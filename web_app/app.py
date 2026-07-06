#!/usr/bin/env python3
"""
Kokoro TTS Web App — FastAPI + SSE
劇本上傳 → LLM 配音分析 → 逐句 TTS 串流
啟動：.venv/Scripts/python.exe -m uvicorn web_app.app:app --reload --port 8000
"""

import asyncio
import concurrent.futures
import io
import json
import os
import re
import subprocess
import tempfile
import time
import uuid
import wave
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── 路徑 ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent   # repo root

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import anthropic

ONNX_FILE  = ROOT / "kokoro.onnx"
STATIC_DIR = Path(__file__).parent / "static"
TEMP_DIR   = Path(__file__).parent / "tmp"
TEMP_DIR.mkdir(exist_ok=True)
SAMPLE_RATE = 24000

# 試聽用示範文本（依語言自動選擇）
PREVIEW_TEXT = {
    "en": (
        "The morning light filtered through the curtains, casting long shadows across the floor. "
        "She picked up her coffee and stared out at the city below. "
        "Today was going to be different — she could feel it."
    ),
    "zh": (
        "清晨的阳光透过窗帘洒落，在地板上投下长长的影子。"
        "她端起咖啡，凝望着窗外的城市。"
        "今天会不同以往——她心里这样感觉着。"
    ),
}

# ── 聲音目錄 ─────────────────────────────────────────────────────────────────

VOICE_CATALOG: Dict[str, dict] = {
    "af_heart":    {"lang": "en", "gender": "female", "group": "American English", "desc": "warm & expressive"},
    "af_bella":    {"lang": "en", "gender": "female", "group": "American English", "desc": "bright & lively"},
    "af_jessica":  {"lang": "en", "gender": "female", "group": "American English", "desc": "clear & professional"},
    "af_nicole":   {"lang": "en", "gender": "female", "group": "American English", "desc": "soft & gentle"},
    "af_sarah":    {"lang": "en", "gender": "female", "group": "American English", "desc": "natural & friendly"},
    "am_adam":     {"lang": "en", "gender": "male",   "group": "American English", "desc": "deep & strong"},
    "am_michael":  {"lang": "en", "gender": "male",   "group": "American English", "desc": "authoritative"},
    "am_echo":     {"lang": "en", "gender": "male",   "group": "American English", "desc": "resonant"},
    "am_eric":     {"lang": "en", "gender": "male",   "group": "American English", "desc": "friendly & casual"},
    "am_liam":     {"lang": "en", "gender": "male",   "group": "American English", "desc": "young & energetic"},
    "bf_emma":     {"lang": "en", "gender": "female", "group": "British English",  "desc": "refined & elegant"},
    "bm_george":   {"lang": "en", "gender": "male",   "group": "British English",  "desc": "distinguished"},
    "bm_lewis":    {"lang": "en", "gender": "male",   "group": "British English",  "desc": "casual & warm"},
    "zf_xiaobei":  {"lang": "zh", "gender": "female", "group": "Mandarin Chinese", "desc": "活潑可愛"},
    "zf_xiaoni":   {"lang": "zh", "gender": "female", "group": "Mandarin Chinese", "desc": "溫柔甜美"},
    "zf_xiaoxiao": {"lang": "zh", "gender": "female", "group": "Mandarin Chinese", "desc": "清晰自然"},
    "zf_xiaoyi":   {"lang": "zh", "gender": "female", "group": "Mandarin Chinese", "desc": "知性優雅"},
    "zm_yunjian":  {"lang": "zh", "gender": "male",   "group": "Mandarin Chinese", "desc": "沉穩有力"},
    "zm_yunxi":    {"lang": "zh", "gender": "male",   "group": "Mandarin Chinese", "desc": "陽光活力"},
    "zm_yunxia":   {"lang": "zh", "gender": "male",   "group": "Mandarin Chinese", "desc": "溫和親切"},
    "zm_yunyang":  {"lang": "zh", "gender": "male",   "group": "Mandarin Chinese", "desc": "成熟穩重"},
}

# ── 劇本解析 ─────────────────────────────────────────────────────────────────

def _parse_fountain(text: str) -> List[Tuple[str, str]]:
    """
    Fountain screenplay format parser.
    Character cues: ALL CAPS line, preceded by blank line, followed by dialogue.
    Parentheticals (stage directions) are stripped.
    Extensions like (V.O.) / (O.S.) / (CONT'D) are removed from the name.
    """
    # Strip boneyard /* ... */ and notes [[ ... ]]
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    text = re.sub(r'\[\[.*?\]\]', '', text, flags=re.DOTALL)

    src = text.splitlines()
    result: List[Tuple[str, str]] = []
    i = 0
    while i < len(src):
        raw = src[i]
        stripped = raw.strip()
        forced = stripped.startswith('@')
        candidate = stripped.lstrip('@').strip()
        # Remove trailing extensions: (V.O.), (O.S.), (CONT'D), etc.
        char_name = re.sub(r'\s*\([^)]*\)\s*$', '', candidate).strip()

        prev_blank  = (i == 0) or (not src[i - 1].strip())
        next_filled = (i + 1 < len(src)) and src[i + 1].strip()
        all_caps    = bool(char_name and char_name == char_name.upper()
                          and re.search(r'[A-Z]', char_name))

        if (forced or all_caps) and prev_blank and next_filled:
            i += 1
            parts: List[str] = []
            while i < len(src):
                dline = src[i].strip()
                if not dline:
                    break
                if re.match(r'^\(.*\)$', dline):   # skip parenthetical
                    i += 1
                    continue
                parts.append(dline)
                i += 1
            if parts and char_name:
                result.append((char_name, ' '.join(parts)))
        else:
            i += 1
    return result


def parse_script(text: str) -> List[Tuple[str, str]]:
    # 1. Fountain: ALL CAPS character cues preceded by blank lines
    fountain = _parse_fountain(text)
    if len(fountain) >= 2:
        return fountain

    # 2. Tag format: [character]dialogue[/character]
    tag_pattern = re.compile(r'\[(\w+)\](.*?)\[/\1\]', re.DOTALL)
    if tag_pattern.search(text):
        lines = []
        for m in tag_pattern.finditer(text):
            char = m.group(1).strip()
            line = m.group(2).strip().replace('\n', ' ')
            if line:
                lines.append((char, line))
        return lines

    # 3. Colon format: Character: dialogue
    lines = []
    colon_pattern = re.compile(r'^([^:：\[\n]{1,40})[：:]\s*(.+)$')
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw or raw.startswith('#') or re.match(r'^[\[\(]', raw):
            continue
        m = colon_pattern.match(raw)
        if m:
            lines.append((m.group(1).strip(), m.group(2).strip()))
    return lines


def voice_lang(voice_id: str) -> str:
    return {
        'a': 'a', 'b': 'b', 'e': 'e', 'f': 'f',
        'h': 'h', 'i': 'i', 'j': 'j', 'p': 'p', 'z': 'z',
    }.get(voice_id[0], 'a')


# ── TTS 引擎（lazy singleton） ───────────────────────────────────────────────

_onnx_synth = None
_pytorch_pipelines: Dict[str, object] = {}


def _get_pytorch_pipeline(lang_code: str):
    from kokoro import KPipeline
    if lang_code not in _pytorch_pipelines:
        _pytorch_pipelines[lang_code] = KPipeline(lang_code=lang_code)
    return _pytorch_pipelines[lang_code]


def _synth_pytorch(text: str, voice: str) -> np.ndarray:
    pipeline = _get_pytorch_pipeline(voice_lang(voice))
    chunks = []
    for _, _, audio in pipeline(text, voice=voice, speed=1.0):
        if audio is not None:
            chunks.append(audio if isinstance(audio, np.ndarray) else audio.numpy())
    return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)


def _build_onnx_synth():
    import onnxruntime as ort
    from kokoro import KModel, KPipeline

    if not ONNX_FILE.exists():
        raise RuntimeError(f"找不到 {ONNX_FILE}，請先執行 test_onnx.py 或 test_onnx_pc.py 產生 ONNX 模型。")

    session  = ort.InferenceSession(str(ONNX_FILE))
    kmodel   = KModel(repo_id="hexgrad/Kokoro-82M").eval()
    vocab    = kmodel.vocab
    pipes: Dict[str, KPipeline] = {}

    def _pipe(lc: str):
        if lc not in pipes:
            pipes[lc] = KPipeline(lang_code=lc, model=kmodel)
        return pipes[lc]

    def _segment(ps, pack):
        ids = [vocab[p] for p in ps if vocab.get(p) is not None]
        if not ids:
            return None
        outputs = session.run(None, {
            "input_ids": np.array([[0, *ids, 0]], dtype=np.int64),
            "style":     pack[len(ps) - 1].numpy(),
            "speed":     np.array([1], dtype=np.int32),
        })
        return outputs[0].squeeze()

    def synth(text: str, voice: str) -> np.ndarray:
        lang = voice_lang(voice)
        pipe = _pipe(lang)
        pack = pipe.load_voice(voice)
        chunks = []
        if lang in 'ab':
            _, tokens = pipe.g2p(text)
            for _, ps, _ in pipe.en_tokenize(tokens):
                if ps:
                    a = _segment(ps, pack)
                    if a is not None:
                        chunks.append(a)
        else:
            ps, _ = pipe.g2p(text)
            if len(ps) > 510:
                ps = ps[:510]
            a = _segment(ps, pack)
            if a is not None:
                chunks.append(a)
        return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)

    return synth


def _get_onnx_synth():
    global _onnx_synth
    if _onnx_synth is None:
        _onnx_synth = _build_onnx_synth()
    return _onnx_synth


def synthesize_line(text: str, voice: str, backend: str) -> np.ndarray:
    if backend == "onnx":
        return _get_onnx_synth()(text, voice)
    return _synth_pytorch(text, voice)


def to_wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def wav_bytes_to_mp3(wav_data: bytes, bitrate: str = "192k") -> bytes:
    fd, tmp_wav = tempfile.mkstemp(suffix='.wav')
    fd2, tmp_mp3 = tempfile.mkstemp(suffix='.mp3')
    os.close(fd)
    os.close(fd2)
    try:
        Path(tmp_wav).write_bytes(wav_data)
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_wav,
             "-codec:a", "libmp3lame", "-b:a", bitrate, tmp_mp3],
            check=True, capture_output=True,
        )
        return Path(tmp_mp3).read_bytes()
    finally:
        os.unlink(tmp_wav)
        os.unlink(tmp_mp3)


# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(title="Kokoro TTS")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


# ── /analyze + /parse ────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text: str


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    lines = parse_script(req.text)
    if not lines:
        raise HTTPException(400, "無法解析劇本（請使用「角色: 台詞」或 [角色]台詞[/角色] 格式）")

    characters = list(dict.fromkeys(c for c, _ in lines))

    # 每個角色取最多 2 句範例
    samples: Dict[str, List[str]] = {}
    for char, text in lines:
        if char not in samples:
            samples[char] = []
        if len(samples[char]) < 2:
            samples[char].append(text[:80])

    sample_block = "\n".join(
        f"  {c}: {' / '.join(ss)}" for c, ss in samples.items()
    )
    voice_block = "\n".join(
        f"  {vid}: [{info['group']}] {info['gender']} — {info['desc']}"
        for vid, info in VOICE_CATALOG.items()
    )

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                "你是 TTS 配音導演。根據角色台詞的語言、性別、年齡、語氣，"
                "從以下聲音列表中為每個角色選出最合適的聲音。\n\n"
                f"可用聲音：\n{voice_block}\n\n"
                f"角色台詞範例：\n{sample_block}\n\n"
                "回傳純 JSON（不含 markdown），格式如下：\n"
                '{"mapping": {"角色名": "voice_id"}, "reasons": {"角色名": "一句理由"}}'
            ),
        }],
    )

    raw = msg.content[0].text.strip()
    raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw).strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        pool = list(VOICE_CATALOG.keys())
        result = {
            "mapping": {c: pool[i % len(pool)] for i, c in enumerate(characters)},
            "reasons": {c: "自動分配" for c in characters},
        }

    return {
        "lines":        [{"char": c, "text": t} for c, t in lines],
        "characters":   characters,
        "voice_map":    result.get("mapping", {}),
        "reasons":      result.get("reasons", {}),
        "voice_catalog": {
            vid: f"{info['group']} · {info['gender']} · {info['desc']}"
            for vid, info in VOICE_CATALOG.items()
        },
    }


# ── /synthesize (SSE) ─────────────────────────────────────────────────────────

class SynthesizeRequest(BaseModel):
    lines:     List[dict]        # [{"char": "Alice", "text": "..."}]
    voice_map: Dict[str, str]
    backend:   str = "onnx"


@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    session_id = uuid.uuid4().hex[:8]
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _push(event: dict):
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def run_tts():
        silence = np.zeros(int(SAMPLE_RATE * 0.4), dtype=np.float32)
        all_audio: List[np.ndarray] = []
        total = len(req.lines)

        _push({"type": "start", "total": total, "backend": req.backend})

        for i, item in enumerate(req.lines):
            char  = item["char"]
            text  = item["text"]
            voice = req.voice_map.get(char, "af_heart")

            _push({"type": "processing", "index": i + 1, "total": total,
                   "char": char, "voice": voice, "text": text[:60]})

            t0 = time.perf_counter()
            try:
                audio = synthesize_line(text, voice, req.backend)
            except Exception as e:
                _push({"type": "error", "message": str(e)})
                _push(None)
                return

            elapsed = time.perf_counter() - t0
            dur = len(audio) / SAMPLE_RATE if len(audio) > 0 else 0
            rtf = round(elapsed / dur, 2) if dur > 0 else 0

            if len(audio) > 0:
                all_audio.append(audio)
                all_audio.append(silence)

            _push({"type": "line", "index": i + 1, "total": total,
                   "char": char, "voice": voice, "text": text[:60],
                   "duration": round(dur, 2), "rtf": rtf})

        if all_audio:
            final = np.concatenate(all_audio)
            wav_data = to_wav_bytes(final)
            (TEMP_DIR / f"{session_id}.wav").write_bytes(wav_data)
            try:
                (TEMP_DIR / f"{session_id}.mp3").write_bytes(wav_bytes_to_mp3(wav_data))
            except Exception as e:
                print(f"MP3 轉檔失敗（將只提供 WAV 下載）：{e}")
            total_dur = round(len(final) / SAMPLE_RATE, 2)
        else:
            total_dur = 0

        _push({"type": "done", "session_id": session_id, "total_duration": total_dur})
        _push(None)  # sentinel

    loop.run_in_executor(concurrent.futures.ThreadPoolExecutor(max_workers=1), run_tts)

    async def event_stream():
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=600)
            except asyncio.TimeoutError:
                break
            if event is None:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── /parse（單一朗讀，不呼叫 LLM） ───────────────────────────────────────────

@app.post("/parse")
async def parse_only(req: AnalyzeRequest):
    lines = parse_script(req.text)
    if not lines:
        raise HTTPException(400, "無法解析劇本格式")
    characters = list(dict.fromkeys(c for c, _ in lines))
    return {
        "lines":        [{"char": c, "text": t} for c, t in lines],
        "characters":   characters,
        "voice_catalog": {
            vid: f"{info['group']} · {info['gender']} · {info['desc']}"
            for vid, info in VOICE_CATALOG.items()
        },
    }


# ── /preview ─────────────────────────────────────────────────────────────────

@app.get("/preview/{voice_id}")
async def preview(voice_id: str):
    if voice_id not in VOICE_CATALOG:
        raise HTTPException(400, "Unknown voice")

    cache_path = TEMP_DIR / f"preview_{voice_id}.wav"
    if cache_path.exists():
        return Response(content=cache_path.read_bytes(), media_type="audio/wav")

    lang = "zh" if voice_id.startswith("z") else "en"
    text = PREVIEW_TEXT[lang]

    loop = asyncio.get_running_loop()

    def _synth():
        try:
            return _get_onnx_synth()(text, voice_id)
        except Exception:
            return _synth_pytorch(text, voice_id)

    audio = await loop.run_in_executor(None, _synth)

    if len(audio) == 0:
        raise HTTPException(500, "Synthesis produced no audio")

    wav_bytes = to_wav_bytes(audio)
    cache_path.write_bytes(wav_bytes)
    return Response(content=wav_bytes, media_type="audio/wav")


# ── /download ─────────────────────────────────────────────────────────────────

@app.get("/download/{session_id}")
async def download(session_id: str, format: str = "wav"):
    if not re.fullmatch(r'[0-9a-f]{8}', session_id):
        raise HTTPException(400, "Invalid session ID")
    if format not in ("wav", "mp3"):
        raise HTTPException(400, "format must be wav or mp3")
    path = TEMP_DIR / f"{session_id}.{format}"
    if not path.exists():
        raise HTTPException(404, "Audio file not found")
    media_type = "audio/mpeg" if format == "mp3" else "audio/wav"
    return Response(
        content=path.read_bytes(),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="kokoro_{session_id}.{format}"'},
    )
