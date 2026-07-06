#!/usr/bin/env python3
# Run with: .venv/bin/python script_to_audio_onnx.py script.txt
"""
劇本轉語音 — ONNX 版（對照 script_to_audio.py 的 PyTorch 版）

推論路徑差異：
  PyTorch 版：KPipeline.__call__() → KModel.forward_with_tokens()  (torch CPU)
  ONNX 版   ：手動 G2P → 取 input_ids + style → ort.InferenceSession.run()

ONNX session 只載入一次，每句台詞直接呼叫 session.run()，
不重複載入模型，冷啟動後每句延遲應接近 RTF ≈ 1x。
"""

import argparse
import re
import sys
import time
import wave
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import onnxruntime as ort

SAMPLE_RATE   = 24000
ONNX_FILE     = "kokoro.onnx"

DEFAULT_VOICE_POOL = [
    "af_heart", "af_bella", "af_jessica", "af_nicole", "af_sarah",
    "am_adam",  "am_michael", "am_echo",  "am_eric",   "am_liam",
    "bf_emma",  "bm_george",  "bm_lewis",
]


# ── 劇本解析（與 PyTorch 版相同） ─────────────────────────────────────────────

def parse_script(text: str) -> List[Tuple[str, str]]:
    lines = []
    tag_pattern = re.compile(r'\[(\w+)\](.*?)\[/\1\]', re.DOTALL)
    if tag_pattern.search(text):
        for m in tag_pattern.finditer(text):
            char = m.group(1).strip()
            line = m.group(2).strip().replace('\n', ' ')
            if line:
                lines.append((char, line))
        return lines
    colon_pattern = re.compile(r'^([^:：\[\n]{1,40})[：:]\s*(.+)$')
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw or raw.startswith('#') or re.match(r'^[\[\(]', raw):
            continue
        m = colon_pattern.match(raw)
        if m:
            lines.append((m.group(1).strip(), m.group(2).strip()))
    return lines


def assign_voices(characters: List[str], override: str = "") -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if override:
        for pair in override.split(','):
            if '=' in pair:
                char, voice = pair.strip().split('=', 1)
                mapping[char.strip()] = voice.strip()
    pool = [v for v in DEFAULT_VOICE_POOL if v not in mapping.values()]
    idx = 0
    for char in characters:
        if char not in mapping:
            mapping[char] = pool[idx % len(pool)]
            idx += 1
    return mapping


def voice_lang(voice_id: str) -> str:
    return {'a':'a','b':'b','e':'e','f':'f','h':'h','i':'i','j':'j','p':'p','z':'z'}.get(voice_id[0], 'a')


# ── ONNX 推論核心 ─────────────────────────────────────────────────────────────

class OnnxKokoro:
    """
    用 onnxruntime 取代 KModel.forward_with_tokens()。
    KPipeline 仍用於 G2P（text → phonemes）和 voice pack 載入，
    但實際的神經網路推論完全走 ONNX。
    """

    def __init__(self, onnx_path: str = ONNX_FILE):
        from kokoro import KModel, KPipeline

        if not Path(onnx_path).exists():
            print(f"找不到 {onnx_path}，請先執行 test_onnx.py 產生 ONNX 模型。")
            sys.exit(1)

        print(f"載入 ONNX session：{onnx_path}  ({Path(onnx_path).stat().st_size/1e6:.0f} MB)")
        t0 = time.perf_counter()
        self.session = ort.InferenceSession(onnx_path)
        print(f"Session 就緒（{time.perf_counter()-t0:.2f}s）")

        # KModel 只用於取 vocab，不跑推論
        print("載入 KModel（取 vocab）…")
        self.kmodel = KModel(repo_id="hexgrad/Kokoro-82M").eval()
        self.vocab   = self.kmodel.vocab

        # 每個 lang_code 的 pipeline（lazy 建立）
        self._pipelines: Dict[str, KPipeline] = {}

    def _get_pipeline(self, lang_code: str):
        from kokoro import KPipeline
        if lang_code not in self._pipelines:
            # model=None → pipeline 不會自行建立 KModel，phonemize only
            self._pipelines[lang_code] = KPipeline(lang_code=lang_code, model=self.kmodel)
        return self._pipelines[lang_code]

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> np.ndarray:
        """
        text  → G2P（KPipeline）→ input_ids + style → ort.run() → waveform
        長句自動由 en_tokenize 分段處理後合併。
        """
        lang   = voice_lang(voice)
        pipeline = self._get_pipeline(lang)

        # 載入 voice pack（510 × 1 × 256）
        pack = pipeline.load_voice(voice)

        chunks: List[np.ndarray] = []

        if lang in 'ab':
            _, tokens = pipeline.g2p(text)
            for gs, ps, _ in pipeline.en_tokenize(tokens):
                if not ps:
                    continue
                audio = self._run_segment(ps, pack, speed)
                if audio is not None:
                    chunks.append(audio)
        else:
            # 非英文：整段送入
            ps, _ = pipeline.g2p(text)
            if len(ps) > 510:
                ps = ps[:510]
            audio = self._run_segment(ps, pack, speed)
            if audio is not None:
                chunks.append(audio)

        return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)

    def _run_segment(self, ps: str, pack: torch.FloatTensor, speed: float) -> np.ndarray:
        """單一音素段 → ort.run() → numpy waveform"""
        ids = [self.vocab[p] for p in ps if self.vocab.get(p) is not None]
        if not ids:
            return None
        input_ids = np.array([[0, *ids, 0]], dtype=np.int64)
        style     = pack[len(ps) - 1].numpy()          # (1, 256)
        spd       = np.array([round(speed)], dtype=np.int32)

        outputs = self.session.run(None, {
            "input_ids": input_ids,
            "style":     style,
            "speed":     spd,
        })
        return outputs[0].squeeze()


# ── 批次生成 ──────────────────────────────────────────────────────────────────

def generate_script_audio(
    engine: OnnxKokoro,
    script_lines: List[Tuple[str, str]],
    voice_map: Dict[str, str],
    gap_ms: int = 400,
) -> np.ndarray:
    silence   = np.zeros(int(SAMPLE_RATE * gap_ms / 1000), dtype=np.float32)
    all_audio = []
    total     = len(script_lines)

    for i, (char, text) in enumerate(script_lines, 1):
        voice = voice_map.get(char, "af_heart")
        print(f"  [{i:3d}/{total}] {char} ({voice}): {text[:50]}{'…' if len(text)>50 else ''}")

        t0    = time.perf_counter()
        audio = engine.synthesize(text, voice)
        elapsed = time.perf_counter() - t0

        if len(audio) > 0:
            dur = len(audio) / SAMPLE_RATE
            print(f"         {dur:.1f}s  ({elapsed:.1f}s,  RTF={elapsed/dur:.2f}x)")
            all_audio.append(audio)
            all_audio.append(silence)
        else:
            print(f"         (no audio)")

    return np.concatenate(all_audio) if all_audio else np.array([], dtype=np.float32)


def save_wav(audio: np.ndarray, path: str):
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="劇本轉語音 — ONNX 版")
    parser.add_argument("script", nargs='?', help="劇本檔案路徑（.txt）")
    parser.add_argument("--voices", default="",
                        help="手動指定音色，例：Alice=af_heart,Bob=am_adam")
    parser.add_argument("--output", default="",
                        help="輸出 WAV 路徑（預設：<script_name>_onnx.wav）")
    parser.add_argument("--gap",  type=int, default=400,
                        help="句間靜音 ms（預設：400）")
    parser.add_argument("--onnx", default=ONNX_FILE,
                        help=f"ONNX 模型路徑（預設：{ONNX_FILE}）")
    args = parser.parse_args()

    if not args.script:
        parser.print_help()
        sys.exit(1)

    script_path = Path(args.script)
    if not script_path.exists():
        print(f"找不到檔案：{script_path}")
        sys.exit(1)

    stem        = script_path.stem
    output_path = args.output or f"{stem}_onnx.wav"

    print(f"\n讀取劇本：{script_path}")
    text  = script_path.read_text(encoding='utf-8')
    lines = parse_script(text)
    if not lines:
        print("無法解析劇本。")
        sys.exit(1)

    characters = list(dict.fromkeys(c for c, _ in lines))
    voice_map  = assign_voices(characters, args.voices)

    print(f"\n角色音色分配：")
    for char, voice in voice_map.items():
        print(f"  {char:20s} → {voice}")

    print(f"\n初始化 ONNX 引擎…")
    engine = OnnxKokoro(onnx_path=args.onnx)

    print(f"\n共 {len(lines)} 句台詞，開始合成…\n")
    t_start = time.perf_counter()
    audio   = generate_script_audio(engine, lines, voice_map, gap_ms=args.gap)
    total_time = time.perf_counter() - t_start

    if len(audio) == 0:
        print("沒有產生任何音訊。")
        sys.exit(1)

    save_wav(audio, output_path)
    total_dur = len(audio) / SAMPLE_RATE

    print(f"\n完成！")
    print(f"  輸出檔案  : {output_path}")
    print(f"  音訊時長  : {total_dur:.1f}s")
    print(f"  總耗時    : {total_time:.1f}s")
    print(f"  整體 RTF  : {total_time/total_dur:.2f}x")


if __name__ == "__main__":
    main()
