#!/usr/bin/env python3
# Run with: .venv/bin/python script_to_audio.py script.txt
"""
劇本轉語音 — 多角色 Kokoro TTS
支援兩種劇本格式：

  格式 A（對話格式）：
    Alice: Hello, how are you?
    Bob: I'm doing great!

  格式 B（標籤格式，TTS-Story 相容）：
    [alice]Hello, how are you?[/alice]
    [bob]I'm doing great![/bob]

用法：
  .venv/bin/python script_to_audio.py script.txt
  .venv/bin/python script_to_audio.py script.txt --voices "Alice=af_heart,Bob=am_adam"
  .venv/bin/python script_to_audio.py script.txt --list-voices
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# 預設音色表（角色名 → Kokoro voice ID）
DEFAULT_VOICE_POOL = [
    "af_heart", "af_bella", "af_jessica", "af_nicole", "af_sarah",
    "am_adam",  "am_michael", "am_echo", "am_eric", "am_liam",
    "bf_emma",  "bm_george",  "bm_lewis",
]

SAMPLE_RATE = 24000


# ── 劇本解析 ─────────────────────────────────────────────────────────────────

def parse_script(text: str) -> List[Tuple[str, str]]:
    """回傳 [(character, line), ...] 列表"""
    lines = []

    # 格式 B：[character]text[/character]
    tag_pattern = re.compile(r'\[(\w+)\](.*?)\[/\1\]', re.DOTALL)
    if tag_pattern.search(text):
        for m in tag_pattern.finditer(text):
            char = m.group(1).strip()
            line = m.group(2).strip().replace('\n', ' ')
            if line:
                lines.append((char, line))
        return lines

    # 格式 A：Character: dialogue
    colon_pattern = re.compile(r'^([^:：\[\n]{1,40})[：:]\s*(.+)$')
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw or raw.startswith('#'):
            continue
        # 跳過舞台指示 [...]  (...)
        if re.match(r'^[\[\(]', raw):
            continue
        m = colon_pattern.match(raw)
        if m:
            lines.append((m.group(1).strip(), m.group(2).strip()))

    return lines


# ── 音色分配 ──────────────────────────────────────────────────────────────────

def assign_voices(characters: List[str], override: str = "") -> Dict[str, str]:
    """將角色名對應到 Kokoro voice ID"""
    mapping: Dict[str, str] = {}

    # 解析手動指定：Alice=af_heart,Bob=am_adam
    if override:
        for pair in override.split(','):
            pair = pair.strip()
            if '=' in pair:
                char, voice = pair.split('=', 1)
                mapping[char.strip()] = voice.strip()

    # 剩餘角色自動從 pool 分配
    pool = [v for v in DEFAULT_VOICE_POOL if v not in mapping.values()]
    idx = 0
    for char in characters:
        if char not in mapping:
            mapping[char] = pool[idx % len(pool)]
            idx += 1

    return mapping


# ── 語音合成 ──────────────────────────────────────────────────────────────────

def generate_script_audio(
    script_lines: List[Tuple[str, str]],
    voice_map: Dict[str, str],
    gap_ms: int = 400,
) -> np.ndarray:
    from kokoro import KPipeline

    # 依語言代碼快取 pipeline（第一次建立時下載模型）
    pipelines: Dict[str, KPipeline] = {}

    def get_pipeline(lang_code: str) -> KPipeline:
        if lang_code not in pipelines:
            print(f"  載入 pipeline [{lang_code}] …")
            pipelines[lang_code] = KPipeline(lang_code=lang_code)
        return pipelines[lang_code]

    def voice_lang(voice_id: str) -> str:
        # Kokoro voice prefix → lang_code
        prefix_map = {
            'a': 'a', 'b': 'b', 'e': 'e', 'f': 'f',
            'h': 'h', 'i': 'i', 'j': 'j', 'p': 'p', 'z': 'z',
        }
        # af_heart → prefix 'a'; bf_emma → 'b'
        return prefix_map.get(voice_id[0], 'a')

    gap_samples = int(SAMPLE_RATE * gap_ms / 1000)
    silence = np.zeros(gap_samples, dtype=np.float32)

    all_audio: List[np.ndarray] = []
    total = len(script_lines)

    for i, (char, text) in enumerate(script_lines, 1):
        voice = voice_map.get(char, "af_heart")
        lang  = voice_lang(voice)
        pipeline = get_pipeline(lang)

        print(f"  [{i:3d}/{total}] {char} ({voice}): {text[:50]}{'…' if len(text)>50 else ''}")

        t0 = time.perf_counter()
        chunks = []
        for _, _, audio in pipeline(text, voice=voice, speed=1.0):
            if audio is not None:
                chunks.append(audio if isinstance(audio, np.ndarray) else audio.numpy())
        elapsed = time.perf_counter() - t0

        if chunks:
            line_audio = np.concatenate(chunks)
            dur = len(line_audio) / SAMPLE_RATE
            print(f"         {dur:.1f}s  ({elapsed:.1f}s,  RTF={elapsed/dur:.2f}x)")
            all_audio.append(line_audio)
            all_audio.append(silence)
        else:
            print(f"         (no audio)")

    return np.concatenate(all_audio) if all_audio else np.array([], dtype=np.float32)


# ── 儲存音訊 ─────────────────────────────────────────────────────────────────

def save_wav(audio: np.ndarray, path: str):
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())


def save_mp3(audio: np.ndarray, path: str, bitrate: str = "192k"):
    tmp_fd, tmp_wav = tempfile.mkstemp(suffix='.wav')
    os.close(tmp_fd)
    try:
        save_wav(audio, tmp_wav)
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_wav,
             "-codec:a", "libmp3lame", "-b:a", bitrate, path],
            check=True, capture_output=True,
        )
    finally:
        os.unlink(tmp_wav)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="劇本轉語音 — Kokoro TTS")
    parser.add_argument("script", nargs='?', help="劇本檔案路徑（.txt）")
    parser.add_argument("--voices", default="",
                        help="手動指定音色，例：Alice=af_heart,Bob=am_adam")
    parser.add_argument("--output", default="",
                        help="輸出路徑（副檔名決定格式：.wav 或 .mp3，預設：<script_name>.wav）")
    parser.add_argument("--format", choices=["wav", "mp3"], default="",
                        help="明確指定輸出格式（優先於副檔名偵測）")
    parser.add_argument("--gap", type=int, default=400,
                        help="每句之間的靜音 ms（預設：400）")
    parser.add_argument("--list-voices", action="store_true",
                        help="列出所有可用 Kokoro 聲音")
    args = parser.parse_args()

    if args.list_voices:
        print("可用聲音（前綴代表語言）：")
        print("  a = 美式英文  b = 英式英文  e = 西班牙文")
        print("  f = 法文      h = 印地文    i = 義大利文")
        print("  j = 日文      p = 葡萄牙文  z = 普通話\n")
        for v in DEFAULT_VOICE_POOL:
            print(f"  {v}")
        return

    if not args.script:
        parser.print_help()
        sys.exit(1)

    script_path = Path(args.script)
    if not script_path.exists():
        print(f"找不到檔案：{script_path}")
        sys.exit(1)

    # 決定格式：--format > 副檔名 > 預設 wav
    fmt = args.format
    if not fmt:
        ext = Path(args.output).suffix.lower() if args.output else ''
        fmt = "mp3" if ext == ".mp3" else "wav"

    default_name = script_path.with_suffix(f'.{fmt}').name
    output_path = args.output or default_name

    print(f"\n讀取劇本：{script_path}")
    text = script_path.read_text(encoding='utf-8')
    lines = parse_script(text)

    if not lines:
        print("無法解析劇本，請確認格式（角色: 台詞 或 [角色]台詞[/角色]）")
        sys.exit(1)

    characters = list(dict.fromkeys(c for c, _ in lines))  # 保持順序去重
    voice_map  = assign_voices(characters, args.voices)

    print(f"\n角色音色分配：")
    for char, voice in voice_map.items():
        print(f"  {char:20s} → {voice}")

    print(f"\n共 {len(lines)} 句台詞，開始合成…\n")
    t_start = time.perf_counter()
    audio = generate_script_audio(lines, voice_map, gap_ms=args.gap)
    total_time = time.perf_counter() - t_start

    if len(audio) == 0:
        print("沒有產生任何音訊。")
        sys.exit(1)

    if fmt == "mp3":
        save_mp3(audio, output_path)
    else:
        save_wav(audio, output_path)
    total_dur = len(audio) / SAMPLE_RATE

    print(f"\n完成！")
    print(f"  輸出檔案  : {output_path}  ({fmt.upper()})")
    print(f"  音訊時長  : {total_dur:.1f}s")
    print(f"  總耗時    : {total_time:.1f}s")
    print(f"  整體 RTF  : {total_time/total_dur:.2f}x")


if __name__ == "__main__":
    main()
