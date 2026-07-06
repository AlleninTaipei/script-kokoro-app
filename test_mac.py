#!/usr/bin/env python3
# Run with: .venv/bin/python test_mac.py
"""Quick smoke test: can this Mac run Kokoro TTS?"""

import sys
import time

SAMPLE_TEXT = "Hello! This is a test of Kokoro text to speech on Apple Silicon."
VOICE = "af_heart"
OUTPUT_WAV = "test_output.wav"


def check_python():
    v = sys.version_info
    ok = v >= (3, 10) and v < (3, 14)
    status = "OK" if ok else "FAIL"
    print(f"[{status}] Python {v.major}.{v.minor}.{v.micro}  (need >=3.10, <3.14)")
    return ok


def check_torch():
    try:
        import torch
        mps = torch.backends.mps.is_available()
        cuda = torch.cuda.is_available()
        device = "mps" if mps else ("cuda" if cuda else "cpu")
        print(f"[ OK] torch {torch.__version__}  |  device: {device}")
        return device
    except ImportError:
        print("[FAIL] torch not installed  →  pip install torch")
        return None


def check_kokoro():
    try:
        from kokoro import KPipeline, KModel  # noqa: F401
        print("[ OK] kokoro imported successfully")
        return True
    except ImportError as e:
        print(f"[FAIL] kokoro import failed: {e}")
        print("       Try:  pip install kokoro")
        return False


def run_inference(device: str):
    import wave
    import numpy as np
    from kokoro import KPipeline

    print(f"\n>> Generating speech on device={device!r} ...")
    print(f"   text  : {SAMPLE_TEXT!r}")
    print(f"   voice : {VOICE}")

    t0 = time.perf_counter()
    pipeline = KPipeline(lang_code="a")

    audio_chunks = []
    for result in pipeline(SAMPLE_TEXT, voice=VOICE, speed=1.0):
        if result.audio is not None:
            audio_chunks.append(result.audio.numpy())

    elapsed = time.perf_counter() - t0

    if not audio_chunks:
        print("[FAIL] No audio generated")
        return False

    audio = np.concatenate(audio_chunks)
    duration = len(audio) / 24000

    with wave.open(OUTPUT_WAV, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())

    rtf = elapsed / duration  # real-time factor (lower = faster)
    print(f"\n[ OK] Done!")
    print(f"   audio duration : {duration:.2f}s")
    print(f"   inference time : {elapsed:.2f}s")
    print(f"   RTF            : {rtf:.3f}x  ({'faster' if rtf < 1 else 'slower'} than real-time)")
    print(f"   output WAV     : {OUTPUT_WAV}")
    return True


def play_audio():
    import subprocess
    print(f"\n>> Playing {OUTPUT_WAV} via afplay ...")
    subprocess.run(["afplay", OUTPUT_WAV], check=True)


def main():
    print("=" * 55)
    print("  Kokoro TTS — Mac smoke test")
    print("=" * 55)

    if not check_python():
        sys.exit(1)

    device = check_torch()
    if device is None:
        sys.exit(1)

    if not check_kokoro():
        sys.exit(1)

    print()
    ok = run_inference(device)
    if not ok:
        sys.exit(1)

    play_audio()
    print("\nAll checks passed. Kokoro runs on this Mac.")


if __name__ == "__main__":
    main()
