#!/usr/bin/env python3
# Run with: .venv/Scripts/python.exe test_onnx_pc.py
"""
Test kokoro-v1 ONNX path on this Windows PC.
Step 1: Export PyTorch model -> kokoro.onnx
Step 2: Run inference with onnxruntime (no torch required at runtime)
Step 3: Play audio via winsound
"""

import time
import wave
from pathlib import Path

import numpy as np
import torch

ONNX_FILE = "kokoro.onnx"
OUTPUT_WAV = "test_onnx_output.wav"
SAMPLE_TEXT = "Hello! This is Kokoro running as an ONNX model, without PyTorch at inference time."
VOICE      = "af_heart"


# ── Step 1: Export ────────────────────────────────────────────────────────────

def export():
    from kokoro import KModel
    from kokoro.model import KModelForONNX
    import onnx

    print("[ 1/3 ] Loading PyTorch model ...")
    kmodel = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True).eval()
    model  = KModelForONNX(kmodel).eval()

    print(f"[ 1/3 ] Exporting -> {ONNX_FILE} ...")
    dummy_ids   = torch.LongTensor([[0, *torch.randint(1, 100, (48,)).tolist(), 0]])
    dummy_style = torch.randn(1, 256)
    dummy_speed = torch.IntTensor([1])

    torch.onnx.export(
        model,
        args=(dummy_ids, dummy_style, dummy_speed),
        f=ONNX_FILE,
        export_params=True,
        input_names=["input_ids", "style", "speed"],
        output_names=["waveform", "duration"],
        opset_version=17,
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "style":     {0: "batch"},
            "speed":     {0: "batch"},
        },
        do_constant_folding=True,
        dynamo=False,  # torch >= 2.6 defaults to the dynamo exporter, which
                       # chokes on this model's data-dependent control flow
    )

    onnx.checker.check_model(onnx.load(ONNX_FILE))
    size_mb = Path(ONNX_FILE).stat().st_size / 1e6
    print(f"[ 1/3 ] Export OK  ({size_mb:.1f} MB)")
    return kmodel   # reuse for tokenisation / voice loading


# ── Step 2: ONNX inference ────────────────────────────────────────────────────

def infer_onnx(kmodel):
    import onnxruntime as ort
    from kokoro import KPipeline

    print("\n[ 2/3 ] Running ONNX inference ...")
    print(f"        text  : {SAMPLE_TEXT!r}")
    print(f"        voice : {VOICE}")

    pipeline = KPipeline(lang_code="a", model=kmodel, device="cpu")
    session  = ort.InferenceSession(ONNX_FILE)

    # tokenise text -> phoneme ids
    def _get_ids_and_style(text, voice_name):
        _, tokens = pipeline.g2p(text)
        ps = []
        for _, ps_chunk, _ in pipeline.en_tokenize(tokens):
            if ps_chunk:
                ps.extend(ps_chunk)
        if len(ps) > 510:
            ps = ps[:510]
        ids = [kmodel.vocab[p] for p in ps if kmodel.vocab.get(p) is not None]
        input_ids = torch.LongTensor([[0, *ids, 0]])

        # voice style vector
        voice_pack = pipeline.load_voice(voice_name)
        style = voice_pack[len(ps) - 1]  # (1, 256)
        return input_ids, style

    t0 = time.perf_counter()
    input_ids, style = _get_ids_and_style(SAMPLE_TEXT, VOICE)
    speed = np.array([1], dtype=np.int32)

    outputs = session.run(None, {
        "input_ids": input_ids.numpy(),
        "style":     style.numpy(),
        "speed":     speed,
    })

    elapsed = time.perf_counter() - t0
    audio   = outputs[0].squeeze()          # float32 waveform
    duration = len(audio) / 24000

    print(f"\n[ 2/3 ] Done!")
    print(f"        audio duration : {duration:.2f}s")
    print(f"        inference time : {elapsed:.2f}s")
    print(f"        RTF            : {elapsed/duration:.3f}x")
    return audio


# ── Step 3: Save & play ───────────────────────────────────────────────────────

def save_and_play(audio):
    import winsound

    print(f"\n[ 3/3 ] Saving -> {OUTPUT_WAV}")
    with wave.open(OUTPUT_WAV, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())

    print(f"[ 3/3 ] Playing via winsound ...")
    winsound.PlaySound(OUTPUT_WAV, winsound.SND_FILENAME)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 58)
    print("  kokoro-v1 ONNX - Windows test")
    print("=" * 58)

    if not Path(ONNX_FILE).exists():
        kmodel = export()
    else:
        print(f"[ 1/3 ] {ONNX_FILE} already exists, skipping export.")
        from kokoro import KModel
        kmodel = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True).eval()

    audio = infer_onnx(kmodel)
    save_and_play(audio)

    print("\nAll done. ONNX path works on this PC.")
