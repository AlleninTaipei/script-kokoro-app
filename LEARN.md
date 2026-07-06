# LEARN.md — Kokoro TTS Script Experiments

> A behind-the-scenes tour of this folder: what it does, why it was built the way it was, and the real engineering lessons hiding inside.

---

## What Is This, Really?

This folder is a **field lab** sitting on top of the Kokoro TTS library. While the parent repo is the whole Kokoro engine (82M-parameter neural network, Python package, JavaScript port), this `script/` folder is where we answered one specific question:

> *"Can a regular Intel Mac turn a multi-character screenplay into a fully voiced audio file — and how fast can we make it?"*

The answer: yes, and the journey to get there taught us a lot about the gap between research-grade PyTorch code and production-ready inference.

---

## The Five Files at a Glance

| File | Role |
|---|---|
| `test_mac.py` | Sanity check: does Kokoro even run on this machine? |
| `test_onnx.py` | Proof-of-concept: export the model to ONNX, run it without PyTorch |
| `script_to_audio.py` | Full pipeline (PyTorch version): screenplay → `.wav` |
| `script_to_audio_onnx.py` | Same pipeline, but ONNX for the neural inference step |
| `example_story.txt` | Sample 3-character story to demo the pipeline |

---

## The Architecture in Plain Language

### The Big Picture

Think of the pipeline as an assembly line in a movie dubbing studio:

1. **Script Supervisor** (`parse_script`) reads the screenplay and produces a list of `(character, line)` pairs.
2. **Casting Director** (`assign_voices`) maps each character name to a specific Kokoro voice ID.
3. **Voice Actors** (Kokoro TTS engine) receive each line and synthesize the audio.
4. **Editor** (`generate_script_audio`) stitches all clips together with brief silences between lines.
5. **Distribution** (`save_wav`) writes the final PCM audio to a `.wav` file.

### Two Engines, Same Assembly Line

The clever design choice here is that **both `script_to_audio.py` and `script_to_audio_onnx.py` share the same assembly line** — they only swap out the "Voice Actor" step:

```
PyTorch path:   KPipeline.__call__()  →  KModel.forward_with_tokens()  (torch CPU)
ONNX path:      KPipeline.g2p()       →  ort.InferenceSession.run()    (ONNX Runtime)
```

The parsing, voice assignment, silence insertion, and WAV writing are identical in both files. Only the neural inference core differs.

---

## The ONNX Story: Why It Matters

### The Problem

On an Intel Mac (x86_64, macOS 12), the highest available PyTorch version is 2.2.2. Apple's MPS (Metal Performance Shaders) GPU acceleration doesn't exist on Intel — only on Apple Silicon. So every inference call runs on plain CPU through PyTorch's kernel, and the Real-Time Factor (RTF) was a painful **5–8x**: synthesizing 1 second of audio took 5–8 seconds.

That's fine for a demo. It's agonizing if you want to synthesize a 7-line story.

### The Fix

ONNX Runtime has highly-optimized CPU kernels that bypass PyTorch's overhead. The same neural network weights, but executed through a leaner runtime, achieved **RTF ≈ 1.1–1.4x** — nearly real-time on the same hardware.

| | PyTorch | ONNX |
|---|---|---|
| RTF (cold) | 5.9x | 1.5x |
| RTF (warm) | 3–8x | 1.1–1.4x |
| Time for 7 lines, 81s audio | 483s | 123s |
| **Speedup** | — | **~4x** |

### How the Export Works (`test_onnx.py`)

Exporting PyTorch to ONNX is like converting a handwritten recipe into a standardized format that any kitchen (runtime) can read:

```python
torch.onnx.export(
    model,
    args=(dummy_ids, dummy_style, dummy_speed),
    f="kokoro.onnx",
    input_names=["input_ids", "style", "speed"],
    output_names=["waveform", "duration"],
    opset_version=17,
    dynamic_axes={...},  # variable-length sequences
    do_constant_folding=True,
)
```

Key details:
- **`opset_version=17`** — the "dialect" of ONNX. Higher versions support more ops; 17 is a safe choice for modern runtimes.
- **`dynamic_axes`** — tells ONNX that `input_ids` can have different lengths per call (different sentences have different numbers of phonemes). Without this, the exported model would only accept exactly the same input shape as the dummy inputs.
- **`do_constant_folding=True`** — a compile-time optimization: pre-compute any sub-expressions that don't depend on runtime inputs. Free speed.
- **`disable_complex=True`** on `KModel` — the STFT operations in the vocoder use complex-number tensors. ONNX doesn't support PyTorch complex ops natively, so Kokoro provides a `disable_complex` flag that swaps them for equivalent real-valued math.

### The Hybrid Trick in `OnnxKokoro`

This is the most interesting engineering pattern in the codebase. Pure ONNX inference still needs two things from the Python side:

1. **G2P (Grapheme-to-Phoneme)**: converting text like "Hello" into phoneme sequences like `/hɛloʊ/`. This is done by the `misaki` library via `KPipeline`.
2. **Voice embeddings**: each voice is a 510×256 style matrix loaded from Hugging Face Hub.

So `OnnxKokoro` does something clever: it loads `KModel` **not to run inference**, but purely to grab the vocabulary (`kmodel.vocab`). Then it passes `model=self.kmodel` to `KPipeline` so the pipeline does G2P and voice loading, while the neural network itself is replaced by ONNX Runtime.

```python
# KModel only used for vocab; ONNX session does the actual inference
self.kmodel = KModel(repo_id="hexgrad/Kokoro-82M").eval()
self.vocab  = self.kmodel.vocab
self._pipelines[lang_code] = KPipeline(lang_code=lang_code, model=self.kmodel)
```

This is a form of **dependency injection** — swapping out one component (the neural backend) without touching the rest of the system.

---

## The Script Parsing Design

### Two Formats, One Parser

The screenplay parser (`parse_script`) handles two common conventions:

**Format A** — human-readable dialogue format:
```
Alice: Hello, how are you?
Bob: I'm doing great!
```

**Format B** — XML-tag format (compatible with TTS-Story projects):
```
[alice]Hello, how are you?[/alice]
[bob]I'm doing great![/bob]
```

The parser tries Format B first (it's unambiguous), then falls back to Format A. This is a classic example of **progressive parsing**: handle the easy structured case first, then use heuristics for the looser format.

The Format A regex is worth studying:
```python
r'^([^:：\[\n]{1,40})[：:]\s*(.+)$'
```
- `[^:：\[\n]{1,40}` — character name: 1–40 chars, none of which are colons, Chinese colons, brackets, or newlines. The 40-char limit prevents stage directions from being misread as character names.
- `[：:]` — matches both the ASCII colon and the full-width Chinese colon `：`, because real-world scripts mix them.

Lines starting with `[` or `(` are silently skipped — those are stage directions, not dialogue.

### Voice Pool Round-Robin

When you have more characters than manually assigned voices, the leftover characters get voices from the default pool in order, cycling if needed:

```python
pool = [v for v in DEFAULT_VOICE_POOL if v not in mapping.values()]
mapping[char] = pool[idx % len(pool)]
```

The `% len(pool)` wrapping ensures you never get an `IndexError` even if a script has 20 characters — they just start cycling through the pool again. Simple, robust, and impossible to crash.

---

## The RTF Metric: Understanding Inference Speed

RTF (Real-Time Factor) is the standard way to measure TTS speed:

```
RTF = inference_time / audio_duration
```

- **RTF < 1.0** = faster than real-time (synthesize 10s of audio in under 10s) → usable in production
- **RTF = 1.0** = exactly real-time
- **RTF > 1.0** = slower than real-time → fine for batch jobs, bad for live use

The code logs RTF per line so you can see exactly where slowdowns happen:

```python
dur = len(line_audio) / SAMPLE_RATE   # actual audio duration
rtf = elapsed / dur                   # how long it took relative to duration
print(f"  {dur:.1f}s  ({elapsed:.1f}s,  RTF={rtf:.2f}x)")
```

---

## The Dependency Hell Problem

This is a cautionary tale every Python developer should hear.

The project targets Intel Mac (x86_64, macOS 12). The dependency chain looks innocent at first, but breaks down like a house of cards:

| Package | Constraint | Reason |
|---|---|---|
| `torch` | `≤ 2.2.2` | PyPI stopped publishing x86_64 macOS wheels at 2.4+ |
| `numpy` | `< 2` | torch 2.2.2 was compiled against NumPy 1.x ABI |
| `transformers` | `< 5` | transformers 5.x requires torch ≥ 2.4 |
| `misaki` (inside kokoro) | Python `< 3.13` | misaki's C extensions don't have 3.13 wheels yet |
| Python | `3.12` | newest version that satisfies everything above |

The fix: use `uv` to pin to Python 3.12 explicitly and install everything in one shot with all constraints:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python kokoro soundfile "numpy<2" "transformers<5" onnx onnxruntime sounddevice
```

**Lesson**: when you're targeting constrained hardware (old CPUs, specific OS versions), always test your dependency tree early. The torch wheel availability is particularly sneaky because `pip install torch` succeeds on Intel Mac — it just silently installs an older version that may conflict with everything else.

---

## Engineering Lessons

### 1. Profiling Before Optimizing

The 4x speedup from ONNX was discovered by first *measuring* the RTF carefully. Without that baseline number, we wouldn't have known whether ONNX was worth the added complexity of the export step. Always measure before you optimize.

### 2. Keep the Hot Path Thin

In `OnnxKokoro._run_segment`, the ONNX call is just 4 lines:

```python
outputs = self.session.run(None, {
    "input_ids": input_ids,
    "style":     style,
    "speed":     spd,
})
return outputs[0].squeeze()
```

No error handling, no logging, no retry logic. This is intentional. Every additional Python operation in the inner loop adds latency. Keep the hot path as thin as possible; handle errors at the outer boundary.

### 3. Lazy Initialization Is Worth It

Both `KPipeline` objects and ONNX sessions are created lazily (only when first needed for a given language). The pattern:

```python
if lang_code not in self._pipelines:
    self._pipelines[lang_code] = KPipeline(lang_code=lang_code, model=self.kmodel)
return self._pipelines[lang_code]
```

A story with 3 characters but only 2 languages will only ever load 2 pipelines. For a script mixing English and Japanese, this saves a real model download and several seconds of startup time.

### 4. WAV Format Is Simpler Than You Think

The WAV writing code reveals a useful pattern:

```python
wf.setnchannels(1)   # mono
wf.setsampwidth(2)   # 16-bit = 2 bytes per sample
wf.setframerate(24000)
wf.writeframes((audio * 32767).astype(np.int16).tobytes())
```

The model outputs `float32` in the range `[-1.0, 1.0]`. To store as 16-bit PCM: multiply by `32767` (the maximum value of a signed 16-bit integer) and cast. That single line is all it takes. Python's `wave` module handles everything else.

### 5. Test Scripts Are Documentation

`test_mac.py` and `test_onnx.py` are not just tests — they're runnable, self-explanatory documentation. Anyone who picks up this project can run `test_mac.py` and know within 30 seconds whether their environment is working. A smoke test that prints clear `[OK]` / `[FAIL]` messages with fix hints is worth more than a README paragraph.

### 6. Handling the Long-Sentence Problem

Neural TTS models have a maximum context length. Kokoro's limit is **510 phonemes** per segment. For English, `KPipeline.en_tokenize()` automatically splits long sentences at natural boundaries (punctuation, clause breaks). For other languages, the code does a hard truncation:

```python
if len(ps) > 510:
    ps = ps[:510]
```

This is an acknowledged trade-off: hard truncation can cut off mid-sentence, but it prevents a crash. A better production solution would split at word boundaries — that's a known future improvement.

---

## What "RTF ≈ 1x" Actually Means for User Experience

At RTF 1.0, synthesizing a 2-minute podcast from a script would take about 2 minutes. That sounds bad, but consider:
- The output is a fully produced audio file, not live streaming.
- The entire synthesis runs offline, no API calls, no usage fees.
- For long-form content (audiobooks, narrated stories), even RTF 3–4x is often acceptable.

The ONNX optimization pushed this project into the "plausibly practical for short content" range on consumer hardware. Adding a GPU (CUDA or MPS on Apple Silicon) would push it to RTF < 0.3x — meaning a 2-minute story synthesizes in under 40 seconds.

---

## How to Run This Project

```bash
# 1. From the repo root, set up the environment
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python kokoro soundfile "numpy<2" "transformers<5" onnx onnxruntime sounddevice

# 2. Verify the environment works
.venv/bin/python script/test_mac.py

# 3. Generate the ONNX model (one-time, ~326 MB)
.venv/bin/python script/test_onnx.py

# 4. Synthesize a screenplay (PyTorch, slow but no setup)
.venv/bin/python script/script_to_audio.py script/example_story.txt

# 5. Synthesize a screenplay (ONNX, ~4x faster)
.venv/bin/python script/script_to_audio_onnx.py script/example_story.txt
```
