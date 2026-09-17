# Validation + benchmark for the unified pure-ONNX engine (scripts/tts_onnx.py)
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))
sys.path.insert(0, str(BASE / "scripts" / "onnx_dev"))

import soundfile as sf

PHONEMES = "salAm hAle SomA Cetor ?ast"
VOICE = str(BASE / "voices" / "male_hello.wav")

from pocket_tts import TTSModel
from pocket_tts.modules.stateful_module import init_states, increment_steps

print("== torch reference ==")
tts = TTSModel.load_model(config=str(BASE / "model" / "v2" / "model.yaml"))
fl = tts.flow_lm

# noise bank identical to OnnxTts(seed=1234) sequential draws
rng = np.random.default_rng(1234)
bank = (rng.standard_normal((120, 1, fl.ldim)) * (0.3**0.5)).astype(np.float32)

from bench_onnx import TorchEngine

torch_eng = TorchEngine(tts)

print("== accuracy ==")
vs = tts.get_state_for_audio_prompt(VOICE)
import copy

vs_snap = copy.deepcopy(vs)
audio_t, lat_t = torch_eng.synthesize(vs, PHONEMES, bank)
print(f"torch: {len(audio_t)/24000:.2f}s, {len(lat_t)} latents")

from tts_onnx import OnnxTts

eng = OnnxTts(seed=1234)

# 1) voice cache agreement (ONNX voice path vs torch)
cache_o, off_o = eng.voice_cache(VOICE)
diffs = []
n = 0
for name, st in vs_snap.items():
    if name.endswith("self_attn"):
        diffs.append(np.abs(cache_o[n, 0, :off_o] - st["cache"][0, 0].numpy()).max())
        diffs.append(np.abs(cache_o[n, 1, :off_o] - st["cache"][1, 0].numpy()).max())
        n += 1
print(f"voice cache diff vs torch: {max(diffs):.3e}")

# 2) full synthesis, same noise; engine voice path (its own conditioning)
audio_o = eng.synthesize(PHONEMES, VOICE, seed=1234)
n = min(len(audio_t), len(audio_o))
rms = np.sqrt((audio_t[:n] ** 2).mean())
rel = np.sqrt(((audio_t[:n] - audio_o[:n]) ** 2).mean()) / rms
print(f"onnx : {len(audio_o)/24000:.2f}s | rel waveform diff vs torch: {rel:.3e}")
print(f"onnx audio peak: {np.abs(audio_o).max():.3f} (non-silent: {np.abs(audio_o).max() > 0.01})")

print("\n== speed (median of 5) ==")


def bench(fn, n=5, warmup=1):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def torch_prod():
    vs = tts.get_state_for_audio_prompt(VOICE)
    with torch.no_grad():
        tts.generate_audio(vs, PHONEMES)


import torch

t1 = bench(torch_prod)
t2 = bench(lambda: eng.synthesize(PHONEMES, VOICE, seed=1234))
dur = len(audio_o) / 24000
print(f"torch production : {t1*1000:6.0f} ms  RTF {dur/t1:.2f}x realtime")
print(f"ONNX unified     : {t2*1000:6.0f} ms  RTF {dur/t2:.2f}x realtime")

sf.write(str(BASE / "output" / "tts_onnx_verify.wav"), audio_o, 24000)
print("saved output/tts_onnx_verify.wav")
