# Final validation: run FlowLM ONNX graph for 3 CHAINED steps with cache
# threading, and compare against the original torch path (noise=0 both sides).
import sys
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

import onnxruntime as ort
from pocket_tts import TTSModel
from pocket_tts.modules.stateful_module import init_states, increment_steps

tts = TTSModel.load_model(config=str(BASE / "model" / "v2" / "model.yaml"))
fl = tts.flow_lm
L = len(fl.transformer.layers)
H = fl.transformer.layers[0].self_attn.num_heads
D = fl.transformer.layers[0].self_attn.dim_per_head
CAP, ldim = 256, fl.ldim

torch.nn.init.normal_ = lambda t, **kw: t.detach().zero_()
torch.nn.init.trunc_normal_ = lambda t, **kw: t.detach().zero_()

text = "salAm hAle SomA Cetor ?ast"
prepared = fl.conditioner.prepare(text)
text_emb = fl.conditioner(prepared)
S_text = text_emb.shape[1]
empty_emb = torch.zeros(1, 0, fl.dim)

# ---- torch reference chain (original code path) ----
ms = init_states(fl, 1, 200)
latents_t = []
seq = torch.full((1, 1, ldim), float("NaN"), dtype=fl.dtype)
with torch.no_grad():
    lat, _ = fl._sample_next_latent(
        sequence=seq, text_embeddings=text_emb, model_state=ms,
        sampler_decode_steps=1, temp=0.3, noise_clamp=None, eos_threshold=-4.0)
    increment_steps(fl, ms, increment=S_text + 1)  # production advances by text+latent
    latents_t.append(lat)
    for _ in range(2):
        lat, _ = fl._sample_next_latent(
            sequence=latents_t[-1].view(1, 1, ldim), text_embeddings=empty_emb,
            model_state=ms, sampler_decode_steps=1, temp=0.3,
            noise_clamp=None, eos_threshold=-4.0)
        increment_steps(fl, ms, increment=1)
        latents_t.append(lat)
print("torch chain:", len(latents_t), "steps")

# ---- clean-room torch chain (spike-4 Step wrapper; same as graph) ----
ns = {"__file__": str(BASE / "scripts" / "spike_onnx4.py")}
src = (BASE / "scripts" / "spike_onnx4.py").read_text(encoding="utf-8")
# keep only definitions (imports + rope + Step class); drop the runtime part
head = src.split("m = Step(fl).eval()")[0]
# cut the runtime sections that call the model while keeping the class def
import re
defs_only = "\n".join(
    line for line in head.splitlines()
    if not line.startswith(("print(", "text =", "prepared =", "text_emb =", "ms_ref", "torch.manual_seed"))
)
# neutralize stray top-level statements from the spike file
defs_only = re.sub(r"^with torch\.no_grad\(\):$", "if False:", defs_only, flags=re.M)
defs_only = re.sub(r"^lat_ref, eos_ref = .*$", "pass", defs_only, flags=re.M)
defs_only = re.sub(r"^backbone_input = .*$", "backbone_input = None", defs_only, flags=re.M)
exec(compile(defs_only, "spike4_head", "exec"), ns)
Step = ns["Step"]

m = Step(fl).eval()
kc = torch.zeros(L, 2, CAP, H, D)
vc = torch.zeros(L, 2, CAP, H, D)
noise = torch.zeros(1, ldim)
latents_c = []
with torch.no_grad():
    lat, _, kc, vc = m(seq, text_emb, torch.zeros((), dtype=torch.long), noise, kc, vc)
    latents_c.append(lat)
    off = S_text + 1
    for _ in range(2):
        lat, _, kc, vc = m(latents_c[-1].view(1, 1, ldim), empty_emb,
                           torch.tensor(off), noise, kc, vc)
        latents_c.append(lat)
        off += 1

dt = [float((a - b).abs().max()) for a, b in zip(latents_c, latents_t)]
print("clean-room vs original torch:", [f"{x:.2e}" for x in dt])

# ---- ONNX chain ----
sess = ort.InferenceSession(str(BASE / "onnx_export" / "flow_lm_step.onnx"),
                            providers=["CPUExecutionProvider"])
sess_gen = ort.InferenceSession(str(BASE / "onnx_export" / "flow_lm_gen_step.onnx"),
                                providers=["CPUExecutionProvider"])
kc_o = np.zeros((L, 2, CAP, H, D), dtype=np.float32)
vc_o = np.zeros((L, 2, CAP, H, D), dtype=np.float32)
latents_o = []
lat, eos, kc_o, vc_o = sess.run(None, {
    "sequence": seq.numpy(), "text_emb": text_emb.detach().numpy(),
    "offset": np.array(0, dtype=np.int64), "noise": noise.numpy(),
    "k_cache": kc_o, "v_cache": vc_o})
latents_o.append(lat)
off = S_text + 1
for _ in range(2):
    lat, eos, kc_o, vc_o = sess_gen.run(None, {
        "sequence": latents_o[-1].reshape(1, 1, ldim), "text_emb": empty_emb.numpy(),
        "offset": np.array(off, dtype=np.int64), "noise": noise.numpy(),
        "k_cache": kc_o, "v_cache": vc_o})
    latents_o.append(lat)
    off += 1

do = [float(np.abs(a - b.numpy()).max()) for a, b in zip(latents_o, latents_c)]
print("ONNX vs clean-room torch:  ", [f"{x:.2e}" for x in do])
print("CHAIN VALIDATION:", "OK" if max(do) < 1e-4 else "MISMATCH")
