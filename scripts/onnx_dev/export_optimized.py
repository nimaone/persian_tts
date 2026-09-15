# Optimized ONNX exports:
#   flow_lm_gen_step_kv.onnx   — gen step returning ONLY the new K/V
#                                ([L,2,S,H,D], ~50KB) instead of the whole
#                                12.6MB cache; host scatters it into its
#                                persistent cache.
#   mimi_decoder_step_kv.onnx  — decode step returning audio + new K/V for the
#                                2 decoder-transformer layers + small conv
#                                states, instead of 56 tensors incl. 33MB of
#                                caches.
# Reuses the verified clean-room implementations from spike_onnx4/5.
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

OUT = BASE / "onnx_export"

# ---------------------------------------------------------------------------
# monkeypatches (verified in spike_onnx5): tensor-only KV write + explicit
# matmul attention (SDPA with bool mask hits a data-dependent export guard)
# ---------------------------------------------------------------------------
import pocket_tts.modules.attention as attn_mod

NEG = -1e9


def complete_kv_tensor(cache, offset, k, v):
    off = offset.reshape(-1)[0]
    S = k.shape[1]
    kv = torch.stack([k, v], dim=0)  # [2,B,S,H,D]
    new_cache = torch.cat([cache[:, :, :off], kv, cache[:, :, off + S :]], dim=2)
    # stash the freshly-written slice so the wrapper can output just that
    state_kv_new = new_cache[:, :, off : off + S]
    valid = new_cache[:, :, : off + S]
    return valid[0], valid[1], new_cache, state_kv_new


def append_and_get_patched(self, k, v, state):
    if state is None:
        # stateless path: only used outside streaming — not reached here
        k_attn = k.permute(0, 2, 1, 3)
        v_attn = v.permute(0, 2, 1, 3)
        pos_k = torch.arange(k_attn.shape[2], device=k_attn.device, dtype=torch.long)
        pos_k = pos_k.view(1, -1).expand(k_attn.shape[0], -1)
        return k_attn, v_attn, pos_k, torch.zeros(k.shape[0], dtype=torch.long)
    k_attn, v_attn, new_cache, kv_new = complete_kv_tensor(state["cache"], state["offset"], k, v)
    state["cache"] = new_cache
    state["kv_new"] = kv_new
    k_attn = k_attn.permute(0, 2, 1, 3)
    v_attn = v_attn.permute(0, 2, 1, 3)
    pos_k = torch.arange(k_attn.shape[2], device=k_attn.device, dtype=torch.long)
    pos_k = pos_k.view(1, -1).expand(k_attn.shape[0], -1)
    pad = state.get("pad")
    offset = state["offset"]
    if pad is not None:
        pos_k = pos_k - pad[:, None]
        offset = offset - pad
    return k_attn, v_attn, pos_k, offset


def attn_forward_patched(self, query, model_state, attn_mask=None):
    """Explicit matmul+softmax attention (SDPA export guard workaround)."""
    from pocket_tts.modules.attention import _build_attention_mask

    state = None if model_state is None else self.get_state(model_state)
    projected = self.in_proj(query)
    b, t, _ = projected.shape
    d = self.dim_per_head
    packed = projected.view(b, t, 3, self.num_heads, d)
    q, k, v = torch.unbind(packed, dim=2)
    rope_offset = self._cache_backend.rope_offset(state, b, q.device)
    q, k = self.rope(q, k, offset=rope_offset)
    q = q.transpose(1, 2)

    k_attn, v_attn, pos_k, offset = self._cache_backend.append_and_get(k, v, state)
    if attn_mask is None:
        if state is None:
            pos = torch.arange(t, device=q.device, dtype=torch.long).view(1, -1)
            attn_mask = _build_attention_mask(pos, pos, self.context)
        else:
            pos_q = offset.view(-1, 1) + torch.arange(t, device=q.device, dtype=torch.long).view(1, -1)
            attn_mask = _build_attention_mask(pos_q, pos_k, self.context)
    if attn_mask.dtype == torch.bool:
        attn_mask = torch.where(attn_mask, 0.0, NEG)
    attn_bias = attn_mask.to(q.dtype)

    scores = q @ k_attn.transpose(-1, -2) * (d**-0.5)
    scores = scores + attn_bias
    w_ = torch.softmax(scores, dim=-1)
    x = w_ @ v_attn
    x = x.transpose(1, 2).reshape(b, t, self.num_heads * d)
    return self.out_proj(x)


attn_mod.complete_kv = lambda cache, offset, k, v: complete_kv_tensor(cache, offset, k, v)[:2]
attn_mod._LinearKVCacheBackend.append_and_get = append_and_get_patched
attn_mod.StreamingMultiheadAttention.forward = attn_forward_patched

from pocket_tts import TTSModel
from pocket_tts.modules.stateful_module import init_states, increment_steps

print("== loading model ==")
tts = TTSModel.load_model(config=str(BASE / "model" / "v2" / "model.yaml"))
fl = tts.flow_lm
mimi = tts.mimi

# ===========================================================================
# 1) FlowLM gen step, K/V-only output
# ===========================================================================
L = len(fl.transformer.layers)
H = fl.transformer.layers[0].self_attn.num_heads
D = fl.transformer.layers[0].self_attn.dim_per_head
CAP = 256
ldim = fl.ldim


def rope(q, k, offset):
    B, T, Hh, Dd = q.shape
    ds = torch.arange(Dd // 2, device=q.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-torch.log(torch.tensor(10000.0)) * 2 / Dd)).view(1, 1, 1, -1)
    ts = torch.arange(T, device=q.device, dtype=torch.float32) + offset.to(torch.float32)
    ts = ts.view(1, -1, 1, 1)
    qr = q.view(B, T, Hh, Dd // 2, 2)[..., 0].float()
    qi = q.view(B, T, Hh, Dd // 2, 2)[..., 1].float()
    kr = k.view(B, T, Hh, Dd // 2, 2)[..., 0].float()
    ki = k.view(B, T, Hh, Dd // 2, 2)[..., 1].float()
    rotr, roti = torch.cos(freqs * ts), torch.sin(freqs * ts)
    qo = torch.stack([qr * rotr - qi * roti, qr * roti + qi * rotr], dim=-1).to(q.dtype)
    ko = torch.stack([kr * rotr - ki * roti, kr * roti + ki * rotr], dim=-1).to(k.dtype)
    return qo.view(B, T, Hh, Dd), ko.view(B, T, Hh, Dd)


class StepKV(torch.nn.Module):
    """One generation step (S=1): full cache IN, new K/V OUT (host scatters)."""

    def __init__(self, fl):
        super().__init__()
        self.fl = fl

    def forward(self, sequence, text_emb, offset, noise, cache):
        fl = self.fl
        bos = fl.bos_emb.view(1, 1, ldim).expand_as(sequence)
        seq = torch.where(torch.isnan(sequence), bos.to(sequence.dtype), sequence)
        x = fl.input_linear(seq)
        x = torch.cat([text_emb, x], dim=1)
        S = x.shape[1]

        k_out = cache
        new_kvs = []
        h = x
        for i, layer in enumerate(fl.transformer.layers):
            attn = layer.self_attn
            xn = layer.norm1(h)
            proj = attn.in_proj(xn)
            packed = proj.view(1, S, 3, H, D)
            q, k, v = torch.unbind(packed, dim=2)
            off = offset + 0
            q, k = rope(q, k, off)
            q = q.transpose(1, 2)
            k_l = k_out[i, 0]
            v_l = k_out[i, 1]
            k_full = torch.cat([k_l[:off], k[0], k_l[off + S :]], dim=0)
            v_full = torch.cat([v_l[:off], v[0], v_l[off + S :]], dim=0)
            new_slot = torch.stack([k_full, v_full], dim=0).unsqueeze(0)
            k_out = torch.cat([k_out[:i], new_slot, k_out[i + 1 :]], dim=0)
            new_kvs.append(torch.stack([k[0], v[0]], dim=0))  # [2,S,H,D]
            k_attn = k_full[: off + S].permute(1, 0, 2).unsqueeze(0)
            v_attn = v_full[: off + S].permute(1, 0, 2).unsqueeze(0)
            Lk = k_attn.shape[2]
            torch._check(Lk >= 1, "cache length must be at least 1")
            pos_k = torch.arange(Lk, device=x.device, dtype=torch.long).view(1, -1)
            pos_q = (off + torch.arange(S, device=x.device, dtype=torch.long)).view(1, -1)
            delta = pos_q[:, :, None] - pos_k[:, None, :]
            mask = (pos_k[:, None, :] >= 0) & (delta >= 0)
            att = F.scaled_dot_product_attention(q, k_attn, v_attn, mask[:, None])
            att = att.transpose(1, 2).reshape(1, S, H * D)
            h = h + attn.out_proj(att)
            xnf = layer.norm2(h)
            h = h + layer.linear2(F.gelu(layer.linear1(xnf), approximate="tanh"))

        h = fl.out_norm(h)
        h_last = h[:, -1].to(torch.float32)
        eos = (fl.out_eos(h_last) > -4.0).to(torch.float32)
        nfr = fl.flow_net.time_embed[0].freqs.shape[0]
        s0 = torch.zeros(1, nfr, dtype=torch.float32)
        t1 = torch.ones(1, nfr, dtype=torch.float32)
        u = fl.flow_net(h_last, s0, t1, noise)
        latent = noise + u
        new_kv = torch.stack(new_kvs, dim=0)  # [L,2,S,H,D]
        return latent, eos, new_kv


torch.nn.init.normal_ = lambda t, **kw: t.detach().zero_()
torch.nn.init.trunc_normal_ = lambda t, **kw: t.detach().zero_()

text = "salAm hAle SomA Cetor ?ast"
prepared = fl.conditioner.prepare(text)
text_emb_full = fl.conditioner(prepared)
ms = init_states(fl, 1, 200)
backbone_input = torch.full((1, 1, ldim), float("NaN"), dtype=fl.dtype)
with torch.no_grad():
    lat_ref, eos_ref = fl._sample_next_latent(
        sequence=backbone_input, text_embeddings=text_emb_full, model_state=ms,
        sampler_decode_steps=1, temp=0.3, noise_clamp=None, eos_threshold=-4.0)
increment_steps(fl, ms, increment=prepared.shape[1] + 1)

cache0 = torch.zeros(L, 2, CAP, H, D)
off_t = torch.tensor(0)
noise0 = torch.zeros(1, ldim)

m = StepKV(fl).eval()
# reference: full-cache graph (already on disk) vs KV graph — same inputs
import onnxruntime as ort

s_full = ort.InferenceSession(str(OUT / "flow_lm_gen_step.onnx"), providers=["CPUExecutionProvider"])

print("== export flow gen step (K/V only) ==")
with torch.no_grad():
    torch.onnx.export(
        m, (backbone_input, torch.zeros(1, 0, fl.dim), off_t, noise0, cache0),
        str(OUT / "flow_lm_gen_step_kv.onnx"),
        input_names=["sequence", "text_emb", "offset", "noise", "cache"],
        output_names=["latent", "eos", "new_kv"],
        opset_version=18,
    )
print(f"OK: {(OUT / 'flow_lm_gen_step_kv.onnx').stat().st_size/1e6:.2f} MB")

# verify: run both graphs on identical inputs (seeded cache), compare latent
# and the scattered cache vs the full graph's cache output.
rng = np.random.default_rng(7)
cache_in = np.zeros((L, 2, CAP, H, D), dtype=np.float32)
cache_in[:, :, :62] = rng.standard_normal((L, 2, 62, H, D)).astype(np.float32)
seq_in = rng.standard_normal((1, 1, ldim)).astype(np.float32)
noise_in = rng.standard_normal((1, ldim)).astype(np.float32)
off_in = 62

lat_full, eos_full, cache_out_full = s_full.run(None, {
    "sequence": seq_in, "text_emb": np.zeros((1, 0, fl.dim), np.float32),
    "offset": np.array(off_in, dtype=np.int64), "noise": noise_in, "cache": cache_in})

s_kv = ort.InferenceSession(str(OUT / "flow_lm_gen_step_kv.onnx"), providers=["CPUExecutionProvider"])
lat_kv, eos_kv, new_kv = s_kv.run(None, {
    "sequence": seq_in, "text_emb": np.zeros((1, 0, fl.dim), np.float32),
    "offset": np.array(off_in, dtype=np.int64), "noise": noise_in, "cache": cache_in})

# host scatter (reshape in case the exporter dropped the size-1 S dim)
nk = new_kv.reshape(L, 2, -1, H, D)
Sn = nk.shape[2]
cache_host = cache_in.copy()
cache_host[:, :, off_in : off_in + Sn] = nk
print(f"latent diff (kv vs full graph): {np.abs(lat_kv - lat_full).max():.2e}")
print(f"cache diff after scatter:      {np.abs(cache_host - cache_out_full).max():.2e}")

# ===========================================================================
# 2) Mimi decoder step, K/V-only outputs for the 2 decoder-transformer layers
# ===========================================================================
print("\n== export mimi decoder step (K/V only) ==")
ms_mimi = init_states(mimi, batch_size=1, sequence_length=2048)

# modules actually used by decode_from_latent:
needed = [k for k in ms_mimi if k.startswith(("decoder.", "upsample.", "decoder_transformer."))]
flat_keys = [(k, kk) for k in needed for kk in ms_mimi[k]]
print(f"state tensors (decode path only): {len(flat_keys)}")
big = [f"{k}.{kk}" for k, kk in flat_keys if kk == "cache"]
print("big cache tensors:", big)


class DecodeStepKV(torch.nn.Module):
    def __init__(self, mimi, keys):
        super().__init__()
        self.mimi = mimi
        self.keys = keys  # [(module, state_key)]

    def forward(self, latent, *state_in):
        ms = {}
        for (name, key), s in zip(self.keys, state_in):
            ms.setdefault(name, {})[key] = s.clone()
        audio = self.mimi.decode_from_latent(latent, ms)
        outs = [audio]
        for name, key in self.keys:
            if key == "cache":
                continue  # big; host applies kv_new instead
            ms[name][key] = ms[name][key] + 16 if key == "offset" else ms[name][key]
            outs.append(ms[name][key])
        for name in ms:
            if "kv_new" in ms[name]:
                outs.append(ms[name]["kv_new"])  # [2,B,S,H,D]
        return tuple(outs)


w = DecodeStepKV(mimi, flat_keys).eval()
state_args = tuple(ms_mimi[n][k].clone() for n, k in flat_keys)
latent_in = torch.randn(1, 1, ldim) * 0.3

with torch.no_grad():
    torch.onnx.export(
        w, (latent_in, *state_args), str(OUT / "mimi_decoder_step_kv.onnx"),
        input_names=["latent"] + [f"s{i}" for i in range(len(flat_keys))],
        output_names=["audio"] + [f"s{i}o" for i in range(len(flat_keys) - 2)] + ["kv0", "kv1"],
        opset_version=18,
    )
print(f"OK: {(OUT / 'mimi_decoder_step_kv.onnx').stat().st_size/1e6:.2f} MB")

# verify vs full-state graph over 2 chained steps
import onnxruntime as ort2

s_dec_full = ort.InferenceSession(str(OUT / "mimi_decoder_step.onnx"), providers=["CPUExecutionProvider"])
names_in_full = [i.name for i in s_dec_full.get_inputs()]
msA = init_states(mimi, batch_size=1, sequence_length=2048)
all_keys_full = [(k, kk) for k, v in msA.items() for kk in v]
lat_a = torch.randn(1, 1, ldim) * 0.3
lat_b = torch.randn(1, 1, ldim) * 0.3

# full graph: 2 steps
st_full = [msA[n][kk].numpy().copy() for n, kk in all_keys_full]
un_a = (lat_a.numpy() * fl.emb_std.numpy() + fl.emb_mean.numpy()).astype(np.float32)
un_b = (lat_b.numpy() * fl.emb_std.numpy() + fl.emb_mean.numpy()).astype(np.float32)
feeds = {names_in_full[0]: un_a}
for nm, arr in zip(names_in_full[1:], st_full):
    feeds[nm] = arr
r1 = s_dec_full.run(None, feeds)
feeds = {names_in_full[0]: un_b}
for nm, arr in zip(names_in_full[1:], r1[1:]):
    feeds[nm] = arr
r2 = s_dec_full.run(None, feeds)

# kv graph: 2 steps with host scatter
s_dec_kv = ort.InferenceSession(str(OUT / "mimi_decoder_step_kv.onnx"), providers=["CPUExecutionProvider"])
kv_in_names = [i.name for i in s_dec_kv.get_inputs()]
kv_out_names = [o.name for o in s_dec_kv.get_outputs()]
print("kv graph inputs:", len(kv_in_names), "outputs:", len(kv_out_names))

st_kv = {f"s{i}": ms_mimi[n][k].numpy().copy() for i, (n, k) in enumerate(flat_keys)}
cache_idx = {i for i, (n, k) in enumerate(flat_keys) if k == "cache"}


def dec_kv_step(un, st, off16):
    feeds = {"latent": un}
    for i, (n, k) in enumerate(flat_keys):
        feeds[f"s{i}"] = st[f"s{i}"]
    res = s_dec_kv.run(None, feeds)
    audio, small, kvs = res[0], res[1:1 + len(flat_keys) - 2], res[1 + len(flat_keys) - 2:]
    # apply small outputs in order (skip cache slots) and scatter new K/V
    j = 0
    for i, (n, k) in enumerate(flat_keys):
        if k == "cache":
            continue
        st[f"s{i}"] = small[j]
        j += 1
    for bi, base in enumerate(sorted(cache_idx)):
        st[f"s{base}"][:, :, off16 : off16 + 16] = kvs[bi]
    return audio, kvs


a1, _ = dec_kv_step(un_a, st_kv, 0)
a2, _ = dec_kv_step(un_b, st_kv, 16)
print(f"decode step1 diff: {np.abs(a1 - r1[0]).max():.2e}")
print(f"decode step2 diff: {np.abs(a2 - r2[0]).max():.2e}")

print("\nexport_optimized done")
