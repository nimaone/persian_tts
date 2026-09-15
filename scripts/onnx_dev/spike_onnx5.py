# Spike 5 (full): Mimi decoder step -> ONNX with all state as graph I/O.
# State components (from init_states): 19 conv caches (previous/first/partial)
# + 2 transformer KV caches (offset/pad/cache). All are plain tensors.
# Strategy: pass the ENTIRE state dict (flattened) in and out of the graph.
# The .item()/int() guard lives only in attention.complete_kv; we monkeypatch
# a tensor-only version for tracing (affine slice ends as in spike 4).
import sys
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

import pocket_tts.modules.attention as attn_mod
from pocket_tts import TTSModel
from pocket_tts.modules.stateful_module import init_states, increment_steps

OUT = BASE / "onnx_export"


def complete_kv_tensor(cache, offset, k, v):
    # Export-friendly reimplementation: build the updated cache with pure
    # tensor ops (concat of prefix + new + suffix, affine slice ends) instead
    # of in-place slice assignment, which torch.export cannot guard on.
    # Returns (k_valid, v_valid, new_cache); the patched append_and_get below
    # writes new_cache back into the state dict (functional update).
    off = offset.reshape(-1)[0]
    S = k.shape[1]
    T = cache.shape[2]
    pre = cache[:, :, :off]
    post = cache[:, :, off + S :]
    new_cache = torch.cat([pre, torch.stack([k, v], dim=0).unsqueeze(1) if False else torch.stack([k, v]), post], dim=2) if False else None
    # cache: [2, B, T, H, D]; k,v: [B, S, H, D]
    kv = torch.stack([k, v], dim=0)  # [2, B, S, H, D]
    new_cache = torch.cat([cache[:, :, :off], kv, cache[:, :, off + S :]], dim=2)
    valid = new_cache[:, :, : off + S]
    return valid[0], valid[1], new_cache


attn_mod.complete_kv = complete_kv_tensor

# append_and_get must consume the new 3-tuple and store the updated cache
# back into the state dict (functional, export-friendly).
_orig_append_and_get = attn_mod._LinearKVCacheBackend.append_and_get


def append_and_get_patched(self, k, v, state):
    if state is None:
        return _orig_append_and_get(self, k, v, state)
    k_attn, v_attn, new_cache = complete_kv_tensor(state["cache"], state["offset"], k, v)
    state["cache"] = new_cache
    k_attn = k_attn.permute(0, 2, 1, 3)
    v_attn = v_attn.permute(0, 2, 1, 3)
    pos_k = torch.arange(k_attn.shape[2], device=k_attn.device, dtype=torch.long)
    pos_k = pos_k.view(1, -1).expand(k_attn.shape[0], -1)
    pad = state.get("pad")
    offset = state["offset"]
    if pad is not None:
        # pad is all-zero in single-sequence decoding; keep the arithmetic
        # unconditional (tensor ops, no data-dependent branch):
        pos_k = pos_k - pad[:, None]
        offset = offset - pad
    return k_attn, v_attn, pos_k, offset


attn_mod._LinearKVCacheBackend.append_and_get = append_and_get_patched

# SDPA guard fix: tell export the mask width is >= 1 (always true here).
_orig_attn_forward = attn_mod.StreamingMultiheadAttention.forward


NEG = -1e9


def attn_forward_patched(self, query, model_state, attn_mask=None):
    """Reimplemented StreamingMultiheadAttention.forward with explicit
    attention (matmul + softmax) instead of F.scaled_dot_product_attention,
    which torch.export cannot guard when the KV length is symbolic."""
    import torch.nn.functional as F
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
    attn_bias = attn_mask.to(q.dtype)  # [B,1,S,Lk] or broadcastable

    scores = q @ k_attn.transpose(-1, -2) * (d ** -0.5)  # [B,H,S,Lk]
    scores = scores + attn_bias
    w_ = torch.softmax(scores, dim=-1)
    x = w_ @ v_attn  # [B,H,S,D]
    x = x.transpose(1, 2).reshape(b, t, self.num_heads * d)
    x = self.out_proj(x)
    return x


attn_mod.StreamingMultiheadAttention.forward = attn_forward_patched

print("== loading ==")
tts = TTSModel.load_model(config=str(BASE / "model" / "v2" / "model.yaml"))
mimi = tts.mimi
fl = tts.flow_lm
ldim = fl.ldim

torch.manual_seed(0)
lat_a = torch.randn(1, 1, ldim) * 0.3
lat_b = torch.randn(1, 1, ldim) * 0.3
un_a = lat_a * fl.emb_std + fl.emb_mean
un_b = lat_b * fl.emb_std + fl.emb_mean

ms = init_states(mimi, batch_size=1, sequence_length=2048)
# snapshot FRESH state before any decode mutates it
flat = []
for k, v in ms.items():
    for kk, vv in v.items():
        flat.append((k, kk, vv.clone()))
print("state tensors:", len(flat))

with torch.no_grad():
    a1 = mimi.decode_from_latent(un_a, ms)
    increment_steps(mimi, ms, increment=16)
    a2 = mimi.decode_from_latent(un_b, ms)
print("torch step1:", tuple(a1.shape), "step2:", tuple(a2.shape))


class DecodeStep(torch.nn.Module):
    def __init__(self, mimi, flat_keys):
        super().__init__()
        self.mimi = mimi
        self.keys = flat_keys  # list of (module_name, state_key)

    def forward(self, latent, *state_in):
        ms = {}
        for (name, key), s in zip([(n, k) for n, k, _ in self.keys], state_in):
            ms.setdefault(name, {})[key] = s.clone()
        audio = self.mimi.decode_from_latent(latent, ms)
        # offsets incremented by 16 (steps_per_latent) — do it in-graph:
        for name in ms:
            for key in ms[name]:
                if key == "offset":
                    ms[name][key] = ms[name][key] + 16
        outs = [audio]
        for name, key, _ in self.keys:
            outs.append(ms[name][key])
        return tuple(outs)


w = DecodeStep(mimi, flat).eval()
state0 = tuple(vv for _, _, vv in flat)

with torch.no_grad():
    r1_t = w(un_a, *state0)
    r2_t = w(un_b, *r1_t[1:])
d1 = (r1_t[0] - a1).abs().max().item()
d2 = (r2_t[0] - a2).abs().max().item()
print(f"wrap vs orig: step1 {d1:.2e}  step2 {d2:.2e}")
print("r1 nan:", torch.isnan(r1_t[0]).any().item(), "r2 nan:", torch.isnan(r2_t[0]).any().item(),
      "| a1 nan:", torch.isnan(a1).any().item(), "a2 nan:", torch.isnan(a2).any().item())

# ONNX export
print("== ONNX export ==")
in_names = ["latent"] + [f"s{i}" for i in range(len(flat))]
out_names = ["audio"] + [f"s{i}o" for i in range(len(flat))]
with torch.no_grad():
    torch.onnx.export(
        w, (un_a, *state0), str(OUT / "mimi_decoder_step.onnx"),
        input_names=in_names, output_names=out_names, opset_version=18,
    )
sz = (OUT / "mimi_decoder_step.onnx").stat().st_size / 1e6
print(f"export OK: {sz:.1f} MB")

import onnxruntime as ort

sess = ort.InferenceSession(str(OUT / "mimi_decoder_step.onnx"), providers=["CPUExecutionProvider"])
names_in = [i.name for i in sess.get_inputs()]
names_out = [o.name for o in sess.get_outputs()]


def run(latent_np, state_list):
    feeds = {names_in[0]: latent_np}
    for n, t in zip(names_in[1:], state_list):
        feeds[n] = t
    return sess.run(None, feeds)


st = [t.numpy() for t in state0]
r1 = run(un_a.numpy(), st)
r2 = run(un_b.numpy(), r1[1:])
da1 = np.abs(r1[0] - a1.numpy()).max()
da2 = np.abs(r2[0] - a2.numpy()).max()
print(f"ORT vs torch: step1 {da1:.2e}  step2 {da2:.2e}")
print("done")
