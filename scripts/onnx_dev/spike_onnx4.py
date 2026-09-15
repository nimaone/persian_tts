# Spike 4 (final): clean-room FlowLM single step -> ONNX with external
# KV-cache + offset as graph inputs. Verifies numerics against the original
# torch path (noise patched to zeros on both sides).
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

from pocket_tts import TTSModel
from pocket_tts.modules.stateful_module import init_states

OUT = BASE / "onnx_export"
OUT.mkdir(exist_ok=True)

print("== loading ==")
tts = TTSModel.load_model(config=str(BASE / "model" / "v2" / "model.yaml"))
fl = tts.flow_lm
ldim, dim = fl.ldim, fl.dim

text = "salAm hAle SomA Cetor ?ast"
prepared = fl.conditioner.prepare(text)
text_emb = fl.conditioner(prepared)
backbone_input = torch.full((1, 1, ldim), float("NaN"), dtype=fl.dtype)

torch.nn.init.normal_ = lambda t, **kw: t.zero_()
torch.nn.init.trunc_normal_ = lambda t, **kw: t.zero_()

ms_ref = init_states(fl, 1, 200)
with torch.no_grad():
    lat_ref, eos_ref = fl._sample_next_latent(
        sequence=backbone_input, text_embeddings=text_emb, model_state=ms_ref,
        sampler_decode_steps=1, temp=0.3, noise_clamp=None, eos_threshold=-4.0,
    )
print("ref latent:", lat_ref.shape, "eos:", eos_ref.shape, float(eos_ref.float()))

# ---------------------------------------------------------------------------
# Clean-room step. Math identical to FlowLMModel.forward for one step:
#   NaN->BOS, input_linear, [text ; x] -> transformer (6 layers, RoPE,
#   external KV cache, affine slice ends on offset) -> out_norm -> out_eos
#   -> flow_net (2 time conds, LSD 1-step from host-supplied noise)
# ---------------------------------------------------------------------------
L = len(fl.transformer.layers)
H = fl.transformer.layers[0].self_attn.num_heads
D = fl.transformer.layers[0].self_attn.dim_per_head
CAP = 256  # cache capacity (static); host keeps offset


def rope(q, k, offset):
    # q,k: [B, S, H, D]
    B, T, Hh, Dd = q.shape
    ds = torch.arange(Dd // 2, device=q.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-torch.log(torch.tensor(10000.0)) * 2 / Dd)).view(1, 1, 1, -1)  # [1,1,1,D/2]
    ts = torch.arange(T, device=q.device, dtype=torch.float32) + offset.to(torch.float32)
    ts = ts.view(1, -1, 1, 1)  # broadcast over [B,T,H,D/2]
    qr, qi = q.view(B, T, Hh, Dd // 2, 2)[..., 0].float(), q.view(B, T, Hh, Dd // 2, 2)[..., 1].float()
    kr, ki = k.view(B, T, Hh, Dd // 2, 2)[..., 0].float(), k.view(B, T, Hh, Dd // 2, 2)[..., 1].float()
    rotr, roti = torch.cos(freqs * ts), torch.sin(freqs * ts)
    qo = torch.stack([qr * rotr - qi * roti, qr * roti + qi * rotr], dim=-1).to(q.dtype)
    ko = torch.stack([kr * rotr - ki * roti, kr * roti + ki * rotr], dim=-1).to(k.dtype)
    return qo.view(B, T, Hh, Dd), ko.view(B, T, Hh, Dd)


class Step(torch.nn.Module):
    def __init__(self, fl):
        super().__init__()
        self.fl = fl  # keep refs to submodules (weights shared)

    def forward(self, sequence, text_emb, offset, noise, cache):
        # sequence [1,1,ldim]; text_emb [1,S,dim]; offset 0-d int64; noise [1,ldim]
        # cache [L,2,CAP,H,D]: slot 0 = K, slot 1 = V per layer
        bos = self.fl.bos_emb.view(1, 1, ldim).expand_as(sequence)
        seq = torch.where(torch.isnan(sequence), bos.to(sequence.dtype), sequence)
        x = self.fl.input_linear(seq)                       # [1,1,dim]
        x = torch.cat([text_emb, x], dim=1)                 # [1,S+1,dim]
        S = x.shape[1]

        k_out = cache
        h = x
        for i, layer in enumerate(self.fl.transformer.layers):
            attn = layer.self_attn
            # --- attention block ---
            xn = layer.norm1(h)
            proj = attn.in_proj(xn)                          # [1,S,3*dim]
            packed = proj.view(1, S, 3, H, D)
            q, k, v = torch.unbind(packed, dim=2)
            off = offset + 0
            q, k = rope(q, k, off)
            q = q.transpose(1, 2)                            # [1,H,S,D]
            # cache write: k_out cache: [L,2,1,CAP,H,D]; this layer's K,V: [B,S,H,D]
            k_l = k_out[i, 0]   # [CAP,H,D]
            v_l = k_out[i, 1]   # [CAP,H,D]  (combined cache: slot1=V)
            k_full = torch.cat([k_l[:off], k[0], k_l[off + S :]], dim=0)   # [CAP,H,D]
            v_full = torch.cat([v_l[:off], v[0], v_l[off + S :]], dim=0)
            # functional write-back: slot [1,2,CAP,H,D] at layer i of cache [L,2,CAP,H,D]
            new_slot = torch.stack([k_full, v_full], dim=0).unsqueeze(0)  # [1,2,CAP,H,D]
            k_out = torch.cat([k_out[:i], new_slot, k_out[i + 1 :]], dim=0)
            k_attn = k_full[: off + S].permute(1, 0, 2).unsqueeze(0)   # [1,H,off+S,D]
            v_attn = v_full[: off + S].permute(1, 0, 2).unsqueeze(0)
            Lk = k_attn.shape[2]
            torch._check(Lk >= 1, "cache length must be at least 1")
            pos_k = torch.arange(Lk, device=x.device, dtype=torch.long).view(1, -1)
            pos_q = (off + torch.arange(S, device=x.device, dtype=torch.long)).view(1, -1)
            delta = pos_q[:, :, None] - pos_k[:, None, :]
            mask = (pos_k[:, None, :] >= 0) & (delta >= 0)   # context is None for flow lm
            att = F.scaled_dot_product_attention(q, k_attn, v_attn, mask[:, None])  # [1,H,S,D]
            att = att.transpose(1, 2).reshape(1, S, H * D)                          # [1,S,dim]
            h = h + attn.out_proj(att)
            # --- ff block ---
            xnf = layer.norm2(h)
            h = h + layer.linear2(F.gelu(layer.linear1(xnf), approximate="tanh"))
        # out: original slices out the text prefix; last position == our -1
        h = fl.out_norm(h)
        h_last = h[:, -1].to(torch.float32)                  # [1,dim]
        eos = (fl.out_eos(h_last) > -4.0).to(torch.float32) # [1,1] bool like original
        # flow net: LSD with 2 time conds, 1 step: s=0,t=1
        # time tensors must match TimestepEmbedder.freqs size (128 here)
        nfr = fl.flow_net.time_embed[0].freqs.shape[0]
        s0 = torch.zeros(1, nfr, dtype=torch.float32)
        t1 = torch.ones(1, nfr, dtype=torch.float32)
        u = fl.flow_net(h_last, s0, t1, noise)               # [1,ldim]
        latent = noise + u
        return latent, eos, k_out


# cache tensors: [L,2,1,CAP,H,D] (L layers stacked)
cache0 = torch.zeros(L, 2, CAP, H, D)
offset = torch.zeros((), dtype=torch.long)
noise = torch.zeros(1, ldim)

m = Step(fl).eval()
with torch.no_grad():
    lat_cr, eos_cr, _ = m(backbone_input, text_emb, offset, noise, cache0)
print("clean-room latent:", lat_cr.shape,
      "| diff vs ref:", (lat_cr - lat_ref).abs().max().item(),
      "| eos diff:", (eos_cr.float() - eos_ref.float()).abs().max().item())

# ---------------------------------------------------------------------------
# ONNX export + ORT verification
# ---------------------------------------------------------------------------
import onnxruntime as ort

print()
print("== ONNX export ==")
with torch.no_grad():
    torch.onnx.export(
        m, (backbone_input, text_emb, offset, noise, cache0),
        str(OUT / "flow_lm_step.onnx"),
        input_names=["sequence", "text_emb", "offset", "noise", "cache"],
        output_names=["latent", "eos", "cache_new"],
        opset_version=18,
    )
sz = (OUT / "flow_lm_step.onnx").stat().st_size / 1e6
print(f"export OK: {sz:.1f} MB")

# second graph: generation step with EMPTY text embedding (S_text=0)
empty_emb = torch.zeros(1, 0, fl.dim)
off6 = torch.tensor(6)
with torch.no_grad():
    torch.onnx.export(
        m, (backbone_input, empty_emb, off6, noise, cache0),
        str(OUT / "flow_lm_gen_step.onnx"),
        input_names=["sequence", "text_emb", "offset", "noise", "cache"],
        output_names=["latent", "eos", "cache_new"],
        opset_version=18,
    )
print(f"gen-step export OK: {(OUT / 'flow_lm_gen_step.onnx').stat().st_size / 1e6:.1f} MB")

sess = ort.InferenceSession(str(OUT / "flow_lm_step.onnx"), providers=["CPUExecutionProvider"])
feeds = {
    "sequence": backbone_input.numpy(),
    "text_emb": text_emb.detach().numpy(),
    "offset": offset.numpy(),
    "noise": noise.numpy(),
    "cache": cache0.numpy(),
}
res = sess.run(None, feeds)
lat_ort = res[0]
d = np.abs(lat_ort - lat_ref.numpy()).max()
print(f"ORT latent diff vs torch: {d:.2e}")
print("done")
