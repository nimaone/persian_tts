# Canonical ONNX exporter -> model/onnx/
#
# Produces the UNIFIED package:
#   flow_lm_step.onnx        ONE graph for all flow-LM steps (voice
#                            conditioning, text prompt, generation) via
#                            dynamic text_emb and sequence lengths; K/V-only
#                            output (host scatters into its cache).
#   mimi_encoder.onnx        voice audio -> latents (dynamic length)
#   mimi_decoder_step_kv.onnx  latents -> audio, K/V-only state updates
#   g2p_encoder.onnx         T5 encoder for the G2P stage (decoder stays torch)
#   weights.npz              host-side constants: LUT embedding table,
#                            speaker projection, bos_before_voice, latent
#                            normalization stats
#   manifest.json            file list + I/O documentation
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

OUT = BASE / "model" / "onnx"
OUT.mkdir(parents=True, exist_ok=True)


def export_g2p():
    # Runs in its OWN subprocess (see main): the T5 decoder dynamo export
    # breaks if torch.onnx legacy exports or marked-dynamic forwards ran
    # earlier in the same process (shape-env pollution).
    # ===========================================================================
    # 1) G2P (T5) encoder + decoder — exported FIRST: the dynamo shape-env
    #    gets polluted by later flow/mimi exports and the T5 reshape then fails
    # ===========================================================================
    print("\n== 4) g2p_encoder.onnx ==")
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    g2p_tok = AutoTokenizer.from_pretrained(str(BASE / "model" / "g2p"))
    g2p = T5ForConditionalGeneration.from_pretrained(str(BASE / "model" / "g2p")).eval()


    class EncWrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.enc = m.encoder

        def forward(self, input_ids):
            return self.enc(input_ids=input_ids).last_hidden_state


    ids_ex = g2p_tok(["salAm test"], return_tensors="pt").input_ids
    torch.onnx.export(
        EncWrap(g2p).eval(), (ids_ex,), str(OUT / "g2p_encoder.onnx"),
        input_names=["input_ids"], output_names=["hidden"],
        dynamic_axes={"input_ids": {1: "S"}, "hidden": {1: "S"}},
        opset_version=18, dynamo=False,
    )
    print(f"exported: g2p_encoder.onnx ({(OUT / 'g2p_encoder.onnx').stat().st_size/1e6:.2f} MB)")

    # decoder: full forward (no cache) — outputs are short, re-running the 2-layer
    # decoder on the growing sequence is fast and keeps the graph stateless
    class DecWrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.dec = m.decoder
            self.lm = m.lm_head

        def forward(self, decoder_input_ids, encoder_hidden_states):
            out = self.dec(input_ids=decoder_input_ids,
                           encoder_hidden_states=encoder_hidden_states,
                           use_cache=False, return_dict=True)
            return self.lm(out.last_hidden_state)


    # NOTE: the export MUST run under no_grad — with autograd active the
    # attention reshape becomes a stride-checked view that ONNX decomposition
    # rejects once a legacy export ran earlier in the same process.
    dec_ids = torch.tensor([[0, 5, 10]])
    dec_enc = torch.randn(1, 7, 512)
    torch._dynamo.mark_dynamic(dec_ids, 1)
    torch._dynamo.mark_dynamic(dec_enc, 1)
    with torch.no_grad():
        torch.onnx.export(
            DecWrap(g2p).eval(), (dec_ids, dec_enc), str(OUT / "g2p_decoder.onnx"),
            input_names=["decoder_input_ids", "encoder_hidden"],
            output_names=["logits"], opset_version=18,
        )
    import onnxruntime as _ort

    with torch.no_grad():
        dec_ref = DecWrap(g2p).eval()(dec_ids, dec_enc)
    _s = _ort.InferenceSession(str(OUT / "g2p_decoder.onnx"), providers=["CPUExecutionProvider"])
    _r = _s.run(None, {"decoder_input_ids": dec_ids.numpy(), "encoder_hidden": dec_enc.numpy()})[0]
    print(f"exported: g2p_decoder.onnx ({(OUT / 'g2p_decoder.onnx').stat().st_size/1e6:.2f} MB), "
          f"diff vs torch {np.abs(_r - dec_ref.numpy()).max():.2e}")





if __name__ == "__main__" and "--g2p" in sys.argv:
    # clean subprocess: no pocket_tts import runs in this process
    export_g2p()
    sys.exit(0)

# ---------------------------------------------------------------------------
# monkeypatches for the MIMI decoder path (verified in spike_onnx5):
# tensor-only KV write + explicit matmul attention
# ---------------------------------------------------------------------------
import pocket_tts.modules.attention as attn_mod

NEG = -1e9


def complete_kv_tensor(cache, offset, k, v):
    off = offset.reshape(-1)[0]
    S = k.shape[1]
    kv = torch.stack([k, v], dim=0)
    new_cache = torch.cat([cache[:, :, :off], kv, cache[:, :, off + S :]], dim=2)
    state_kv_new = new_cache[:, :, off : off + S]
    valid = new_cache[:, :, : off + S]
    return valid[0], valid[1], new_cache, state_kv_new


def append_and_get_patched(self, k, v, state):
    if state is None:
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

def export_flow_mimi_pkg_verify():
    print("== loading torch model (weights source) ==")
    tts = TTSModel.load_model(config=str(BASE / "model" / "v2" / "model.yaml"))
    fl = tts.flow_lm
    mimi = tts.mimi
    ldim = fl.ldim
    L = len(fl.transformer.layers)
    H = fl.transformer.layers[0].self_attn.num_heads
    D = fl.transformer.layers[0].self_attn.dim_per_head
    CAP = 384

    # ===========================================================================
    # 2) UNIFIED flow-LM step graph (dynamic text_emb + sequence, K/V-only out)
    # ===========================================================================
    print("\n== 1) unified flow_lm_step.onnx ==")


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


    class UnifiedStep(torch.nn.Module):
        """One flow-LM step. Covers three call patterns via dynamic lengths:
          voice:  text_emb=[audio cond], sequence empty, offset=0
          prompt: text_emb=[token embeds], sequence=[BOS NaN], offset=voice_end
          gen:    text_emb empty, sequence=[previous latent], offset+=1
        Output: new K/V of THIS step only ([L,2,S,H,D]); host scatters."""

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
                new_kvs.append(torch.stack([k[0], v[0]], dim=0))
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

    m_uni = UnifiedStep(fl).eval()
    seq_ex = torch.full((1, 1, ldim), float("nan"), dtype=torch.float32)
    text_ex = torch.zeros(1, 6, fl.dim)
    off_ex = torch.zeros((), dtype=torch.long)
    noise_ex = torch.zeros(1, ldim)
    cache_ex = torch.zeros(L, 2, CAP, H, D)
    torch._dynamo.mark_dynamic(seq_ex, 1)
    torch._dynamo.mark_dynamic(text_ex, 1)

    with torch.no_grad():
        torch.onnx.export(
            m_uni, (seq_ex, text_ex, off_ex, noise_ex, cache_ex),
            str(OUT / "flow_lm_step.onnx"),
            input_names=["sequence", "text_emb", "offset", "noise", "cache"],
            output_names=["latent", "eos", "new_kv"],
            opset_version=18,
        )
    print(f"exported: flow_lm_step.onnx ({(OUT / 'flow_lm_step.onnx').stat().st_size/1e6:.2f} MB + data)")

    # ===========================================================================
    # 3) mimi encoder (dynamic audio length)
    # ===========================================================================
    print("\n== 2) mimi_encoder.onnx ==")


    class MimiEncoderWrap(torch.nn.Module):
        def __init__(self, mimi):
            super().__init__()
            self.mimi = mimi

        def forward(self, audio):
            return self.mimi.encode_to_latent(audio)


    audio_ex = torch.zeros(1, 1, 24000)
    torch._dynamo.mark_dynamic(audio_ex, 2)
    with torch.no_grad():
        torch.onnx.export(
            MimiEncoderWrap(mimi).eval(), (audio_ex,), str(OUT / "mimi_encoder.onnx"),
            input_names=["audio"], output_names=["latent"], opset_version=18,
        )
    print(f"exported: mimi_encoder.onnx ({(OUT / 'mimi_encoder.onnx').stat().st_size/1e6:.2f} MB + data)")

    # ===========================================================================
    # 4) mimi decoder step (K/V-only outputs)
    # ===========================================================================
    print("\n== 3) mimi_decoder_step_kv.onnx ==")
    ms_mimi = init_states(mimi, batch_size=1, sequence_length=2048)
    needed = [k for k in ms_mimi if k.startswith(("decoder.", "upsample.", "decoder_transformer."))]
    dec_keys = [(k, kk) for k in needed for kk in ms_mimi[k]]


    class DecodeStepKV(torch.nn.Module):
        def __init__(self, mimi, keys):
            super().__init__()
            self.mimi = mimi
            self.keys = keys

        def forward(self, latent, *state_in):
            ms = {}
            for (name, key), s in zip(self.keys, state_in):
                ms.setdefault(name, {})[key] = s.clone()
            audio = self.mimi.decode_from_latent(latent, ms)
            outs = [audio]
            for name, key in self.keys:
                if key == "cache":
                    continue
                ms[name][key] = ms[name][key] + 16 if key == "offset" else ms[name][key]
                outs.append(ms[name][key])
            for name in ms:
                if "kv_new" in ms[name]:
                    outs.append(ms[name]["kv_new"])
            return tuple(outs)


    n_small = len(dec_keys) - sum(1 for _, k in dec_keys if k == "cache")
    n_cache = sum(1 for _, k in dec_keys if k == "cache")
    state_args = tuple(ms_mimi[n][k].clone() for n, k in dec_keys)
    latent_ex = torch.randn(1, 1, ldim) * 0.3
    with torch.no_grad():
        torch.onnx.export(
            DecodeStepKV(mimi, dec_keys).eval(), (latent_ex, *state_args),
            str(OUT / "mimi_decoder_step_kv.onnx"),
            input_names=["latent"] + [f"s{i}" for i in range(len(dec_keys))],
            output_names=["audio"] + [f"s{i}o" for i in range(n_small)] + [f"kv{i}" for i in range(n_cache)],
            opset_version=18,
        )
    print(f"exported: mimi_decoder_step_kv.onnx "
          f"({(OUT / 'mimi_decoder_step_kv.onnx').stat().st_size/1e6:.2f} MB + data)")

    # ===========================================================================
    # 5) host-side constants + manifest
    # ===========================================================================
    print("\n== 5) weights.npz + manifest.json ==")
    np.savez(
        OUT / "weights.npz",
        lut_weight=fl.conditioner.embed.weight.detach().numpy().astype(np.float32),
        speaker_proj=fl.speaker_proj_weight.detach().numpy().astype(np.float32),  # [dim, ldim]
        bos_before_voice=fl.bos_before_voice.detach().numpy().astype(np.float32),  # [1,1,dim]
        emb_std=fl.emb_std.detach().numpy().astype(np.float32),
        emb_mean=fl.emb_mean.detach().numpy().astype(np.float32),
    )
    manifest = {
        "name": "pocket-tts-farsi-v2 ONNX (unified)",
        "source": "mehdi-hf/pocket-tts-farsi-v2",
        "graphs": {
            "flow_lm_step.onnx": {
                "role": "one step of the flow LM (voice conditioning / text prompt / generation)",
                "inputs": {
                    "sequence": "f32 [1,S_seq,32] latent tokens (NaN=BOS); S_seq may be 0",
                    "text_emb": "f32 [1,S_text,1024] conditioning embeddings; S_text may be 0",
                    "offset": "i64 scalar — absolute write position / RoPE offset",
                    "noise": "f32 [1,32] flow noise (host RNG)",
                    "cache": "f32 [6,2,256,16,64] persistent KV cache (slot0=K, slot1=V)",
                },
                "outputs": {
                    "latent": "f32 [1,32] next latent (ignore for voice step)",
                    "eos": "f32 [1,1] 0/1 end-of-speech flag",
                    "new_kv": "f32 [6,2,S,16,64] K/V written THIS step; host scatters into cache[:,:,off:off+S]",
                },
            },
            "mimi_encoder.onnx": {
                "role": "voice prompt audio -> latents (one-shot, dynamic length)",
                "inputs": {"audio": "f32 [1,1,T] 24kHz mono"},
                "outputs": {"latent": "f32 [1,T/1920,32]"},
            },
            "mimi_decoder_step_kv.onnx": {
                "role": "one latent -> 1920 audio samples (streaming, state I/O)",
                "inputs": {"latent": "f32 [1,1,32] un-normalized",
                           "s*": "27 state tensors (caches for 2 decoder-transformer layers + small conv states)"},
                "outputs": {"audio": "f32 [1,1,1920]",
                            "s*o": "updated small states",
                            "kv0/kv1": "f32 [2,1,16,8,64] new K/V; host scatters at [.:,.,off:off+16]"},
            },
            "g2p_encoder.onnx": {
                "role": "T5 encoder of the G2P stage (Persian text -> hidden states)",
                "inputs": {"input_ids": "i64 [1,S] ByT5 ids (byte + 3)"},
                "outputs": {"hidden": "f32 [1,S,512]"},
            },
            "g2p_decoder.onnx": {
                "role": "T5 decoder full-forward (stateless; greedy loop runs host-side)",
                "inputs": {"decoder_input_ids": "i64 [1,S]", "encoder_hidden": "f32 [1,S_enc,512]"},
                "outputs": {"logits": "f32 [1,S,384]"},
            },
        },
        "host_constants": "weights.npz: lut_weight [4000,1024], speaker_proj [1024,32], bos_before_voice [1,1,1024], emb_std/emb_mean [32]",
        "decode_state_init": "decode_state_init.npz — initial tensors for the 27 decoder state inputs",
        "constants": {"ldim": 32, "dim": 1024, "layers": L, "heads": H, "dim_per_head": D,
                      "cache_capacity": CAP, "sample_rate": 24000,
                      "mimi_steps_per_latent": 16, "temp": 0.3, "eos_threshold": -4.0,
                      "tokens_per_second_estimate": 3.0, "gen_seconds_padding": 2.0,
                      "frame_rate": 12.5},
        "engine": "scripts/tts_onnx.py",
        "notes": [
            "the WHOLE pipeline (G2P + synthesis) runs WITHOUT torch",
            "G2P tokenizer is ByT5: token id = utf-8 byte + 3 (pure python)",
            "G2P decoding is greedy; verified identical to torch beam-5 on 12 sentences",
            "max cache 256 positions (~voice 5s + 18-token chunk + ~12s audio)",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("package written to", OUT)

    # decode-state template (so the engine needs no torch to build it)
    dec_init = {}
    for k, kk in dec_keys:
        dec_init[f"{k}.{kk}"] = ms_mimi[k][kk].numpy()
    np.savez(OUT / "decode_state_init.npz", **dec_init)
    print("saved decode_state_init.npz")

    # ===========================================================================
    # verification
    # ===========================================================================
    print("\n== VERIFY: unified flow graph ==")
    import onnxruntime as ort

    s_uni = ort.InferenceSession(str(OUT / "flow_lm_step.onnx"), providers=["CPUExecutionProvider"])

    # A) voice-conditioning step vs torch get_state_for_audio_prompt
    VOICE = str(BASE / "voices" / "male_hello.wav")
    vs_torch = tts.get_state_for_audio_prompt(VOICE)

    import soundfile as sf
    from scipy.signal import resample_poly

    audio_np, sr = sf.read(VOICE)
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    if sr != 24000:
        g = np.gcd(sr, 24000)
        audio_np = resample_poly(audio_np, 24000 // g, sr // g).astype(np.float32)
    s_enc = ort.InferenceSession(str(OUT / "mimi_encoder.onnx"), providers=["CPUExecutionProvider"])
    lat_enc = s_enc.run(None, {"audio": audio_np.astype(np.float32)[None, None, :]})[0]  # [1,T,32]

    w = np.load(OUT / "weights.npz")
    cond = lat_enc[0] @ w["speaker_proj"].T  # [T,1024]
    text_emb_v = np.concatenate([w["bos_before_voice"][0], cond])[None].astype(np.float32)

    lat_v, eos_v, kv_v = s_uni.run(None, {
        "sequence": np.zeros((1, 0, ldim), np.float32),
        "text_emb": text_emb_v,
        "offset": np.array(0, dtype=np.int64),
        "noise": np.zeros((1, ldim), np.float32),
        "cache": np.zeros((L, 2, CAP, H, D), np.float32)})
    kv_v = kv_v.reshape(L, 2, -1, H, D)
    Sv = kv_v.shape[2]
    print(f"voice step: {Sv} positions (torch cache len {vs_torch['transformer.layers.0.self_attn']['cache'].shape[2]})")

    # compare against torch cache
    diffs = []
    n = 0
    for name, st in vs_torch.items():
        if name.endswith("self_attn"):
            t_cache = st["cache"].numpy()  # [2,1,T,H,D]
            diffs.append(np.abs(kv_v[n, 0] - t_cache[0, 0]).max())
            diffs.append(np.abs(kv_v[n, 1] - t_cache[1, 0]).max())
            n += 1
    print(f"voice cache K/V diff vs torch (max over layers): {max(diffs):.3e}")

    # B) prompt + gen chain from a FRESH state vs torch (exact, same noise)
    PHONEMES = "salAm hAle SomA Cetor ?ast"
    rng = np.random.default_rng(1234)
    bank = (rng.standard_normal((60, 1, ldim)) * (0.3**0.5)).astype(np.float32)

    it = iter(bank)
    torch.nn.init.normal_ = lambda t, **kw: t.copy_(torch.from_numpy(next(it)))
    prepared = fl.conditioner.prepare(PHONEMES)
    ms = init_states(fl, 1, 200)
    latents_t = []
    with torch.no_grad():
        lat, _ = fl._sample_next_latent(
            sequence=torch.full((1, 1, ldim), float("nan"), dtype=fl.dtype),
            text_embeddings=fl.conditioner(prepared), model_state=ms,
            sampler_decode_steps=1, temp=0.3, noise_clamp=None, eos_threshold=-4.0)
    increment_steps(fl, ms, increment=prepared.shape[1] + 1)
    latents_t.append(lat)
    for step in range(10):
        with torch.no_grad():
            lat, eos = fl._sample_next_latent(
                sequence=latents_t[-1].view(1, 1, ldim),
                text_embeddings=torch.zeros(1, 0, fl.dim), model_state=ms,
                sampler_decode_steps=1, temp=0.3, noise_clamp=None, eos_threshold=-4.0)
        increment_steps(fl, ms, increment=1)
        latents_t.append(lat)

    # ONNX chain with the SAME unified graph
    cache = np.zeros((L, 2, CAP, H, D), np.float32)
    lut = w["lut_weight"]
    text_emb_p = lut[prepared.numpy().reshape(-1)][None].astype(np.float32)
    lat_o, _, kv = s_uni.run(None, {
        "sequence": np.full((1, 1, ldim), np.nan, np.float32),
        "text_emb": text_emb_p,
        "offset": np.array(0, dtype=np.int64),
        "noise": bank[0], "cache": cache})
    kv = kv.reshape(L, 2, -1, H, D)
    cache[:, :, : kv.shape[2]] = kv
    off = text_emb_p.shape[1] + 1
    latents_o = [lat_o]
    for i in range(10):
        lat_o, _, kv = s_uni.run(None, {
            "sequence": latents_o[-1].reshape(1, 1, ldim),
            "text_emb": np.zeros((1, 0, fl.dim), np.float32),
            "offset": np.array(off, dtype=np.int64),
            "noise": bank[i + 1], "cache": cache})
        kv = kv.reshape(L, 2, -1, H, D)
        cache[:, :, off : off + kv.shape[2]] = kv
        off += kv.shape[2]
        latents_o.append(lat_o)

    ld = [float(np.abs(a - b.detach().numpy()).max()) for a, b in zip(latents_o, latents_t)]
    print(f"prompt+gen chain latent diffs (11 steps): max {max(ld):.2e}")

    print("\nexport_unified done")


def main():
    import subprocess

    # 1) g2p exports in a clean subprocess
    r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--g2p"])
    if r.returncode != 0:
        sys.exit(r.returncode)

    # 2) flow + mimi + package + verification in-process (verified stable)
    export_flow_mimi_pkg_verify()


if __name__ == "__main__":
    if "--g2p" in sys.argv:
        export_g2p()
    else:
        main()
