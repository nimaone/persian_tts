# Benchmark of the three optimization paths vs the stock engines:
#   1. K/V-only output graphs (host scatters new K/V into persistent caches)
#   2. IOBinding zero-copy input binding (persistent numpy buffers)
#   3. Hybrid engine: FlowLM in torch + Mimi (encoder+decoder) in ONNX
import sys
import time
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

import onnxruntime as ort
from pocket_tts import TTSModel
from pocket_tts.modules.stateful_module import init_states, increment_steps

ORT_DIR = BASE / "onnx_export"
PHONEMES = "salAm hAle SomA Cetor ?ast"
VOICE = str(BASE / "voices" / "female_hello.wav")
TEMP, EOS_THRESHOLD = 0.3, -4.0


def noise_bank(n, ldim, seed=1234):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, 1, ldim)) * (TEMP**0.5)).astype(np.float32)


def bench(fn, n=10, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


class OptimizedOnnxEngine:
    """Full ONNX engine using K/V-only graphs + optional IOBinding."""

    def __init__(self, tts, iobinding=False):
        fl = tts.flow_lm
        self.tts = tts
        self.fl = fl
        self.ldim = fl.ldim
        self.L = len(fl.transformer.layers)
        self.H = fl.transformer.layers[0].self_attn.num_heads
        self.D = fl.transformer.layers[0].self_attn.dim_per_head
        self.CAP = 256
        self.std = fl.emb_std.detach().numpy()
        self.mean = fl.emb_mean.detach().numpy()
        self.iobinding = iobinding

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.s_prompt = ort.InferenceSession(str(ORT_DIR / "flow_lm_step.onnx"), opts,
                                             providers=["CPUExecutionProvider"])
        self.s_gen = ort.InferenceSession(str(ORT_DIR / "flow_lm_gen_step_kv.onnx"), opts,
                                          providers=["CPUExecutionProvider"])
        self.s_dec = ort.InferenceSession(str(ORT_DIR / "mimi_decoder_step_kv.onnx"), opts,
                                          providers=["CPUExecutionProvider"])

        # mimi decode key layout (must match export_optimized.py flat_keys)
        ms = init_states(tts.mimi, batch_size=1, sequence_length=2048)
        needed = [k for k in ms if k.startswith(("decoder.", "upsample.", "decoder_transformer."))]
        self.dec_keys = [(k, kk) for k in needed for kk in ms[k]]
        self.dec_cache_idx = [i for i, (n, k) in enumerate(self.dec_keys) if k == "cache"]
        self.dec_small_names = [f"s{i}" for i, (n, k) in enumerate(self.dec_keys) if k != "cache"]
        # output layout: audio, small states (in dec_keys order, caches skipped), kv0, kv1
        n_small = len(self.dec_keys) - len(self.dec_cache_idx)
        self.dec_out_small = list(range(1, 1 + n_small))
        self.dec_out_kv = list(range(1 + n_small, 1 + n_small + len(self.dec_cache_idx)))

        if iobinding:
            self.ib = self.s_gen.io_binding()
            # persistent input buffers
            self.buf_seq = np.zeros((1, 1, self.ldim), np.float32)
            self.buf_noise = np.zeros((1, self.ldim), np.float32)
            self.buf_cache = np.zeros((self.L, 2, self.CAP, self.H, self.D), np.float32)
            self.buf_off = np.zeros((), np.int64)

    def seed_flow_cache(self, voice_state):
        kc = np.zeros((self.L, 2, self.CAP, self.H, self.D), dtype=np.float32)
        n = 0
        offset = None
        for name, st in voice_state.items():
            if name.endswith("self_attn"):
                t = st["cache"].shape[2]
                kc[n, 0, :t] = st["cache"][0, 0].numpy()
                kc[n, 1, :t] = st["cache"][1, 0].numpy()
                n += 1
                if offset is None:
                    offset = int(st["offset"][0])
        return kc, offset

    def _gen_step(self, seq, cache, off, noise):
        if not self.iobinding:
            lat, eos, new_kv = self.s_gen.run(None, {
                "sequence": seq, "text_emb": np.zeros((1, 0, self.fl.dim), np.float32),
                "offset": np.array(off, dtype=np.int64), "noise": noise, "cache": cache})
            nk = new_kv.reshape(self.L, 2, -1, self.H, self.D)
            cache[:, :, off : off + nk.shape[2]] = nk
            return lat, eos, cache

        # IOBinding path: bind persistent buffers (zero-copy), run, scatter
        ib = self.ib
        ib.clear_binding_inputs(); ib.clear_binding_outputs()
        self.buf_seq[...] = seq
        self.buf_noise[...] = noise
        self.buf_cache[...] = cache
        self.buf_off[()] = off
        ib.bind_cpu_input("sequence", self.buf_seq)
        ib.bind_cpu_input("text_emb", np.zeros((1, 0, self.fl.dim), np.float32))
        ib.bind_cpu_input("offset", self.buf_off)
        ib.bind_cpu_input("noise", self.buf_noise)
        ib.bind_cpu_input("cache", self.buf_cache)
        ib.bind_output("latent"); ib.bind_output("eos"); ib.bind_output("new_kv")
        self.s_gen.run_with_iobinding(ib)
        lat = ib.get_outputs()[0].numpy()
        eos = ib.get_outputs()[1].numpy()
        new_kv = ib.get_outputs()[2].numpy()
        nk = new_kv.reshape(self.L, 2, -1, self.H, self.D)
        cache = self.buf_cache.copy()
        cache[:, :, off : off + nk.shape[2]] = nk
        return lat, eos, cache

    def synthesize(self, cache, off, phonemes, bank, max_frames=80):
        fl = self.fl
        prepared = fl.conditioner.prepare(phonemes)
        text_emb = fl.conditioner.embed.weight.detach().numpy()[prepared.numpy().reshape(-1)][None].astype(np.float32)
        bos = np.full((1, 1, self.ldim), np.nan, dtype=np.float32)
        lat, eos, cache = self.s_prompt.run(None, {
            "sequence": bos, "text_emb": text_emb,
            "offset": np.array(off, dtype=np.int64), "noise": bank[0], "cache": cache})
        latents = [lat]
        off += text_emb.shape[1] + 1

        eos_step = None
        i = 1
        for step in range(max_frames):
            lat, eos, cache = self._gen_step(latents[-1].reshape(1, 1, self.ldim), cache, off, bank[i])
            i += 1
            off += 1
            if bool(eos.reshape(-1)[0]) and eos_step is None:
                eos_step = step
            if eos_step is not None and step >= eos_step + 2:
                break
            latents.append(lat)

        # mimi decode (K/V graph + host scatter)
        st = {f"s{i}": init_states(self.tts.mimi, 1, 2048)[n][k].numpy().copy()
              for i, (n, k) in enumerate(self.dec_keys)}
        out = []
        off16 = 0
        for lt in latents:
            un = (lt.reshape(1, 1, self.ldim) * self.std + self.mean).astype(np.float32)
            feeds = {"latent": un}
            for j, nm in enumerate(self.dec_small_names):
                # small names enumerate non-cache slots in order
                idx = [i for i, (n, k) in enumerate(self.dec_keys) if k != "cache"][j]
                feeds[f"s{idx}"] = st[f"s{idx}"]
            for base in self.dec_cache_idx:
                feeds[f"s{base}"] = st[f"s{base}"]
            res = self.s_dec.run(None, feeds)
            out.append(res[0])
            for j, o in enumerate(self.dec_out_small):
                idx = [i for i, (n, k) in enumerate(self.dec_keys) if k != "cache"][j]
                st[f"s{idx}"] = res[o]
            for bi, base in enumerate(self.dec_cache_idx):
                st[f"s{base}"][:, :, off16 : off16 + 16] = res[self.dec_out_kv[bi]]
            off16 += 16
        return np.concatenate(out, axis=2)[0, 0], latents


class HybridEngine:
    """FlowLM in torch (in-place cache) + Mimi encoder/decoder in ONNX."""

    def __init__(self, tts):
        self.tts = tts
        self.fl = tts.flow_lm
        self.ldim = tts.flow_lm.ldim
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.s_enc = ort.InferenceSession(str(ORT_DIR / "mimi_encoder.onnx"), opts,
                                          providers=["CPUExecutionProvider"])
        self.s_dec = ort.InferenceSession(str(ORT_DIR / "mimi_decoder_step_kv.onnx"), opts,
                                          providers=["CPUExecutionProvider"])
        ms = init_states(tts.mimi, batch_size=1, sequence_length=2048)
        needed = [k for k in ms if k.startswith(("decoder.", "upsample.", "decoder_transformer."))]
        self.dec_keys = [(k, kk) for k in needed for kk in ms[k]]
        self.dec_cache_idx = [i for i, (n, k) in enumerate(self.dec_keys) if k == "cache"]
        n_small = len(self.dec_keys) - len(self.dec_cache_idx)
        self.dec_out_small = list(range(1, 1 + n_small))
        self.dec_out_kv = list(range(1 + n_small, 1 + n_small + len(self.dec_cache_idx)))

    def voice_state_onnx(self, voice_wav):
        """Voice conditioning with ONNX mimi encoder + numpy speaker_proj,
        flow voice step still in torch (hybrid)."""
        import soundfile as sf
        from pocket_tts.data.audio_utils import convert_audio

        audio, sr = sf.read(voice_wav)
        audio_t = torch.from_numpy(audio.astype(np.float32))
        if audio_t.ndim == 1:
            audio_t = audio_t.unsqueeze(0)          # [1, T] like audio_read
        audio_t = convert_audio(audio_t, sr, 24000, 1)  # [1, T]
        lat = self.s_enc.run(None, {"audio": audio_t.unsqueeze(0).numpy()})[0]  # [1,T,32]
        lat_t = torch.from_numpy(lat)
        cond = lat_t @ self.fl.speaker_proj_weight.T  # [1,T,dim]
        if self.fl.insert_bos_before_voice:
            cond = torch.cat([self.fl.bos_before_voice, cond], dim=1)
        ms = init_states(self.fl, batch_size=1, sequence_length=cond.shape[1] + 80)
        self.tts._run_flow_lm_and_increment_step(model_state=ms, audio_conditioning=cond)
        return ms

    def synthesize(self, voice_ms, phonemes, bank, max_frames=80):
        fl = self.fl
        prepared = fl.conditioner.prepare(phonemes)
        self.tts._expand_kv_cache(voice_ms, sequence_length=100 + prepared.shape[1] + max_frames + 8)

        it = iter(bank)
        orig_normal = torch.nn.init.normal_
        torch.nn.init.normal_ = lambda t, **kw: t.copy_(torch.from_numpy(next(it)))
        ms = voice_ms
        latents = []
        try:
            with torch.no_grad():
                lat, _ = fl._sample_next_latent(
                    sequence=torch.full((1, 1, fl.ldim), float("nan"), dtype=fl.dtype),
                    text_embeddings=fl.conditioner(prepared), model_state=ms,
                    sampler_decode_steps=1, temp=TEMP, noise_clamp=None, eos_threshold=EOS_THRESHOLD)
            increment_steps(fl, ms, increment=prepared.shape[1] + 1)
            latents.append(lat)

            eos_step = None
            empty = torch.zeros(1, 0, fl.dim)
            for step in range(max_frames):
                with torch.no_grad():
                    lat, eos = fl._sample_next_latent(
                        sequence=latents[-1].view(1, 1, fl.ldim), text_embeddings=empty,
                        model_state=ms, sampler_decode_steps=1, temp=TEMP,
                        noise_clamp=None, eos_threshold=EOS_THRESHOLD)
                increment_steps(fl, ms, increment=1)
                if bool(eos) and eos_step is None:
                    eos_step = step
                if eos_step is not None and step >= eos_step + 2:
                    break
                latents.append(lat)
        finally:
            torch.nn.init.normal_ = orig_normal

        # ONNX mimi decode
        st = {f"s{i}": init_states(self.tts.mimi, 1, 2048)[n][k].numpy().copy()
              for i, (n, k) in enumerate(self.dec_keys)}
        std, mean = fl.emb_std.numpy(), fl.emb_mean.numpy()
        out = []
        off16 = 0
        small_idx = [i for i, (n, k) in enumerate(self.dec_keys) if k != "cache"]
        for lt in latents:
            un = (lt.view(1, 1, fl.ldim).numpy() * std + mean).astype(np.float32)
            feeds = {"latent": un}
            for idx in small_idx:
                feeds[f"s{idx}"] = st[f"s{idx}"]
            for base in self.dec_cache_idx:
                feeds[f"s{base}"] = st[f"s{base}"]
            res = self.s_dec.run(None, feeds)
            out.append(res[0])
            for j, o in enumerate(self.dec_out_small):
                st[f"s{small_idx[j]}"] = res[o]
            for bi, base in enumerate(self.dec_cache_idx):
                st[f"s{base}"][:, :, off16 : off16 + 16] = res[self.dec_out_kv[bi]]
            off16 += 16
        return np.concatenate(out, axis=2)[0, 0], latents


def main():
    print("== loading ==")
    tts = TTSModel.load_model(config=str(BASE / "model" / "v2" / "model.yaml"))
    fl = tts.flow_lm
    bank = noise_bank(120, fl.ldim)

    import copy

    from bench_onnx import TorchEngine  # reuse verified torch mirror engine

    torch_eng = TorchEngine(tts)
    opt_eng = OptimizedOnnxEngine(tts, iobinding=False)
    opt_eng_io = OptimizedOnnxEngine(tts, iobinding=True)
    hyb_eng = HybridEngine(tts)

    # ---------------- accuracy ----------------
    print("\n== ACCURACY (identical noise) ==")
    vs = tts.get_state_for_audio_prompt(VOICE)
    vs_snap = copy.deepcopy(vs)
    audio_t, lat_t = torch_eng.synthesize(vs, PHONEMES, bank)
    cache_seed, off_seed = opt_eng.seed_flow_cache(vs_snap)
    audio_o, lat_o = opt_eng.synthesize(cache_seed, off_seed, PHONEMES, bank)
    n = min(len(audio_t), len(audio_o))
    rms = np.sqrt((audio_t[:n] ** 2).mean())
    print(f"torch {len(audio_t)/24000:.2f}s | onnx-opt {len(audio_o)/24000:.2f}s | "
          f"rel diff {np.sqrt(((audio_t[:n]-audio_o[:n])**2).mean())/rms:.3e}")

    # hybrid accuracy (its own voice state via ONNX encoder)
    vs_h = hyb_eng.voice_state_onnx(VOICE)
    audio_h, lat_h = hyb_eng.synthesize(vs_h, PHONEMES, bank)
    n = min(len(audio_t), len(audio_h))
    print(f"hybrid {len(audio_h)/24000:.2f}s | rel diff vs torch "
          f"{np.sqrt(((audio_t[:n]-audio_h[:n])**2).mean())/rms:.3e}")

    # ---------------- component speed ----------------
    print("\n== SPEED: flow gen step (40-step chain) ==")

    def torch_chain():
        ms = tts.get_state_for_audio_prompt(VOICE)
        with torch.no_grad():
            prepared = fl.conditioner.prepare(PHONEMES)
            tts._expand_kv_cache(ms, sequence_length=tts._flow_lm_current_end(ms) + 60)
            fl._sample_next_latent(
                sequence=torch.full((1, 1, fl.ldim), float("nan"), dtype=fl.dtype),
                text_embeddings=fl.conditioner(prepared), model_state=ms,
                sampler_decode_steps=1, temp=TEMP, noise_clamp=None, eos_threshold=EOS_THRESHOLD)
            increment_steps(fl, ms, prepared.shape[1] + 1)
            seq = torch.zeros(1, 1, fl.ldim)
            for _ in range(40):
                lat, _ = fl._sample_next_latent(
                    sequence=seq, text_embeddings=torch.zeros(1, 0, fl.dim),
                    model_state=ms, sampler_decode_steps=1, temp=TEMP,
                    noise_clamp=None, eos_threshold=EOS_THRESHOLD)
                increment_steps(fl, ms, increment=1)
                seq = lat.view(1, 1, fl.ldim)

    def onnx_kv_chain(engine):
        vs = tts.get_state_for_audio_prompt(VOICE)
        cache, off = engine.seed_flow_cache(vs)
        prepared = fl.conditioner.prepare(PHONEMES)
        te = fl.conditioner.embed.weight.detach().numpy()[prepared.numpy().reshape(-1)][None].astype(np.float32)
        bos = np.full((1, 1, engine.ldim), np.nan, np.float32)
        _, _, cache = engine.s_prompt.run(None, {
            "sequence": bos, "text_emb": te, "offset": np.array(off, dtype=np.int64),
            "noise": np.zeros((1, fl.ldim), np.float32), "cache": cache})
        off += te.shape[1] + 1
        seq = np.zeros((1, 1, engine.ldim), np.float32)
        for i in range(40):
            lat, _, cache = engine._gen_step(seq, cache, off, np.zeros((1, fl.ldim), np.float32))
            off += 1
            seq = lat.reshape(1, 1, engine.ldim)

    t1 = bench(torch_chain, n=5, warmup=1)
    t2 = bench(lambda: onnx_kv_chain(opt_eng), n=5, warmup=1)
    t3 = bench(lambda: onnx_kv_chain(opt_eng_io), n=5, warmup=1)
    print(f"torch            : {t1/40:6.1f} ms/step")
    print(f"onnx K/V-only    : {t2/40:6.1f} ms/step")
    print(f"onnx K/V+IOBind  : {t3/40:6.1f} ms/step")

    # ---------------- mimi decode speed ----------------
    print("\n== SPEED: mimi decode step (40-step chain) ==")

    def torch_dec():
        ms = init_states(tts.mimi, 1, 2048)
        with torch.no_grad():
            for i in range(40):
                un = torch.from_numpy(bank[i].reshape(1, 1, fl.ldim)) * fl.emb_std + fl.emb_mean
                tts.mimi.decode_from_latent(un, ms)
                increment_steps(tts.mimi, ms, 16)

    def onnx_dec(engine):
        st = {f"s{i}": init_states(tts.mimi, 1, 2048)[n][k].numpy().copy()
              for i, (n, k) in enumerate(engine.dec_keys)}
        small_idx = [i for i, (n, k) in enumerate(engine.dec_keys) if k != "cache"]
        off16 = 0
        for i in range(40):
            un = (bank[i].reshape(1, 1, engine.ldim) * engine.std + engine.mean).astype(np.float32)
            feeds = {"latent": un}
            for idx in small_idx:
                feeds[f"s{idx}"] = st[f"s{idx}"]
            for base in engine.dec_cache_idx:
                feeds[f"s{base}"] = st[f"s{base}"]
            res = engine.s_dec.run(None, feeds)
            for j, o in enumerate(engine.dec_out_small):
                st[f"s{small_idx[j]}"] = res[o]
            for bi, base in enumerate(engine.dec_cache_idx):
                st[f"s{base}"][:, :, off16:off16+16] = res[engine.dec_out_kv[bi]]
            off16 += 16

    t1 = bench(torch_dec, n=5, warmup=1)
    t2 = bench(lambda: onnx_dec(opt_eng), n=5, warmup=1)
    print(f"torch            : {t1/40:6.1f} ms/step")
    print(f"onnx K/V-only    : {t2/40:6.1f} ms/step")

    # ---------------- full synthesis ----------------
    print("\n== SPEED: full synthesis (2.8s audio) ==")

    def full_torch_prod():
        vs = tts.get_state_for_audio_prompt(VOICE)
        with torch.no_grad():
            tts.generate_audio(vs, PHONEMES)

    def full_onnx_opt():
        vs = tts.get_state_for_audio_prompt(VOICE)
        c, o = opt_eng.seed_flow_cache(vs)
        opt_eng.synthesize(c, o, PHONEMES, bank)

    def full_onnx_opt_io():
        vs = tts.get_state_for_audio_prompt(VOICE)
        c, o = opt_eng_io.seed_flow_cache(vs)
        opt_eng_io.synthesize(c, o, PHONEMES, bank)

    def full_hybrid():
        vs = hyb_eng.voice_state_onnx(VOICE)
        hyb_eng.synthesize(vs, PHONEMES, bank)

    for name, fn in [("torch production (baseline)", full_torch_prod),
                     ("onnx K/V-only", full_onnx_opt),
                     ("onnx K/V+IOBinding", full_onnx_opt_io),
                     ("hybrid (flow-torch + mimi-onnx)", full_hybrid)]:
        t = bench(fn, n=5, warmup=1)
        print(f"{name:34s}: {t:6.0f} ms  RTF {len(audio_t)/24000/(t/1000):.2f}x realtime")

    print("\nbenchmark done")


if __name__ == "__main__":
    main()
