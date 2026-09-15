# ONNX inference engine + benchmark vs the stock PyTorch engine.
#
# Engine pieces (host = numpy + sentencepiece; compute = ORT):
#   - voice prompt conditioning: torch model used ONCE at setup to seed the
#     flow-LM KV cache (identical for both engines -> fair comparison)
#   - text prompt step: flow_lm_step.onnx   (S_text fixed at export time)
#   - generation loop:  flow_lm_gen_step.onnx (EOS loop, host noise)
#   - audio decode:     mimi_decoder_step.onnx (per latent, state I/O)
#
# Benchmarks:
#   speed  : per-step timings (torch vs ORT) + full synthesis wall time
#   accuracy: identical noise bank fed to both engines -> waveform diff
import queue
import sys
import threading
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
PHONEMES = "salAm hAle SomA Cetor ?ast"          # 6 tokens == exported prompt graph
VOICE = str(BASE / "voices" / "female_hello.wav")
TEMP = 0.3
EOS_THRESHOLD = -4.0


# ============================================================================
# shared helpers
# ============================================================================
def noise_bank(n, ldim, seed=1234):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, 1, ldim)) * (TEMP**0.5)).astype(np.float32)


class TorchEngine:
    """Sequential mirror of the production loop, driven at the same level as
    the ONNX engine, with noise drawn from the shared bank."""

    def __init__(self, tts):
        self.tts = tts
        self.fl = tts.flow_lm
        self.mimi = tts.mimi
        self.noise_iter = None

    def _patch_noise(self, bank):
        it = iter(bank)
        self.draw_log = []

        def normal_(t, **kw):
            arr = next(it)
            self.draw_log.append((len(self.draw_log), float(arr.reshape(-1)[0])))
            t.copy_(torch.from_numpy(arr))
            return t

        self._orig_normal = torch.nn.init.normal_
        torch.nn.init.normal_ = normal_

    def _unpatch_noise(self):
        torch.nn.init.normal_ = self._orig_normal

    def synthesize(self, voice_state, phonemes, bank, max_frames=80):
        fl = self.fl
        prepared = fl.conditioner.prepare(phonemes)
        empty = torch.zeros(1, 0, fl.dim)
        self._patch_noise(bank)

        ms = voice_state  # mutated in place
        # production grows the KV cache before prompting text (_generate)
        prepared = fl.conditioner.prepare(phonemes)
        current_end = self.tts._flow_lm_current_end(ms)
        self.tts._expand_kv_cache(ms, sequence_length=current_end + prepared.shape[1] + max_frames + 8)
        latents = []
        try:
            with torch.no_grad():
                # text prompt step
                lat, _ = fl._sample_next_latent(
                    sequence=torch.full((1, 1, fl.ldim), float("nan"), dtype=fl.dtype),
                    text_embeddings=fl.conditioner(prepared), model_state=ms,
                    sampler_decode_steps=1, temp=TEMP, noise_clamp=None,
                    eos_threshold=EOS_THRESHOLD)
                increment_steps(fl, ms, increment=prepared.shape[1] + 1)
                latents.append(lat)

                eos_step = None
                for step in range(max_frames):
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
            self._unpatch_noise()

        # mimi decode (sequential)
        ms_mimi = init_states(self.mimi, batch_size=1, sequence_length=2048)
        chunks = []
        with torch.no_grad():
            for lat in latents:
                un = lat.view(1, 1, fl.ldim) * fl.emb_std + fl.emb_mean
                audio = self.mimi.decode_from_latent(un, ms_mimi)
                increment_steps(self.mimi, ms_mimi, increment=16)
                chunks.append(audio)
        return torch.cat(chunks, dim=-1)[0, 0].numpy(), latents


class OnnxEngine:
    """Full ONNX generation loop (flow LM + mimi decode)."""

    def __init__(self, tts, ort_dir=ORT_DIR):
        fl = tts.flow_lm
        self.fl = fl
        self.tts = tts
        self.ldim = fl.ldim
        self.L = len(fl.transformer.layers)
        self.H = fl.transformer.layers[0].self_attn.num_heads
        self.D = fl.transformer.layers[0].self_attn.dim_per_head
        self.CAP = 256

        self.spk_std = fl.emb_std.detach().numpy()
        self.spk_mean = fl.emb_mean.detach().numpy()

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.s_prompt = ort.InferenceSession(str(ort_dir / "flow_lm_step.onnx"), opts,
                                             providers=["CPUExecutionProvider"])
        self.s_gen = ort.InferenceSession(str(ort_dir / "flow_lm_gen_step.onnx"), opts,
                                          providers=["CPUExecutionProvider"])
        self.s_dec = ort.InferenceSession(str(ort_dir / "mimi_decoder_step.onnx"), opts,
                                          providers=["CPUExecutionProvider"])
        self.dec_in = [i.name for i in self.s_dec.get_inputs()]

        # mimi decoder state template from torch init (shapes only)
        ms = init_states(tts.mimi, batch_size=1, sequence_length=2048)
        self.dec_state_keys = [(k, kk) for k, v in ms.items() for kk in v]

    def seed_flow_cache(self, voice_state):
        """Extract per-layer K/V from torch ModelState into stacked cache."""
        kc = np.zeros((self.L, 2, self.CAP, self.H, self.D), dtype=np.float32)
        n = 0
        offset = None
        for name, st in voice_state.items():
            if name.endswith("self_attn"):
                t = st["cache"].shape[2]
                kc[n, 0, :t] = st["cache"][0, 0].numpy()
                kc[n, 1, :t] = st["cache"][1, 0].numpy()
                n += 1
            if name.endswith("self_attn") and offset is None:
                offset = int(st["offset"][0])
        assert n == self.L
        return kc, offset

    def dec_state_init(self):
        ms = init_states(self.tts.mimi, batch_size=1, sequence_length=2048)
        return [vv.numpy().copy() for v in ms.values() for vv in v.values()]

    def synthesize(self, cache, off, phonemes, bank, max_frames=80, threaded_decode=False):
        fl = self.fl

        # -- text prompt step (host tokenization == torch conditioner.prepare)
        prepared = fl.conditioner.prepare(phonemes)  # host-side tokenizer
        text_emb = fl.conditioner.embed.weight.detach().numpy()[prepared.numpy().reshape(-1)][None]
        text_emb = text_emb.astype(np.float32)
        noise = bank[0]
        bos = np.full((1, 1, self.ldim), np.nan, dtype=np.float32)
        lat, eos, cache = self.s_prompt.run(None, {
            "sequence": bos, "text_emb": text_emb,
            "offset": np.array(off, dtype=np.int64), "noise": noise, "cache": cache})
        latents = [lat]
        off += text_emb.shape[1] + 1

        # -- generation loop
        eos_step = None
        i = 1
        for step in range(max_frames):
            lat, eos, cache = self.s_gen.run(None, {
                "sequence": latents[-1].reshape(1, 1, self.ldim),
                "text_emb": np.zeros((1, 0, fl.dim), dtype=np.float32),
                "offset": np.array(off, dtype=np.int64), "noise": bank[i],
                "cache": cache})
            i += 1
            off += 1
            if bool(eos.reshape(-1)[0]) and eos_step is None:
                eos_step = step
            if eos_step is not None and step >= eos_step + 2:
                break
            latents.append(lat)

        # -- mimi decode
        def decode_all(latent_list):
            st = self.dec_state_init()
            out = []
            for lt in latent_list:
                un = (lt.reshape(1, 1, self.ldim) * self.spk_std + self.spk_mean).astype(np.float32)
                feeds = {self.dec_in[0]: un}
                for name, arr in zip(self.dec_in[1:], st):
                    feeds[name] = arr
                res = self.s_dec.run(None, feeds)
                out.append(res[0])
                st = res[1:]
            return np.concatenate(out, axis=2)[0, 0]

        if not threaded_decode:
            return decode_all(latents), latents

        # threaded: decode in a worker while "generating" (here: generation is
        # already done, so this only measures decode throughput alone)
        q = queue.Queue()
        out_chunks = []

        def worker():
            st = self.dec_state_init()
            while True:
                item = q.get()
                if item is None:
                    break
                un = (item.reshape(1, 1, self.ldim) * self.spk_std + self.spk_mean).astype(np.float32)
                feeds = {self.dec_in[0]: un}
                for name, arr in zip(self.dec_in[1:], st):
                    feeds[name] = arr
                res = self.s_dec.run(None, feeds)
                out_chunks.append(res[0])
                st = res[1:]
                q.task_done()

        th = threading.Thread(target=worker)
        th.start()
        for lt in latents:
            q.put(lt)
        q.put(None)
        th.join()
        return np.concatenate(out_chunks, axis=2)[0, 0], latents


# ============================================================================
# benchmark
# ============================================================================
def bench(fn, n=30, warmup=5):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts)), float(np.mean(ts)), float(np.min(ts))


def main():
    print("== loading torch model ==")
    tts = TTSModel.load_model(config=str(BASE / "model" / "v2" / "model.yaml"))
    fl = tts.flow_lm
    mimi = tts.mimi

    print("== building voice state (setup) ==")
    voice_state = tts.get_state_for_audio_prompt(VOICE)
    voice_state_copy = None  # torch engine mutates; rebuild per run

    torch_eng = TorchEngine(tts)
    onnx_eng = OnnxEngine(tts)

    # ------------------------------------------------------------------
    # accuracy: same noise bank
    # ------------------------------------------------------------------
    print("\n== ACCURACY (identical noise sequence) ==")
    bank = noise_bank(120, fl.ldim)
    vs1 = tts.get_state_for_audio_prompt(VOICE)
    import copy
    vs1_snapshot = copy.deepcopy(vs1)          # pristine copy for seeding
    audio_t, lat_t = torch_eng.synthesize(vs1, PHONEMES, bank)
    cache_seed, off_seed = onnx_eng.seed_flow_cache(vs1_snapshot)
    audio_o, lat_o = onnx_eng.synthesize(cache_seed, off_seed, PHONEMES, bank)
    print(f"torch audio: {audio_t.shape[0]/24000:.2f}s | onnx audio: {audio_o.shape[0]/24000:.2f}s")
    n = min(len(audio_t), len(audio_o))
    diff = np.abs(audio_t[:n] - audio_o[:n])
    rms = np.sqrt((audio_t[:n] ** 2).mean())
    print(f"waveform max diff: {diff.max():.3e} | rms of signal: {rms:.4f} "
          f"| rel diff: {np.sqrt(((audio_t[:n]-audio_o[:n])**2).mean())/rms:.3e}")
    # per-step latent diffs
    ld = [float(np.abs(a - b.detach().numpy()).max()) for a, b in zip(lat_o, lat_t)]
    print(f"latent diffs: max {max(ld):.2e}, mean {np.mean(ld):.2e} over {len(ld)} steps")
    print("first 10 per-step:", [f"{x:.2e}" for x in ld[:10]])
    print("n_torch_latents:", len(lat_t), "n_onnx_latents:", len(lat_o))
    # first latent comparison in detail
    print("torch lat[0][:6]:", lat_t[0].detach().numpy().reshape(-1)[:6])
    print("onnx  lat[0][:6]:", lat_o[0].reshape(-1)[:6])
    print("bank[0][:6]:", bank[0].reshape(-1)[:6])
    print("torch draws (idx, first val) first 8:", torch_eng.draw_log[:8])
    print("bank first vals 0..7:", [f"{bank[j].reshape(-1)[0]:.6f}" for j in range(8)])

    # ------------------------------------------------------------------
    # speed: component benchmarks
    # ------------------------------------------------------------------
    print("\n== SPEED: components ==")
    import soundfile as sf

    voice_audio, sr = sf.read(VOICE)
    va = torch.from_numpy(voice_audio.astype(np.float32))[None, None, :]

    # mimi encoder (voice prompt, ~5 s)
    t_med, t_mean, t_min = bench(lambda: mimi.encode_to_latent(va))
    print(f"mimi encoder  torch : {t_med:7.1f} ms (median of 30)")
    t_med2, _, _ = bench(lambda: onnx_eng_s_enc.run(None, {"audio": va.numpy()}))
    print(f"mimi encoder  ONNX  : {t_med2:7.1f} ms   -> {t_med/t_med2:.2f}x")

    # flow gen step: chain 40 steps through both engines
    def torch_gen_chain():
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

    def onnx_gen_chain():
        cache, off = onnx_eng.seed_flow_cache(tts.get_state_for_audio_prompt(VOICE))
        prepared = fl.conditioner.prepare(PHONEMES)
        te = fl.conditioner.embed.weight.detach().numpy()[prepared.numpy().reshape(-1)][None].astype(np.float32)
        bos = np.full((1, 1, fl.ldim), np.nan, dtype=np.float32)
        _, _, cache = onnx_eng.s_prompt.run(None, {
            "sequence": bos, "text_emb": te, "offset": np.array(off, dtype=np.int64),
            "noise": np.zeros((1, fl.ldim), np.float32), "cache": cache})
        off += te.shape[1] + 1
        seq = np.zeros((1, 1, fl.ldim), np.float32)
        for i in range(40):
            lat, _, cache = onnx_eng.s_gen.run(None, {
                "sequence": seq, "text_emb": np.zeros((1, 0, fl.dim), np.float32),
                "offset": np.array(off, dtype=np.int64),
                "noise": np.zeros((1, fl.ldim), np.float32), "cache": cache})
            off += 1
            seq = lat.reshape(1, 1, fl.ldim)

    t_med, _, _ = bench(torch_gen_chain, n=5, warmup=1)
    t_med2, _, _ = bench(onnx_gen_chain, n=5, warmup=1)
    print(f"flow 40 steps torch : {t_med:7.1f} ms ({t_med/40:.1f} ms/step)")
    print(f"flow 40 steps ONNX  : {t_med2:7.1f} ms ({t_med2/40:.1f} ms/step)  -> {t_med/t_med2:.2f}x")

    # mimi decode step
    def torch_dec_chain():
        ms = init_states(mimi, batch_size=1, sequence_length=2048)
        with torch.no_grad():
            for i in range(40):
                un = torch.from_numpy(bank[i].reshape(1, 1, fl.ldim)) * fl.emb_std + fl.emb_mean
                mimi.decode_from_latent(un, ms)
                increment_steps(mimi, ms, 16)

    def onnx_dec_chain():
        st = onnx_eng.dec_state_init()
        for i in range(40):
            un = (bank[i].reshape(1, 1, fl.ldim) * onnx_eng.spk_std + onnx_eng.spk_mean).astype(np.float32)
            feeds = {onnx_eng.dec_in[0]: un}
            for name, arr in zip(onnx_eng.dec_in[1:], st):
                feeds[name] = arr
            res = onnx_eng.s_dec.run(None, feeds)
            st = res[1:]

    t_med, _, _ = bench(torch_dec_chain, n=5, warmup=1)
    t_med2, _, _ = bench(onnx_dec_chain, n=5, warmup=1)
    print(f"mimi 40 steps torch : {t_med:7.1f} ms ({t_med/40:.1f} ms/step)")
    print(f"mimi 40 steps ONNX  : {t_med2:7.1f} ms ({t_med2/40:.1f} ms/step)  -> {t_med/t_med2:.2f}x")

    # ------------------------------------------------------------------
    # speed: full synthesis
    # ------------------------------------------------------------------
    print("\n== SPEED: full synthesis ==")

    def full_torch_seq():
        vs = tts.get_state_for_audio_prompt(VOICE)
        torch_eng.synthesize(vs, PHONEMES, bank)

    def full_onnx_seq():
        vs = tts.get_state_for_audio_prompt(VOICE)
        c, o = onnx_eng.seed_flow_cache(vs)
        onnx_eng.synthesize(c, o, PHONEMES, bank)

    def full_torch_prod():
        vs = tts.get_state_for_audio_prompt(VOICE)
        with torch.no_grad():
            tts.generate_audio(vs, PHONEMES)

    for name, fn in [("torch sequential (mirror)", full_torch_seq),
                     ("torch production (threaded)", full_torch_prod),
                     ("ONNX sequential", full_onnx_seq)]:
        t_med, t_mean, t_min = bench(fn, n=5, warmup=1)
        print(f"{name:30s}: {t_med:7.0f} ms  (min {t_min:.0f})  "
              f"RTF {audio_t.shape[0]/24000/(t_med/1000):.2f}x realtime")

    print("\nbenchmark done")


if __name__ == "__main__":
    # mimi encoder session for the component benchmark
    onnx_eng_s_enc = ort.InferenceSession(str(ORT_DIR / "mimi_encoder.onnx"),
                                          providers=["CPUExecutionProvider"])
    main()
