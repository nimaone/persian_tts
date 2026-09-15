# Pure-ONNX inference engine for pocket-tts-farsi-v2.
#
# The WHOLE pipeline — Persian text -> phonemes (G2P) -> speech — runs on
# onnxruntime + numpy + sentencepiece + pure-python helpers. NO torch.
# Persian text is auto-detected (Arabic-script codepoints) and phonemised
# with the ONNX G2P (scripts/g2p_onnx.py); phoneme strings pass through.
#
# Package: model/onnx/ (see manifest.json)
#
# CLI:
#   python scripts/tts_onnx.py "salAm hAle SomA Cetor ?ast" voices/female_hello.wav out.wav
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import sentencepiece as spm

BASE = Path(__file__).resolve().parent.parent
PKG = BASE / "model" / "onnx"

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?؟])\s+")
_PHRASE_SPLIT = re.compile(r"(?<=[،؛:—–,;:])\s+")


def split_sentences(text: str) -> list[str]:
    return [t.strip() for t in _SENTENCE_SPLIT.split(text.strip()) if t.strip()]


def split_phrases(sentence: str) -> list[str]:
    """Punctuation-aware phrase units. G2P discards punctuation, so a chunker
    working on phonemes alone cannot see where the writer paused (model card:
    'anything chunking the result cuts on token count alone'). Splitting the
    TEXT first keeps commas / dashes / colons as the pause points they are."""
    return [t.strip() for t in _PHRASE_SPLIT.split(sentence.strip()) if t.strip()]


# A phrase ending in one of these is a lead-in ("سؤال اصلی:") whose whole
# purpose is the pause that follows it.
_STRONG_LEADIN = (":", "؛", "—", "–")


def merge_short_phrases(phrases: list[str], min_words: int = 4,
                        keep_standalone=None) -> list[str]:
    """Very short phrases (< min_words words) read badly as standalone
    chunks, so they merge into a neighbour — EXCEPT lead-ins ending in
    strong punctuation: however short, "سؤال اصلی:" must stay standalone
    because the pause after the colon is exactly what the writer asked for.
    `keep_standalone(p)` lets the caller veto that protection (an unclean
    phonemisation). Must run on TEXT (before phonemisation): phonemes carry
    no punctuation."""
    def _protected(p: str) -> bool:
        return p.rstrip().endswith(_STRONG_LEADIN) and (
            keep_standalone is None or keep_standalone(p))

    out: list[str] = []
    for p in phrases:
        if out and len(out[-1].split()) < min_words and not _protected(out[-1]):
            out[-1] += " " + p
        else:
            out.append(p)
    if len(out) >= 2 and len(out[-1].split()) < min_words and not _protected(out[-2]):
        out[-2] += " " + out.pop()
    return out


def plan_phrases(sentence: str, g2p) -> list[tuple[str, float]]:
    """One text sentence -> [(phonemes, gap_before_seconds), ...].
    Punctuation-aware splitting AND tiny-phrase merging happen here, where
    the punctuation is still visible. Gap: 0.20 s at a normal phrase start,
    0.26 s after a colon/semicolon lead-in — how a reader delivers it."""
    cache: dict[str, bool] = {}

    def clean(tp: str) -> bool:
        # GE2P is noisy on 1-word inputs ("نکته:" -> "nokte nokte"): a
        # lead-in may only stand alone if its phoneme word count matches
        # its text word count; otherwise it merges forward like any other
        # tiny phrase (merged text gives the G2P the context it needs).
        if tp not in cache:
            ph = g2p.phonemise(tp, keep_ezafe=True)
            cache[tp] = bool(ph) and len(ph.split()) == len(tp.split())
        return cache[tp]

    tps = merge_short_phrases(split_phrases(sentence), keep_standalone=clean)
    out: list[tuple[str, float]] = []
    prev_tp: str | None = None
    for tp in tps:
        ph = g2p.phonemise(tp, keep_ezafe=True)
        if not ph:
            continue
        strong = prev_tp is not None and prev_tp.rstrip().endswith((":", "؛"))
        out.append((ph, 0.26 if strong else 0.20))
        prev_tp = tp
    return out


class OnnxTts:
    def __init__(self, pkg_dir=PKG, seed=0):
        self.dir = Path(pkg_dir)
        man = json.loads((self.dir / "manifest.json").read_text(encoding="utf-8"))
        c = man["constants"]
        self.ldim, self.dim = c["ldim"], c["dim"]
        self.L, self.H, self.D, self.CAP = c["layers"], c["heads"], c["dim_per_head"], c["cache_capacity"]
        self.sample_rate = c["sample_rate"]
        self.steps_per_latent = c["mimi_steps_per_latent"]
        self.temp = c["temp"]
        self.eos_threshold = c["eos_threshold"]
        self.tps_est = c["tokens_per_second_estimate"]
        self.gen_pad = c["gen_seconds_padding"]
        self.frame_rate = c["frame_rate"]

        w = np.load(self.dir / "weights.npz")
        self.lut = w["lut_weight"]
        self.spk_proj = w["speaker_proj"]          # [dim, ldim]
        self.bos_voice = w["bos_before_voice"][0]  # [1, dim]
        self.emb_std, self.emb_mean = w["emb_std"], w["emb_mean"]

        self.sp = spm.SentencePieceProcessor(model_file=str(BASE / "model" / "v2" / "tokenizer_ph.model"))
        self.rng = np.random.default_rng(seed)

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        prov = ["CPUExecutionProvider"]
        self.s_flow = ort.InferenceSession(str(self.dir / "flow_lm_step.onnx"), opts, providers=prov)
        self.s_enc = ort.InferenceSession(str(self.dir / "mimi_encoder.onnx"), opts, providers=prov)
        self.s_dec = ort.InferenceSession(str(self.dir / "mimi_decoder_step_kv.onnx"), opts, providers=prov)

        # decoder state layout: inputs s0..sN follow the export's dec_keys
        # order; npz keys carry "module.state_key" so we can find the caches.
        self.dec_in = [i.name for i in self.s_dec.get_inputs()]
        self.dec_out = [o.name for o in self.s_dec.get_outputs()]
        init = np.load(self.dir / "decode_state_init.npz")
        self.dec_init = [init[k] for k in init.files]  # npz order == s0..sN
        n_state = len(self.dec_in) - 1
        self.cache_slots = [i for i, k in enumerate(init.files) if k.endswith(".cache")]
        self.small_slots = [i for i in range(n_state) if i not in self.cache_slots]
        self.dec_out_small = [i for i, n in enumerate(self.dec_out) if n.endswith("o") and n != "audio"]
        self.dec_out_kv = [i for i, n in enumerate(self.dec_out) if n.startswith("kv")]

    # ------------------------------------------------------------------
    def _flow_step(self, seq, text_emb, offset, noise, cache):
        S = seq.shape[1] + text_emb.shape[1]
        if offset + S > self.CAP:
            raise RuntimeError(
                f"flow cache overflow: offset {offset} + {S} new tokens exceeds "
                f"capacity {self.CAP} — the text chunk is too long for one "
                f"generation pass (max ~18 phoneme tokens per chunk)")
        lat, eos, kv = self.s_flow.run(None, {
            "sequence": seq, "text_emb": text_emb,
            "offset": np.array(offset, dtype=np.int64),
            "noise": noise, "cache": cache})
        kv = kv.reshape(self.L, 2, -1, self.H, self.D)
        S = kv.shape[2]
        cache = cache.copy()
        cache[:, :, offset : offset + S] = kv
        return lat, eos, cache, offset + S

    def voice_cache(self, wav_path):
        """Voice prompt -> seeded flow KV cache. Pure ONNX + numpy."""
        import soundfile as sf
        from scipy.signal import resample_poly

        audio, sr = sf.read(wav_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        if sr != self.sample_rate:
            g = np.gcd(int(sr), self.sample_rate)
            audio = resample_poly(audio, self.sample_rate // g, sr // g).astype(np.float32)

        lat = self.s_enc.run(None, {"audio": audio[None, None, :]})[0]  # [1,T,ldim]
        cond = (lat[0] @ self.spk_proj.T).astype(np.float32)            # [T,dim]
        text_emb = np.concatenate([self.bos_voice, cond])[None]         # [1,T+1,dim]

        cache = np.zeros((self.L, 2, self.CAP, self.H, self.D), np.float32)
        _, _, cache, off = self._flow_step(
            np.zeros((1, 0, self.ldim), np.float32), text_emb, 0,
            np.zeros((1, self.ldim), np.float32), cache)
        return cache, off

    # ------------------------------------------------------------------
    def _decode_all(self, latents):
        st = [a.copy() for a in self.dec_init]
        out = []
        off = 0
        for lt in latents:
            un = (lt.reshape(1, 1, self.ldim) * self.emb_std + self.emb_mean).astype(np.float32)
            feeds = {"latent": un}
            for i in range(len(st)):
                feeds[self.dec_in[1 + i]] = st[i]
            res = self.s_dec.run(None, feeds)
            out.append(res[0])
            for j, o in enumerate(self.dec_out_small):
                st[self.small_slots[j]] = res[o]
            for j, o in enumerate(self.dec_out_kv):
                st[self.cache_slots[j]][:, :, off : off + self.steps_per_latent] = res[o]
            off += self.steps_per_latent
        return np.concatenate(out, axis=2)[0, 0]

    def synthesize_text(self, text, voice_wav, seed=None, pace=1.0):
        """Persian text OR phonemes -> audio. Persian is auto-detected;
        Persian text is split into sentences and punctuation-delimited phrases
        BEFORE phonemisation so pauses land where the writer put them."""
        if any("؀" <= ch <= "ۿ" for ch in text):
            from g2p_onnx import OnnxG2P

            if not hasattr(self, "_g2p"):
                self._g2p = OnnxG2P(self.dir)
            # per-sentence phrase plans: phrase gaps (0.20/0.26 s) stay
            # shorter than sentence pauses (0.28 s), mirroring the server
            sentences = []
            for sent in split_sentences(text):
                plan = plan_phrases(sent, self._g2p)
                if plan:
                    sentences.append(plan)
            print("phonemes:", " ".join(" ".join(p for p, _ in s) for s in sentences))
            parts, silence = [], np.zeros(int(0.28 * self.sample_rate), np.float32)
            for plan in sentences:
                parts.append(self.synthesize(plan, voice_wav, pace=pace))
                parts.append(silence)
            return np.concatenate(parts[:-1]) if len(parts) > 1 else parts[0]
        return self.synthesize(text, voice_wav, seed=seed, pace=pace)

    def chunk_phonemes(self, phonemes: str, max_tokens: int = 18) -> list[str]:
        """Word-boundary packing of a phoneme string into model-sized chunks.

        The model is trained on ~11-token utterances; 18 is the safe budget and
        at 21+ generations stop terminating (model card). Never breaks after an
        ezafe marker ("1") so bound phrases like "?eqtesAde1 ?AmrikA" stay in
        one chunk. The "1" is stripped from the returned chunks.
        """
        words = [w for w in phonemes.split() if w]
        chunks, cur = [], []
        for w in words:
            candidate = " ".join(cur + [w])
            n = len(self.sp.encode(candidate, out_type=int))
            over = cur and n > max_tokens
            bound = (
                (cur and cur[-1].endswith("1"))   # marked ezafe pair
                or w in self._LIGHT_VERBS          # compound verb ("Sekannde miSavad")
            )
            if over and not (bound and n <= max_tokens + 4):
                chunks.append(" ".join(cur))
                cur = [w]
            else:
                # bound pairs stay together even if slightly over budget
                # (hard cap +4: a chained run of ezafe heads must not grow
                # a chunk into runaway territory, 21+ tokens stop terminating)
                cur = cur + [w]
        if cur:
            chunks.append(" ".join(cur))
        chunks = self._fix_boundaries(chunks)
        # a trailing 1-2 token chunk reads badly (the model wants >= a few
        # tokens); merge it into the previous chunk even slightly over budget
        if len(chunks) >= 2:
            tail = len(self.sp.encode(chunks[-1], out_type=int))
            if tail <= 2:
                chunks[-2] = chunks[-2] + " " + chunks[-1]
                chunks.pop()
        return [c.replace("1", "") for c in chunks]

    # Persian function words that must not dangle at a chunk end: a
    # preposition/conjunction without its object makes the model pause after
    # it, which the listener hears as a strange mid-phrase stop.
    _FUNCTION_WORDS = {
        "dar", "be", "az", "tA", "va", "ke", "rA", "bA", "bedune",
        "age", "vali", "yA", "barAye", "vase", "dAr", "mi",
    }
    # Conjunctions bind to their LEFT operand ("A va B"): a chunk must not
    # start with one, or the coordinated pair is split by the chunk pause.
    _CONJUNCTIONS = {"va", "yA", "vali", "amA", "hattA", "ke"}
    # Persian light verbs complete the previous word's compound verb
    # ("Sekannde miSavad"); breaking right before one splits the verb.
    _LIGHT_VERBS = {
        "miSavad", "miSavand", "miSavam", "miSavid", "miSavim",
        "Savad", "Savand", "Sod", "Sodand", "Sodan",
        "mikonad", "mikonand", "mikonam", "mikonid", "mikonim",
        "kard", "karde", "konad", "konand",
        "dahad", "dahand", "dad", "dade",
        "dArad", "dArand", "dAsht", "Ast", "?ast", "bud", "budand",
    }

    def _fix_boundaries(self, chunks: list[str], min_words: int = 3) -> list[str]:
        """Move words across chunk boundaries so no boundary splits a bound
        phrase. Each rule moves the previous chunk's LAST word down, then the
        same boundary is re-checked (rules chain):
        (a) an ezafe-marked head ("X1") must not END a chunk — its modifier
            is stranded in the next chunk ("... ?ettesAlAte1 | momken");
        (b) an ezafe phrase should not START a chunk detached from the word
            it attaches to ("fAylhA | ruye1 vindoz");
        (c) a function word must not END a chunk ("... Savad dar | mostanadAt");
        (d) a conjunction must not START a chunk ("... pAydAr | va qAbele ...")
            — it binds to its left operand;
        (e) unmarked compounds: G2P does not always emit the "1" marker (ZWNJ
            compounds like قابل‌اعتماد come out as "qAbele ?e?temAd"), so an
            "…e | ?…" pattern across a boundary is treated as a broken word.
        Rules (b)-(e) never move a word that is itself bound to its LEFT
        neighbour (an ezafe modifier "…?ettesAlAte1 momken", or a light verb
        "Sekannde miSavad") — that would trade one split for a worse one
        (this is exactly how "اتصالات | ممکن" used to get broken).
        """
        i = 1
        while i < len(chunks):
            prev, nxt = chunks[i - 1].split(), chunks[i].split()
            left_bound = len(prev) >= 2 and (
                prev[-2].endswith("1") or prev[-1] in self._LIGHT_VERBS)
            bad = False
            if len(prev) > min_words:
                if prev[-1].endswith("1"):
                    bad = True      # (a) marked ezafe head at chunk end
                elif not left_bound and nxt and nxt[0].endswith("1"):
                    bad = True      # (b) ezafe phrase detached from its host
                elif not left_bound and prev[-1] in self._FUNCTION_WORDS:
                    bad = True      # (c) dangling preposition/conjunction
                elif not left_bound and nxt and nxt[0] in self._CONJUNCTIONS:
                    bad = True      # (d) conjunction split from its operand
                elif (not left_bound and prev[-1].endswith("e")
                      and nxt and nxt[0].startswith("?")):
                    bad = True      # (e) unmarked compound split (qAbele | ?e?temAd)
            if bad:
                chunks[i - 1] = " ".join(prev[:-1])
                chunks[i] = prev[-1] + " " + chunks[i]
            else:
                i += 1
        return chunks

    def synthesize(self, phonemes, voice_wav, seed=None, pace=1.0):
        """phonemes: a phoneme string OR a list of phrase strings. In list mode
        every phrase is a pause unit (punctuation-aware splitting is done in
        the text layer); a phrase longer than ~18 tokens is sub-chunked. Each
        chunk is generated from the pristine voice state (mirrors production:
        copy_state=True — a continued 100+ position context is out of
        distribution and causes early EOS = dropped words). `pace`
        (0.6..1.5) time-stretches the final audio uniformly."""
        pace = float(np.clip(pace, 0.6, 1.5))
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        voice_cache, voice_off = self.voice_cache(voice_wav)

        if isinstance(phonemes, str):
            phrases = [(phonemes, None)]
        else:
            phrases = [p if isinstance(p, tuple) else (p, None) for p in phonemes]
        phrases = [(p.strip(), g) for p, g in phrases if p and p.strip()]
        # NOTE: tiny-phrase merging is deliberately NOT done here. It needs
        # the original punctuation, so it lives in the text layer
        # (plan_phrases / merge_short_phrases); merging again here would
        # re-absorb a short colon lead-in that the caller kept standalone.

        segments = []  # (audio, gap_before_seconds)
        for pi, (phrase, pgap) in enumerate(phrases):
            for ci, chunk in enumerate(self.chunk_phonemes(phrase)):
                audio = self._generate_with_retry(voice_cache, voice_off, chunk)
                if pi == 0 and ci == 0:
                    gap = 0.0
                elif ci == 0:
                    gap = pgap if pgap is not None else 0.20
                else:
                    gap = 0.12
                segments.append((audio, gap))
        audio = self._stitch(segments)
        if abs(pace - 1.0) >= 0.03:
            from pedalboard import time_stretch

            audio = np.ascontiguousarray(audio, dtype=np.float32)
            audio = time_stretch(audio, self.sample_rate,
                                 stretch_factor=pace).reshape(-1)
        return audio.astype(np.float32)

    def _generate_with_retry(self, voice_cache, voice_off, chunk, attempts=2):
        """Generate one chunk, retrying when quality is bad: (1) speech too
        short = early EOS = dropped words, (2) speech too long = runaway,
        (3) a long mid-chunk silence — the model sometimes goes quiet for
        0.4-0.7 s at a random word, heard as a weird mid-phrase stop. All
        stochastic per the model card; a fresh attempt usually lands clean.
        Returns the decoded audio trimmed to actual speech. The duration
        check uses speech, not latent count: the model often trails 1-2 s
        of near-silence before EOS, masking a broken chunk."""
        tokens = len(self.sp.encode(chunk, out_type=int))
        expected_speech = tokens / self.tps_est  # ~seconds of speech
        best, best_key = None, None
        for _ in range(attempts):
            latents, _, _ = self._generate_chunk(voice_cache.copy(), voice_off, chunk)
            audio = self._decode_all(latents)
            s0, e0 = self._speech_bounds(audio)
            speech = audio[s0:e0]
            dur = (e0 - s0) / self.sample_rate
            dead = self._max_internal_silence(speech)
            dur_ok = 0.45 * expected_speech <= dur <= 1.6 * expected_speech
            key = (dur_ok, 0.0 if dead <= 0.35 else -dead, dur)
            if best_key is None or key > best_key:
                best, best_key = speech, key
            if dur_ok and dead <= 0.35:
                break
        return best

    def _max_internal_silence(self, speech, rel_floor=0.03) -> float:
        """Longest quiet run inside a trimmed chunk (dead air the model
        inserted mid-generation), in seconds."""
        if len(speech) < 3:
            return 0.0
        rms = float(np.sqrt((speech ** 2).mean()))
        if rms < 1e-6:
            return 0.0
        quiet = (np.abs(speech) < rel_floor * rms).astype(np.int8)
        edges = np.diff(np.concatenate(([0], quiet, [0])))
        starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
        if len(starts) == 0:
            return 0.0
        return float((ends - starts).max()) / self.sample_rate

    def _speech_bounds(self, p, rel_floor=0.03, head_keep=0.02, tail_keep=0.06,
                       min_run=0.03):
        """Indices of actual speech: the model pads generations with near-
        silence (up to ~2 s) before EOS, and that dead air is what makes
        chunked output feel disjointed. Threshold is relative to the chunk's
        own RMS so quiet voices are not clipped; onsets must be sustained
        (>= min_run) so a single murmur sample does not count as speech."""
        rms = float(np.sqrt((p ** 2).mean()))
        if rms < 1e-6:
            return 0, len(p)
        loud = np.abs(p) >= rel_floor * rms
        # windowed: a position is speech if >=60% of its 2*min_run window is loud
        w = max(1, int(min_run * self.sample_rate))
        kernel = np.ones(2 * w) / (2 * w)
        dens = np.convolve(loud.astype(np.float32), kernel, mode="same")
        solid = dens >= 0.6
        idx = np.where(solid)[0]
        if len(idx) == 0:
            return 0, len(p)
        sr = self.sample_rate
        start = max(0, idx[0] - int(head_keep * sr))
        end = min(len(p), idx[-1] + int(tail_keep * sr))
        return start, end

    def _stitch(self, segments, chunk_gap=0.12, phrase_gap=0.20):
        """Join chunk audios into one continuous-sounding piece: loudness
        matched to the first chunk (each chunk is generated fresh and their
        levels differ by up to ~1.6x), 8 ms declick fades, and fixed short
        pauses instead of the variable 1-2 s of model-generated dead air.
        `segments` is a list of (audio, gap_before) pairs — a phrase start
        (punctuation position) gets a slightly longer pause than an
        intra-phrase chunk boundary."""
        if not segments:
            return np.zeros(0, dtype=np.float32)
        target = float(np.sqrt((segments[0][0] ** 2).mean()))
        f = max(1, int(0.008 * self.sample_rate))
        out = []
        for p, gap in segments:
            rms = float(np.sqrt((p ** 2).mean()))
            if rms > 1e-6:
                p = p * float(np.clip(target / rms, 0.75, 1.35))
            if len(p) > 2 * f:
                p = p.copy()
                p[:f] *= np.linspace(0.0, 1.0, f, dtype=np.float32)
                p[-f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
            if gap > 0:
                out.append(np.zeros(int(gap * self.sample_rate), dtype=np.float32))
            out.append(p)
        audio = np.concatenate(out)
        # peak guard: loudness matching can push peaks past full scale
        peak = float(np.abs(audio).max()) if len(audio) else 0.0
        if peak > 0.98:
            audio = audio * (0.98 / peak)
        return self._compress_pauses(audio)

    def _compress_pauses(self, audio, max_pause=0.38, keep=0.28, rel_floor=0.03):
        """The model sometimes goes silent mid-chunk (or before EOS) even
        after the retry in _generate_with_retry. Remaining dead-air stretches
        longer than `max_pause` are shortened to `keep` seconds with small
        fades, so the flow of speech stays continuous. Deliberate pauses —
        sentence 0.28 s, colon lead-in 0.26 s, phrase 0.20 s, chunk 0.12 s —
        all sit below `max_pause` and are untouched."""
        rms = float(np.sqrt((audio ** 2).mean()))
        if rms < 1e-6:
            return audio
        quiet = (np.abs(audio) < rel_floor * rms).astype(np.int8)
        sr = self.sample_rate
        f = max(1, int(0.008 * sr))
        edges = np.diff(np.concatenate(([0], quiet, [0])))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        out, pos = [], 0
        for a, b in zip(starts, ends):
            if (b - a) > int(max_pause * sr):
                out.append(audio[pos:a])
                seg = audio[a : min(a + int(keep * sr), b)].copy()
                if len(seg) > 2 * f:
                    seg[-f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
                out.append(seg)
                nxt = audio[b : b + f]
                if len(nxt) == f:
                    nxt = (nxt * np.linspace(0.0, 1.0, f, dtype=np.float32)).astype(np.float32)
                out.append(nxt)
                pos = b + f
        out.append(audio[pos:])
        return np.concatenate(out)

    def _generate_chunk(self, cache, off, chunk):
        tokens = self.sp.encode(chunk, out_type=int)
        text_emb = self.lut[np.asarray(tokens)][None].astype(np.float32)

        noise = (self.rng.standard_normal((1, self.ldim)) * (self.temp**0.5)).astype(np.float32)
        lat, _, cache, off = self._flow_step(
            np.full((1, 1, self.ldim), np.nan, np.float32), text_emb, off, noise, cache)
        latents = [lat]

        words = len(chunk.split())
        frames_after_eos = (3 if words <= 4 else 1) + 2
        max_gen_len = int(np.ceil((len(tokens) / self.tps_est + self.gen_pad) * self.frame_rate))

        eos_step = None
        for step in range(max_gen_len):
            noise = (self.rng.standard_normal((1, self.ldim)) * (self.temp**0.5)).astype(np.float32)
            lat, eos, cache, off = self._flow_step(
                latents[-1].reshape(1, 1, self.ldim),
                np.zeros((1, 0, self.dim), np.float32), off, noise, cache)
            if bool(eos.reshape(-1)[0]) and eos_step is None:
                eos_step = step
            if eos_step is not None and step >= eos_step + frames_after_eos:
                break
            latents.append(lat)
        return latents, cache, off


def main():
    args = sys.argv[1:]
    text = args[0] if args else "سلام، حال شما چطور است؟"
    voice = args[1] if len(args) > 1 else str(BASE / "voices" / "female_hello.wav")
    out = args[2] if len(args) > 2 else str(BASE / "output" / "tts_onnx.wav")

    eng = OnnxTts()
    t0 = time.perf_counter()
    audio = eng.synthesize_text(text, voice)
    dt = time.perf_counter() - t0

    import soundfile as sf

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, audio, eng.sample_rate)
    print(f"text: {text}")
    print(f"generated {len(audio)/eng.sample_rate:.2f}s audio in {dt:.2f}s -> {out}")


if __name__ == "__main__":
    main()
