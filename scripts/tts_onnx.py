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
#   python scripts/tts_onnx.py "salAm hAle SomA Cetor ?ast" voices/male_hello.wav out.wav
import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import sentencepiece as spm

BASE = Path(__file__).resolve().parent.parent
PKG = BASE / "model" / "onnx"

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?؟])\s+")
# phrase joints: after punctuation, and around parentheticals — "(...)" is a
# prosodic unit a reader sets off with small pauses on both sides
_PHRASE_SPLIT = re.compile(r"(?<=[،؛:—–,;:)])\s+|\s+(?=\()")


def split_sentences(text: str) -> list[str]:
    return [t.strip() for t in _SENTENCE_SPLIT.split(text.strip()) if t.strip()]


def split_phrases(sentence: str) -> list[str]:
    """Punctuation-aware phrase units. G2P discards punctuation, so a chunker
    working on phonemes alone cannot see where the writer paused (model card:
    'anything chunking the result cuts on token count alone'). Splitting the
    TEXT first keeps commas / dashes / colons as the pause points they are.
    Phrases with no letters/digits are debris — "— —" typed as two dashes
    leaves a lone "—", and a standalone "..." sentence is all punctuation —
    and would otherwise reach the G2P as an empty string ("normalisation
    emptied the text" -> HTTP 400 on perfectly valid Persian)."""
    return [t for t in _PHRASE_SPLIT.split(sentence.strip())
            if t.strip() and _letter_words(t)]


# A phrase ending in one of these is a lead-in ("سؤال اصلی:") whose whole
# purpose is the pause that follows it.
_STRONG_LEADIN = (":", "؛", "—", "–")

# pause lengths shared by the plan/pack layers (seconds): a strong lead-in
# (colon/semicolon/dash) gets a real stop, a comma a short breath; the
# sentence pause is inserted by the caller — server and CLI both read
# SENTENCE_GAP so the two paths cannot drift apart
_STRONG_GAP = 0.26
_PHRASE_GAP = 0.16
SENTENCE_GAP = 0.45
# gap at a fresh-chunk boundary inside one sentence (an unrelated knob from
# the _compress_pauses keep=0.12, which shortens leftover dead air)
_CHUNK_GAP = 0.12


def _letter_words(tp: str) -> list[str]:
    """Words with at least one letter/digit — a lone "—" or "..." is not a
    word, and counting it as one breaks the phoneme-word-count check."""
    return [w for w in tp.split() if any(ch.isalnum() for ch in w)]


def merge_short_phrases(phrases: list[str], min_words: int = 2,
                        keep_standalone=None) -> list[str]:
    """Punctuation is a pause: a phrase with >= min_words words keeps its
    own pause slot (the reader asked for it with the comma/paren/colon).
    Only 1-word fragments merge forward — they read badly standalone and
    GE2P is noisy on 1-word inputs. Phrases ending in strong punctuation
    ("سؤال اصلی:", "…است —") stay standalone however short, and
    `keep_standalone(p)` lets the caller veto that for an unclean
    phonemisation. Must run on TEXT (before phonemisation): phonemes carry
    no punctuation."""
    def _protected(p: str) -> bool:
        return p.rstrip().endswith(_STRONG_LEADIN) and (
            keep_standalone is None or keep_standalone(p))

    out: list[str] = []
    for p in phrases:
        if out and len(_letter_words(out[-1])) < min_words and not _protected(out[-1]):
            out[-1] += " " + p
        else:
            out.append(p)
    if (len(out) >= 2 and len(_letter_words(out[-1])) < min_words
            and not _protected(out[-2])):
        # NB: pop BEFORE the store — `out[-2] += " " + out.pop()` would
        # resolve the store index AFTER the list shrank, silently
        # overwriting the phrase one slot earlier (dropped + duplicated
        # phrases, heard as missing words).
        last = out.pop()
        out[-1] += " " + last
    return out


# Text-layer prepositions that start a DETACHABLE adjunct ("برای X", "بدون
# X"). Tight-binding ones (از/با/در/به/تا — "از آن"، "به دست آورد"، "۱ تا ۲")
# are deliberately excluded: cutting before them reads far worse than wherever
# the phoneme packer lands.
_TEXT_PREPS = {"برای", "بدون", "درباره", "مثل"}


def split_long_phrase(tp: str, max_words: int = 9) -> list[str]:
    """A phrase longer than max_words needs 2+ chunks anyway (18-token
    budget); cut it before its last preposition (>=4 words before, >=3
    after) so the packer's inevitable break lands at a natural joint like
    '…سیم‌کشی | برای ساختن…' instead of inside 'یک شبکهٔ عصبی'."""
    words = tp.split()
    if len(words) <= max_words:
        return [tp]
    cut = None
    for i, w in enumerate(words):
        if w in _TEXT_PREPS and i >= 4 and len(words) - i >= 3:
            cut = i
    if cut is None:
        return [tp]
    return (split_long_phrase(" ".join(words[:cut]), max_words)
            + split_long_phrase(" ".join(words[cut:]), max_words))


# Light verbs in TEXT form (ZWNJ/space-stripped). A phrase must not START
# with one: the TTS model refuses to lead an utterance with a bare verbal
# enclitic and drops the word — "…جدیدی — | می‌شود شبکه را…" loses «می‌شود»
# on every voice — while «دیده می‌شود» at a chunk end reads fine. Kept in
# sync with the phoneme-side _LIGHT_VERBS (داد/دارد/می‌دهد families included).
_TEXT_LIGHT_VERBS = {
    "است", "هست", "هستم", "هستی", "هستیم", "هستید", "هستند",
    "بود", "بودم", "بودی", "بودیم", "بودید", "بودند", "باشد", "باشند",
    "شد", "شدم", "شدی", "شدیم", "شدید", "شدند", "شود", "شوند",
    "کرد", "کردم", "کردی", "کردیم", "کردید", "کردند", "کرده",
    "کنم", "کنی", "کند", "کنیم", "کنید", "کنند",
    "داد", "دادم", "دادی", "دادیم", "دادید", "دادند", "داده",
    "بدهد", "بدهند",
    "میشود", "میشوند", "میکند", "میکنند", "میکرد", "میکردند",
    "میداد", "میدادند", "میدهد", "میدهند",
    "دارد", "دارم", "داری", "داریم", "دارید", "دارند", "داشت",
    "میباشد", "میباشند",
}


def merge_leading_light_verbs(phrases: list[str]) -> list[str]:
    """Merge any phrase that starts with a light verb — or with the object
    marker «را», which clings to the previous phrase's noun — into the
    previous phrase, so the clitic follows its host and is actually spoken.
    The dash/colon pause that preceded it gives way to a small intra-chunk
    gap after the verb — a minor prosody cost against a dropped word."""
    out: list[str] = []
    for p in phrases:
        words = _letter_words(p)
        first = words[0].strip("«»()\"'.,;:!?،؛:-") if words else ""
        key = first.replace("\u200c", "")
        if out and (key in _TEXT_LIGHT_VERBS or key == "را"):
            out[-1] += " " + p
        else:
            out.append(p)
    return out


def plan_phrases(sentence: str, g2p, tokenizer=None) -> list[tuple[str, float]]:
    """One text sentence -> [(phonemes, gap_before_seconds), ...].
    Punctuation-aware splitting AND tiny-phrase merging happen here, where
    the punctuation is still visible. Gaps mirror how a reader delivers
    the text: 0.16 s at a comma (a short breath), 0.26 s after a strong
    lead-in (colon/semicolon/dash). The sentence-final pause (0.45 s) is
    inserted by the caller, outside the per-sentence synthesis.
    `tokenizer` (the engine's SentencePiece model) gates split_long_phrase
    on the real budget — phoneme TOKENS, not text words: a 10-word phrase
    of short words is ~17 tokens and fits one chunk, so splitting it only
    inserted a pause the writer never asked for. Without it, the old
    word-count heuristic applies."""
    cache: dict[str, bool] = {}

    def clean(tp: str) -> bool:
        # GE2P is noisy on 1-word inputs ("نکته:" -> "nokte nokte"): a
        # lead-in may only stand alone if its phoneme word count matches
        # its text word count; otherwise it merges forward like any other
        # tiny phrase (merged text gives the G2P the context it needs).
        # The text side is transliterated AND normalised first — "self-
        # recurrency" is one text word but two Persian words, and "۱۳۱۳:"
        # is one text word but five G2P words (numbers expand); the raw
        # count wrongly vetoed such lead-ins.
        if tp not in cache:
            from g2p_onnx import transliterate_text  # puts model/v2 on sys.path
            from normalize_fa import normalize_for_model
            want = len(_letter_words(
                normalize_for_model(transliterate_text(tp))
                .replace("؟", "").replace("?", "").replace(":", "")))
            ph = g2p.phonemise(tp, keep_ezafe=True)
            cache[tp] = bool(ph) and len(ph.split()) == want
        return cache[tp]

    merged = merge_short_phrases(split_phrases(sentence), keep_standalone=clean)
    merged = merge_leading_light_verbs(merged)
    out: list[tuple[str, float]] = []
    prev_tp: str | None = None
    for tp in merged:
        ph_tp = g2p.phonemise(tp, keep_ezafe=True)
        if (tokenizer is not None and ph_tp
                and len(tokenizer.encode(ph_tp, out_type=int)) <= 20):
            parts = [tp]          # fits one chunk even with bound-pair slack
        else:
            parts = split_long_phrase(tp)
        for q in parts:
            ph = ph_tp if q == tp else g2p.phonemise(q, keep_ezafe=True)
            if not ph:
                continue
            strong = prev_tp is not None and prev_tp.rstrip().endswith(_STRONG_LEADIN)
            out.append((ph, _STRONG_GAP if strong else _PHRASE_GAP))
            prev_tp = q
    return out


def pack_phrases(plan: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """PACK mode (the UI's «یکپارچه» option): merge adjacent phrases into
    one breathing unit unless a strong lead-in (gap 0.26 — colon/semicolon/
    dash) demands its pause. The comma pauses give way to the chunker's
    shorter joints, so the sentence flows in fewer, longer breaths — the
    model ends every chunk with its own sentence-final fall, so fewer
    chunks mean fewer artificial sentence-ends. Split mode (the plan as-is)
    keeps every punctuation pause."""
    groups: list[tuple[str, float]] = []
    for ph, gap in plan:
        if groups and gap != _STRONG_GAP:
            groups[-1] = (groups[-1][0] + " " + ph, groups[-1][1])
        else:
            groups.append((ph, gap))
    return groups


def trim_hot_onset(audio: np.ndarray, sr: int, head_ms: int = 300,
                   thresh_db: float = 4.0,
                   min_keep_s: float = 1.5) -> tuple[np.ndarray, int]:
    """Drop a hot opening from a reference-voice prompt. -> (audio, dropped_ms)

    A prompt whose first syllable is much louder than the rest of the sample
    makes the model replay that onset instead of the first word of a chunk.
    With voices/male_hello.wav (first 300 ms = +9.3 dB over the sample's own
    speech level) the 4-token tail chunk «شبکه شکننده میشود» opened with a
    280 ms burst: peak 1.39 (past full scale), +8 dB over the chunk's own body
    and 0.86-correlated with the prompt's first 250 ms — so the word «شبکه» is
    masked and the peak guard ducks the whole render by ~3 dB. The other two
    builtin references start with ~250 ms of near-silence (-48 / -36 dB) and
    never do this. Measured remedy: drop the leading 300 ms (four seeds and
    the full sentence come out clean; head peak 1.385 -> 0.025, head rms
    9 -> 27 dB below the chunk body). A prompt that opens at a normal level is
    returned untouched, so this costs nothing for voices that are already fine
    — and it also covers uploads, which only get trimmed to 5 s and never had
    their onset looked at. Prompts too short to lose `head_ms` keep their burst
    (min_keep_s): a 0.9 s prompt would hurt the clone more than the leak."""
    if len(audio) < sr:                       # too short to judge
        return audio, 0
    h = max(1, int(0.01 * sr))
    nf = len(audio) // h
    peak = float(np.abs(audio).max())
    if nf < 8 or peak < 1e-6:
        return audio, 0
    frames = np.sqrt(np.array([(audio[i * h:(i + 1) * h] ** 2).mean()
                               for i in range(nf)], dtype=np.float64))
    speech = frames[frames > 0.02 * peak]     # frames carrying real audio
    if not len(speech):
        return audio, 0
    typical = float(np.median(speech))
    head = frames[: max(1, int(head_ms / 10))]
    if float(np.sqrt((head ** 2).mean())) <= typical * 10 ** (thresh_db / 20):
        return audio, 0
    drop = int(head_ms / 1000 * sr)
    if len(audio) - drop < min_keep_s * sr:
        return audio, 0
    out = audio[drop:].copy()
    f = max(1, int(0.02 * sr))                # the new start may be mid-wave
    if len(out) > 2 * f:
        out[:f] *= np.linspace(0.0, 1.0, f, dtype=np.float32)
    return out, drop * 1000 // sr


class OnnxTts:
    # voice-prompt KV cache budget: ~19 MB per entry -> ~76 MB held
    VOICE_MEMO_MAX = 4

    def __init__(self, pkg_dir=PKG, seed=None):
        # seed=None draws OS entropy: every run differs, so the README's
        # "rebuild a bad generation" advice actually works. Pass an int
        # (CLI --seed) to reproduce a run exactly.
        self.dir = Path(pkg_dir)
        man = json.loads((self.dir / "manifest.json").read_text(encoding="utf-8"))
        c = man["constants"]
        self.ldim, self.dim = c["ldim"], c["dim"]
        self.L, self.H, self.D, self.CAP = c["layers"], c["heads"], c["dim_per_head"], c["cache_capacity"]
        self.sample_rate = c["sample_rate"]
        self.steps_per_latent = c["mimi_steps_per_latent"]
        self.temp = c["temp"]
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
        # voice prompt KV caches, memoised by (path, mtime): encoding a
        # reference voice costs ~0.5 s and synthesis runs per SENTENCE —
        # a 6-sentence paragraph paid it 6 times. Entries are ~19 MB each,
        # so VOICE_MEMO_MAX caps the cache at ~76 MB (LRU).
        self._voice_memo: dict = {}
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # chunks generate in parallel (4 workers); the models are small and
        # batch-1 steps barely use extra threads — cap per-session threads
        # to avoid 4x oversubscription on an 8-core box
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
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
        # write in place: every caller owns its cache exclusively (the
        # shared voice cache is copied per attempt in _generate_with_retry)
        # — a defensive copy here cost ~5.4 ms x ~45 steps per chunk
        cache[:, :, offset : offset + S] = kv
        return lat, eos, cache, offset + S

    def voice_cache(self, wav_path):
        """Voice prompt -> seeded flow KV cache. Pure ONNX + numpy.
        Memoised by (path, mtime): the returned cache is SHARED and must
        never be written to — callers copy it before generating (they do:
        _generate_with_retry copies per attempt)."""
        import soundfile as sf
        from scipy.signal import resample_poly

        p = Path(wav_path)
        key = (str(p.resolve()), p.stat().st_mtime)
        hit = self._voice_memo.pop(key, None)
        if hit is not None:
            self._voice_memo[key] = hit  # refresh recency
            return hit

        audio, sr = sf.read(wav_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        if sr != self.sample_rate:
            g = np.gcd(int(sr), self.sample_rate)
            audio = resample_poly(audio, self.sample_rate // g, sr // g).astype(np.float32)
        # model card: prompts beyond 5 s are out of distribution (the
        # upload endpoint already trims; this guards the CLI/engine path)
        if len(audio) > 5 * self.sample_rate:
            audio = audio[: 5 * self.sample_rate]
        audio, dropped = trim_hot_onset(audio, self.sample_rate)
        if dropped:
            print(f"voice prompt {p.name}: dropped the first {dropped} ms — a "
                  f"hot opening syllable gets replayed as a burst over the "
                  f"first word of every chunk")

        lat = self.s_enc.run(None, {"audio": audio[None, None, :]})[0]  # [1,T,ldim]
        cond = (lat[0] @ self.spk_proj.T).astype(np.float32)            # [T,dim]
        text_emb = np.concatenate([self.bos_voice, cond])[None]         # [1,T+1,dim]

        cache = np.zeros((self.L, 2, self.CAP, self.H, self.D), np.float32)
        _, _, cache, off = self._flow_step(
            np.zeros((1, 0, self.ldim), np.float32), text_emb, 0,
            np.zeros((1, self.ldim), np.float32), cache)
        self._voice_memo[key] = (cache, off)
        while len(self._voice_memo) > self.VOICE_MEMO_MAX:
            self._voice_memo.pop(next(iter(self._voice_memo)))
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

    def synthesize_text(self, text, voice_wav, seed=None, pace=1.0, mode="split"):
        """Persian text OR phonemes -> audio. Persian is auto-detected;
        Persian text is split into sentences and punctuation-delimited phrases
        BEFORE phonemisation so pauses land where the writer put them.
        mode="pack" merges comma-delimited phrases into longer breathing
        units (pack_phrases) — punctuation pauses survive only at strong
        lead-ins (colon/semicolon/dash)."""
        if any("\u0600" <= ch <= "\u06FF" for ch in text):
            from g2p_onnx import OnnxG2P

            if not hasattr(self, "_g2p"):
                self._g2p = OnnxG2P(self.dir)
            # per-sentence phrase plans: phrase gaps (0.16/0.26 s) stay
            # clearly shorter than sentence pauses (0.45 s), mirroring the
            # server — a reader breathes at a comma but stops at a period
            sentences = []
            for sent in split_sentences(text):
                plan = plan_phrases(sent, self._g2p, self.sp)
                if mode == "pack":
                    plan = pack_phrases(plan)
                if plan:
                    sentences.append(plan)
            print("phonemes:", " ".join(" ".join(p for p, _ in s) for s in sentences))
            parts, silence = [], np.zeros(int(SENTENCE_GAP * self.sample_rate), np.float32)
            for plan in sentences:
                parts.append(self.synthesize(plan, voice_wav, pace=pace))
                parts.append(silence)
            if not parts:  # text was all punctuation debris
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(parts[:-1]) if len(parts) > 1 else parts[0]
        return self.synthesize(text, voice_wav, seed=seed, pace=pace)

    # Phonetic prepositions: when a forced chunk break lands just after a
    # preposition's first word or two, the PP has barely started — the break
    # moves back to before the preposition ("…biStar | ?az SabakehAye …"
    # reads far better than "…?az SabakehAye | tasAdofi …").
    _PREPS = {"?az", "bA", "dar", "be", "barAye", "tA", "ruye", "bedune", "vase"}

    def _pack_words(self, words, max_tokens=18, slack=2) -> list[str]:
        """Greedy word-boundary packing (the loop body of chunk_phonemes):
        fill a chunk to max_tokens + slack, keep bound pairs (ezafe "1" /
        light verb / "rA") together up to max_tokens+2, and on a forced break
        look back for the best joint. Chunks keep the ezafe marker attached.

        `slack` is the headroom the fill may use before a break is forced. It
        defaults to the +2 that _enforce_budget and the bound-pair rule below
        already treat as safe (21+ tokens stop terminating), so a clause that
        fits the hard cap is read in one breath instead of being split
        mid-clause: a 20-token clause used to be cut right after «rA», leaving
        a 4-token tail chunk that starts late and reads as a comma pause.
        chunk_phonemes re-packs with slack=0 (the plain 18-token target) when
        the wider fill would leave a runt tail."""
        chunks, cur = [], []
        for w in words:
            candidate = " ".join(cur + [w])
            n = len(self.sp.encode(candidate, out_type=int))
            over = cur and n > max_tokens + slack
            bound = (
                (cur and cur[-1].endswith("1"))   # marked ezafe pair
                or w in self._LIGHT_VERBS         # compound verb ("Sekannde miSavad")
                or w == "rA"                      # object marker clings to its noun
            )
            if over and not (bound and n <= max_tokens + 2):
                # a break is forced — pick the best joint near the overflow:
                # 1. right after a light verb (it completes its host, and
                #    what follows starts a fresh unit: "…jadidi miSavad |
                #    Sabake rA beture…" beats "…Sabake rA | beture…")
                # 2. right after the object marker "rA" (end of the object
                #    NP = start of the predicate — always a safe joint:
                #    "…?amalkard rA | bA hazineye kamtari be dast ?Avardand"
                #    keeps the final verb phrase in one breath)
                # 3. before a preposition the break would strand (at most
                #    one word between prep and break: "…biStar | ?az
                #    SabakehAye …" not "…?az SabakehAye | tasAdofi …")
                cut = len(cur)
                for k in range(len(cur) - 1, max(len(cur) - 5, -1), -1):
                    if cur[k] in self._LIGHT_VERBS or cur[k] == "rA":
                        cut = k + 1
                        break
                if cut == len(cur):
                    for k in range(len(cur) - 1, max(len(cur) - 3, -1), -1):
                        if cur[k] in self._PREPS and len(cur) - k <= 2:
                            cut = k
                            break
                if (cut >= 3
                        and len(self.sp.encode(" ".join(cur[:cut]), out_type=int)) >= 6):
                    # guard: a cut must not create a tiny chunk (<6 tokens)
                    # — the model is unreliable on those ("har taklif rA"
                    # came out as garbage on whole voices)
                    chunks.append(" ".join(cur[:cut]))
                    cur = cur[cut:]
                else:
                    chunks.append(" ".join(cur))
                    cur = []
                cur = cur + [w]
            else:
                # bound pairs stay together even if slightly over budget
                # (hard cap +2: 21+ tokens stop terminating, so the slack
                # must never reach past 20)
                cur = cur + [w]
        if cur:
            chunks.append(" ".join(cur))
        return chunks

    def chunk_phonemes(self, phonemes: str, max_tokens: int = 18) -> list[str]:
        """Word-boundary packing of a phoneme string into model-sized chunks.

        The model is trained on ~11-token utterances; 18 is the safe budget and
        at 21+ generations stop terminating (model card), so the fill aims at
        max_tokens+2 — the cap _enforce_budget already enforces — and drops back
        to the plain 18 only when the wider fill would end in a runt (a short
        chunk is generated on its own, starts late and drags dead air in).
        Never breaks after an ezafe marker ("1") so bound phrases like
        "?eqtesAde1 ?AmrikA" stay in one chunk. The "1" is stripped from the
        returned chunks.
        """
        words = [w for w in phonemes.split() if w]

        def pack(slack: int) -> list[str]:
            c = self._pack_words(words, max_tokens, slack=slack)
            c = self._fix_boundaries(c)
            c = self._enforce_budget(c, max_tokens)
            # a trailing 1-2 token chunk reads badly (the model wants >= a few
            # tokens); merge it into the previous chunk even slightly over
            # budget — but never past 19 tokens (21+ stop terminating)
            if len(c) >= 2:
                tail = len(self.sp.encode(c[-1], out_type=int))
                merged = len(self.sp.encode(c[-2] + " " + c[-1], out_type=int))
                if tail <= 2 and merged <= 19:
                    c[-2] = c[-2] + " " + c[-1]
                    c.pop()
            return c

        chunks = pack(slack=2)
        if len(chunks) >= 2 and (
                len(_letter_words(chunks[-1])) < 3
                or len(self.sp.encode(chunks[-1], out_type=int)) < 6):
            # the wider fill left a runt tail — re-pack the whole phrase with
            # the plain 18-token target (pre-slack behaviour; measured cost of
            # NOT doing this: a 2-token "?ast" chunk + 1.3 s of dead air)
            chunks = pack(slack=0)
        return [c.replace("1", "") for c in chunks]

    def _enforce_budget(self, chunks, max_tokens):
        """_fix_boundaries moves words across boundaries without checking
        length, and a moved word can push the receiving chunk past the
        terminating budget — re-pack any chunk over max_tokens+2 (a
        19-token chunk was observed in the wild; the old +4 slack allowed
        22 in theory while 21+ already stop terminating)."""
        for _ in range(3):
            if all(len(self.sp.encode(c, out_type=int)) <= max_tokens + 2
                   for c in chunks):
                return chunks
            repacked = []
            for c in chunks:
                if len(self.sp.encode(c, out_type=int)) <= max_tokens + 2:
                    repacked.append(c)
                else:
                    repacked.extend(self._pack_words(c.split(), max_tokens))
            chunks = self._fix_boundaries(repacked)
        return chunks

    # Persian function words that must not dangle at a chunk end: a
    # preposition/conjunction without its object makes the model pause after
    # it, which the listener hears as a strange mid-phrase stop. "beture"/
    # "besurate" ("به‌طور/به‌صورت X") always need the word they qualify.
    # "rA" is NOT here: it never dangles — it clings to the noun before it
    # and is handled as a left-binding clitic.
    _FUNCTION_WORDS = {
        "dar", "be", "?az", "tA", "va", "ke", "bA", "bedune",
        "age", "vali", "yA", "barAye", "vase", "dAr", "mi",
        "beture", "besurate",
    }
    # Left-binding words: a chunk must not START with one — "A va B" and
    # "ketAb rA" belong together, so the previous chunk's last word moves
    # down to keep the pair intact.
    _CONJUNCTIONS = {"va", "yA", "vali", "amA", "hattA", "ke", "rA"}
    # Persian light verbs complete the previous word's compound verb
    # ("Sekannde miSavad", "neSAn dAdand"); breaking right before one splits
    # the verb. NB: GE2P writes long-a as "A" — the dah-family is "dAd…",
    # never "dad…" (a lowercase typo there silently disables the rule).
    _LIGHT_VERBS = {
        "miSavad", "miSavand", "miSavam", "miSavid", "miSavim",
        "Savad", "Savand", "Sod", "Sodand", "Sodan",
        "mikonad", "mikonand", "mikonam", "mikonid", "mikonim",
        "kard", "karde", "konad", "konand",
        "dAd", "dAde", "dAdand", "dAdam", "dAdi", "dAdim", "dAdid",
        "dAhad", "dAhand", "midAd", "midAdand",
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

        NB: these rules run per PHRASE (chunk_phonemes is called per
        phrase), so a chunk that OPENS a phrase is deliberately never
        touched — a leading "و"/"که" there follows the writer's own
        comma/dash pause, which is normal Persian clause prosody; moving
        it before the pause would fight the punctuation the plan exists
        to honour.
        """
        i = 1
        while i < len(chunks):
            prev, nxt = chunks[i - 1].split(), chunks[i].split()
            left_bound = len(prev) >= 2 and (
                prev[-2].endswith("1") or prev[-1] in self._LIGHT_VERBS
                or prev[-1] == "rA")
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
        jobs = []      # (chunk, gap) — generation units
        for pi, (phrase, pgap) in enumerate(phrases):
            for ci, chunk in enumerate(self.chunk_phonemes(phrase)):
                if pi == 0 and ci == 0:
                    gap = 0.0
                elif ci == 0:
                    gap = pgap if pgap is not None else _PHRASE_GAP
                else:
                    gap = _CHUNK_GAP
                jobs.append((chunk, gap))

        # chunks are independent (each generates from the pristine voice
        # state), and batch-1 flow steps barely saturate the cores —
        # generate them in parallel with per-chunk rngs
        seeds = self.rng.integers(0, 2**63, size=max(1, len(jobs)))
        if len(jobs) > 1:
            from concurrent.futures import ThreadPoolExecutor

            def run(i):
                return self._generate_with_retry(
                    voice_cache, voice_off, jobs[i][0],
                    rng=np.random.default_rng(int(seeds[i])))

            workers = min(4, os.cpu_count() or 4)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                audios = list(ex.map(run, range(len(jobs))))
        else:
            audios = [self._generate_with_retry(
                voice_cache, voice_off, jobs[0][0],
                rng=np.random.default_rng(int(seeds[0])))]
        segments = [(a, jobs[i][1]) for i, a in enumerate(audios)]
        audio = self._stitch(segments)
        if abs(pace - 1.0) >= 0.03:
            from pedalboard import time_stretch

            audio = np.ascontiguousarray(audio, dtype=np.float32)
            audio = time_stretch(audio, self.sample_rate,
                                 stretch_factor=pace).reshape(-1)
        return audio.astype(np.float32)

    @staticmethod
    def _window_mean(x: np.ndarray, w: int) -> np.ndarray:
        """'same'-mode moving average in O(n): np.convolve with a w-sample
        kernel is O(n*w) — ~0.4 s on 20 s of audio — and this runs on every
        retry attempt and again when stitching. Output length is always
        len(x) (np.convolve 'same' returns max(len(x), w), which let index
        math run past the signal on inputs shorter than the window)."""
        n = len(x)
        c = np.concatenate(([0.0], np.cumsum(x.astype(np.float64))))
        k = np.arange(n + w - 1)
        lo = np.maximum(k - w + 1, 0)
        hi = np.minimum(k + 1, n)
        full = (c[hi] - c[lo]) / w
        start = (w - 1) // 2
        return full[start: start + n].astype(np.float32)

    def _dead_air_regions(self, p, rel_floor=0.03, win=0.25, density=0.90):
        """Spans of dead air the model inserted mid-chunk, as (start, end)
        sample indices. Windowed quiet-density based: the model scatters
        tiny blips through its silences, which chop a 1.2 s pause into sub-
        threshold runs that defeat any longest-run check. A region needs
        >=density of a win-second window below rel_floor*rms, then expands
        to its true quiet boundaries."""
        rms = float(np.sqrt((p ** 2).mean()))
        if rms < 1e-6 or len(p) < 3:
            return []
        sr = self.sample_rate
        quiet = np.abs(p) < rel_floor * rms
        w = max(1, int(win * sr))
        dens = self._window_mean(quiet.astype(np.float32), w)
        solid = dens >= density
        edges = np.diff(np.concatenate(([0], solid.astype(np.int8), [0])))
        starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
        out = []
        for a, b in zip(starts, ends):
            i, j = int(a), int(b)
            while i > 0 and quiet[i - 1]:
                i -= 1
            while j < len(p) and quiet[j]:
                j += 1
            out.append((i, j))
        merged: list[tuple[int, int]] = []
        for i, j in out:
            if merged and i <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], j))
            else:
                merged.append((i, j))
        return merged

    def _solid_fraction(self, speech, floor=0.05, win=0.04) -> float:
        """Fraction of the chunk's duration carrying real speech energy
        (windowed rms >= an absolute floor). A degraded generation produces
        one loud burst and then near-silence: chunk-level rms and duration
        checks both pass, but only 10-40% of the audio is speech — the
        words are simply not there (confirmed by ASR)."""
        if len(speech) < 3:
            return 1.0
        w = max(1, int(win * self.sample_rate))
        n = len(speech) // w
        if n == 0:
            return 1.0
        solid = sum(1 for i in range(n)
                    if float(np.sqrt((speech[i * w:(i + 1) * w] ** 2).mean())) >= floor)
        return solid / n

    def _generate_with_retry(self, voice_cache, voice_off, chunk, attempts=3,
                             rng=None):
        """Generate one chunk, retrying when quality is bad. `rng` lets
        parallel callers give each chunk its own independent noise stream
        (the engine's shared rng is not thread-safe). Failure modes,
        all stochastic per the model card; a fresh attempt usually lands
        clean:
        (1) near-silent output (a generation that EOS'd into nothing) —
            caught by an ABSOLUTE rms floor, because every relative check
            is blind on a silent chunk (its own rms is ~0, so all of it
            looks "loud" and none of it looks "quiet");
        (2) burst-then-silence degradation — one loud syllable then
            mumbling: caught by the solid-speech fraction;
        (3) speech too short = early EOS = dropped words, or too long =
            runaway (the manifest's tps_est=3 is ~2x conservative vs the
            real 4-7 tokens/s, so the window is wide);
        (4) a long mid-chunk dead-air stretch — the model sometimes goes
            quiet for ~1 s at a random word, heard as a weird mid-phrase
            stop (measured windowed, so noise blips inside the silence
            cannot hide it).
        Preference order when no attempt is fully clean: loud beats silent,
        enough actual speech beats garbage, more speech beats less (dropped
        words read short), then less dead air. Returns trimmed speech."""
        tokens = len(self.sp.encode(chunk, out_type=int))
        expected_speech = tokens / self.tps_est  # ~seconds (conservative)
        best, best_key = None, None
        for _ in range(attempts):
            latents, _, _ = self._generate_chunk(voice_cache.copy(), voice_off, chunk,
                                                   rng=rng)
            audio = self._decode_all(latents)
            s0, e0 = self._speech_bounds(audio)
            speech = audio[s0:e0]
            # strip model dead air BEFORE measuring (and returning): a
            # chunk often says its words then trails ~1 s of pre-EOS
            # silence, which used to fail the density gate and trigger
            # two pointless retries — 3x the cost on clean chunks
            speech = self._compress_pauses(speech, max_pause=0.30, keep=0.12)
            dur = len(speech) / self.sample_rate
            rms = float(np.sqrt((speech ** 2).mean()))
            loud_ok = rms >= 0.02
            frac = self._solid_fraction(speech)
            # actual speech seconds: a short chunk often says its words fast
            # and then trails ~1 s of pre-EOS silence — total duration and
            # solid fraction both look bad while the WORDS are fine, and a
            # dense garbage attempt used to win the ranking on them.
            solid_dur = frac * dur
            dur_ok = 0.30 * expected_speech <= solid_dur <= 1.6 * expected_speech
            regions = self._dead_air_regions(speech)
            dead = max((j - i) for i, j in regions) / self.sample_rate if regions else 0.0
            key = (loud_ok, dur_ok, solid_dur, 0.0 if dead <= 0.35 else -dead)
            if best_key is None or key > best_key:
                best, best_key = speech, key
            if loud_ok and dur_ok and frac >= 0.55 and dead <= 0.35:
                break
        return best

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
        dens = self._window_mean(loud.astype(np.float32), 2 * w)
        solid = dens >= 0.6
        idx = np.where(solid)[0]
        if len(idx) == 0:
            return 0, len(p)
        sr = self.sample_rate
        start = max(0, idx[0] - int(head_keep * sr))
        end = min(len(p), idx[-1] + int(tail_keep * sr))
        return start, end

    def _stitch(self, segments):
        """Join chunk audios into one continuous-sounding piece: loudness
        matched to the MEDIAN chunk level (chunks are generated fresh and
        their levels differ wildly — a female_narration segment measured at 3x
        its median), plus 8 ms declick fades and fixed short pauses instead
        of the variable 1-2 s of model-generated dead air. The gain clip is
        deliberately wide (0.4-2.5): the old ±35% bound left such an outlier
        2.25x above the rest — a 7 dB jump between adjacent segments; the
        final peak guard still protects the output. `segments` is a list of
        (audio, gap_before) pairs — a phrase start (punctuation position)
        gets a slightly longer pause than an intra-phrase chunk boundary."""
        if not segments:
            return np.zeros(0, dtype=np.float32)
        levels = [float(np.sqrt((p ** 2).mean())) for p, _ in segments if len(p)]
        target = float(np.median(levels)) if levels else 0.0
        f = max(1, int(0.008 * self.sample_rate))
        out = []
        for p, gap in segments:
            rms = float(np.sqrt((p ** 2).mean()))
            if rms > 1e-6 and target > 1e-6:
                p = p * float(np.clip(target / rms, 0.4, 2.5))
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
        """Remaining dead-air stretches (whatever survived the retry in
        _generate_with_retry) longer than `max_pause` are shortened to
        `keep` seconds with small fades, so the flow of speech stays
        continuous. Regions come from windowed quiet density — the model
        peppers its silences with tiny blips that would chop a 1 s pause
        into sub-threshold runs. Deliberate intra-sentence pauses — strong
        lead-in 0.26 s, phrase 0.16 s, chunk 0.12 s — stay below `max_pause`
        untouched; the 0.45 s sentence pause is inserted by the caller,
        outside this pass."""
        regions = [r for r in self._dead_air_regions(audio, rel_floor=rel_floor)
                   if (r[1] - r[0]) > int(max_pause * self.sample_rate)]
        if not regions:
            return audio
        sr = self.sample_rate
        f = max(1, int(0.008 * sr))
        out, pos = [], 0
        for a, b in regions:
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

    def _generate_chunk(self, cache, off, chunk, rng=None):
        tokens = self.sp.encode(chunk, out_type=int)
        text_emb = self.lut[np.asarray(tokens)][None].astype(np.float32)

        r = rng if rng is not None else self.rng
        noise = (r.standard_normal((1, self.ldim)) * (self.temp**0.5)).astype(np.float32)
        lat, _, cache, off = self._flow_step(
            np.full((1, 1, self.ldim), np.nan, np.float32), text_emb, off, noise, cache)
        latents = [lat]

        words = len(chunk.split())
        frames_after_eos = (3 if words <= 4 else 1) + 2
        max_gen_len = int(np.ceil((len(tokens) / self.tps_est + self.gen_pad) * self.frame_rate))

        eos_step = None
        for step in range(max_gen_len):
            noise = (r.standard_normal((1, self.ldim)) * (self.temp**0.5)).astype(np.float32)
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
    ap = argparse.ArgumentParser(
        description="Persian TTS via the pure-ONNX engine (no torch)")
    ap.add_argument("text", nargs="?", default="سلام، حال شما چطور است؟",
                    help="Persian text or phonemes")
    ap.add_argument("voice", nargs="?",
                    default=str(BASE / "voices" / "male_hello.wav"),
                    help="reference voice WAV")
    ap.add_argument("out", nargs="?",
                    default=str(BASE / "output" / "tts_onnx.wav"),
                    help="output WAV path")
    ap.add_argument("--seed", type=int, default=None,
                    help="reproduce a run exactly")
    ap.add_argument("--pack", action="store_true",
                    help="merge comma phrases into longer breaths")
    args = ap.parse_args()

    eng = OnnxTts(seed=args.seed)
    t0 = time.perf_counter()
    audio = eng.synthesize_text(args.text, args.voice,
                                mode="pack" if args.pack else "split")
    dt = time.perf_counter() - t0

    import soundfile as sf

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, audio, eng.sample_rate)
    print(f"text: {args.text}")
    print(f"generated {len(audio)/eng.sample_rate:.2f}s audio in {dt:.2f}s -> {args.out}")


if __name__ == "__main__":
    main()
