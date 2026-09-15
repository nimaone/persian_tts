# Pure-ONNX G2P: Persian text -> phoneme string, NO torch.
#
# Components:
#   model/onnx/g2p_encoder.onnx   T5 encoder (token ids -> hidden)
#   model/onnx/g2p_decoder.onnx   T5 decoder full-forward (ids + enc hidden -> logits)
#   ByT5 tokenizer                byte -> id (id = byte + 3), pure python
#   model/v2/normalize_fa.py      Persian normalization, pure python
# Greedy decoding host-side (encoder is re-run once; decoder re-runs on the
# growing sequence — the model is 2 layers, outputs are short, so this is
# both fast and cache-free).
import re
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

BASE = Path(__file__).resolve().parent.parent
PKG = BASE / "model" / "onnx"
sys.path.insert(0, str(BASE / "model" / "v2"))

from normalize_fa import normalize_for_model

# ByT5: 0=<pad> 1=</s> 2=<unk>, bytes start at 3 (verified against HF tokenizer)
PAD, EOS, UNK = 0, 1, 2

# ---- Latin -> Persian transliteration ----------------------------------
# normalize_for_model strips Latin entirely and the G2P model has only ever
# seen Persian script, so an English word in the text is silently dropped
# from the audio ("(sparse) است" -> "است"). Transliterate it to Persian
# script first — the same convention Persian speakers use (ویندوز، سرور).
_LATIN_DIGRAPHS = {
    "sh": "ش", "ch": "چ", "th": "ث", "ph": "ف", "gh": "غ", "kh": "خ",
    "wh": "و", "ck": "ک", "qu": "کو", "oo": "و", "ee": "ی", "ea": "ی",
    "ou": "او", "au": "او", "ai": "ای", "ay": "ای", "ey": "ای",
    "oi": "اوی", "oy": "اوی",
}
_LATIN_CHAR = {
    "a": "ا", "b": "ب", "c": "ک", "d": "د", "e": "", "f": "ف", "g": "گ",
    "h": "ه", "i": "ی", "j": "ج", "k": "ک", "l": "ل", "m": "م", "n": "ن",
    "o": "او", "p": "پ", "q": "ق", "r": "ر", "s": "س", "t": "ت", "u": "و",
    "v": "و", "w": "و", "x": "کس", "y": "ی", "z": "ز",
}
# short all-caps words are read letter by letter (GPU -> جی‌پی‌یو)
_LETTER_NAMES = {
    "a": "ای", "b": "بی", "c": "سی", "d": "دی", "e": "ای", "f": "اف",
    "g": "جی", "h": "اچ", "i": "آی", "j": "جی", "k": "کی", "l": "ال",
    "m": "ام", "n": "ان", "o": "او", "p": "پی", "q": "کیو", "r": "آر",
    "s": "اس", "t": "تی", "u": "یو", "v": "وی", "w": "دبلیو", "x": "ایکس",
    "y": "وای", "z": "زد",
}
# Common English words whose letter-by-letter transliteration comes out
# wrong; Persian tech convention instead (echo -> اکو not چاو, network ->
# نت‌ورک not نتواورک, windows -> ویندوز not وینداووس).
_LATIN_EXCEPTIONS = {
    "echo": "اکو", "state": "استیت", "network": "نت‌ورک",
    "windows": "ویندوز", "notebook": "نوت‌بوک", "photoshop": "فتوشاپ",
    "python": "پایتون", "recurrency": "ریکارانسی",
}
_LATIN_WORD = re.compile("[A-Za-z][A-Za-z'-]*")
_CLUSTER_START = set("پتکبجچذژزصضثفگسش")


def _transliterate_word(w: str) -> str:
    # hyphenated compounds transliterate part by part ("self-recurrency" ->
    # "سلف ریکارانسی", not the letter-mush of the whole run)
    if "-" in w.strip("-"):
        parts = [p for p in w.split("-") if p]
        if len(parts) > 1:
            return " ".join(_transliterate_word(p) for p in parts)
    lw = w.lower()
    if lw in _LATIN_EXCEPTIONS:
        return _LATIN_EXCEPTIONS[lw]
    if w.isupper() and 2 <= len(w) <= 5 and w.isalpha():
        return "‌".join(_LETTER_NAMES[c] for c in w.lower())
    out, i = [], 0
    while i < len(lw):
        two = lw[i : i + 2]
        if two in _LATIN_DIGRAPHS:
            out.append(_LATIN_DIGRAPHS[two])
            i += 2
        else:
            out.append(_LATIN_CHAR.get(lw[i], ""))
            i += 1
    s = "".join(out)
    # English onsets like sp/st/sk are impossible in Persian: اِسپارس not سپارس
    if len(s) >= 2 and s[0] == "س" and s[1] in _CLUSTER_START:
        s = "ا" + s
    return s or w
TO_PHONEMES = str.maketrans({"/": "a", "a": "A", "@": "?", "$": "S", "c": "C"})

# GE2P contextual misreadings, fixed on the phoneme output: "بعد همین..."
# comes out "bo?d" (= بود) while "بعد از..." is correctly "ba?d".
_PRON_FIX = {"bo?d": "ba?d"}

# The G2P degrades on long inputs: clean to ~40 words, repetitive collage
# from ~45 (measured on natural text; an earlier audit saw 5% failures
# already at 30). Punctuation normally keeps phrases far below this — the
# hole is a run-on sentence with no punctuation and no detachable
# preposition, which would reach the G2P whole. Split such text at a
# conjunction (fallback: hard cut) so no G2P call exceeds the window;
# windows are decoded separately and their phonemes rejoined, so callers
# still get one string.
_G2P_WINDOW_WORDS = 30
_G2P_CUT_WORDS = {"و", "یا", "ولی", "اما", "که", "زیرا", "چون", "پس",
                  "سپس", "بنابراین", "همچنین"}


def _g2p_windows(text: str) -> list[str]:
    words = text.split()
    if len(words) <= _G2P_WINDOW_WORDS:
        return [text]
    cut = None
    for i in range(_G2P_WINDOW_WORDS // 2,
                   min(len(words) - 4, _G2P_WINDOW_WORDS - 3)):
        if words[i] in _G2P_CUT_WORDS:
            cut = i
    if cut is None:
        cut = _G2P_WINDOW_WORDS // 2
    return (_g2p_windows(" ".join(words[:cut]))
            + _g2p_windows(" ".join(words[cut:])))


def transliterate_text(text: str) -> str:
    """Latin -> Persian transliteration only (the text pre-processing step
    of phonemise, without the G2P). Lets callers count the words the G2P
    will actually see ("self-recurrency" is ONE text word but TWO Persian
    words after transliteration)."""
    return _LATIN_WORD.sub(lambda m: _transliterate_word(m.group(0)), text)


def encode(text: str) -> list[int]:
    return [b + 3 for b in text.encode("utf-8")]


def decode(ids) -> str:
    # ids 3..258 are the 256 byte tokens; 259..383 are T5 <extra_id>s (the
    # decoder's vocab is 384) which training targets never emit — drop them
    # instead of crashing bytes() on out-of-range values
    return bytes(i - 3 for i in ids if 3 <= i <= 258).decode("utf-8", errors="replace")


class OnnxG2P:
    def __init__(self, pkg_dir=PKG):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        prov = ["CPUExecutionProvider"]
        self.s_enc = ort.InferenceSession(str(pkg_dir / "g2p_encoder.onnx"), opts, providers=prov)
        self.s_dec = ort.InferenceSession(str(pkg_dir / "g2p_decoder.onnx"), opts, providers=prov)
        self.max_len = 512

    def _decode_greedy(self, text: str) -> str:
        ids = encode(text)
        if not ids:
            return ""
        enc = self.s_enc.run(None, {"input_ids": np.asarray([ids], dtype=np.int64)})[0]

        dec = [PAD]  # decoder_start_token_id = 0
        for _ in range(self.max_len):
            logits = self.s_dec.run(None, {
                "decoder_input_ids": np.asarray([dec], dtype=np.int64),
                "encoder_hidden": enc})[0]
            nxt = int(np.argmax(logits[0, -1]))
            if nxt == EOS:
                break
            dec.append(nxt)
        return decode(dec[1:])

    def phonemise(self, text: str, keep_ezafe: bool = False) -> str:
        text = _LATIN_WORD.sub(lambda m: _transliterate_word(m.group(0)), text)
        text = normalize_for_model(text)
        # normalize keeps ":" — and a trailing colon makes GE2P repeat the
        # final word ("باقی‌ماندهٔ صادقانه:" -> "…sAdeqAne sAdeqAne",
        # "نکته:" -> "nokte nokte"); the punctuation's prosodic job is done
        # by the phrase splitter, so it is worthless to the G2P anyway
        text = text.replace("؟", "").replace("?", "").replace(":", "")
        raw = " ".join(self._decode_greedy(w) for w in _g2p_windows(text))
        # "1" marks the ezafe. The chunker needs it to avoid splitting a bound
        # noun phrase ("?eqtesAde1 ?AmrikA"); it is stripped per chunk before
        # the TTS model sees the text.
        out = raw.translate(TO_PHONEMES)
        out = " ".join(_PRON_FIX.get(w, w) for w in out.split())
        return out if keep_ezafe else out.replace("1", "")


if __name__ == "__main__":
    sentences = [
        "سلام، حال شما چطور است؟",
        "اقتصاد آمریکا را تا سال ۲۰۳۰ تغییر دهد.",
        "مادر کتاب را روی میز اتاق گذاشت و پنجره را باز کرد.",
        "شرکت آنتروپیک مدلی اقتصادی منتشر کرده است.",
        "امروز هوا خیلی خوب است و من می‌خواهم بیرون بروم.",
        "دانشگاه تهران در سال ۱۳۱۳ تأسیس شد.",
    ]
    g2p = OnnxG2P()
    for s in sentences:
        print(f"{s}\n  -> {g2p.phonemise(s)}")
