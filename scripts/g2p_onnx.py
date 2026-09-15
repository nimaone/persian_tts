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
TO_PHONEMES = str.maketrans({"/": "a", "a": "A", "@": "?", "$": "S", "c": "C"})


def encode(text: str) -> list[int]:
    return [b + 3 for b in text.encode("utf-8")]


def decode(ids) -> str:
    return bytes(i - 3 for i in ids if i >= 3).decode("utf-8", errors="replace")


class OnnxG2P:
    def __init__(self, pkg_dir=PKG):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        prov = ["CPUExecutionProvider"]
        self.s_enc = ort.InferenceSession(str(pkg_dir / "g2p_encoder.onnx"), opts, providers=prov)
        self.s_dec = ort.InferenceSession(str(pkg_dir / "g2p_decoder.onnx"), opts, providers=prov)
        self.max_len = 512

    def phonemise(self, text: str, keep_ezafe: bool = False) -> str:
        text = normalize_for_model(text)
        text = text.replace("؟", "").replace("?", "")
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
        raw = decode(dec[1:])
        # "1" marks the ezafe. The chunker needs it to avoid splitting a bound
        # noun phrase ("?eqtesAde1 ?AmrikA"); it is stripped per chunk before
        # the TTS model sees the text.
        out = raw.translate(TO_PHONEMES)
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
