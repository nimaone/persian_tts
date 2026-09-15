# Persian TTS pipeline: normalize -> G2P phonemes -> pocket-tts synthesis
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

BASE = Path(__file__).resolve().parent.parent
G2P_DIR = BASE / "model" / "g2p"
V2_DIR = BASE / "model" / "v2"

sys.path.insert(0, str(V2_DIR))
from normalize_fa import normalize_for_model  # shipped with the model repo

TO_PHONEMES = str.maketrans({"/": "a", "a": "A", "@": "?", "$": "S", "c": "C"})


def load_g2p():
    tok = AutoTokenizer.from_pretrained(str(G2P_DIR))
    model = T5ForConditionalGeneration.from_pretrained(str(G2P_DIR)).eval()
    return tok, model


def phonemise(text: str, tok, g2p) -> str:
    text = normalize_for_model(text)
    text = text.replace("؟", "").replace("?", "")
    enc = tok([text], add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        out = g2p.generate(**enc, num_beams=5, max_length=512, early_stopping=True)
    raw = tok.batch_decode(out, skip_special_tokens=True)[0].strip()
    # "1" marks the ezafe; strip it before the model sees the text
    return raw.translate(TO_PHONEMES).replace("1", "")


def load_tts():
    from pocket_tts import TTSModel

    return TTSModel.load_model(config=str(V2_DIR / "model.yaml"))


def synthesize(text: str, voice: str | Path, out_path: str | Path, tts=None, g2p=None):
    """Full pipeline for one sentence. Reuse tts=(model, state) / g2p=(tok, model) if given."""
    tok, g2p_model = g2p or load_g2p()
    model = tts or load_tts()

    ph = phonemise(text, tok, g2p_model)
    print(f"Persian : {text}")
    print(f"Phonemes: {ph}")

    voice_state = model.get_state_for_audio_prompt(str(voice))
    t0 = time.time()
    audio = model.generate_audio(voice_state, ph)
    dt = time.time() - t0

    import soundfile as sf  # noqa: TID252

    sf.write(str(out_path), audio, model.sample_rate)
    dur = audio.shape[-1] / model.sample_rate
    print(f"Generated {dur:.2f}s of audio in {dt:.1f}s -> {out_path}")
    return audio, model.sample_rate
