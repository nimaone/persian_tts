# Demo web server for the pure-ONNX Persian TTS engine.
#
#   ./env/Scripts/python.exe scripts/server.py     -> http://127.0.0.1:8000
#
# Endpoints:
#   GET  /                  the single-page UI (web/index.html)
#   GET  /api/voices        available reference voices (builtin + uploaded)
#   POST /api/tts           {text, voice} -> {id, phonemes, duration, ...}
#   GET  /api/audio/{id}    generated WAV
#   POST /api/voice/upload  upload a custom voice (auto-trimmed to 5 s)
import io
import re
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

WEB = BASE / "web" / "index.html"
UPLOAD_DIR = BASE / "uploads" / "voices"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

SR = 24000
PAUSE_S = 0.25          # pause between sentences (model card recommendation)
MAX_TEXT = 800          # keep demo requests bounded
MAX_VOICES = 64

BUILTIN_VOICE_META = {
    "female_hello.wav": ("بانو · صمیمی", "صدای زن، لحن آرام و دوستانه"),
    "female_short.wav": ("بانو · روایت", "صدای زن، روایت‌گر"),
    "male_news.wav": ("آقا · خبری", "صدای مرد، لحن خبرگزاری"),
}

app = FastAPI(title="پارسی‌گو — Persian TTS demo")

_engine = None
_engine_lock = threading.Lock()
_audio_store: dict[str, dict] = {}
_store_lock = threading.Lock()

# punctuation-aware phrase splitting lives with the engine (single source of
# truth for where pauses may fall)
from tts_onnx import plan_phrases  # noqa: E402


def get_engine():
    global _engine
    if _engine is None:
        from tts_onnx import OnnxTts

        _engine = OnnxTts()
        if not hasattr(_engine, "_g2p"):
            from g2p_onnx import OnnxG2P

            _engine._g2p = OnnxG2P(_engine.dir)
    return _engine


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!؟])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def list_voices() -> list[dict]:
    out = []
    for f in sorted((BASE / "voices").glob("*.wav")):
        name, desc = BUILTIN_VOICE_META.get(f.name, (f.stem, ""))
        out.append({"id": f.name, "name": name, "desc": desc, "builtin": True})
    for f in sorted(UPLOAD_DIR.glob("*.wav")):
        out.append({
            "id": f"upload:{f.name}", "name": f.name[:22],
            "desc": "صدای بارگذاری‌شده", "builtin": False,
        })
    return out[:MAX_VOICES]


def voice_path(voice_id: str) -> Path:
    if voice_id.startswith("upload:"):
        p = UPLOAD_DIR / Path(voice_id[7:]).name
    else:
        p = BASE / "voices" / Path(voice_id).name
    if not p.is_file():
        raise HTTPException(404, "voice not found")
    return p


class TTSRequest(BaseModel):
    text: str
    voice: str
    pace: float = 1.0


@app.get("/")
def index():
    return FileResponse(WEB)


@app.get("/api/voices")
def voices():
    return {"voices": list_voices()}


@app.post("/api/tts")
def tts(req: TTSRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "متن خالی است")
    if len(text) > MAX_TEXT:
        raise HTTPException(400, f"متن طولانی است (حداکثر {MAX_TEXT} نویسه)")

    engine = get_engine()
    pace = float(min(max(req.pace, 0.6), 1.5))
    phonemes_all, chunks = [], []

    with _engine_lock:
        for sent in split_sentences(text):
            # punctuation-aware plan: each phrase (comma/dash/colon-delimited)
            # is a pause unit with its own gap, and short lead-ins ending in
            # strong punctuation ("سؤال اصلی:") stay standalone
            try:
                plan = plan_phrases(sent, engine._g2p, engine.sp)
            except ValueError as e:
                raise HTTPException(400, "متن فارسی معتبری پیدا نشد") from e
            if not plan:
                continue
            # model card: retry a runaway once (stochastic; 2nd attempt usually ends)
            tokens = sum(len(engine.sp.encode(p.replace("1", ""), out_type=int))
                         for p, _ in plan)
            cap = tokens / engine.tps_est + engine.gen_pad + 1
            for attempt in range(2):
                audio = engine.synthesize(plan, voice_path(req.voice), pace=pace)
                if len(audio) / SR <= cap + 2.0:  # multi-chunk texts run longer
                    break
            phonemes_all.append(" ".join(p.replace("1", "") for p, _ in plan))
            chunks.append(audio)
            chunks.append(np.zeros(int(PAUSE_S / pace * SR), dtype=audio.dtype))

    if not chunks:
        raise HTTPException(400, "متنی برای ساخت صدا پیدا نشد")
    audio = np.concatenate(chunks[:-1])  # drop trailing pause
    duration = len(audio) / SR

    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV")
    audio_id = uuid.uuid4().hex[:12]
    with _store_lock:
        _audio_store[audio_id] = {
            "wav": buf.getvalue(),
            "text": text,
            "voice": req.voice,
            "duration": duration,
            "phonemes": " ".join(phonemes_all),
        }
    return {
        "id": audio_id,
        "phonemes": " ".join(phonemes_all),
        "duration": round(duration, 2),
        "pace": pace,
        "sentences": len(phonemes_all),
        "elapsed": None,
    }


@app.get("/api/audio/{audio_id}")
def audio(audio_id: str):
    with _store_lock:
        item = _audio_store.get(audio_id)
    if item is None:
        raise HTTPException(404, "not found")
    return Response(content=item["wav"], media_type="audio/wav")


@app.post("/api/voice/upload")
async def upload_voice(file: UploadFile = File(...)):
    import scipy.signal as sig

    data = await file.read()
    try:
        wav, sr = sf.read(io.BytesIO(data))
    except Exception:
        raise HTTPException(400, "فایل صوتی خوانده نشد — WAV پیشنهاد می‌شود")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if sr != SR:
        g = np.gcd(int(sr), SR)
        wav = sig.resample_poly(wav, SR // g, sr // g).astype(np.float32)
    if len(wav) < SR:  # < 1 s is too short to clone
        raise HTTPException(400, "صدای مرجع باید حداقل ۱ ثانیه باشد")
    wav = wav[: 5 * SR]  # model card: prompts beyond 5 s are out of distribution

    vid = f"upload:{uuid.uuid4().hex[:8]}.wav"
    sf.write(UPLOAD_DIR / vid[7:], wav, SR, format="WAV")
    return {"id": vid, "name": vid[7:], "seconds": round(len(wav) / SR, 1)}


if __name__ == "__main__":
    print("loading engine (first request may take a moment)...")
    get_engine()
    print("demo:  http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
