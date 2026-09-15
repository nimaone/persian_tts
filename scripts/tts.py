# CLI: synthesize a Persian sentence to a WAV file.
# Usage:
#   python scripts/tts.py "متن فارسی" [voice.wav] [out.wav]
# Defaults: voices/female_hello.wav -> output/tts_out.wav
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from persian_tts import BASE, synthesize


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "سلام، حال شما چطور است؟"
    voice = sys.argv[2] if len(sys.argv) > 2 else str(BASE / "voices" / "female_hello.wav")
    out_path = sys.argv[3] if len(sys.argv) > 3 else str(BASE / "output" / "tts_out.wav")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    synthesize(text, voice, out_path)


if __name__ == "__main__":
    main()
