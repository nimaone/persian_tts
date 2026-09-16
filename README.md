# Persian TTS (pocket-tts-farsi-v2)

تبدیل متن فارسی به گفتار با کلونینگ صدا، کاملاً آفلاین روی CPU.

## ساختار پروژه

```
persian_tts/
├── env/               محیط مجازی پایتون (torch, transformers, pocket-tts)
├── model/
│   ├── v2/            مدل اصلی TTS (mehdi-hf/pocket-tts-farsi-v2)
│   ├── g2p/           مدل G2P فارسی→فونم (mehdi-hf/Homo-GE2PE-Persian-HF)
│   └── onnx/          پکیج ONNX یکپارچه (بدون torch) + manifest.json
├── voices/            صداهای مرجع برای کلونینگ (حداکثر ۵ ثانیه)
├── output/            فایل‌های WAV تولیدشده
├── scripts/
│   ├── persian_tts.py ماژول خط لوله (load_g2p, phonemise, synthesize)
│   ├── tts.py         CLI اجرای سریع (torch)
│   ├── tts_onnx.py    موتور ONNX خالص (بدون torch)
│   ├── export_unified.py  سازنده پکیج model/onnx/
│   └── onnx_dev/      اسکریپت‌های توسعه ONNX (spike/bench)
└── README.md
```

## راه‌اندازی از کلون تازه

وزن‌های مدل در گیت نیستند (۹۲۰MB) — این سه مرحله:

```bash
# ۱) محیط مجازی و وابستگی‌ها (برای مسیر torch؛ مسیر ONNX فقط onnxruntime/numpy/scipy/soundfile/sentencepiece/pedalboard می‌خواهد)
python -m venv env
./env/Scripts/python.exe -m pip install -r requirements.txt

# ۲) دانلود مدل‌ها از HuggingFace (در دسترس است، بدون پراکسی)
mkdir -p model/v2 model/g2p
B=https://huggingface.co/mehdi-hf/pocket-tts-farsi-v2/resolve/main
for f in model.yaml normalize_fa.py tokenizer_ph.model model.safetensors; do
  curl -L -o "model/v2/$f" "$B/$f"; done
G=https://huggingface.co/mehdi-hf/Homo-GE2PE-Persian-HF/resolve/main
for f in config.json generation_config.json tokenizer_config.json added_tokens.json model.safetensors; do
  curl -L -o "model/g2p/$f" "$G/$f"; done

# ۳) ساخت پکیج ONNX یکپارچه (اختیاری — برای موتور بدون torch و دموی وب)
./env/Scripts/python.exe scripts/export_unified.py
```

## دموی وب (UI)

```bash
./env/Scripts/python.exe scripts/server.py
# سپس در مرورگر: http://127.0.0.1:8000
```

- رابط فارسی RTL با تم تیره مدرن (`web/index.html`) — بدون نیاز به اینترنت
- انتخاب صدای مرجع + **بارگذاری صدای خودتان** (خودکار به ۵ ثانیه بریده می‌شود)
- نمایش فونم‌های تولیدشده، پلیر با موج‌نگار (waveform) و seek، دانلود WAV، تاریخچه
- متن چندجمله‌ای خودکار جمله‌به‌جمله ساخته و با مکث ۰.۲۵ ثانیه‌ای چسبانده می‌شود
- همان موتور ONNX بدون torch — کل سرور فقط onnxruntime/numpy/scipy/soundfile می‌خواهد

## استفاده

```bash
# جمله ساده با صدای پیش‌فرض (زن)
./env/Scripts/python.exe scripts/tts.py "سلام، حال شما چطور است؟"

# صدای مرد خبرنگار با خروجی دلخواه
./env/Scripts/python.exe scripts/tts.py "متن شما" voices/male_news.wav output/my.wav

# در کد پایتون
import sys; sys.path.insert(0, "scripts")
from persian_tts import load_g2p, load_tts, synthesize
g2p = load_g2p(); tts = load_tts()
synthesize("سلام دنیا", "voices/female_hello.wav", "output/x.wav", tts=tts, g2p=g2p)
```

## نکات مهم

- **متن طولانی**: جمله‌ها را جدا کنید، هر جمله را جدا فونم‌کنید و خروجی‌ها را با ~۰.۲۵ ثانیه سکوت به هم بچسبانید (برای مکث در جای درست).
- **صدای مرجع**: حتماً ≤ ۵ ثانیه باشد؛ طولانی‌تر باعث می‌شود مدل جمله خودش را ادامه دهد به جای متن شما.
- **تکرار بی‌پایان**: اگر خروجی تکراری شد (fanAvariiii...)، دوباره بسازید — تصادفی است و معمولاً بار دوم درست تمام می‌شود. (در CLI نسخهٔ ONNX هر اجرا seed تازه می‌گیرد؛ برای بازتولید دقیق یک خروجی، `--seed 42` بدهید.)
- **فونم‌ها مستقیم**: مدل متن فارسی خام نمی‌فهمد؛ همیشه از `phonemise()` رد کنید.
- **نسخه ONNX (کاملاً بدون torch، حتی G2P)**: `./env/Scripts/python.exe scripts/tts_onnx.py "سلام، حال شما چطور است؟" voices/female_hello.wav out.wav` — متن فارسی را مستقیم می‌گیرد (جزئیات: `docs/onnx-feasibility.md` و `docs/onnx-optimizations.md`)

## اعتبارها و مجوزها

وزن‌های مدل ساختهٔ این مخزن نیستند؛ این پروژه از مدل‌های منتشرشدهٔ دیگران استفاده می‌کند:

| بخش | مدل / کد | سازنده | مجوز |
|---|---|---|---|
| TTS فارسی + کلونینگ صدا | [`mehdi-hf/pocket-tts-farsi-v2`](https://huggingface.co/mehdi-hf/pocket-tts-farsi-v2) | `mallahyari` — کد و خط لولهٔ آموزش: [`mallahyari/pocket-tts`](https://github.com/mallahyari/pocket-tts) | **CC-BY-NC-4.0** |
| G2P فارسی→فونم | [`mehdi-hf/Homo-GE2PE-Persian-HF`](https://huggingface.co/mehdi-hf/Homo-GE2PE-Persian-HF) | بازبسته‌بندی از [`MahtaFetrat/Homo-GE2PE-Persian`](https://huggingface.co/MahtaFetrat/Homo-GE2PE-Persian) نوشتهٔ **Elnaz Rahmati** و همکاران | MIT © 2025 Elnaz Rahmati |
| کتابخانه و معماری پایه | [`pocket-tts`](https://pypi.org/project/pocket-tts/) نسخهٔ ۳.۱.۰ | [Kyutai](https://kyutai.org) | MIT |

**هشدار مجوز — استفادهٔ تجاری ممنوع:** مدل TTS با مجوز **CC-BY-NC-4.0** منتشر شده است (محدودیتی که از دادهٔ آموزش به ارث رسیده) و استفادهٔ تجاری از خروجی آن مجاز نیست. برای استفادهٔ تجاری باید از سازنده مجوز بگیرید یا سراغ مدل دیگری بروید.

**دربارهٔ G2P:** آنچه در `mehdi-hf/Homo-GE2PE-Persian-HF` قرار دارد مدل جدیدی نیست؛ وزن‌ها بایت‌به‌بایت همان `MahtaFetrat/Homo-GE2PE-Persian` هستند و تغییر فقط در بسته‌بندی است (فرمت `from_pretrained` بدون وابستگی به Parsivar). همهٔ اعتبار مدل به نویسندگان اصلی می‌رسد.

کد این مخزن (اسکریپت‌ها و رابط وب) نوشتهٔ همین پروژه است؛ مدل‌ها و کتابخانهٔ پایه متعلق به سازندگان بالا هستند.

## نکته فنی نصب (پچ فورک)

پکیج PyPI نسخه ۳.۱.۰ سه فلگ `capitalize_first_letter` و... را نمی‌شناسد که بدون آن‌ها
کلمه‌ی اول هر chunk حذف می‌شود. سه فایل فورک `mallahyari/pocket-tts` روی
`env/Lib/site-packages/pocket_tts/` جایگزین شده‌اند:
`utils/config.py`, `models/text_chunking.py`, `models/tts_model.py`.
اگر محیط را از نو ساختید، این سه فایل را دوباره از
`https://cdn.jsdelivr.net/gh/mallahyari/pocket-tts@main/<path>` بگیرید و جایگزین کنید.
