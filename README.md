# پارسی‌گو — تبدیل متن فارسی به گفتار (pocket-tts-farsi-v2)

تبدیل متن فارسی به گفتار با **کلونینگ صدا**، کاملاً آفلاین روی **CPU** — بدون GPU، بدون
ارتباط با اینترنت در زمان اجرا. متن فارسی را مستقیم می‌گیرد، فونم‌سازی و سنتز را خودش
انجام می‌دهد و خروجی WAV با نرخ نمونه ۲۴ کیلوهرتز میدهد.

دو مسیر اجرا دارد:

- **موتور ONNX خالص** (`scripts/tts_onnx.py` + `scripts/server.py`) — بدون torch،
  وابستگی‌های نصب‌شده ~۶۰MB (در برابر ~۱.۶GB با torch)، هم‌سرعت موتور اصلی.
- **مسیر torch اصلی** (`scripts/tts.py`) — خط لوله مرجع pocket-tts؛ برای توسعه و
  صحت‌سنجی.

دموی وب: `./env/Scripts/python.exe scripts/server.py` → http://127.0.0.1:8000

## ساختار پروژه

```
persian_tts/
├── env/               محیط مجازی پایتون (torch, transformers, pocket-tts)
├── model/
│   ├── v2/            مدل اصلی TTS (mehdi-hf/pocket-tts-farsi-v2)
│   ├── g2p/           مدل G2P فارسی→فونم (mehdi-hf/Homo-GE2PE-Persian-HF)
│   └── onnx/          پکیج ONNX یکپارچه (~۴۸۰MB) + manifest.json
├── voices/            صداهای مرجع داخلی (≤ ۵ ثانیه)
├── output/            فایل‌های WAV تولیدشده (در گیت نیست)
├── uploads/           صداهای بارگذاری‌شده از طریق دمو (در گیت نیست)
├── docs/              مستندات فنی: امکان‌سنجی و بهینه‌سازی ONNX
├── scripts/
│   ├── persian_tts.py ماژول خط لوله torch (load_g2p, phonemise, synthesize)
│   ├── tts.py         CLI مسیر torch
│   ├── tts_onnx.py    موتور و CLI مسیر ONNX (بدون torch)
│   ├── g2p_onnx.py    G2P خالص ONNX (قابل استفاده مستقل)
│   ├── server.py      سرور دموی وب (FastAPI) + API
│   ├── export_unified.py  سازندهٔ پکیج model/onnx/ (یکبار، با torch)
│   ├── test_tts_onnx.py   صحت‌سنجی و بنچمارک موتور ONNX در برابر torch
│   └── onnx_dev/      اسکریپت‌های توسعه ONNX (spike/bench؛ تاریخی)
├── web/index.html     رابط فارسی RTL دموی وب
└── README.md
```

## راه‌اندازی از کلون تازه

وزن‌های مدل در گیت نیستند (~۹۲۰MB) — این سه مرحله:

```bash
# ۱) محیط مجازی و وابستگی‌ها
python -m venv env
./env/Scripts/python.exe -m pip install -r requirements.txt
```

`requirements.txt` دو بخش دارد و می‌توانید فقط بخش موردنیاز را نصب کنید:

- **مسیر torch** (`scripts/tts.py`): `pocket-tts`، `transformers`، `soundfile`
- **مسیر ONNX** (`scripts/tts_onnx.py` و دمو): `onnxruntime`، `numpy`، `scipy`،
  `soundfile`، `sentencepiece` (+ `fastapi`/`uvicorn` برای سرور، `onnx`/`onnxscript`
  و `pedalboard` فقط برای ساخت مجدد پکیج و کنترل سرعت)

```bash
# ۲) دانلود مدل‌ها از HuggingFace (در دسترس است، بدون پراکسی)
mkdir -p model/v2 model/g2p
B=https://huggingface.co/mehdi-hf/pocket-tts-farsi-v2/resolve/main
for f in model.yaml normalize_fa.py tokenizer_ph.model model.safetensors; do
  curl -L -o "model/v2/$f" "$B/$f"; done
G=https://huggingface.co/mehdi-hf/Homo-GE2PE-Persian-HF/resolve/main
for f in config.json generation_config.json tokenizer_config.json added_tokens.json model.safetensors; do
  curl -L -o "model/g2p/$f" "$G/$f"; done

# ۳) ساخت پکیج ONNX یکپارچه (یکبار نیاز به torch دارد؛ بعد از آن torch دیگر لازم نیست)
./env/Scripts/python.exe scripts/export_unified.py
```

> **نکته:** مرحلهٔ ۳ به torch نیاز دارد چون گراف‌ها را از مدل اصلی export می‌کند، اما
> *خروجی* آن (`model/onnx/`) کاملاً بدون torch اجرا می‌شود. وقتی پکیج ساخته شد، می‌توانید
> مسیر ONNX را با همان وابستگی‌های ~۶۰MB اجرا کنید.

## دموی وب (UI)

```bash
./env/Scripts/python.exe scripts/server.py
# سپس در مرورگر: http://127.0.0.1:8000
```

- رابط فارسی RTL با تم تیره مدرن (`web/index.html`) — بدون نیاز به اینترنت
- انتخاب صدای مرجع + **بارگذاری صدای خودتان** (WAV/MP3/OGG/FLAC؛ خودکار به ۵ ثانیه
  بریده، یک کاناله و به ۲۴kHz ресample می‌شود؛ حداقل ۱ ثانیه لازم است)
- **دو حالت گفتار**: «با مکث ویرگول‌ها» (split — هر ویرگول/خط تیره مکث خودش را
  می‌گیرد) و «یکپارچه و روان» (pack — عبارات ویرگولی در نفس‌های بلندتر ادغام
  می‌شوند؛ مکث دونقطه/خط تیره/پایان جمله حفظ می‌شود)
- **کنترل سرعت گفتار** ۰٫۷× تا ۱٫۳۵× (در سرور ۰٫۶–۱٫۵ محدود می‌شود)
- نمایش فونم‌های تولیدشده، پلیر با موج‌نگار و seek، دانلود WAV، تاریخچهٔ ۸ مورد اخیر
- همان موتور ONNX بدون torch — کل سرور فقط onnxruntime/numpy/scipy/soundfile می‌خواهد

صداهای مرجع داخلی:

| فایل | نام در دمو | توضیح |
|---|---|---|
| `voices/female_hello.wav` | بانو · صمیمی | صدای زن، لحن آرام و دوستانه |
| `voices/female_short.wav` | بانو · روایت | صدای زن، روایت‌گر |
| `voices/male_news.wav` | آقا · خبری | صدای مرد، لحن خبرگزاری |

محدودیت دمو: حداکثر ۸۰۰ نویسه متن در هر درخواست.

## استفاده از خط فرمان

### مسیر ONNX (توصیه‌شده — بدون torch)

```bash
./env/Scripts/python.exe scripts/tts_onnx.py "متن فارسی" [صدای-مرجع.wav] [خروجی.wav] [--seed N] [--pack]
```

| آرگومان | پیشفرض | توضیح |
|---|---|---|
| متن | `سلام، حال شما چطور است؟` | متن فارسی **یا** رشتهٔ فونم (خودکار تشخیص داده می‌شود) |
| صدای مرجع | `voices/female_hello.wav` | فایل WAV مرجع (≤ ۵ ثانیه) |
| خروجی | `output/tts_onnx.wav` | مسیر فایل خروجی |
| `--seed N` | تصادفی | برای بازتولید دقیق یک خروجی |
| `--pack` | خالی | حالت «یکپارچه و روان» (ادغام عبارات ویرگولی) |

متن چندجمله‌ای بهصورت خودکار جمله‌بندی، فونم‌سازی و با مکث مناسب چسبانده می‌شود.

### مسیر torch

```bash
# جمله ساده با صدای پیش‌فرض (زن)
./env/Scripts/python.exe scripts/tts.py "سلام، حال شما چطور است؟"

# صدای مرد خبرنگار با خروجی دلخواه
./env/Scripts/python.exe scripts/tts.py "متن شما" voices/male_news.wav output/my.wav
```

> مسیر torch متن را **جمله به جمله** پردازش می‌کند (هر بار فونم‌سازی + سنتز). برای
> متن طولانی، جمله‌ها را خودتان جدا کنید و خروجی‌ها را با ~۰٫۴۵ ثانیه سکوت به هم
> بچسبانید — مسیر ONNX این کار را خودش انجام میدهد.

### در کد پایتون

```python
import sys; sys.path.insert(0, "scripts")

# مسیر ONNX (بدون torch)
from tts_onnx import OnnxTts
eng = OnnxTts()                       # seed= برای بازتولید
audio = eng.synthesize_text("سلام دنیا", "voices/female_hello.wav", mode="pack")
# یا مستقیماً روی فونم: eng.synthesize("salAm donyA", voice, pace=1.0)

# مسیر torch
from persian_tts import load_g2p, load_tts, synthesize
g2p = load_g2p(); tts = load_tts()
synthesize("سلام دنیا", "voices/female_hello.wav", "output/x.wav", tts=tts, g2p=g2p)

# G2P بهتنهایی
from g2p_onnx import OnnxG2P
print(OnnxG2P().phonemise("اقتصاد آمریکا"))   # -> "?eqtesAde ?AmrikA"
# keep_ezafe=True نشانگر اضافه را نگه می‌دارد: "?eqtesAde1 ?AmrikA"
```

## API سرور (برای توسعه‌دهندگان)

`scripts/server.py` یک FastAPI است با این نقاط پایانی:

| روش | مسیر | ورودی | خروجی |
|---|---|---|---|
| `GET` | `/` | — | صفحهٔ دمو (`web/index.html`) |
| `GET` | `/api/voices` | — | `{voices: [{id, name, desc, builtin}]}` |
| `POST` | `/api/tts` | `{text, voice, pace?, mode?}` | `{id, phonemes, duration, pace, mode, sentences}` |
| `GET` | `/api/audio/{id}` | — | WAV (`audio/wav`) |
| `POST` | `/api/voice/upload` | `multipart/form-data` (فیلد `file`) | `{id, name, seconds}` |

نمونه:

```bash
curl -X POST localhost:8000/api/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"سلام، حال شما چطور است؟","voice":"female_hello.wav","mode":"pack"}'
# -> {"id":"a1b2c3d4e5f6","phonemes":"salAm hAle SomA Cetor ?ast","duration":2.41,...}
# صدا:  curl localhost:8000/api/audio/a1b2c3d4e5f6 --output out.wav
```

- `voice` یا نام یک صدای داخلی است یا `upload:NAME.wav` (از `/api/voices`).
- `mode` باید `split` یا `pack` باشد؛ `pace` در ۰٫۶–۱٫۵ محدود می‌شود.
- صداهای بارگذاری‌شده در `uploads/voices/` ذخیره می‌شوند (در گیت نیستند).

## نکات مهم و رفع مشکل

- **صدای مرجع**: حتماً ≤ ۵ ثانیه باشد؛ طولانی‌تر باعث می‌شود مدل جملهٔ خودش را ادامه
  دهد به جای متن شما. در دمو و موتور ONNX برش خودکار انجام می‌شود. صدایی که هجای
  اولش بسیار بلندتر از بقیه است نیز بهطور خودکار اصلاح می‌شود (در غیر این صورت مدل
  آن شروع را روی کلمهٔ اول هر بخش بازتولید می‌کند).
- **تکرار یا خروجی معیوب**: موتور ONNX هر بخش را تا ۳ بار تولید می‌کند و بهترین را
  برمی‌گزیند (سکوت، انفجار-سکوت، کلمات افتاده، طول غیرعادی، سکوت میان‌جمله). اگر باز
  هم مشکل بود، دوباره اجرا کنید — تصادفی است. برای بازتولید دقیق `--seed 42` بدهید.
- **فونم‌ها مستقیم**: مدل متن فارسی خام نمی‌فهمد؛ همیشه از `phonemise()` یا
  `synthesize_text()` رد کنید. (در CLI مسیر ONNX، دادن رشتهٔ فونم هم پشتیبانی می‌شود.)
- **کلمات انگلیسی**: متن فارسی وسط انگلیسی (مثل «ویندوز»، «network») خودکار به خط
  فارسی تبدیل می‌شود — در غیر این صورت حذف می‌شوند.
- **متن طولانی بدون نقطه‌گذاری**: G2P در پنجره‌های ≤۳۰ کلمه با برش روی حرف ربط اجرا
  می‌شود؛ جملهٔ سراسری بدون نقطه‌گذاری از ~۴۵ کلمه به بالا وگرنه خروجی تکراری میداد.

## معماری و کیفیت صدا

چون مدل تصادفی (stochastic) است و گاهی خروجی معیوب میدهد، موتور ONNX لایه‌های
کنترل کیفیت قابل توجهی دارد (جزئیات و اعداد در `docs/`):

- **برنامه‌ریزی عبارات آگاه از نقطه‌گذاری** — ویرگول ۰٫۱۶s (نفس کوتاه)، دونقطه/خط
  تیره ۰٫۲۶s (مکث واقعی)، پایان جمله ۰٫۴۵s؛ عبارات کوتاه و افعال سبک آغازین ادغام
  می‌شوند تا کلمه‌ای نیفتد.
- **بسته‌بندی چانکها روی مرز واژه** — بودجهٔ ~۱۸ توکن؛ اصلاح مرز برای اضافه، حرف ربط،
  افعال سبک و حرف اضافه؛ چانکهای یک‌دو توکنی که بد درمی‌آیند ادغام می‌شوند.
- **تولید موازی** چانکهای مستقل (تا ۴ worker) + کش صدای مرجع (LRU، ۴ مورد).
- **تطبیق بلندی** بخشها به میانه + گارد پیک + fade کوتاه برای خروجی پیوسته.
- **فشرده‌سازی سکوت**های اضافی که مدل در میان جمله تولید می‌کند.
- **اعتبارسنجی**: موج خروجی ONNX در برابر torch با نویز یکسان ۷٫۵e-۴ اختلاف
  (ناشنیدنی) و هم‌سرعت (RTF ~۰٫۸۸×؛ جزئیات در `scripts/test_tts_onnx.py`).

## مستندات فنی

- [`docs/onnx-feasibility.md`](docs/onnx-feasibility.md) — امکان‌سنجی export ONNX:
  پنج گراف مدل، موانع export و راه‌حلها، نتایج عددی هر جزء.
- [`docs/onnx-optimizations.md`](docs/onnx-optimizations.md) — بهینه‌سازی‌های پیاده‌شده
  (K/V-only، IOBinding، موتور هیبریدی)، اندازه‌گیری‌ها، و یکپارچه‌سازی نهایی.

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
کلمهٔ اول هر chunk حذف می‌شود. سه فایل فورک `mallahyari/pocket-tts` روی
`env/Lib/site-packages/pocket_tts/` جایگزین شده‌اند:
`utils/config.py`, `models/text_chunking.py`, `models/tts_model.py`.
اگر محیط را از نو ساختید، این سه فایل را دوباره از
`https://cdn.jsdelivr.net/gh/mallahyari/pocket-tts@main/<path>` بگیرید و جایگزین کنید.
(این پچ فقط مسیر torch را نیاز دارد؛ مسیر ONNX از پکیج PyPI استفاده نمی‌کند.)
