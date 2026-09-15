# امکان‌سنجی ONNX برای pocket-tts-farsi-v2

> نتیجه: **ممکن است** — با اسپایک‌های عملی اثبات شد. هر ۴ جزء مدل به ONNX تبدیل و
> از نظر عددی با PyTorch تأیید شدند. الگوی `whisper_flow_farsi` (فایل‌های .onnx +
> حلقه اجرای Python) قابل بازتولید است، اما برخلاف ASR که «یک فایل model.onnx»
> کافی بود، اینجا مدل ۵ گراف جدا نیاز دارد چون تولید گفتار خودبازگشتی (autoregressive)
> و stateful است.

## نتایج اسپایک (سپتامبر ۲۰۲۶)

| جزء | گراف | نتیجه عددی (اختلاف با PyTorch) | حجم |
|---|---|---|---|
| Mimi encoder (صدای مرجع → latent) | `mimi_encoder.onnx` | 2.7e-05 | ~39 MB |
| Mimi decoder (latent → صدا، یک گام + state) | `mimi_decoder_step.onnx` | 2.4e-07 (۲ گام زنجیره‌ای) | ~41 MB |
| FlowLM گام prompt (متن + BOS → latent) | `flow_lm_step.onnx` | 1.19e-06 | ~341 MB* |
| FlowLM گام تولید (latent → latent) | `flow_lm_gen_step.onnx` | 3.2e-06 (۳ گام زنجیره‌ای) | ~341 MB* |
| G2P (T5) encoder | `g2p_encoder.onnx` | 2.2e-07 | ~60 MB |

\* وزن‌ها در فایل `.data` کناری هستند (torch.onnx > 2GB پشتیبانی نمی‌کند؛ با float32 ~341MB).

اسکریپت‌های تولید: `scripts/spike_onnx*.py`، صحت‌سنجی زنجیره‌ای: `scripts/spike_chain_test.py`

## ریشه مشکلات export و راه‌حل‌ها

کد اصلی pocket-tts مستقیماً قابل export نیست. سه مانع اصلی:

1. **گاردهای وابسته به داده در KV-cache** — `complete_kv` در
   `pocket_tts/modules/attention.py:14` مقدار offset را با `.item()` به Python
   برمی‌گرداند و با آن اندیس‌گذاری می‌کند؛ `torch.export` روی slice وابسته به
   tensor خطای `GuardOnDataDependentSymNode` می‌دهد.
   **راه‌حل:** بازنویسی clean-room گام تبدیل (spike_onnx4.py) که offset را به‌صورت
   ورودی 0-d int64 گراف می‌گیرد و همه slice-ها را با انت‌های affine
   (`k_l[:off]`, `k_l[off+S:]`) می‌سازد. نتیجه: تطابق 0.0 با کد اصلی در eager.

2. **SDPA با mask بولی روی طول نمادین** — تجزیه ONNX آن روی `Eq(len, 0)` گارد
   می‌گذارد. **راه‌حل:** attention صریح (matmul + softmax + bias شناور) به‌جای
   `F.scaled_dot_product_attention`.

3. **نویز تصادفی در گراف** — `torch.nn.init.normal_` در `flow_lm.forward` قابل
   trace نیست (و seed در ONNX قابل حمل نیست). **راه‌حل:** نویز باید مثل
   whisper_flow_farsi در host (numpy) نمونه‌گیری شود و به‌عنوان ورودی `noise`
   به گراف داده شود. در تست‌ها نویز صفر گذاشته شد تا دو مسیر قابل مقایسه باشند.

همچنین: state مدل (dict of dict of tensors) باید مسطح و به‌صورت ورودی/خروجی
صریح گراف رد شود — ۱۸ tensor برای FlowLM و ۵۶ tensor برای Mimi decoder.

## تفاوت با whisper_flow_farsi

| | ASR (Shenava) | TTS (pocket-tts) |
|---|---|---|
| تعداد گراف | ۱ (`model.onnx`) | ۵ گراف |
| حالت | بدون state (CTC یک‌باره) | stateful + خودبازگشتی (۱۲.۵ گام/ثانیه) |
| حلقه اجرا | یک‌بار decode | حلقه گام‌به‌گام با threading کَش |
| نویز | ندارد | نمونه‌گیری در host و تزریق به گراف |

## آنچه برای نسخه کامل مانده (برآورد: ۳ تا ۵ روز کار)

1. **گراف prompt با صدا**: گام اول تولید شامل conditioning صدای مرجع است
   (`speaker_proj` + latents صدا) — باید مثل گام متن export شود.
2. **G2P decoder**: فقط encoder تست شد؛ decoder T5 با حلقه greedy در Python
   (beam=5 هم ممکن است ولی در ONNX پیچیده است — greedy برای G2P معمولاً کافی است،
   باید از نظر کیفیت G2P مقایسه شود).
3. **runtime کامل** (`tts_onnx.py`): حلقه تولید با ORT — نمونه‌گیری نویز،
   threading کَش K/V بین گام‌ها، تشخیص EOS، فراخوانی mimi decoder در هر گام،
   chunking متن (۲۲ توکن) با قوانین ezafe.
4. **تست انتها-به-انتها**: تولید چند جمله و مقایسه WER/شنیداری با خروجی PyTorch.
5. **بهینه‌سازی (اختیاری)**: int8 quantization با ORT (pocket-tts خودش با torchao
   ~۲۷٪ سرعت و ~۴۸٪ حافظه بهتر می‌شود؛ ORT می‌تواند مشابه بدهد).

## مزیت‌های نسخه ONNX

- بدون وابستگی به torch (۱.۵GB+) — فقط onnxruntime (~۵۰MB)
- قابل اجرا در C#/.NET، جاوا، Node.js و موبایل
- startup سریع‌تر (بدون بارگذاری PyTorch)
- اما برای این پروژه که torch نصب است و روی CPU کار می‌کند، سود عملیاتی محدود است؛
  ارزش اصلی در استقرار سبک (installer مثل دیکته‌یار) است.

## بنچمارک سرعت و دقت (ONNX در برابر موتور اصلی PyTorch)

شرایط تست: همان جمله (`salAm hAle SomA Cetor ?ast`)، همان صدای مرجع (female_hello)،
**همان دنباله نویز** برای هر دو موتور (seed=1234) — یعنی تنها متغیر، موتور اجراست.
(اسکریپت: `scripts/bench_onnx.py`)

### دقت

| معیار | نتیجه |
|---|---|
| طول صدا | هر دو 2.80s — EOS در همان گام (۳۵ latent) |
| اختلاف waveform (max) | 5.1e-04 |
| اختلاف نسبی RMS | **3.7e-04** (ناشنیدنی) |
| اختلاف latent در هر گام | ~1e-06 (پایدار، بدون واگرایی در ۳۵ گام) |

یعنی خروجی ONNX عملاً بیت‌به‌بیتِ شنیداری با PyTorch یکسان است.

### سرعت (ویندوز، CPU، تک‌نخسته؛ median)

| مؤلفه | PyTorch | ONNX | نسبت |
|---|---:|---:|---|
| Mimi encoder (کدگذاری صدای مرجع ۵ ثانیه) | 805 ms | **373 ms** | **ONNX ۲.۲x سریع‌تر** |
| FlowLM هر گام تولید | 67.6 ms | 78.8 ms | ONNX ۱۴٪ کندتر |
| Mimi decode هر گام | 24.7 ms | 28.7 ms | ONNX ۱۴٪ کندتر |
| سنتز کامل (۲.۸ ثانیه صدا، شامل setup صدا) | 3431 ms (تولیدی threaded: 3450) | 3789 ms | ONNX ~۱۰٪ کندتر |
| ضریب زمان واقعی (RTF) | 0.81x | 0.74x | هر دو سریع‌تر از realtime |

**چرا گام‌های ONNX کمی کندترند؟** الگوی KV-cache به‌صورت ورودی/خروجی گراف یعنی
کپی ~۱۲.۶MB در هر گام، در حالی که torch کش را درجا به‌روز می‌کند. راه‌های بهبود
(برای نسخه کامل): IOBinding با بافر از پیش تخصیص‌یافته، بازگرداندن فقط K/V جدید
(scatter در host)، یا int8 — که فعلاً به‌دلیل shape inference ناقصِ sliceهای
داینامیک روی این گراف ممکن نشد (`Incomplete symbolic shape inference`).

**نکته:** کانولوشن‌های Mimi در ORT به‌وضوح سریع‌ترند (۲.۲x)؛ ترنسفورمر کوچک
FlowLM نه. یک موتور هیبریدی (Mimi در ONNX + FlowLM در torch) می‌تواند از هر دو
جهت بهترین باشد.

> **به‌روزرسانی:** این سه مسیر بهینه‌سازی (K/V-only، IOBinding، هیبریدی)
> پیاده‌سازی و اندازه‌گیری شدند — موتور هیبریدی **۱۳٪ سریع‌تر از موتور اصلی**
> شد. نتایج کامل: [onnx-optimizations.md](onnx-optimizations.md)

## نتیجه‌گیری

مسیر فنی باز است و سخت‌ترین بخش‌ها (KV-cache stateful و attention نمادین) حل و
تأیید شده‌اند. برای شروع، پیشنهاد: ابتدا runtime کامل روی همین ۵ گراف ساخته شود،
بعد از تأیید خروجی صوتی، quantization و بسته‌بندی.
