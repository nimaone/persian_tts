# نوتبوک‌های Colab

نوت‌بوک آمادهٔ اجرا برای موتور pocket. کد از GitHub clone می‌شود،
وزن‌ها از Hugging Face دانلود می‌شوند و موتور خودش بین **CUDA / CPU** بنچمارک
می‌گیرد و بردنده را اجرا می‌کند (نصب پیش‌فرض: `onnxruntime-gpu` برای T4 کولاب).

| نوت‌بوک | موتور | وزن‌ها | وضعیت |
|---|---|---|---|
| [`colab_pocket.ipynb`](colab_pocket.ipynb) | pocket (پارسی‌گو) | [`Nimaone/pocket-tts-farsi-v2-onnx`](https://huggingface.co/Nimaone/pocket-tts-farsi-v2-onnx) | ✅ آمادهٔ اجرا |

## قبل از اجرا

1. **پوش کد**: نوت‌بوک `main` را از GitHub می‌گیرد — آخرین کامیت‌ها را push کن:
   `git push origin main`
2. **GPU runtime**: در Colab — `Runtime → Change runtime type → T4 GPU`.
3. اگر خواستی provider را اجباری کنی: سلول تنظیمات → `PARSIGO_EP = "cpu" | "cuda" | "dml"`.
