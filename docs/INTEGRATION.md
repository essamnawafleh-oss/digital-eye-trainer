# Using the fine-tuned model in Digital Eye

<div dir="rtl">

**بالعربية:** افتح ملف `DIGITAL_EYE_SETTINGS.txt` داخل مجلد التصدير وانسخ الإعداد إلى العين الرقمية.
لا يلزم تعديل أي كود. النموذج الأصلي يبقى في مكانه؛ للرجوع إليه احذف `rec_model_dir`.
اترك `rec_model_name` كما هو: `arabic_PP-OCRv5_mobile_rec`.

</div>

## 1. What to change

Digital Eye selects the recogniser with two existing options (`document_parser_core/config.py`,
`PaddleConfig.rec_model_name` / `rec_model_dir`). Each export writes them, with the real path,
to `DIGITAL_EYE_SETTINGS.txt`:

```python
options = {
    "paddle": {
        "rec_model_name": "arabic_PP-OCRv5_mobile_rec",   # keep this name
        "rec_model_dir": r"C:\...\workspace\exports\arabic_PP-OCRv5_mobile_rec_ft_run1",
    }
}
result = parse_document("permit.pdf", None, options)
```

* `rec_model_name` must stay `arabic_PP-OCRv5_mobile_rec`: PaddleX applies the Arabic bidi
  post-processing based on this name, and the fine-tuned model was trained for exactly that.
* **Fallback:** the original folder (`<models_dir>/paddle/arabic_PP-OCRv5_mobile_rec`) is never
  touched. Remove `rec_model_dir` to go back.
* Before use: `python -m eye_trainer verify-export <folder>` (checksums + loads in PaddleOCR).
* Detection and the English cross-check model are unchanged.

## 2. Measured inside Digital Eye (synthetic data)

Digital Eye's own benchmark (`benchmarks/evaluate.py`, unchanged code), run twice by
`scripts/compare_in_digital_eye.py`: once with the original recogniser, once with the sample
model from [SAMPLE_EVALUATION_REPORT.md](SAMPLE_EVALUATION_REPORT.md). Options:
`{"vlm": {"enabled": false}, "strict_network_guard": true}`; only `rec_model_dir` differs.

Data: a **new** set made with Digital Eye's generator
(`benchmarks/make_synthetic_dataset.py --n 12 --seed 4242`): 45 documents (36 building permits —
12 Arabic, 12 English, 12 mixed — and 9 coordinate tables; clean / mild / heavy scan quality;
PNG, JPEG, image-only PDF). Its font, DejaVu Sans, was **not** used to train the model, and the
generator, layouts and page rendering differ from the trainer's.

| Metric | Original | Fine-tuned | Better is |
|---|---:|---:|---|
| False acceptance (wrong value returned as trustworthy) | 0.4% | 0.0% | lower |
| Field exact match | 84.0% | **98.7%** | higher |
| Critical fields exact match (permit, plot, area) | 71.3% | **97.2%** | higher |
| Required-field coverage | 87.2% | 99.5% | higher |
| Wrong values that were flagged | 97.2% | 100.0% | higher |
| Correct values still flagged for review | 14.3% | 14.9% | lower |
| Coordinate values within tolerance | 96.1% | 96.1% | higher |
| Seconds per document | 3.74 | 3.97 | lower |

| Language | Fields | Correct (orig → tuned) | False acceptances | Flagged for review |
|---|---:|---:|---:|---:|
| Arabic | 72 | 37 → **70** | 1 → 0 | 38 → 14 |
| Mixed | 72 | 72 → 72 | 0 → 0 | 2 → 0 |
| English | 72 | 71 → 71 | 0 → 0 | 13 → 13 |

Raw output: [sample_digital_eye_comparison.md](sample_digital_eye_comparison.md) /
[.json](sample_digital_eye_comparison.json).

What this shows and what it does not:

* The exported folder **drops into Digital Eye with a configuration change only** and works
  through its whole pipeline (detection, cross-check, extraction, review flags, offline guard).
* Arabic permits improved most: Arabic-Indic areas with `٫` and Latin numbers in Arabic lines
  are now read. Many Arabic fields are still flagged for review because Digital Eye
  deliberately sends every Arabic-Indic OCR number to review; that rule could be relaxed only
  after real-document measurements.
* English and coordinate results do not change: Digital Eye reads Latin-only lines with the
  English model, which was not fine-tuned.
* **Synthetic only**: 45 generated documents, one font. Real scans will be harder. Repeat this
  comparison with your own labelled benchmark set (Digital Eye manifest format) to know.

Reproduce:

```bash
<digital-eye python> <digital-eye>/benchmarks/make_synthetic_dataset.py /tmp/de_set --n 12 --seed 4242
python scripts/compare_in_digital_eye.py --digital-eye <digital-eye> --python <digital-eye python> \
    --dataset /tmp/de_set --exported workspace/exports/<name>
```

## 3. Suggested change in Digital Eye (not applied)

Fine-tuning cannot fix the reordering done by PaddleX after recognition
(see [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md)). If you want it fixed, the change belongs in
Digital Eye's PaddleOCR provider, for example: take PaddleX's text, and for lines that contain
Arabic letters re-apply the bidi algorithm with a right-to-left paragraph direction
(`get_display(visual_text, base_dir="R")`) on the raw visual-order output instead of the
automatic guess. This needs access to the raw output (or reversing PaddleX's step) and its own
tests; it was **not implemented or measured** here because this project must not modify
Digital Eye. The "raw model" view in every evaluation report shows what such a change could gain.
