# Feasibility report — fine-tuning `arabic_PP-OCRv5_mobile_rec` locally

Date: 2026-10-24. Every statement is marked **VERIFIED** (run here with a minimal working
example), **SOURCE** (read in the installed source code, not executed) or **UNVERIFIED**.

> **ملخص بالعربية:** التدريب الإضافي (fine-tuning) للنموذج العربي ممكن على جهازك وبدون إنترنت.
> لم نحتج إلى ملفات التدريب الرسمية المحجوبة؛ استطعنا تحويل ملفات النموذج نفسها التي تستخدمها العين
> الرقمية إلى نموذج قابل للتدريب، وتحققنا أن النتيجة مطابقة تماماً قبل أي تدريب. النموذج بعد التدريب
> يُصدَّر بنفس الصيغة، ويعمل في العين الرقمية بتغيير الإعدادات فقط. اكتشفنا أيضاً أن جزءاً من مشكلة ترتيب
> الكلمات في الأسطر المختلطة سببه خطوة معالجة داخل PaddleX وليس النموذج، وهذا لا يُصلَح بالتدريب.

## 1. Machine used for development and tests

| Item | Value |
|---|---|
| OS | Linux x86_64 (container) |
| Python | 3.13.17 |
| CPU / RAM / disk | 4 cores, 15.7 GB RAM, ~30 GB free |
| GPU | **none** (no `nvidia-smi`, Paddle CPU build) |
| Network | PyPI, GitHub, Hugging Face reachable. **Blocked:** `*.bcebos.com` (Baidu, hosts the official *training* checkpoints), `paddlepaddle.org.cn` (GPU wheels), ModelScope, AIStudio |

Windows and GPU training could **not** be tested (UNVERIFIED). Windows wheels for every
pinned package exist for Python 3.13 (checked with `pip download --platform win_amd64`).

## 2. How recognition fine-tuning works in this version

| Question | Finding | Status |
|---|---|---|
| Does `pip install paddleocr==3.7.0` include training code? | No. The pip package is inference-only. Training lives in the PaddleOCR GitHub repo (`tools/train.py`, `ppocr/`), tag `v3.7.0` (commit `b03f464`). | VERIFIED |
| Training config | `configs/rec/PP-OCRv5/multi_language/arabic_PP-OCRv5_mobile_rec.yaml`: PP-LCNetV3 (scale 0.95) + SVTR neck + CTC head, plus an NRTR head used only during training; dictionary `ppocrv5_arabic_dict.txt` (747 characters). `en_PP-OCRv5_mobile_rec` has the identical architecture. | VERIFIED |
| Weights needed to fine-tune | Upstream expects `arabic_PP-OCRv5_mobile_rec_pretrained.pdparams` from `paddle-model-ecology.bj.bcebos.com`. **Not reachable here**, and the Hugging Face repo `PaddlePaddle/arabic_PP-OCRv5_mobile_rec` only has the inference export (`inference.json`, `inference.pdiparams`, `inference.yml`). | VERIFIED |
| Can the inference weights be trained? | **Yes.** The export is *re-parameterised*: PP-LCNetV3's multi-branch blocks are fused into single convolutions, and the NRTR head is dropped (234 tensors instead of 884). We build the model in the same fused form and load the 234 tensors. Their order in the file is not named; it is recovered by exporting a copy of our model in which every tensor holds a unique constant. Result: **bit-identical output** to the official inference program (max abs diff 0.0) on random input, and **identical text to `paddleocr.TextRecognition` on 60/60 synthetic lines**. | VERIFIED |
| Training without the NRTR head | We fine-tune with CTC loss only (the NRTR head was only an auxiliary training signal and is not part of the deployed model). | VERIFIED (trains, see §4) |
| Pooling difference | In training mode the backbone pools to a fixed 40 time steps; at inference to width/8. Identical for a 48×320 input; we always train with the inference pooling so long lines keep their resolution. | SOURCE + VERIFIED |
| Dictionary coverage | `٫` (U+066B), `٬`, Arabic-Indic digits, ASCII digits, `²`, Arabic `ي ك` **and** Persian `ی ک` are all in the dictionary. No head resizing is needed; Persian outputs can be trained away. | VERIFIED |
| Export to Digital Eye's format | `paddle.jit.save` of our model + a copy of the original `inference.yml` gives a folder that `paddleocr.TextRecognition(model_name="arabic_PP-OCRv5_mobile_rec", model_dir=...)` loads and that gives the same text as the trained model. | VERIFIED |
| oneDNN | Kept off (`enable_mkldnn=False`) everywhere, as in Digital Eye. Training does not use oneDNN-specific code. | VERIFIED (CPU) |

## 3. A finding that changes the training labels: PaddleX's bidi step

`paddlex/inference/models/text_recognition/predictor.py` (3.7.2) runs
`bidi.algorithm.get_display(text)` on the output of `arabic_PP-OCRv5_mobile_rec`
(SOURCE). The **raw** model therefore outputs characters in **visual** order, left to right
as they appear in the image (VERIFIED: raw `ةبقعلا عقوملا`, PaddleOCR shows `الموقع العقبة`).

Consequences:

1. Training targets must be in visual order. Labels are typed and stored in normal (logical)
   order; the trainer converts them with the same `python-bidi` library (bracket mirroring
   included). Training on logical-order labels would fight what the model already knows.
2. **Part of Digital Eye's weakness #3 is not a model error.** `get_display()` guesses the
   paragraph direction from the first strong character of the *visual* string. For an
   Arabic line whose leftmost word is Latin (`رقم الرخصة: BP-2024/0153`), it guesses
   left-to-right and the runs come out swapped (`BP-2024/0153 :رقم الرخصة`) even when every
   character is right. On 3,000 generated lines, 13% could not survive this round trip
   **even with perfect recognition**. Hyphenated dates in Arabic lines are a dangerous case:
   `التاريخ: 2001-08-16` comes back as `التاريخ: 16-08-2001` (VERIFIED, see
   `tests/test_textutil.py`). Fine-tuning cannot fix this; Digital Eye would have to change
   that step (not done here, by request — see KNOWN_LIMITATIONS.md).
3. Every report therefore shows two views: **what Digital Eye receives** (after the bidi
   step, compared with the typed label) and the **raw model** (visual order, pure recognition).

## 4. Minimal working run (vertical slice) — VERIFIED

Synthetic lines only (no real documents involved):

1. Load base inference weights → trainable model (≈5 s, verified identical outputs).
2. Fine-tune with CTC loss, Adam, warm-up + cosine schedule, BatchNorm statistics frozen,
   batch 16: **≈4–5 s per step on 4 CPU cores** (≈3–4 steps/minute alone on the machine).
   Resume after interruption verified (stop after 2 steps, restart, finish at 4).
3. Export → folder with `inference.json/.pdiparams/.yml`.
4. Load the exported folder with `paddleocr.TextRecognition` → identical text to the trained model.
5. Measure before/after on a held-out synthetic test set (different documents and seed).
6. Load the exported model in **Digital Eye** (unchanged code, `paddle.rec_model_dir`
   option only) and run its own benchmark — see `docs/INTEGRATION.md`.
7. The whole pipeline (2 training steps) also ran inside a Linux network namespace with no
   network interfaces (`tests/test_offline.py`).

Measured results are in `docs/SAMPLE_EVALUATION_REPORT.md` and `docs/INTEGRATION.md`.

## 5. Data preparation, detection, PDFs

| Item | Choice | Status |
|---|---|---|
| Detection | `PP-OCRv5_mobile_det` through `paddleocr.TextDetection`, `limit_type="max"`, `limit_side_len=960`, oneDNN off — Digital Eye's settings | VERIFIED |
| Cropping | `paddlex...CropByPolys(det_box_type="quad")`, the function PaddleOCR's own pipeline uses | VERIFIED |
| Suggestions | Arabic model through `TextRecognition` (exactly Digital Eye's text) + English model as second opinion; Latin-only lines use the English reading first (Digital Eye's rule) | VERIFIED |
| PDF rendering | `pypdfium2` (Apache-2.0/BSD-3) at 200 dpi, Digital Eye's default DPI. PyMuPDF was avoided because of its AGPL licence | VERIFIED |
| Synthetic rendering | Pillow 13 with libraqm 0.11 (HarfBuzz + FriBiDi) for shaping and bidi | VERIFIED on Linux; Pillow's Windows wheels are documented to include raqm (UNVERIFIED here) |

## 6. GPU

`train` picks the GPU automatically when `paddle.device.is_compiled_with_cuda()` and at
least one CUDA device exist, otherwise the CPU. The GPU build must be installed separately
(`install.bat gpu` / `./install.sh gpu`: `paddlepaddle-gpu==3.3.1` from Paddle's CUDA 12.6
index). **UNVERIFIED:** no GPU here and Paddle's wheel index was blocked. You need an
NVIDIA driver that supports CUDA 12.6 (check the minimum driver version on Paddle's
installation page for 3.3.1 before installing). If anything fails, uninstall `paddlepaddle-gpu`, reinstall `paddlepaddle==3.3.1` and train on
the CPU — slower, same results.

## 7. Decisions

* Separate repository and environment; Digital Eye is not modified.
* Model code vendored from PaddleOCR v3.7.0 (4 files, Apache-2.0, `eye_trainer/vendor/ppocr_rec/NOTICE`)
  instead of cloning the 250 MB repo at install time (no git needed, works offline, Windows-friendly).
* Labels stored in logical order; the visual conversion happens only for training.
* Model selection on the validation set uses what Digital Eye receives.
* Export is refused unless the test-set evaluation passed.
