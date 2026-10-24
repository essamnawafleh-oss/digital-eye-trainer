# Licences

This repository's own code: **Apache-2.0** (see `LICENSE`).

## Bundled files

| Files | Licence | Source |
|---|---|---|
| `eye_trainer/vendor/ppocr_rec/*.py` (4 files) | Apache-2.0 | PaddleOCR v3.7.0, commit `b03f464`; licence and modification notice in that folder (`LICENSE`, `NOTICE`) |
| `fonts/NotoNaskhArabic.ttf`, `fonts/NotoSansArabic.ttf` | SIL Open Font License 1.1 (`fonts/OFL-Noto*.txt`) | google/fonts repository (`ofl/notonaskharabic`, `ofl/notosansarabic`) |
| `fonts/Amiri-Regular.ttf` | SIL OFL 1.1 (`fonts/OFL-Amiri.txt`) | google/fonts `ofl/amiri` |
| `fonts/Cairo.ttf` | SIL OFL 1.1 (`fonts/OFL-Cairo.txt`) | google/fonts `ofl/cairo` |
| `fonts/ScheherazadeNew-Regular.ttf` | SIL OFL 1.1 (`fonts/OFL-ScheherazadeNew.txt`) | google/fonts `ofl/scheherazadenew` |

The OFL allows bundling and redistribution with the licence text; the fonts are not sold on
their own and are not modified.

## Models (downloaded at install time, not stored in git)

| Model | Licence | Checked |
|---|---|---|
| `PaddlePaddle/PP-OCRv5_mobile_det` | Apache-2.0 | model card front matter `license: apache-2.0` (checked by `download-models`) |
| `PaddlePaddle/arabic_PP-OCRv5_mobile_rec` | Apache-2.0 | same |
| `PaddlePaddle/en_PP-OCRv5_mobile_rec` | Apache-2.0 | same |

A fine-tuned model is a derivative of the Apache-2.0 base model. Apache-2.0 allows that; keep
the licence notice if you ever distribute it (you should not distribute it: it may contain
traces of your documents, see `docs/PRIVACY.md`).

## Python dependencies (`requirements.txt`, from package metadata)

| Package | Version | Licence |
|---|---|---|
| paddlepaddle | 3.3.1 | Apache-2.0 |
| paddleocr | 3.7.0 | Apache-2.0 |
| paddlex | 3.7.2 | Apache-2.0 |
| opencv-contrib-python | 4.10.0.84 | Apache-2.0 |
| python-bidi | 0.6.12 | **LGPL-3.0** — used unmodified as a library (it is also what PaddleX itself uses); fine for this tool. If you ship a product, keep it replaceable (it is a separate package) |
| pillow | 13.0.0 | MIT-CMU (HPND) |
| numpy | 2.3.5 | BSD-3-Clause |
| pyyaml | 6.0.2 | MIT |
| pypdfium2 | 5.14.0 | Apache-2.0 / BSD-3-Clause (PDFium: BSD-3-Clause) |
| fonttools | 4.67.0 | MIT |
| huggingface_hub | 2.3.0 | Apache-2.0 |
| pytest (tests only) | 9.1.1 | MIT |

Transitive dependencies pulled in by paddlex (pandas, pydantic, requests, modelscope,
aistudio-sdk, …) are Apache-2.0, MIT, BSD, MPL-2.0 (certifi, tqdm) or PSF. **PyMuPDF (AGPL)
is not used.** The local web page uses no external libraries, fonts or CDNs.
