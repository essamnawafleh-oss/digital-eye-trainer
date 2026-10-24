# Digital Eye Trainer

A local, **offline** tool that fine-tunes the Arabic text recogniser used by
[Digital Eye](https://github.com/essamnawafleh-oss/digital-eye)
(`arabic_PP-OCRv5_mobile_rec`, PaddleOCR 3.7.0) on your own documents, and exports a model
that drops into Digital Eye with a configuration change only.

> العربية: [README.ar.md](README.ar.md) — شرح مبسّط لغير المبرمجين.

```
your PDFs/images ──► import (detect + crop + suggest) ──► label in the browser ──┐
synthetic lines (numbers, units, dates, IDs, mixed) ─────────────────────────────┤
                                                                                  ▼
             export (Digital Eye format + checksums) ◄── evaluate (before/after, per document split) ◄── train (CPU/GPU, resumable)
```

## Privacy (non-negotiable)

* Documents never leave the computer. Import, labelling, training, evaluation and export run
  inside a strict **offline guard** (`eye_trainer/offline.py`): any non-loopback connection or
  DNS lookup raises, proxy variables are removed and local proxy ports are blocked.
  The internet is used only by `install` and `download-models`, which never touch documents.
* Proven by tests: the in-process guard (`tests/test_offline.py`) and a full pipeline run
  inside a Linux network namespace with **no network interfaces** (`unshare -rn`).
* The labelling page is served on `127.0.0.1` only, refuses foreign `Host` headers and needs
  a per-session token; it loads no external scripts, fonts or styles.
* No telemetry, no cloud services, no external APIs or hosted models.
* Shareable reports contain only counts and rates (tested: no document text in reports).
  Per-line errors go to `workspace/private/` — never share that folder.
* **A fine-tuned model can memorise frequent strings (names, plot numbers). Treat it as
  confidential.** See [docs/PRIVACY.md](docs/PRIVACY.md).

## Install (once, needs internet)

| | CPU (everywhere) | NVIDIA GPU (unverified) |
|---|---|---|
| Windows | double-click `install.bat` | `install.bat gpu` |
| Linux | `./install.sh` | `./install.sh gpu` |

Requires Python 3.10–3.13 (tested: 3.13). Creates `.venv/` (separate from Digital Eye),
installs pinned versions (`requirements.txt`), downloads the three base models from Hugging
Face (Apache-2.0, ~21 MB) into `workspace/models/`, and prints a machine check.
If the GPU build fails, use the CPU install: training is slower but identical.

## Use

`start.bat` / `./start.sh` opens `http://127.0.0.1:8765` with four tabs:
**Import → Label → Train → Evaluate & export.** The same steps exist on the command line:

```bash
python -m eye_trainer check                    # Python, OS, GPU, RAM, disk, models
python -m eye_trainer import /path/to/scans    # PDFs/images -> line crops + suggestions
python -m eye_trainer ui                       # labelling page
python -m eye_trainer synth --train 6000       # synthetic lines (train/val/test)
python -m eye_trainer train --run run1 --steps 3000      # resumes automatically
python -m eye_trainer evaluate --run run1      # exit code 3 if the new model is worse
python -m eye_trainer export --run run1        # refused unless the evaluation passed
python -m eye_trainer verify-export workspace/exports/<name>
python -m eye_trainer demo --steps 300         # synthetic-only end-to-end check
```

## How it works (short)

* **Weights.** Official *training* checkpoints are on Baidu servers (blocked where this was
  built). Instead the trainer rebuilds the recogniser in its fused (re-parameterised) form and
  loads Digital Eye's own inference weights into it, then checks the output is identical to
  the official inference program before training. See [docs/FEASIBILITY_REPORT.md](docs/FEASIBILITY_REPORT.md).
* **Labels** are typed in normal (logical) order. The model outputs visual order and PaddleX
  converts it back with `python-bidi`; training targets are converted the same way.
* **Data.** Import uses Digital Eye's detector, settings and cropping, and pre-fills each line
  with exactly what Digital Eye reads today (English model as second opinion). Lines that are
  uncertain or contain digits are shown first.
* **Split by document.** Every document is assigned to train, validation or test once, and
  stays there; the second document imported becomes the first test document. Files in the same
  sub-folder of the import folder (pages or re-scans of one document) always share a split. Evaluation
  refuses to run if a test document was used in training.
* **Training.** CTC loss, Adam, warm-up + cosine schedule, frozen BatchNorm statistics,
  light augmentation, real and synthetic lines mixed 50/50 by default. Checkpoints every 50
  steps; the best checkpoint on the validation set (measured as Digital Eye sees it) is kept.
  CPU or NVIDIA GPU (automatic).
* **Evaluation.** Before vs after on the test set: character error rate, line exact match,
  exact match of numbers (Arabic-Indic and ASCII, every digit and separator), IDs, mixed lines,
  Persian-letter outputs; in two views (what Digital Eye receives / raw model). Fails loudly if
  the new model is worse overall in either view.
* **Export.** `inference.json/.pdiparams/.yml` + `manifest.json` (SHA-256) +
  `DIGITAL_EYE_SETTINGS.txt` + `CONFIDENTIAL.txt` + the report. The exported folder is loaded
  with `paddleocr.TextRecognition` and must give the same text as the trained model.
  Integration: [docs/INTEGRATION.md](docs/INTEGRATION.md).

## Results so far (synthetic data only)

A 50-minute CPU run on 6,000 synthetic lines, then Digital Eye's own benchmark (unchanged
code, 45 freshly generated documents in a font not used for training), original → fine-tuned:
field exact match 84.0% → 98.7%, critical fields 71.3% → 97.2%, Arabic permit fields correct
37 → 70 of 72, false acceptances 1 → 0. Mixed-line run order is still limited by PaddleX's bidi
step. Details: [docs/SAMPLE_EVALUATION_REPORT.md](docs/SAMPLE_EVALUATION_REPORT.md),
[docs/INTEGRATION.md](docs/INTEGRATION.md). **No real document has been used.** Numbers on
synthetic lines show that the pipeline works and can fix the targeted weaknesses; they do not
predict accuracy on your scans.

## Tests

`run_tests.sh` / `run_tests.bat` (≈3 min on 4 CPU cores; 46 tests, all passing on Linux). Synthetic data only. Tests that need
the base models are skipped if they are not downloaded.

## Documents

* [docs/FEASIBILITY_REPORT.md](docs/FEASIBILITY_REPORT.md) — what was verified, how fine-tuning works here
* [docs/SAMPLE_EVALUATION_REPORT.md](docs/SAMPLE_EVALUATION_REPORT.md) — sample report from synthetic data
* [docs/INTEGRATION.md](docs/INTEGRATION.md) — using the model in Digital Eye, measured there
* [docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md)
* [docs/PRIVACY.md](docs/PRIVACY.md)
* [docs/LICENSES.md](docs/LICENSES.md)

## Not done yet (planned)

The same pipeline for the detection model (`PP-OCRv5_mobile_det`) and for the English
cross-check recogniser. The English recogniser has the same architecture and already loads
(tested), but training/evaluation for it is not wired up.
