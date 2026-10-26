# Known limitations

## Not verified

* **No real document has been used.** Development, tests and all published numbers use
  synthetic lines and pages. Real scans (other fonts, stamps, handwriting, photos, faded
  paper) will behave differently. The first real numbers will come from your machine.
* **Windows**: verified on GitHub's Windows Server runner (full test suite + the packaged exe:
  check, short train/evaluate/export, page start). Not yet run on a desktop Windows 10/11 PC, and
  `install.bat` / `start.bat` themselves are not exercised by CI (the exe path is). Found there and
  fixed: Pillow's Windows wheels need `fribidi-0.dll` for Arabic shaping.
* **GPU training**: not run (no GPU available, Paddle's GPU wheel index blocked). The device
  is selected automatically; CPU fallback is the tested path.
* **macOS**: not tested.
* Python versions other than 3.13.

## What fine-tuning cannot fix

* **PaddleX's bidi post-processing.** For `arabic_PP-OCRv5_mobile_rec`, PaddleX 3.7.2 runs
  `bidi.algorithm.get_display()` on the model output with an *automatic* paragraph direction.
  * An Arabic line whose **leftmost** word is Latin (`رقم الرخصة: BP-2024/0153`) is returned
    with its runs swapped (`BP-2024/0153 :رقم الرخصة`), even when every character is right.
  * **Hyphenated dates inside Arabic lines are reordered**: `التاريخ: 2001-08-16` arrives as
    `التاريخ: 16-08-2001`. With an ambiguous day/month this can produce a plausible wrong date.
  * On generated lines, about 13% of lines cannot come back exactly through this step, whatever
    the model does. The "raw model" view in the reports shows recognition quality without it.
  * A fix belongs in Digital Eye (e.g. calling `get_display(raw, base_dir="R")` for lines whose
    visual text contains Arabic letters, or reading the raw output). This repository does not
    modify Digital Eye; the suggestion is in `docs/INTEGRATION.md`.
* Detection errors (lines merged or cut by the detector) are not addressed: only the
  recogniser is trained. A crop that the detector cut badly should be marked "unreadable".

## Training

* CTC-only fine-tuning: the NRTR auxiliary head used in upstream training is not available
  from the inference export and is not used. Upstream also trains from the non-fused
  (multi-branch) network; we train the fused network. Both are fine-tuning choices that the
  measurements so far support, but they differ from Baidu's own recipe.
* BatchNorm statistics are frozen (small batches); if your documents look very different from
  the base model's data, this may limit adaptation.
* On a 4-core CPU a step (batch 16) takes ≈2.5 s (measured: 1,200 steps in 50 minutes), so
  1,000 steps ≈ 42 minutes, longer if the computer is busy. Large runs need patience or a GPU.
* Catastrophic forgetting is controlled by mixing synthetic general lines (50/50 by default)
  and checked by the test-set comparison, but text types absent from both your labels and the
  generator (e.g. long English paragraphs) are not measured.
* Characters outside the 747-character dictionary cannot be learnt; such lines are skipped
  and reported by code point. Extending the dictionary would need a new output layer.
* Lines too long for their width in CTC terms (more characters than width/8 time steps) are
  skipped and counted.

## Evaluation

* A test set needs documents that are never used for training. With only a few labelled test
  documents the comparison is noisy (the report prints a rough ±interval and warns below 30
  lines).
* The synthetic test set comes from the same generator and fonts as the synthetic training
  data (different seed and documents): good for checking the pipeline and the targeted
  weaknesses, optimistic as an accuracy estimate. Short generated strings (e.g. a bare number)
  can occur in both training and test data by chance.
* "Numbers" and "IDs" are found with regular expressions in the ground truth; a number counts
  as correct only if the identical token appears in the output. Unusual formats may be missed.
* Per-category drops are warnings, not failures; only the overall comparison blocks export.

## Labelling

* The browser shows the label in the paragraph direction you choose; the comparison with the
  crop is visual. Mixed lines need attention (the preview helps).
* Multi-page TIFFs are imported page by page; HEIC and Office files are not supported.
* Only one labelling page per workspace at a time is supported.
* Re-importing an identical file is skipped (same SHA-256). A re-scan of the same paper is a
  different file: to keep it in the same split as the original, put all files of one document
  in the same **sub-folder** of the import folder (files in one sub-folder form a group that
  always shares a split). Files placed directly in the import folder are independent documents.

## Not implemented yet

* Fine-tuning the detection model and the English cross-check recogniser (planned; the English
  recogniser already loads with the same code).
