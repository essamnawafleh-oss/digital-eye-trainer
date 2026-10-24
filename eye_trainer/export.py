"""Export a fine-tuned model in Digital Eye's format (PaddleX inference folder).

Output folder::

    <name>/
      inference.json, inference.pdiparams, inference.yml   <- the model (drop-in)
      manifest.json                                       <- SHA-256 of every file + provenance
      DIGITAL_EYE_SETTINGS.txt                            <- what to paste into Digital Eye
      CONFIDENTIAL.txt                                    <- why this folder must stay private
      evaluation_report.md / .json                        <- copy of the passing report (numbers only)

The original model is never modified; Digital Eye keeps it as the fallback.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

from .messages import say
from .model import export_inference, model_name_of

CONFIDENTIAL = """CONFIDENTIAL MODEL / نموذج سرّي

AR: هذا النموذج تدرّب على وثائقك الخاصة. نماذج التعرّف على النص قد "تحفظ" نصوصاً تتكرر كثيراً في
بيانات التدريب (مثل الأسماء وأرقام القطع والرخص)، ويمكن أحياناً استخراج شيء منها من النموذج نفسه.
عامِل هذا المجلد كما تعامل الوثائق الأصلية: لا ترفعه إلى الإنترنت، ولا ترسله لأحد، ولا تشاركه مع أي
خدمة سحابية، واحفظ نسخه في مكان آمن.

EN: This model was fine-tuned on your private documents. Text recognisers can memorise strings
that appear often in their training data (names, plot and permit numbers), and some of that can
sometimes be recovered from the weights. Treat this folder like the documents themselves: do not
upload, e-mail or publish it, and do not share it with cloud services.
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def settings_text(model_dir: Path, base_name: str, base_dir_hint: str) -> str:
    return f"""Digital Eye settings for the fine-tuned recogniser / إعدادات العين الرقمية للنموذج المدرَّب
==========================================================================================

AR: انسخ أحد الخيارين التاليين في إعدادات العين الرقمية. لا يلزم أي تعديل آخر في الكود.
    للرجوع إلى النموذج الأصلي احذف السطر rec_model_dir أو أعده إلى المجلد الأصلي.
EN: Use ONE of the two options below. No code change is needed. To go back to the original
    model, remove rec_model_dir (or point it back to the original folder).

1) Python options passed to parse_document(...):

    options = {{
        "paddle": {{
            "rec_model_name": "{base_name}",
            "rec_model_dir": r"{model_dir}",
        }}
    }}

2) As a JSON options file (e.g. for benchmarks/evaluate.py --options opts.json):

    {json.dumps({"paddle": {"rec_model_name": base_name, "rec_model_dir": str(model_dir)}}, ensure_ascii=False)}

Keep the original model folder ({base_dir_hint}) as the fallback.
rec_model_name must stay "{base_name}": PaddleX uses this name to apply the Arabic
bidi post-processing, and the fine-tuned model was trained for it.

Verify the files before use:  python -m eye_trainer verify-export "{model_dir}"
"""


def export_run(model, base_model_dir: str | Path, out_dir: str | Path, run_info: dict,
               report_json: Path | None, force_unevaluated: bool = False) -> Path:
    """Export after checking that an evaluation passed (or force_unevaluated)."""
    passed = False
    if report_json is not None and Path(report_json).is_file():
        passed = bool(json.loads(Path(report_json).read_text(encoding="utf-8"))["verdict"]["passed"])
    if not passed and not force_unevaluated:
        raise RuntimeError(say(
            "لا يمكن التصدير: لا يوجد تقييم ناجح لهذا النموذج على مجموعة الاختبار (أو أنه أسوأ من الأصلي).",
            "Refusing to export: no passing test-set evaluation for this model (or it is worse than the original)."))
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(say(f"المجلد {out} موجود وليس فارغاً.", f"{out} already exists and is not empty."))
    export_inference(model, base_model_dir, out)
    base_name = model_name_of(base_model_dir)
    (out / "CONFIDENTIAL.txt").write_text(CONFIDENTIAL, encoding="utf-8")
    (out / "DIGITAL_EYE_SETTINGS.txt").write_text(settings_text(out.resolve(), base_name, str(base_model_dir)),
                                                  encoding="utf-8")
    if report_json is not None and Path(report_json).is_file():
        shutil.copy2(report_json, out / "evaluation_report.json")
        md = Path(report_json).with_suffix(".md")
        if md.is_file():
            shutil.copy2(md, out / "evaluation_report.md")
    files = {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()}
    manifest = {
        "model_name": base_name,
        "format": "PaddleX inference (inference.json + inference.pdiparams + inference.yml)",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_model": {"dir_name": Path(base_model_dir).name,
                       "inference.pdiparams_sha256": sha256(Path(base_model_dir) / "inference.pdiparams")},
        "evaluation_passed": passed,
        "exported_without_passing_evaluation": not passed,
        "run": run_info,
        "files_sha256": files,
        "confidential": True,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def verify_files(model_dir: str | Path) -> list[str]:
    """Compare files with manifest.json. Returns a list of problems (empty = OK)."""
    d = Path(model_dir)
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    problems = []
    for name, digest in man["files_sha256"].items():
        p = d / name
        if not p.is_file():
            problems.append(f"missing: {name}")
        elif sha256(p) != digest:
            problems.append(f"checksum mismatch: {name}")
    return problems


def check_loads_in_paddleocr(model_dir: str | Path, images: list, expected: list[str] | None = None,
                             device: str = "cpu") -> list[str]:
    """Load the folder with paddleocr.TextRecognition (the API Digital Eye uses) and read images.

    Returns the recognised texts. If `expected` is given, raise when they differ."""
    from paddleocr import TextRecognition

    rec = TextRecognition(model_name=model_name_of(model_dir), model_dir=str(model_dir), device=device,
                          enable_mkldnn=False)
    texts = [str(o["rec_text"]) for o in rec.predict([str(i) for i in images], batch_size=1)]
    if expected is not None and texts != expected:
        bad = sum(a != b for a, b in zip(texts, expected))
        raise RuntimeError(f"exported model gives different text than the trained model on {bad} of {len(texts)} lines")
    return texts
