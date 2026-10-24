"""Run Digital Eye's own benchmark twice — original recogniser vs fine-tuned — and compare.

Digital Eye is NOT modified: the fine-tuned model is selected with the documented options
`paddle.rec_model_name` / `paddle.rec_model_dir` only.

    python scripts/compare_in_digital_eye.py \
        --digital-eye C:/code/digital-eye \
        --python C:/code/digital-eye/.venv/Scripts/python.exe \
        --dataset C:/data/benchmark_set          (a folder with Digital Eye's manifest.json)
        --exported workspace/exports/<name>
        [--models workspace/models] [--out workspace/reports/digital_eye_comparison]

To create a synthetic benchmark set with Digital Eye's generator:
    <digital-eye python> <digital-eye>/benchmarks/make_synthetic_dataset.py OUT --n 12 --seed 4242

Output: comparison.md / comparison.json with numbers only (shareable). Digital Eye's full
result files contain field values (document text) and are written to <out>/private/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MODELS = ("PP-OCRv5_mobile_det", "arabic_PP-OCRv5_mobile_rec", "en_PP-OCRv5_mobile_rec")
KEYS = [("false_acceptance_rate", "False acceptance (wrong value returned as trustworthy)", False),
        ("field_exact_match", "Field exact match", True),
        ("critical_field_exact_match", "Critical fields exact match (permit, plot, area)", True),
        ("field_coverage", "Required-field coverage", True),
        ("uncertainty_detection_rate", "Wrong values that were flagged", True),
        ("unnecessary_review_rate", "Correct values still flagged for review", False),
        ("coordinate_numeric_accuracy", "Coordinate values within tolerance", True),
        ("mean_cer", "Page text CER", False),
        ("seconds_mean", "Seconds per document", False)]


def by_tag(result: dict, manifest: list[dict]) -> dict:
    """Field accuracy per language tag (counts only)."""
    tags = {e["file"]: set(e.get("tags", [])) for e in manifest}
    out: dict[str, dict] = {}
    for doc in result["documents"]:
        for tag in ("ar", "en", "mixed"):
            if tag in tags.get(doc["file"], set()):
                c = out.setdefault(tag, {"fields": 0, "correct": 0, "false_acceptance": 0, "flagged": 0})
                for f in doc["fields"]:
                    c["fields"] += 1
                    c["correct"] += f["correct"]
                    c["false_acceptance"] += f["false_acceptance"]
                    c["flagged"] += f["flagged_for_review"]
    return out


def run(de: Path, py: str, dataset: Path, options: dict, out_json: Path) -> dict:
    opts = out_json.with_suffix(".options.json")
    opts.write_text(json.dumps(options), encoding="utf-8")
    r = subprocess.run([py, str(de / "benchmarks" / "evaluate.py"), str(dataset), "--options", str(opts),
                        "--out", str(out_json)], capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-4000:])
        raise SystemExit(f"Digital Eye benchmark failed (exit {r.returncode})")
    return json.loads(out_json.read_text(encoding="utf-8"))


def fmt(v) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.1f}%" if isinstance(v, float) and v <= 1.0 else str(v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--digital-eye", type=Path, required=True)
    ap.add_argument("--python", default=sys.executable, help="Python of Digital Eye's environment")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--exported", type=Path, required=True)
    ap.add_argument("--models", type=Path, default=Path("workspace/models"))
    ap.add_argument("--out", type=Path, default=Path("workspace/reports/digital_eye_comparison"))
    args = ap.parse_args(argv)

    private = args.out / "private"
    private.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="de_models_"))
    for m in MODELS:  # Digital Eye expects <models_dir>/paddle/<name>
        shutil.copytree(args.models / m, tmp / "paddle" / m, ignore=shutil.ignore_patterns(".cache"))
    base = {"models_dir": str(tmp), "vlm": {"enabled": False}, "strict_network_guard": True}
    tuned = {**base, "paddle": {"rec_model_name": "arabic_PP-OCRv5_mobile_rec",
                                "rec_model_dir": str(args.exported.resolve())}}
    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    print("running Digital Eye with the original model ...")
    before = run(args.digital_eye, args.python, args.dataset, base, private / "original.json")
    print("running Digital Eye with the fine-tuned model ...")
    after = run(args.digital_eye, args.python, args.dataset, tuned, private / "fine_tuned.json")
    shutil.rmtree(tmp, ignore_errors=True)

    comp = {"documents": before["summary"]["documents"],
            "summary": {k: {"original": before["summary"].get(k), "fine_tuned": after["summary"].get(k)} for k, _, _ in KEYS},
            "by_language": {"original": by_tag(before, manifest), "fine_tuned": by_tag(after, manifest)}}
    (args.out / "comparison.json").write_text(json.dumps(comp, indent=2), encoding="utf-8")
    lines = ["# Digital Eye: original vs fine-tuned recogniser", "",
             "Numbers only. Same documents, same Digital Eye code and options; only `paddle.rec_model_dir` differs.", "",
             f"Documents: {comp['documents']}", "",
             "| Metric | Original | Fine-tuned | Better is |", "|---|---:|---:|---|"]
    for k, label, higher in KEYS:
        s = comp["summary"][k]
        lines.append(f"| {label} | {fmt(s['original'])} | {fmt(s['fine_tuned'])} | {'higher' if higher else 'lower'} |")
    lines += ["", "| Language | Fields | Correct (orig → tuned) | False acceptances (orig → tuned) | Flagged (orig → tuned) |",
              "|---|---:|---:|---:|---:|"]
    for tag in ("ar", "mixed", "en"):
        b, a = comp["by_language"]["original"].get(tag), comp["by_language"]["fine_tuned"].get(tag)
        if b and a:
            lines.append(f"| {tag} | {b['fields']} | {b['correct']} → {a['correct']} | "
                         f"{b['false_acceptance']} → {a['false_acceptance']} | {b['flagged']} → {a['flagged']} |")
    (args.out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
