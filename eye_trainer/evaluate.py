"""Before/after evaluation.

Shareable reports contain only counts and rates: no document text, names, file names
or images. Per-line errors (which do contain text) go to a separate file under
``workspace/private/`` that is clearly marked as not to be shared.
"""

from __future__ import annotations

import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import paddle

from . import textutil as tu
from .data import Charset, Record, batch_tensor, load_bgr

CATEGORY_ORDER = ["all", "arabic", "latin", "mixed_script", "arabic_indic_digits", "ascii_digits",
                  "ascii_digits_in_arabic", "arabic_decimal", "ids"]
CATEGORY_AR = {
    "all": "كل الأسطر", "arabic": "عربي فقط", "latin": "لاتيني فقط", "mixed_script": "مختلط عربي/لاتيني",
    "arabic_indic_digits": "أرقام هندية (٠١٢)", "ascii_digits": "أرقام لاتينية (012)",
    "ascii_digits_in_arabic": "أرقام لاتينية داخل سطر عربي", "arabic_decimal": "فاصلة عشرية ٫", "ids": "أرقام/رموز معرّفة",
}


def predict(model, charset: Charset, records: list[Record], batch_size: int = 32) -> list[tuple[str, float]]:
    """Recognise each record's image. Same preprocessing and greedy decoding as paddlex."""
    model.eval()
    images = [load_bgr(r.image) for r in records]
    ratios = [im.shape[1] / float(im.shape[0]) for im in images]
    order = sorted(range(len(records)), key=lambda i: ratios[i])  # paddlex also sorts by aspect ratio
    out: list[tuple[str, float] | None] = [None] * len(records)
    with paddle.no_grad():
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            x = batch_tensor([images[i] for i in idx])
            probs = model(paddle.to_tensor(x)).numpy()
            for k, i in enumerate(idx):
                out[i] = charset.decode(probs[k])
    return out  # type: ignore[return-value]


@dataclass
class Acc:
    lines: int = 0
    exact: int = 0
    edits: int = 0
    chars: int = 0
    numbers: int = 0
    numbers_ok: int = 0
    ids: int = 0
    ids_ok: int = 0
    persian_letters: int = 0

    def as_dict(self) -> dict:
        def rate(a, b):
            return None if b == 0 else round(a / b, 4)
        return {
            "lines": self.lines,
            "cer": rate(self.edits, self.chars),
            "line_exact_match": rate(self.exact, self.lines),
            "numbers": self.numbers,
            "number_exact_match": rate(self.numbers_ok, self.numbers),
            "ids": self.ids,
            "id_exact_match": rate(self.ids_ok, self.ids),
            "persian_letter_outputs": self.persian_letters,
        }


def score(gts: list[str], preds: list[str]) -> dict:
    """Metrics per category. A number/ID counts as correct only if the identical token
    (all digits and separators) appears in the prediction."""
    acc: dict[str, Acc] = {c: Acc() for c in CATEGORY_ORDER}
    for gt, pred in zip(gts, preds, strict=True):
        g, p = tu.compare_for_metrics(gt), tu.compare_for_metrics(pred)
        d = tu.levenshtein(g, p)
        gn, pn = tu.numbers(g), tu.numbers(p)
        gi, pi = tu.ids(g), tu.ids(p)
        persian = sum(p.count(c) for c in tu.PERSIAN_TO_ARABIC)
        for c in tu.categories(g):
            a = acc[c]
            a.lines += 1
            a.exact += g == p
            a.edits += d
            a.chars += max(1, len(g))
            a.numbers += len(gn)
            a.numbers_ok += tu.token_hits(gn, pn)
            a.ids += len(gi)
            a.ids_ok += tu.token_hits(gi, pi)
            a.persian_letters += persian
    return {c: acc[c].as_dict() for c in CATEGORY_ORDER}


VIEWS = {
    "digital_eye": "What Digital Eye receives (after PaddleX's bidi step) vs. the typed label",
    "raw_model": "Raw model output vs. the label in visual order (pure recognition quality)",
}


def evaluate_records(model, charset: Charset, records: list[Record]) -> tuple[dict, list[str]]:
    """Metrics in both views, plus the Digital-Eye-view text of each line."""
    raw = [t for t, _ in predict(model, charset, records)]
    shown = [tu.digital_eye_view(t) for t in raw]
    metrics = {
        "digital_eye": score([r.text for r in records], shown),
        "raw_model": score([r.target for r in records], raw),
    }
    return metrics, shown


def _overall_problems(view: str, b: dict, a: dict) -> list[str]:
    out = []
    if a["cer"] is not None and b["cer"] is not None and a["cer"] > b["cer"]:
        out.append(f"[{view}] character error rate got worse: {b['cer']:.4f} -> {a['cer']:.4f}")
    if a["line_exact_match"] is not None and b["line_exact_match"] is not None and a["line_exact_match"] < b["line_exact_match"]:
        out.append(f"[{view}] line exact match got worse: {b['line_exact_match']:.4f} -> {a['line_exact_match']:.4f}")
    return out


def verdict(before: dict, after: dict, min_lines: int = 30) -> dict:
    """The new model must not be worse than the original on the test set.

    Worse = higher character error rate or lower line exact-match overall, in either
    view. Drops inside a category are listed as warnings."""
    problems, warnings = [], []
    for view in VIEWS:
        problems += _overall_problems(view, before[view]["all"], after[view]["all"])
    b_de, a_de = before["digital_eye"], after["digital_eye"]
    for cat in CATEGORY_ORDER[1:]:
        bc, ac = b_de[cat], a_de[cat]
        if ac["lines"] and bc["cer"] is not None and ac["cer"] is not None and ac["cer"] > bc["cer"]:
            warnings.append(f"{cat}: character error rate got worse ({bc['cer']:.4f} -> {ac['cer']:.4f}, {ac['lines']} lines)")
        for key in ("number_exact_match", "id_exact_match"):
            if bc[key] is not None and ac[key] is not None and ac[key] < bc[key]:
                warnings.append(f"{cat}: {key} got worse ({bc[key]:.4f} -> {ac[key]:.4f})")
    n = b_de["all"]["lines"]
    if n < min_lines:
        warnings.append(f"only {n} test lines: the comparison is not statistically reliable")
    return {"passed": not problems, "problems": problems, "warnings": warnings}


def exact_match_ci(acc: dict) -> str:
    """Rough 95% interval for a rate measured on n lines (normal approximation)."""
    p, n = acc["line_exact_match"], acc["lines"]
    if p is None or n == 0:
        return "n/a"
    half = 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / n)
    return f"±{half * 100:.1f} pts"


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "paddlepaddle": paddle.__version__,
        "device": paddle.device.get_device(),
    }


def build_report(before: dict, after: dict, info: dict) -> dict:
    return {
        "report": "digital-eye-trainer evaluation",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "privacy": "This file contains only counts and rates. No document text, names, file names or images.",
        "info": info,
        "environment": environment(),
        "views": VIEWS,
        "before": before,
        "after": after,
        "verdict": verdict(before, after),
    }


def _fmt(v):
    return "—" if v is None else f"{v * 100:.1f}%"


def _table(b: dict, a: dict) -> list[str]:
    rows = ["| Category / الفئة | Lines | CER before | CER after | Exact before | Exact after | Numbers before | Numbers after | IDs before | IDs after |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in CATEGORY_ORDER:
        if not b[c]["lines"]:
            continue
        rows.append(
            f"| {c} / {CATEGORY_AR[c]} | {b[c]['lines']} | {_fmt(b[c]['cer'])} | {_fmt(a[c]['cer'])} | "
            f"{_fmt(b[c]['line_exact_match'])} | {_fmt(a[c]['line_exact_match'])} | "
            f"{_fmt(b[c]['number_exact_match'])} ({b[c]['numbers']}) | {_fmt(a[c]['number_exact_match'])} | "
            f"{_fmt(b[c]['id_exact_match'])} ({b[c]['ids']}) | {_fmt(a[c]['id_exact_match'])} |")
    return rows


def report_markdown(rep: dict) -> str:
    b, a, v = rep["before"], rep["after"], rep["verdict"]
    lines = [
        "# Evaluation report / تقرير التقييم",
        "",
        f"Created: {rep['created']}  ",
        "Privacy: only counts and rates; no document text, names, file names or images.  ",
        "الخصوصية: أرقام ونِسَب فقط، بلا نصوص أو أسماء أو صور من الوثائق.",
        "",
        "## Result / النتيجة",
        "",
        ("**PASSED** — the new model is not worse than the original on the test set. / **نجح** — النموذج الجديد ليس أسوأ من الأصلي."
         if v["passed"] else
         "**FAILED** — the new model is WORSE than the original. Do not use it. / **فشل** — النموذج الجديد أسوأ من الأصلي، لا تستخدمه."),
        "",
    ]
    lines += [f"- ❌ {p}" for p in v["problems"]] + [f"- ⚠️ {w}" for w in v["warnings"]]
    lines += ["", "## Test set / مجموعة الاختبار", ""]
    lines += [f"- {k}: {val}" for k, val in rep["info"].items()]
    lines += ["", "CER = character error rate (lower is better / أقل أفضل). Exact = whole line correct / السطر صحيح بالكامل.",
              "Numbers / IDs: share of numbers / identifiers reproduced exactly, every digit and separator (count in brackets).", ""]
    for view, title in (("digital_eye", "What Digital Eye receives / ما يصل إلى العين الرقمية"),
                        ("raw_model", "Raw model output / مخرجات النموذج الخام")):
        lines += [f"## {title}", "", rep["views"][view] + ".", ""] + _table(b[view], a[view]) + [""]
    de_b, de_a = b["digital_eye"]["all"], a["digital_eye"]["all"]
    lines += [f"Persian letters (ی ک) in output: before {de_b['persian_letter_outputs']}, after {de_a['persian_letter_outputs']}.  ",
              f"Line exact-match uncertainty on {de_a['lines']} lines: {exact_match_ci(de_a)} (95%, rough).", "",
              "## Environment", ""]
    lines += [f"- {k}: {val}" for k, val in rep["environment"].items()]
    return "\n".join(lines) + "\n"


def write_private_errors(path: Path, records: list[Record], before: list[str], after: list[str]) -> None:
    """Per-line differences for YOUR eyes only (contains document text)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# PRIVATE - contains document text. Do NOT share this file. / خاص - يحتوي نصوص الوثائق، لا تشاركه\n")
        for r, b, a in zip(records, before, after):
            if tu.compare_for_metrics(r.text) != tu.compare_for_metrics(a) or b != a:
                fh.write(json.dumps({"image": str(r.image), "truth": r.text, "before": b, "after": a},
                                    ensure_ascii=False) + "\n")
