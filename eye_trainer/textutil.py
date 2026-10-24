"""Text normalisation, bidi conversion, line categories and error metrics.

Two orders matter (verified with paddlex 3.7.2, see docs/FEASIBILITY_REPORT.md):

* **logical** order: the order you type text. Labels are stored this way.
* **visual** order: characters as they appear in the image from left to right.
  arabic_PP-OCRv5_mobile_rec is trained to output this order. PaddleX then calls
  ``bidi.algorithm.get_display(raw_text)`` (predictor.py) to turn it back into
  logical order before Digital Eye sees it.

So training targets are ``to_visual(label)``, and "what Digital Eye sees" is
``digital_eye_view(raw_output)``. The paragraph direction of a label is right-to-left
when its first strong character is Arabic, unless the label says otherwise.
"""

from __future__ import annotations

import re
import unicodedata

from bidi.algorithm import get_display

ARABIC_INDIC = "٠١٢٣٤٥٦٧٨٩"
EXT_ARABIC_INDIC = "۰۱۲۳۴۵۶۷۸۹"
AR_DECIMAL = "٫"  # ٫
AR_THOUSANDS = "٬"  # ٬
PERSIAN_TO_ARABIC = {"ی": "ي", "ک": "ك", "ۀ": "ة"}  # ی ک ۀ -> ي ك ة

_AR_LETTER = re.compile(r"[ء-غف-يٱ-ۓۺ-ۼ]")
_LATIN = re.compile(r"[A-Za-z]")
_ASCII_DIGIT = re.compile(r"[0-9]")
_AR_DIGIT = re.compile(r"[٠-٩۰-۹]")
# A number: digits of one system, with optional grouping/decimal separators.
NUMBER_RE = re.compile(
    r"[0-9]+(?:[.,][0-9]+)*|[٠-٩]+(?:[٫٬.,][٠-٩]+)*|[۰-۹]+(?:[٫٬.,][۰-۹]+)*")
# An identifier: letters and digits joined by - or /, containing at least one digit and one letter,
# or digit groups joined by - or / (plot/permit numbers like 2024/0153).
ID_RE = re.compile(r"(?=[\w\-/]*\d)(?=[\w\-/]*[A-Za-z])[A-Za-z0-9]+(?:[\-/][A-Za-z0-9]+)+|\d+(?:[\-/]\d+)+|"
                   r"[٠-٩]+(?:[\-/][٠-٩]+)+")


def normalize_label(text: str, persian_to_arabic: bool = True) -> str:
    """NFC, unify whitespace, optionally map Persian yeh/keheh to Arabic letters."""
    text = unicodedata.normalize("NFC", text)
    if persian_to_arabic:
        text = text.translate(str.maketrans(PERSIAN_TO_ARABIC))
    text = text.replace("‏", "").replace("‎", "").replace("؜", "")  # bidi marks
    return " ".join(text.split())


def has_arabic(text: str) -> bool:
    return bool(_AR_LETTER.search(text))


def has_latin(text: str) -> bool:
    return bool(_LATIN.search(text))


def first_strong_rtl(text: str) -> bool:
    for ch in text:
        bidi = unicodedata.bidirectional(ch)
        if bidi in ("R", "AL"):
            return True
        if bidi == "L":
            return False
    return False


def to_visual(text: str, direction: str = "auto") -> str:
    """Logical -> visual (left-to-right image) order, with bracket mirroring."""
    if direction == "auto":
        direction = "rtl" if first_strong_rtl(text) else "ltr"
    return get_display(text, base_dir="R" if direction == "rtl" else "L")


def digital_eye_view(raw: str) -> str:
    """Exactly what paddlex returns for the Arabic model: get_display() with automatic direction."""
    return get_display(raw)


def categories(text: str) -> set[str]:
    """Categories a ground-truth line belongs to (used for per-category metrics)."""
    cats = {"all"}
    ar, lat = has_arabic(text), has_latin(text)
    if ar and lat:
        cats.add("mixed_script")
    elif ar:
        cats.add("arabic")
    elif lat:
        cats.add("latin")
    if _AR_DIGIT.search(text):
        cats.add("arabic_indic_digits")
    if _ASCII_DIGIT.search(text):
        cats.add("ascii_digits")
        if ar:
            cats.add("ascii_digits_in_arabic")
    if AR_DECIMAL in text:
        cats.add("arabic_decimal")
    if ID_RE.search(text):
        cats.add("ids")
    return cats


def numbers(text: str) -> list[str]:
    return NUMBER_RE.findall(text)


def ids(text: str) -> list[str]:
    return ID_RE.findall(text)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def compare_for_metrics(text: str) -> str:
    """What is compared: NFC and collapsed whitespace only. Persian letters are NOT
    mapped here, so an output with U+06CC counts as an error (Digital Eye needs Arabic)."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def token_hits(gt_tokens: list[str], pred_tokens: list[str]) -> int:
    """How many ground-truth tokens appear in the prediction (multiset match)."""
    pool = list(pred_tokens)
    hits = 0
    for t in gt_tokens:
        if t in pool:
            pool.remove(t)
            hits += 1
    return hits
