"""Synthetic text-line generator (Arabic / English / mixed) with scanner-like degradation.

Every sample is generated here; no real document is involved. Rendering uses Pillow's
libraqm layout engine for correct Arabic shaping and bidi ordering. Labels are in
logical order. Lines are grouped into synthetic "documents" so the document-level
train/validation/test split applies to synthetic data as well.

Fonts: the .ttf/.otf files in the bundled ``fonts/`` folder plus any folder you pass
(e.g. ``workspace/fonts``). A font is used for a line only if it has a glyph for every
character of that line.
"""

from __future__ import annotations

import io
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, features

from . import textutil as tu

BUNDLED_FONTS = Path(__file__).resolve().parents[1] / "fonts"
AR_DIGITS = str.maketrans("0123456789", tu.ARABIC_INDIC)

FIRST = ["محمد", "أحمد", "خالد", "سامي", "ليلى", "رنا", "يوسف", "هالة", "عمر", "نور", "زينب", "علي", "مريم",
         "إبراهيم", "فاطمة", "سعيد", "هيثم", "ياسمين", "كريم", "عيسى", "يحيى", "منى", "رياض", "سلمى"]
LAST = ["الخطيب", "النوافلة", "العبادي", "الزعبي", "الحسن", "المصري", "الشريف", "القاسم", "الكيلاني",
        "البدور", "الحياري", "العموري", "الطراونة", "اليوسف", "المجالي", "الكردي"]
PLACES = ["الجبيهة الشمالي", "تلاع العلي", "أبو نصير", "خلدا", "الرابية", "شفا بدران", "ماركا", "الكرك",
          "إربد", "الزرقاء", "مادبا", "السلط", "جرش", "عجلون", "العقبة", "ناعور", "وادي السير"]
AR_LABELS = ["رقم الرخصة", "رقم القطعة", "رقم الحوض", "اسم المالك", "تاريخ الإصدار", "تاريخ الانتهاء",
             "المساحة الكلية", "مساحة البناء", "الحوض", "القرية", "اللوحة", "المنطقة", "الاستعمال",
             "نسبة الطابقية", "الارتداد الأمامي", "عدد الطوابق", "رقم المعاملة", "رقم الطلب", "الإحداثي الشرقي",
             "الإحداثي الشمالي", "المنسوب", "رقم النقطة", "المهندس المسؤول", "رقم الهاتف"]
EN_LABELS = ["Permit No", "Plot No", "Basin No", "Owner Name", "Issue Date", "Expiry Date", "Total Area",
             "Built-up Area", "Village", "Sheet", "Zone", "Land Use", "Floors", "Transaction No", "Easting",
             "Northing", "Elevation", "Point", "Engineer", "Phone"]
AR_WORDS = ["بناء", "سكني", "تجاري", "صناعي", "زراعي", "طابق", "أرضي", "أول", "ثاني", "سطح", "مخطط", "موقع",
            "تنظيم", "ترخيص", "مستودع", "مكاتب", "شقة", "سور", "حديقة", "مواقف", "وزارة", "الأشغال", "البلدية",
            "دائرة", "الأراضي", "والمساحة", "حسب", "المخطط", "المرفق", "ملاحظات", "يوجد", "لا", "مع", "على", "من"]
EN_WORDS = ["Building", "Permit", "Residential", "Commercial", "Ground", "Floor", "Site", "Plan", "Municipality",
            "Department", "Lands", "Survey", "Notes", "Approved", "Valid", "until", "per", "attached", "drawing"]
AR_UNITS = ["م2", "م²", "م", "دونم", "كم", "سم", "م3"]
EN_UNITS = ["m2", "m²", "m", "km", "sqm", "cm", "m3"]
ID_PREFIX = ["BP", "LIC", "P", "TR", "AMM", "ZR", "IRB", "PL"]


def raqm_available() -> bool:
    return bool(features.check("raqm"))


# ------------------------------------------------------------------ fonts ------
@dataclass
class Font:
    path: Path
    cmap: frozenset

    def covers(self, text: str) -> bool:
        return all(ord(c) in self.cmap or c.isspace() for c in text)


def load_fonts(extra_dirs: list[str | Path] | None = None) -> list[Font]:
    from fontTools.ttLib import TTFont

    paths: list[Path] = []
    for d in [BUNDLED_FONTS, *(extra_dirs or [])]:
        d = Path(d)
        if d.is_dir():
            paths += sorted(p for p in d.iterdir() if p.suffix.lower() in (".ttf", ".otf"))
    fonts = []
    for p in paths:
        try:
            fonts.append(Font(p, frozenset(TTFont(str(p), lazy=True).getBestCmap())))
        except Exception:  # unreadable font file: skip, the caller reports how many loaded
            continue
    return fonts


# ------------------------------------------------------------------ text -------
def _num(rng: random.Random, lo: float, hi: float, decimals: int) -> float:
    return round(rng.uniform(lo, hi), decimals)


def fmt_number(value: float, decimals: int, arabic_digits: bool, grouping: bool) -> str:
    s = f"{value:,.{decimals}f}" if grouping else f"{value:.{decimals}f}"
    if arabic_digits:
        s = s.replace(",", tu.AR_THOUSANDS).replace(".", tu.AR_DECIMAL).translate(AR_DIGITS)
    return s


def _date(rng: random.Random, arabic_digits: bool) -> str:
    y, m, d = rng.randint(1990, 2030), rng.randint(1, 12), rng.randint(1, 28)
    s = rng.choice([f"{d:02d}/{m:02d}/{y}", f"{y}-{m:02d}-{d:02d}", f"{d}/{m}/{y}", f"{y}/{m:02d}/{d:02d}"])
    return s.translate(AR_DIGITS) if arabic_digits else s


def _id(rng: random.Random) -> str:
    return rng.choice([
        f"{rng.choice(ID_PREFIX)}-{rng.randint(1990, 2030)}/{rng.randint(1, 9999):04d}",
        f"{rng.choice(ID_PREFIX)}{rng.randint(100, 99999)}",
        f"{rng.randint(1, 999)}/{rng.randint(1, 99)}",
        f"{rng.choice(ID_PREFIX)}-{rng.randint(10, 999)}-{rng.randint(1, 99)}",
    ])


def _name(rng: random.Random) -> str:
    return " ".join([rng.choice(FIRST), rng.choice(FIRST), rng.choice(LAST)][: rng.choice([2, 3, 3])])


def make_line(rng: random.Random) -> str:
    """One random line. Weighted towards the known weaknesses of the base model."""
    ar_dig = rng.random() < 0.5
    kind = rng.choices(
        ["ar_number_unit", "ar_label_ascii", "ar_label_id", "ar_date", "ar_name", "ar_text", "en_label",
         "en_text", "coords", "mixed_en_first", "ar_place", "number_only"],
        weights=[16, 14, 12, 8, 7, 6, 8, 4, 8, 7, 5, 5])[0]
    if kind == "ar_number_unit":
        v = fmt_number(_num(rng, 1, 99999, rng.choice([0, 1, 2, 2, 3])), 2 if rng.random() < .5 else rng.choice([0, 1, 3]),
                       ar_dig, rng.random() < 0.25)
        unit = rng.choice(AR_UNITS) if rng.random() < 0.8 else rng.choice(EN_UNITS)
        return f"{rng.choice(AR_LABELS)}: {v} {unit}" if rng.random() < 0.8 else f"{v} {unit}"
    if kind == "ar_label_ascii":
        v = rng.choice([str(rng.randint(1, 99999)), fmt_number(_num(rng, 1, 9999, 2), 2, False, rng.random() < .3)])
        sep = rng.choice([": ", ":", " "])
        return f"{rng.choice(AR_LABELS)}{sep}{v}"
    if kind == "ar_label_id":
        return f"{rng.choice(AR_LABELS)}: {_id(rng)}"
    if kind == "ar_date":
        return f"{rng.choice(['تاريخ الإصدار', 'تاريخ الانتهاء', 'التاريخ', 'تاريخ الطلب'])}: {_date(rng, ar_dig)}"
    if kind == "ar_name":
        return f"{rng.choice(['اسم المالك', 'المالك', 'المهندس المسؤول', 'اسم المستدعي'])}: {_name(rng)}"
    if kind == "ar_text":
        return " ".join(rng.choice(AR_WORDS) for _ in range(rng.randint(2, 7)))
    if kind == "ar_place":
        return f"{rng.choice(['الحوض', 'القرية', 'المنطقة', 'الموقع'])}: {rng.choice(PLACES)} {rng.choice(['', str(rng.randint(1, 40)), str(rng.randint(1, 40)).translate(AR_DIGITS)])}".strip()
    if kind == "en_label":
        v = rng.choice([_id(rng), str(rng.randint(1, 9999)), _date(rng, False),
                        f"{fmt_number(_num(rng, 50, 9999, 2), 2, False, rng.random() < .4)} {rng.choice(EN_UNITS)}"])
        return f"{rng.choice(EN_LABELS)}: {v}"
    if kind == "en_text":
        return " ".join(rng.choice(EN_WORDS) for _ in range(rng.randint(2, 6)))
    if kind == "coords":
        e, n = _num(rng, 200000, 260000, 2), _num(rng, 1100000, 1180000, 2)
        if ar_dig:
            return f"{rng.randint(1, 30)}".translate(AR_DIGITS) + f"  {fmt_number(e, 2, True, False)}  {fmt_number(n, 2, True, False)}"
        return rng.choice([f"{rng.randint(1, 30)}  {e:.2f}  {n:.2f}", f"E: {e:.2f}  N: {n:.2f}",
                           f"الإحداثي الشرقي: {e:.2f}"])
    if kind == "mixed_en_first":
        return rng.choice(["Building Permit / رخصة بناء", f"{rng.choice(EN_LABELS)}: {_name(rng)}",
                           f"{rng.choice(EN_LABELS)} / {rng.choice(AR_LABELS)}: {_id(rng)}"])
    v = fmt_number(_num(rng, 0, 99999, 2), rng.choice([0, 2]), ar_dig, rng.random() < .3)
    return v


# ------------------------------------------------------------------ rendering --
def render_line(text: str, font: Font, size: int, rng: random.Random) -> Image.Image:
    """Render one line in logical order with libraqm (shaping + bidi), black on paper."""
    if not raqm_available():
        raise RuntimeError("Pillow was built without libraqm: Arabic cannot be rendered correctly. "
                           "Install the official Pillow wheel (pip install --force-reinstall pillow).")
    f = ImageFont.truetype(str(font.path), size, layout_engine=ImageFont.Layout.RAQM)
    direction = "rtl" if tu.first_strong_rtl(text) else "ltr"
    left, top, right, bottom = f.getbbox(text, direction=direction)
    pad_x, pad_y = rng.randint(2, int(size * 0.6) + 2), rng.randint(2, int(size * 0.35) + 2)
    w, h = right - left + 2 * pad_x, bottom - top + 2 * pad_y
    paper = rng.randint(215, 255)
    img = Image.new("L", (max(w, 8), max(h, 8)), paper)
    ink = rng.randint(0, 70)
    ImageDraw.Draw(img).text((pad_x - left, pad_y - top), text, font=f, fill=ink, direction=direction)
    return img


def degrade(img: Image.Image, rng: random.Random, strength: float) -> Image.Image:
    """Scanner-like defects: skew, blur, low resolution, noise, JPEG. strength in [0, 1]."""
    if strength <= 0:
        return img.convert("RGB")
    paper = int(np.median(np.asarray(img)))
    angle = rng.uniform(-2.0, 2.0) * strength
    if abs(angle) > 0.05:
        img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=paper)
    if rng.random() < 0.6 * strength:
        f = rng.uniform(0.45, 0.85)
        small = img.resize((max(8, int(img.width * f)), max(8, int(img.height * f))), Image.BILINEAR)
        img = small.resize(img.size, Image.BILINEAR)
    blur = rng.uniform(0, 1.3) * strength
    if blur > 0.15:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    arr = np.asarray(img).astype(np.float32)
    nrng = np.random.default_rng(rng.randrange(2**32))
    arr += nrng.normal(0, rng.uniform(0, 20) * strength, arr.shape)
    if rng.random() < 0.3 * strength:  # uneven illumination
        grad = np.linspace(rng.uniform(-25, 0), rng.uniform(0, 25), arr.shape[1])[None, :]
        arr += grad
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGB")
    if rng.random() < 0.8 * strength:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=rng.randint(int(35 + 50 * (1 - strength)), 92))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return img


@dataclass
class Sample:
    image: Image.Image
    text: str
    font: str
    doc_id: str


def generate(n: int, seed: int, fonts: list[Font], strength: tuple[float, float] = (0.0, 1.0),
             lines_per_doc: int = 40, doc_prefix: str = "synth", size_range: tuple[int, int] = (22, 44)):
    """Yield n samples. Lines are grouped into synthetic documents of `lines_per_doc`."""
    if not fonts:
        raise RuntimeError(f"no fonts found (bundled folder: {BUNDLED_FONTS})")
    rng = random.Random(seed)
    made = 0
    while made < n:
        text = tu.normalize_label(make_line(rng))
        usable = [f for f in fonts if f.covers(text)]
        if not usable or not text:
            continue
        font = rng.choice(usable)
        img = render_line(text, font, rng.randint(*size_range), rng)
        img = degrade(img, rng, rng.uniform(*strength))
        yield Sample(img, text, font.path.name, f"{doc_prefix}-{seed}-{made // lines_per_doc:04d}")
        made += 1


def write_dataset(out_dir: str | Path, n: int, seed: int, fonts: list[Font], strength=(0.0, 1.0),
                  lines_per_doc: int = 40, doc_prefix: str = "synth") -> Path:
    """Write images + labels.jsonl ({"image", "text", "doc_id", "source", "font"})."""
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    with open(out / "labels.jsonl", "w", encoding="utf-8") as fh:
        for i, s in enumerate(generate(n, seed, fonts, strength, lines_per_doc, doc_prefix)):
            name = f"images/{seed}_{i:06d}.png"
            s.image.save(out / name)
            fh.write(json.dumps({"image": name, "text": s.text, "doc_id": s.doc_id, "source": "synthetic",
                                 "font": s.font}, ensure_ascii=False) + "\n")
    return out


# ------------------------------------------------------------------ pages ------
def render_page(lines: list[str], fonts: list[Font], rng: random.Random, width: int = 1654,
                strength: float = 0.3) -> Image.Image:
    """A synthetic document page (A4 at 200 dpi) with one text line per row, right-aligned
    for right-to-left lines. Used to test the document import pipeline."""
    page = Image.new("L", (width, int(width * 1.414)), 255)
    y = 120
    for text in lines:
        usable = [f for f in fonts if f.covers(text)] or fonts
        line = render_line(text, rng.choice(usable), rng.randint(30, 40), rng)
        if y + line.height > page.height - 100:
            break
        x = width - 140 - line.width if tu.first_strong_rtl(text) else 140
        page.paste(line, (max(0, x), y))
        y += line.height + rng.randint(25, 50)
    return degrade(page, rng, strength) if strength > 0 else page.convert("RGB")

