"""Line datasets, preprocessing (identical to PaddleX inference) and batching.

A dataset is a folder with ``labels.jsonl``; each row is
``{"image": "<path relative to the folder or absolute>", "text": "...", "doc_id": "...", "source": "..."}``.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import synth
from . import textutil as tu
from .model import INPUT_HEIGHT, MIN_INPUT_WIDTH

MAX_INPUT_WIDTH = 3200  # paddlex OCRReisizeNormImg.max_imgW


@dataclass
class Record:
    image: Path
    text: str
    doc_id: str
    source: str = "real"
    direction: str = "auto"  # paragraph direction of the label: auto | rtl | ltr

    @property
    def target(self) -> str:
        """Training target in the model's (visual) order."""
        return tu.to_visual(self.text, self.direction)


def read_dataset(folder: str | Path) -> list[Record]:
    folder = Path(folder)
    out = []
    with open(folder / "labels.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                p = Path(r["image"])
                out.append(Record(p if p.is_absolute() else folder / p, r["text"], r["doc_id"],
                                  r.get("source", "real"), r.get("direction", "auto")))
    return out


def write_dataset(folder: str | Path, records: list[Record]) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "labels.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps({"image": str(r.image), "text": r.text, "doc_id": r.doc_id, "source": r.source,
                                "direction": r.direction}, ensure_ascii=False) + "\n")
    return folder


# ------------------------------------------------------------- encoding --------
class Charset:
    def __init__(self, classes: list[str]):
        self.classes = classes
        self.index = {c: i for i, c in enumerate(classes) if i > 0}

    def encode(self, text: str) -> list[int] | None:
        try:
            return [self.index[c] for c in text]
        except KeyError:
            return None

    def unsupported(self, text: str) -> set[str]:
        return {c for c in text if c not in self.index}

    def decode(self, probs: np.ndarray) -> tuple[str, float]:
        """Greedy CTC decode, as paddlex CTCLabelDecode: argmax, merge repeats, drop blank."""
        idx = probs.argmax(-1)
        conf = probs.max(-1)
        keep = np.ones(len(idx), dtype=bool)
        keep[1:] = idx[1:] != idx[:-1]
        keep &= idx != 0
        chars = [self.classes[i] for i in idx[keep]]
        score = float(conf[keep].mean()) if keep.any() else 0.0
        return "".join(chars), score


def ctc_fits(label: list[int], width: int) -> bool:
    """CTC needs one time step per character plus one blank between repeated characters."""
    steps = time_steps(width) - 1
    repeats = sum(1 for a, b in zip(label, label[1:]) if a == b)
    return len(label) + repeats <= steps


# ------------------------------------------------------------- images ----------
def load_bgr(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.ascontiguousarray(np.asarray(img)[:, :, ::-1])


def resized_width(h: int, w: int) -> int:
    return min(MAX_INPUT_WIDTH, int(math.ceil(INPUT_HEIGHT * w / float(h))))


def batch_tensor(images: list[np.ndarray]) -> np.ndarray:
    """Resize, normalise and pad a batch exactly like paddlex 3.7.2 TextRecPredictor:
    the padded width is int(48 * max(320/48, widest aspect ratio in the batch)), each
    image is resized to height 48 keeping its aspect ratio (capped at that width) and
    scaled to [-1, 1]; the rest stays 0. Images are BGR uint8 arrays."""
    max_ratio = max([MIN_INPUT_WIDTH / INPUT_HEIGHT] + [im.shape[1] / float(im.shape[0]) for im in images])
    width = int(INPUT_HEIGHT * max_ratio)
    out = np.zeros((len(images), 3, INPUT_HEIGHT, min(width, MAX_INPUT_WIDTH)), dtype="float32")
    for k, im in enumerate(images):
        h, w = im.shape[:2]
        if width > MAX_INPUT_WIDTH:
            rw = MAX_INPUT_WIDTH
        else:
            rw = min(width, int(math.ceil(INPUT_HEIGHT * w / float(h))))
        r = cv2.resize(im, (rw, INPUT_HEIGHT)).astype("float32").transpose((2, 0, 1)) / 255.0
        out[k, :, :, :rw] = (r - 0.5) / 0.5
    return out


def time_steps(width: int) -> int:
    return width // 8


def augment(bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    """Light, label-preserving augmentation for real and synthetic crops."""
    img = Image.fromarray(bgr[:, :, ::-1].copy()).convert("L")
    if rng.random() < 0.5:  # crop margin jitter (detector boxes vary)
        w, h = img.size
        dx, dy = int(w * rng.uniform(0, 0.03)), int(h * rng.uniform(0, 0.12))
        img = img.crop((rng.randint(0, dx), rng.randint(0, dy), w - rng.randint(0, dx), h - rng.randint(0, dy)))
    if rng.random() < 0.3:  # horizontal stretch
        w, h = img.size
        img = img.resize((max(8, int(w * rng.uniform(0.85, 1.15))), h), Image.BILINEAR)
    img = synth.degrade(img, rng, rng.uniform(0.0, 0.5))
    arr = np.asarray(img.convert("RGB"))[:, :, ::-1]
    return np.ascontiguousarray(arr)


# ------------------------------------------------------------- batching --------
@dataclass
class Prepared:
    record: Record
    label: list[int]
    width: int  # resized width before padding


def prepare(records: list[Record], charset: Charset) -> tuple[list[Prepared], dict]:
    """Encode labels and drop lines the model cannot learn. Returns (kept, reasons)."""
    kept, skipped = [], {"unsupported_characters": 0, "empty": 0, "too_long_for_width": 0, "unreadable_image": 0}
    unsupported: set[str] = set()
    for r in records:
        if not r.text.strip():
            skipped["empty"] += 1
            continue
        label = charset.encode(r.target)
        if label is None:
            skipped["unsupported_characters"] += 1
            unsupported |= charset.unsupported(r.target)
            continue
        try:
            with Image.open(r.image) as im:
                w, h = im.size
        except Exception:
            skipped["unreadable_image"] += 1
            continue
        rw = resized_width(h, w)
        if not ctc_fits(label, max(MIN_INPUT_WIDTH, rw)):
            skipped["too_long_for_width"] += 1
            continue
        kept.append(Prepared(r, label, rw))
    # Unsupported characters are reported by Unicode code point only, never as text.
    skipped["unsupported_codepoints"] = sorted(f"U+{ord(c):04X}" for c in unsupported)
    return kept, skipped


def batches(items: list[Prepared], batch_size: int, rng: random.Random):
    """Endless shuffled batches; items of similar width are grouped to limit padding."""
    while True:
        order = list(items)
        rng.shuffle(order)
        chunk = batch_size * 20
        for start in range(0, len(order), chunk):
            group = sorted(order[start:start + chunk], key=lambda p: p.width)
            bs = [group[i:i + batch_size] for i in range(0, len(group), batch_size)]
            rng.shuffle(bs)
            yield from (b for b in bs if b)


def make_train_batch(items: list[Prepared], rng: random.Random, do_augment: bool = True):
    images, labels = [], []
    for p in items:
        bgr = load_bgr(p.record.image)
        if do_augment:
            bgr = augment(bgr, rng)
        images.append(bgr)
        labels.append(p.label)
    x = batch_tensor(images)
    steps = time_steps(x.shape[3])
    keep = [k for k, lab in enumerate(labels) if ctc_fits(lab, x.shape[3])]  # augmentation may shrink width
    if not keep:
        return x[:0], np.zeros((0, 1), dtype="int32"), np.zeros(0, dtype="int64"), steps
    x = x[keep]
    labels = [labels[k] for k in keep]
    max_len = max(len(lab) for lab in labels)
    y = np.zeros((len(labels), max_len), dtype="int32")
    for k, lab in enumerate(labels):
        y[k, : len(lab)] = lab
    lengths = np.array([len(lab) for lab in labels], dtype="int64")
    return x, y, lengths, steps
