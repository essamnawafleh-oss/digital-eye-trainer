"""Import documents: PDF/images -> pages -> text-line detection -> crops -> suggestions.

Uses the same detector, settings and cropping as Digital Eye (PaddleOCR 3.7.0):
PP-OCRv5_mobile_det, limit_type "max", limit_side_len 960, oneDNN off, quad crops by
paddlex CropByPolys. Each crop is pre-filled with the current Arabic model's reading
(exactly what Digital Eye would get) and the English model's reading as a second opinion.
Runs inside the offline guard.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from . import textutil as tu
from .messages import say
from .offline import offline
from .store import Store, file_doc_id, priority
from .workspace import BASE_REC, CROSS_REC, DET, Workspace

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PDF_EXT = {".pdf"}
RENDER_DPI = 200  # Digital Eye's default render_dpi


def find_documents(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXT | PDF_EXT)


def pages_of(path: Path, dpi: int = RENDER_DPI):
    """Yield RGB PIL pages. PDFs are rendered with pypdfium2 (Apache-2.0/BSD)."""
    if path.suffix.lower() in PDF_EXT:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(path))
        try:
            for i in range(len(pdf)):
                yield pdf[i].render(scale=dpi / 72).to_pil().convert("RGB")
        finally:
            pdf.close()
    else:
        with Image.open(path) as im:
            frames = getattr(im, "n_frames", 1)
            for i in range(frames):
                im.seek(i)
                yield im.convert("RGB")


def group_of(path: Path, folder: Path) -> str | None:
    """Files in the same sub-folder of the import folder form one document group
    (all pages / re-scans of one permit). Files directly in the folder are their own group.
    Only a hash of the sub-folder path is stored."""
    import hashlib

    rel = path.resolve().parent.relative_to(folder.resolve())
    if rel == Path("."):
        return None
    return "g" + hashlib.sha256(rel.as_posix().encode("utf-8")).hexdigest()[:16]


class Importer:
    def __init__(self, ws: Workspace, device: str = "cpu"):
        self.ws = ws
        self.device = device
        self._det = self._rec = self._cross = self._crop = None

    def _load(self):
        if self._det is not None:
            return
        for name in (DET, BASE_REC, CROSS_REC):
            if not (self.ws.model_dir(name) / "inference.yml").is_file():
                raise RuntimeError(say(f"النموذج {name} غير موجود. شغّل أولاً: تنزيل النماذج.",
                                       f"model {name} is missing; run download-models first"))
        from paddleocr import TextDetection, TextRecognition
        from paddlex.inference.pipelines.components import CropByPolys

        common = dict(device=self.device, enable_mkldnn=False)
        self._det = TextDetection(model_name=DET, model_dir=str(self.ws.model_dir(DET)),
                                  limit_type="max", limit_side_len=960, **common)
        self._rec = TextRecognition(model_name=BASE_REC, model_dir=str(self.ws.model_dir(BASE_REC)), **common)
        self._cross = TextRecognition(model_name=CROSS_REC, model_dir=str(self.ws.model_dir(CROSS_REC)), **common)
        self._crop = CropByPolys(det_box_type="quad")

    def import_folder(self, folder: str | Path, store: Store, log=print) -> dict:
        files = find_documents(Path(folder))
        summary = {"files": len(files), "imported": 0, "already_imported": 0, "failed": 0, "lines": 0}
        if not files:
            log(say("لم يُعثر على ملفات PDF أو صور في المجلد.", "No PDF or image files found in that folder."))
            return summary
        with offline():
            self._load()
            for n, path in enumerate(files, 1):
                doc_id = file_doc_id(path)
                if store.has_doc(doc_id):
                    summary["already_imported"] += 1
                    continue
                try:
                    lines, pages = self._import_one(path, doc_id)
                except Exception as exc:  # keep going with the other files
                    summary["failed"] += 1
                    log(say(f"تعذّر استيراد الملف رقم {n}: {type(exc).__name__}",
                            f"could not import file #{n}: {type(exc).__name__}: {exc}"))
                    continue
                store.add_doc(doc_id, path.name, "pdf" if path.suffix.lower() in PDF_EXT else "image", pages,
                              group=group_of(path, Path(folder)))
                store.add_lines(lines)
                summary["imported"] += 1
                summary["lines"] += len(lines)
                log(say(f"ملف {n} من {len(files)}: {len(lines)} سطر", f"file {n}/{len(files)}: {len(lines)} lines"))
        store.assign_splits()
        return summary

    def _import_one(self, path: Path, doc_id: str) -> tuple[list[dict], int]:
        crop_dir = self.ws.root / "crops" / doc_id
        crop_dir.mkdir(parents=True, exist_ok=True)
        rows, pages = [], 0
        for page_no, page in enumerate(pages_of(path), 1):
            pages += 1
            bgr = np.ascontiguousarray(np.asarray(page)[:, :, ::-1])
            det = next(iter(self._det.predict(bgr)))
            polys = [np.asarray(p, dtype=np.float32) for p in det["dt_polys"]]
            polys.sort(key=lambda p: (round(float(p[:, 1].min()) / 10), -float(p[:, 0].max())))
            crops = [c for c in self._crop(bgr, polys)]
            keep = [i for i, c in enumerate(crops) if c.size and c.shape[0] >= 4 and c.shape[1] >= 4]
            if not keep:
                continue
            crops = [crops[i] for i in keep]
            polys = [polys[i] for i in keep]
            main = list(self._rec.predict(crops, batch_size=8))
            alt = list(self._cross.predict(crops, batch_size=8))
            for idx, (crop, poly, m, a) in enumerate(zip(crops, polys, main, alt)):
                rel = Path("crops") / doc_id / f"p{page_no:03d}_l{idx:03d}.png"
                Image.fromarray(crop[:, :, ::-1]).save(self.ws.root / rel)
                text, score = str(m["rec_text"]), float(m["rec_score"])
                alt_text, alt_score = str(a["rec_text"]), float(a["rec_score"])
                if not tu.has_arabic(text) and not tu.has_arabic(alt_text):
                    # Digital Eye's rule: Latin/digit-only lines are read by the English model.
                    text, score, alt_text, alt_score = alt_text, alt_score, text, score
                rows.append({"doc_id": doc_id, "page": page_no, "idx": idx, "crop": rel.as_posix(),
                             "poly": np.asarray(poly).round(1).tolist(), "suggestion": text, "score": score,
                             "alt_text": alt_text, "alt_score": alt_score,
                             "priority": priority(text, score, alt_text)})
        return rows, pages
