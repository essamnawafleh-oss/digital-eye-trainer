"""Workspace folder layout. Everything the tool reads or writes about your documents
lives under one folder (default: ``./workspace``), which is git-ignored."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_REC = "arabic_PP-OCRv5_mobile_rec"
CROSS_REC = "en_PP-OCRv5_mobile_rec"
DET = "PP-OCRv5_mobile_det"
ALL_MODELS = (DET, BASE_REC, CROSS_REC)


@dataclass
class Workspace:
    root: Path

    @classmethod
    def default(cls, path: str | Path | None = None) -> "Workspace":
        p = Path(path or os.environ.get("EYE_TRAINER_WORKSPACE") or "workspace").expanduser().resolve()
        ws = cls(p)
        for sub in ("models", "inbox", "crops", "synthetic", "datasets", "runs", "reports", "private",
                    "exports", "fonts"):
            (p / sub).mkdir(parents=True, exist_ok=True)
        return ws

    def model_dir(self, name: str) -> Path:
        return self.root / "models" / name

    @property
    def base_rec(self) -> Path:
        return self.model_dir(BASE_REC)

    def run_dir(self, name: str) -> Path:
        return self.root / "runs" / name

    def report_dir(self, name: str) -> Path:
        return self.root / "reports" / name

    def synthetic(self, split: str) -> Path:
        return self.root / "synthetic" / split

    def real(self, split: str) -> Path:
        return self.root / "datasets" / f"real_{split}"
