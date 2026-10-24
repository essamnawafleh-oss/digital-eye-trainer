"""Shared fixtures. All test data is synthetic and generated on the fly.

Tests that need the base models look for them in $EYE_TRAINER_TEST_MODELS, then in
./workspace/models, and are skipped (with the reason) if they are not there.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ("PP-OCRv5_mobile_det", "arabic_PP-OCRv5_mobile_rec", "en_PP-OCRv5_mobile_rec")


def models_root() -> Path | None:
    for cand in (os.environ.get("EYE_TRAINER_TEST_MODELS"), ROOT / "workspace" / "models"):
        if cand and all((Path(cand) / m / "inference.pdiparams").is_file() for m in MODEL_NAMES):
            return Path(cand)
    return None


requires_models = pytest.mark.skipif(models_root() is None, reason="base models not downloaded "
                                     "(run: python -m eye_trainer download-models, or set EYE_TRAINER_TEST_MODELS)")


@pytest.fixture(scope="session")
def base_models() -> Path:
    root = models_root()
    if root is None:
        pytest.skip("base models not downloaded")
    return root


@pytest.fixture(scope="session")
def fonts():
    from eye_trainer import synth

    return synth.load_fonts()


@pytest.fixture()
def workspace(tmp_path, base_models):
    """A fresh workspace with the base models copied in."""
    from eye_trainer.workspace import Workspace

    ws = Workspace.default(tmp_path / "ws")
    for m in MODEL_NAMES:
        shutil.copytree(base_models / m, ws.model_dir(m), ignore=shutil.ignore_patterns(".cache"))
    return ws


@pytest.fixture()
def empty_workspace(tmp_path):
    from eye_trainer.workspace import Workspace

    return Workspace.default(tmp_path / "ws")
