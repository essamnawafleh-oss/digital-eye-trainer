"""Entry point of the packaged application (DigitalEyeTrainer.exe).

Double-click: opens the local page. With arguments it behaves like
``python -m eye_trainer <command>`` (e.g. ``DigitalEyeTrainer.exe check``).

The workspace (your documents, labels, models) is the ``workspace`` folder next to the
exe, or ``%LOCALAPPDATA%\\DigitalEyeTrainer\\workspace`` if that folder is not writable.
The three base models are bundled and copied into the workspace on first start, so
the packaged application never needs the internet.
"""

from __future__ import annotations

import multiprocessing
import os
import shutil
import sys
from pathlib import Path


def _app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def _writable(folder: Path) -> bool:
    try:
        folder.mkdir(parents=True, exist_ok=True)
        probe = folder / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def _workspace() -> Path:
    if os.environ.get("EYE_TRAINER_WORKSPACE"):
        return Path(os.environ["EYE_TRAINER_WORKSPACE"])
    here = _app_dir() / "workspace"
    if _writable(here):
        return here
    local = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "DigitalEyeTrainer" / "workspace"
    local.mkdir(parents=True, exist_ok=True)
    return local


def _install_bundled_models(ws: Path) -> None:
    bundled = Path(getattr(sys, "_MEIPASS", _app_dir())) / "base_models"
    if not bundled.is_dir():
        return
    for src in bundled.iterdir():
        dst = ws / "models" / src.name
        if src.is_dir() and not (dst / "inference.pdiparams").is_file():
            shutil.copytree(src, dst, dirs_exist_ok=True)


def _point_paddle_at_bundle() -> None:
    """Paddle loads libmklml / libiomp lazily from <site-packages>/paddle/libs
    (paddle/base/core.py, set_paddle_lib_path). A frozen app has no site-packages, so
    give Paddle the bundled folder explicitly."""
    if not getattr(sys, "frozen", False):
        return
    libs = Path(getattr(sys, "_MEIPASS", _app_dir())) / "paddle" / "libs"
    if os.name == "nt":
        os.add_dll_directory(str(libs))
        os.environ["PATH"] = str(libs) + os.pathsep + os.environ.get("PATH", "")
    import paddle.base.core as core

    core._set_paddle_lib_path(str(libs))


def main() -> int:
    multiprocessing.freeze_support()
    _point_paddle_at_bundle()
    ws = _workspace()
    os.environ["EYE_TRAINER_WORKSPACE"] = str(ws)
    _install_bundled_models(ws)
    from eye_trainer.cli import main as cli
    from eye_trainer.messages import configure_console

    configure_console()
    double_click = len(sys.argv) == 1
    try:
        code = cli(["ui"] if double_click else sys.argv[1:])
    except Exception:  # keep the window open so the message can be read
        import traceback

        traceback.print_exc()
        code = 1
    if double_click and code not in (0, 130):
        input("\nاضغط Enter للإغلاق / Press Enter to close ")
    return code


if __name__ == "__main__":
    sys.exit(main())
