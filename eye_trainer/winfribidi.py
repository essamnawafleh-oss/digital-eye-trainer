"""FriBiDi for Pillow on Windows.

Pillow's Windows wheels contain libraqm and HarfBuzz but load FriBiDi (LGPL) at run time
from ``fribidi-0.dll``; without it Pillow silently falls back to basic layout and Arabic
is drawn unshaped (verified on GitHub's Windows runner, Pillow 13.0.0). This module
installs the DLL from a pinned conda-forge build (checksums below) during
``download-models`` and makes it findable before Pillow loads its font module.

FriBiDi: LGPL-2.1-or-later, unmodified binary; licence text in third_party/fribidi/COPYING,
source code at https://github.com/fribidi/fribidi/tree/v1.0.17.
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

PACKAGE_URL = "https://conda.anaconda.org/conda-forge/win-64/fribidi-1.0.17-hfd05255_0.conda"
PACKAGE_SHA256 = "30a93a5132923a5679e0729a02bf5b36052c84cc437decc14a35caee5e276f46"
DLL_MEMBER = "Library/bin/fribidi-0.dll"
DLL_SHA256 = "eb5002db275aeb841cf2a599bb2b8894424df4be4022aef089b7a98ee29cb8a5"
DLL_NAME = "fribidi-0.dll"
BIN_DIR = Path(__file__).resolve().parent / "_bin"


def _candidates() -> list[Path]:
    dirs = [BIN_DIR]
    if getattr(sys, "frozen", False):
        dirs.append(Path(sys.executable).resolve().parent)
    return dirs


def enable() -> bool:
    """Make fribidi-0.dll findable (call before PIL.ImageFont is imported)."""
    if sys.platform != "win32":
        return True
    for d in _candidates():
        if (d / DLL_NAME).is_file():
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(str(d))
            except OSError:
                pass
            return True
    return False


def install(dest: Path = BIN_DIR) -> Path:
    """Download the pinned package, verify both checksums, extract the DLL."""
    import zstandard

    target = dest / DLL_NAME
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == DLL_SHA256:
        return target
    data = urllib.request.urlopen(PACKAGE_URL, timeout=120).read()
    if hashlib.sha256(data).hexdigest() != PACKAGE_SHA256:
        raise RuntimeError("FriBiDi download has an unexpected checksum; not installed")
    outer = zipfile.ZipFile(io.BytesIO(data))
    inner_name = next(n for n in outer.namelist() if n.startswith("pkg-"))
    raw = zstandard.ZstdDecompressor().decompressobj().decompress(outer.read(inner_name))
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        dll = tar.extractfile(DLL_MEMBER).read()
    if hashlib.sha256(dll).hexdigest() != DLL_SHA256:
        raise RuntimeError("FriBiDi DLL has an unexpected checksum; not installed")
    dest.mkdir(parents=True, exist_ok=True)
    target.write_bytes(dll)
    return target
