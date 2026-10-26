# PyInstaller spec: one-folder Windows/Linux build of Digital Eye Trainer.
#   pyinstaller packaging/DigitalEyeTrainer.spec --noconfirm
# Expects the three base models in ./base_models/<name>/ (see .github/workflows/windows-exe.yml).
from importlib.metadata import distributions
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

ROOT = Path(SPECPATH).parent
datas, binaries, hiddenimports = [], [], []
for pkg in ("paddle", "paddlex", "paddleocr", "pypdfium2", "pypdfium2_raw", "cv2", "bidi", "fontTools"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h
# Paddle opens these with dlopen()/LoadLibrary at run time, from the folder of the core
# library, which PyInstaller places at the top of the bundle: put copies next to it.
import paddle as _paddle

_libs = Path(_paddle.__file__).parent / "libs"
for pattern in ("*mklml*", "*iomp*", "*warpctc*", "*warprnnt*"):
    binaries += [(str(f), ".") for f in _libs.glob(pattern)]
# paddlex checks optional dependencies through package metadata at run time.
for dist in distributions():
    try:
        datas += copy_metadata(dist.metadata["Name"])
    except Exception:
        pass
hiddenimports += collect_submodules("eye_trainer")
datas += [
    (str(ROOT / "fonts"), "fonts"),
    (str(ROOT / "eye_trainer" / "ui" / "static"), "eye_trainer/ui/static"),
    (str(ROOT / "eye_trainer" / "vendor" / "ppocr_rec" / "LICENSE"), "eye_trainer/vendor/ppocr_rec"),
    (str(ROOT / "eye_trainer" / "vendor" / "ppocr_rec" / "NOTICE"), "eye_trainer/vendor/ppocr_rec"),
    (str(ROOT / "base_models"), "base_models"),
    (str(ROOT / "LICENSE"), "."),
    (str(ROOT / "docs" / "LICENSES.md"), "."),
]

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "IPython", "jupyter", "torch", "tensorflow"],
    # Paddle's dynamic-to-static export reads the model's Python source: keep .py files.
    module_collection_mode={"eye_trainer": "py", "paddle": "pyz+py"},
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="DigitalEyeTrainer", console=True,
          upx=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="DigitalEyeTrainer")
