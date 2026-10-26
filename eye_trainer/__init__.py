"""Digital Eye Trainer: local, offline fine-tuning of the Arabic PP-OCRv5 text recogniser."""

__version__ = "0.1.0"

from . import winfribidi as _winfribidi

_winfribidi.enable()  # Windows: let Pillow find fribidi-0.dll before it loads its font module
