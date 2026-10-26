# FriBiDi (Windows only)

Pillow's Windows wheels need `fribidi-0.dll` for Arabic layout (libraqm). It is not stored in
git: `python -m eye_trainer download-models` downloads the unmodified conda-forge build
`fribidi-1.0.17-hfd05255_0` (checksums pinned in `eye_trainer/winfribidi.py`) into
`eye_trainer/_bin/`, and the Windows .exe ships it next to `DigitalEyeTrainer.exe`.

Licence: LGPL-2.1-or-later (`COPYING`). Source: https://github.com/fribidi/fribidi/tree/v1.0.17
You may replace the DLL with any compatible FriBiDi build.
