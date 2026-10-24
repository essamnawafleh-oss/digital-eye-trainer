#!/usr/bin/env bash
# One-time installation (needs internet ONCE). Linux / macOS.
#   ./install.sh          CPU (works everywhere)
#   ./install.sh gpu      NVIDIA GPU (needs a matching CUDA driver; see README)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
  done
fi
[ -n "$PY" ] || { echo "Python 3.10-3.13 is required / يلزم تثبيت بايثون 3.10-3.13"; exit 1; }
echo "== Python: $($PY --version)"
[ -d .venv ] || "$PY" -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt
if [ "${1:-}" = "gpu" ]; then
  echo "== Installing the GPU build of PaddlePaddle (CUDA 12.6). UNVERIFIED by the authors - see README."
  python -m pip uninstall -y paddlepaddle
  python -m pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
fi
echo "== Downloading the base models (Apache-2.0, ~21 MB) / تنزيل النماذج الأساسية"
python -m eye_trainer download-models
python -m eye_trainer check
echo
echo "== Done. Start with ./start.sh  /  انتهى التثبيت. شغّل ./start.sh"
