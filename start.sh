#!/usr/bin/env bash
# Opens the local page (http://127.0.0.1:8765). Works without internet.
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] || { echo "Run ./install.sh first / شغّل ./install.sh أولاً"; exit 1; }
. .venv/bin/activate
exec python -m eye_trainer ui "$@"
