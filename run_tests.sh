#!/usr/bin/env bash
# Runs the automated tests (synthetic data only). Takes a few minutes on a CPU.
set -euo pipefail
cd "$(dirname "$0")"
. .venv/bin/activate
python -m pip install -q pytest==9.1.1
python -m pytest "$@"
