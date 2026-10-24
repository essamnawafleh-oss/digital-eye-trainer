#!/usr/bin/env bash
# System-level offline proof (Linux): runs a command in a network namespace that has
# no network interfaces, so nothing it starts can reach any network.
#   scripts/verify_offline.sh python -m eye_trainer demo --steps 20 --lines 200
#   scripts/verify_offline.sh python -m eye_trainer ui --no-browser
set -euo pipefail
cd "$(dirname "$0")/.."
if ! unshare -rn true 2>/dev/null; then
  echo "unshare -rn is not permitted on this system: disconnect the network cable / Wi-Fi instead."
  exit 2
fi
[ -d .venv ] && . .venv/bin/activate
[ "$#" -gt 0 ] || set -- python -m eye_trainer demo --steps 20 --lines 200
exec unshare -rn "$@"
