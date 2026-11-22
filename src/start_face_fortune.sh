#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
VENV="${FACE_FORTUNE_VENV:-/home/willtek/work/infer_env}"
if [[ -f "$VENV/bin/activate" ]]; then
    source "$VENV/bin/activate"
else
    echo "Warning: venv not found at $VENV. Using current Python interpreter."
fi
python web_app.py --camera "${1:-1}" --host 0.0.0.0 --port "${FACE_FORTUNE_PORT:-5000}"
