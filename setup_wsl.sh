#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_BASE="${XDG_CACHE_HOME:-${HOME}/.cache}"
ENV_DIR="$CACHE_BASE/loudspeaker-time-fem-venv"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  uv venv "$ENV_DIR"
fi
uv pip install --python "$ENV_DIR/bin/python" -e "$PROJECT_DIR[test]"
echo "WSL environment ready: $ENV_DIR"
