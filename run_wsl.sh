#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_BASE="${XDG_CACHE_HOME:-${HOME}/.cache}"
ENV_DIR="$CACHE_BASE/loudspeaker-time-fem-venv"
SCRATCH_DIR="$CACHE_BASE/loudspeaker-time-fem-runs"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  "$PROJECT_DIR/setup_wsl.sh"
fi
mkdir -p "$SCRATCH_DIR"
"$ENV_DIR/bin/python" "$PROJECT_DIR/cli.py" run \
  --config "$PROJECT_DIR/configs/transient_70Hz.json" \
  --scratch-root "$SCRATCH_DIR" \
  --outdir "$PROJECT_DIR/runs/transient_70Hz"
