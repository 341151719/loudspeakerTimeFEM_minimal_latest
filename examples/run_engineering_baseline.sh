#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
PYTHONPATH=src python3 cli.py run \
  --config configs/transient_70Hz_nonlinear_comsol_physical_abc.json \
  --scratch-root /tmp/loudspeaker-time-fem-runs \
  --outdir runs/transient_70Hz_nonlinear_comsol_physical_abc
