#!/usr/bin/env python3
"""最小独立包自检：只装配生产模型，不依赖历史结果或 COMSOL。"""
from __future__ import annotations

import json
from pathlib import Path

from loudspeaker_time_fem.config import load_config
from loudspeaker_time_fem.model import build_transient_model


ROOT = Path(__file__).resolve().parent


def main() -> int:
    config_path = ROOT / "configs/transient_70Hz_nonlinear_comsol_physical_abc.json"
    config, resolved = load_config(config_path)
    model = build_transient_model(config, resolved)
    required = [
        ROOT / "inputs/comsol_transient_mesh.mphtxt",
        ROOT / "inputs/nonlinear_magnetic_law_20260728.json",
        ROOT / "inputs/frequency_mainline/best_model/coupled_solver.py",
        ROOT / "inputs/frequency_mainline/inputs/meshes/comsol_geometry_polyline_coarse_2p5mm.msh",
        ROOT / "README_CN.md",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    report = {
        "status": "PASS" if not missing else "FAIL",
        "project": ROOT.name,
        "metadata": model.metadata,
        "missing_required_evidence": missing,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
