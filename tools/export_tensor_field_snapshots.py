#!/usr/bin/env python3
"""Export four native FEM magnetic fields at the diagnostic transient extrema.

The transient solver consumes the frozen scalar tensor surface.  This helper
does not alter that solve: it re-solves the native magnetic FEM at the four
Python-trajectory extrema so the handoff retains representative A_phi, B and
mu_r VTU evidence rather than presenting the scalar ROM as a field solution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_tensor_magnetic_coenergy import (
    MagneticFEM,
    build_context,
    refine_tagged_mesh,
    write_pilot_vtu,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="export four native magnetic VTU extrema")
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--transient", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    scan = args.scan.resolve()
    transient = args.transient.resolve()
    out = args.out.resolve()
    summary = json.loads((scan / "pilot_summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "passed":
        raise RuntimeError("native pilot must pass before exporting transient field snapshots")
    frame = pd.read_csv(transient / "all_probes_timeseries.csv")
    required = {"time_s", "coil_displacement_m", "current_A"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"transient export missing {sorted(required.difference(frame.columns))}")
    candidates = {
        "max_positive_displacement": int(frame["coil_displacement_m"].idxmax()),
        "max_negative_displacement": int(frame["coil_displacement_m"].idxmin()),
        "max_positive_current": int(frame["current_A"].idxmax()),
        "max_negative_current": int(frame["current_A"].idxmin()),
    }
    context = build_context()
    mesh = context["mesh"]
    level = int(summary["selected_mesh_level"])
    for _ in range(level):
        mesh = refine_tagged_mesh(mesh)
    solver = MagneticFEM(mesh, context["mainline"], mesh_level=level)
    field_dir = out / "field_snapshots"
    field_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, index in candidates.items():
        row = frame.iloc[index]
        x_m = float(row["coil_displacement_m"])
        current_A = float(row["current_A"])
        result = solver.solve_point(x_m, current_A, zero_initial=True)
        filename = f"transient_{label}_x_{x_m * 1e3:+.6f}mm_i_{current_A:+.6f}A_L{level}.vtu".replace("+", "p").replace("-", "m")
        path = field_dir / filename
        write_pilot_vtu(path, result, mesh)
        rows.append({
            "label": label,
            "time_s": float(row["time_s"]),
            "x_m": x_m,
            "current_A": current_A,
            "mesh_level": level,
            "vtu": str(path.relative_to(out)),
            "sha256": sha256(path),
            "residual_A": float(result["residual_A"]),
            "residual_mu": float(result["residual_mu"]),
            "pde_residual": float(result["pde_residual"]),
            "B_max_T": float(result["B_max_T"]),
        })
    manifest = {
        "schema_version": 1,
        "kind": "native_tensor_transient_field_snapshots",
        "source": "Python diagnostic transient trajectory extrema; native FEM re-solves, not ROM field translation",
        "scan_input_hash": summary["input_hash"],
        "mesh_level": level,
        "fields": ["A_phi_Wb_per_m", "B_r_T", "B_z_T", "B_norm_T", "mu_r", "domain_id"],
        "rows": rows,
    }
    (out / "field_snapshot_manifest.csv").write_text(pd.DataFrame(rows).to_csv(index=False, float_format="%.12e"), encoding="utf-8")
    (out / "field_snapshot_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
