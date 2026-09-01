#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fit a low-order generalized-coordinate Kms(x) ROM from measured tabular data."
    )
    p.add_argument("csv", type=Path)
    p.add_argument("--x-column", default="x_mm")
    p.add_argument("--kms-column", default="Kms_N_per_mm")
    p.add_argument("--x-unit", choices=["m", "mm"], default="mm")
    p.add_argument("--kms-unit", choices=["N/m", "N/mm"], default="N/mm")
    p.add_argument("--order", type=int, default=4)
    p.add_argument("--scale-mm", type=float, default=None)
    p.add_argument("--limit-mm", type=float, default=None)
    p.add_argument("--out", type=Path, required=True)
    return p


def main() -> int:
    args = _parser().parse_args()
    frame = pd.read_csv(args.csv)
    x = frame[args.x_column].to_numpy(dtype=float, copy=True)
    kms = frame[args.kms_column].to_numpy(dtype=float, copy=True)
    if args.x_unit == "mm":
        x *= 1e-3
    if args.kms_unit == "N/mm":
        kms *= 1e3
    order = int(args.order)
    if order < 1:
        raise ValueError("order must be >= 1")
    mask = np.isfinite(x) & np.isfinite(kms)
    x, kms = x[mask], kms[mask]
    if len(x) < order + 2:
        raise ValueError("insufficient finite samples for requested ROM order")
    idx = np.argsort(x)
    x, kms = x[idx], kms[idx]
    if x[0] > 0.0 or x[-1] < 0.0:
        raise ValueError("measured displacement range must span q=0")
    kms0 = float(np.interp(0.0, x, kms))
    if kms0 <= 0.0:
        raise ValueError("interpolated Kms(0) must be positive")
    measured_symmetric_limit = float(min(abs(x[0]), abs(x[-1])))
    scale = (
        float(args.scale_mm) * 1e-3
        if args.scale_mm is not None
        else measured_symmetric_limit
    )
    limit = (
        float(args.limit_mm) * 1e-3
        if args.limit_mm is not None
        else measured_symmetric_limit
    )
    if scale <= 0 or limit <= 0:
        raise ValueError("scale/limit must be positive")
    if limit > measured_symmetric_limit * (1.0 + 1e-12):
        raise ValueError(
            "declared ROM limit exceeds the displacement range measured on both sides of q=0"
        )
    inside = np.abs(x) <= limit * (1.0 + 1e-12)
    xfit, kfit = x[inside], kms[inside]
    xi = xfit / scale
    ratio = kfit / kms0
    # Constrain c0=1 so the normalized curve is exact at the rest position.
    A = np.column_stack([xi**power for power in range(1, order + 1)])
    tail, *_ = np.linalg.lstsq(A, ratio - 1.0, rcond=None)
    coeff = np.r_[1.0, tail]
    predicted = np.polynomial.polynomial.polyval(xi, coeff)
    error = predicted - ratio
    payload = {
        "kind": "polynomial_secant_stiffness_ratio",
        "displacement_scale_m": scale,
        "displacement_limit_m": limit,
        "stiffness_ratio_power_coefficients": [float(v) for v in coeff],
        "metadata": {
            "provenance": "fitted from measured Kms(x) table",
            "source_csv_name": args.csv.name,
            "source_csv_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
            "source_x_column": args.x_column,
            "source_kms_column": args.kms_column,
            "measured_Kms0_N_m": kms0,
            "fit_order": order,
            "fit_samples": int(len(xfit)),
            "fit_rms_ratio_error": float(np.sqrt(np.mean(error**2))),
            "fit_max_abs_ratio_error": float(np.max(np.abs(error))),
            "note": (
                "Production correlation should calibrate the FEM small-signal suspension stiffness "
                "to measured Kms(0), then use this normalized curve as the nonlinear correction shape."
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["metadata"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
