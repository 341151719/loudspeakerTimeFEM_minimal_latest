#!/usr/bin/env python3
"""Strict common-basis comparison of raw COMSOL and Python transient exports."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


PROBE_MAP = {
    "python_axis_near_actual": "p_axis_near_0p10m_Pa",
    "python_axis_boundary_actual": "p_axis_boundary93_0p165m_Pa",
    "python_axis_rear_actual": "p_axis_rear_m0p12m_Pa",
    "python_offaxis_actual": "p_offaxis_45deg_0p10m_Pa",
    "common_axis_0p14m": "p_common_axis_0p14m_Pa",
    "common_rear_physical_m0p10": "p_rear_physical_m0p10_Pa",
}
# This coordinate lies between the physical-domain/PML interface (R=0.115 m)
# and the COMSOL outer boundary. A physical-domain ABC model cannot reproduce
# the pressure field inside COMSOL's transformed PML coordinates. Preserve and
# report it, but do not use it as a radiation-field acceptance criterion.
DIAGNOSTIC_ONLY_PRESSURE_PROBES = {"python_axis_rear_actual"}
GLOBAL_MAP = {
    "coil_current_A": "current_A",
    "coil_displacement_m": "coil_displacement_m",
    "dynamic_BL_N_A": "dynamic_BL_N_A",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_harmonics(time: np.ndarray, value: np.ndarray, f0: float, nh: int = 10):
    columns = [np.ones_like(time)]
    for order in range(1, nh + 1):
        omega = 2 * math.pi * f0 * order
        columns.extend([np.sin(omega * time), np.cos(omega * time)])
    design = np.column_stack(columns)
    coefficient, *_ = np.linalg.lstsq(design, value, rcond=None)
    complex_peak = np.array(
        [coefficient[2 * n] - 1j * coefficient[2 * n - 1] for n in range(1, nh + 1)]
    )
    return coefficient, complex_peak, design @ coefficient


def phase_difference_deg(candidate: complex, reference: complex) -> float:
    return float(np.degrees(np.angle(candidate / reference)))


def compare_signal(name, tc, yc, tp, yp, f0, python_cycle, metric_rows, harmonic_rows):
    period = 1.0 / f0
    clo, chi = 3 * period - 1e-10, 4 * period - 1e-10
    plo, phi = (python_cycle - 1) * period - 1e-10, python_cycle * period - 1e-10
    cmask = (tc >= clo) & (tc < chi)
    pmask = (tp >= plo) & (tp < phi)
    tc1, yc1 = tc[cmask], yc[cmask]
    tp1, yp1 = tp[pmask], yp[pmask]
    if len(tc1) < 22 or len(tp1) < 22:
        raise RuntimeError(f"{name}: insufficient last-cycle samples: COMSOL={len(tc1)} Python={len(tp1)}")
    cc, hc, fitc = fit_harmonics(tc1, yc1, f0)
    cp, hp, _ = fit_harmonics(tp1, yp1, f0)
    # Compare on identical COMSOL times using the Python trigonometric fit.
    design_common = [np.ones_like(tc1)]
    for order in range(1, 11):
        omega = 2 * math.pi * f0 * order
        design_common.extend([np.sin(omega * tc1), np.cos(omega * tc1)])
    fitp_common = np.column_stack(design_common) @ cp
    nrmse = float(np.sqrt(np.mean((fitp_common - yc1) ** 2)) / max(np.sqrt(np.mean(yc1**2)), 1e-300))
    thdc = float(np.linalg.norm(hc[1:]) / max(abs(hc[0]), 1e-300))
    thdp = float(np.linalg.norm(hp[1:]) / max(abs(hp[0]), 1e-300))
    metric_rows.append(
        {
            "signal": name,
            "comsol_samples": len(tc1),
            "python_samples": len(tp1),
            "comsol_dc": float(cc[0]),
            "python_dc": float(cp[0]),
            "comsol_H1_peak": float(abs(hc[0])),
            "python_H1_peak": float(abs(hp[0])),
            "H1_amplitude_relative_error": float(abs(abs(hp[0]) - abs(hc[0])) / max(abs(hc[0]), 1e-300)),
            "H1_phase_python_minus_comsol_deg": phase_difference_deg(hp[0], hc[0]),
            "comsol_THD": thdc,
            "python_THD": thdp,
            "THD_absolute_percentage_point_error": 100 * abs(thdp - thdc),
            "THD_relative_error": float(abs(thdp - thdc) / max(thdc, 1e-300)),
            "common_time_waveform_NRMSE": nrmse,
        }
    )
    for order, (a, b) in enumerate(zip(hc, hp), 1):
        harmonic_rows.append(
            {
                "signal": name,
                "harmonic": order,
                "frequency_Hz": order * f0,
                "comsol_peak": abs(a),
                "python_peak": abs(b),
                "amplitude_relative_error": abs(abs(b) - abs(a)) / max(abs(a), 1e-300),
                "python_minus_comsol_dB": 20 * math.log10(max(abs(b), 1e-300) / max(abs(a), 1e-300)),
                "phase_python_minus_comsol_deg": phase_difference_deg(b, a),
                "comsol_ratio_to_H1": abs(a) / max(abs(hc[0]), 1e-300),
                "python_ratio_to_H1": abs(b) / max(abs(hp[0]), 1e-300),
            }
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comsol", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--f0", type=float, default=70.0)
    parser.add_argument(
        "--python-cycle",
        type=int,
        default=4,
        help="Python 结果用于比较的周期编号；COMSOL 基准固定为第 4 周期",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cg_path = args.comsol / "global_timeseries.csv"
    cp_path = args.comsol / "pressure_points_timeseries.csv"
    py_path = args.python / "all_probes_timeseries.csv"
    cg, cplong, py = pd.read_csv(cg_path), pd.read_csv(cp_path), pd.read_csv(py_path)
    # Remesh junctions occur twice in COMSOL's concatenated dataset; keep the
    # post-remesh value so junctions do not receive extra least-squares weight.
    cg = cg.drop_duplicates("time_s", keep="last").sort_values("time_s")
    cp = cplong.pivot_table(index="time_s", columns="probe_name", values="p_Pa", aggfunc="last", dropna=False).sort_index()
    metric_rows, harmonic_rows = [], []
    for comsol_name, python_name in GLOBAL_MAP.items():
        if comsol_name in cg and cg[comsol_name].notna().all():
            compare_signal(comsol_name, cg.time_s.to_numpy(), cg[comsol_name].to_numpy(), py.time_s.to_numpy(), py[python_name].to_numpy(), args.f0, args.python_cycle, metric_rows, harmonic_rows)
    unavailable = []
    for comsol_name, python_name in PROBE_MAP.items():
        if comsol_name not in cp or not cp[comsol_name].notna().all():
            unavailable.append({"signal": "pressure_" + comsol_name, "reason": "coordinate outside COMSOL acoustic mesh or undefined"})
            continue
        compare_signal("pressure_" + comsol_name, cp.index.to_numpy(), cp[comsol_name].to_numpy(), py.time_s.to_numpy(), py[python_name].to_numpy(), args.f0, args.python_cycle, metric_rows, harmonic_rows)
    metrics = pd.DataFrame(metric_rows)
    harmonics = pd.DataFrame(harmonic_rows)
    metrics.to_csv(args.out / "comparison_metrics.csv", index=False, float_format="%.12e")
    harmonics.to_csv(args.out / "harmonic_comparison.csv", index=False, float_format="%.12e")

    criteria = {}
    for row in metric_rows:
        diagnostic_signal = row["signal"] in {
            "pressure_" + name for name in DIAGNOSTIC_ONLY_PRESSURE_PROBES
        }
        if (
            row["signal"] in {"coil_current_A", "coil_displacement_m"}
            or (row["signal"].startswith("pressure_") and not diagnostic_signal)
        ):
            criteria[row["signal"] + ":H1"] = bool(row["H1_amplitude_relative_error"] <= 0.10 and abs(row["H1_phase_python_minus_comsol_deg"]) <= 10)
    near = next(row for row in metric_rows if row["signal"] == "pressure_python_axis_near_actual")
    criteria["near_axis_THD"] = bool(near["THD_absolute_percentage_point_error"] <= 0.5 and near["THD_relative_error"] <= 0.20)
    if "dynamic_BL_N_A" in cg and cg.dynamic_BL_N_A.notna().all():
        period = 1 / args.f0
        mask = (cg.time_s >= 3 * period - 1e-10) & (cg.time_s < 4 * period - 1e-10)
        cmin, cmax = float(cg.loc[mask, "dynamic_BL_N_A"].min()), float(cg.loc[mask, "dynamic_BL_N_A"].max())
        plo, phi = (args.python_cycle - 1) * period - 1e-10, args.python_cycle * period - 1e-10
        pmin, pmax = float(py.loc[(py.time_s >= plo) & (py.time_s < phi), "dynamic_BL_N_A"].min()), float(py.loc[(py.time_s >= plo) & (py.time_s < phi), "dynamic_BL_N_A"].max())
        criteria["dynamic_BL_range_endpoints"] = bool(abs(pmin-cmin)/abs(cmin) <= .05 and abs(pmax-cmax)/abs(cmax) <= .05)
    research_criteria = {}
    current_metric = next(
        row for row in metric_rows if row["signal"] == "coil_current_A"
    )
    bl_metric = next(
        row for row in metric_rows if row["signal"] == "dynamic_BL_N_A"
    )
    research_criteria["coil_current_THD"] = bool(
        current_metric["THD_absolute_percentage_point_error"] <= 0.5
        and current_metric["THD_relative_error"] <= 0.10
    )
    research_criteria["dynamic_BL_H1"] = bool(
        bl_metric["H1_amplitude_relative_error"] <= 0.10
        and abs(bl_metric["H1_phase_python_minus_comsol_deg"]) <= 10.0
    )
    research_criteria["dynamic_BL_waveform"] = bool(
        bl_metric["common_time_waveform_NRMSE"] <= 0.01
    )
    summary = {
        "status": "comparison_completed",
        "source_hashes": {str(p): sha256(p) for p in (cg_path, cp_path, py_path)},
        "f0_Hz": args.f0,
        "window": {
            "comsol": "3*T0 <= t < 4*T0",
            "python": f"{args.python_cycle - 1}*T0 <= t < {args.python_cycle}*T0",
        },
        "criteria": criteria,
        "all_registered_physics_criteria_pass": all(criteria.values()),
        "research_criteria": research_criteria,
        "all_research_criteria_pass": all(research_criteria.values()),
        "diagnostic_only_signals": [
            "pressure_" + name for name in sorted(DIAGNOSTIC_ONLY_PRESSURE_PROBES)
        ],
        "unavailable_registered_signals": unavailable,
        "metrics": metric_rows,
        "note": "COMSOL numerical-convergence acceptance is evaluated separately and is required for the final validation claim.",
    }
    (args.out / "comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
