#!/usr/bin/env python3
"""Run the registered tensor-vs-refined-COMSOL comparison and gates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASELINE_SUMMARY = ROOT / "comsol_validation/raw/physical_abc_v3_comparison/comparison_summary.json"
OLD_COUPLED_SUMMARY = ROOT / "comsol_validation/processed/python_physical_abc_coupled_coenergy/comparison_summary.json"
COMSOL = ROOT / "comsol_validation/raw/comsol_refined_mesh_v2"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metrics_by_signal(summary: dict) -> dict[str, dict]:
    return {str(row["signal"]): row for row in summary.get("metrics", [])}


def invoke_common_compare(tensor: Path, out: Path) -> None:
    command = [
        sys.executable,
        str(ROOT / "comsol_validation/compare_comsol_python.py"),
        "--comsol",
        str(COMSOL),
        "--python",
        str(tensor),
        "--out",
        str(out),
        "--f0",
        "70",
        "--python-cycle",
        "4",
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def make_long_metrics(out: Path, raw: pd.DataFrame, baseline: dict, old: dict) -> pd.DataFrame:
    rows = []
    source = {"baseline": metrics_by_signal(baseline), "old_coupled": metrics_by_signal(old)}
    for _, row in raw.iterrows():
        signal = str(row["signal"])
        for model in ("COMSOL", "tensor"):
            peak = row["comsol_H1_peak"] if model == "COMSOL" else row["python_H1_peak"]
            phase = 0.0 if model == "COMSOL" else row["H1_phase_python_minus_comsol_deg"]
            thd = row["comsol_THD"] if model == "COMSOL" else row["python_THD"]
            nrmse = 0.0 if model == "COMSOL" else row["common_time_waveform_NRMSE"]
            rows.append({"signal": signal, "model": model, "H1_peak": peak, "H1_phase_deg": phase, "THD": thd, "waveform_NRMSE": nrmse, "H1_amplitude_relative_error": 0.0 if model == "COMSOL" else row["H1_amplitude_relative_error"], "H1_phase_error_deg": abs(phase)})
        for model, metrics in source.items():
            old_row = metrics.get(signal)
            if old_row is None:
                continue
            rows.append({"signal": signal, "model": model, "H1_peak": old_row.get("python_H1_peak"), "H1_phase_deg": old_row.get("H1_phase_python_minus_comsol_deg"), "THD": old_row.get("python_THD"), "waveform_NRMSE": old_row.get("common_time_waveform_NRMSE"), "H1_amplitude_relative_error": old_row.get("H1_amplitude_relative_error"), "H1_phase_error_deg": abs(old_row.get("H1_phase_python_minus_comsol_deg", 0.0))})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "comparison_metrics.csv", index=False, float_format="%.12e")
    return frame


def _harmonic_coefficients(time_s: np.ndarray, values: np.ndarray, f0: float) -> np.ndarray:
    """Return DC..H10 complex coefficients on the registered cycle window."""
    time_s = np.asarray(time_s, dtype=float)
    values = np.asarray(values, dtype=float)
    columns = [np.ones_like(time_s)]
    for order in range(1, 11):
        omega = 2.0 * np.pi * f0 * order
        columns.extend([np.sin(omega * time_s), np.cos(omega * time_s)])
    coefficient, *_ = np.linalg.lstsq(np.column_stack(columns), values, rcond=None)
    return np.asarray(
        [complex(coefficient[0])]
        + [complex(coefficient[2 * order] - 1j * coefficient[2 * order - 1]) for order in range(1, 11)]
    )


def _last_cycle(time_s: np.ndarray, values: np.ndarray, f0: float, cycle: int) -> np.ndarray:
    period = 1.0 / f0
    mask = (time_s >= (cycle - 1) * period - 1e-10) & (time_s < cycle * period - 1e-10)
    if np.count_nonzero(mask) < 22:
        raise RuntimeError(f"harmonic signal has too few samples in cycle {cycle}")
    return _harmonic_coefficients(time_s[mask], values[mask], f0)


def make_harmonic(out: Path, tensor: Path, baseline: Path) -> None:
    """Export real/imaginary H1-H10 values for COMSOL, baseline and tensor."""
    raw = pd.read_csv(out / "harmonic_comparison.csv")
    comsol_global = pd.read_csv(COMSOL / "global_timeseries.csv").drop_duplicates("time_s", keep="last").sort_values("time_s")
    comsol_pressure = pd.read_csv(COMSOL / "pressure_points_timeseries.csv")
    comsol_pressure = comsol_pressure.pivot_table(index="time_s", columns="probe_name", values="p_Pa", aggfunc="last").sort_index()
    tensor_frame = pd.read_csv(tensor / "all_probes_timeseries.csv")
    baseline_frame = pd.read_csv(baseline / "all_probes_timeseries.csv")
    old_path = ROOT / "runs/transient_70Hz_nonlinear_comsol_physical_abc_coupled_coenergy_diagnostic/all_probes_timeseries.csv"
    old_frame = pd.read_csv(old_path) if old_path.is_file() else None
    mappings = {
        "coil_current_A": (comsol_global, "coil_current_A", "current_A"),
        "coil_displacement_m": (comsol_global, "coil_displacement_m", "coil_displacement_m"),
        "dynamic_BL_N_A": (comsol_global, "dynamic_BL_N_A", "dynamic_BL_N_A"),
        "pressure_python_axis_near_actual": (comsol_pressure, "pressure_python_axis_near_actual", "p_axis_near_0p10m_Pa"),
        "pressure_python_offaxis_actual": (comsol_pressure, "pressure_python_offaxis_actual", "p_offaxis_45deg_0p10m_Pa"),
        "pressure_common_rear_physical_m0p10": (comsol_pressure, "pressure_common_rear_physical_m0p10", "p_rear_physical_m0p10_Pa"),
    }
    rows = []
    raw_signals = set(raw["signal"].astype(str))
    for signal, (comsol_frame, comsol_column, python_column) in mappings.items():
        if signal not in raw_signals or comsol_column not in comsol_frame or python_column not in tensor_frame or python_column not in baseline_frame:
            continue
        ctime = comsol_frame.index.to_numpy(dtype=float) if comsol_frame is comsol_pressure else comsol_frame["time_s"].to_numpy(dtype=float)
        cvalues = comsol_frame[comsol_column].to_numpy(dtype=float)
        ccoef = _last_cycle(ctime, cvalues, 70.0, 4)
        tcoef = _last_cycle(tensor_frame.time_s.to_numpy(dtype=float), tensor_frame[python_column].to_numpy(dtype=float), 70.0, 4)
        bcoef = _last_cycle(baseline_frame.time_s.to_numpy(dtype=float), baseline_frame[python_column].to_numpy(dtype=float), 70.0, 4)
        ocoef = _last_cycle(old_frame.time_s.to_numpy(dtype=float), old_frame[python_column].to_numpy(dtype=float), 70.0, 4) if old_frame is not None and python_column in old_frame else None
        reference_h1 = abs(ccoef[1])
        for order in range(1, 11):
            model_coefficients = [("COMSOL", ccoef), ("baseline", bcoef), ("tensor", tcoef)]
            if ocoef is not None:
                model_coefficients.insert(2, ("old_coupled", ocoef))
            for model, coefficient in model_coefficients:
                value = coefficient[order]
                reference = ccoef[order]
                rows.append({
                    "signal": signal,
                    "model": model,
                    "harmonic": order,
                    "frequency_Hz": order * 70.0,
                    "complex_real": value.real,
                    "complex_imag": value.imag,
                    "peak": abs(value),
                    "phase_deg": float(np.degrees(np.angle(value))),
                    "relative_to_COMSOL_H1": abs(value) / max(reference_h1, 1e-300),
                    "amplitude_relative_error_vs_COMSOL": abs(abs(value) - abs(reference)) / max(abs(reference), 1e-300) if model != "COMSOL" else 0.0,
                    "phase_error_vs_COMSOL_deg": float(np.degrees(np.angle(value / reference))) if model != "COMSOL" and abs(reference) > 1e-300 else 0.0,
                })
    pd.DataFrame(rows).to_csv(out / "harmonic_H1_H10.csv", index=False, float_format="%.12e")


def make_waveforms(out: Path, tensor: Path, baseline: Path) -> None:
    comsol_global = pd.read_csv(COMSOL / "global_timeseries.csv").drop_duplicates("time_s", keep="last").sort_values("time_s")
    tensor_frame = pd.read_csv(tensor / "all_probes_timeseries.csv")
    base_frame = pd.read_csv(baseline / "all_probes_timeseries.csv")
    period = 1.0 / 70.0
    cmask = (comsol_global.time_s >= 3 * period - 1e-10) & (comsol_global.time_s < 4 * period - 1e-10)
    tmask = (tensor_frame.time_s >= 3 * period - 1e-10) & (tensor_frame.time_s < 4 * period - 1e-10)
    bmask = (base_frame.time_s >= 3 * period - 1e-10) & (base_frame.time_s < 4 * period - 1e-10)
    c = comsol_global.loc[cmask].reset_index(drop=True)
    t = tensor_frame.loc[tmask].reset_index(drop=True)
    b = base_frame.loc[bmask].reset_index(drop=True)
    mappings = {"coil_current_A": ("current_A", "current_A"), "coil_displacement_m": ("coil_displacement_m", "coil_displacement_m"), "dynamic_BL_N_A": ("dynamic_BL_N_A", "dynamic_BL_N_A"), "pressure_axis_near": ("pressure_python_axis_near_actual", "p_axis_near_0p10m_Pa"), "pressure_offaxis": ("pressure_python_offaxis_actual", "p_offaxis_45deg_0p10m_Pa"), "pressure_rear_physical": ("pressure_common_rear_physical_m0p10", "p_rear_physical_m0p10_Pa")}
    rows = []
    for name, (ccol, pcol) in mappings.items():
        if ccol not in c or pcol not in t or pcol not in b:
            continue
        tv = np.interp(c.time_s, t.time_s, t[pcol]); bv = np.interp(c.time_s, b.time_s, b[pcol])
        for time_s, cv, bv0, tv0 in zip(c.time_s, c[ccol], bv, tv): rows.append({"time_s": time_s, "signal": name, "COMSOL": cv, "baseline": bv0, "tensor": tv0, "tensor_minus_COMSOL": tv0 - cv, "baseline_minus_COMSOL": bv0 - cv})
    pd.DataFrame(rows).to_csv(out / "waveform_error_timeseries.csv", index=False, float_format="%.12e")


def make_attribution(out: Path, tensor: Path) -> None:
    from loudspeaker_time_fem.nonlinear_law import NonlinearMagneticLaw
    from loudspeaker_time_fem.tensor_coenergy import TensorCoenergyLaw

    global_frame = pd.read_csv(COMSOL / "global_timeseries.csv").drop_duplicates("time_s", keep="last").sort_values("time_s")
    period = 1.0 / 70.0
    global_frame = global_frame.loc[
        (global_frame.time_s >= 3.0 * period - 1e-10)
        & (global_frame.time_s < 4.0 * period - 1e-10)
    ].reset_index(drop=True)
    law = TensorCoenergyLaw.from_json(ROOT / "inputs/nonlinear_magnetic_coenergy_tensor_20260801.json")
    old = NonlinearMagneticLaw.from_json(ROOT / "inputs/nonlinear_magnetic_law_20260728.json")
    x_c = global_frame["coil_displacement_m"].to_numpy(dtype=float); i_c = global_frame["coil_current_A"].to_numpy(dtype=float)
    tensor_bl = np.asarray(law.effective_bl(x_c, i_c), dtype=float)
    old_bl = np.asarray([old.bl(x) for x in x_c], dtype=float)
    own = pd.read_csv(tensor / "magnetic_timeseries.csv").sort_values("t")
    own_t = own["t"].to_numpy(dtype=float)
    own_x = np.interp(global_frame.time_s, own_t, own["x"].to_numpy(dtype=float))
    own_i = np.interp(global_frame.time_s, own_t, own["i"].to_numpy(dtype=float))
    own_bl = np.interp(global_frame.time_s, own_t, own["BL_secant"].to_numpy(dtype=float))
    frame = pd.DataFrame({
        "t": global_frame.time_s,
        "x_C_m": x_c,
        "i_C_A": i_c,
        "COMSOL_BL_N_A": global_frame["dynamic_BL_N_A"],
        "tensor_BL_on_COMSOL_path_N_A": tensor_bl,
        "baseline_BL_on_COMSOL_path_N_A": old_bl,
        "x_P_m": own_x,
        "i_P_A": own_i,
        "tensor_BL_on_tensor_path_N_A": own_bl,
    })
    frame.to_csv(out / "attribution_trajectory.csv", index=False, float_format="%.12e")
    metrics = {}
    for name, values in {
        "COMSOL": global_frame["dynamic_BL_N_A"].to_numpy(dtype=float),
        "tensor_on_COMSOL_path": tensor_bl,
        "baseline_on_COMSOL_path": old_bl,
        "tensor_own_path": own_bl,
    }.items():
        coefficients = _harmonic_coefficients(global_frame.time_s.to_numpy(dtype=float), values, 70.0)
        metrics[name] = {
            "H1_peak": abs(coefficients[1]),
            "H1_phase_deg": float(np.degrees(np.angle(coefficients[1]))),
            "min_N_A": float(np.min(values)),
            "max_N_A": float(np.max(values)),
        }
    (out / "attribution_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def build_gate(out: Path, tensor: Path, baseline: dict, tensor_summary: dict) -> dict:
    raw = load_summary(out / "comparison_summary.json")
    rows = metrics_by_signal(raw)
    research = {}
    current = rows["coil_current_A"]
    bl = rows["dynamic_BL_N_A"]
    research["dynamic_BL_H1_amplitude"] = {"measured": bl["H1_amplitude_relative_error"], "limit": 0.10, "pass": bl["H1_amplitude_relative_error"] <= 0.10}
    research["dynamic_BL_H1_phase"] = {"measured": abs(bl["H1_phase_python_minus_comsol_deg"]), "limit": 10.0, "pass": abs(bl["H1_phase_python_minus_comsol_deg"]) <= 10.0}
    research["dynamic_BL_waveform_NRMSE"] = {"measured": bl["common_time_waveform_NRMSE"], "limit": 0.01, "pass": bl["common_time_waveform_NRMSE"] <= 0.01}
    research["coil_current_THD_absolute_pp"] = {"measured": current["THD_absolute_percentage_point_error"], "limit": 0.5, "pass": current["THD_absolute_percentage_point_error"] <= 0.5}
    research["coil_current_THD_relative"] = {"measured": current["THD_relative_error"], "limit": 0.10, "pass": current["THD_relative_error"] <= 0.10}
    baseline_rows = metrics_by_signal(baseline)
    engineering = {}
    degradation = []
    engineering_signals = ["coil_current_A", "coil_displacement_m", "pressure_python_axis_near_actual", "pressure_python_offaxis_actual", "pressure_common_rear_physical_m0p10"]
    for signal in engineering_signals:
        if signal not in rows:
            engineering[signal] = {"pass": False, "reason": "missing registered comparison signal"}
            continue
        tensor_row = rows[signal]
        engineering[signal] = {
            "H1_amplitude_relative_error": tensor_row["H1_amplitude_relative_error"],
            "H1_phase_absolute_error_deg": abs(tensor_row["H1_phase_python_minus_comsol_deg"]),
            "waveform_NRMSE": tensor_row["common_time_waveform_NRMSE"],
            "pass": tensor_row["H1_amplitude_relative_error"] <= 0.10 and abs(tensor_row["H1_phase_python_minus_comsol_deg"]) <= 10.0,
        }
        if signal not in baseline_rows: continue
        b, t = baseline_rows[signal], rows[signal]
        amp_delta = t["H1_amplitude_relative_error"] - b["H1_amplitude_relative_error"]
        phase_delta = abs(t["H1_phase_python_minus_comsol_deg"]) - abs(b["H1_phase_python_minus_comsol_deg"])
        limit_amp = max(0.002, 0.10 * b["H1_amplitude_relative_error"])
        failure = amp_delta > limit_amp or phase_delta > 0.2
        degradation.append({"signal": signal, "amplitude_error_delta": amp_delta, "amplitude_limit": limit_amp, "phase_abs_error_delta_deg": phase_delta, "phase_limit_deg": 0.2, "pass": not failure})
    gates = {
        **research,
        "engineering_main_quantities": {"measured": engineering, "pass": all(item.get("pass", False) for item in engineering.values())},
        "engineering_no_material_degradation": {"measured": degradation, "pass": all(item["pass"] for item in degradation)},
    }
    all_pass = all(bool(item.get("pass", False)) for item in gates.values())
    result = {"schema_version": 1, "status": "PASS" if all_pass else "FAIL", "gates": gates, "evidence": {"comparison_summary": str(out / "comparison_summary.json"), "comparison_metrics": str(out / "comparison_metrics.csv"), "tensor_summary": str(tensor / "summary.json"), "baseline_summary": str(BASELINE_SUMMARY)}, "production_decision": "promote" if all_pass else "retain diagnostic", "reference_response_used_for_fit": False}
    (out / "gate_decision.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def build_manifest(out: Path) -> None:
    files = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json": files.append({"path": str(path.relative_to(out)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    (out / "artifact_manifest.json").write_text(json.dumps({"schema_version": 1, "generated_by": "tools/build_tensor_comparison.py", "files": files}, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="compare tensor transient against refined COMSOL")
    parser.add_argument("--tensor", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=ROOT / "runs/transient_70Hz_nonlinear_comsol_physical_abc")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    invoke_common_compare(args.tensor, args.out)
    baseline = load_summary(BASELINE_SUMMARY); old = load_summary(OLD_COUPLED_SUMMARY); tensor_summary = load_summary(args.tensor / "summary.json")
    raw = pd.read_csv(args.out / "comparison_metrics.csv"); raw.to_csv(args.out / "comparison_metrics_raw.csv", index=False, float_format="%.12e")
    make_long_metrics(args.out, raw, baseline, old); make_harmonic(args.out, args.tensor, args.baseline); make_waveforms(args.out, args.tensor, args.baseline)
    raw_summary = load_summary(args.out / "comparison_summary.json")
    bl = metrics_by_signal(raw_summary).get("dynamic_BL_N_A", {})
    if bl and (bl.get("H1_amplitude_relative_error", 1) > .1 or abs(bl.get("H1_phase_python_minus_comsol_deg", 99)) > 10 or bl.get("common_time_waveform_NRMSE", 99) > .01): make_attribution(args.out, args.tensor)
    gate = build_gate(args.out, args.tensor, baseline, tensor_summary)
    raw_summary["gate_decision"] = gate; raw_summary["tensor_coenergy"] = True; (args.out / "comparison_summary.json").write_text(json.dumps(raw_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    build_manifest(args.out)
    print(json.dumps(gate, ensure_ascii=False, indent=2)); return 0 if gate["status"] == "PASS" else 2


if __name__ == "__main__": raise SystemExit(main())
