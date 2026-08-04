from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .model import TransientModel
from .solver import TransientResult


def _jsonable(value: Any):
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _spectrum(
    time_s: np.ndarray,
    values: np.ndarray,
    f0: float,
    harmonics: int,
    p_ref: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    centered = values - np.mean(values, axis=0, keepdims=True)
    columns = [np.ones_like(time_s)]
    for order in range(1, harmonics + 1):
        omega = 2.0 * math.pi * order * f0
        columns.extend([np.sin(omega * time_s), np.cos(omega * time_s)])
    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design, centered, rcond=None)
    rows = []
    harmonic_peaks = []
    for order in range(1, harmonics + 1):
        sin_coeff = coefficients[2 * order - 1]
        cos_coeff = coefficients[2 * order]
        amplitudes = np.hypot(sin_coeff, cos_coeff)
        harmonic_peaks.append(amplitudes)
        row = {"harmonic": order, "frequency_Hz": float(order * f0)}
        for name, amplitude in enumerate(amplitudes):
            row[f"probe_{name}_peak_Pa"] = float(amplitude)
            row[f"probe_{name}_SPL_dB"] = float(
                20.0 * math.log10(max(amplitude / math.sqrt(2.0), 1e-300) / p_ref)
            )
        rows.append(row)
    h = np.asarray(harmonic_peaks)
    thd = np.sqrt(np.sum(h[1:, :] ** 2, axis=0)) / np.maximum(h[0, :], 1e-300)
    return pd.DataFrame(rows), thd


def _cycle_convergence(values: np.ndarray, steps_per_period: int) -> np.ndarray:
    """Relative RMS difference between the last two complete cycles."""
    n = int(steps_per_period)
    if len(values) < 2 * n + 1:
        return np.full(values.shape[1:] or (), np.nan)
    previous = values[-2 * n - 1 : -n - 1]
    current = values[-n - 1 : -1]
    difference = np.sqrt(np.mean((current - previous) ** 2, axis=0))
    scale = np.sqrt(np.mean(current**2, axis=0))
    return difference / np.maximum(scale, 1e-300)


def export_result(model: TransientModel, result: TransientResult, outdir: str | Path) -> dict:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    float_format = model.config["export"].get("csv_float_format", "%.12e")
    probe_columns = {
        f"p_{name}_Pa": result.pressure_probes_Pa[:, i]
        for i, name in enumerate(model.probes.names)
    }
    time_frame = pd.DataFrame(
        {
            "time_s": result.time_s,
            "voltage_V": result.voltage_V,
            "current_A": result.current_A,
            "coil_displacement_m": result.coil_displacement_m,
            "coil_velocity_m_s": result.coil_velocity_m_s,
            "dynamic_BL_N_A": result.dynamic_BL_N_A,
            "incremental_inductance_H": result.incremental_inductance_H,
            "nonlinear_iterations": result.nonlinear_iterations,
            "ale_normalized_gap_margin": result.ale_normalized_gap_margin,
            "p_axis_1m_baffled_piston_Pa": result.farfield_axis_Pa,
            **probe_columns,
            "structural_kinetic_energy_J": result.energy_J[:, 0],
            "structural_strain_energy_J": result.energy_J[:, 1],
            "coil_magnetic_energy_J": result.energy_J[:, 2],
            "copper_loss_power_W": result.energy_J[:, 3],
            "structural_damping_power_W": result.energy_J[:, 4],
        }
    )
    if result.magnetic_flux_Wb is not None:
        time_frame["magnetic_flux_linkage_Wb"] = result.magnetic_flux_Wb
        time_frame["coenergy_J"] = result.coenergy_J
        time_frame["electromagnetic_force_N"] = result.electromagnetic_force_N
        time_frame["BL_tangent_N_A"] = result.tangent_BL_N_A
        time_frame["dforce_dx_N_m"] = result.dforce_dx_N_m
        time_frame["dforce_di_N_A"] = result.dforce_di_N_A
        time_frame["newton_residual"] = result.newton_residual
    time_path = out / "all_probes_timeseries.csv"
    time_frame.to_csv(time_path, index=False, float_format=float_format)

    extra_paths: list[Path] = []
    if result.magnetic_flux_Wb is not None:
        magnetic_frame = pd.DataFrame(
            {
                "t": result.time_s,
                "x": result.coil_displacement_m,
                "i": result.current_A,
                "F": result.electromagnetic_force_N,
                "psi": result.magnetic_flux_Wb,
                "W": result.coenergy_J,
                "BL_secant": result.dynamic_BL_N_A,
                "BL_tangent": result.tangent_BL_N_A,
                "L_incremental": result.incremental_inductance_H,
                "W_xx": result.dforce_dx_N_m,
                "W_xi": result.dforce_di_N_A,
                "Newton_iterations": result.nonlinear_iterations,
                "Newton_residual": result.newton_residual,
                "coordinate_margin": result.ale_normalized_gap_margin,
            }
        )
        magnetic_path = out / "magnetic_timeseries.csv"
        magnetic_frame.to_csv(magnetic_path, index=False, float_format=float_format)
        extra_paths.append(magnetic_path)
        balance = np.asarray(result.energy_balance_W, dtype=float)
        energy_frame = pd.DataFrame(
            {
                "t": result.time_s,
                "input_power_W": balance[:, 0],
                "copper_loss_W": balance[:, 1],
                "structural_damping_W": balance[:, 2],
                "electromagnetic_power_W": balance[:, 3],
                "mechanical_storage_J": balance[:, 4],
                "magnetic_storage_J": balance[:, 5],
                "coenergy_J": balance[:, 6],
                "discrete_balance_residual_W": balance[:, 7],
                # A deliberately labelled far-field proxy; the transient
                # balance is not treated as a closed acoustic energy law.
                "acoustic_farfield_power_proxy_W": (
                    result.farfield_axis_Pa**2
                    / (float(model.config["air"]["rho0_kg_m3"]) * float(model.config["air"]["c0_m_s"]))
                    * 4.0 * np.pi * float(model.config.get("exterior", {}).get("axis_distance_m", 1.0)) ** 2
                ),
            }
        )
        energy_path = out / "energy_balance_timeseries.csv"
        energy_frame.to_csv(energy_path, index=False, float_format=float_format)
        extra_paths.append(energy_path)

    f0 = float(model.config["drive"]["frequency_Hz"])
    steps_per_period = int(model.config["time"]["steps_per_period"])
    last_cycles = int(model.config["export"].get("last_cycles_for_spectrum", 1))
    count = last_cycles * steps_per_period + 1
    spectrum_values = np.column_stack(
        [result.pressure_probes_Pa[-count:], result.farfield_axis_Pa[-count:]]
    )
    spectrum, thd = _spectrum(
        result.time_s[-count:],
        spectrum_values,
        f0,
        int(model.config["export"].get("harmonics", 10)),
        float(model.config["air"]["p_ref_Pa"]),
    )
    spectrum_path = out / "spectrum_and_harmonics.csv"
    spectrum.to_csv(spectrum_path, index=False, float_format=float_format)

    snapshot_path = out / "field_snapshots.npz"
    np.savez_compressed(
        snapshot_path,
        time_s=result.snapshot_times_s,
        acoustic_points_rz_m=model.acoustic.mesh.points_rz_m[
            model.acoustic.acoustic_nodes_global
        ],
        pressure_Pa=result.pressure_snapshots_Pa,
        solid_points_rz_m=model.solid.points_rz_m,
        solid_displacement_m=result.solid_displacement_snapshots_m,
        coil_displacement_m=np.interp(
            result.snapshot_times_s, result.time_s, result.coil_displacement_m
        ),
        dynamic_BL_N_A=np.interp(
            result.snapshot_times_s, result.time_s, result.dynamic_BL_N_A
        ),
        incremental_inductance_H=np.interp(
            result.snapshot_times_s, result.time_s, result.incremental_inductance_H
        ),
    )
    summary = {
        "status": "completed",
        "model": model.metadata,
        "runtime": result.runtime,
        "drive": model.config["drive"],
        "time": model.config["time"],
        "probe_names": model.probes.names + ["axis_1m_baffled_piston"],
        "THD_first_probe": float(thd[0]),
        "THD_by_probe": {
            name: float(value)
            for name, value in zip(
                model.probes.names + ["axis_1m_baffled_piston"], thd
            )
        },
        "cycle_steady_state": {
            "definition": "relative RMS difference between final two complete cycles",
            "current": float(
                _cycle_convergence(result.current_A[:, None], steps_per_period)[0]
            ),
            "coil_displacement": float(
                _cycle_convergence(
                    result.coil_displacement_m[:, None], steps_per_period
                )[0]
            ),
            "pressure_by_probe": {
                name: float(value)
                for name, value in zip(
                    model.probes.names,
                    _cycle_convergence(
                        result.pressure_probes_Pa, steps_per_period
                    ),
                )
            },
        },
        "harmonic_crosscheck": result.harmonic,
        "tensor_coenergy": result.magnetic_flux_Wb is not None,
        "magnetic_diagnostics": (
            {
                "flux_min_max_Wb": [float(np.min(result.magnetic_flux_Wb)), float(np.max(result.magnetic_flux_Wb))],
                "coenergy_min_max_J": [float(np.min(result.coenergy_J)), float(np.max(result.coenergy_J))],
                "force_min_max_N": [float(np.min(result.electromagnetic_force_N)), float(np.max(result.electromagnetic_force_N))],
                "secant_BL_min_max_N_A": [float(np.min(result.dynamic_BL_N_A)), float(np.max(result.dynamic_BL_N_A))],
                "tangent_BL_min_max_N_A": [float(np.min(result.tangent_BL_N_A)), float(np.max(result.tangent_BL_N_A))],
                "incremental_L_min_max_H": [float(np.min(result.incremental_inductance_H)), float(np.max(result.incremental_inductance_H))],
                "newton_iterations_max": int(np.max(result.nonlinear_iterations)),
                "newton_iterations_mean": float(np.mean(result.nonlinear_iterations[1:])),
                "newton_residual_max": float(np.max(result.newton_residual)),
                "coordinate_margin_min": float(np.min(result.ale_normalized_gap_margin)),
                "energy_balance_residual_max_W": float(np.max(np.abs(result.energy_balance_W[:, 7]))),
            }
            if result.magnetic_flux_Wb is not None
            else None
        ),
        "scope": {
            "implemented": (
                "native tensor coenergy W(x,i), consistent Newton derivatives and moving-winding diagnostic ROM"
                if result.magnetic_flux_Wb is not None
                else "field-derived nonlinear magnetic coenergy and moving-coil ALE ROM"
                if model.nonlinear_law is not None
                else "linear transient structure-acoustic-electric coupling"
            ),
            "not_implemented": (
                [
                    "full-domain per-step magnetic remeshing",
                    "eddy-current magnetic auxiliary states",
                    "exact auxiliary-differential-equation transient PML",
                    "time-domain convolution form of narrow-region thermoviscous acoustics",
                ]
                if model.nonlinear_law is not None
                else [
                    "moving mesh and automatic remeshing",
                    "displacement-dependent nonlinear B-H/BL",
                    "exact auxiliary-differential-equation transient PML",
                    "time-domain convolution form of narrow-region thermoviscous acoustics",
                ]
            ),
        },
    }
    summary_path = out / "summary.json"
    summary_path.write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        path.name: {"bytes": path.stat().st_size, "sha256": _file_sha256(path)}
        for path in (time_path, spectrum_path, snapshot_path, summary_path, *extra_paths)
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return _jsonable(summary)
