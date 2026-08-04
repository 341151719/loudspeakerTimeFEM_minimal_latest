#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.linalg import splu

from loudspeaker_time_fem.config import load_config
from loudspeaker_time_fem.model import build_transient_model
from loudspeaker_time_fem.solver import _harmonic_solution


def fit_harmonic(time: np.ndarray, values: np.ndarray, frequency: float) -> complex:
    omega = 2.0 * np.pi * frequency
    design = np.column_stack(
        (np.ones_like(time), np.sin(omega * time), np.cos(omega * time))
    )
    coefficient, *_ = np.linalg.lstsq(design, values, rcond=None)
    return complex(coefficient[2] - 1j * coefficient[1])


def p2_edge_shape(t: float) -> np.ndarray:
    return np.array(
        [(1.0 - t) * (1.0 - 2.0 * t), t * (2.0 * t - 1.0), 4.0 * t * (1.0 - t)]
    )


def project_python_motion(model, quadrature: pd.DataFrame, solid_h1: np.ndarray) -> np.ndarray:
    tags = set(model.metadata["G"]["reference_interface_boundaries"])
    candidates = []
    for va, vb, tag in model.solid.boundary_edges:
        if int(tag) not in tags:
            continue
        midpoint = model.solid.edge_mid_nodes[tuple(sorted((int(va), int(vb))))]
        candidates.append(
            (
                int(va),
                int(vb),
                int(midpoint),
                model.solid.points_rz_m[int(va)],
                model.solid.points_rz_m[int(vb)],
            )
        )
    full = np.zeros(model.solid.ndof, dtype=complex)
    full[model.solid_free_dofs] = solid_h1
    vector = full.reshape(-1, 2)
    out = np.zeros(len(quadrature), dtype=complex)
    for row in quadrature.itertuples(index=False):
        point = np.array([row.r_m, row.z_m])
        best = None
        for va, vb, midpoint, q0, q1 in candidates:
            direction = q1 - q0
            tau = float(
                np.clip(
                    np.dot(point - q0, direction)
                    / max(float(np.dot(direction, direction)), 1e-30),
                    0.0,
                    1.0,
                )
            )
            distance = float(np.linalg.norm(point - (q0 + tau * direction)))
            if best is None or distance < best[0]:
                best = (distance, va, vb, midpoint, tau)
        _, va, vb, midpoint, tau = best
        displacement = p2_edge_shape(tau) @ vector[[va, vb, midpoint]]
        out[int(row.sample_id)] = displacement @ np.array([row.normal_r, row.normal_z])
    return out


def weighted_metrics(reference: np.ndarray, candidate: np.ndarray, weight: np.ndarray) -> dict:
    norm_reference = np.sqrt(np.sum(weight * np.abs(reference) ** 2))
    norm_candidate = np.sqrt(np.sum(weight * np.abs(candidate) ** 2))
    difference = np.sqrt(np.sum(weight * np.abs(candidate - reference) ** 2))
    inner = np.sum(weight * np.conj(reference) * candidate)
    denominator = max(norm_reference * norm_candidate, 1e-30)
    return {
        "reference_rms_m": float(norm_reference / np.sqrt(np.sum(weight))),
        "candidate_rms_m": float(norm_candidate / np.sqrt(np.sum(weight))),
        "relative_complex_L2_error": float(difference / max(norm_reference, 1e-30)),
        "complex_correlation_magnitude": float(abs(inner) / denominator),
        "correlation_phase_deg": float(np.degrees(np.angle(inner))),
    }


def assemble_source(model, quadrature: pd.DataFrame, displacement: np.ndarray) -> np.ndarray:
    omega = 2.0 * np.pi * float(model.config["drive"]["frequency_Hz"])
    rho0 = float(model.config["air"]["rho0_kg_m3"])
    source = np.zeros(model.n_pressure, dtype=complex)
    amap = model.acoustic.acoustic_node_map
    lookup = {int(base): local for local, base in enumerate(model.pressure_free_dofs)}
    for row in quadrature.itertuples(index=False):
        value = rho0 * omega**2 * row.axisym_weight_m2 * displacement[row.sample_id]
        source[lookup[amap[row.pressure_global_a]]] += row.pressure_shape_a * value
        source[lookup[amap[row.pressure_global_b]]] += row.pressure_shape_b * value
    return source


def complex_columns(prefix: str, value: complex) -> dict[str, float]:
    return {
        f"{prefix}_real": float(np.real(value)),
        f"{prefix}_imag": float(np.imag(value)),
        f"{prefix}_magnitude": float(np.abs(value)),
        f"{prefix}_phase_deg": float(np.degrees(np.angle(value))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--quadrature", type=Path, required=True)
    parser.add_argument("--comsol-interface", type=Path, required=True)
    parser.add_argument("--comsol-global", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    model = build_transient_model(config, config_path)
    quadrature = pd.read_csv(args.quadrature).sort_values("sample_id")
    interface = np.load(args.comsol_interface)
    comsol_motion = interface["normal_displacement_H1_m"]

    frequency = float(config["drive"]["frequency_Hz"])
    voltage = float(config["drive"]["voltage_peak_V"])
    harmonic_x, harmonic_i = _harmonic_solution(model, frequency, voltage)
    # Convert cosine-voltage solution to the sine-drive convention used by the
    # exported COMSOL time traces.
    python_solid = -1j * harmonic_x[: model.n_solid]
    python_motion_raw = project_python_motion(model, quadrature, python_solid)

    global_data = pd.read_csv(args.comsol_global)
    period = 1.0 / frequency
    mask = (global_data.time_s >= 3.0 * period - 1e-10) & (
        global_data.time_s < 4.0 * period - 1e-10
    )
    comsol_coil = fit_harmonic(
        global_data.time_s.to_numpy()[mask],
        global_data.coil_displacement_m.to_numpy()[mask],
        frequency,
    )
    bl = max(abs(float(model.metadata["BL_axial_N_per_A"])), 1e-30)
    python_coil = complex(-1j * (model.back_emf_vector @ harmonic_x[: model.n_solid]) / bl)
    coil_scale = comsol_coil / python_coil
    python_motion = python_motion_raw * coil_scale
    weight = quadrature.axisym_weight_m2.to_numpy()

    ns = model.n_solid
    omega = 2.0 * np.pi * frequency
    dynamic = (
        model.K[ns:, ns:].astype(complex)
        - omega**2 * model.M[ns:, ns:].astype(complex)
        + 1j * omega * model.C[ns:, ns:].astype(complex)
    ).tocsc()
    factor = splu(dynamic)
    probe_names = model.probes.names

    rows = []
    for boundary, samples in quadrature.groupby("boundary_entity", sort=True):
        ids = samples.sample_id.to_numpy(dtype=int)
        boundary_weight = weight[ids]
        metrics = weighted_metrics(
            comsol_motion[ids], python_motion[ids], boundary_weight
        )
        mask_samples = np.zeros(len(quadrature), dtype=complex)
        mask_samples[ids] = comsol_motion[ids]
        comsol_probe = np.asarray(
            model.probes.pressure_matrix @ factor.solve(assemble_source(model, quadrature, mask_samples))
        )
        mask_samples[ids] = python_motion[ids]
        python_probe = np.asarray(
            model.probes.pressure_matrix @ factor.solve(assemble_source(model, quadrature, mask_samples))
        )
        row = {
            "boundary_entity": int(boundary),
            "sample_count": int(len(ids)),
            "axisym_area_m2": float(np.sum(boundary_weight)),
            **metrics,
        }
        for name, cvalue, pvalue in zip(probe_names, comsol_probe, python_probe):
            row.update(complex_columns(f"{name}_comsol_motion_pressure_Pa", cvalue))
            row.update(complex_columns(f"{name}_python_motion_pressure_Pa", pvalue))
        rows.append(row)

    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out.with_suffix(".csv")
    table.to_csv(csv_path, index=False)
    global_metrics = weighted_metrics(comsol_motion, python_motion, weight)
    source_comsol = assemble_source(model, quadrature, comsol_motion)
    source_python = assemble_source(model, quadrature, python_motion)
    probe_comsol = np.asarray(model.probes.pressure_matrix @ factor.solve(source_comsol))
    probe_python = np.asarray(model.probes.pressure_matrix @ factor.solve(source_python))
    result = {
        "status": "completed",
        "comparison": "COMSOL versus Python interface normal displacement H1",
        "normalization": "Python interface motion complex-scaled to COMSOL coil displacement H1",
        "coil_displacement_H1_m": {
            "comsol": {"real": comsol_coil.real, "imag": comsol_coil.imag, "magnitude": abs(comsol_coil)},
            "python_raw": {"real": python_coil.real, "imag": python_coil.imag, "magnitude": abs(python_coil)},
            "python_to_comsol_complex_scale": {
                "real": coil_scale.real,
                "imag": coil_scale.imag,
                "magnitude": abs(coil_scale),
                "phase_deg": float(np.degrees(np.angle(coil_scale))),
            },
        },
        "global_interface_metrics": global_metrics,
        "full_probe_pressure_from_interface_motion": {
            name: {
                "comsol_motion_magnitude_Pa": float(abs(cvalue)),
                "python_motion_magnitude_Pa": float(abs(pvalue)),
                "python_over_comsol_magnitude": float(abs(pvalue) / max(abs(cvalue), 1e-30)),
                "python_minus_comsol_phase_deg": float(np.degrees(np.angle(pvalue / cvalue))),
            }
            for name, cvalue, pvalue in zip(probe_names, probe_comsol, probe_python)
        },
        "per_boundary_csv": str(csv_path),
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
