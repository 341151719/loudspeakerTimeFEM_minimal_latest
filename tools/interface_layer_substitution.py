#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

from loudspeaker_time_fem.config import load_config
from loudspeaker_time_fem.model import build_transient_model


def fit_harmonic(time: np.ndarray, values: np.ndarray, frequency: float) -> np.ndarray:
    omega = 2.0 * np.pi * frequency
    design = np.column_stack(
        (np.ones_like(time), np.sin(omega * time), np.cos(omega * time))
    )
    coefficient, *_ = np.linalg.lstsq(design, values, rcond=None)
    return coefficient[2] - 1j * coefficient[1]


def interpolate_trace(
    coordinates: np.ndarray, values: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    tree = cKDTree(coordinates)
    count = min(4, len(coordinates))
    distance, index = tree.query(targets, k=count)
    if count == 1:
        return values[np.asarray(index)]
    exact = distance[:, 0] < 1e-12
    weights = 1.0 / np.maximum(distance, 1e-12) ** 2
    weights /= weights.sum(axis=1, keepdims=True)
    result = np.sum(weights * values[index], axis=1)
    result[exact] = values[index[exact, 0]]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--quadrature", type=Path, required=True)
    parser.add_argument("--comsol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    model = build_transient_model(config, config_path)
    motion = pd.read_csv(args.motion)
    quadrature = pd.read_csv(args.quadrature).sort_values("sample_id")
    times = np.sort(motion.time_s.unique())
    sample_displacement = np.zeros((len(times), len(quadrature)), dtype=float)
    trace_choice: dict[int, str] = {}

    for boundary, samples in quadrature.groupby("boundary_entity", sort=False):
        boundary_motion = motion[motion.boundary_entity == boundary]
        up_scale = max(
            boundary_motion.up_displacement_r_m.abs().max(),
            boundary_motion.up_displacement_z_m.abs().max(),
        )
        down_scale = max(
            boundary_motion.down_displacement_r_m.abs().max(),
            boundary_motion.down_displacement_z_m.abs().max(),
        )
        prefix = "up" if up_scale >= down_scale else "down"
        trace_choice[int(boundary)] = prefix
        targets = samples[["r_m", "z_m"]].to_numpy()
        normals = samples[["normal_r", "normal_z"]].to_numpy()
        sample_ids = samples.sample_id.to_numpy(dtype=int)
        for time_index, time in enumerate(times):
            frame = boundary_motion[np.isclose(boundary_motion.time_s, time)]
            coordinates = frame[["R_m", "Z_m"]].to_numpy()
            radial = interpolate_trace(
                coordinates,
                frame[f"{prefix}_displacement_r_m"].to_numpy(),
                targets,
            )
            axial = interpolate_trace(
                coordinates,
                frame[f"{prefix}_displacement_z_m"].to_numpy(),
                targets,
            )
            sample_displacement[time_index, sample_ids] = (
                normals[:, 0] * radial + normals[:, 1] * axial
            )

    f0 = float(config["drive"]["frequency_Hz"])
    displacement_h1 = fit_harmonic(times, sample_displacement, f0)
    omega = 2.0 * np.pi * f0
    rho0 = float(config["air"]["rho0_kg_m3"])
    source = np.zeros(model.n_pressure, dtype=complex)
    amap = model.acoustic.acoustic_node_map
    free_lookup = {
        int(base): local for local, base in enumerate(model.pressure_free_dofs)
    }
    for row in quadrature.itertuples(index=False):
        value = rho0 * omega * omega * row.axisym_weight_m2 * displacement_h1[row.sample_id]
        edge = tuple(sorted((int(row.pressure_global_a), int(row.pressure_global_b))))
        midpoint = getattr(model.acoustic, "edge_midpoint_nodes", {}).get(edge)
        if midpoint is None:
            weighted_nodes = (
                (int(row.pressure_global_a), float(row.pressure_shape_a)),
                (int(row.pressure_global_b), float(row.pressure_shape_b)),
            )
        else:
            t = float(row.quadrature_t)
            if getattr(model.acoustic, "pressure_order", 1) == 2:
                weighted_nodes = (
                    (int(row.pressure_global_a), (1.0 - t) * (1.0 - 2.0 * t)),
                    (int(row.pressure_global_b), t * (2.0 * t - 1.0)),
                    (int(midpoint), 4.0 * t * (1.0 - t)),
                )
            elif t <= 0.5:
                weighted_nodes = (
                    (int(row.pressure_global_a), 1.0 - 2.0 * t),
                    (int(midpoint), 2.0 * t),
                )
            else:
                weighted_nodes = (
                    (int(row.pressure_global_b), 2.0 * t - 1.0),
                    (int(midpoint), 2.0 * (1.0 - t)),
                )
        for global_node, shape in weighted_nodes:
            source[free_lookup[amap[global_node]]] += shape * value

    ns = model.n_solid
    mass = model.M[ns:, ns:].astype(complex)
    damping = model.C[ns:, ns:].astype(complex)
    stiffness = model.K[ns:, ns:].astype(complex)
    dynamic = (stiffness - omega * omega * mass + 1j * omega * damping).tocsc()
    pressure_h1 = splu(dynamic).solve(source)
    probe_h1 = np.asarray(model.probes.pressure_matrix @ pressure_h1)

    comsol_long = pd.read_csv(args.comsol / "pressure_points_timeseries.csv")
    comsol = comsol_long.pivot_table(
        index="time_s", columns="probe_name", values="p_Pa", aggfunc="last"
    ).sort_index()
    t = comsol.index.to_numpy()
    mask = (t >= 3.0 / f0 - 1e-10) & (t < 4.0 / f0 - 1e-10)
    mapping = {
        "axis_near_0p10m": "python_axis_near_actual",
        "axis_rear_m0p12m": "python_axis_rear_actual",
        "offaxis_45deg_0p10m": "python_offaxis_actual",
        "comsol_native_point6": None,
        "rear_physical_m0p10": "common_rear_physical_m0p10",
    }
    rows = []
    for index, name in enumerate(model.probes.names):
        reference_name = mapping.get(name)
        if reference_name is None:
            continue
        reference = fit_harmonic(t[mask], comsol[reference_name].to_numpy()[mask], f0)
        candidate = probe_h1[index]
        rows.append(
            {
                "probe": name,
                "comsol_probe": reference_name,
                "comsol_H1_peak_Pa": abs(reference),
                "substitution_H1_peak_Pa": abs(candidate),
                "amplitude_relative_error": abs(abs(candidate) - abs(reference))
                / abs(reference),
                "phase_substitution_minus_comsol_deg": float(
                    np.degrees(np.angle(candidate / reference))
                ),
            }
        )
    volume_displacement = np.sum(
        quadrature.axisym_weight_m2.to_numpy() * displacement_h1
    )
    result = {
        "status": "completed",
        "substitution": "COMSOL interface displacement H1 -> Python native physical-domain acoustic operator",
        "trace_choice_by_boundary": trace_choice,
        "interface_volume_displacement_H1_m3": {
            "real": float(volume_displacement.real),
            "imag": float(volume_displacement.imag),
            "magnitude": float(abs(volume_displacement)),
        },
        "metrics": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    npz_path = args.out.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        sample_id=quadrature.sample_id.to_numpy(dtype=int),
        normal_displacement_H1_m=displacement_h1,
        acoustic_source_H1=source,
        pressure_H1_Pa=pressure_h1,
        probe_H1_Pa=probe_h1,
    )
    result["compressed_interface_h1"] = str(npz_path)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
