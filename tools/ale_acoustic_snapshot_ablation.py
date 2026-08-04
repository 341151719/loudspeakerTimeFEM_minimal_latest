#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

from loudspeaker_time_fem.comsol_mesh import ComsolMesh
from loudspeaker_time_fem.config import load_config
from loudspeaker_time_fem.model import (
    _build_probe_map,
    _outer_radiation_matrices,
    build_transient_model,
)
from loudspeaker_time_fem.native_acoustic import build_native_acoustic


def harmonic(time: np.ndarray, values: np.ndarray, frequency: float) -> np.ndarray:
    omega = 2.0 * np.pi * frequency
    design = np.column_stack(
        (np.ones_like(time), np.sin(omega * time), np.cos(omega * time))
    )
    coefficient, *_ = np.linalg.lstsq(design, values, rcond=None)
    return coefficient[2] - 1j * coefficient[1]


def morphed_mesh(model, solid_displacement: np.ndarray, outer_radius: float) -> ComsolMesh:
    acoustic = model.acoustic
    mesh = acoustic.mesh
    selected = set(acoustic.acoustic_domains)
    interfaces = mesh.acoustic_interface_edges(selected)
    interface_nodes = sorted({node for edge in interfaces for node in edge[:2]})
    acoustic_points = mesh.points_rz_m
    solid_tree = cKDTree(model.solid.points_rz_m)
    _distance, nearest = solid_tree.query(acoustic_points[interface_nodes])
    boundary_displacement = solid_displacement[nearest]
    radius = np.linalg.norm(acoustic_points, axis=1)
    outer_nodes = np.flatnonzero(np.abs(radius - outer_radius) < 3e-4)
    axis_nodes = np.flatnonzero(np.abs(acoustic_points[:, 0]) < 1e-12)
    fixed = np.unique(
        np.concatenate((np.asarray(interface_nodes), outer_nodes, axis_nodes))
    )
    fixed_local = np.array(
        [acoustic.acoustic_node_map[int(node)] for node in fixed if int(node) in acoustic.acoustic_node_map]
    )
    prescribed = np.zeros((len(acoustic.acoustic_nodes_global), 2))
    for node, value in zip(interface_nodes, boundary_displacement):
        if node in acoustic.acoustic_node_map:
            prescribed[acoustic.acoustic_node_map[node]] = value
    for node in axis_nodes:
        if int(node) in acoustic.acoustic_node_map:
            prescribed[acoustic.acoustic_node_map[int(node)]] = 0.0
    all_local = np.arange(len(acoustic.acoustic_nodes_global))
    free = np.setdiff1d(all_local, fixed_local)
    stiffness = acoustic.Kp.tocsc()
    if len(free):
        solve = splu(stiffness[free][:, free])
        for component in range(2):
            prescribed[free, component] = solve.solve(
                -(stiffness[free][:, fixed_local] @ prescribed[fixed_local, component])
            )
    points = acoustic_points.copy()
    points[acoustic.acoustic_nodes_global] += prescribed
    return ComsolMesh(points, mesh.cells, mesh.entities)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--interface", type=Path, required=True)
    parser.add_argument("--comsol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    model = build_transient_model(config, config_path)
    snapshots = np.load(args.snapshots)
    interface = np.load(args.interface)
    displacement = snapshots["solid_displacement_m"]
    coil = snapshots["coil_displacement_m"]
    cases = {"zero": None, "negative": int(np.argmin(coil)), "positive": int(np.argmax(coil))}
    frequency = float(config["drive"]["frequency_Hz"])
    omega = 2.0 * np.pi * frequency
    source = interface["acoustic_source_H1"]
    long = pd.read_csv(args.comsol / "pressure_points_timeseries.csv")
    frame = long.pivot_table(index="time_s", columns="probe_name", values="p_Pa", aggfunc="last")
    time = frame.index.to_numpy()
    mask = (time >= 3 / frequency - 1e-10) & (time < 4 / frequency - 1e-10)
    mapping = {
        "axis_near_0p10m": "python_axis_near_actual",
        "offaxis_45deg_0p10m": "python_offaxis_actual",
        "rear_physical_m0p10": "common_rear_physical_m0p10",
    }
    evaluation_config = copy.deepcopy(config)
    evaluation_config["probes"] = [
        item for item in config["probes"] if item["name"] in mapping
    ]
    truth = {name: harmonic(time[mask], frame[column].to_numpy()[mask], frequency) for name, column in mapping.items()}
    rows = []
    contract = config["acoustic_contract"]
    absorbing = config["absorbing_layer"]
    for label, index in cases.items():
        mesh_override = None if index is None else morphed_mesh(
            model, displacement[index], float(absorbing["outer_radius_m"])
        )
        acoustic = build_native_acoustic(
            config_path.parent.parent / contract["mesh"],
            set(contract["acoustic_domains"]),
            set(contract["pml_domains"]),
            int(contract.get("uniform_refinement_levels", 0)),
            int(contract.get("pressure_order", 1)),
            mesh_override,
        )
        pf = np.arange(len(acoustic.acoustic_nodes_global))
        probes = _build_probe_map(evaluation_config, acoustic, pf)
        damping, boundary_stiffness, _ = _outer_radiation_matrices(
            acoustic, pf, float(absorbing["outer_radius_m"]),
            float(config["air"]["c0_m_s"]), True,
        )
        dynamic = (
            acoustic.Kp + boundary_stiffness
            - omega**2 * acoustic.Mp / float(config["air"]["c0_m_s"])**2
            + 1j * omega * damping
        ).tocsc()
        solved = splu(dynamic).solve(source)
        probe_values = dict(zip(probes.names, np.asarray(probes.pressure_matrix @ solved)))
        minimum_area_ratio = 1.0
        if index is not None:
            base = model.acoustic.mesh.points_rz_m[model.acoustic.triangles_global[:, :3]]
            moved = acoustic.mesh.points_rz_m[acoustic.triangles_global[:, :3]]
            def area(p):
                first = p[:, 1] - p[:, 0]
                second = p[:, 2] - p[:, 0]
                return np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
            minimum_area_ratio = float(np.min(area(moved) / area(base)))
        for name, reference in truth.items():
            value = probe_values[name]
            rows.append({
                "case": label, "coil_displacement_m": 0.0 if index is None else float(coil[index]),
                "minimum_area_ratio": minimum_area_ratio, "probe": name,
                "amplitude_relative_error": float(abs(abs(value)-abs(reference))/abs(reference)),
                "phase_error_deg": float(np.degrees(np.angle(value/reference))),
            })
    result = {"status": "completed", "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
