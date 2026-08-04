#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from scipy.special import eval_legendre

from loudspeaker_time_fem.config import load_config
from loudspeaker_time_fem.model import build_transient_model
from loudspeaker_time_fem.spherical_nrbc import outgoing_modal_impedance


def fit_harmonic(time: np.ndarray, values: np.ndarray, frequency: float) -> complex:
    omega = 2.0 * np.pi * frequency
    design = np.column_stack(
        (np.ones_like(time), np.sin(omega * time), np.cos(omega * time))
    )
    coefficient, *_ = np.linalg.lstsq(design, values, rcond=None)
    return complex(coefficient[2] - 1j * coefficient[1])


def boundary_operators(model, radius: float, frequency: float, lmax: int):
    """Return the local first-order ABC and exact axisymmetric spherical DtN."""
    acoustic = model.acoustic
    amap = acoustic.acoustic_node_map
    lookup = {int(base): local for local, base in enumerate(model.pressure_free_dofs)}
    boundary_nodes = set()
    segments = []
    tolerance = max(2e-5, 2e-3 * radius)
    for ga, gb in acoustic.mesh.line_cells:
        ga, gb = int(ga), int(gb)
        if ga not in amap or gb not in amap:
            continue
        p0, p1 = acoustic.mesh.points_rz_m[[ga, gb]]
        if max(abs(np.linalg.norm(p0) - radius), abs(np.linalg.norm(p1) - radius)) > tolerance:
            continue
        a, b = int(amap[ga]), int(amap[gb])
        if a not in lookup or b not in lookup:
            continue
        ia, ib = lookup[a], lookup[b]
        boundary_nodes.update((ia, ib))
        segments.append((ia, ib, p0, p1))
    nodes = np.array(sorted(boundary_nodes), dtype=int)
    node_lookup = {node: index for index, node in enumerate(nodes)}
    mass = np.zeros((len(nodes), len(nodes)))
    moment = np.zeros((len(nodes), lmax + 1))
    xg, wg = np.polynomial.legendre.leggauss(max(8, lmax + 2))
    for ia, ib, p0, p1 in segments:
        length = float(np.linalg.norm(p1 - p0))
        local = [node_lookup[ia], node_lookup[ib]]
        for xi, weight in zip(xg, wg):
            t = 0.5 * (xi + 1.0)
            shape = np.array([1.0 - t, t])
            point = (1.0 - t) * p0 + t * p1
            surface_weight = (
                2.0 * np.pi * max(float(point[0]), 1e-12) * length * 0.5 * weight
            )
            cos_theta = float(point[1] / radius)
            mass[np.ix_(local, local)] += surface_weight * np.outer(shape, shape)
            for order in range(lmax + 1):
                moment[local, order] += (
                    surface_weight * shape * eval_legendre(order, cos_theta)
                )
    omega = 2.0 * np.pi * frequency
    c0 = float(model.config["air"]["c0_m_s"])
    k = omega / c0
    x = k * radius
    order = np.arange(lmax + 1)
    # exp(+i wt) convention => outgoing radial wave is hankel of second kind.
    dtn_added = outgoing_modal_impedance(
        frequency, radius, c0, int(lmax)
    )
    modal_norm = 4.0 * np.pi * radius**2 / (2.0 * order + 1.0)
    first = (1.0 / radius + 1j * k) * mass
    # Retain the local l=0 ABC on the unresolved trace complement and apply
    # exact modal corrections only to the explicitly resolved harmonics.
    exact = first + (moment * ((dtn_added - dtn_added[0]) / modal_norm)) @ moment.T
    return nodes, first, exact, dtn_added


def sparse_boundary_delta(size: int, nodes: np.ndarray, delta: np.ndarray):
    row, col = np.meshgrid(nodes, nodes, indexing="ij")
    return coo_matrix((delta.ravel(), (row.ravel(), col.ravel())), shape=(size, size)).tocsc()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--interface", type=Path, required=True)
    parser.add_argument("--comsol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--orders", default="0,1,2,4,8,12,16")
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    model = build_transient_model(config, config_path)
    frequency = float(config["drive"]["frequency_Hz"])
    omega = 2.0 * np.pi * frequency
    radius = float(config["absorbing_layer"]["outer_radius_m"])
    interface = np.load(args.interface)
    source = interface["acoustic_source_H1"]
    ns = model.n_solid
    base = (
        model.K[ns:, ns:].astype(complex)
        - omega**2 * model.M[ns:, ns:].astype(complex)
        + 1j * omega * model.C[ns:, ns:].astype(complex)
    ).tocsc()

    long = pd.read_csv(args.comsol / "pressure_points_timeseries.csv")
    pressure = long.pivot_table(
        index="time_s", columns="probe_name", values="p_Pa", aggfunc="last"
    ).sort_index()
    time = pressure.index.to_numpy()
    period = 1.0 / frequency
    use = (time >= 3.0 * period - 1e-10) & (time < 4.0 * period - 1e-10)
    mapping = {
        "axis_near_0p10m": "python_axis_near_actual",
        "offaxis_45deg_0p10m": "python_offaxis_actual",
        "rear_physical_m0p10": "common_rear_physical_m0p10",
    }
    reference = {
        name: fit_harmonic(time[use], pressure[column].to_numpy()[use], frequency)
        for name, column in mapping.items()
    }
    rows = []
    for order in [int(value) for value in args.orders.split(",")]:
        nodes, first, exact, eigenvalues = boundary_operators(
            model, radius, frequency, order
        )
        dynamic = base + sparse_boundary_delta(
            model.n_pressure, nodes, exact - first
        )
        solved = splu(dynamic).solve(source)
        probes = np.asarray(model.probes.pressure_matrix @ solved)
        probe_map = dict(zip(model.probes.names, probes))
        row = {
            "lmax": order,
            "boundary_dofs": int(len(nodes)),
            "dtn_l0_real_1_per_m": float(eigenvalues[0].real),
            "dtn_l0_imag_1_per_m": float(eigenvalues[0].imag),
        }
        for name, value in reference.items():
            candidate = probe_map[name]
            row[f"{name}_amplitude_relative_error"] = float(
                abs(abs(candidate) - abs(value)) / abs(value)
            )
            row[f"{name}_phase_error_deg"] = float(
                np.degrees(np.angle(candidate / value))
            )
        rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out.with_suffix(".csv"), index=False)
    result = {
        "status": "completed",
        "purpose": "frequency-only exact spherical DtN diagnostic; not the transient production boundary",
        "frequency_Hz": frequency,
        "radius_m": radius,
        "rows": rows,
        "csv": str(args.out.with_suffix(".csv")),
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
