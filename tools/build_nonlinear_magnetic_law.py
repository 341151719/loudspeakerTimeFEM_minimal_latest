#!/usr/bin/env python3
"""Rebuild BL(x) and lambda(i) from the native frequency-mainline magnetic FEM."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import meshio
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mainline",
        default=str(ROOT.parent / "00_MAINLINE/loudspeakerFEM_current_20260717"),
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "inputs/nonlinear_magnetic_law_20260728.json"),
    )
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--tol", type=float, default=1e-5)
    args = parser.parse_args()
    base = Path(args.mainline).resolve()
    sys.path[:0] = [str(base / "src"), str(base / "best_model")]
    from loudspeaker_axisym_fem.axisym_magnetics import (
        _assemble_frequency_matrices,
        _assemble_linear_system,
        _coil_area,
        _default_dirichlet_nodes,
        _element_fields,
        _flux_linkage_from_A,
        compute_bl_from_elements,
        effective_mu_r_from_B,
        load_tagged_meshio,
    )
    from loudspeaker_axisym_fem.comsol_driver_model import SOFT_IRON_BH_TABLE
    from p2_axisym_solid import build_p2_solid

    mesh_path = base / "inputs/meshes/comsol_geometry_polyline_coarse_2p5mm.msh"
    vtu_path = base / "inputs/comsol_reference/magnetostatic_converged_55iter.vtu"
    mesh = load_tagged_meshio(mesh_path)
    solid = build_p2_solid(mesh)
    vtu = meshio.read(vtu_path)
    triangles = next(
        np.asarray(block.data, int) for block in vtu.cells if block.type == "triangle"
    )
    xy = np.asarray(vtu.points[:, :2], float)
    centers = xy[triangles].mean(axis=1)
    fields = vtu.cell_data_dict
    Br = np.asarray(fields["B_r_T"]["triangle"], float)
    tree = cKDTree(centers)

    coil_elements = []
    coil_area = 0.0
    for connection, domain in zip(solid.triangles6, solid.domains):
        if int(domain) not in (17, 18, 19):
            continue
        points = solid.points_rz_m[connection[:3]]
        area = 0.5 * abs(
            float(
                np.linalg.det(
                    np.column_stack((points[1] - points[0], points[2] - points[0]))
                )
            )
        )
        coil_elements.append((points.mean(axis=0), area))
        coil_area += area

    shifts = np.linspace(-0.004, 0.004, 33)
    bl_samples = []
    for shift in shifts:
        integral = 0.0
        for center, area in coil_elements:
            query = center + np.array([0.0, shift])
            distance, index = tree.query(query, k=8)
            weight = 1.0 / np.maximum(distance, 2e-5) ** 2
            br = float(np.sum(weight * Br[index]) / np.sum(weight))
            integral += -100.0 * 2.0 * math.pi * query[0] * br * area
        bl_samples.append(integral / coil_area)
    coefficients = np.polynomial.chebyshev.chebfit(
        shifts / 0.004, np.asarray(bl_samples), 6
    )

    fixed = _default_dirichlet_nodes(
        mesh, (1, 2, 3, 4, 5, 83, 84, 85, 86, 87, 88, 89, 94)
    )
    soft = np.isin(mesh.tri_domains.astype(int), [6, 23])
    sigma = np.zeros(mesh.n_triangles)
    magnetic_coil_area = _coil_area(mesh, (17, 18, 19))
    current_values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    bl_current = []
    total_flux = []
    iterations = []
    timings = []
    for current in current_values:
        started = time.perf_counter()
        mu = np.ones(mesh.n_triangles)
        mu[soft] = 700.0
        potential = np.zeros(mesh.n_nodes)
        for iteration in range(1, args.max_iter + 1):
            K, permanent_rhs, free = _assemble_linear_system(
                mesh,
                mu,
                magnet_domains=(24,),
                remanence_T=0.4,
                dirichlet_nodes=fixed,
            )
            _, _, current_rhs, current_free, _ = _assemble_frequency_matrices(
                mesh,
                mu,
                sigma,
                coil_domains=(17, 18, 19),
                N0=100,
                unit_current_A=float(current),
                dirichlet_nodes=fixed,
            )
            if not np.array_equal(free, current_free):
                raise RuntimeError("magnetic free-DOF maps differ")
            updated = np.zeros(mesh.n_nodes)
            updated[free] = spsolve(K, permanent_rhs + current_rhs)
            residual = np.linalg.norm(updated - potential) / max(
                np.linalg.norm(updated), 1e-30
            )
            potential = updated
            Br_i, Bz_i, Bnorm_i, _ = _element_fields(mesh, potential, mu)
            target = mu.copy()
            target[soft] = np.clip(
                effective_mu_r_from_B(Bnorm_i[soft], SOFT_IRON_BH_TABLE),
                1.0,
                4000.0,
            )
            mu[soft] = 0.9 * mu[soft] + 0.1 * target[soft]
            if residual < args.tol and iteration >= 2:
                break
        bl_current.append(
            compute_bl_from_elements(
                mesh, Br_i, coil_domains=(17, 18, 19), N0=100
            )
        )
        total_flux.append(
            float(
                np.real(
                    _flux_linkage_from_A(
                        mesh,
                        potential,
                        coil_domains=(17, 18, 19),
                        N0=100,
                        coil_area_m2=magnetic_coil_area,
                    )
                )
            )
        )
        iterations.append(iteration)
        timings.append(time.perf_counter() - started)

    zero = int(np.where(current_values == 0.0)[0][0])
    current_flux = np.asarray(total_flux) - total_flux[zero]
    flux_coefficients_descending = np.polyfit(current_values, current_flux, 4)
    flux_coefficients_ascending = flux_coefficients_descending[::-1]
    # Force exact zero flux at zero current.
    flux_coefficients_ascending[0] = 0.0
    bl0 = float(bl_current[zero])
    center_value = float(np.polynomial.chebyshev.chebval(0.0, coefficients))
    coefficients[0] += bl0 - center_value
    result = {
        "kind": "field_derived_separable_magnetic_coenergy_ROM",
        "displacement_limit_m": 0.004,
        "bl_chebyshev_coordinate": "x/displacement_limit_m",
        "bl_chebyshev_coefficients_N_A": coefficients.tolist(),
        "current_limit_A": 1.0,
        "lambda_polynomial_coefficients_Wb_ascending": flux_coefficients_ascending.tolist(),
        "raw_displacement_scan": {
            "method": "8-neighbour inverse-distance interpolation of Br, shifted coil quadrature, followed by degree-6 Chebyshev compression",
            "shift_mm": (1000.0 * shifts).tolist(),
            "BL_N_A": list(map(float, bl_samples)),
        },
        "raw_current_scan": {
            "method": "native nonlinear B_inverse magnetostatic FEM with permanent magnet plus homogenized 100-turn coil current",
            "current_A": current_values.tolist(),
            "BL_N_A": list(map(float, bl_current)),
            "total_flux_linkage_Wb": total_flux,
            "current_flux_linkage_Wb": current_flux.tolist(),
            "iterations_each": iterations,
            "seconds_each": timings,
            "residual_target": args.tol,
        },
        "coenergy_contract": {
            "force": "F = BL(x)*i",
            "motional_emf": "e = d/dt integral_0^x BL(s) ds",
            "inductive_voltage": "d(lambda_i(i))/dt",
            "note": "BL(x) and lambda_i(i) are kept separable so force and back-EMF remain energy conjugate",
        },
    }
    Path(args.out).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
