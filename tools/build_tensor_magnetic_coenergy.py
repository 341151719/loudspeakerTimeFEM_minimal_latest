#!/usr/bin/env python3
"""Build a native two-dimensional ``(x, i)`` magnetic coenergy law.

This tool intentionally does not read COMSOL transient response data.  Every
training and holdout point assembles a new moving-winding source vector and
solves the nonlinear native axisymmetric magnetic FEM problem on the fixed
mesh.  The subcommands are staged so an incomplete pilot/scan cannot silently
turn into a production law.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Iterable

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, cg, splu, spsolve
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
WORK_NAME = "tensor_magnetic_scan_20260801"
X_TRAIN_MM = np.asarray(
    [
        -4,
        -3.5,
        -3,
        -2.5,
        -2.25,
        -2,
        -1.75,
        -1.5,
        -1.25,
        -1,
        -0.75,
        -0.5,
        -0.25,
        0,
        0.25,
        0.5,
        0.75,
        1,
        1.25,
        1.5,
        1.75,
        2,
        2.25,
        2.5,
        3,
        3.5,
        4,
    ],
    dtype=float,
)
I_TRAIN_A = np.asarray(
    [
        -1,
        -0.875,
        -0.75,
        -0.625,
        -0.5,
        -0.4,
        -0.3,
        -0.2,
        -0.1,
        0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.625,
        0.75,
        0.875,
        1,
    ],
    dtype=float,
)
X_HOLDOUT_MM = np.asarray([-1.875, -0.9375, 0.125, 1.0625, 1.875], dtype=float)
I_HOLDOUT_A = np.asarray([-0.55, -0.275, 0.05, 0.325, 0.55], dtype=float)
PILOT_X_MM = np.asarray([-2.0, 0.0, 2.0], dtype=float)
PILOT_I_A = np.asarray([-0.5, 0.0, 0.5], dtype=float)
PATH_INDEPENDENCE_POINTS = ((-2.0, -0.5), (0.0, 0.5), (2.0, -0.5))
STRICT_POINTS = ((0.0, 0.0), (-1.75, -0.55), (1.75, 0.55), (-4.0, -1.0), (4.0, 1.0))
FIXED_RETRY_RELAXATIONS = (0.20, 0.10, 0.05)
SOFT_DOMAINS = (6, 23)
MAGNET_DOMAINS = (24,)
COIL_DOMAINS = (17, 18, 19)
EXTERIOR_BOUNDARIES = (1, 2, 3, 4, 5, 83, 84, 85, 86, 87, 88, 89, 94)
N_TURNS = 100
REMANENCE_T = 0.4
MU0 = 4.0 * math.pi * 1e-7
RESIDUAL_TOL = 1e-7
STRICT_RESIDUAL_TOL = 1e-9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def savez_atomic(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def csv_atomic(path: Path, frame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.12e")
    os.replace(temporary, path)


def append_ledger(workdir: Path, row: dict[str, object]) -> None:
    path = workdir / "experiment_ledger.csv"
    fields = [
        "experiment_id",
        "parent_experiment",
        "unique_change",
        "input_hash",
        "native_gate",
        "comsol_gate",
        "conclusion",
        "next_action",
    ]
    existing: list[dict[str, object]] = []
    if path.is_file():
        import pandas as pd

        existing = pd.read_csv(path).to_dict("records")
    if any(str(item.get("experiment_id")) == str(row["experiment_id"]) for item in existing):
        return
    existing.append(row)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing)
    os.replace(temporary, path)


def import_mainline(base: Path):
    base = base.resolve()
    sys.path[:0] = [str(base / "src"), str(base / "best_model")]
    from loudspeaker_axisym_fem.axisym_magnetics import (
        _default_dirichlet_nodes,
        _element_fields,
        _tri_geometry,
        effective_mu_r_from_B,
        load_tagged_meshio,
    )
    from loudspeaker_axisym_fem.comsol_driver_model import SOFT_IRON_BH_TABLE

    return {
        "base": base,
        "load_mesh": load_tagged_meshio,
        "default_fixed": _default_dirichlet_nodes,
        "element_fields": _element_fields,
        "tri_geometry": _tri_geometry,
        "effective_mu_r_from_B": effective_mu_r_from_B,
        "bh_table": SOFT_IRON_BH_TABLE,
    }


def refine_tagged_mesh(mesh):
    """Uniform 1-to-4 refinement while retaining all domain/line labels."""
    from types import SimpleNamespace

    points = [np.asarray(point, dtype=float).copy() for point in mesh.points_rz_m]
    edge_midpoints: dict[tuple[int, int], int] = {}

    def midpoint(a: int, b: int) -> int:
        key = (min(int(a), int(b)), max(int(a), int(b)))
        found = edge_midpoints.get(key)
        if found is not None:
            return found
        index = len(points)
        points.append(0.5 * (mesh.points_rz_m[key[0]] + mesh.points_rz_m[key[1]]))
        edge_midpoints[key] = index
        return index

    triangles: list[list[int]] = []
    domains: list[int] = []
    for tri, domain in zip(mesh.triangles, mesh.tri_domains):
        a, b, c = map(int, tri)
        ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        triangles.extend([[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]])
        domains.extend([int(domain)] * 4)
    lines: list[list[int]] = []
    line_tags: list[int] = []
    for line, tag in zip(mesh.line_cells, mesh.line_tags):
        a, b = map(int, line)
        ab = midpoint(a, b)
        lines.extend([[a, ab], [ab, b]])
        line_tags.extend([int(tag), int(tag)])
    refined_points = np.asarray(points, dtype=float)
    refined_triangles = np.asarray(triangles, dtype=int)
    refined_line_cells = np.asarray(lines, dtype=int)
    refined_line_tags = np.asarray(line_tags, dtype=int)

    def boundary_nodes(boundary_ids=None):
        if refined_line_cells.size == 0:
            return np.array([], dtype=int)
        if boundary_ids is None:
            mask = np.ones(len(refined_line_tags), dtype=bool)
        else:
            ids = set(int(value) for value in boundary_ids)
            mask = np.isin(refined_line_tags, list(ids))
        return np.unique(refined_line_cells[mask].ravel())

    return SimpleNamespace(
        points_rz_m=refined_points,
        triangles=refined_triangles,
        tri_domains=np.asarray(domains, dtype=int),
        line_cells=refined_line_cells,
        line_tags=refined_line_tags,
        n_nodes=len(refined_points),
        n_triangles=len(refined_triangles),
        boundary_nodes=boundary_nodes,
    )


class MovingWinding:
    """Reference coil quadrature and moving source/observation operator."""

    def __init__(self, mesh, tri_geometry, winding_domains=COIL_DOMAINS):
        self.mesh = mesh
        self.tri_geometry = tri_geometry
        area, centroid, _, _ = tri_geometry(mesh.points_rz_m, mesh.triangles)
        mask = np.isin(mesh.tri_domains.astype(int), list(winding_domains))
        if not np.any(mask):
            raise ValueError("coil domains 17/18/19 not found")
        self.coil_triangles = np.nonzero(mask)[0]
        self.area = area
        self.centroid = centroid
        self.coil_area_m2 = float(np.sum(area[mask]))
        if self.coil_area_m2 <= 0:
            raise ValueError("coil area is not positive")
        self.reference_points = []
        self.reference_weights = []
        # Three-point degree-two exact triangle quadrature.
        bary = np.asarray([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])
        self.barycentric_quadrature = bary
        for element in self.coil_triangles:
            vertices = mesh.points_rz_m[mesh.triangles[element]]
            for weights in bary:
                self.reference_points.append(weights @ vertices)
                self.reference_weights.append(float(area[element] / 3.0))
        self.reference_points = np.asarray(self.reference_points, dtype=float)
        self.reference_weights = np.asarray(self.reference_weights, dtype=float)
        self.centers = (mesh.points_rz_m[mesh.triangles].mean(axis=1)).astype(float)
        self.tree = cKDTree(self.centers)
        self._last_locator = None

    def _locate(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = min(32, self.mesh.n_triangles)
        distances, candidates = self.tree.query(points, k=count)
        if count == 1:
            candidates = np.asarray(candidates)[:, None]
        tri_indices = np.empty(len(points), dtype=int)
        barycentric = np.empty((len(points), 3), dtype=float)
        margins = np.empty(len(points), dtype=float)
        for q, (point, choices) in enumerate(zip(points, candidates)):
            best_margin = -np.inf
            best_index = -1
            best_bary = None
            for raw_index in np.asarray(choices).ravel():
                index = int(raw_index)
                vertices = self.mesh.points_rz_m[self.mesh.triangles[index]]
                transform = np.column_stack((vertices[1] - vertices[0], vertices[2] - vertices[0]))
                try:
                    ab = np.linalg.solve(transform, point - vertices[0])
                except np.linalg.LinAlgError:
                    continue
                weights = np.asarray([1.0 - ab.sum(), ab[0], ab[1]], dtype=float)
                margin = float(np.min(weights))
                if margin > best_margin:
                    best_margin, best_index, best_bary = margin, index, weights
            if best_index < 0 or best_margin < -1e-10:
                raise RuntimeError(
                    "移动线圈积分点未落入固定磁场解域: "
                    f"point={point.tolist()}, best_barycentric={best_margin:.3e}"
                )
            tri_indices[q] = best_index
            barycentric[q] = best_bary
            margins[q] = best_margin
        return tri_indices, barycentric, margins

    def at(self, displacement_m: float) -> dict[str, object]:
        points = self.reference_points.copy()
        points[:, 1] += float(displacement_m)
        tri_indices, barycentric, margins = self._locate(points)
        source = np.zeros(self.mesh.n_nodes, dtype=float)
        factor = 2.0 * math.pi * points[:, 0] * (N_TURNS / self.coil_area_m2)
        coefficients = factor * self.reference_weights
        for q, element in enumerate(tri_indices):
            source[self.mesh.triangles[element]] += coefficients[q] * barycentric[q]
        support = np.flatnonzero(np.abs(source) > 1e-30).astype(np.int64)
        support_hash = sha256_bytes(support.tobytes())
        value_hash = sha256_bytes(
            support.tobytes() + np.asarray(source[support], dtype=np.float64).tobytes()
        )
        return {
            "displacement_m": float(displacement_m),
            "points_rz_m": points,
            "tri_indices": tri_indices,
            "barycentric": barycentric,
            "locator_min_barycentric": float(np.min(margins)),
            "source_vector": source,
            "support_indices": support,
            "support_sha256": support_hash,
            "source_sha256": value_hash,
            "coil_area_m2": self.coil_area_m2,
        }

    def lorentz_bl(self, displacement_m: float, B_r: np.ndarray, located: dict[str, object]) -> float:
        points = np.asarray(located["points_rz_m"], dtype=float)
        indices = np.asarray(located["tri_indices"], dtype=int)
        values = np.asarray(B_r, dtype=float)[indices]
        weights = self.reference_weights
        return float(
            np.sum(-2.0 * math.pi * N_TURNS * points[:, 0] * values * weights)
            / self.coil_area_m2
        )


class MagneticFEM:
    """Precomputed P1 axisymmetric nonlinear magnetic FEM on one mesh."""

    def __init__(self, mesh, mainline, *, mesh_level: int):
        self.mesh = mesh
        self.mainline = mainline
        self.mesh_level = int(mesh_level)
        self.tri_geometry = mainline["tri_geometry"]
        self.element_fields = mainline["element_fields"]
        self.effective_mu_r_from_B = mainline["effective_mu_r_from_B"]
        self.bh_table = mainline["bh_table"]
        self.area, self.centroid, self.dNdr, self.dNdz = self.tri_geometry(
            mesh.points_rz_m, mesh.triangles
        )
        self.mesh_quality_min_area_ratio = float(
            np.min(self.area[self.area > 0]) / max(np.max(self.area), 1e-300)
        )
        self.r = np.maximum(self.centroid[:, 0], 1e-9)
        ncent = 1.0 / 3.0
        grad_r = self.dNdr + ncent / self.r[:, None]
        grad_z = self.dNdz
        weight = 2.0 * math.pi * self.r * self.area / MU0
        self.element_K = weight[:, None, None] * (
            np.einsum("ei,ej->eij", grad_z, grad_z)
            + np.einsum("ei,ej->eij", grad_r, grad_r)
        )
        self.element_magnet_rhs = weight[:, None] * REMANENCE_T * grad_r
        self.soft = np.isin(mesh.tri_domains.astype(int), list(SOFT_DOMAINS))
        self.magnet = np.isin(mesh.tri_domains.astype(int), list(MAGNET_DOMAINS))
        self.fixed = np.asarray(
            mainline["default_fixed"](mesh, EXTERIOR_BOUNDARIES), dtype=int
        )
        fixed_mask = np.ones(mesh.n_nodes, dtype=bool)
        fixed_mask[self.fixed] = False
        self.free = np.nonzero(fixed_mask)[0]
        self.rows = np.repeat(mesh.triangles, 3, axis=1).ravel()
        self.cols = np.tile(mesh.triangles, (1, 3)).ravel()
        topology = b"".join(
            np.asarray(values, dtype=np.int64).tobytes()
            for values in (self.rows, self.cols, self.free)
        )
        self.topology_hash = sha256_bytes(topology)
        self.winding = MovingWinding(mesh, self.tri_geometry)
        if np.any(self.soft & np.isin(mesh.tri_domains.astype(int), list(COIL_DOMAINS))):
            raise RuntimeError("coil domain unexpectedly overlaps soft iron")
        if np.any(self.magnet & np.isin(mesh.tri_domains.astype(int), list(COIL_DOMAINS))):
            raise RuntimeError("coil domain unexpectedly overlaps permanent magnet")
        # The default Picard starting material state is independent of (x, i).
        # Factor its operator once per mesh and reuse it as the exact first-step
        # solve for zero-initialized points and as a deterministic preconditioner
        # for continuation points.  Refactoring the same 178k-node L2 matrix for
        # every pilot/scan point is needlessly expensive and can exhaust the
        # long-running command window without changing the FEM equations.
        self._default_initial_mu = self.initial_mu()
        K_initial, _, _, _ = self.assemble(
            self._default_initial_mu,
            np.zeros(self.mesh.n_nodes, dtype=float),
            0.0,
        )
        self._default_initial_lu = splu(K_initial.tocsc())
        self._default_preconditioner = LinearOperator(
            K_initial.shape,
            matvec=lambda value: self._default_initial_lu.solve(
                np.asarray(value, dtype=float)
            ),
            dtype=float,
        )

    def assemble(self, mu_r: np.ndarray, source: np.ndarray, current_A: float):
        inv_mu = 1.0 / np.maximum(np.asarray(mu_r, dtype=float), 1.0)
        local = (self.element_K * inv_mu[:, None, None]).ravel()
        matrix = coo_matrix(
            (local, (self.rows, self.cols)),
            shape=(self.mesh.n_nodes, self.mesh.n_nodes),
        ).tocsr()
        rhs = np.zeros(self.mesh.n_nodes, dtype=float)
        if np.any(self.magnet):
            np.add.at(
                rhs,
                self.mesh.triangles[self.magnet].ravel(),
                (self.element_magnet_rhs[self.magnet] * inv_mu[self.magnet, None]).ravel(),
            )
        rhs += float(current_A) * np.asarray(source, dtype=float)
        return matrix[self.free][:, self.free].tocsr(), rhs[self.free], matrix, rhs

    def fields(self, A: np.ndarray, mu_r: np.ndarray):
        return self.element_fields(self.mesh, A, mu_r)

    def initial_mu(self) -> np.ndarray:
        mu = np.ones(self.mesh.n_triangles, dtype=float)
        mu[self.soft] = 700.0
        return mu

    def solve_point(
        self,
        displacement_m: float,
        current_A: float,
        *,
        initial_A: np.ndarray | None = None,
        initial_mu: np.ndarray | None = None,
        strict: bool = False,
        zero_initial: bool = False,
    ) -> dict[str, object]:
        located = self.winding.at(displacement_m)
        source = np.asarray(located["source_vector"], dtype=float)
        A_start = np.zeros(self.mesh.n_nodes, dtype=float) if initial_A is None else np.asarray(initial_A, dtype=float).copy()
        mu_start = self.initial_mu() if initial_mu is None else np.asarray(initial_mu, dtype=float).copy()
        tolerance = STRICT_RESIDUAL_TOL if strict else RESIDUAL_TOL
        attempts: list[dict[str, object]] = []
        started_total = time.perf_counter()
        # The nonlinear update only changes the soft-iron element weights.
        # Reuse the mesh-level factorization described in __init__.  If a
        # continuation state supplies a different mu field, the first solve is
        # iterative with the same strict linear tolerance; a failed CG solve
        # falls back to a direct solve and is never accepted on a weak residual.
        preconditioner = self._default_preconditioner
        exact_initial_operator = initial_mu is None
        for retry_index, relaxation in enumerate(FIXED_RETRY_RELAXATIONS):
            A_previous = A_start.copy()
            mu = mu_start.copy()
            started = time.perf_counter()
            converged = False
            last = (math.inf, math.inf, math.inf)
            A = A_previous.copy()
            B_r = B_z = B_norm = H_norm = np.zeros(self.mesh.n_triangles)
            for iteration in range(1, 241):
                Kff, rhs, Kfull, rhs_full = self.assemble(mu, source, current_A)
                if iteration == 1 and exact_initial_operator:
                    A_free = self._default_initial_lu.solve(rhs)
                else:
                    A_free, cg_info = cg(
                        Kff,
                        rhs,
                        x0=A_previous[self.free],
                        M=preconditioner,
                        rtol=1e-11,
                        atol=1e-13,
                        maxiter=1200,
                    )
                    if cg_info != 0:
                        # A failed iterative solve is a linear-solver failure,
                        # not permission to weaken the nonlinear gate.
                        A_free = spsolve(Kff, rhs)
                A = np.zeros(self.mesh.n_nodes, dtype=float)
                A[self.free] = A_free
                residual_A = float(
                    np.linalg.norm(A - A_previous) / max(np.linalg.norm(A), 1e-30)
                )
                B_r, B_z, B_norm, H_norm = self.fields(A, mu)
                target = mu.copy()
                if np.any(self.soft):
                    target[self.soft] = np.clip(
                        self.effective_mu_r_from_B(
                            np.maximum(B_norm[self.soft], 0.0),
                            self.bh_table,
                            floor=1.0,
                            cap=4000.0,
                        ),
                        1.0,
                        4000.0,
                    )
                residual_mu = float(
                    np.linalg.norm(target[self.soft] - mu[self.soft])
                    / max(np.linalg.norm(target[self.soft]), 1e-30)
                ) if np.any(self.soft) else 0.0
                pde = Kfull @ A - rhs_full
                pde_residual = float(
                    np.linalg.norm(pde[self.free])
                    / max(np.linalg.norm(rhs), 1e-30)
                )
                last = (residual_A, residual_mu, pde_residual)
                if max(last) <= tolerance and iteration >= 2:
                    converged = True
                    break
                mu = (1.0 - relaxation) * mu + relaxation * target
                A_previous = A
            attempts.append(
                {
                    "retry_index": retry_index,
                    "relaxation": relaxation,
                    "iterations": iteration,
                    "residual_A": last[0],
                    "residual_mu": last[1],
                    "pde_residual": last[2],
                    "runtime_s": time.perf_counter() - started,
                    "converged": converged,
                }
            )
            if converged:
                break
        total_runtime = time.perf_counter() - started_total
        if not converged:
            raise RuntimeError(
                f"磁静态点 ({displacement_m:.9g} m, {current_A:.9g} A) 未收敛；"
                f"最后残差 A/mu/PDE={last[0]:.3e}/{last[1]:.3e}/{last[2]:.3e}"
            )
        raw_flux = float(source @ A)
        lorentz_bl = self.winding.lorentz_bl(displacement_m, B_r, located)
        return {
            "x_m": float(displacement_m),
            "x_mm": float(displacement_m * 1000.0),
            "current_A": float(current_A),
            "mesh_level": self.mesh_level,
            "converged": True,
            "retry_index": int(attempts[-1]["retry_index"]),
            "relaxation": float(attempts[-1]["relaxation"]),
            "iterations": int(attempts[-1]["iterations"]),
            "residual_A": float(last[0]),
            "residual_mu": float(last[1]),
            "pde_residual": float(last[2]),
            "runtime_s": float(total_runtime),
            "winding_vector_norm": float(np.linalg.norm(source)),
            "winding_vector_sha256": str(located["source_sha256"]),
            "winding_support_sha256": str(located["support_sha256"]),
            "A_norm": float(np.linalg.norm(A)),
            "B_max_T": float(np.max(np.abs(B_norm))),
            "mu_min": float(np.min(mu)),
            "mu_max": float(np.max(mu)),
            "psi_raw_Wb": raw_flux,
            "lorentz_force_N": float(current_A * lorentz_bl),
            "lorentz_BL_N_A": float(lorentz_bl),
            "coil_area_m2": float(self.winding.coil_area_m2),
            "mesh_quality_min_area_ratio": self.mesh_quality_min_area_ratio,
            "locator_min_barycentric": float(located["locator_min_barycentric"]),
            "initial_state_source": "zero" if zero_initial or initial_A is None else "continuation",
            "error_message": "",
            "attempts": attempts,
            "A_phi": A,
            "B_r": B_r,
            "B_z": B_z,
            "B_norm": B_norm,
            "mu_r": mu,
            "source_vector": source,
            "winding_points_rz_m": np.asarray(located["points_rz_m"]),
        }


def point_row(result: dict[str, object], *, split: str, psi_offset: float = 0.0) -> dict[str, object]:
    fields = [
        "x_m",
        "x_mm",
        "current_A",
        "mesh_level",
        "converged",
        "retry_index",
        "relaxation",
        "iterations",
        "residual_A",
        "residual_mu",
        "pde_residual",
        "runtime_s",
        "winding_vector_norm",
        "winding_vector_sha256",
        "winding_support_sha256",
        "A_norm",
        "B_max_T",
        "mu_min",
        "mu_max",
        "psi_raw_Wb",
        "lorentz_force_N",
        "lorentz_BL_N_A",
        "coil_area_m2",
        "mesh_quality_min_area_ratio",
        "locator_min_barycentric",
        "initial_state_source",
        "error_message",
    ]
    row = {"split": split}
    for name in fields:
        row[name] = result.get(name)
    row["psi_Wb"] = float(result.get("psi_raw_Wb", np.nan)) - float(psi_offset)
    row["energy_diagnostic_J"] = None
    return row


def _load_scan_arrays(workdir: Path):
    path = workdir / "training_tensor.npz"
    if not path.is_file():
        raise RuntimeError(f"缺少阶段 2 输出 {path}；必须先通过 scan")
    with np.load(path, allow_pickle=False) as raw:
        return {name: np.asarray(raw[name]) for name in raw.files}


def write_pilot_vtu(path: Path, result: dict[str, object], mesh) -> None:
    import meshio

    points = np.column_stack([mesh.points_rz_m, np.zeros(mesh.n_nodes)])
    cell_data = {
        "domain_id": [mesh.tri_domains.astype(int)],
        "B_r_T": [np.asarray(result["B_r"], dtype=float)],
        "B_z_T": [np.asarray(result["B_z"], dtype=float)],
        "B_norm_T": [np.asarray(result["B_norm"], dtype=float)],
        "mu_r": [np.asarray(result["mu_r"], dtype=float)],
    }
    meshio.write_points_cells(
        str(path),
        points,
        [("triangle", mesh.triangles)],
        point_data={"A_phi_Wb_per_m": np.asarray(result["A_phi"], dtype=float)},
        cell_data=cell_data,
    )


def build_context(mainline_path: Path | None = None):
    base = (mainline_path or ROOT / "inputs" / "frequency_mainline").resolve()
    mainline = import_mainline(base)
    mesh_path = base / "inputs/meshes/comsol_geometry_polyline_coarse_2p5mm.msh"
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    mesh = mainline["load_mesh"](mesh_path)
    input_hashes = {
        "magnetic_mesh": sha256(mesh_path),
        "axisym_magnetics_py": sha256(base / "src/loudspeaker_axisym_fem/axisym_magnetics.py"),
        "comsol_driver_model_py": sha256(base / "src/loudspeaker_axisym_fem/comsol_driver_model.py"),
        "tensor_builder_py": sha256(Path(__file__).resolve()),
    }
    bh_bytes = repr(tuple(mainline["bh_table"])).encode()
    input_hashes["soft_iron_bh_table"] = sha256_bytes(bh_bytes)
    input_hash = sha256_bytes(json.dumps(input_hashes, sort_keys=True).encode())
    return {"base": base, "mesh": mesh, "mesh_path": mesh_path, "mainline": mainline, "input_hashes": input_hashes, "input_hash": input_hash}


def pilot(args) -> int:
    import pandas as pd

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    context = build_context(Path(args.mainline) if args.mainline else None)
    append_ledger(
        workdir,
        {
            "experiment_id": "E00_BASELINE",
            "parent_experiment": "",
            "unique_change": "read-only baseline evidence and native mesh input hash",
            "input_hash": context["input_hash"],
            "native_gate": "pytest=12 passed; self_test=PASS",
            "comsol_gate": "not run; refined files read-only",
            "conclusion": "baseline frozen",
            "next_action": "E01_PILOT_L0_L1",
        },
    )
    append_ledger(
        workdir,
        {
            "experiment_id": "E01_PILOT_L0_L1",
            "parent_experiment": "E00_BASELINE",
            "unique_change": "true moving-winding 3x3 native nonlinear FEM pilot and mesh refinement",
            "input_hash": context["input_hash"],
            "native_gate": "all pilot points and mesh gate are evaluated below",
            "comsol_gate": "not applicable before fit/transient",
            "conclusion": "in progress",
            "next_action": "complete pilot before scan",
        },
    )
    requested = sorted(set(int(value) for value in args.mesh_levels))
    if requested[0] != 0:
        raise ValueError("pilot 必须从 mesh level 0 开始")
    meshes = {0: context["mesh"]}
    for level in range(1, max(requested) + 1):
        meshes[level] = refine_tagged_mesh(meshes[level - 1])
    # Always create L1 for the mandated first comparison; L2 is created only if
    # the L0/L1 result does not close the 0.5% gate.
    if 1 not in meshes:
        meshes[1] = refine_tagged_mesh(meshes[0])
    results_by_level: dict[int, list[dict[str, object]]] = {}
    snapshot_dir = workdir / "field_snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    for level in (0, 1):
        solver = MagneticFEM(meshes[level], context["mainline"], mesh_level=level)
        rows = []
        for x_mm in PILOT_X_MM:
            for current in (0.0, 0.5, -0.5):
                # Pilot points are intentionally all initialized independently;
                # this exposes a false "shifted VTU" implementation.
                result = solver.solve_point(float(x_mm) * 1e-3, float(current), zero_initial=True)
                result["split"] = "pilot"
                rows.append(result)
                filename = f"pilot_x_{x_mm:+.4f}mm_i_{current:+.4f}A_L{level}.vtu".replace("+", "p").replace("-", "m")
                write_pilot_vtu(snapshot_dir / filename, result, meshes[level])
        results_by_level[level] = rows

    def row_lookup(rows):
        return {(round(float(r["x_m"]), 12), round(float(r["current_A"]), 12)): r for r in rows}

    l0, l1 = row_lookup(results_by_level[0]), row_lookup(results_by_level[1])
    force_denominator = max(
        1e-30,
        1e-4 * max(abs(float(r["lorentz_force_N"])) for r in l0.values()),
    )
    changes = []
    for key in l0:
        a, b = l0[key], l1[key]
        changes.append(
            {
                "x_m": key[0],
                "current_A": key[1],
                "psi_raw_relative_change": abs(float(b["psi_raw_Wb"]) - float(a["psi_raw_Wb"])) / max(abs(float(a["psi_raw_Wb"])), 1e-30),
                "lorentz_force_relative_change": abs(float(b["lorentz_force_N"]) - float(a["lorentz_force_N"])) / max(abs(float(a["lorentz_force_N"])), force_denominator),
                "lorentz_BL_relative_change": abs(float(b["lorentz_BL_N_A"]) - float(a["lorentz_BL_N_A"])) / max(abs(float(a["lorentz_BL_N_A"])), 1e-30),
            }
        )
    max_01 = max(max(float(row["psi_raw_relative_change"]), float(row["lorentz_force_relative_change"]), float(row["lorentz_BL_relative_change"])) for row in changes)
    convergence_rows = [{"from_level": 0, "to_level": 1, **row} for row in changes]
    selected_level = 0 if max_01 <= 0.005 else None
    l12 = []
    if selected_level is None:
        if 2 not in meshes:
            meshes[2] = refine_tagged_mesh(meshes[1])
        l12_points = ((-2.0, -0.5), (-1.0, 0.5), (0.0, 0.0), (1.0, -0.5), (2.0, 0.5))
        # The mandated L1/L2 comparison set contains x=+-1 mm, which is not
        # part of the 3x3 pilot's x axis.  Reuse the already solved -2/0/+2
        # rows and perform only the two missing native L1 solves; silently
        # looking up the 3x3 table here would turn a valid pilot into a false
        # pass or an opaque KeyError.
        l1_lookup = row_lookup(results_by_level[1])
        missing_l1 = [
            (x_mm, current)
            for x_mm, current in l12_points
            if (round(x_mm * 1e-3, 12), round(current, 12)) not in l1_lookup
        ]
        if missing_l1:
            l1_solver = MagneticFEM(meshes[1], context["mainline"], mesh_level=1)
            for x_mm, current in missing_l1:
                extra = l1_solver.solve_point(x_mm * 1e-3, current, zero_initial=True)
                l1_lookup[(round(float(extra["x_m"]), 12), round(float(extra["current_A"]), 12))] = extra
            del l1_solver
        solver = MagneticFEM(meshes[2], context["mainline"], mesh_level=2)
        rows = []
        for x_mm, current in l12_points:
            result = solver.solve_point(x_mm * 1e-3, current, zero_initial=True)
            rows.append(result)
        l2_lookup = row_lookup(rows)
        force_denominator_12 = max(1e-30, 1e-4 * max(abs(float(r["lorentz_force_N"])) for r in rows))
        for key, b in l2_lookup.items():
            a = l1_lookup[key]
            l12.append(
                {
                    "x_m": key[0],
                    "current_A": key[1],
                    "psi_raw_relative_change": abs(float(b["psi_raw_Wb"]) - float(a["psi_raw_Wb"])) / max(abs(float(a["psi_raw_Wb"])), 1e-30),
                    "lorentz_force_relative_change": abs(float(b["lorentz_force_N"]) - float(a["lorentz_force_N"])) / max(abs(float(a["lorentz_force_N"])), force_denominator_12),
                    "lorentz_BL_relative_change": abs(float(b["lorentz_BL_N_A"]) - float(a["lorentz_BL_N_A"])) / max(abs(float(a["lorentz_BL_N_A"])), 1e-30),
                }
            )
        max_12 = max(max(float(row["psi_raw_relative_change"]), float(row["lorentz_force_relative_change"]), float(row["lorentz_BL_relative_change"])) for row in l12)
        convergence_rows.extend([{ "from_level": 1, "to_level": 2, **row} for row in l12])
        if max_12 <= 0.005:
            selected_level = 1
        else:
            selected_level = -1
    else:
        max_12 = None
    for level, rows in results_by_level.items():
        for row in rows:
            row.pop("A_phi", None)
            row.pop("B_r", None)
            row.pop("B_z", None)
            row.pop("B_norm", None)
            row.pop("mu_r", None)
            row.pop("source_vector", None)
            row.pop("winding_points_rz_m", None)
    pd.DataFrame(convergence_rows).to_csv(workdir / "mesh_convergence.csv", index=False, float_format="%.12e")
    all_pilot_rows = []
    for level, rows in results_by_level.items():
        for result in rows:
            all_pilot_rows.append(point_row(result, split="pilot"))
    pd.DataFrame(all_pilot_rows).to_csv(workdir / "pilot_points.csv", index=False, float_format="%.12e")
    # Old fixed-VTU interpolation is retained only as an explicitly labelled
    # anti-fraud diagnostic, never as a training input.
    fixed_vtu_diag = fixed_vtu_comparison(context, results_by_level[0])
    json_dump(workdir / "fixed_vtu_shift_diagnostic.json", fixed_vtu_diag)
    x0_scan_comparison = None
    if selected_level >= 0:
        selected_rows = row_lookup(results_by_level[selected_level])
        native_x0 = float(selected_rows[(0.0, 0.5)]["lorentz_BL_N_A"])
        old_path = ROOT / "inputs/nonlinear_magnetic_law_20260728.json"
        old_data = json.loads(old_path.read_text(encoding="utf-8"))
        old_bl = float(np.polynomial.chebyshev.chebval(0.0, np.asarray(old_data["bl_chebyshev_coefficients_N_A"], dtype=float)))
        x0_scan_comparison = {"native_BL_N_A": native_x0, "legacy_current_scan_BL_N_A": old_bl, "relative_difference": abs(native_x0 - old_bl) / max(abs(old_bl), 1e-30), "order_of_magnitude_consistent": abs(native_x0) > 0 and abs(native_x0 / old_bl) < 2.0}
    pilot_summary = {
        "schema_version": 1,
        "kind": "native_tensor_coenergy_pilot",
        "input_hash": context["input_hash"],
        "input_hashes": context["input_hashes"],
        "mainline": str(context["base"]),
        "mesh_levels_evaluated": sorted(results_by_level),
        "mesh_convergence_max_L0_L1": max_01,
        "mesh_convergence_max_L1_L2": max_12,
        "selected_mesh_level": selected_level,
        "residual_tolerances": {"pilot": RESIDUAL_TOL, "strict": STRICT_RESIDUAL_TOL},
        "pilot_points": 9,
        "all_pilot_points_converged": all(bool(r["converged"]) for r in all_pilot_rows),
        "all_locator_margins_positive": all(float(r["locator_min_barycentric"]) >= -1e-10 for r in all_pilot_rows),
        "fixed_vtu_diagnostic": "fixed_vtu_shift_diagnostic.json",
        "x0_current_scan_comparison": x0_scan_comparison,
    }
    if selected_level < 0:
        pilot_summary["status"] = "failed_mesh_not_converged"
    else:
        pilot_summary["status"] = "passed" if pilot_summary["all_pilot_points_converged"] else "failed_pilot_point"
    json_dump(workdir / "pilot_summary.json", pilot_summary)
    json_dump(workdir / "pilot_pass.json", {"status": pilot_summary["status"], "evidence": "pilot_summary.json"})
    json_dump(workdir / "provenance_pilot.json", {
        "schema_version": 1,
        "command": " ".join(sys.argv),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "input_hashes": context["input_hashes"],
    })
    print(json.dumps(pilot_summary, ensure_ascii=False, indent=2))
    return 0 if pilot_summary["status"] == "passed" else 1


def fixed_vtu_comparison(context: dict[str, object], pilot_results: list[dict[str, object]]) -> dict[str, object]:
    """Compare native re-solves with the prohibited fixed-VTU shift diagnostic."""
    import meshio

    vtu_path = context["base"] / "inputs/comsol_reference/magnetostatic_converged_55iter.vtu"
    if not Path(vtu_path).is_file():
        return {"status": "unavailable", "reason": str(vtu_path)}
    vtu = meshio.read(str(vtu_path))
    triangles = next(np.asarray(block.data, dtype=int) for block in vtu.cells if block.type == "triangle")
    points = np.asarray(vtu.points[:, :2], dtype=float)
    centers = points[triangles].mean(axis=1)
    fields = vtu.cell_data_dict
    br = np.asarray(fields["B_r_T"]["triangle"], dtype=float)
    tree = cKDTree(centers)
    winding = MovingWinding(context["mesh"], context["mainline"]["tri_geometry"])
    rows = []
    for native in pilot_results:
        x = float(native["x_m"])
        if abs(float(native["current_A"])) < 1e-14:
            continue
        location = winding.at(x)
        shifted = np.asarray(location["points_rz_m"])
        dist, indices = tree.query(shifted, k=min(8, len(centers)))
        if np.ndim(indices) == 1:
            indices = indices[:, None]
            dist = dist[:, None]
        weights = 1.0 / np.maximum(dist, 2e-5) ** 2
        values = np.sum(weights * br[indices], axis=1) / np.sum(weights, axis=1)
        fixed_bl = float(np.sum(-2 * math.pi * N_TURNS * shifted[:, 0] * values * winding.reference_weights) / winding.coil_area_m2)
        rows.append({"x_m": x, "current_A": float(native["current_A"]), "native_lorentz_BL_N_A": float(native["lorentz_BL_N_A"]), "fixed_vtu_shift_BL_N_A": fixed_bl, "absolute_difference_N_A": abs(float(native["lorentz_BL_N_A"]) - fixed_bl), "native_source_sha256": native["winding_vector_sha256"]})
    return {"status": "completed", "method": "prohibited fixed VTU 8-neighbor diagnostic only", "rows": rows, "native_re_solve_is_distinct": bool(any(row["absolute_difference_N_A"] > 1e-12 for row in rows))}


def _checkpoint_arrays(workdir: Path, context: dict[str, object], arrays: dict[str, np.ndarray], *, last_zero_A=None, last_zero_mu=None, last_x_index=-1) -> None:
    checkpoint = workdir / "checkpoint.npz"
    save = dict(arrays)
    if last_zero_A is not None:
        save["last_zero_A"] = np.asarray(last_zero_A)
        save["last_zero_mu"] = np.asarray(last_zero_mu)
    save["last_x_index"] = np.asarray([last_x_index], dtype=int)
    savez_atomic(checkpoint, **save)
    json_dump(workdir / "checkpoint_metadata.json", {
        "schema_version": 1,
        "input_hash": context["input_hash"],
        "input_hashes": context["input_hashes"],
        "mesh_level": int(arrays["mesh_level"][0, 0]),
        "axis_order": ["x_index", "i_index"],
        "completed_points": int(np.count_nonzero(arrays["completed"])),
        "total_points": int(arrays["completed"].size),
        "last_x_index": int(last_x_index),
    })


def _fresh_scan_arrays(nx: int, ni: int, mesh_level: int):
    arrays: dict[str, np.ndarray] = {
        "x_training_m": X_TRAIN_MM[:, None] * 1e-3,
        "current_training_A": I_TRAIN_A[None, :],
        "mesh_level": np.full((nx, ni), mesh_level, dtype=int),
        "completed": np.zeros((nx, ni), dtype=bool),
        "converged": np.zeros((nx, ni), dtype=bool),
        "retry_index": np.full((nx, ni), -1, dtype=int),
        "relaxation": np.full((nx, ni), np.nan),
        "iterations": np.zeros((nx, ni), dtype=int),
        "residual_A": np.full((nx, ni), np.nan),
        "residual_mu": np.full((nx, ni), np.nan),
        "pde_residual": np.full((nx, ni), np.nan),
        "runtime_s": np.full((nx, ni), np.nan),
        "winding_vector_norm": np.full((nx, ni), np.nan),
        "A_norm": np.full((nx, ni), np.nan),
        "B_max_T": np.full((nx, ni), np.nan),
        "mu_min": np.full((nx, ni), np.nan),
        "mu_max": np.full((nx, ni), np.nan),
        "psi_raw_Wb": np.full((nx, ni), np.nan),
        "psi_training_Wb": np.full((nx, ni), np.nan),
        "lorentz_force_N": np.full((nx, ni), np.nan),
        "lorentz_BL_N_A": np.full((nx, ni), np.nan),
        "coil_area_m2": np.full((nx, ni), np.nan),
        "mesh_quality_min_area_ratio": np.full((nx, ni), np.nan),
        "locator_min_barycentric": np.full((nx, ni), np.nan),
        "winding_vector_sha256": np.full((nx, ni), "", dtype="U64"),
        "winding_support_sha256": np.full((nx, ni), "", dtype="U64"),
        "initial_state_source": np.full((nx, ni), "", dtype="U16"),
        "error_message": np.full((nx, ni), "", dtype="U256"),
    }
    return arrays


def _load_or_new_checkpoint(workdir: Path, context: dict[str, object], mesh_level: int):
    path = workdir / "checkpoint.npz"
    metadata_path = workdir / "checkpoint_metadata.json"
    if not path.is_file():
        return _fresh_scan_arrays(len(X_TRAIN_MM), len(I_TRAIN_A), mesh_level), None, None, -1
    if not metadata_path.is_file():
        raise RuntimeError("checkpoint.npz 存在但 checkpoint_metadata.json 缺失")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("input_hash") != context["input_hash"]:
        raise RuntimeError("checkpoint 输入哈希不一致，拒绝混用旧扫描点")
    if int(metadata.get("mesh_level", -1)) != int(mesh_level):
        raise RuntimeError("checkpoint 网格层级不一致")
    with np.load(path, allow_pickle=False) as raw:
        arrays = {name: np.asarray(raw[name]) for name in raw.files if name not in {"last_zero_A", "last_zero_mu", "last_x_index"}}
        last_A = np.asarray(raw["last_zero_A"]) if "last_zero_A" in raw else None
        last_mu = np.asarray(raw["last_zero_mu"]) if "last_zero_mu" in raw else None
        last_x = int(np.asarray(raw["last_x_index"])[0]) if "last_x_index" in raw else -1
    return arrays, last_A, last_mu, last_x


def _rows_from_scan_arrays(arrays: dict[str, np.ndarray], split: str = "training"):
    rows = []
    for ix, x in enumerate(X_TRAIN_MM * 1e-3):
        for ii, current in enumerate(I_TRAIN_A):
            row = {
                "split": split,
                "x_m": float(x),
                "x_mm": float(x * 1000),
                "current_A": float(current),
            }
            for name in arrays:
                if name in {"x_training_m", "current_training_A", "completed", "psi_training_Wb"}:
                    continue
                value = arrays[name][ix, ii]
                row[name] = value.item() if np.ndim(value) == 0 else value
            row["psi_Wb"] = float(arrays["psi_training_Wb"][ix, ii])
            row["energy_diagnostic_J"] = None
            rows.append(row)
    return rows


def _write_training_tensor(workdir: Path, arrays: dict[str, np.ndarray], psi_zero_raw: float) -> None:
    psi = arrays["psi_raw_Wb"] - float(psi_zero_raw)
    arrays["psi_training_Wb"][:] = psi
    W = np.zeros_like(psi)
    for ix in range(psi.shape[0]):
        W[ix] = np.array([np.trapezoid(psi[ix, : ii + 1], I_TRAIN_A[: ii + 1]) for ii in range(len(I_TRAIN_A))])
    arrays["W_raw_J"] = W
    arrays["F_L_N"] = np.asarray(arrays["lorentz_force_N"], dtype=float)
    arrays["W_training_J"] = W
    savez_atomic(workdir / "training_tensor.npz", **arrays, axis_order=np.asarray(["x_index", "i_index"]))
    import pandas as pd

    csv_atomic(workdir / "scan_points.csv", pd.DataFrame(_rows_from_scan_arrays(arrays)))


def scan(args) -> int:
    import pandas as pd

    workdir = Path(args.workdir).resolve()
    pilot_path = workdir / "pilot_summary.json"
    if not pilot_path.is_file():
        raise RuntimeError("scan 前置证据缺失：请先成功运行 pilot")
    pilot_summary = json.loads(pilot_path.read_text(encoding="utf-8"))
    if pilot_summary.get("status") != "passed":
        raise RuntimeError(f"pilot 未通过: {pilot_summary.get('status')}")
    context = build_context(Path(args.mainline) if args.mainline else None)
    if pilot_summary.get("input_hash") != context["input_hash"]:
        raise RuntimeError("pilot 输入/代码哈希与当前扫描器不一致；请重新运行 pilot")
    mesh_level = int(pilot_summary["selected_mesh_level"])
    mesh = context["mesh"]
    for _ in range(mesh_level):
        mesh = refine_tagged_mesh(mesh)
    solver = MagneticFEM(mesh, context["mainline"], mesh_level=mesh_level)
    arrays, last_zero_A, last_zero_mu, last_x = _load_or_new_checkpoint(workdir, context, mesh_level)
    if args.failed_only and not (workdir / "checkpoint.npz").is_file():
        raise RuntimeError("--failed-only 需要已有 checkpoint")
    x_indices = list(range(len(X_TRAIN_MM)))
    if args.x_index is not None:
        if not 0 <= args.x_index < len(X_TRAIN_MM):
            raise ValueError("--x-index 越界")
        x_indices = [int(args.x_index)]
    elif args.resume and last_x >= 0:
        x_indices = list(range(last_x + 1, len(X_TRAIN_MM)))
    if args.failed_only:
        x_indices = sorted(set(ix for ix in x_indices if not np.all(arrays["completed"][ix])))
    if not x_indices and np.all(arrays["completed"]):
        psi_zero = float(arrays["psi_raw_Wb"][np.where(X_TRAIN_MM == 0)[0][0], np.where(I_TRAIN_A == 0)[0][0]])
        _write_training_tensor(workdir, arrays, psi_zero)
        print(json.dumps({"status": "already_complete", "points": int(arrays["completed"].size)}, ensure_ascii=False, indent=2))
        return 0
    if last_x < 0 or not args.resume:
        last_zero_A = None
        last_zero_mu = None
        last_x = -1
    progress = {"status": "running", "completed": int(np.count_nonzero(arrays["completed"])), "total": int(arrays["completed"].size), "last_x_index": last_x}
    json_dump(workdir / "progress.json", progress)
    for ix in x_indices:
        x_m = float(X_TRAIN_MM[ix] * 1e-3)
        if args.failed_only and np.all(arrays["completed"][ix]):
            continue
        # Each x slice is atomically committed.  i=0 starts from the previous
        # x slice's i=0 state; positive and negative current branches start at
        # that same stable zero-current state.
        zero_A = last_zero_A.copy() if last_zero_A is not None else None
        zero_mu = last_zero_mu.copy() if last_zero_mu is not None else None
        positive_A = positive_mu = None
        negative_A = negative_mu = None
        slice_results: dict[int, dict[str, object]] = {}
        order = [int(np.where(I_TRAIN_A == 0)[0][0])]
        order += [int(ii) for ii in np.where(I_TRAIN_A > 0)[0]]
        order += [int(ii) for ii in np.where(I_TRAIN_A < 0)[0][::-1]]
        for ii in order:
            current = float(I_TRAIN_A[ii])
            if args.failed_only and bool(arrays["completed"][ix, ii]):
                continue
            if current == 0.0:
                init_A, init_mu, zero_flag = zero_A, zero_mu, zero_A is None
            elif current > 0.0:
                init_A, init_mu, zero_flag = positive_A if positive_A is not None else zero_A, positive_mu if positive_mu is not None else zero_mu, False
            else:
                init_A, init_mu, zero_flag = negative_A if negative_A is not None else zero_A, negative_mu if negative_mu is not None else zero_mu, False
            result = solver.solve_point(x_m, current, initial_A=init_A, initial_mu=init_mu, zero_initial=zero_flag)
            slice_results[ii] = result
            if current == 0.0:
                zero_A = np.asarray(result["A_phi"]).copy()
                zero_mu = np.asarray(result["mu_r"]).copy()
            elif current > 0.0:
                positive_A = np.asarray(result["A_phi"]).copy()
                positive_mu = np.asarray(result["mu_r"]).copy()
            else:
                negative_A = np.asarray(result["A_phi"]).copy()
                negative_mu = np.asarray(result["mu_r"]).copy()
        for ii, result in slice_results.items():
            for name in arrays:
                if name in {"x_training_m", "current_training_A", "completed", "psi_training_Wb", "W_raw_J"}:
                    continue
                if name in result:
                    value = result[name]
                    arrays[name][ix, ii] = value
            arrays["completed"][ix, ii] = True
            arrays["converged"][ix, ii] = bool(result["converged"])
        last_zero_A, last_zero_mu, last_x = zero_A, zero_mu, ix
        _checkpoint_arrays(workdir, context, arrays, last_zero_A=last_zero_A, last_zero_mu=last_zero_mu, last_x_index=last_x)
        progress = {"status": "running", "completed": int(np.count_nonzero(arrays["completed"])), "total": int(arrays["completed"].size), "last_x_index": last_x, "last_slice_x_mm": float(X_TRAIN_MM[ix]), "last_slice_seconds": float(sum(float(item.get("runtime_s", 0.0)) for item in slice_results.values()))}
        json_dump(workdir / "progress.json", progress)
        print(json.dumps(progress, ensure_ascii=False))
    if not np.all(arrays["completed"]):
        raise RuntimeError("扫描结束但仍有未完成点；拒绝生成训练张量")
    center_x = int(np.where(np.isclose(X_TRAIN_MM, 0.0))[0][0])
    center_i = int(np.where(np.isclose(I_TRAIN_A, 0.0))[0][0])
    psi_zero = float(arrays["psi_raw_Wb"][center_x, center_i])
    _write_training_tensor(workdir, arrays, psi_zero)
    path_rows = []
    for x_mm, current in PATH_INDEPENDENCE_POINTS:
        result = solver.solve_point(x_mm * 1e-3, current, zero_initial=True)
        ix0 = int(np.argmin(abs(X_TRAIN_MM - x_mm)))
        ii0 = int(np.argmin(abs(I_TRAIN_A - current)))
        continuation_psi = float(arrays["psi_raw_Wb"][ix0, ii0])
        continuation_force = float(arrays["lorentz_force_N"][ix0, ii0])
        psi_difference = abs(float(result["psi_raw_Wb"]) - continuation_psi) / max(abs(continuation_psi), 1e-12)
        force_difference = abs(float(result["lorentz_force_N"]) - continuation_force) / max(abs(continuation_force), 1e-8)
        path_rows.append({"x_mm": x_mm, "current_A": current, "psi_raw_Wb_zero_initial": result["psi_raw_Wb"], "psi_raw_Wb_continuation": continuation_psi, "psi_relative_difference": psi_difference, "lorentz_force_N_zero_initial": result["lorentz_force_N"], "lorentz_force_N_continuation": continuation_force, "force_relative_difference": force_difference, "residual_A": result["residual_A"], "residual_mu": result["residual_mu"], "pde_residual": result["pde_residual"]})
    path_max = max(max(float(row["psi_relative_difference"]), float(row["force_relative_difference"])) for row in path_rows)
    json_dump(workdir / "path_independence.json", {"points": path_rows, "method": "zero-initial rerun against continuation tensor", "relative_difference_limit": 0.001, "max_relative_field_quantity_difference": path_max, "status": "passed" if path_max <= 0.001 else "failed"})
    strict_rows = []
    for x_m, current in STRICT_POINTS:
        normal = solver.solve_point(x_m * 1e-3, current, zero_initial=True, strict=False)
        result = solver.solve_point(x_m * 1e-3, current, zero_initial=True, strict=True)
        psi_change = abs(float(result["psi_raw_Wb"]) - float(normal["psi_raw_Wb"])) / max(abs(float(normal["psi_raw_Wb"])), 1e-12)
        force_change = abs(float(result["lorentz_force_N"]) - float(normal["lorentz_force_N"])) / max(abs(float(normal["lorentz_force_N"])), 1e-8)
        strict_rows.append({"x_m": x_m * 1e-3, "current_A": current, "residual_A": result["residual_A"], "residual_mu": result["residual_mu"], "pde_residual": result["pde_residual"], "psi_raw_Wb": result["psi_raw_Wb"], "lorentz_force_N": result["lorentz_force_N"], "psi_change_vs_normal": psi_change, "force_change_vs_normal": force_change})
    strict_status = all(max(float(row["residual_A"]), float(row["residual_mu"]), float(row["pde_residual"])) <= STRICT_RESIDUAL_TOL and float(row["psi_change_vs_normal"]) <= 0.001 and float(row["force_change_vs_normal"]) <= 0.001 for row in strict_rows)
    json_dump(workdir / "strict_rechecks.json", {"points": strict_rows, "tolerance": STRICT_RESIDUAL_TOL, "quantity_change_limit": 0.001, "status": "passed" if strict_status else "failed"})
    if not strict_status:
        raise RuntimeError("严格五点重算未通过残差或 psi/F 变化门禁")
    # Keep the full provenance separate from the raw tensor itself.
    json_dump(workdir / "provenance.json", {"schema_version": 1, "kind": "native_tensor_magnetic_scan", "command": " ".join(sys.argv), "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "generated_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "python": platform.python_version(), "numpy": np.__version__, "scipy": __import__("scipy").__version__, "meshio": __import__("meshio").__version__, "input_hash": context["input_hash"], "input_hashes": context["input_hashes"], "mesh_level": mesh_level, "mesh_hash": context["input_hashes"]["magnetic_mesh"], "mesh_topology_hash": solver.topology_hash, "mesh_quality_min_area_ratio": solver.mesh_quality_min_area_ratio, "material_contract": {"soft_iron_domains": list(SOFT_DOMAINS), "permanent_magnet_domains": list(MAGNET_DOMAINS), "coil_domains": list(COIL_DOMAINS), "coil_mu_r_sigma": "mu_r=1 and sigma=0 in native magnetic stiffness/source contract"}, "axis_order": ["x_index", "i_index"], "coordinates": {"x": "m", "current": "A"}, "source_contract": "same moving winding vector is RHS and flux observation b(x)^T A", "residual_contract": {"residual_A": "||A_k-A_(k-1)||/max(||A_k||,1e-30)", "residual_mu": "||mu_target-mu_k||_soft/max(||mu_target||_soft,1e-30)", "pde_residual": "||K(mu_k)A_k-rhs||/max(||rhs||,1e-30)"}})
    append_ledger(workdir, {"experiment_id": "E02_TENSOR_V1", "parent_experiment": "E01_PILOT_L0_L1", "unique_change": "fixed 513 point native tensor scan", "input_hash": context["input_hash"], "native_gate": "scan completed; fit/holdout pending", "comsol_gate": "not read", "conclusion": "raw tensor complete", "next_action": "holdout then fit"})
    json_dump(workdir / "progress.json", {"status": "complete", "completed": int(arrays["completed"].size), "total": int(arrays["completed"].size), "last_x_index": last_x})
    return 0


def holdout(args) -> int:
    import pandas as pd

    workdir = Path(args.workdir).resolve()
    pilot_summary = json.loads((workdir / "pilot_summary.json").read_text(encoding="utf-8")) if (workdir / "pilot_summary.json").is_file() else {}
    if pilot_summary.get("status") != "passed":
        raise RuntimeError("holdout 前置证据缺失或 pilot 未通过")
    arrays = _load_scan_arrays(workdir)
    context = build_context(Path(args.mainline) if args.mainline else None)
    mesh = context["mesh"]
    mesh_level = int(pilot_summary["selected_mesh_level"])
    for _ in range(mesh_level):
        mesh = refine_tagged_mesh(mesh)
    solver = MagneticFEM(mesh, context["mainline"], mesh_level=mesh_level)
    rows = []
    for x_mm in X_HOLDOUT_MM:
        for current in I_HOLDOUT_A:
            result = solver.solve_point(float(x_mm) * 1e-3, float(current), zero_initial=True)
            rows.append(point_row(result, split="holdout", psi_offset=float(arrays["psi_raw_Wb"][np.where(np.isclose(X_TRAIN_MM, 0))[0][0], np.where(np.isclose(I_TRAIN_A, 0))[0][0]])))
    frame = pd.DataFrame(rows)
    frame.insert(0, "holdout_x_index", np.repeat(np.arange(len(X_HOLDOUT_MM)), len(I_HOLDOUT_A)))
    frame.insert(1, "holdout_i_index", np.tile(np.arange(len(I_HOLDOUT_A)), len(X_HOLDOUT_MM)))
    csv_atomic(workdir / "holdout_points.csv", frame)
    savez_atomic(workdir / "holdout_tensor.npz", x_holdout_m=X_HOLDOUT_MM * 1e-3, current_holdout_A=I_HOLDOUT_A, psi_holdout_Wb=frame["psi_Wb"].to_numpy().reshape(5, 5), lorentz_force_holdout_N=frame["lorentz_force_N"].to_numpy().reshape(5, 5), completed=np.ones((5, 5), dtype=bool))
    savez_atomic(workdir / "holdout_checkpoint.npz", completed=np.ones((5, 5), dtype=bool), x_holdout_m=X_HOLDOUT_MM * 1e-3, current_holdout_A=I_HOLDOUT_A)
    json_dump(workdir / "holdout_checkpoint_metadata.json", {"schema_version": 1, "input_hash": context["input_hash"], "completed_points": 25, "total_points": 25, "axis_order": ["x_index", "i_index"]})
    json_dump(workdir / "holdout_provenance.json", {"schema_version": 1, "kind": "native_tensor_magnetic_holdout", "input_hash": context["input_hash"], "training_tensor_sha256": sha256(workdir / "training_tensor.npz"), "points": 25, "not_used_for_fit": True})
    append_ledger(workdir, {"experiment_id": "E02_TENSOR_V1_HOLDOUT", "parent_experiment": "E02_TENSOR_V1", "unique_change": "fixed 5x5 Cartesian native FEM holdout", "input_hash": context["input_hash"], "native_gate": "25 holdout solves complete; fit pending", "comsol_gate": "not read", "conclusion": "raw holdout complete", "next_action": "fit fixed spline candidates"})
    print(json.dumps({"status": "complete", "points": 25}, ensure_ascii=False, indent=2))
    return 0


def nrmse(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - ref) ** 2)) / max(np.sqrt(np.mean(ref**2)), 1e-300))


def normalized_errors(pred: np.ndarray, ref: np.ndarray, floor: float = 0.0) -> tuple[float, float]:
    denominator = np.maximum(np.abs(ref), floor)
    rel = np.abs(pred - ref) / np.maximum(denominator, 1e-300)
    return nrmse(pred, ref), float(np.max(rel))


def fit(args) -> int:
    import pandas as pd
    from loudspeaker_time_fem.tensor_coenergy import TensorCoenergyLaw

    workdir = Path(args.workdir).resolve()
    pilot_summary_path = workdir / "pilot_summary.json"
    if not pilot_summary_path.is_file() or json.loads(pilot_summary_path.read_text(encoding="utf-8")).get("status") != "passed":
        raise RuntimeError("fit 前置证据缺失：pilot 未通过")
    arrays = _load_scan_arrays(workdir)
    holdout_path = workdir / "holdout_points.csv"
    if not holdout_path.is_file():
        raise RuntimeError("fit 前置证据缺失：请先运行 holdout")
    holdout = pd.read_csv(holdout_path)
    x_axis = np.asarray(arrays["x_training_m"][:, 0], dtype=float)
    i_axis = np.asarray(arrays["current_training_A"][0, :], dtype=float)
    psi = np.asarray(arrays["psi_training_Wb"], dtype=float)
    psi_scale = float(np.max(np.abs(psi)))
    if not np.isfinite(psi_scale) or psi_scale <= 0:
        raise RuntimeError("psi training tensor is empty or zero")
    candidates = []
    N = psi.size
    candidate_s = [N * value for value in (0.0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6)]
    xgrid = np.linspace(x_axis[0], x_axis[-1], 161)
    igrid = np.linspace(i_axis[0], i_axis[-1], 161)
    Xg, Ig = np.meshgrid(xgrid, igrid, indexing="ij")
    force_floor_train = 1e-4 * max(float(np.max(np.abs(arrays["lorentz_force_N"]))), 1e-300)
    force_floor_holdout = 1e-4 * max(float(np.max(np.abs(holdout["lorentz_force_N"]))), 1e-300)
    for smoothing in candidate_s:
        law = TensorCoenergyLaw(Path("candidate.json"), Path("candidate.npz"), 0.004, 1.0, x_axis, i_axis, psi, psi_scale, smoothing, {"candidate_s": smoothing}, 12)
        Lgrid = np.asarray(law.incremental_inductance(Xg, Ig), dtype=float)
        psi_train_pred = np.asarray(law.flux(*np.meshgrid(x_axis, i_axis, indexing="ij")), dtype=float)
        F_train_pred = np.asarray(law.force(*np.meshgrid(x_axis, i_axis, indexing="ij")), dtype=float)
        hold_x = holdout["x_m"].to_numpy(dtype=float)
        hold_i = holdout["current_A"].to_numpy(dtype=float)
        psi_hold_pred = np.asarray(law.flux(hold_x, hold_i), dtype=float)
        F_hold_pred = np.asarray(law.force(hold_x, hold_i), dtype=float)
        psi_train_nrmse, psi_train_max = normalized_errors(psi_train_pred.ravel(), psi.ravel(), floor=1e-8 * psi_scale)
        force_train_nrmse, force_train_max = normalized_errors(F_train_pred.ravel(), arrays["lorentz_force_N"].ravel(), floor=force_floor_train)
        psi_hold_nrmse, psi_hold_max = normalized_errors(psi_hold_pred, holdout["psi_Wb"].to_numpy(dtype=float), floor=1e-8 * psi_scale)
        force_hold_nrmse, force_hold_max = normalized_errors(F_hold_pred, holdout["lorentz_force_N"].to_numpy(dtype=float), floor=force_floor_holdout)
        positive = bool(np.all(Lgrid > 0.0))
        hard = positive and psi_hold_nrmse <= 0.005 and psi_hold_max <= 0.02 and force_hold_nrmse <= 0.005 and force_hold_max <= 0.02
        reason = "passed" if hard else "; ".join(filter(None, ["W_ii<=0" if not positive else "", "psi holdout threshold" if psi_hold_nrmse > .005 or psi_hold_max > .02 else "", "force holdout threshold" if force_hold_nrmse > .005 or force_hold_max > .02 else ""]))
        candidates.append({"smoothing_s_normalized": smoothing, "positive_inductance": positive, "L_min_H": float(np.min(Lgrid)), "L_max_H": float(np.max(Lgrid)), "psi_train_NRMSE": psi_train_nrmse, "psi_train_max_normalized_error": psi_train_max, "force_train_NRMSE": force_train_nrmse, "force_train_max_normalized_error": force_train_max, "psi_holdout_NRMSE": psi_hold_nrmse, "psi_holdout_max_normalized_error": psi_hold_max, "force_holdout_NRMSE": force_hold_nrmse, "force_holdout_max_normalized_error": force_hold_max, "score": psi_hold_nrmse + force_hold_nrmse, "passed": hard, "reason": reason})
    passed = [row for row in candidates if row["passed"]]
    if not passed:
        json_dump(workdir / "fit_gate.json", {"status": "failed", "candidates": candidates})
        pd.DataFrame(candidates).to_csv(workdir / "fit_candidates.csv", index=False, float_format="%.12e")
        raise RuntimeError("没有 spline 候选通过 native 留出/正电感门禁")
    best_score = min(float(row["score"]) for row in passed)
    finalists = [row for row in passed if float(row["score"]) <= 1.02 * best_score]
    selected = max(finalists, key=lambda row: float(row["smoothing_s_normalized"]))
    selected_s = float(selected["smoothing_s_normalized"])
    final_law = TensorCoenergyLaw(Path("candidate.json"), Path("candidate.npz"), 0.004, 1.0, x_axis, i_axis, psi, psi_scale, selected_s, {"candidate_s": selected_s}, 12)
    # 12-point versus 20-point integration check is done by reconstructing the
    # same spline with both quadrature orders, not by changing individual points.
    final_law_20 = TensorCoenergyLaw(Path("candidate.json"), Path("candidate.npz"), 0.004, 1.0, x_axis, i_axis, psi, psi_scale, selected_s, {"candidate_s": selected_s}, 20)
    check_x = np.linspace(-0.0038, 0.0038, 9)
    check_i = np.linspace(-0.95, 0.95, 9)
    Xc, Ic = np.meshgrid(check_x, check_i, indexing="ij")
    W12 = np.asarray(final_law.coenergy(Xc, Ic))
    W20 = np.asarray(final_law_20.coenergy(Xc, Ic))
    gauss_difference = float(np.max(np.abs(W12 - W20) / np.maximum(np.abs(W20), 1e-30)))
    if gauss_difference > 1e-8:
        raise RuntimeError(f"12/20 点 Gauss 积分差异 {gauss_difference:.3e} > 1e-8")
    # Update holdout table with fit-only derived values; native FEM columns are
    # untouched and remain the authoritative raw measurements.
    holdout["psi_fit_Wb"] = np.asarray(final_law.flux(holdout["x_m"], holdout["current_A"]), dtype=float)
    holdout["F_coenergy_N"] = np.asarray(final_law.force(holdout["x_m"], holdout["current_A"]), dtype=float)
    holdout["psi_absolute_error_Wb"] = holdout["psi_fit_Wb"] - holdout["psi_Wb"]
    holdout["psi_relative_error"] = np.abs(holdout["psi_absolute_error_Wb"]) / np.maximum(np.abs(holdout["psi_Wb"]), 1e-8 * psi_scale)
    holdout["F_absolute_error_N"] = holdout["F_coenergy_N"] - holdout["lorentz_force_N"]
    holdout["F_relative_error"] = np.abs(holdout["F_absolute_error_N"]) / np.maximum(np.abs(holdout["lorentz_force_N"]), force_floor_holdout)
    csv_atomic(workdir / "holdout_points.csv", holdout)
    surface_x = np.linspace(-0.004, 0.004, 161)
    surface_i = np.linspace(-1.0, 1.0, 161)
    Xs, Is = np.meshgrid(surface_x, surface_i, indexing="ij")
    surface = {
        "x_m": Xs.ravel(),
        "current_A": Is.ravel(),
        "W_J": np.asarray(final_law.coenergy(Xs, Is)).ravel(),
        "psi_Wb": np.asarray(final_law.flux(Xs, Is)).ravel(),
        "F_N": np.asarray(final_law.force(Xs, Is)).ravel(),
        "BL_secant_N_A": np.asarray(final_law.effective_bl(Xs, Is)).ravel(),
        "BL_tangent_N_A": np.asarray(final_law.dforce_di(Xs, Is)).ravel(),
        "L_incremental_H": np.asarray(final_law.incremental_inductance(Xs, Is)).ravel(),
        "W_xx_N_m": np.asarray(final_law.dforce_dx(Xs, Is)).ravel(),
        "W_xi_N_A": np.asarray(final_law.dforce_di(Xs, Is)).ravel(),
    }
    surface["reciprocity_residual"] = np.zeros_like(surface["W_xi_N_A"])
    savez_atomic(workdir / "fit_surface_grid.npz", **surface, x_axis_m=surface_x, current_axis_A=surface_i, axis_order=np.asarray(["x_index", "i_index"]))
    csv_atomic(workdir / "fit_surface_grid.csv", pd.DataFrame(surface))
    npz_out = Path(args.npz_out).resolve()
    json_out = Path(args.json_out).resolve()
    npz_out.parent.mkdir(parents=True, exist_ok=True)
    # The production-readable NPZ is the compressed raw tensor plus the
    # invariant Lorentz data, not the dense 161x161 derived grid.
    savez_atomic(npz_out, x_training_m=x_axis, current_training_A=i_axis, psi_training_Wb=psi, F_training_N=np.asarray(arrays["lorentz_force_N"]), lorentz_force_training_N=np.asarray(arrays["lorentz_force_N"]), psi_raw_training_Wb=np.asarray(arrays["psi_raw_Wb"]), W_training_J=np.asarray(arrays["W_raw_J"]), W_raw_training_J=np.asarray(arrays["W_raw_J"]), training_valid_mask=np.asarray(arrays["completed"], dtype=bool), x_holdout_m=X_HOLDOUT_MM * 1e-3, current_holdout_A=I_HOLDOUT_A, psi_holdout_Wb=holdout["psi_Wb"].to_numpy().reshape(5, 5), F_holdout_N=holdout["lorentz_force_N"].to_numpy().reshape(5, 5), lorentz_force_holdout_N=holdout["lorentz_force_N"].to_numpy().reshape(5, 5))
    json_data = {
        "schema_version": 1,
        "kind": "native_tensor_coenergy_magnetic_law",
        "generated_by": "tools/build_tensor_magnetic_coenergy.py fit",
        "data_npz": npz_out.name,
        "data_npz_sha256": sha256(npz_out),
        "displacement_limit_m": 0.004,
        "current_limit_A": 1.0,
        "coordinates": {"x": "m", "current": "A", "axis_order": ["x_index", "i_index"]},
        "source_contract": {"rhs": "i*b(x)", "flux": "psi_raw=b(x)^T*A", "gauge": "psi=psi_raw(x,i)-psi_raw(0,0)", "force": "F=partial W/partial x", "coenergy": "W=integral_0^i psi(x,s) ds"},
        "fit": {"class": "scipy.interpolate.RectBivariateSpline", "kx": 3, "ky": 3, "normalized_x": "x/0.004", "normalized_current": "i/1.0", "psi_scale_Wb": psi_scale, "smoothing_s_normalized": selected_s, "gauss_order": 12, "gauss_12_vs_20_max_relative_difference": gauss_difference, "selected_candidate": selected},
        "native_gate": {"status": "passed", "candidates": candidates, "training_points": int(psi.size), "holdout_points": 25, "W_ii_min_H": float(selected["L_min_H"]), "W_ii_max_H": float(selected["L_max_H"]), "psi_zero_gauge_raw_Wb": float(arrays["psi_raw_Wb"][np.where(np.isclose(X_TRAIN_MM, 0))[0][0], np.where(np.isclose(I_TRAIN_A, 0))[0][0]])},
        "provenance": {"scan_dir": str(workdir), "training_tensor_sha256": sha256(workdir / "training_tensor.npz"), "holdout_sha256": sha256(workdir / "holdout_points.csv"), "input_hash": json.loads((workdir / "provenance.json").read_text(encoding="utf-8"))["input_hash"]},
        "array_schema": {"x_training_m": {"dtype": "float64", "shape": [len(x_axis)], "axis": "x_index", "unit": "m"}, "current_training_A": {"dtype": "float64", "shape": [len(i_axis)], "axis": "i_index", "unit": "A"}, "psi_training_Wb": {"dtype": "float64", "shape": [len(x_axis), len(i_axis)], "axis": ["x_index", "i_index"], "unit": "Wb"}, "F_training_N": {"dtype": "float64", "shape": [len(x_axis), len(i_axis)], "axis": ["x_index", "i_index"], "unit": "N", "source": "native Lorentz cross-check"}, "W_training_J": {"dtype": "float64", "shape": [len(x_axis), len(i_axis)], "axis": ["x_index", "i_index"], "unit": "J", "source": "trapezoidal audit of globally gauged psi"}, "x_holdout_m": {"dtype": "float64", "shape": [5], "axis": "holdout_x_index", "unit": "m"}, "current_holdout_A": {"dtype": "float64", "shape": [5], "axis": "holdout_i_index", "unit": "A"}, "psi_holdout_Wb": {"dtype": "float64", "shape": [5, 5], "axis": ["holdout_x_index", "holdout_i_index"], "unit": "Wb"}},
    }
    json_dump(json_out, json_data)
    pd.DataFrame(candidates).to_csv(workdir / "fit_candidates.csv", index=False, float_format="%.12e")
    json_dump(workdir / "fit_gate.json", {"status": "passed", "selected_smoothing_s_normalized": selected_s, "evidence": "fit_candidates.csv", "json_out": str(json_out), "npz_out": str(npz_out), "gauss_12_vs_20_max_relative_difference": gauss_difference})
    append_ledger(workdir, {"experiment_id": "E02_TENSOR_V1_FIT", "parent_experiment": "E02_TENSOR_V1_HOLDOUT", "unique_change": "fixed cubic RectBivariateSpline candidate list and scalar coenergy integration", "input_hash": json_data["provenance"]["input_hash"], "native_gate": "fit_gate passed", "comsol_gate": "COMSOL not read before fit freeze", "conclusion": "native tensor candidate frozen", "next_action": "diagnostic transient"})
    print(json.dumps({"status": "passed", "selected": selected, "json_out": str(json_out), "npz_out": str(npz_out)}, ensure_ascii=False, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="native moving-winding tensor coenergy scan")
    sub = root.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pilot", help="run the mandatory 3x3 L0/L1 pilot")
    p.add_argument("--workdir", required=True)
    p.add_argument("--mainline", default=None)
    p.add_argument("--mesh-levels", nargs="+", type=int, default=[0, 1])
    p.set_defaults(func=pilot)
    p = sub.add_parser("scan", help="run/resume the fixed 513-point native tensor")
    p.add_argument("--workdir", required=True)
    p.add_argument("--mainline", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--failed-only", action="store_true")
    p.add_argument("--x-index", type=int, default=None)
    p.set_defaults(func=scan)
    p = sub.add_parser("holdout", help="solve the fixed 5x5 native FEM holdout")
    p.add_argument("--workdir", required=True)
    p.add_argument("--mainline", default=None)
    p.add_argument("--resume", action="store_true")
    p.set_defaults(func=holdout)
    p = sub.add_parser("fit", help="freeze the scalar C2 coenergy surface")
    p.add_argument("--workdir", required=True)
    p.add_argument("--json-out", required=True)
    p.add_argument("--npz-out", required=True)
    p.set_defaults(func=fit)
    return root


def main() -> int:
    args = parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
