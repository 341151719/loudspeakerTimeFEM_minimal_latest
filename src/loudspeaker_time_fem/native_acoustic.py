from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from .comsol_mesh import ComsolMesh, load_comsol_mphtxt_mesh


ACOUSTIC_DOMAINS = {1, 2, 3, 5, 6}
PML_DOMAINS = {1, 6}
STRUCTURE_DOMAINS = {4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19}


@dataclass
class NativeAcousticModel:
    mesh: ComsolMesh
    acoustic_nodes_global: np.ndarray
    acoustic_node_map: dict[int, int]
    pressure_free_dofs: np.ndarray
    pressure_dirichlet_dofs: np.ndarray
    triangles_global: np.ndarray
    edge_midpoint_nodes: dict[tuple[int, int], int]
    uniform_refinement_levels: int
    pressure_order: int
    acoustic_domains: frozenset[int]
    pml_domains: frozenset[int]
    Kp: csr_matrix
    Mp: csr_matrix
    Mp_pml: csr_matrix


def _triangle_area_grads(points: np.ndarray) -> tuple[float, np.ndarray]:
    r0, z0 = points[0]
    r1, z1 = points[1]
    r2, z2 = points[2]
    determinant = (r1 - r0) * (z2 - z0) - (r2 - r0) * (z1 - z0)
    area = 0.5 * abs(determinant)
    if area <= 0:
        raise ValueError("degenerate acoustic triangle")
    gradients = np.array(
        [
            [(z1 - z2) / determinant, (r2 - r1) / determinant],
            [(z2 - z0) / determinant, (r0 - r2) / determinant],
            [(z0 - z1) / determinant, (r1 - r0) / determinant],
        ]
    )
    return area, gradients


def _barycentric_integral(area: float, exponents: tuple[int, int, int]) -> float:
    numerator = 2.0 * area
    for exponent in exponents:
        numerator *= math.factorial(exponent)
    return numerator / math.factorial(sum(exponents) + 2)


def _element_matrices(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Axisymmetric P1 matrices with exact integration of the linear radius."""
    area, gradients = _triangle_area_grads(points)
    radial_integral = area * float(np.mean(points[:, 0]))
    stiffness = 2.0 * math.pi * radial_integral * (gradients @ gradients.T)
    mass = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                powers = [0, 0, 0]
                powers[i] += 1
                powers[j] += 1
                powers[k] += 1
                mass[i, j] += (
                    2.0
                    * math.pi
                    * points[k, 0]
                    * _barycentric_integral(area, tuple(powers))
                )
    return stiffness, mass


def _element_matrices_p2(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Axisymmetric affine-geometry P2 matrices using degree-five quadrature."""
    area, barycentric_gradients = _triangle_area_grads(points)
    quadrature = [
        ((1 / 3, 1 / 3, 1 / 3), 0.225),
        ((0.470142064105115, 0.470142064105115, 0.059715871789770), 0.132394152788506),
        ((0.470142064105115, 0.059715871789770, 0.470142064105115), 0.132394152788506),
        ((0.059715871789770, 0.470142064105115, 0.470142064105115), 0.132394152788506),
        ((0.101286507323456, 0.101286507323456, 0.797426985353087), 0.125939180544827),
        ((0.101286507323456, 0.797426985353087, 0.101286507323456), 0.125939180544827),
        ((0.797426985353087, 0.101286507323456, 0.101286507323456), 0.125939180544827),
    ]
    stiffness = np.zeros((6, 6))
    mass = np.zeros((6, 6))
    for barycentric, weight in quadrature:
        l0, l1, l2 = barycentric
        shape = np.array(
            [
                l0 * (2 * l0 - 1),
                l1 * (2 * l1 - 1),
                l2 * (2 * l2 - 1),
                4 * l0 * l1,
                4 * l1 * l2,
                4 * l2 * l0,
            ]
        )
        gradients = np.vstack(
            [
                (4 * l0 - 1) * barycentric_gradients[0],
                (4 * l1 - 1) * barycentric_gradients[1],
                (4 * l2 - 1) * barycentric_gradients[2],
                4 * (l0 * barycentric_gradients[1] + l1 * barycentric_gradients[0]),
                4 * (l1 * barycentric_gradients[2] + l2 * barycentric_gradients[1]),
                4 * (l2 * barycentric_gradients[0] + l0 * barycentric_gradients[2]),
            ]
        )
        radius = float(np.dot(np.asarray(barycentric), points[:, 0]))
        factor = 2.0 * math.pi * radius * area * weight
        stiffness += factor * (gradients @ gradients.T)
        mass += factor * np.outer(shape, shape)
    return stiffness, mass


def build_native_acoustic(
    mesh_path: str | Path,
    acoustic_domains: set[int] | None = None,
    pml_domains: set[int] | None = None,
    uniform_refinement_levels: int = 0,
    pressure_order: int = 1,
    mesh_override: ComsolMesh | None = None,
) -> NativeAcousticModel:
    mesh = mesh_override if mesh_override is not None else load_comsol_mphtxt_mesh(mesh_path)
    selected = set(ACOUSTIC_DOMAINS if acoustic_domains is None else acoustic_domains)
    pml_selected = set(PML_DOMAINS if pml_domains is None else pml_domains)
    triangles, domains = mesh.triangulated_domains(selected)
    order = int(pressure_order)
    if order not in (1, 2):
        raise ValueError("native acoustic pressure_order must be 1 or 2")
    levels = int(uniform_refinement_levels)
    if order == 2 and levels:
        raise ValueError("P2 pressure and uniform P1 refinement are separate diagnostics")
    if levels not in (0, 1):
        raise ValueError("native acoustic uniform_refinement_levels currently supports 0 or 1")
    edge_midpoint_nodes: dict[tuple[int, int], int] = {}
    if levels == 1 or order == 2:
        points = mesh.points_rz_m.tolist()

        def midpoint_node(a: int, b: int) -> int:
            key = tuple(sorted((int(a), int(b))))
            if key not in edge_midpoint_nodes:
                edge_midpoint_nodes[key] = len(points)
                points.append(
                    (0.5 * (mesh.points_rz_m[key[0]] + mesh.points_rz_m[key[1]])).tolist()
                )
            return edge_midpoint_nodes[key]

        refined_triangles: list[list[int]] = []
        refined_domains: list[int] = []
        for (a, b, c), domain in zip(triangles, domains):
            ab = midpoint_node(a, b)
            bc = midpoint_node(b, c)
            ca = midpoint_node(c, a)
            if order == 2:
                refined_triangles.append([a, b, c, ab, bc, ca])
                refined_domains.append(int(domain))
            else:
                refined_triangles.extend(
                    ([a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca])
                )
                refined_domains.extend([int(domain)] * 4)
        triangles = np.asarray(refined_triangles, dtype=int)
        domains = np.asarray(refined_domains, dtype=int)
        mesh = ComsolMesh(np.asarray(points, dtype=float), mesh.cells, mesh.entities)
    acoustic_nodes = np.unique(triangles)
    node_map = {int(global_node): local for local, global_node in enumerate(acoustic_nodes)}
    size = len(acoustic_nodes)
    rows: list[int] = []
    cols: list[int] = []
    kvals: list[float] = []
    mvals: list[float] = []
    pmlvals: list[float] = []
    for triangle, domain in zip(triangles, domains):
        local = [node_map[int(node)] for node in triangle]
        vertices = mesh.points_rz_m[triangle[:3]]
        stiffness, mass = (
            _element_matrices_p2(vertices) if order == 2 else _element_matrices(vertices)
        )
        for i in range(len(triangle)):
            for j in range(len(triangle)):
                rows.append(local[i])
                cols.append(local[j])
                kvals.append(float(stiffness[i, j]))
                mvals.append(float(mass[i, j]))
                pmlvals.append(
                    float(mass[i, j]) if int(domain) in pml_selected else 0.0
                )
    indices = (rows, cols)
    stiffness = coo_matrix((kvals, indices), shape=(size, size)).tocsr()
    mass = coo_matrix((mvals, indices), shape=(size, size)).tocsr()
    pml_mass = coo_matrix((pmlvals, indices), shape=(size, size)).tocsr()
    return NativeAcousticModel(
        mesh=mesh,
        acoustic_nodes_global=acoustic_nodes,
        acoustic_node_map=node_map,
        pressure_free_dofs=np.arange(size, dtype=int),
        pressure_dirichlet_dofs=np.empty(0, dtype=int),
        triangles_global=triangles,
        edge_midpoint_nodes=edge_midpoint_nodes,
        uniform_refinement_levels=levels,
        pressure_order=order,
        acoustic_domains=frozenset(selected),
        pml_domains=frozenset(pml_selected),
        Kp=stiffness,
        Mp=mass,
        Mp_pml=pml_mass,
    )


def _edge_shapes(t: float) -> np.ndarray:
    return np.array([(1.0 - t) * (1.0 - 2.0 * t), t * (2.0 * t - 1.0), 4.0 * t * (1.0 - t)])


def _cell_sides(mesh: ComsolMesh) -> dict[tuple[int, int], list[tuple[int, np.ndarray]]]:
    sides: dict[tuple[int, int], list[tuple[int, np.ndarray]]] = {}
    for kind in ("tri", "quad"):
        for cell, domain in zip(mesh.cells[kind], mesh.entities[kind]):
            centroid = np.mean(mesh.points_rz_m[cell], axis=0)
            ordered = cell if kind == "tri" else cell[[0, 1, 3, 2]]
            for a, b in zip(ordered, np.roll(ordered, -1)):
                key = tuple(sorted((int(a), int(b))))
                sides.setdefault(key, []).append((int(domain), centroid))
    return sides


def assemble_native_nonconforming_G(
    acoustic: NativeAcousticModel,
    solid: Any,
    reference_interface_tags: set[int],
) -> tuple[csr_matrix, dict[str, Any]]:
    """Project the old P2 structure onto the native transient acoustic interface.

    The native COMSOL interface is the integration geometry. At each quadrature
    point, P2 structural shape functions are evaluated on the closest reference
    structural edge. The same matrix and its transpose are used in the two weak
    equations, preserving discrete work reciprocity.
    """
    candidates = []
    for va, vb, tag in solid.boundary_edges:
        if int(tag) not in reference_interface_tags:
            continue
        p0 = solid.points_rz_m[int(va)]
        p1 = solid.points_rz_m[int(vb)]
        midpoint = solid.edge_mid_nodes[tuple(sorted((int(va), int(vb))))]
        candidates.append((int(va), int(vb), int(midpoint), p0, p1))
    if not candidates:
        raise ValueError("no reference structural interface edges")

    interfaces = acoustic.mesh.acoustic_interface_edges(
        set(acoustic.acoustic_domains), STRUCTURE_DOMAINS
    )
    sides = _cell_sides(acoustic.mesh)
    xg, wg = np.polynomial.legendre.leggauss(4)
    parameters = 0.5 * (xg + 1.0)
    weights = 0.5 * wg
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    projection_distances = []
    boundary_entities = []
    for ga, gb, acoustic_domain, structure_domain, boundary_entity in interfaces:
        p0 = acoustic.mesh.points_rz_m[ga]
        p1 = acoustic.mesh.points_rz_m[gb]
        tangent = p1 - p0
        length = float(np.linalg.norm(tangent))
        if length <= 0:
            continue
        normal = np.array([tangent[1], -tangent[0]]) / length
        adjacent = sides[tuple(sorted((ga, gb)))]
        acoustic_centroid = next(c for d, c in adjacent if d == acoustic_domain)
        structure_centroid = next(c for d, c in adjacent if d == structure_domain)
        if np.dot(normal, structure_centroid - acoustic_centroid) < 0:
            normal = -normal
        midpoint_global = acoustic.edge_midpoint_nodes.get(tuple(sorted((ga, gb))))
        pressure_nodes = [acoustic.acoustic_node_map[ga], acoustic.acoustic_node_map[gb]]
        if midpoint_global is not None:
            pressure_nodes.append(acoustic.acoustic_node_map[midpoint_global])
        for t, weight in zip(parameters, weights):
            point = (1.0 - t) * p0 + t * p1
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
            distance, va, vb, midpoint, tau = best
            projection_distances.append(distance)
            ns = _edge_shapes(tau)
            if acoustic.pressure_order == 2:
                np_shape = _edge_shapes(t)
            elif midpoint_global is None:
                np_shape = np.array([1.0 - t, t])
            elif t <= 0.5:
                np_shape = np.array([1.0 - 2.0 * t, 0.0, 2.0 * t])
            else:
                np_shape = np.array([0.0, 2.0 * t - 1.0, 2.0 * (1.0 - t)])
            axisymmetric_weight = (
                2.0 * math.pi * max(float(point[0]), 1e-12) * length * weight
            )
            for i, structural_node in enumerate((va, vb, midpoint)):
                for j, pressure_node in enumerate(pressure_nodes):
                    value = axisymmetric_weight * ns[i] * np_shape[j]
                    rows.extend((2 * structural_node, 2 * structural_node + 1))
                    cols.extend((pressure_node, pressure_node))
                    values.extend((normal[0] * value, normal[1] * value))
        boundary_entities.append(boundary_entity)
    matrix = coo_matrix(
        (values, (rows, cols)), shape=(solid.ndof, len(acoustic.acoustic_nodes_global))
    ).tocsr()
    distances = np.asarray(projection_distances)
    return matrix, {
        "G_shape": list(matrix.shape),
        "native_interface_edges": len(interfaces),
        "native_interface_boundary_entities": sorted(set(boundary_entities)),
        "reference_interface_boundaries": sorted(reference_interface_tags),
        "quadrature_order": 4,
        "pressure_trace_order": acoustic.pressure_order,
        "pressure_trace_segments_per_native_edge": (
            2 if acoustic.uniform_refinement_levels else 1
        ),
        "structure_trace_order": 2,
        "mesh_coupling": "native_transient_acoustic_to_reference_P2_closest_edge",
        "projection_distance_median_m": float(np.median(distances)),
        "projection_distance_p95_m": float(np.quantile(distances, 0.95)),
        "projection_distance_max_m": float(np.max(distances)),
        "reciprocity_contract": "same G used for pressure work and rho*G.T acceleration source",
    }
