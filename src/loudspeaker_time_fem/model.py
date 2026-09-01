from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.sparse import bmat, coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import splu

from .config import resolve_base_mainline


@dataclass
class ProbeMap:
    names: list[str]
    requested_rz_m: np.ndarray
    actual_rz_m: np.ndarray
    pressure_dofs: np.ndarray
    pressure_matrix: csr_matrix
    distances_m: np.ndarray


@dataclass
class TransientModel:
    config: dict[str, Any]
    config_path: Path
    base_mainline: Path
    M: csr_matrix
    C: csr_matrix
    K: csr_matrix
    force_vector: np.ndarray
    back_emf_vector: np.ndarray
    R_ohm: float
    L_H: float
    n_solid: int
    n_pressure: int
    solid_free_dofs: np.ndarray
    pressure_free_dofs: np.ndarray
    solid: Any
    acoustic: Any
    mesh: Any
    G: csr_matrix
    probes: ProbeMap
    pml_sigma_by_pressure_dof: np.ndarray
    nonlinear_law: Any | None
    suspension_law: Any | None
    metadata: dict[str, Any]


def _import_frequency_mainline(base: Path):
    for entry in (base / "src", base / "best_model"):
        value = str(entry)
        if value not in sys.path:
            sys.path.insert(0, value)
    from loudspeaker_axisym_fem.axisym_magnetics import load_tagged_meshio
    from loudspeaker_axisym_fem.stage4C_acoustic_structure import (
        build_stage4C_acoustic_structure_model,
    )
    from p2_axisym_solid import (
        assemble_lorentz_force,
        assemble_p2_G,
        build_p2_solid,
    )
    return (
        load_tagged_meshio,
        build_stage4C_acoustic_structure_model,
        build_p2_solid,
        assemble_p2_G,
        assemble_lorentz_force,
    )


def _stiffness_proportional_damping(
    solid: Any,
    reference_Hz: float,
    reference_by_label_Hz: dict[str, float] | None = None,
) -> csr_matrix:
    out = csr_matrix((solid.ndof, solid.ndof), dtype=float)
    for domain, matrix in solid.K_by_domain.items():
        material = solid.materials[int(domain)]
        label_reference = (reference_by_label_Hz or {}).get(
            str(material.label), reference_Hz
        )
        omega_ref = 2.0 * math.pi * float(label_reference)
        coefficient_s = float(material.beta_dK) + float(material.loss_factor) / omega_ref
        if coefficient_s:
            out = out + coefficient_s * matrix
    return out.tocsr()


def _pml_sigma(
    acoustic_points: np.ndarray,
    inner_radius_m: float,
    outer_radius_m: float,
    order: int,
    c0: float,
    target_amplitude: float,
) -> np.ndarray:
    radius = np.linalg.norm(acoustic_points, axis=1)
    thickness = float(outer_radius_m - inner_radius_m)
    if thickness <= 0:
        raise ValueError("absorbing_layer outer_radius_m 必须大于 inner_radius_m")
    xi = np.clip((radius - inner_radius_m) / thickness, 0.0, 1.0)
    # Integral sigma/c dx = -log(target); sigma=sigma_max*xi^order.
    sigma_max = -(order + 1) * c0 * math.log(max(target_amplitude, 1e-12)) / thickness
    return sigma_max * xi**order


def _build_probe_map(config: dict[str, Any], acoustic: Any, pressure_free: np.ndarray) -> ProbeMap:
    points = acoustic.mesh.points_rz_m[acoustic.acoustic_nodes_global]
    free_lookup = {int(global_dof): local for local, global_dof in enumerate(pressure_free)}
    if hasattr(acoustic, "triangles_global"):
        global_triangles = np.asarray(acoustic.triangles_global, dtype=int)
    elif hasattr(acoustic.mesh, "triangles"):
        global_triangles = np.asarray(acoustic.mesh.triangles, dtype=int)
        keep = np.all(
            np.isin(global_triangles, acoustic.acoustic_nodes_global), axis=1
        )
        global_triangles = global_triangles[keep]
    else:
        raise ValueError("acoustic model does not expose triangle topology")
    local_triangles = np.asarray(
        [
            [acoustic.acoustic_node_map[int(node)] for node in triangle]
            for triangle in global_triangles
        ],
        dtype=int,
    )
    names: list[str] = []
    requested: list[list[float]] = []
    actual: list[np.ndarray] = []
    dofs: list[int] = []
    distances: list[float] = []
    matrix_rows: list[int] = []
    matrix_cols: list[int] = []
    matrix_values: list[float] = []
    for item in config.get("probes", []):
        target = np.array([float(item["r_m"]), float(item["z_m"])])
        best = None
        for triangle in local_triangles:
            vertices = points[triangle[:3]]
            if np.any(target < vertices.min(axis=0) - 1e-12) or np.any(
                target > vertices.max(axis=0) + 1e-12
            ):
                continue
            transform = np.column_stack(
                (vertices[1] - vertices[0], vertices[2] - vertices[0])
            )
            try:
                ab = np.linalg.solve(transform, target - vertices[0])
            except np.linalg.LinAlgError:
                continue
            weights = np.array([1.0 - ab.sum(), ab[0], ab[1]])
            margin = float(np.min(weights))
            if margin >= -1e-9 and (best is None or margin > best[0]):
                best = (margin, triangle, weights)
        names.append(str(item["name"]))
        requested.append(target.tolist())
        row = len(names) - 1
        if best is None:
            action = str(item.get("outside_domain_action", "error"))
            if action != "nearest_boundary_diagnostic":
                raise ValueError(
                    f"probe {item['name']!r} at {target.tolist()} is outside the "
                    "solved acoustic domain; set outside_domain_action="
                    "'nearest_boundary_diagnostic' only for an explicitly "
                    "non-physical diagnostic"
                )
            candidates = np.asarray(pressure_free, dtype=int)
            delta = points[candidates] - target
            j = int(np.argmin(np.einsum("ij,ij->i", delta, delta)))
            base_dof = int(candidates[j])
            actual_point = points[base_dof]
            interpolation = [(free_lookup[base_dof], 1.0)]
        else:
            _, triangle, weights = best
            if len(triangle) == 6:
                l0, l1, l2 = weights
                interpolation_weights = np.array(
                    [
                        l0 * (2 * l0 - 1),
                        l1 * (2 * l1 - 1),
                        l2 * (2 * l2 - 1),
                        4 * l0 * l1,
                        4 * l1 * l2,
                        4 * l2 * l0,
                    ]
                )
            else:
                interpolation_weights = weights
            available = [
                (free_lookup[int(node)], float(weight))
                for node, weight in zip(triangle, interpolation_weights)
                if int(node) in free_lookup
            ]
            if len(available) != len(triangle):
                raise ValueError("probe triangle includes constrained pressure DOF")
            interpolation = available
            actual_point = target.copy()
            base_dof = int(triangle[int(np.argmax(np.abs(interpolation_weights)))])
        for column, value in interpolation:
            matrix_rows.append(row)
            matrix_cols.append(column)
            matrix_values.append(value)
        actual.append(actual_point)
        dofs.append(free_lookup[base_dof])
        distances.append(float(np.linalg.norm(actual_point - target)))
    if not names:
        raise ValueError("至少需要一个压力探针")
    return ProbeMap(
        names,
        np.asarray(requested, float),
        np.asarray(actual, float),
        np.asarray(dofs, int),
        coo_matrix(
            (matrix_values, (matrix_rows, matrix_cols)),
            shape=(len(names), len(pressure_free)),
        ).tocsr(),
        np.asarray(distances, float),
    )


def _outer_radiation_matrices(
    acoustic: Any,
    pressure_free: np.ndarray,
    outer_radius_m: float,
    c0: float,
    curvature_correction: bool,
    robin_stiffness_1_per_m: float | None = None,
    robin_damping_s_per_m: float | None = None,
    robin_legendre_real_1_per_m: list[float] | None = None,
    robin_legendre_imag_1_per_m: list[float] | None = None,
    reference_omega_rad_s: float | None = None,
) -> tuple[csr_matrix, csr_matrix, int]:
    """Assemble spherical first-order outgoing-wave boundary matrices.

    For p=f(t-r/c)/r, dp/dr + p/r + p_t/c = 0. The p/r term is
    essential when kR is small; omitting it turns the low-frequency response
    into an artificial damping-controlled cavity.
    """
    amap = acoustic.acoustic_node_map
    free_lookup = {int(base): local for local, base in enumerate(pressure_free)}
    rows: list[int] = []
    cols: list[int] = []
    damping_values: list[float] = []
    stiffness_values: list[float] = []
    segments = 0
    tolerance = max(2e-5, 2e-3 * outer_radius_m)
    for segment in acoustic.mesh.line_cells:
        ga, gb = map(int, segment)
        if ga not in amap or gb not in amap:
            continue
        p0 = acoustic.mesh.points_rz_m[ga]
        p1 = acoustic.mesh.points_rz_m[gb]
        if max(
            abs(float(np.linalg.norm(p0)) - outer_radius_m),
            abs(float(np.linalg.norm(p1)) - outer_radius_m),
        ) > tolerance:
            continue
        base_nodes = [ga, gb]
        midpoint = getattr(acoustic, "edge_midpoint_nodes", {}).get(
            tuple(sorted((ga, gb)))
        )
        if midpoint is not None:
            base_nodes.append(midpoint)
        local_nodes = [int(amap[node]) for node in base_nodes]
        if any(node not in free_lookup for node in local_nodes):
            continue
        length = float(np.linalg.norm(p1 - p0))
        xg, wg = np.polynomial.legendre.leggauss(3)
        local_damping = np.zeros((len(local_nodes), len(local_nodes)))
        local_stiffness = np.zeros((len(local_nodes), len(local_nodes)))
        for xi, weight in zip(xg, wg):
            t = 0.5 * (xi + 1.0)
            if getattr(acoustic, "pressure_order", 1) == 2:
                shape = np.array(
                    [
                        (1.0 - t) * (1.0 - 2.0 * t),
                        t * (2.0 * t - 1.0),
                        4.0 * t * (1.0 - t),
                    ]
                )
            elif midpoint is None:
                shape = np.array([1.0 - t, t])
            elif t <= 0.5:
                shape = np.array([1.0 - 2.0 * t, 0.0, 2.0 * t])
            else:
                shape = np.array([0.0, 2.0 * t - 1.0, 2.0 * (1.0 - t)])
            radius = max(float(((1.0 - t) * p0 + t * p1)[0]), 1e-12)
            weighted_shape = (
                2.0
                * math.pi
                * radius
                * length
                * 0.5
                * weight
                * np.outer(shape, shape)
            )
            point = (1.0 - t) * p0 + t * p1
            if robin_legendre_real_1_per_m is not None:
                cosine = float(point[1] / outer_radius_m)
                stiffness_coefficient = float(
                    np.polynomial.legendre.legval(cosine, robin_legendre_real_1_per_m)
                )
                imag_coefficient = float(
                    np.polynomial.legendre.legval(
                        cosine, robin_legendre_imag_1_per_m or [0.0]
                    )
                )
                damping_coefficient = imag_coefficient / float(reference_omega_rad_s)
            else:
                stiffness_coefficient = (
                    float(robin_stiffness_1_per_m)
                    if robin_stiffness_1_per_m is not None
                    else 1.0 / outer_radius_m
                )
                damping_coefficient = (
                    float(robin_damping_s_per_m)
                    if robin_damping_s_per_m is not None
                    else 1.0 / c0
                )
            local_damping += damping_coefficient * weighted_shape
            if curvature_correction:
                local_stiffness += stiffness_coefficient * weighted_shape
        local_dofs = [free_lookup[node] for node in local_nodes]
        for i in range(len(local_dofs)):
            for j in range(len(local_dofs)):
                rows.append(local_dofs[i])
                cols.append(local_dofs[j])
                damping_values.append(float(local_damping[i, j]))
                stiffness_values.append(float(local_stiffness[i, j]))
        segments += 1
    damping = coo_matrix(
        (damping_values, (rows, cols)),
        shape=(len(pressure_free), len(pressure_free)),
    ).tocsr()
    stiffness = coo_matrix(
        (stiffness_values, (rows, cols)),
        shape=(len(pressure_free), len(pressure_free)),
    ).tocsr()
    return damping, stiffness, segments


def build_transient_model(config: dict[str, Any], config_path: Path) -> TransientModel:
    base = resolve_base_mainline(config, config_path)
    (
        load_mesh,
        build_acoustic,
        build_solid,
        assemble_G,
        assemble_lorentz,
    ) = _import_frequency_mainline(base)
    mesh_path = base / config["mesh"]
    mphtxt_path = base / config["mphtxt"]
    magnetic_path = base / config["magnetostatic_vtu"]
    mesh = load_mesh(mesh_path)
    c0 = float(config["air"]["c0_m_s"])
    rho0 = float(config["air"]["rho0_kg_m3"])
    solid = build_solid(mesh)
    reference_acoustic = build_acoustic(
        mesh, mphtxt_path, solid_uniform_refine=0, c0=c0
    )
    reference_G, reference_G_info = assemble_G(
        reference_acoustic, solid, pressure_operator=None
    )
    acoustic_contract = config.get("acoustic_contract", {})
    if acoustic_contract.get("kind") == "comsol_transient_native_geometry":
        from .native_acoustic import (
            assemble_native_nonconforming_G,
            build_native_acoustic,
        )

        native_mesh_path = Path(acoustic_contract["mesh"])
        if not native_mesh_path.is_absolute():
            native_mesh_path = config_path.parent.parent / native_mesh_path
        acoustic = build_native_acoustic(
            native_mesh_path,
            set(map(int, acoustic_contract.get("acoustic_domains", [1, 2, 3, 5, 6]))),
            set(map(int, acoustic_contract.get("pml_domains", [1, 6]))),
            int(acoustic_contract.get("uniform_refinement_levels", 0)),
            int(acoustic_contract.get("pressure_order", 1)),
        )
        G_full, G_info = assemble_native_nonconforming_G(
            acoustic,
            solid,
            set(map(int, reference_G_info["interface_boundaries"])),
        )
        output_mesh = acoustic.mesh
    else:
        acoustic = reference_acoustic
        G_full, G_info = reference_G, reference_G_info
        native_mesh_path = None
        output_mesh = mesh
    force_full_complex, lorentz_info = assemble_lorentz(solid, magnetic_path)
    force_full = np.asarray(force_full_complex.real, float)

    sf = np.asarray(solid.free_dofs, int)
    # The frequency mainline marks the outer truncation as pressure Dirichlet for
    # legacy non-PML branches. In time domain we retain every acoustic DOF and
    # replace that truncation with a first-order radiation condition plus sponge.
    pf = np.arange(len(acoustic.acoustic_nodes_global), dtype=int)
    Ms = solid.M[sf][:, sf].tocsr()
    Ks = solid.K_real[sf][:, sf].tocsr()
    Cs = _stiffness_proportional_damping(
        solid,
        float(config["structure"]["damping_reference_Hz"]),
        {
            str(key): float(value)
            for key, value in config["structure"]
            .get("damping_reference_by_label_Hz", {})
            .items()
        },
    )[sf][:, sf].tocsr()
    Ma = (acoustic.Mp[pf][:, pf] / (c0 * c0)).tocsr()
    Ka = acoustic.Kp[pf][:, pf].tocsr()
    G = G_full[sf][:, pf].tocsr()

    absorbing = config["absorbing_layer"]
    acoustic_points = acoustic.mesh.points_rz_m[acoustic.acoustic_nodes_global]
    sigma_all = _pml_sigma(
        acoustic_points,
        float(absorbing["inner_radius_m"]),
        float(absorbing["outer_radius_m"]),
        int(absorbing["polynomial_order"]),
        c0,
        float(absorbing["target_one_way_amplitude"]),
    )
    # Lump only the reused PML-domain mass. This makes C positive diagonal and
    # avoids a frequency-dependent complex coordinate map in the time system.
    pml_lumped_all = np.asarray(acoustic.Mp_pml.sum(axis=1)).ravel()
    pml_damping = 2.0 * sigma_all[pf] * pml_lumped_all[pf] / (c0 * c0)
    Ca = diags(pml_damping, format="csr")
    radiation_segments = 0
    if bool(absorbing.get("outer_first_order_radiation_condition", True)):
        radiation, radiation_stiffness, radiation_segments = _outer_radiation_matrices(
            acoustic,
            pf,
            float(absorbing["outer_radius_m"]),
            c0,
            bool(absorbing.get("outer_curvature_correction", False)),
            absorbing.get("robin_stiffness_1_per_m"),
            absorbing.get("robin_damping_s_per_m"),
            absorbing.get("robin_legendre_real_1_per_m"),
            absorbing.get("robin_legendre_imag_1_per_m"),
            absorbing.get("reference_omega_rad_s"),
        )
        Ca = Ca + radiation
        Ka = Ka + radiation_stiffness

    ns = len(sf)
    npres = len(pf)
    zero_sp = csr_matrix((ns, npres))
    zero_ps = csr_matrix((npres, ns))
    # Weak time form:
    # Ms*u_dd + Cs*u_d + Ks*u - G*p = g*i
    # Ma*p_dd + Ca*p_d + Ka*p + rho0*G.T*u_dd = 0
    M = bmat([[Ms, zero_sp], [rho0 * G.T, Ma]], format="csr")
    C = bmat([[Cs, zero_sp], [zero_ps, Ca]], format="csr")
    K = bmat([[Ks, -G], [zero_ps, Ka]], format="csr")
    force = force_full[sf]
    probes = _build_probe_map(config, acoustic, pf)
    bl = float(np.sum(force_full[1::2]))
    metadata = {
        "formulation": "linear_monolithic_structure_acoustic_electric",
        "mesh_path": str(mesh_path),
        "acoustic_contract": {
            **acoustic_contract,
            "native_mesh_path": (
                str(native_mesh_path) if native_mesh_path is not None else None
            ),
        },
        "mphtxt_path": str(mphtxt_path),
        "magnetostatic_vtu": str(magnetic_path),
        "n_total_second_order_dofs": int(ns + npres),
        "n_solid_free_dofs": int(ns),
        "n_pressure_free_dofs": int(npres),
        "matrix_nnz": {"M": int(M.nnz), "C": int(C.nnz), "K": int(K.nnz)},
        "G": G_info,
        "lorentz": lorentz_info,
        "BL_axial_N_per_A": bl,
        "absorbing_layer": {
            **absorbing,
            "sigma_max_per_s": float(np.max(sigma_all)),
            "active_pressure_dofs": int(np.count_nonzero(pml_damping)),
            "outer_radiation_segments": int(radiation_segments),
            "note": "reuses frequency-domain PML geometry; time-domain polynomial sponge, not complex-coordinate frequency PML",
        },
        "probe_mapping": [
            {
                "name": name,
                "requested_rz_m": requested.tolist(),
                "actual_rz_m": actual.tolist(),
                "distance_m": float(distance),
                "mapping_kind": (
                    "exact_or_element_interpolation"
                    if distance <= 1e-10
                    else "nearest_boundary_diagnostic"
                ),
            }
            for name, requested, actual, distance in zip(
                probes.names,
                probes.requested_rz_m,
                probes.actual_rz_m,
                probes.distances_m,
            )
        ],
    }
    suspension_law = None
    suspension_cfg = config.get("mechanical_nonlinearity", {})
    if bool(suspension_cfg.get("enabled", False)):
        from .suspension_rom import SuspensionROM

        bl_for_coordinate = float(bl)
        if abs(bl_for_coordinate) <= 1e-12:
            raise RuntimeError("cannot define suspension generalized coordinate because BL is zero")
        h_susp = force / bl_for_coordinate
        reference_mode = str(
            suspension_cfg.get("reference_stiffness_mode", "linear_fem_suspension_regions")
        )
        static_solution = splu(Ks.tocsc()).solve(h_susp)
        total_compliance_m_N = float(h_susp @ static_solution)
        if total_compliance_m_N <= 0.0:
            raise RuntimeError("linear FEM generalized compliance must be positive")
        total_generalized_stiffness = 1.0 / total_compliance_m_N
        static_ritz_shape = static_solution / total_compliance_m_N
        region_contributions: dict[int, float] = {}
        if reference_mode == "linear_fem_suspension_regions":
            suspension_domains = [int(v) for v in suspension_cfg.get("suspension_domain_ids", [20, 25])]
            for domain in suspension_domains:
                if domain not in solid.K_by_domain:
                    raise KeyError(f"suspension domain {domain} is absent from structural FEM")
                k_domain = solid.K_by_domain[domain][sf][:, sf]
                region_contributions[domain] = float(
                    static_ritz_shape @ (k_domain @ static_ritz_shape)
                )
            reference_stiffness = float(sum(region_contributions.values()))
            if reference_stiffness <= 0.0:
                raise RuntimeError("projected suspension-region stiffness must be positive")
            compliance_m_N = 1.0 / reference_stiffness
            reference_source = (
                "static Ritz projection sum(phi^T K_domain phi) over suspension domains "
                + str(suspension_domains)
            )
        elif reference_mode == "linear_fem_static_compliance":
            reference_stiffness = total_generalized_stiffness
            compliance_m_N = total_compliance_m_N
            reference_source = "1/(h^T Ks^-1 h) from assembled linear FEM"
            suspension_domains = []
        elif reference_mode == "explicit":
            reference_stiffness = float(suspension_cfg["reference_stiffness_N_m"])
            compliance_m_N = 1.0 / reference_stiffness
            reference_source = "explicit config value"
            suspension_domains = []
        else:
            raise ValueError(
                "mechanical_nonlinearity.reference_stiffness_mode must be "
                "'linear_fem_suspension_regions', 'linear_fem_static_compliance' or 'explicit'"
            )
        suspension_path = Path(suspension_cfg["law"])
        if not suspension_path.is_absolute():
            suspension_path = config_path.parent.parent / suspension_path
        suspension_law = SuspensionROM.from_json(
            suspension_path, reference_stiffness_N_m=reference_stiffness
        )
        metadata["mechanical_nonlinearity"] = {
            **suspension_cfg,
            **suspension_law.diagnostics(),
            "reference_stiffness_source": reference_source,
            "reference_compliance_equivalent_m_N": compliance_m_N,
            "linear_fem_total_generalized_stiffness_N_m": total_generalized_stiffness,
            "linear_fem_total_compliance_m_N": total_compliance_m_N,
            "suspension_domain_ids": suspension_domains,
            "suspension_region_stiffness_contributions_N_m": {
                str(domain): value for domain, value in region_contributions.items()
            },
            "suspension_fraction_of_generalized_stiffness": (
                reference_stiffness / total_generalized_stiffness
                if reference_mode == "linear_fem_suspension_regions" else None
            ),
            "coordinate_definition": "q = h^T u, h = Lorentz force vector / BL(0)",
            "force_contract": "F_s(q)=Kms(q)q; FEM receives DeltaF=F_s-Kms(0)q",
            "tangent_contract": "dDeltaF/dq=d[Kms(q)q]/dq-Kms(0)",
            "small_signal_preservation": "DeltaF(0)=0 and dDeltaF/dq|0=0",
        }

    nonlinear_law = None
    nonlinear_cfg = config.get("nonlinear", {})
    if bool(nonlinear_cfg.get("enabled", False)):
        from .nonlinear_law import NonlinearMagneticLaw

        law_path = Path(nonlinear_cfg["law"])
        if not law_path.is_absolute():
            law_path = config_path.parent.parent / law_path
        nonlinear_law = NonlinearMagneticLaw.from_json(law_path)
        is_tensor_coenergy = nonlinear_law.__class__.__name__ == "TensorCoenergyLaw"
        metadata["formulation"] = (
            "monolithic_structure_acoustic_electric_with_"
            + (
                "native_tensor_coenergy_magnetic_ROM"
                if is_tensor_coenergy
                else "field_derived_nonlinear_magnetic_ALE_ROM"
            )
        )
        if is_tensor_coenergy:
            coenergy_contract = {
                "force": "F = partial W/partial x",
                "flux": "psi = partial W/partial i",
                "coenergy": "W = integral_0^i psi(x,s) ds",
                "reciprocity": "dF/di = dpsi/dx = W_xi",
                "source": "native nonlinear FEM tensor; no transient COMSOL response identification",
            }
        elif bool(config["nonlinear"].get("coupled_coenergy_enabled", False)):
            coenergy_contract = {
                "force": "F = i*[BL(x)+deltaBL(i)]",
                "flux": "psi = lambda_i(i)+integral_0^x BL(s)ds+x*[deltaBL(i)+i*deltaBL'(i)]",
                "reciprocity": "dF/di = dpsi/dx by construction",
                "source": "native magnetostatic displacement and current scans; no transient COMSOL response identification",
            }
        else:
            coenergy_contract = nonlinear_law.metadata["coenergy_contract"]
        metadata["nonlinear"] = {
            **nonlinear_cfg,
            "law_path": str(nonlinear_law.path),
            "law_kind": nonlinear_law.__class__.__name__,
            "displacement_limit_m": nonlinear_law.displacement_limit_m,
            "current_limit_A": nonlinear_law.current_limit_A,
            "coenergy_contract": coenergy_contract,
        }
    if suspension_law is not None:
        metadata["formulation"] = metadata["formulation"] + "_with_suspension_Kms_ROM"
    return TransientModel(
        config=config,
        config_path=config_path,
        base_mainline=base,
        M=M,
        C=C,
        K=K,
        force_vector=force,
        back_emf_vector=force.copy(),
        R_ohm=float(config["drive"]["Rdc_ohm"]),
        L_H=float(config["drive"]["Lcoil_H"]),
        n_solid=ns,
        n_pressure=npres,
        solid_free_dofs=sf,
        pressure_free_dofs=pf,
        solid=solid,
        acoustic=acoustic,
        mesh=output_mesh,
        G=G,
        probes=probes,
        pml_sigma_by_pressure_dof=sigma_all[pf],
        nonlinear_law=nonlinear_law,
        suspension_law=suspension_law,
        metadata=metadata,
    )
