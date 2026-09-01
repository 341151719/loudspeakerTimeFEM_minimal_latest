from __future__ import annotations

import math
import time

import numpy as np
from scipy.sparse.linalg import splu

from .model import TransientModel
from .solver import TransientResult, _fit_harmonic, drive_voltage
from .tensor_coenergy import TensorCoenergyLaw


def _suspension_rom_terms(model: TransientModel, q_m: float) -> tuple[float, float, float, float, float]:
    """Return (correction force, correction tangent, potential correction, Kms secant, Kms tangent).

    The assembled FEM already supplies the small-signal linear stiffness.  The
    ROM therefore contributes only the nonlinear deviation from that tangent.
    """
    law = model.suspension_law
    if law is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    return (
        float(law.correction_force(q_m)),
        float(law.correction_tangent(q_m)),
        float(law.correction_potential(q_m)),
        float(law.secant_stiffness(q_m)),
        float(law.tangent_stiffness(q_m)),
    )


def tensor_newton_residual_jacobian(
    law: TensorCoenergyLaw,
    effective: np.ndarray,
    hfull: np.ndarray,
    state: np.ndarray,
    current_A: float,
    rhs_mech: np.ndarray,
    q_previous: float,
    current_previous_A: float,
    dt: float,
    resistance_ohm: float,
    voltage_average_V: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the complete tensor-coenergy Newton residual and Jacobian.

    This small reference assembly is also used by the derivative tests.  The
    production loop exploits its rank-one mechanical block algebraically, but
    the returned dense expression makes the contract unambiguous.
    """
    x = np.asarray(state[:-1], dtype=float)
    i = float(state[-1])
    hfull = np.asarray(hfull, dtype=float)
    q = float(hfull @ x)
    law.check_coordinates(q, i)
    F = float(law.force(q, i))
    F_x = float(law.dforce_dx(q, i))
    F_i = float(law.dforce_di(q, i))
    psi = float(law.flux(q, i))
    psi_previous = float(law.flux(q_previous, current_previous_A))
    L = float(law.incremental_inductance(q, i))
    if L <= 0.0:
        raise RuntimeError("W_ii must remain positive in Newton assembly")
    residual_mech = effective @ x - rhs_mech - hfull * F
    residual_electric = (
        0.5 * resistance_ohm * (i + current_previous_A)
        + (psi - psi_previous) / dt
        - voltage_average_V
    )
    effective_dense = effective.toarray() if hasattr(effective, "toarray") else np.asarray(effective, dtype=float)
    jacobian = np.zeros((len(state), len(state)), dtype=float)
    jacobian[:-1, :-1] = effective_dense - F_x * np.outer(hfull, hfull)
    jacobian[:-1, -1] = -hfull * F_i
    jacobian[-1, :-1] = (F_i / dt) * hfull
    jacobian[-1, -1] = 0.5 * resistance_ohm + L / dt
    return np.r_[residual_mech, residual_electric], jacobian


def _solve_tensor_coenergy_transient(
    model: TransientModel, law: TensorCoenergyLaw
) -> TransientResult:
    """Solve the transient system with the exact W-derived Newton tangent.

    The legacy separable law below remains untouched for regression.  This
    branch uses the same ``W_xi`` for mechanical/electrical coupling and a
    rank-one Sherman--Morrison update for the low-rank structural coordinate.
    """
    cfg = model.config
    ncfg = cfg["nonlinear"]
    f0 = float(cfg["drive"]["frequency_Hz"])
    cycles = float(cfg["time"]["cycles"])
    steps_per_period = int(cfg["time"]["steps_per_period"])
    dt = 1.0 / (f0 * steps_per_period)
    n_steps = int(round(cycles * steps_per_period))
    times = np.arange(n_steps + 1, dtype=float) * dt
    voltage = drive_voltage(model, times)
    beta = float(cfg["time"].get("newmark_beta", 0.25))
    gamma = float(cfg["time"].get("newmark_gamma", 0.5))
    if beta <= 0 or gamma <= 0:
        raise ValueError("Newmark beta/gamma 必须为正")

    ns = model.n_solid
    n = model.M.shape[0]
    a0 = 1.0 / (beta * dt * dt)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)
    effective = (model.K + a0 * model.M + a1 * model.C).tocsc()
    started_factor = time.perf_counter()
    lu = splu(effective)
    factor_seconds = time.perf_counter() - started_factor

    base_bl = float(model.metadata.get("BL_axial_N_per_A", law.effective_bl(0.0, 0.0)))
    if abs(base_bl) <= 1e-12:
        raise RuntimeError("结构力形函数 BL 基准为零")
    h = model.force_vector / base_bl
    hfull = np.zeros(n, dtype=float)
    hfull[:ns] = h
    inverse_h = lu.solve(hfull)
    h_inverse_h = float(hfull @ inverse_h)

    x = np.zeros(n, dtype=float)
    velocity = np.zeros(n, dtype=float)
    acceleration = np.zeros(n, dtype=float)
    current = np.zeros(n_steps + 1, dtype=float)
    pressure_probes = np.zeros((n_steps + 1, len(model.probes.names)), dtype=float)
    coil_x = np.zeros(n_steps + 1, dtype=float)
    coil_v = np.zeros(n_steps + 1, dtype=float)
    coil_a = np.zeros(n_steps + 1, dtype=float)
    magnetic_flux = np.zeros(n_steps + 1, dtype=float)
    coenergy = np.zeros(n_steps + 1, dtype=float)
    electromagnetic_force = np.zeros(n_steps + 1, dtype=float)
    tangent_bl = np.zeros(n_steps + 1, dtype=float)
    incremental_l = np.zeros(n_steps + 1, dtype=float)
    force_xx = np.zeros(n_steps + 1, dtype=float)
    force_xi = np.zeros(n_steps + 1, dtype=float)
    suspension_enabled = model.suspension_law is not None
    suspension_force = np.zeros(n_steps + 1, dtype=float) if suspension_enabled else None
    suspension_secant = np.zeros(n_steps + 1, dtype=float) if suspension_enabled else None
    suspension_tangent = np.zeros(n_steps + 1, dtype=float) if suspension_enabled else None
    suspension_correction = np.zeros(n_steps + 1, dtype=float) if suspension_enabled else None
    suspension_energy = np.zeros(n_steps + 1, dtype=float) if suspension_enabled else None
    sm_denominator = np.ones(n_steps + 1, dtype=float)
    energy = np.zeros((n_steps + 1, 5), dtype=float)
    balance = np.zeros((n_steps + 1, 8), dtype=float)
    iterations = np.zeros(n_steps + 1, dtype=int)
    newton_residual = np.zeros(n_steps + 1, dtype=float)
    gap_margin = np.ones(n_steps + 1, dtype=float)
    gap_half_width = float(ncfg.get("ale_gap_half_width_m", law.displacement_limit_m))
    q_limit = min(
        law.displacement_limit_m,
        model.suspension_law.displacement_limit_m if suspension_enabled else law.displacement_limit_m,
    )
    Ms = model.M[:ns, :ns]
    Ks = model.K[:ns, :ns]
    Cs = model.C[:ns, :ns]

    phases = np.asarray(cfg["time"].get("field_snapshot_phases_deg", []), float)
    snapshot_cycle = int(cfg["time"].get("snapshot_cycle", math.ceil(cycles)))
    snapshot_targets = ((snapshot_cycle - 1) + phases / 360.0) / f0
    snapshot_indices = {
        int(np.clip(round(value / dt), 0, n_steps)) for value in snapshot_targets
    }
    snapshot_times: list[float] = []
    pressure_snapshots: list[np.ndarray] = []
    solid_snapshots: list[np.ndarray] = []

    def observe(index: int) -> None:
        u = x[:ns]
        p = x[ns:]
        vu = velocity[:ns]
        q = float(h @ u)
        i = float(current[index])
        law.check_coordinates(q, i)
        psi = float(law.flux(q, i))
        W = float(law.coenergy(q, i))
        F = float(law.force(q, i))
        L = float(law.incremental_inductance(q, i))
        if L <= 0.0:
            raise RuntimeError(f"W_ii 非正: x={q:.6g}, i={i:.6g}, L={L:.6g}")
        pressure_probes[index] = model.probes.pressure_matrix @ p
        coil_x[index] = q
        coil_v[index] = float(h @ vu)
        coil_a[index] = float(h @ acceleration[:ns])
        magnetic_flux[index] = psi
        coenergy[index] = W
        electromagnetic_force[index] = F
        tangent_bl[index] = float(law.effective_bl(q, i))
        incremental_l[index] = L
        force_xx[index] = float(law.dforce_dx(q, i))
        force_xi[index] = float(law.dforce_di(q, i))
        corr_force, corr_tangent, corr_potential, kms_secant, kms_tangent = _suspension_rom_terms(model, q)
        if suspension_enabled:
            suspension_force[index] = float(model.suspension_law.restoring_force(q))
            suspension_secant[index] = kms_secant
            suspension_tangent[index] = kms_tangent
            suspension_correction[index] = corr_force
            suspension_energy[index] = corr_potential
        rank_coefficient = force_xx[index] - corr_tangent
        sm_denominator[index] = 1.0 - rank_coefficient * h_inverse_h
        gap_margin[index] = 1.0 - abs(q) / gap_half_width
        kinetic = 0.5 * float(vu @ (Ms @ vu))
        potential = 0.5 * float(u @ (Ks @ u)) + corr_potential
        magnetic = float(law.magnetic_energy(q, i))
        copper = model.R_ohm * i * i
        damping = float(vu @ (Cs @ vu))
        energy[index] = [kinetic, potential, magnetic, copper, damping]
        balance[index, :7] = [
            float(voltage[index] * i),
            copper,
            damping,
            F * coil_v[index],
            kinetic + potential,
            magnetic,
            W,
        ]
        if index in snapshot_indices:
            p_full = np.zeros(len(model.acoustic.acoustic_nodes_global))
            p_full[model.pressure_free_dofs] = p
            u_full = np.zeros(model.solid.ndof)
            u_full[model.solid_free_dofs] = u
            snapshot_times.append(times[index])
            pressure_snapshots.append(p_full)
            solid_snapshots.append(u_full.reshape(-1, 2))

    observe(0)
    newton_tol = float(ncfg.get("newton_tolerance", 1e-8))
    max_newton = int(ncfg.get("newton_max_iterations", 12))
    q_previous = 0.0
    loop_started = time.perf_counter()
    for step in range(n_steps):
        rhs_mech = (
            model.M @ (a0 * x + a2 * velocity + a3 * acceleration)
            + model.C @ (a1 * x + a4 * velocity + a5 * acceleration)
        )
        x_guess = x + dt * velocity + 0.5 * dt * dt * acceleration
        i_guess = float(current[step])
        flux_previous = float(law.flux(q_previous, current[step]))
        converged = False
        final_relative = math.inf
        for iteration in range(1, max_newton + 1):
            q = float(h @ x_guess[:ns])
            law.check_coordinates(q, i_guess)
            F = float(law.force(q, i_guess))
            F_x = float(law.dforce_dx(q, i_guess))
            F_i = float(law.dforce_di(q, i_guess))
            psi = float(law.flux(q, i_guess))
            Ldiff = float(law.incremental_inductance(q, i_guess))
            if Ldiff <= 0.0:
                raise RuntimeError(
                    f"W_ii 非正，拒绝 Newton: x={q:.9g}, i={i_guess:.9g}, L={Ldiff:.9g}"
                )
            corr_force, corr_tangent, _, _, _ = _suspension_rom_terms(model, q)
            residual_mech = effective @ x_guess - rhs_mech - hfull * F + hfull * corr_force
            residual_electric = (
                0.5 * model.R_ohm * (i_guess + current[step])
                + (psi - flux_previous) / dt
                - 0.5 * (voltage[step + 1] + voltage[step])
            )
            final_relative = max(
                float(np.linalg.norm(residual_mech)) / max(float(np.linalg.norm(rhs_mech)), 1.0),
                abs(float(residual_electric)) / max(abs(float(voltage[step + 1])), 1.0),
            )
            if final_relative < newton_tol:
                converged = True
                break
            # Mechanical Newton block is A - (F_x - dDeltaF_s/dq) h h^T.
            # Electromagnetic and suspension nonlinearities therefore remain one rank-1 update.
            rank_coefficient = F_x - corr_tangent
            denominator = 1.0 - rank_coefficient * h_inverse_h
            if abs(denominator) < 1e-10:
                raise RuntimeError("combined magnetic/Kms Sherman--Morrison denominator near zero")

            def solve_tangent(rhs: np.ndarray) -> np.ndarray:
                base = lu.solve(rhs)
                return base + rank_coefficient * inverse_h * float(hfull @ base) / denominator

            y = solve_tangent(-residual_mech)
            z = F_i * inverse_h / denominator
            electric_row = (F_i / dt) * hfull
            electric_diagonal = 0.5 * model.R_ohm + Ldiff / dt
            schur = electric_diagonal + float(electric_row @ z)
            if abs(schur) < 1e-12:
                raise RuntimeError("tensor coenergy Newton Schur denominator near zero")
            delta_i = (-float(residual_electric) - float(electric_row @ y)) / schur
            delta_x = y + z * delta_i
            scale = 1.0
            valid_trial = False
            for _ in range(24):
                q_trial = float(h @ (x_guess[:ns] + scale * delta_x[:ns]))
                i_trial = i_guess + scale * delta_i
                if abs(q_trial) <= q_limit and abs(i_trial) <= law.current_limit_A:
                    valid_trial = True
                    break
                scale *= 0.5
            if not valid_trial:
                raise RuntimeError("Newton step cannot keep magnetic/suspension coordinates inside declared range")
            x_guess += scale * delta_x
            i_guess += scale * delta_i
        if not converged:
            raise RuntimeError(
                f"tensor coenergy Newton 在 step={step + 1}, t={times[step + 1]:.6g}s "
                f"未于 {max_newton} 次内收敛，relative={final_relative:.3e}"
            )
        x_new = x_guess
        acceleration_new = a0 * (x_new - x) - a2 * velocity - a3 * acceleration
        velocity_new = velocity + dt * (
            (1.0 - gamma) * acceleration + gamma * acceleration_new
        )
        x, velocity, acceleration = x_new, velocity_new, acceleration_new
        current[step + 1] = i_guess
        iterations[step + 1] = iteration
        newton_residual[step + 1] = final_relative
        q_previous = float(h @ x[:ns])
        observe(step + 1)
    loop_seconds = time.perf_counter() - loop_started

    exterior = cfg.get("exterior", {})
    radius = float(exterior.get("axis_distance_m", 1.0))
    piston_radius = float(exterior.get("effective_piston_radius_m", 0.07))
    piston_area = math.pi * piston_radius**2
    rho0 = float(cfg["air"]["rho0_kg_m3"])
    c0 = float(cfg["air"]["c0_m_s"])
    delay = radius / c0
    farfield_axis = (
        rho0
        * piston_area
        * np.interp(times - delay, times, coil_a, left=0.0)
        / (2.0 * math.pi * radius)
    )
    total_storage = energy[:, 0] + energy[:, 1] + energy[:, 2]
    d_storage = np.gradient(total_storage, dt)
    balance[:, 7] = balance[:, 0] - balance[:, 1] - balance[:, 2] - d_storage
    secant_bl = np.divide(
        electromagnetic_force,
        current,
        out=tangent_bl.copy(),
        where=np.abs(current) > 1e-8,
    )
    last_cycles = int(cfg["export"].get("last_cycles_for_spectrum", 1))
    start = max(0, n_steps - last_cycles * steps_per_period)
    harmonic = {
        "frequency_Hz": f0,
        "tensor_coenergy": True,
        "fundamental_probe_complex_Pa": _fit_harmonic(times[start:], pressure_probes[start:], f0),
        "fundamental_current_complex_A": complex(_fit_harmonic(times[start:], current[start:], f0)),
        "fundamental_axis_farfield_complex_Pa": complex(_fit_harmonic(times[start:], farfield_axis[start:], f0)),
        "BL_tangent_min_max_N_A": [float(np.min(tangent_bl)), float(np.max(tangent_bl))],
        "BL_secant_min_max_N_A": [float(np.min(secant_bl)), float(np.max(secant_bl))],
        "incremental_L_min_max_H": [float(np.min(incremental_l)), float(np.max(incremental_l))],
        "current_min_max_A": [float(np.min(current)), float(np.max(current))],
        "coil_displacement_min_max_m": [float(np.min(coil_x)), float(np.max(coil_x))],
        "newton_iterations_max": int(np.max(iterations)),
        "newton_iterations_mean": float(np.mean(iterations[1:])),
        "newton_residual_max": float(np.max(newton_residual)),
        "ale_gap_margin_min": float(np.min(gap_margin)),
        "energy_balance_residual_max_W": float(np.max(np.abs(balance[:, 7]))),
        "suspension_Kms_ROM_enabled": suspension_enabled,
        "sherman_morrison_denominator_min_abs": float(np.min(np.abs(sm_denominator))),
        **(
            {
                "Kms_secant_min_max_N_m": [float(np.min(suspension_secant)), float(np.max(suspension_secant))],
                "Kms_tangent_min_max_N_m": [float(np.min(suspension_tangent)), float(np.max(suspension_tangent))],
                "suspension_restoring_force_min_max_N": [float(np.min(suspension_force)), float(np.max(suspension_force))],
            }
            if suspension_enabled else {}
        ),
    }
    return TransientResult(
        time_s=times,
        voltage_V=voltage,
        current_A=current,
        coil_displacement_m=coil_x,
        coil_velocity_m_s=coil_v,
        farfield_axis_Pa=farfield_axis,
        pressure_probes_Pa=pressure_probes,
        energy_J=energy,
        dynamic_BL_N_A=secant_bl,
        incremental_inductance_H=incremental_l,
        nonlinear_iterations=iterations,
        ale_normalized_gap_margin=gap_margin,
        snapshot_times_s=np.asarray(snapshot_times),
        pressure_snapshots_Pa=np.asarray(pressure_snapshots),
        solid_displacement_snapshots_m=np.asarray(solid_snapshots),
        harmonic=harmonic,
        runtime={
            "factorization_seconds": factor_seconds,
            "time_loop_seconds": loop_seconds,
            "total_solver_seconds": factor_seconds + loop_seconds,
            "steps": float(n_steps),
            "dt_s": dt,
        },
        magnetic_flux_Wb=magnetic_flux,
        coenergy_J=coenergy,
        electromagnetic_force_N=electromagnetic_force,
        tangent_BL_N_A=tangent_bl,
        dforce_dx_N_m=force_xx,
        dforce_di_N_A=force_xi,
        newton_residual=newton_residual,
        energy_balance_W=balance,
        suspension_restoring_force_N=suspension_force,
        suspension_secant_stiffness_N_m=suspension_secant,
        suspension_tangent_stiffness_N_m=suspension_tangent,
        suspension_correction_force_N=suspension_correction,
        suspension_correction_energy_J=suspension_energy,
        sherman_morrison_denominator=sm_denominator,
    )


def solve_nonlinear_transient(model: TransientModel) -> TransientResult:
    law = model.nonlinear_law
    if law is None:
        raise ValueError("配置未启用 nonlinear magnetic law")
    if isinstance(law, TensorCoenergyLaw):
        return _solve_tensor_coenergy_transient(model, law)
    cfg = model.config
    ncfg = cfg["nonlinear"]
    f0 = float(cfg["drive"]["frequency_Hz"])
    cycles = float(cfg["time"]["cycles"])
    steps_per_period = int(cfg["time"]["steps_per_period"])
    dt = 1.0 / (f0 * steps_per_period)
    n_steps = int(round(cycles * steps_per_period))
    times = np.arange(n_steps + 1, dtype=float) * dt
    voltage = drive_voltage(model, times)
    beta = float(cfg["time"].get("newmark_beta", 0.25))
    gamma = float(cfg["time"].get("newmark_gamma", 0.5))
    ns = model.n_solid
    n = model.M.shape[0]
    a0 = 1.0 / (beta * dt * dt)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)
    effective = (model.K + a0 * model.M + a1 * model.C).tocsc()
    t_factor = time.perf_counter()
    lu = splu(effective)
    factor_seconds = time.perf_counter() - t_factor

    bl0 = law.bl(0.0)
    dynamic_bl_enabled = bool(ncfg.get("dynamic_BL_enabled", True))
    nonlinear_inductance_enabled = bool(
        ncfg.get("nonlinear_inductance_enabled", True)
    )
    coupled_coenergy_enabled = bool(ncfg.get("coupled_coenergy_enabled", False))
    linear_inductance = law.incremental_inductance(0.0)

    def bl_value(q: float, i: float = 0.0) -> float:
        base = law.bl(q) if dynamic_bl_enabled else bl0
        return base + (
            law.bl_current_correction(i) if coupled_coenergy_enabled else 0.0
        )

    def bl_derivative(q: float) -> float:
        return law.dbl_dx(q) if dynamic_bl_enabled else 0.0

    def motional_flux(q: float) -> float:
        return law.motional_flux(q) if dynamic_bl_enabled else bl0 * q

    def current_flux(i: float) -> float:
        return law.current_flux(i) if nonlinear_inductance_enabled else linear_inductance * i

    def differential_inductance(i: float) -> float:
        return (
            law.incremental_inductance(i)
            if nonlinear_inductance_enabled
            else linear_inductance
        )

    def total_flux(q: float, i: float) -> float:
        if coupled_coenergy_enabled:
            return law.coupled_flux(q, i)
        return current_flux(i) + motional_flux(q)
    h = model.force_vector / bl0
    hfull = np.zeros(n)
    hfull[:ns] = h
    inverse_h = lu.solve(hfull)
    h_inverse_h = float(hfull @ inverse_h)

    x = np.zeros(n)
    velocity = np.zeros(n)
    acceleration = np.zeros(n)
    current = np.zeros(n_steps + 1)
    pressure_probes = np.zeros((n_steps + 1, len(model.probes.names)))
    coil_x = np.zeros(n_steps + 1)
    coil_v = np.zeros(n_steps + 1)
    coil_a = np.zeros(n_steps + 1)
    energy = np.zeros((n_steps + 1, 5))
    dynamic_bl = np.full(n_steps + 1, bl0)
    incremental_l = np.full(n_steps + 1, linear_inductance)
    suspension_enabled = model.suspension_law is not None
    suspension_force = np.zeros(n_steps + 1, dtype=float) if suspension_enabled else None
    suspension_secant = np.zeros(n_steps + 1, dtype=float) if suspension_enabled else None
    suspension_tangent = np.zeros(n_steps + 1, dtype=float) if suspension_enabled else None
    suspension_correction = np.zeros(n_steps + 1, dtype=float) if suspension_enabled else None
    suspension_energy = np.zeros(n_steps + 1, dtype=float) if suspension_enabled else None
    sm_denominator = np.ones(n_steps + 1, dtype=float)
    iterations = np.zeros(n_steps + 1, dtype=int)
    gap_margin = np.ones(n_steps + 1)
    gap_half_width = float(ncfg.get("ale_gap_half_width_m", law.displacement_limit_m))
    q_limit = min(
        law.displacement_limit_m,
        model.suspension_law.displacement_limit_m if suspension_enabled else law.displacement_limit_m,
    )
    Ms = model.M[:ns, :ns]
    Ks = model.K[:ns, :ns]
    Cs = model.C[:ns, :ns]

    phases = np.asarray(cfg["time"].get("field_snapshot_phases_deg", []), float)
    snapshot_cycle = int(cfg["time"].get("snapshot_cycle", math.ceil(cycles)))
    snapshot_targets = ((snapshot_cycle - 1) + phases / 360.0) / f0
    snapshot_indices = set(
        int(np.clip(round(value / dt), 0, n_steps)) for value in snapshot_targets
    )
    snapshot_times: list[float] = []
    pressure_snapshots: list[np.ndarray] = []
    solid_snapshots: list[np.ndarray] = []

    def observe(index: int):
        u = x[:ns]
        p = x[ns:]
        vu = velocity[:ns]
        q = float(h @ u)
        pressure_probes[index] = model.probes.pressure_matrix @ p
        coil_x[index] = q
        coil_v[index] = float(h @ vu)
        coil_a[index] = float(h @ acceleration[:ns])
        dynamic_bl[index] = bl_value(q, current[index])
        incremental_l[index] = differential_inductance(current[index])
        corr_force, corr_tangent, corr_potential, kms_secant, kms_tangent = _suspension_rom_terms(model, q)
        if suspension_enabled:
            suspension_force[index] = float(model.suspension_law.restoring_force(q))
            suspension_secant[index] = kms_secant
            suspension_tangent[index] = kms_tangent
            suspension_correction[index] = corr_force
            suspension_energy[index] = corr_potential
        rank_coefficient = current[index] * bl_derivative(q) - corr_tangent
        sm_denominator[index] = 1.0 - rank_coefficient * h_inverse_h
        gap_margin[index] = 1.0 - abs(q) / gap_half_width
        energy[index] = [
            0.5 * float(vu @ (Ms @ vu)),
            0.5 * float(u @ (Ks @ u)) + corr_potential,
            float(
                0.5 * linear_inductance * current[index] ** 2
                if not nonlinear_inductance_enabled
                else current[index] * law.current_flux(current[index])
                - law.flux_polynomial.integ()(current[index])
                + (
                    q
                    * current[index] ** 2
                    * law.dbl_current_di(current[index])
                    if coupled_coenergy_enabled
                    else 0.0
                )
            ),
            model.R_ohm * current[index] ** 2,
            float(vu @ (Cs @ vu)),
        ]
        if index in snapshot_indices:
            p_full = np.zeros(len(model.acoustic.acoustic_nodes_global))
            p_full[model.pressure_free_dofs] = p
            u_full = np.zeros(model.solid.ndof)
            u_full[model.solid_free_dofs] = u
            snapshot_times.append(times[index])
            pressure_snapshots.append(p_full)
            solid_snapshots.append(u_full.reshape(-1, 2))

    observe(0)
    newton_tol = float(ncfg.get("newton_tolerance", 1e-8))
    max_newton = int(ncfg.get("newton_max_iterations", 12))
    q_previous = 0.0
    t_loop = time.perf_counter()
    for step in range(n_steps):
        rhs_mech = (
            model.M @ (a0 * x + a2 * velocity + a3 * acceleration)
            + model.C @ (a1 * x + a4 * velocity + a5 * acceleration)
        )
        x_guess = x + dt * velocity + 0.5 * dt * dt * acceleration
        i_guess = float(current[step])
        flux_previous = total_flux(q_previous, float(current[step]))
        converged = False
        for iteration in range(1, max_newton + 1):
            q = float(h @ x_guess[:ns])
            law.check_coordinates(q, i_guess)
            bl = bl_value(q, i_guess)
            dbl = bl_derivative(q)
            Ldiff = differential_inductance(i_guess)
            flux = total_flux(q, i_guess)
            correction_prime = (
                law.dbl_current_di(i_guess) if coupled_coenergy_enabled else 0.0
            )
            correction_second = (
                law.d2bl_current_di2(i_guess) if coupled_coenergy_enabled else 0.0
            )
            force_di = bl + i_guess * correction_prime
            flux_dq = force_di
            flux_di = Ldiff + (
                q * (2.0 * correction_prime + i_guess * correction_second)
                if coupled_coenergy_enabled
                else 0.0
            )
            corr_force, corr_tangent, _, _, _ = _suspension_rom_terms(model, q)
            residual_mech = effective @ x_guess - rhs_mech - hfull * bl * i_guess + hfull * corr_force
            residual_electric = (
                0.5 * model.R_ohm * (i_guess + current[step])
                + (flux - flux_previous) / dt
                - 0.5 * (voltage[step + 1] + voltage[step])
            )
            relative = max(
                float(np.linalg.norm(residual_mech))
                / max(float(np.linalg.norm(rhs_mech)), 1.0),
                abs(float(residual_electric))
                / max(abs(float(voltage[step + 1])), 1.0),
            )
            if relative < newton_tol:
                converged = True
                break

            # A + dDeltaF_s/dq*h*h^T - i*dBL/dq*h*h^T
            # = A - rank_coefficient*h*h^T, retaining one Sherman-Morrison update.
            rank_coefficient = i_guess * dbl - corr_tangent
            denominator = 1.0 - rank_coefficient * h_inverse_h
            if abs(denominator) < 1e-10:
                raise RuntimeError("非线性结构切线的 Sherman-Morrison 分母接近零")

            def solve_tangent(rhs: np.ndarray) -> np.ndarray:
                base = lu.solve(rhs)
                return base + (
                    rank_coefficient
                    * inverse_h
                    * float(hfull @ base)
                    / denominator
                )

            y = solve_tangent(-residual_mech)
            # Mechanical derivative wrt current is -h*BL, so solve(-c)=solve(h*BL).
            z = force_di * inverse_h / denominator
            electric_row = (flux_dq / dt) * hfull
            electric_diagonal = 0.5 * model.R_ohm + flux_di / dt
            schur = electric_diagonal + float(electric_row @ z)
            delta_i = (
                -float(residual_electric) - float(electric_row @ y)
            ) / schur
            delta_x = y + z * delta_i
            # Guard field-derived coordinate limits without hiding an excursion.
            scale = 1.0
            for _ in range(12):
                q_trial = float(h @ (x_guess[:ns] + scale * delta_x[:ns]))
                i_trial = i_guess + scale * delta_i
                if (
                    abs(q_trial) <= 0.999 * q_limit
                    and abs(i_trial) <= 0.999 * law.current_limit_A
                ):
                    break
                scale *= 0.5
            x_guess += scale * delta_x
            i_guess += scale * delta_i
        if not converged:
            raise RuntimeError(
                f"非线性 Newton 在 step={step + 1}, t={times[step + 1]:.6g}s "
                f"未于 {max_newton} 次内收敛"
            )

        x_new = x_guess
        acceleration_new = a0 * (x_new - x) - a2 * velocity - a3 * acceleration
        velocity_new = velocity + dt * (
            (1.0 - gamma) * acceleration + gamma * acceleration_new
        )
        x, velocity, acceleration = x_new, velocity_new, acceleration_new
        current[step + 1] = i_guess
        iterations[step + 1] = iteration
        q_previous = float(h @ x[:ns])
        observe(step + 1)
    loop_seconds = time.perf_counter() - t_loop

    exterior = cfg.get("exterior", {})
    radius = float(exterior.get("axis_distance_m", 1.0))
    piston_radius = float(exterior.get("effective_piston_radius_m", 0.07))
    piston_area = math.pi * piston_radius**2
    rho0 = float(cfg["air"]["rho0_kg_m3"])
    c0 = float(cfg["air"]["c0_m_s"])
    delay = radius / c0
    farfield_axis = (
        rho0
        * piston_area
        * np.interp(times - delay, times, coil_a, left=0.0)
        / (2.0 * math.pi * radius)
    )
    last_cycles = int(cfg["export"].get("last_cycles_for_spectrum", 1))
    start = max(0, n_steps - last_cycles * steps_per_period)
    harmonic = {
        "frequency_Hz": f0,
        "nonlinear": True,
        "dynamic_BL_enabled": dynamic_bl_enabled,
        "nonlinear_inductance_enabled": nonlinear_inductance_enabled,
        "coupled_coenergy_enabled": coupled_coenergy_enabled,
        "fundamental_probe_complex_Pa": _fit_harmonic(
            times[start:], pressure_probes[start:], f0
        ),
        "fundamental_current_complex_A": complex(
            _fit_harmonic(times[start:], current[start:], f0)
        ),
        "fundamental_axis_farfield_complex_Pa": complex(
            _fit_harmonic(times[start:], farfield_axis[start:], f0)
        ),
        "BL_min_max_N_A": [float(dynamic_bl.min()), float(dynamic_bl.max())],
        "incremental_L_min_max_H": [
            float(incremental_l.min()),
            float(incremental_l.max()),
        ],
        "current_min_max_A": [float(current.min()), float(current.max())],
        "coil_displacement_min_max_m": [float(coil_x.min()), float(coil_x.max())],
        "newton_iterations_max": int(iterations.max()),
        "newton_iterations_mean": float(iterations[1:].mean()),
        "ale_gap_margin_min": float(gap_margin.min()),
        "suspension_Kms_ROM_enabled": suspension_enabled,
        "sherman_morrison_denominator_min_abs": float(np.min(np.abs(sm_denominator))),
        **(
            {
                "Kms_secant_min_max_N_m": [float(np.min(suspension_secant)), float(np.max(suspension_secant))],
                "Kms_tangent_min_max_N_m": [float(np.min(suspension_tangent)), float(np.max(suspension_tangent))],
                "suspension_restoring_force_min_max_N": [float(np.min(suspension_force)), float(np.max(suspension_force))],
            }
            if suspension_enabled else {}
        ),
    }
    return TransientResult(
        time_s=times,
        voltage_V=voltage,
        current_A=current,
        coil_displacement_m=coil_x,
        coil_velocity_m_s=coil_v,
        farfield_axis_Pa=farfield_axis,
        pressure_probes_Pa=pressure_probes,
        energy_J=energy,
        dynamic_BL_N_A=dynamic_bl,
        incremental_inductance_H=incremental_l,
        nonlinear_iterations=iterations,
        ale_normalized_gap_margin=gap_margin,
        snapshot_times_s=np.asarray(snapshot_times),
        pressure_snapshots_Pa=np.asarray(pressure_snapshots),
        solid_displacement_snapshots_m=np.asarray(solid_snapshots),
        harmonic=harmonic,
        runtime={
            "factorization_seconds": factor_seconds,
            "time_loop_seconds": loop_seconds,
            "total_solver_seconds": factor_seconds + loop_seconds,
            "steps": float(n_steps),
            "dt_s": dt,
        },
        suspension_restoring_force_N=suspension_force,
        suspension_secant_stiffness_N_m=suspension_secant,
        suspension_tangent_stiffness_N_m=suspension_tangent,
        suspension_correction_force_N=suspension_correction,
        suspension_correction_energy_J=suspension_energy,
        sherman_morrison_denominator=sm_denominator,
    )
