from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np
from scipy.sparse import bmat, csc_matrix, csr_matrix
from scipy.sparse.linalg import splu

from .model import TransientModel


@dataclass
class TransientResult:
    time_s: np.ndarray
    voltage_V: np.ndarray
    current_A: np.ndarray
    coil_displacement_m: np.ndarray
    coil_velocity_m_s: np.ndarray
    farfield_axis_Pa: np.ndarray
    pressure_probes_Pa: np.ndarray
    energy_J: np.ndarray
    dynamic_BL_N_A: np.ndarray
    incremental_inductance_H: np.ndarray
    nonlinear_iterations: np.ndarray
    ale_normalized_gap_margin: np.ndarray
    snapshot_times_s: np.ndarray
    pressure_snapshots_Pa: np.ndarray
    solid_displacement_snapshots_m: np.ndarray
    harmonic: dict[str, Any]
    runtime: dict[str, float]
    magnetic_flux_Wb: np.ndarray | None = None
    coenergy_J: np.ndarray | None = None
    electromagnetic_force_N: np.ndarray | None = None
    tangent_BL_N_A: np.ndarray | None = None
    dforce_dx_N_m: np.ndarray | None = None
    dforce_di_N_A: np.ndarray | None = None
    newton_residual: np.ndarray | None = None
    energy_balance_W: np.ndarray | None = None


def soft_start(t_s: np.ndarray | float, duration_s: float) -> np.ndarray:
    t = np.asarray(t_s, dtype=float)
    if duration_s <= 0:
        return np.ones_like(t)
    x = np.clip(t / duration_s, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def drive_voltage(model: TransientModel, t_s: np.ndarray | float) -> np.ndarray:
    cfg = model.config["drive"]
    t = np.asarray(t_s, dtype=float)
    f0 = float(cfg["frequency_Hz"])
    ramp = soft_start(t, float(cfg.get("soft_start_cycles", 1.0)) / f0)
    return float(cfg["voltage_peak_V"]) * np.sin(2.0 * math.pi * f0 * t) * ramp


def _harmonic_solution(model: TransientModel, frequency_Hz: float, voltage_peak_V: float):
    omega = 2.0 * math.pi * float(frequency_Hz)
    dynamic = (model.K - omega * omega * model.M + 1j * omega * model.C).tocsc()
    n = dynamic.shape[0]
    electrical_row = np.zeros(n, complex)
    electrical_row[: model.n_solid] = 1j * omega * model.back_emf_vector
    force_column = np.zeros(n, complex)
    force_column[: model.n_solid] = -model.force_vector
    system = bmat(
        [
            [dynamic, csc_matrix(force_column[:, None])],
            [csc_matrix(electrical_row[None, :]), csc_matrix([[model.R_ohm + 1j * omega * model.L_H]])],
        ],
        format="csc",
    )
    rhs = np.r_[np.zeros(n, complex), complex(voltage_peak_V)]
    solution = splu(system).solve(rhs)
    return solution[:-1], complex(solution[-1])


def _fit_harmonic(time_s: np.ndarray, values: np.ndarray, frequency_Hz: float) -> np.ndarray:
    omega = 2.0 * math.pi * float(frequency_Hz)
    design = np.column_stack(
        [np.sin(omega * time_s), np.cos(omega * time_s), np.ones_like(time_s)]
    )
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    # Real signal = Re(P*exp(iwt)); sin coefficient a and cos coefficient b -> P=b-i*a.
    return coefficients[1] - 1j * coefficients[0]


def solve_transient(model: TransientModel) -> TransientResult:
    cfg = model.config
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

    n = model.M.shape[0]
    ns = model.n_solid
    a0 = 1.0 / (beta * dt * dt)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)
    effective = (model.K + a0 * model.M + a1 * model.C).tocsc()

    force_col = np.zeros(n)
    force_col[:ns] = -model.force_vector
    electric_row = np.zeros(n)
    electric_row[:ns] = 0.5 * a1 * model.back_emf_vector
    electric_scalar = model.L_H / dt + 0.5 * model.R_ohm
    coupled = bmat(
        [
            [effective, csc_matrix(force_col[:, None])],
            [csc_matrix(electric_row[None, :]), csc_matrix([[electric_scalar]])],
        ],
        format="csc",
    )
    t_factor = time.perf_counter()
    lu = splu(coupled)
    factor_seconds = time.perf_counter() - t_factor

    x = np.zeros(n)
    velocity = np.zeros(n)
    acceleration = np.zeros(n)
    current = np.zeros(n_steps + 1)
    pressure_probes = np.zeros((n_steps + 1, len(model.probes.names)))
    coil_x = np.zeros(n_steps + 1)
    coil_v = np.zeros(n_steps + 1)
    coil_a = np.zeros(n_steps + 1)
    energy = np.zeros((n_steps + 1, 5))
    bl = max(abs(float(model.metadata["BL_axial_N_per_A"])), 1e-15)

    phases = np.asarray(cfg["time"].get("field_snapshot_phases_deg", []), float)
    snapshot_cycle = int(cfg["time"].get("snapshot_cycle", math.ceil(cycles)))
    snapshot_targets = ((snapshot_cycle - 1) + phases / 360.0) / f0
    snapshot_indices = set(
        int(np.clip(round(value / dt), 0, n_steps)) for value in snapshot_targets
    )
    snapshot_times: list[float] = []
    pressure_snapshots: list[np.ndarray] = []
    solid_snapshots: list[np.ndarray] = []
    Ms = model.M[:ns, :ns]
    Ks = model.K[:ns, :ns]
    Cs = model.C[:ns, :ns]

    def observe(index: int):
        u = x[:ns]
        p = x[ns:]
        vu = velocity[:ns]
        pressure_probes[index, :] = model.probes.pressure_matrix @ p
        coil_x[index] = float(model.back_emf_vector @ u / bl)
        coil_v[index] = float(model.back_emf_vector @ vu / bl)
        coil_a[index] = float(model.back_emf_vector @ acceleration[:ns] / bl)
        kinetic = 0.5 * float(vu @ (Ms @ vu))
        potential = 0.5 * float(u @ (Ks @ u))
        electric = 0.5 * model.L_H * current[index] ** 2
        resistive_power = model.R_ohm * current[index] ** 2
        structural_damping_power = float(vu @ (Cs @ vu))
        energy[index] = [
            kinetic,
            potential,
            electric,
            resistive_power,
            structural_damping_power,
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
    t_loop = time.perf_counter()
    for step in range(n_steps):
        rhs_mech = (
            model.M @ (a0 * x + a2 * velocity + a3 * acceleration)
            + model.C @ (a1 * x + a4 * velocity + a5 * acceleration)
        )
        velocity_constant = -a1 * x[:ns] - a4 * velocity[:ns] - a5 * acceleration[:ns]
        rhs_electric = (
            (model.L_H / dt - 0.5 * model.R_ohm) * current[step]
            + 0.5 * (voltage[step] + voltage[step + 1])
            - 0.5 * float(model.back_emf_vector @ velocity[:ns])
            - 0.5 * float(model.back_emf_vector @ velocity_constant)
        )
        solved = lu.solve(np.r_[rhs_mech, rhs_electric])
        x_new = solved[:-1]
        acceleration_new = a0 * (x_new - x) - a2 * velocity - a3 * acceleration
        velocity_new = velocity + dt * (
            (1.0 - gamma) * acceleration + gamma * acceleration_new
        )
        x, velocity, acceleration = x_new, velocity_new, acceleration_new
        current[step + 1] = solved[-1]
        observe(step + 1)
    loop_seconds = time.perf_counter() - t_loop

    exterior = cfg.get("exterior", {})
    radius = float(exterior.get("axis_distance_m", 1.0))
    piston_radius = float(exterior.get("effective_piston_radius_m", 0.07))
    piston_area = math.pi * piston_radius**2
    rho0 = float(cfg["air"]["rho0_kg_m3"])
    c0 = float(cfg["air"]["c0_m_s"])
    delay = radius / c0
    delayed_acceleration = np.interp(
        times - delay, times, coil_a, left=0.0, right=float(coil_a[-1])
    )
    farfield_axis = rho0 * piston_area * delayed_acceleration / (2.0 * math.pi * radius)

    last_cycles = int(cfg["export"].get("last_cycles_for_spectrum", 1))
    start = max(0, n_steps - last_cycles * steps_per_period)
    harmonic_x, harmonic_i = _harmonic_solution(
        model, f0, float(cfg["drive"]["voltage_peak_V"])
    )
    transient_probe = _fit_harmonic(times[start:], pressure_probes[start:], f0)
    harmonic_probe = model.probes.pressure_matrix @ harmonic_x[ns:]
    # The direct solve uses a real (cosine-reference) voltage phasor while the
    # transient source is sin(omega*t), hence the -i reference rotation.
    harmonic_probe_sine_reference = -1j * harmonic_probe
    harmonic_current_sine_reference = -1j * harmonic_i
    harmonic_coil_displacement = (
        model.back_emf_vector @ harmonic_x[:ns] / bl
    )
    harmonic_farfield_cos_reference = (
        rho0
        * piston_area
        * (-((2.0 * math.pi * f0) ** 2) * harmonic_coil_displacement)
        / (2.0 * math.pi * radius)
        * np.exp(-1j * 2.0 * math.pi * f0 * delay)
    )
    transient_farfield = complex(_fit_harmonic(times[start:], farfield_axis[start:], f0))
    harmonic_farfield_sine_reference = -1j * harmonic_farfield_cos_reference
    probe_error = np.abs(transient_probe - harmonic_probe_sine_reference) / np.maximum(
        np.abs(harmonic_probe), 1e-30
    )
    harmonic = {
        "frequency_Hz": f0,
        "transient_probe_complex_Pa": transient_probe,
        "direct_harmonic_probe_complex_Pa": harmonic_probe_sine_reference,
        "probe_relative_error": probe_error,
        "transient_current_complex_A": complex(_fit_harmonic(times[start:], current[start:], f0)),
        "direct_harmonic_current_complex_A": harmonic_current_sine_reference,
        "transient_axis_farfield_complex_Pa": transient_farfield,
        "direct_harmonic_axis_farfield_complex_Pa": harmonic_farfield_sine_reference,
        "axis_farfield_relative_error": float(
            abs(transient_farfield - harmonic_farfield_sine_reference)
            / max(abs(harmonic_farfield_sine_reference), 1e-30)
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
        dynamic_BL_N_A=np.full_like(times, bl),
        incremental_inductance_H=np.full_like(times, model.L_H),
        nonlinear_iterations=np.zeros_like(times, dtype=int),
        ale_normalized_gap_margin=np.ones_like(times),
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
    )
