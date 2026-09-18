"""Reduced-order non-axisymmetric loudspeaker radiation analysis.

This module is the first layer of the 3-D extension.  It represents structural
motion with circumferential Fourier harmonics and evaluates the radiated field
with a Rayleigh surface integral.  A non-axisymmetric basket can modulate the
rear source, and a rectangular enclosure can be included with a documented
Kirchhoff physical-optics approximation.

It is not a replacement for a volume-meshed nonlinear 3-D FEM.  In particular,
mode amplitudes are inputs until the full 3-D structural eigen/transient solver
is connected.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AnnularSource:
    xyz_m: np.ndarray
    area_weights_m2: np.ndarray
    radius_m: np.ndarray
    azimuth_rad: np.ndarray
    radial_centres_m: np.ndarray
    radial_ring_areas_m2: np.ndarray
    radial_points: int
    azimuthal_points: int


@dataclass(frozen=True)
class SurfacePanels:
    centres_m: np.ndarray
    normals: np.ndarray
    areas_m2: np.ndarray


def complex_amplitude(magnitude: float, phase_deg: float = 0.0) -> complex:
    return float(magnitude) * np.exp(1j * np.deg2rad(float(phase_deg)))


def annular_source(
    outer_radius_m: float,
    inner_radius_m: float = 0.0,
    radial_points: int = 20,
    azimuthal_points: int = 96,
) -> AnnularSource:
    """Build midpoint quadrature for a flat annular diaphragm."""
    outer = float(outer_radius_m)
    inner = float(inner_radius_m)
    nr = int(radial_points)
    nphi = int(azimuthal_points)
    if not (0.0 <= inner < outer):
        raise ValueError("inner_radius_m must satisfy 0 <= inner < outer")
    if nr < 2 or nphi < 8:
        raise ValueError("source quadrature requires radial_points>=2 and azimuthal_points>=8")
    edges = np.linspace(inner, outer, nr + 1)
    radius = 0.5 * (edges[:-1] + edges[1:])
    ring_areas = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    phi = 2.0 * math.pi * np.arange(nphi) / nphi
    rr, pp = np.meshgrid(radius, phi, indexing="ij")
    xyz = np.column_stack(
        [
            (rr * np.cos(pp)).ravel(),
            (rr * np.sin(pp)).ravel(),
            np.zeros(rr.size),
        ]
    )
    weights = np.repeat(ring_areas / nphi, nphi)
    return AnnularSource(
        xyz_m=xyz,
        area_weights_m2=weights,
        radius_m=rr.ravel(),
        azimuth_rad=pp.ravel(),
        radial_centres_m=radius,
        radial_ring_areas_m2=ring_areas,
        radial_points=nr,
        azimuthal_points=nphi,
    )


def _mode_phasor(mode: dict[str, Any], component: str) -> complex:
    magnitude_key = f"{component}_amplitude_m_s_peak"
    phase_key = f"{component}_phase_deg"
    if magnitude_key in mode:
        return complex_amplitude(mode[magnitude_key], mode.get(phase_key, 0.0))
    if component == "cosine" and "amplitude_m_s_peak" in mode:
        return complex_amplitude(mode["amplitude_m_s_peak"], mode.get("phase_deg", 0.0))
    return 0.0j


def synthesize_modal_velocity(
    source: AnnularSource,
    modes: list[dict[str, Any]],
) -> np.ndarray:
    """Synthesize complex normal velocity from real Fourier mode pairs.

    Each mode supports cosine/sine amplitudes and phases.  ``radial_power``
    controls the centre regularity, while ``radial_nodes`` creates concentric
    phase reversals for a breakup family.
    """
    velocity = np.zeros(len(source.xyz_m), dtype=complex)
    outer = float(np.max(source.radius_m))
    x = np.clip(source.radius_m / max(outer, 1e-15), 0.0, 1.0)
    for item in modes:
        order = int(item["order"])
        if order < 0:
            raise ValueError("circumferential mode order must be non-negative")
        power = float(item.get("radial_power", order if order else 0.0))
        nodes = int(item.get("radial_nodes", 0))
        edge_taper = float(item.get("edge_taper_power", 0.0))
        radial = np.power(np.maximum(x, 1e-12), power)
        if nodes:
            radial *= np.cos(nodes * math.pi * x)
        if edge_taper:
            radial *= np.power(np.maximum(1.0 - x * x, 0.0), edge_taper)
        cosine = _mode_phasor(item, "cosine")
        sine = _mode_phasor(item, "sine")
        angular = cosine * np.cos(order * source.azimuth_rad)
        if order:
            angular += sine * np.sin(order * source.azimuth_rad)
        velocity += radial * angular
    return velocity


def fourier_coefficients(
    values: np.ndarray,
    max_order: int,
    azimuth_axis: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return coefficients for f=a0+sum(am cos(m phi)+bm sin(m phi))."""
    data = np.moveaxis(np.asarray(values), azimuth_axis, -1)
    nphi = data.shape[-1]
    phi = 2.0 * math.pi * np.arange(nphi) / nphi
    cosine = np.empty(data.shape[:-1] + (int(max_order) + 1,), dtype=complex)
    sine = np.zeros_like(cosine)
    cosine[..., 0] = np.mean(data, axis=-1)
    for order in range(1, int(max_order) + 1):
        cosine[..., order] = 2.0 * np.mean(data * np.cos(order * phi), axis=-1)
        sine[..., order] = 2.0 * np.mean(data * np.sin(order * phi), axis=-1)
    return cosine, sine


def reconstruct_fourier(
    azimuth_rad: np.ndarray,
    cosine: np.ndarray,
    sine: np.ndarray,
) -> np.ndarray:
    phi = np.asarray(azimuth_rad, dtype=float)
    a = np.asarray(cosine)
    b = np.asarray(sine)
    if a.shape != b.shape or a.shape[-1] < 1:
        raise ValueError("cosine and sine coefficient arrays must have matching order axes")
    out = np.expand_dims(a[..., 0], -1) * np.ones_like(phi, dtype=complex)
    for order in range(1, a.shape[-1]):
        out += np.expand_dims(a[..., order], -1) * np.cos(order * phi)
        out += np.expand_dims(b[..., order], -1) * np.sin(order * phi)
    return out


def modal_metrics(
    source: AnnularSource,
    velocity_m_s_peak: np.ndarray,
    frequency_Hz: float,
    max_order: int,
) -> dict[str, Any]:
    field = np.asarray(velocity_m_s_peak, dtype=complex).reshape(
        source.radial_points, source.azimuthal_points
    )
    cosine, sine = fourier_coefficients(field, max_order=max_order)
    energy = np.empty(max_order + 1, dtype=float)
    energy[0] = float(np.sum(source.radial_ring_areas_m2 * np.abs(cosine[:, 0]) ** 2))
    for order in range(1, max_order + 1):
        energy[order] = float(
            0.5
            * np.sum(
                source.radial_ring_areas_m2
                * (np.abs(cosine[:, order]) ** 2 + np.abs(sine[:, order]) ** 2)
            )
        )
    total = max(float(np.sum(energy)), 1e-300)
    radius = source.radial_centres_m
    weight = source.radial_ring_areas_m2
    denominator = float(np.sum(weight * radius * radius))
    slope_x = np.sum(weight * radius * cosine[:, 1]) / denominator if max_order >= 1 else 0.0j
    slope_y = np.sum(weight * radius * sine[:, 1]) / denominator if max_order >= 1 else 0.0j
    omega = 2.0 * math.pi * float(frequency_Hz)
    tilt_x = slope_x / (1j * omega)
    tilt_y = slope_y / (1j * omega)
    return {
        "modal_energy_fraction": (energy / total).tolist(),
        "dominant_order": int(np.argmax(energy)),
        "rocking_velocity_gradient_x_per_s": [float(slope_x.real), float(slope_x.imag)],
        "rocking_velocity_gradient_y_per_s": [float(slope_y.real), float(slope_y.imag)],
        "rocking_tilt_x_rad_peak": [float(tilt_x.real), float(tilt_x.imag)],
        "rocking_tilt_y_rad_peak": [float(tilt_y.real), float(tilt_y.imag)],
        "rocking_tilt_magnitude_rad_peak": float(np.sqrt(abs(tilt_x) ** 2 + abs(tilt_y) ** 2)),
        "higher_order_breakup_fraction": float(np.sum(energy[2:]) / total),
    }


def basket_transmission(
    azimuth_rad: np.ndarray,
    spokes: int,
    spoke_width_deg: float,
    rotation_deg: float = 0.0,
    open_transmission: float = 1.0,
    blocked_transmission: float = 0.05,
    edge_softness_deg: float = 1.5,
) -> np.ndarray:
    """Angular rear-radiation transmission of equally spaced basket spokes."""
    count = int(spokes)
    if count < 1:
        return np.full_like(np.asarray(azimuth_rad, float), float(open_transmission))
    phi = np.asarray(azimuth_rad, dtype=float) - np.deg2rad(float(rotation_deg))
    pitch = 2.0 * math.pi / count
    distance = np.abs((phi + 0.5 * pitch) % pitch - 0.5 * pitch)
    half_width = 0.5 * np.deg2rad(float(spoke_width_deg))
    softness = max(np.deg2rad(float(edge_softness_deg)), 1e-9)
    open_fraction = 0.5 * (1.0 + np.tanh((distance - half_width) / softness))
    return float(blocked_transmission) + (
        float(open_transmission) - float(blocked_transmission)
    ) * open_fraction


def angular_mode_coupling(
    transmission: np.ndarray,
    max_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Galerkin coupling matrix in the complex exp(i*m*phi) basis."""
    profile = np.asarray(transmission, dtype=complex)
    phi = 2.0 * math.pi * np.arange(len(profile)) / len(profile)
    orders = np.arange(-int(max_order), int(max_order) + 1)
    matrix = np.empty((len(orders), len(orders)), dtype=complex)
    for i, target in enumerate(orders):
        for j, source in enumerate(orders):
            matrix[i, j] = np.mean(profile * np.exp(1j * (source - target) * phi))
    return orders, matrix


def rayleigh_pressure(
    source_xyz_m: np.ndarray,
    area_weights_m2: np.ndarray,
    velocity_m_s_peak: np.ndarray,
    observers_xyz_m: np.ndarray,
    frequency_Hz: float,
    rho0_kg_m3: float = 1.2041,
    c0_m_s: float = 343.2,
    chunk_size: int = 256,
) -> np.ndarray:
    """Evaluate the baffled Rayleigh integral at arbitrary 3-D observers."""
    source = np.asarray(source_xyz_m, dtype=float)
    observers = np.atleast_2d(np.asarray(observers_xyz_m, dtype=float))
    strength = np.asarray(area_weights_m2, float) * np.asarray(velocity_m_s_peak, complex)
    k = 2.0 * math.pi * float(frequency_Hz) / float(c0_m_s)
    coefficient = 1j * float(rho0_kg_m3) * float(c0_m_s) * k / (2.0 * math.pi)
    out = np.empty(len(observers), dtype=complex)
    for start in range(0, len(observers), int(chunk_size)):
        stop = min(start + int(chunk_size), len(observers))
        delta = observers[start:stop, None, :] - source[None, :, :]
        distance = np.linalg.norm(delta, axis=2)
        if np.any(distance < 1e-9):
            raise ValueError("Rayleigh observer coincides with a source quadrature point")
        green = np.exp(-1j * k * distance) / distance
        out[start:stop] = coefficient * (green @ strength)
    return out


def rectangular_enclosure_panels(
    width_m: float,
    height_m: float,
    depth_m: float,
    centre_xy_m: tuple[float, float] = (0.0, 0.0),
    divisions: tuple[int, int, int] = (14, 16, 12),
) -> SurfacePanels:
    """Panelize the four sides and rear of a box whose front plane is z=0."""
    width, height, depth = map(float, (width_m, height_m, depth_m))
    nx, ny, nz = map(int, divisions)
    if min(width, height, depth) <= 0 or min(nx, ny, nz) < 2:
        raise ValueError("enclosure dimensions must be positive and divisions >= 2")
    cx, cy = map(float, centre_xy_m)
    xs = np.linspace(cx - width / 2, cx + width / 2, nx + 1)
    ys = np.linspace(cy - height / 2, cy + height / 2, ny + 1)
    zs = np.linspace(-depth, 0.0, nz + 1)
    centres: list[list[float]] = []
    normals: list[list[float]] = []
    areas: list[float] = []

    def grid_face(a_edges, b_edges, point, normal):
        for ia in range(len(a_edges) - 1):
            for ib in range(len(b_edges) - 1):
                a = 0.5 * (a_edges[ia] + a_edges[ia + 1])
                b = 0.5 * (b_edges[ib] + b_edges[ib + 1])
                centres.append(point(a, b))
                normals.append(normal)
                areas.append((a_edges[ia + 1] - a_edges[ia]) * (b_edges[ib + 1] - b_edges[ib]))

    grid_face(ys, zs, lambda y, z: [xs[0], y, z], [-1.0, 0.0, 0.0])
    grid_face(ys, zs, lambda y, z: [xs[-1], y, z], [1.0, 0.0, 0.0])
    grid_face(xs, zs, lambda x, z: [x, ys[0], z], [0.0, -1.0, 0.0])
    grid_face(xs, zs, lambda x, z: [x, ys[-1], z], [0.0, 1.0, 0.0])
    grid_face(xs, ys, lambda x, y: [x, y, -depth], [0.0, 0.0, -1.0])
    return SurfacePanels(np.asarray(centres), np.asarray(normals), np.asarray(areas))


def kirchhoff_rigid_scattering(
    panels: SurfacePanels,
    incident_pressure_Pa_peak: np.ndarray,
    observers_xyz_m: np.ndarray,
    frequency_Hz: float,
    c0_m_s: float = 343.2,
    chunk_size: int = 256,
) -> np.ndarray:
    """High-frequency physical-optics estimate for a rigid enclosure.

    The rigid-surface pressure is approximated as twice the incident pressure.
    Edge release in the finite panel integral produces a diffraction estimate,
    but this is not a full BEM solution and is reported as such in metadata.
    """
    observers = np.atleast_2d(np.asarray(observers_xyz_m, float))
    incident = np.asarray(incident_pressure_Pa_peak, complex)
    if incident.shape != panels.areas_m2.shape:
        raise ValueError("incident pressure must contain one value per enclosure panel")
    k = 2.0 * math.pi * float(frequency_Hz) / float(c0_m_s)
    strength = -2.0 * incident * panels.areas_m2
    out = np.empty(len(observers), dtype=complex)
    for start in range(0, len(observers), int(chunk_size)):
        stop = min(start + int(chunk_size), len(observers))
        delta = observers[start:stop, None, :] - panels.centres_m[None, :, :]
        distance = np.linalg.norm(delta, axis=2)
        rhat = delta / np.maximum(distance[:, :, None], 1e-15)
        normal_projection = np.einsum("opk,pk->op", rhat, panels.normals)
        derivative = (
            np.exp(-1j * k * distance)
            / (4.0 * math.pi)
            * (1j * k / distance + 1.0 / distance**2)
            * normal_projection
        )
        out[start:stop] = derivative @ strength
    return out


def observation_plane(extent_m: float, z_m: float, points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(-float(extent_m), float(extent_m), int(points))
    x, y = np.meshgrid(axis, axis, indexing="xy")
    xyz = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, float(z_m))])
    return x, y, xyz


def observation_sphere(radius_m: float, polar_points: int, azimuthal_points: int):
    polar = np.linspace(0.0, math.pi, int(polar_points))
    azimuth = np.linspace(-math.pi, math.pi, int(azimuthal_points), endpoint=False)
    tt, pp = np.meshgrid(polar, azimuth, indexing="ij")
    radius = float(radius_m)
    xyz = np.column_stack(
        [
            (radius * np.sin(tt) * np.cos(pp)).ravel(),
            (radius * np.sin(tt) * np.sin(pp)).ravel(),
            (radius * np.cos(tt)).ravel(),
        ]
    )
    return tt, pp, xyz


def analyze(config: dict[str, Any]) -> dict[str, Any]:
    """Run a configured circumferential source and 3-D acoustic analysis."""
    frequency = float(config["frequency_Hz"])
    air = config.get("air", {})
    rho0 = float(air.get("rho0_kg_m3", 1.2041))
    c0 = float(air.get("c0_m_s", 343.2))
    diaphragm = config["diaphragm"]
    source = annular_source(
        diaphragm["outer_radius_m"],
        diaphragm.get("inner_radius_m", 0.0),
        diaphragm.get("radial_points", 20),
        diaphragm.get("azimuthal_points", 96),
    )
    velocity = synthesize_modal_velocity(source, list(config["modes"]))
    max_order = int(config.get("max_mode_order", max(int(m["order"]) for m in config["modes"])))
    metrics = modal_metrics(source, velocity, frequency, max_order)

    basket = config.get("basket", {})
    transmission = basket_transmission(
        source.azimuth_rad,
        basket.get("spokes", 0),
        basket.get("spoke_width_deg", 0.0),
        basket.get("rotation_deg", 0.0),
        basket.get("open_transmission", 1.0),
        basket.get("blocked_transmission", 0.05),
        basket.get("edge_softness_deg", 1.5),
    )
    _, coupling = angular_mode_coupling(
        basket_transmission(
            2.0 * math.pi * np.arange(720) / 720,
            basket.get("spokes", 0),
            basket.get("spoke_width_deg", 0.0),
            basket.get("rotation_deg", 0.0),
            basket.get("open_transmission", 1.0),
            basket.get("blocked_transmission", 0.05),
            basket.get("edge_softness_deg", 1.5),
        ),
        max_order,
    )

    observation = config.get("observation", {})
    plane_extent = float(observation.get("near_extent_m", 0.16))
    plane_points = int(observation.get("near_points", 61))
    near_offset = float(observation.get("near_offset_m", 0.04))
    x, y, front_xyz = observation_plane(plane_extent, near_offset, plane_points)
    _, _, rear_xyz = observation_plane(plane_extent, -near_offset, plane_points)
    front_pressure = rayleigh_pressure(
        source.xyz_m, source.area_weights_m2, velocity, front_xyz, frequency, rho0, c0
    )
    rear_velocity = -velocity * transmission
    rear_pressure = rayleigh_pressure(
        source.xyz_m, source.area_weights_m2, rear_velocity, rear_xyz, frequency, rho0, c0
    )

    enclosure_info: dict[str, Any] = {"enabled": False, "method": "none"}
    rear_exterior_pressure = None
    rear_exterior_z_m = None
    enclosure = config.get("enclosure", {})
    if bool(enclosure.get("enabled", False)):
        panels = rectangular_enclosure_panels(
            enclosure["width_m"],
            enclosure["height_m"],
            enclosure["depth_m"],
            tuple(enclosure.get("centre_xy_m", (0.0, 0.0))),
            tuple(enclosure.get("divisions", (14, 16, 12))),
        )
        incident_panels = rayleigh_pressure(
            source.xyz_m,
            source.area_weights_m2,
            rear_velocity,
            panels.centres_m,
            frequency,
            rho0,
            c0,
        )
        front_pressure += kirchhoff_rigid_scattering(
            panels, incident_panels, front_xyz, frequency, c0
        )
        rear_exterior_z_m = -float(enclosure["depth_m"]) - near_offset
        _, _, rear_exterior_xyz = observation_plane(
            plane_extent, rear_exterior_z_m, plane_points
        )
        rear_exterior_pressure = rayleigh_pressure(
            source.xyz_m,
            source.area_weights_m2,
            rear_velocity,
            rear_exterior_xyz,
            frequency,
            rho0,
            c0,
        )
        rear_exterior_pressure += kirchhoff_rigid_scattering(
            panels, incident_panels, rear_exterior_xyz, frequency, c0
        )
        enclosure_info = {
            "enabled": True,
            "method": "Kirchhoff rigid physical-optics approximation",
            "panel_count": int(len(panels.areas_m2)),
            "limitation": "finite-panel diffraction estimate; not a converged 3-D BEM/FEM",
        }

    polar, azimuth, far_xyz = observation_sphere(
        observation.get("far_radius_m", 2.0),
        observation.get("far_polar_points", 37),
        observation.get("far_azimuthal_points", 72),
    )
    front_half = far_xyz[:, 2] >= 0.0
    far_pressure = np.empty(len(far_xyz), dtype=complex)
    for mask, source_velocity in ((front_half, velocity), (~front_half, rear_velocity)):
        far_pressure[mask] = rayleigh_pressure(
            source.xyz_m,
            source.area_weights_m2,
            source_velocity,
            far_xyz[mask],
            frequency,
            rho0,
            c0,
        )
    if enclosure_info["enabled"]:
        far_pressure += kirchhoff_rigid_scattering(
            panels, incident_panels, far_xyz, frequency, c0
        )
    far_abs = np.abs(far_pressure)
    far_relative_db = 20.0 * np.log10(np.maximum(far_abs / max(float(np.max(far_abs)), 1e-300), 1e-12))
    sphere_weight = np.sin(polar)
    front_power_proxy = float(np.sum(sphere_weight[polar <= math.pi / 2] * far_abs.reshape(polar.shape)[polar <= math.pi / 2] ** 2))
    rear_power_proxy = float(np.sum(sphere_weight[polar > math.pi / 2] * far_abs.reshape(polar.shape)[polar > math.pi / 2] ** 2))
    metrics.update(
        {
            "basket_mean_amplitude_transmission": float(np.mean(transmission)),
            "basket_mode_coupling_offdiagonal_ratio": float(
                np.linalg.norm(coupling - np.diag(np.diag(coupling)))
                / max(np.linalg.norm(coupling), 1e-300)
            ),
            "front_to_rear_power_proxy_ratio": float(front_power_proxy / max(rear_power_proxy, 1e-300)),
        }
    )
    return {
        "metadata": {
            "analysis_kind": "circumferential_reduced_order_3d_acoustics",
            "frequency_Hz": frequency,
            "structural_contract": "user/config supplied Fourier mode amplitudes",
            "radiation_method": "baffled Rayleigh surface integral",
            "enclosure": enclosure_info,
            "not_yet_modeled": [
                "3-D structural eigenvalue solution",
                "geometric-nonlinear buckling and mode-amplitude prediction",
                "converged full-wave enclosure FEM/BEM",
            ],
        },
        "metrics": metrics,
        "source": source,
        "velocity_m_s_peak": velocity,
        "basket_transmission": transmission,
        "near_x_m": x,
        "near_y_m": y,
        "front_pressure_Pa_peak": front_pressure.reshape(x.shape),
        "rear_pressure_Pa_peak": rear_pressure.reshape(x.shape),
        "rear_exterior_pressure_Pa_peak": (
            None if rear_exterior_pressure is None else rear_exterior_pressure.reshape(x.shape)
        ),
        "rear_exterior_z_m": rear_exterior_z_m,
        "far_polar_rad": polar,
        "far_azimuth_rad": azimuth,
        "far_pressure_Pa_peak": far_pressure.reshape(polar.shape),
        "far_relative_dB": far_relative_db.reshape(polar.shape),
    }


def export_analysis(result: dict[str, Any], outdir: str | Path) -> dict[str, Any]:
    """Write machine-readable results without embedding binary data in JSON."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    source: AnnularSource = result["source"]
    fields_path = out / "asymmetry3d_fields.npz"
    np.savez_compressed(
        fields_path,
        source_xyz_m=source.xyz_m,
        source_area_weights_m2=source.area_weights_m2,
        source_velocity_m_s_peak=result["velocity_m_s_peak"],
        basket_transmission=result["basket_transmission"],
        near_x_m=result["near_x_m"],
        near_y_m=result["near_y_m"],
        front_pressure_Pa_peak=result["front_pressure_Pa_peak"],
        rear_pressure_Pa_peak=result["rear_pressure_Pa_peak"],
        rear_exterior_pressure_Pa_peak=(
            np.empty((0, 0), dtype=complex)
            if result["rear_exterior_pressure_Pa_peak"] is None
            else result["rear_exterior_pressure_Pa_peak"]
        ),
        rear_exterior_z_m=np.array(
            np.nan if result["rear_exterior_z_m"] is None else result["rear_exterior_z_m"]
        ),
        far_polar_rad=result["far_polar_rad"],
        far_azimuth_rad=result["far_azimuth_rad"],
        far_pressure_Pa_peak=result["far_pressure_Pa_peak"],
        far_relative_dB=result["far_relative_dB"],
    )
    summary = {
        **result["metadata"],
        "metrics": result["metrics"],
        "files": {"fields_npz": fields_path.name},
    }
    summary_path = out / "asymmetry3d_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {**summary, "summary_path": str(summary_path), "fields_path": str(fields_path)}
