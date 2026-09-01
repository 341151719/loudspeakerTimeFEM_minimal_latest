import math

import numpy as np

from loudspeaker_time_fem.asymmetry3d import (
    analyze,
    angular_mode_coupling,
    annular_source,
    basket_transmission,
    fourier_coefficients,
    modal_metrics,
    rayleigh_pressure,
    reconstruct_fourier,
    rectangular_enclosure_panels,
)


def test_fourier_roundtrip_preserves_nonaxisymmetric_modes():
    phi = 2.0 * math.pi * np.arange(128) / 128
    values = 1.2 + (0.4 - 0.2j) * np.cos(phi) + 0.7j * np.sin(phi)
    values += -0.33 * np.cos(4 * phi) + (0.1 + 0.05j) * np.sin(4 * phi)
    cosine, sine = fourier_coefficients(values, max_order=6)
    rebuilt = reconstruct_fourier(phi, cosine, sine)
    np.testing.assert_allclose(rebuilt, values, atol=2e-14)


def test_m1_velocity_recovers_rigid_rocking_tilt():
    source = annular_source(0.08, radial_points=16, azimuthal_points=96)
    frequency = 700.0
    tilt = 1.7e-3
    omega = 2.0 * math.pi * frequency
    velocity = 1j * omega * tilt * source.radius_m * np.cos(source.azimuth_rad)
    metrics = modal_metrics(source, velocity, frequency, max_order=4)
    np.testing.assert_allclose(metrics["rocking_tilt_x_rad_peak"], [tilt, 0.0], atol=2e-14)
    assert metrics["dominant_order"] == 1
    assert metrics["higher_order_breakup_fraction"] < 1e-28


def test_five_spoke_basket_couples_only_fivefold_order_differences():
    phi = 2.0 * math.pi * np.arange(720) / 720
    transmission = basket_transmission(phi, 5, 12.0, rotation_deg=7.0)
    orders, coupling = angular_mode_coupling(transmission, max_order=6)
    for i, target in enumerate(orders):
        for j, source in enumerate(orders):
            if (source - target) % 5:
                assert abs(coupling[i, j]) < 2e-12
    assert abs(coupling[np.where(orders == 0)[0][0], np.where(orders == 5)[0][0]]) > 1e-3


def test_rectangular_enclosure_panel_area_and_count():
    width, height, depth = 0.23, 0.28, 0.16
    nx, ny, nz = 6, 8, 5
    panels = rectangular_enclosure_panels(width, height, depth, divisions=(nx, ny, nz))
    assert len(panels.areas_m2) == 2 * ny * nz + 2 * nx * nz + nx * ny
    expected_area = 2 * height * depth + 2 * width * depth + width * height
    np.testing.assert_allclose(np.sum(panels.areas_m2), expected_area, rtol=2e-15)
    np.testing.assert_allclose(np.linalg.norm(panels.normals, axis=1), 1.0)


def test_pure_m1_source_cancels_on_axis_but_radiates_off_axis():
    source = annular_source(0.08, radial_points=12, azimuthal_points=96)
    velocity = source.radius_m * np.cos(source.azimuth_rad)
    observers = np.array([[0.0, 0.0, 0.5], [0.2, 0.0, 0.5]])
    pressure = rayleigh_pressure(
        source.xyz_m,
        source.area_weights_m2,
        velocity,
        observers,
        1200.0,
    )
    assert abs(pressure[0]) < 1e-12 * abs(pressure[1])
    assert abs(pressure[1]) > 0.0


def test_small_analysis_exports_both_near_fields_and_full_sphere():
    result = analyze(
        {
            "frequency_Hz": 900.0,
            "max_mode_order": 3,
            "diaphragm": {
                "outer_radius_m": 0.05,
                "radial_points": 5,
                "azimuthal_points": 24,
            },
            "modes": [
                {"order": 0, "amplitude_m_s_peak": 0.2},
                {"order": 1, "cosine_amplitude_m_s_peak": 0.03},
                {"order": 3, "cosine_amplitude_m_s_peak": 0.02},
            ],
            "basket": {"spokes": 3, "spoke_width_deg": 14.0},
            "enclosure": {"enabled": False},
            "observation": {
                "near_points": 9,
                "far_polar_points": 7,
                "far_azimuthal_points": 12,
            },
        }
    )
    assert result["front_pressure_Pa_peak"].shape == (9, 9)
    assert result["rear_pressure_Pa_peak"].shape == (9, 9)
    assert result["rear_exterior_pressure_Pa_peak"] is None
    assert result["far_relative_dB"].shape == (7, 12)
    assert np.ptp(result["far_relative_dB"], axis=1).max() > 0.0
    assert result["metadata"]["enclosure"]["method"] == "none"


def test_m0_limit_recovers_azimuthally_invariant_far_field():
    result = analyze(
        {
            "frequency_Hz": 600.0,
            "max_mode_order": 2,
            "diaphragm": {
                "outer_radius_m": 0.05,
                "radial_points": 6,
                "azimuthal_points": 48,
            },
            "modes": [{"order": 0, "amplitude_m_s_peak": 0.2}],
            "basket": {"spokes": 0},
            "enclosure": {"enabled": False},
            "observation": {
                "near_points": 7,
                "far_polar_points": 9,
                "far_azimuthal_points": 16,
            },
        }
    )
    magnitude = np.abs(result["far_pressure_Pa_peak"])
    np.testing.assert_allclose(
        magnitude,
        np.broadcast_to(magnitude[:, :1], magnitude.shape),
        rtol=2e-13,
        atol=1e-13,
    )
