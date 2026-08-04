from __future__ import annotations

import numpy as np
from pathlib import Path

from loudspeaker_time_fem.config import (
    assert_native_production_config,
    load_config,
    resolve_base_mainline,
)
from loudspeaker_time_fem.model import build_transient_model
from loudspeaker_time_fem.nonlinear_law import NonlinearMagneticLaw
from loudspeaker_time_fem.export import _cycle_convergence
from loudspeaker_time_fem.solver import soft_start
from loudspeaker_time_fem.spherical_nrbc import outgoing_modal_impedance


def test_soft_start_is_smooth_and_bounded():
    t = np.linspace(-1.0, 2.0, 301)
    y = soft_start(t, 1.0)
    assert np.all(y >= 0.0)
    assert np.all(y <= 1.0)
    assert y[0] == 0.0
    assert y[-1] == 1.0
    assert np.all(np.diff(y) >= -1e-14)


def test_soft_start_zero_duration_is_one():
    np.testing.assert_allclose(soft_start(np.array([0.0, 1.0]), 0.0), 1.0)


def test_nonlinear_magnetic_law_is_energy_conjugate():
    root = Path(__file__).resolve().parents[1]
    law = NonlinearMagneticLaw.from_json(
        root / "inputs/nonlinear_magnetic_law_20260728.json"
    )
    for x in np.linspace(-0.003, 0.003, 7):
        step = 1e-8
        derivative = (
            law.motional_flux(x + step) - law.motional_flux(x - step)
        ) / (2 * step)
        np.testing.assert_allclose(derivative, law.bl(x), rtol=2e-8, atol=2e-8)
    for current in np.linspace(-0.8, 0.8, 5):
        assert law.incremental_inductance(current) > 0.0


def test_default_config_uses_bundled_frequency_mainline(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    config, config_path = load_config(root / "configs/transient_70Hz.json")
    monkeypatch.setenv("LOUDSPEAKER_FREQUENCY_MAINLINE", "/missing/external/mainline")
    expected = (root / "inputs/frequency_mainline").resolve()
    assert resolve_base_mainline(config, config_path) == expected


def test_outside_probe_requires_explicit_diagnostic_opt_in():
    root = Path(__file__).resolve().parents[1]
    config, config_path = load_config(
        root / "configs/transient_70Hz_nonlinear_comsol_physical_abc.json"
    )
    probe = next(item for item in config["probes"] if item["name"] == "axis_rear_m0p12m")
    probe.pop("outside_domain_action")
    import pytest

    with pytest.raises(ValueError, match="outside the solved acoustic domain"):
        build_transient_model(config, config_path)


def test_native_production_rejects_reference_identified_boundary():
    root = Path(__file__).resolve().parents[1]
    config, _ = load_config(
        root
        / "configs/transient_70Hz_nonlinear_comsol_interface_up_robin_legendre2.json"
    )
    import pytest

    with pytest.raises(ValueError, match="native production rejects"):
        assert_native_production_config(config)


def test_tensor_coenergy_candidate_is_diagnostic_only():
    root = Path(__file__).resolve().parents[1]
    config, _ = load_config(
        root
        / "configs/transient_70Hz_nonlinear_comsol_physical_abc_tensor_coenergy_diagnostic.json"
    )
    import pytest

    with pytest.raises(ValueError, match="native production rejects"):
        assert_native_production_config(config)


def test_exact_spherical_nrbc_monopole_and_low_frequency_modes():
    radius = 0.115
    frequency = 70.0
    sound_speed = 343.2035820928282
    values = outgoing_modal_impedance(frequency, radius, sound_speed, 4)
    k = 2.0 * np.pi * frequency / sound_speed
    np.testing.assert_allclose(values[0], 1.0 / radius + 1j * k, rtol=2e-14)
    assert np.all(values[1:].real > values[:-1].real)
    assert np.all(values.imag >= 0.0)


def test_cycle_convergence_is_zero_for_repeated_periods():
    phase = np.linspace(0.0, 2.0 * np.pi, 33)
    cycle = np.sin(phase)
    values = np.concatenate([cycle[:-1], cycle[:-1], cycle])[:, None]
    np.testing.assert_allclose(_cycle_convergence(values, 32), 0.0, atol=1e-15)
