from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from loudspeaker_time_fem.nonlinear_law import NonlinearMagneticLaw
from loudspeaker_time_fem.tensor_coenergy import TensorCoenergyLaw


def _synthetic_law(order: int = 12) -> TensorCoenergyLaw:
    x = np.linspace(-0.004, 0.004, 9)
    i = np.linspace(-1.0, 1.0, 9)
    X, I = np.meshgrid(x, i, indexing="ij")
    # A polynomial of degree <=3 in each normalized coordinate.  It contains
    # PM motion flux, current nonlinearity and a genuine x-i coupling while
    # retaining a comfortably positive differential inductance.
    psi = (
        0.0012 * (X / 0.004 + 0.15 * (X / 0.004) ** 2)
        + 0.0017 * I
        + 0.00012 * (X / 0.004) * I
        + 0.00003 * I**3
    )
    return TensorCoenergyLaw(
        Path("synthetic.json"),
        Path("synthetic.npz"),
        0.004,
        1.0,
        x,
        i,
        psi - psi[4, 9 // 2],
        float(np.max(np.abs(psi))),
        0.0,
        {"kind": "synthetic"},
        order,
    )


def test_tensor_api_has_one_scalar_surface_and_positive_inductance():
    law = _synthetic_law()
    assert law.coenergy(0.001, 0.0) == pytest.approx(0.0, abs=1e-14)
    assert law.flux(0.0, 0.0) == pytest.approx(0.0, abs=1e-14)
    assert law.force(0.001, 0.0) == pytest.approx(0.0, abs=1e-14)
    values = law.incremental_inductance(
        np.linspace(-0.0039, 0.0039, 17)[:, None], np.linspace(-0.99, 0.99, 17)
    )
    assert np.min(values) > 0.0
    assert law.dforce_di(0.001, 0.2) == law.dflux_dx(0.001, 0.2)


def test_tensor_derivatives_converge_against_centered_finite_difference():
    law = _synthetic_law()
    rng = np.random.default_rng(20260801)
    points = np.column_stack(
        [rng.uniform(-0.0039, 0.0039, 64), rng.uniform(-0.98, 0.98, 64)]
    )
    for x, i in points:
        dx = 1e-6
        di = 1e-4
        force_di_h = (law.force(x, i + di) - law.force(x, i - di)) / (2 * di)
        force_di_h2 = (law.force(x, i + di / 2) - law.force(x, i - di / 2)) / di
        flux_dx_h = (law.flux(x + dx, i) - law.flux(x - dx, i)) / (2 * dx)
        flux_dx_h2 = (law.flux(x + dx / 2, i) - law.flux(x - dx / 2, i)) / dx
        np.testing.assert_allclose(force_di_h, law.dforce_di(x, i), rtol=1e-5, atol=2e-8)
        np.testing.assert_allclose(force_di_h2, law.dforce_di(x, i), rtol=1e-5, atol=2e-8)
        np.testing.assert_allclose(flux_dx_h, law.dflux_dx(x, i), rtol=1e-5, atol=2e-8)
        np.testing.assert_allclose(flux_dx_h2, law.dflux_dx(x, i), rtol=1e-5, atol=2e-8)
        force_x_h = (law.force(x + dx, i) - law.force(x - dx, i)) / (2 * dx)
        force_x_h2 = (law.force(x + dx / 2, i) - law.force(x - dx / 2, i)) / dx
        np.testing.assert_allclose(force_x_h, law.dforce_dx(x, i), rtol=1e-5, atol=2e-7)
        assert abs(force_x_h2 - law.dforce_dx(x, i)) <= abs(force_x_h - law.dforce_dx(x, i)) + 1e-7


def test_tensor_boundaries_are_inclusive_and_outside_is_rejected():
    law = _synthetic_law()
    law.check_coordinates(-0.004, -1.0)
    law.check_coordinates(0.004, 1.0)
    with pytest.raises(RuntimeError, match="超出磁场扫描范围"):
        law.check_coordinates(0.004000001, 0.0)
    with pytest.raises(RuntimeError, match="超出非线性磁场扫描范围"):
        law.check_coordinates(0.0, -1.000001)


def test_tensor_schema_dispatch_rejects_unknown_kind(tmp_path: Path):
    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps({"kind": "not-a-law"}), encoding="utf-8")
    with pytest.raises(ValueError, match="未知磁律"):
        NonlinearMagneticLaw.from_json(unknown)


def test_tensor_axis_order_and_negative_inductance_contracts():
    law = _synthetic_law()
    assert law.psi_training_Wb.shape == (law.x_axis_m.size, law.current_axis_A.size)
    assert law.x_axis_m[0] < law.x_axis_m[-1]
    assert law.current_axis_A[0] < law.current_axis_A[-1]
    assert np.min(law.incremental_inductance(np.linspace(-.0039, .0039, 11)[:, None], np.linspace(-.99, .99, 11))) > 1e-6
    # The scalar API must not manufacture BL=F/i at zero current.
    np.testing.assert_allclose(law.effective_bl(0.001, 0.0), law.dforce_di(0.001, 0.0), rtol=0, atol=0)


def test_tensor_gauge_and_force_sign_are_not_hidden_by_a_second_fit():
    law = _synthetic_law()
    x = np.asarray([-0.003, 0.0, 0.003])
    i = np.asarray([-0.7, 0.0, 0.8])
    X, I = np.meshgrid(x, i, indexing="ij")
    assert np.all(np.isfinite(law.coenergy(X, I)))
    assert np.all(np.isfinite(law.force(X, I)))
    # Maxwell reciprocity is the exact same analytic value, not two fitted surfaces.
    np.testing.assert_array_equal(law.dforce_di(X, I), law.dflux_dx(X, I))


def test_real_tensor_schema_when_fit_artifact_exists():
    path = Path("inputs/nonlinear_magnetic_coenergy_tensor_20260801.json")
    if not path.is_file():
        pytest.skip("native tensor fit is produced by the staged scan")
    law = NonlinearMagneticLaw.from_json(path)
    assert isinstance(law, TensorCoenergyLaw)
    assert law.metadata["schema_version"] == 1
    assert law.x_axis_m.size >= 27
    assert law.current_axis_A.size >= 19
