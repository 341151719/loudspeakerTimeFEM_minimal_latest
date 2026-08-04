from __future__ import annotations

import numpy as np
import pytest

from loudspeaker_time_fem.nonlinear_solver import tensor_newton_residual_jacobian
from test_tensor_coenergy import _synthetic_law


def test_complete_tensor_newton_jacobian_matches_directional_difference():
    law = _synthetic_law()
    rng = np.random.default_rng(20260801)
    n = 7
    matrix = np.diag(np.linspace(2.0, 4.0, n))
    h = rng.normal(size=n)
    state = np.r_[rng.uniform(-0.001, 0.001, n), 0.15]
    rhs = rng.normal(size=n) * 0.1
    residual, jacobian = tensor_newton_residual_jacobian(
        law,
        matrix,
        h,
        state,
        0.15,
        rhs,
        0.0002,
        0.12,
        1e-4,
        8.0,
        0.2,
    )
    direction = rng.normal(size=n + 1)
    direction /= np.linalg.norm(direction)
    hstep = 1e-7
    plus = state + hstep * direction
    minus = state - hstep * direction
    plus_residual, _ = tensor_newton_residual_jacobian(
        law, matrix, h, plus, 0.15, rhs, 0.0002, 0.12, 1e-4, 8.0, 0.2
    )
    minus_residual, _ = tensor_newton_residual_jacobian(
        law, matrix, h, minus, 0.15, rhs, 0.0002, 0.12, 1e-4, 8.0, 0.2
    )
    finite_difference = (plus_residual - minus_residual) / (2 * hstep)
    np.testing.assert_allclose(
        finite_difference,
        jacobian @ direction,
        rtol=2e-6,
        atol=2e-8,
    )


def test_mixed_derivative_is_the_same_object_contract():
    law = _synthetic_law()
    for x in (-0.003, 0.0, 0.003):
        for i in (-0.7, 0.2, 0.8):
            assert law.dforce_di(x, i) == law.dflux_dx(x, i)


def test_zero_voltage_newton_step_is_passive_and_stationary():
    law = _synthetic_law()
    n = 4
    matrix = np.diag(np.linspace(2.0, 3.0, n))
    h = np.zeros(n)
    state = np.zeros(n + 1)
    residual, jacobian = tensor_newton_residual_jacobian(
        law, matrix, h, state, 0.0, np.zeros(n), 0.0, 0.0, 1e-4, 8.0, 0.0
    )
    np.testing.assert_allclose(residual, 0.0, atol=1e-14)
    assert np.all(np.linalg.eigvalsh(jacobian) > 0.0)
    assert law.magnetic_energy(0.0, 0.0) == 0.0


def test_newton_rejects_nonpositive_incremental_inductance():
    law = _synthetic_law()
    law.incremental_inductance = lambda _x, _i: -1.0  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="W_ii"):
        tensor_newton_residual_jacobian(
            law, np.eye(2), np.zeros(2), np.zeros(3), 0.0,
            np.zeros(2), 0.0, 0.0, 1e-4, 8.0, 0.0
        )
