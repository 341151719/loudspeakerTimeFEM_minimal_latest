import numpy as np

from loudspeaker_time_fem.nonlinear_law import NonlinearMagneticLaw


def test_coupled_coenergy_maxwell_reciprocity():
    law = NonlinearMagneticLaw.from_json(
        "inputs/nonlinear_magnetic_law_20260728.json"
    )
    for q in (-1.5e-3, 0.0, 1.5e-3):
        for current in (-0.5, 0.0, 0.5):
            dq = 1e-8
            di = 1e-6
            dflux_dq = (
                law.coupled_flux(q + dq, current)
                - law.coupled_flux(q - dq, current)
            ) / (2 * dq)
            force_plus = (current + di) * law.coupled_force_factor(q, current + di)
            force_minus = (current - di) * law.coupled_force_factor(q, current - di)
            dforce_di = (force_plus - force_minus) / (2 * di)
            np.testing.assert_allclose(dflux_dq, dforce_di, rtol=2e-8, atol=2e-8)
