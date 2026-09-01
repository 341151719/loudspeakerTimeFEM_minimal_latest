from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
from scipy.sparse import csr_matrix


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inputs" / "frequency_mainline" / "src"))
sys.path.insert(0, str(ROOT / "inputs" / "frequency_mainline" / "best_model"))

from p2_axisym_solid import complex_stiffness  # noqa: E402


def _minimal_cone_model():
    material = SimpleNamespace(loss_factor=0.04, beta_dK=0.0)
    return SimpleNamespace(
        ndof=1,
        K_by_domain={21: csr_matrix([[2.0]])},
        materials={21: material},
    )


def test_breakup_diagnostic_can_apply_stiffness_multiplier_full_band():
    model = _minimal_cone_model()
    regions = {"cone_1": csr_matrix([[2.0]])}
    multipliers = {"cone_1": 2.5}

    production_transition = complex_stiffness(
        model,
        2.0 * np.pi * 100.0,
        100.0,
        regions,
        multipliers,
        transition_start_Hz=2500.0,
        transition_end_Hz=4500.0,
    )
    full_band_diagnostic = complex_stiffness(
        model,
        2.0 * np.pi * 100.0,
        100.0,
        regions,
        multipliers,
        transition_start_Hz=0.0,
        transition_end_Hz=1.0,
    )

    assert np.isclose(production_transition[0, 0].real, 2.0)
    assert np.isclose(full_band_diagnostic[0, 0].real, 5.0)
