from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from loudspeaker_time_fem.suspension_rom import SuspensionROM


def _law(tmp_path: Path) -> SuspensionROM:
    path = tmp_path / "kms.json"
    path.write_text(
        json.dumps(
            {
                "kind": "polynomial_secant_stiffness_ratio",
                "displacement_scale_m": 0.004,
                "displacement_limit_m": 0.004,
                "stiffness_ratio_power_coefficients": [1.0, 0.08, 0.8, 0.12, 0.35],
            }
        ),
        encoding="utf-8",
    )
    return SuspensionROM.from_json(path, reference_stiffness_N_m=3000.0)


def test_secant_definition_and_small_signal_preservation(tmp_path: Path):
    law = _law(tmp_path)
    np.testing.assert_allclose(
        law.restoring_force(0.0012), law.secant_stiffness(0.0012) * 0.0012, rtol=1e-14
    )
    assert abs(law.correction_force(0.0)) < 1e-15
    assert abs(law.correction_tangent(0.0)) < 1e-12
    assert abs(law.secant_stiffness(0.0) - 3000.0) < 1e-12
    assert abs(law.tangent_stiffness(0.0) - 3000.0) < 1e-12


def test_tangent_matches_force_finite_difference(tmp_path: Path):
    law = _law(tmp_path)
    for q in (-0.003, -0.001, 0.0015, 0.003):
        eps = 1e-8
        numeric = (law.restoring_force(q + eps) - law.restoring_force(q - eps)) / (2 * eps)
        np.testing.assert_allclose(numeric, law.tangent_stiffness(q), rtol=2e-9, atol=2e-6)


def test_correction_potential_derivative_matches_correction_force(tmp_path: Path):
    law = _law(tmp_path)
    for q in (-0.003, -0.001, 0.0015, 0.003):
        eps = 1e-8
        numeric = (law.correction_potential(q + eps) - law.correction_potential(q - eps)) / (2 * eps)
        np.testing.assert_allclose(numeric, law.correction_force(q), rtol=5e-9, atol=2e-8)


def test_combined_sherman_morrison_matches_direct_tangent(tmp_path: Path):
    """Kms tangent and magnetic dBL/dx collapse to the same rank-one direction."""
    law = _law(tmp_path)
    rng = np.random.default_rng(42)
    B = rng.normal(size=(12, 12))
    A = B.T @ B + 4.0 * np.eye(12)
    h = rng.normal(size=12)
    q = 0.0024
    i = 0.45
    dbl_dx = -120.0  # N/A/m, representative sign only; algebra is the target.
    kms_correction_tangent = law.correction_tangent(q)
    rank_coefficient = i * dbl_dx - kms_correction_tangent
    direct = A - rank_coefficient * np.outer(h, h)
    rhs = rng.normal(size=12)

    Ainv_h = np.linalg.solve(A, h)
    denominator = 1.0 - rank_coefficient * float(h @ Ainv_h)
    base = np.linalg.solve(A, rhs)
    sherman = base + rank_coefficient * Ainv_h * float(h @ base) / denominator
    expected = np.linalg.solve(direct, rhs)
    np.testing.assert_allclose(sherman, expected, rtol=2e-12, atol=2e-12)


def test_declared_range_rejects_excursion(tmp_path: Path):
    law = _law(tmp_path)
    try:
        law.restoring_force(0.0041)
    except ValueError as exc:
        assert "outside declared range" in str(exc)
    else:
        raise AssertionError("expected declared coordinate range to be enforced")


def test_klippel_table_fit_recovers_polynomial_without_absolute_path(tmp_path: Path):
    x_mm = np.linspace(-4.0, 4.0, 41)
    xi = x_mm / 4.0
    coefficients = np.array([1.0, 0.08, 0.8, 0.12, 0.35])
    kms_N_mm = 3.0 * np.polynomial.polynomial.polyval(xi, coefficients)
    csv_path = tmp_path / "measured_kms.csv"
    np.savetxt(
        csv_path,
        np.column_stack([x_mm, kms_N_mm]),
        delimiter=",",
        header="x_mm,Kms_N_per_mm",
        comments="",
    )
    out = tmp_path / "fitted.json"
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(root / "tools/fit_suspension_kms_rom.py"),
            str(csv_path),
            "--order",
            "4",
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    np.testing.assert_allclose(
        payload["stiffness_ratio_power_coefficients"], coefficients, atol=2e-14
    )
    metadata = payload["metadata"]
    assert metadata["source_csv_name"] == csv_path.name
    assert len(metadata["source_csv_sha256"]) == 64
    assert str(tmp_path) not in json.dumps(payload)


def test_fit_limit_cannot_extrapolate_past_shorter_measured_side(tmp_path: Path):
    csv_path = tmp_path / "asymmetric_range.csv"
    np.savetxt(
        csv_path,
        np.array([[-4.0, 3.2], [-2.0, 3.05], [0.0, 3.0], [1.0, 3.02], [3.0, 3.15]]),
        delimiter=",",
        header="x_mm,Kms_N_per_mm",
        comments="",
    )
    root = Path(__file__).resolve().parents[1]
    failed = subprocess.run(
        [
            sys.executable,
            str(root / "tools/fit_suspension_kms_rom.py"),
            str(csv_path),
            "--order",
            "2",
            "--limit-mm",
            "4",
            "--out",
            str(tmp_path / "invalid.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert failed.returncode != 0
    assert "exceeds the displacement range measured on both sides" in failed.stderr
