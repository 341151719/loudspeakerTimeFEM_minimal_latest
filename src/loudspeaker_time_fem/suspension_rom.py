from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SuspensionROM:
    """One-coordinate conservative suspension ROM for nonlinear Kms(q).

    The ROM follows the Klippel secant-stiffness convention

        F_s(q) = Kms(q) * q.

    The full FEM already contains the small-signal linear structural stiffness,
    therefore this class exposes an *incremental correction*

        Delta F_s(q) = F_s(q) - Kms(0) * q

    which has zero force and zero tangent correction at q=0.  This keeps the
    validated small-signal FEM untouched while adding large-signal suspension
    hardening/asymmetry through the generalized coil coordinate q=h^T u.
    """

    path: Path
    reference_stiffness_N_m: float
    displacement_scale_m: float
    displacement_limit_m: float
    stiffness_ratio_coefficients: np.ndarray
    metadata: dict[str, Any]

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        reference_stiffness_N_m: float,
    ) -> "SuspensionROM":
        source = Path(path).resolve()
        data = json.loads(source.read_text(encoding="utf-8"))
        kind = str(data.get("kind", "polynomial_secant_stiffness_ratio"))
        if kind != "polynomial_secant_stiffness_ratio":
            raise ValueError(
                "unsupported suspension ROM kind; expected "
                "'polynomial_secant_stiffness_ratio'"
            )
        scale = float(data["displacement_scale_m"])
        limit = float(data.get("displacement_limit_m", scale))
        coeff = np.asarray(data["stiffness_ratio_power_coefficients"], dtype=float)
        if scale <= 0.0 or limit <= 0.0:
            raise ValueError("suspension ROM displacement scale/limit must be positive")
        if coeff.ndim != 1 or len(coeff) < 1:
            raise ValueError("suspension ROM coefficients must be a non-empty 1D list")
        if not np.all(np.isfinite(coeff)):
            raise ValueError("suspension ROM coefficients must be finite")
        if abs(float(coeff[0]) - 1.0) > 1e-10:
            raise ValueError(
                "stiffness ratio must equal 1 at q=0 so the FEM small-signal tangent is preserved"
            )
        k0 = float(reference_stiffness_N_m)
        if not np.isfinite(k0) or k0 <= 0.0:
            raise ValueError("reference stiffness must be positive")
        law = cls(
            path=source,
            reference_stiffness_N_m=k0,
            displacement_scale_m=scale,
            displacement_limit_m=limit,
            stiffness_ratio_coefficients=coeff,
            metadata={**data.get("metadata", {}), "kind": kind},
        )
        # Reject nonphysical ROMs in their declared operating range.
        q = np.linspace(-limit, limit, 2001)
        if float(np.min(law.secant_stiffness(q))) <= 0.0:
            raise ValueError("suspension ROM has non-positive secant stiffness in declared range")
        if float(np.min(law.tangent_stiffness(q))) <= 0.0:
            raise ValueError("suspension ROM has non-positive incremental stiffness in declared range")
        return law

    def _xi(self, q_m: np.ndarray | float) -> np.ndarray:
        return np.asarray(q_m, dtype=float) / self.displacement_scale_m

    def _check(self, q_m: np.ndarray | float) -> None:
        q = np.asarray(q_m, dtype=float)
        if np.any(np.abs(q) > self.displacement_limit_m * (1.0 + 1e-12)):
            raise ValueError(
                f"suspension ROM coordinate outside declared range +/-{self.displacement_limit_m:g} m"
            )

    def stiffness_ratio(self, q_m: np.ndarray | float) -> np.ndarray | float:
        self._check(q_m)
        xi = self._xi(q_m)
        value = np.polynomial.polynomial.polyval(xi, self.stiffness_ratio_coefficients)
        return float(value) if np.ndim(q_m) == 0 else value

    def stiffness_ratio_derivative_per_m(
        self, q_m: np.ndarray | float
    ) -> np.ndarray | float:
        self._check(q_m)
        xi = self._xi(q_m)
        derivative_coeff = np.polynomial.polynomial.polyder(
            self.stiffness_ratio_coefficients
        )
        value = (
            np.polynomial.polynomial.polyval(xi, derivative_coeff)
            / self.displacement_scale_m
        )
        return float(value) if np.ndim(q_m) == 0 else value

    def secant_stiffness(self, q_m: np.ndarray | float) -> np.ndarray | float:
        value = self.reference_stiffness_N_m * np.asarray(self.stiffness_ratio(q_m))
        return float(value) if np.ndim(q_m) == 0 else value

    def restoring_force(self, q_m: np.ndarray | float) -> np.ndarray | float:
        q = np.asarray(q_m, dtype=float)
        value = q * np.asarray(self.secant_stiffness(q_m))
        return float(value) if np.ndim(q_m) == 0 else value

    def tangent_stiffness(self, q_m: np.ndarray | float) -> np.ndarray | float:
        q = np.asarray(q_m, dtype=float)
        ratio = np.asarray(self.stiffness_ratio(q_m))
        dr_dq = np.asarray(self.stiffness_ratio_derivative_per_m(q_m))
        value = self.reference_stiffness_N_m * (ratio + q * dr_dq)
        return float(value) if np.ndim(q_m) == 0 else value

    def correction_force(self, q_m: np.ndarray | float) -> np.ndarray | float:
        q = np.asarray(q_m, dtype=float)
        value = np.asarray(self.restoring_force(q_m)) - self.reference_stiffness_N_m * q
        return float(value) if np.ndim(q_m) == 0 else value

    def correction_tangent(self, q_m: np.ndarray | float) -> np.ndarray | float:
        value = np.asarray(self.tangent_stiffness(q_m)) - self.reference_stiffness_N_m
        return float(value) if np.ndim(q_m) == 0 else value

    def correction_potential(self, q_m: np.ndarray | float) -> np.ndarray | float:
        """Conservative potential whose derivative is correction_force(q)."""
        self._check(q_m)
        q = np.asarray(q_m, dtype=float)
        xi = q / self.displacement_scale_m
        # Delta U = K0*s^2 * integral xi * (ratio(xi)-1) dxi.
        delta_coeff = self.stiffness_ratio_coefficients.copy()
        delta_coeff[0] -= 1.0
        force_xi_coeff = np.r_[0.0, delta_coeff]  # xi * (ratio - 1)
        potential_coeff = np.polynomial.polynomial.polyint(force_xi_coeff)
        value = (
            self.reference_stiffness_N_m
            * self.displacement_scale_m**2
            * np.polynomial.polynomial.polyval(xi, potential_coeff)
        )
        return float(value) if np.ndim(q_m) == 0 else value

    def diagnostics(self) -> dict[str, Any]:
        q = np.linspace(-self.displacement_limit_m, self.displacement_limit_m, 2001)
        ks = np.asarray(self.secant_stiffness(q))
        kt = np.asarray(self.tangent_stiffness(q))
        return {
            "kind": self.metadata.get("kind"),
            "law_path": str(self.path),
            "reference_stiffness_N_m": self.reference_stiffness_N_m,
            "displacement_scale_m": self.displacement_scale_m,
            "displacement_limit_m": self.displacement_limit_m,
            "stiffness_ratio_power_coefficients": self.stiffness_ratio_coefficients.tolist(),
            "secant_stiffness_min_max_N_m": [float(np.min(ks)), float(np.max(ks))],
            "tangent_stiffness_min_max_N_m": [float(np.min(kt)), float(np.max(kt))],
            **{k: v for k, v in self.metadata.items() if k != "kind"},
        }
