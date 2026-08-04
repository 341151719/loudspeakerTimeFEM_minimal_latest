from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from numpy.polynomial import Chebyshev, Polynomial


@dataclass
class NonlinearMagneticLaw:
    path: Path
    displacement_limit_m: float
    current_limit_A: float
    bl_polynomial: Chebyshev
    flux_polynomial: Polynomial
    metadata: dict
    current_bl_correction: Polynomial

    @classmethod
    def from_json(cls, path: str | Path) -> "NonlinearMagneticLaw":
        source = Path(path).resolve()
        data = json.loads(source.read_text(encoding="utf-8"))
        kind = data.get("kind")
        if kind == "native_tensor_coenergy_magnetic_law":
            from .tensor_coenergy import TensorCoenergyLaw

            return TensorCoenergyLaw.from_json(source)  # type: ignore[return-value]
        if kind != "field_derived_separable_magnetic_coenergy_ROM":
            raise ValueError(
                f"未知磁律 kind={kind!r}；拒绝把新 schema 静默解释为旧可分磁律"
            )
        xlim = float(data["displacement_limit_m"])
        bl = Chebyshev(
            np.asarray(data["bl_chebyshev_coefficients_N_A"], float),
            domain=[-xlim, xlim],
        )
        flux = Polynomial(
            np.asarray(data["lambda_polynomial_coefficients_Wb_ascending"], float)
        )
        raw_current = data.get("raw_current_scan", {})
        current_samples = np.asarray(raw_current.get("current_A", [0.0]), float)
        bl_samples = np.asarray(raw_current.get("BL_N_A", [float(bl(0.0))]), float)
        degree = min(2, len(current_samples) - 1)
        correction = Polynomial.fit(
            current_samples, bl_samples - float(bl(0.0)), degree
        ).convert()
        return cls(
            source,
            xlim,
            float(data["current_limit_A"]),
            bl,
            flux,
            data,
            correction,
        )

    def check_coordinates(self, displacement_m: float, current_A: float) -> None:
        if abs(displacement_m) > self.displacement_limit_m:
            raise RuntimeError(
                f"音圈位移 {displacement_m:.6g} m 超出磁场扫描范围 "
                f"±{self.displacement_limit_m:.6g} m"
            )
        if abs(current_A) > self.current_limit_A:
            raise RuntimeError(
                f"电流 {current_A:.6g} A 超出非线性磁场扫描范围 "
                f"±{self.current_limit_A:.6g} A"
            )

    def bl(self, displacement_m: float) -> float:
        return float(self.bl_polynomial(displacement_m))

    def dbl_dx(self, displacement_m: float) -> float:
        return float(self.bl_polynomial.deriv()(displacement_m))

    def motional_flux(self, displacement_m: float) -> float:
        primitive = self.bl_polynomial.integ()
        return float(primitive(displacement_m) - primitive(0.0))

    def current_flux(self, current_A: float) -> float:
        return float(self.flux_polynomial(current_A))

    def incremental_inductance(self, current_A: float) -> float:
        return float(self.flux_polynomial.deriv()(current_A))

    def bl_current_correction(self, current_A: float) -> float:
        return float(self.current_bl_correction(current_A))

    def dbl_current_di(self, current_A: float) -> float:
        return float(self.current_bl_correction.deriv()(current_A))

    def d2bl_current_di2(self, current_A: float) -> float:
        return float(self.current_bl_correction.deriv(2)(current_A))

    def coupled_force_factor(self, displacement_m: float, current_A: float) -> float:
        return self.bl(displacement_m) + self.bl_current_correction(current_A)

    def coupled_flux(self, displacement_m: float, current_A: float) -> float:
        correction = self.bl_current_correction(current_A)
        dcorrection = self.dbl_current_di(current_A)
        return (
            self.current_flux(current_A)
            + self.motional_flux(displacement_m)
            + displacement_m * (correction + current_A * dcorrection)
        )
