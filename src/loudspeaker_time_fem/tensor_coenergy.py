"""Tensor-product magnetic coenergy law.

The data file consumed here is deliberately small and boring: it contains the
raw native-FEM ``psi(x, i)`` tensor and a JSON sidecar freezes the spline
choice and provenance.  All mechanical/electrical quantities are derivatives
of that one scalar surface.  Keeping this implementation independent from the
legacy separable law is important: an old JSON file must never silently select
the new physics.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.interpolate import RectBivariateSpline


class TensorCoenergyLaw:
    """C2 tensor coenergy representation backed by a cubic B-spline.

    Coordinates are checked before every public evaluation.  The spline is
    fitted in normalized coordinates, while the public API uses SI units.
    ``psi`` is the globally gauged increment ``psi_raw(x, i)-psi_raw(0, 0)``;
    no x-dependent zeroing is performed.
    """

    def __init__(
        self,
        path: Path,
        npz_path: Path,
        displacement_limit_m: float,
        current_limit_A: float,
        x_axis_m: np.ndarray,
        current_axis_A: np.ndarray,
        psi_training_Wb: np.ndarray,
        psi_scale_Wb: float,
        smoothing_s_normalized: float,
        metadata: dict,
        gauss_order: int = 12,
    ) -> None:
        self.path = Path(path).resolve()
        self.npz_path = Path(npz_path).resolve()
        self.displacement_limit_m = float(displacement_limit_m)
        self.current_limit_A = float(current_limit_A)
        self.x_axis_m = np.asarray(x_axis_m, dtype=float)
        self.current_axis_A = np.asarray(current_axis_A, dtype=float)
        self.psi_training_Wb = np.asarray(psi_training_Wb, dtype=float)
        self.psi_scale_Wb = float(psi_scale_Wb)
        self.smoothing_s_normalized = float(smoothing_s_normalized)
        self.metadata = metadata
        self.gauss_order = int(gauss_order)
        if self.x_axis_m.ndim != 1 or self.current_axis_A.ndim != 1:
            raise ValueError("tensor axes must be one-dimensional")
        if self.psi_training_Wb.shape != (
            len(self.x_axis_m),
            len(self.current_axis_A),
        ):
            raise ValueError(
                "psi tensor axis order must be [x_index, i_index]; got "
                f"{self.psi_training_Wb.shape}"
            )
        if len(self.x_axis_m) < 4 or len(self.current_axis_A) < 4:
            raise ValueError("cubic tensor spline needs at least four points per axis")
        if not np.all(np.diff(self.x_axis_m) > 0) or not np.all(
            np.diff(self.current_axis_A) > 0
        ):
            raise ValueError("tensor axes must be strictly increasing")
        if self.psi_scale_Wb <= 0 or not np.isfinite(self.psi_scale_Wb):
            raise ValueError("psi_scale_Wb must be finite and positive")
        self._x_scale = self.displacement_limit_m
        self._i_scale = self.current_limit_A
        xhat = self.x_axis_m / self._x_scale
        ihat = self.current_axis_A / self._i_scale
        normalized = self.psi_training_Wb / self.psi_scale_Wb
        self._spline = RectBivariateSpline(
            xhat,
            ihat,
            normalized,
            kx=3,
            ky=3,
            s=self.smoothing_s_normalized,
        )
        self._psi_zero_normalized = float(
            np.asarray(self._spline.ev(0.0, 0.0), dtype=float).reshape(-1)[0]
        )
        nodes, weights = np.polynomial.legendre.leggauss(self.gauss_order)
        self._gauss_nodes = nodes
        self._gauss_weights = weights

    @classmethod
    def from_json(cls, path: str | Path) -> "TensorCoenergyLaw":
        source = Path(path).resolve()
        data = json.loads(source.read_text(encoding="utf-8"))
        schema = data.get("schema_version")
        if data.get("kind") != "native_tensor_coenergy_magnetic_law":
            raise ValueError(
                f"{source}: expected native_tensor_coenergy_magnetic_law, "
                f"got {data.get('kind')!r}"
            )
        if schema not in (1, "1", "1.0"):
            raise ValueError(f"{source}: unsupported tensor coenergy schema {schema!r}")
        npz_name = data.get("data_npz")
        if not npz_name:
            raise ValueError(f"{source}: missing data_npz")
        npz_path = (source.parent / str(npz_name)).resolve()
        if not npz_path.is_file():
            raise FileNotFoundError(f"tensor coenergy NPZ not found: {npz_path}")
        expected_hash = data.get("data_npz_sha256")
        if expected_hash:
            import hashlib

            digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
            if digest != expected_hash:
                raise ValueError(
                    f"{source}: NPZ SHA256 mismatch, expected {expected_hash}, got {digest}"
                )
        with np.load(npz_path, allow_pickle=False) as raw:
            required = {"x_training_m", "current_training_A", "psi_training_Wb"}
            missing = sorted(required.difference(raw.files))
            if missing:
                raise ValueError(f"{npz_path}: missing arrays {missing}")
            x_axis = np.asarray(raw["x_training_m"], dtype=float)
            i_axis = np.asarray(raw["current_training_A"], dtype=float)
            psi = np.asarray(raw["psi_training_Wb"], dtype=float)
        fit = data.get("fit", {})
        return cls(
            source,
            npz_path,
            float(data["displacement_limit_m"]),
            float(data["current_limit_A"]),
            x_axis,
            i_axis,
            psi,
            float(fit["psi_scale_Wb"]),
            float(fit["smoothing_s_normalized"]),
            data,
            int(fit.get("gauss_order", 12)),
        )

    def check_coordinates(
        self,
        displacement_m: float | np.ndarray,
        current_A: float | np.ndarray,
    ) -> None:
        x = np.asarray(displacement_m, dtype=float)
        i = np.asarray(current_A, dtype=float)
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(i)):
            raise RuntimeError("磁律坐标必须是有限数")
        x_bad = (x < -self.displacement_limit_m) | (x > self.displacement_limit_m)
        i_bad = (i < -self.current_limit_A) | (i > self.current_limit_A)
        if np.any(x_bad):
            value = float(x.flat[np.flatnonzero(x_bad)[0]])
            raise RuntimeError(
                f"音圈位移 {value:.12g} m 超出磁场扫描范围 "
                f"[-{self.displacement_limit_m:.12g}, {self.displacement_limit_m:.12g}] m"
            )
        if np.any(i_bad):
            value = float(i.flat[np.flatnonzero(i_bad)[0]])
            raise RuntimeError(
                f"电流 {value:.12g} A 超出非线性磁场扫描范围 "
                f"[-{self.current_limit_A:.12g}, {self.current_limit_A:.12g}] A"
            )

    @staticmethod
    def _return(value: np.ndarray | float):
        array = np.asarray(value, dtype=float)
        return float(array) if array.ndim == 0 else array

    def _eval_normalized(
        self,
        displacement_m: float | np.ndarray,
        current_A: float | np.ndarray,
        dx: int = 0,
        dy: int = 0,
    ):
        self.check_coordinates(displacement_m, current_A)
        x, i = np.broadcast_arrays(
            np.asarray(displacement_m, dtype=float),
            np.asarray(current_A, dtype=float),
        )
        value = self._spline.ev(
            (x / self._x_scale).ravel(),
            (i / self._i_scale).ravel(),
            dx=dx,
            dy=dy,
        ).reshape(x.shape)
        return self._return(value)

    def _psi_normalized(self, displacement_m, current_A):
        return np.asarray(
            self._eval_normalized(displacement_m, current_A), dtype=float
        ) - self._psi_zero_normalized

    def _derivative(self, displacement_m, current_A, dx: int, dy: int):
        scale = self.psi_scale_Wb / (self._x_scale**dx * self._i_scale**dy)
        value = np.asarray(
            self._eval_normalized(displacement_m, current_A, dx=dx, dy=dy),
            dtype=float,
        )
        return self._return(scale * value)

    def _integral(self, function: Callable, displacement_m, current_A):
        self.check_coordinates(displacement_m, current_A)
        x, i = np.broadcast_arrays(
            np.asarray(displacement_m, dtype=float),
            np.asarray(current_A, dtype=float),
        )
        nodes = self._gauss_nodes
        weights = self._gauss_weights
        samples_i = 0.5 * i[..., None] * (nodes + 1.0)
        samples_x = np.broadcast_to(x[..., None], samples_i.shape)
        values = np.asarray(function(samples_x, samples_i), dtype=float)
        result = 0.5 * i * np.sum(values * weights, axis=-1)
        return self._return(result)

    def flux(self, displacement_m, current_A):
        """Incremental magnetic flux linkage ``W_i`` in Wb."""
        value = self.psi_scale_Wb * self._psi_normalized(displacement_m, current_A)
        return self._return(value)

    dW_di = flux

    def dpsi_dx(self, displacement_m, current_A):
        return self._derivative(displacement_m, current_A, 1, 0)

    def dpsi_di(self, displacement_m, current_A):
        return self._derivative(displacement_m, current_A, 0, 1)

    def d2psi_dx2(self, displacement_m, current_A):
        return self._derivative(displacement_m, current_A, 2, 0)

    def coenergy(self, displacement_m, current_A):
        return self._integral(self.flux, displacement_m, current_A)

    def dW_dx(self, displacement_m, current_A):
        return self._integral(self.dpsi_dx, displacement_m, current_A)

    force = dW_dx

    def dforce_dx(self, displacement_m, current_A):
        return self._integral(self.d2psi_dx2, displacement_m, current_A)

    def dforce_di(self, displacement_m, current_A):
        """The mixed derivative ``W_xi = psi_x``."""
        return self.dpsi_dx(displacement_m, current_A)

    def dflux_dx(self, displacement_m, current_A):
        return self.dforce_di(displacement_m, current_A)

    def incremental_inductance(self, displacement_m, current_A):
        """Return ``W_ii`` with the public coordinate order ``(x, i)``."""
        return self.dpsi_di(displacement_m, current_A)

    def effective_bl(self, displacement_m, current_A):
        x, i = np.broadcast_arrays(
            np.asarray(displacement_m, dtype=float),
            np.asarray(current_A, dtype=float),
        )
        force = np.asarray(self.force(x, i), dtype=float)
        tangent = np.asarray(self.dforce_di(x, i), dtype=float)
        value = np.array(tangent, dtype=float, copy=True)
        np.divide(force, i, out=value, where=np.abs(i) > 1e-8)
        return self._return(value)

    def magnetic_energy(self, displacement_m, current_A):
        x, i = np.broadcast_arrays(
            np.asarray(displacement_m, dtype=float),
            np.asarray(current_A, dtype=float),
        )
        return self._return(i * np.asarray(self.flux(x, i)) - np.asarray(self.coenergy(x, i)))

    def evaluate(self, displacement_m: float, current_A: float) -> dict[str, float]:
        self.check_coordinates(displacement_m, current_A)
        return {
            "x_m": float(displacement_m),
            "current_A": float(current_A),
            "W_J": float(self.coenergy(displacement_m, current_A)),
            "psi_Wb": float(self.flux(displacement_m, current_A)),
            "F_N": float(self.force(displacement_m, current_A)),
            "W_xx_N_m": float(self.dforce_dx(displacement_m, current_A)),
            "W_xi_N_A": float(self.dforce_di(displacement_m, current_A)),
            "W_ii_H": float(self.incremental_inductance(displacement_m, current_A)),
            "BL_secant_N_A": float(self.effective_bl(displacement_m, current_A)),
            "BL_tangent_N_A": float(self.dforce_di(displacement_m, current_A)),
        }
