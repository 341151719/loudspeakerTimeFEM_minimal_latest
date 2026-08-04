#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from loudspeaker_time_fem.config import load_config
from loudspeaker_time_fem.model import build_transient_model
from loudspeaker_time_fem.nonlinear_solver import solve_nonlinear_transient


def fit_h1(time: np.ndarray, value: np.ndarray, f0: float) -> complex:
    omega = 2.0 * np.pi * f0
    design = np.column_stack(
        (np.ones_like(time), np.sin(omega * time), np.cos(omega * time))
    )
    coefficient, *_ = np.linalg.lstsq(design, value, rcond=None)
    return complex(coefficient[2], -coefficient[1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--comsol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    base, config_path = load_config(args.config)
    pressure = pd.read_csv(args.comsol / "pressure_points_timeseries.csv").pivot_table(
        index="time_s", columns="probe_name", values="p_Pa", aggfunc="last"
    )
    f0 = float(base["drive"]["frequency_Hz"])
    lo, hi = 3.0 / f0 - 1e-10, 4.0 / f0 - 1e-10
    cmask = (pressure.index.to_numpy() >= lo) & (pressure.index.to_numpy() < hi)
    reference_names = [
        "python_axis_near_actual",
        "python_axis_rear_actual",
        "python_offaxis_actual",
    ]
    reference = {
        name: fit_h1(
            pressure.index.to_numpy()[cmask],
            pressure[name].to_numpy()[cmask],
            f0,
        )
        for name in reference_names
    }
    variants = [
        ("sponge_0p1_abc", 0.1, True),
        ("sponge_0p5_abc", 0.5, True),
        ("sponge_0p9_abc", 0.9, True),
        ("no_sponge_abc", 1.0, True),
        ("no_sponge_reflective", 1.0, False),
    ]
    rows = []
    for variant, target, radiation in variants:
        config = copy.deepcopy(base)
        config["absorbing_layer"]["target_one_way_amplitude"] = target
        config["absorbing_layer"]["outer_first_order_radiation_condition"] = radiation
        model = build_transient_model(config, config_path)
        try:
            result = solve_nonlinear_transient(model)
        except RuntimeError as error:
            rows.append(
                {
                    "variant": variant,
                    "target_one_way_amplitude": target,
                    "outer_abc": radiation,
                    "signal": "solver",
                    "status": "failed",
                    "message": str(error),
                }
            )
            continue
        pmask = (result.time_s >= lo) & (result.time_s < hi)
        for index, reference_name in enumerate(reference_names):
            candidate = fit_h1(
                result.time_s[pmask],
                result.pressure_probes_Pa[pmask, index],
                f0,
            )
            truth = reference[reference_name]
            rows.append(
                {
                    "variant": variant,
                    "target_one_way_amplitude": target,
                    "outer_abc": radiation,
                    "signal": reference_name,
                    "status": "completed",
                    "message": "",
                    "comsol_H1_peak_Pa": abs(truth),
                    "python_H1_peak_Pa": abs(candidate),
                    "amplitude_relative_error": abs(abs(candidate) - abs(truth))
                    / abs(truth),
                    "phase_python_minus_comsol_deg": float(
                        np.degrees(np.angle(candidate / truth))
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False, float_format="%.12e")
    print(
        json.dumps(
            frame.to_dict(orient="records"), ensure_ascii=False, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
