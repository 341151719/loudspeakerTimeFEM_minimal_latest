#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from loudspeaker_time_fem.native_acoustic import (
    STRUCTURE_DOMAINS,
    _cell_sides,
    build_native_acoustic,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--domains", default="2,3,5")
    args = parser.parse_args()
    domains = {int(value) for value in args.domains.split(",")}
    acoustic = build_native_acoustic(args.mesh, domains, set())
    interfaces = acoustic.mesh.acoustic_interface_edges(domains, STRUCTURE_DOMAINS)
    sides = _cell_sides(acoustic.mesh)
    xg, wg = np.polynomial.legendre.leggauss(4)
    rows = []
    sample = 0
    for edge_index, (ga, gb, adomain, sdomain, boundary) in enumerate(interfaces):
        p0 = acoustic.mesh.points_rz_m[ga]
        p1 = acoustic.mesh.points_rz_m[gb]
        tangent = p1 - p0
        length = float(np.linalg.norm(tangent))
        normal = np.array([tangent[1], -tangent[0]]) / length
        adjacent = sides[tuple(sorted((ga, gb)))]
        ac = next(c for d, c in adjacent if d == adomain)
        structure = next(c for d, c in adjacent if d == sdomain)
        if np.dot(normal, structure - ac) < 0:
            normal = -normal
        for xi, weight in zip(xg, wg):
            t = 0.5 * (xi + 1.0)
            point = (1.0 - t) * p0 + t * p1
            rows.append(
                {
                    "sample_id": sample,
                    "edge_index": edge_index,
                    "quadrature_t": t,
                    "r_m": point[0],
                    "z_m": point[1],
                    "normal_r": normal[0],
                    "normal_z": normal[1],
                    "axisym_weight_m2": 2.0
                    * np.pi
                    * max(float(point[0]), 1e-12)
                    * length
                    * 0.5
                    * weight,
                    "pressure_global_a": ga,
                    "pressure_global_b": gb,
                    "pressure_shape_a": 1.0 - t,
                    "pressure_shape_b": t,
                    "boundary_entity": boundary,
                    "acoustic_domain": adomain,
                    "structure_domain": sdomain,
                }
            )
            sample += 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False, float_format="%.17g")
    print(f"samples={len(rows)} edges={len(interfaces)} output={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
