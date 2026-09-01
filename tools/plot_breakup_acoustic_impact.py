#!/usr/bin/env python3
"""Render separate source/near-field/far-field breakup impact stories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import meshio
import numpy as np
import pandas as pd

from plot_breakup_story_posters import (
    BG,
    CASES,
    GOLD,
    INK,
    MUTED,
    PANEL_BG,
    PHASE_CMAP,
    STAGES,
    _catalog,
    _draw_disc,
    _select_stages,
)


STAGE_COPY = {
    "piston": {
        "headline": "同相面源建立连续波前",
        "near": "近场保持大尺度连续压力区",
        "far": "远场能量集中于稳定主瓣",
    },
    "onset": {
        "headline": "第一圈反相运动开始声学抵消",
        "near": "近场出现第一条干涉节点带",
        "far": "远场辐射开始重新分配",
    },
    "developed": {
        "headline": "多重反相环把辐射切成多个声源区",
        "near": "近场形成密集热点与零压区",
        "far": "主瓣、旁瓣与深零点重新组织",
    },
}


def _phase_reference(solid_vtu: Path) -> float:
    mesh = meshio.read(solid_vtu)
    tags = np.asarray(mesh.cell_data["domain_id"][0], dtype=int)
    cells = np.asarray(mesh.cells[0].data[tags == 21], dtype=int)
    nodes = np.unique(cells)
    radius = np.asarray(mesh.points[nodes, 0], dtype=float)
    uz = (
        np.asarray(mesh.point_data["u_z_m_real"], dtype=float)[nodes]
        + 1j * np.asarray(mesh.point_data["u_z_m_imag"], dtype=float)[nodes]
    )
    return float(np.angle(np.mean(uz[radius <= np.quantile(radius, 0.12)])))


def _acoustic_triangles(mesh: meshio.Mesh) -> np.ndarray:
    triangles = []
    for index, block in enumerate(mesh.cells):
        tags = np.asarray(mesh.cell_data["is_PML"][index], dtype=int)
        cells = np.asarray(block.data[tags == 0], dtype=int)
        if block.type == "triangle":
            triangles.extend(cells.tolist())
        elif block.type == "triangle6":
            for a, b, c, ab, bc, ca in cells:
                triangles.extend([[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]])
    return np.asarray(triangles, dtype=int)


def _draw_near_field(ax, acoustic_vtu: Path, solid_vtu: Path, phase: float) -> None:
    acoustic = meshio.read(acoustic_vtu)
    points = np.asarray(acoustic.points[:, :2], dtype=float)
    triangles = _acoustic_triangles(acoustic)
    pressure = (
        np.asarray(acoustic.point_data["p_Pa_peak_real"], dtype=float)
        + 1j * np.asarray(acoustic.point_data["p_Pa_peak_imag"], dtype=float)
    )
    signed = np.real(pressure * np.exp(-1j * phase))
    physical_nodes = np.unique(triangles)
    scale = float(np.quantile(np.abs(signed[physical_nodes]), 0.985))
    values = np.clip(signed / max(scale, 1e-30), -1.0, 1.0)
    levels = np.linspace(-1, 1, 101)
    for sign in (1.0, -1.0):
        triangulation = mtri.Triangulation(sign * points[:, 0] * 1000, points[:, 1] * 1000, triangles)
        ax.tricontourf(triangulation, values, levels=levels, cmap=PHASE_CMAP, vmin=-1, vmax=1, extend="both")

    solid = meshio.read(solid_vtu)
    sx, sz = solid.points[:, 0] * 1000, solid.points[:, 1] * 1000
    ax.scatter(sx, sz, s=0.45, color=GOLD, alpha=0.62, linewidths=0)
    ax.scatter(-sx, sz, s=0.45, color=GOLD, alpha=0.62, linewidths=0)
    ax.axvline(0, color="#7f98a8", lw=0.5, alpha=0.35)
    ax.set(xlim=(-170, 170), ylim=(-165, 170))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(PANEL_BG)


def _draw_far_field(ax, directivity_csv: Path) -> None:
    frame = pd.read_csv(directivity_csv)
    theta = np.deg2rad(frame["theta_deg"].to_numpy(float))
    relative = frame["relative_dB"].to_numpy(float)
    radius = np.clip((relative + 30.0) / 30.0, 0.0, 1.0)
    ax.set_facecolor(PANEL_BG)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_ylim(0, 1.03)
    for width, alpha in ((12, 0.04), (6, 0.10), (2.2, 0.95)):
        ax.plot(theta, radius, color="#35d6ff", lw=width, alpha=alpha)
    ax.fill(theta, radius, color="#0aa9dd", alpha=0.28)
    ax.fill_between(theta, 0, radius, color="#0d5f8b", alpha=0.16)
    ax.set_xticks(np.deg2rad([-90, -60, -30, 0, 30, 60, 90]))
    ax.set_xticklabels(["侧", "", "", "轴向", "", "", "侧"], color=MUTED, fontsize=9)
    ax.set_yticks([1 / 3, 2 / 3, 1])
    ax.set_yticklabels([])
    ax.grid(color="#7292a4", alpha=0.18, lw=0.7)
    ax.spines["polar"].set_color("#315064")
    ax.spines["polar"].set_linewidth(0.8)


def _save_story(outdir: Path, case_key: str, case: dict, stage_info, record: dict) -> dict:
    stage_key, number, stage_title, english, _ = stage_info
    freq = float(record["frequency_Hz"])
    source_dir = Path(record["source"]).parent
    solid_vtu = Path(record["source"])
    acoustic_vtu = source_dir / f"acoustic_{freq:g}Hz.vtu"
    directivity_csv = source_dir / f"directivity_{freq:g}Hz.csv"
    phase = _phase_reference(solid_vtu)

    fig = plt.figure(figsize=(19.2, 10.8), facecolor=BG)
    fig.text(0.045, 0.925, f"{case['title']}  ·  {stage_title}", color=INK, fontsize=28, fontweight="bold")
    fig.text(0.045, 0.875, f"{freq/1000:g} kHz", color=GOLD, fontsize=16, fontweight="bold")
    fig.text(0.14, 0.878, STAGE_COPY[stage_key]["headline"], color=MUTED, fontsize=13)
    fig.text(0.955, 0.925, f"{number}  {english}", color="#294252", fontsize=18, fontweight="bold", ha="right")
    fig.text(0.955, 0.875, case["params"], color=MUTED, fontsize=11.5, ha="right")

    source_ax = fig.add_axes([0.035, 0.18, 0.28, 0.61], facecolor=PANEL_BG)
    near_ax = fig.add_axes([0.345, 0.18, 0.35, 0.61], facecolor=PANEL_BG)
    far_ax = fig.add_axes([0.725, 0.20, 0.245, 0.58], projection="polar", facecolor=PANEL_BG)
    _draw_disc(source_ax, record, callout=stage_key != "piston")
    _draw_near_field(near_ax, acoustic_vtu, solid_vtu, phase)
    _draw_far_field(far_ax, directivity_csv)

    headings = [
        (0.035, "SOURCE", "振膜源面"),
        (0.345, "NEAR FIELD", "近场干涉"),
        (0.725, "FAR FIELD", "远场辐射"),
    ]
    for x, english_heading, chinese in headings:
        fig.text(x, 0.815, english_heading, color=GOLD, fontsize=10, fontweight="bold")
        fig.text(x, 0.786, chinese, color=INK, fontsize=15, fontweight="bold")
    fig.text(0.035, 0.115, "红橙 / 蓝青：相对音圈参考的同相 / 反相运动", color=MUTED, fontsize=9.5)
    fig.text(0.345, 0.115, STAGE_COPY[stage_key]["near"], color=MUTED, fontsize=10.5)
    fig.text(0.725, 0.115, STAGE_COPY[stage_key]["far"], color=MUTED, fontsize=10.5)
    fig.text(0.5, 0.05, "结构分割改变近场干涉，再重塑远场能量方向", color=GOLD, fontsize=11, ha="center")

    stem = f"{case_key}_{stage_key}_{freq:g}Hz_acoustic_impact"
    png = outdir / f"{stem}.png"
    svg = outdir / f"{stem}.svg"
    fig.savefig(png, dpi=200, facecolor=BG)
    fig.savefig(svg, facecolor=BG)
    plt.close(fig)
    return {
        "case": case_key,
        "stage": stage_key,
        "frequency_Hz": freq,
        "png": str(png),
        "svg": str(svg),
        "solid_vtu": str(solid_vtu),
        "acoustic_vtu": str(acoustic_vtu),
        "directivity_csv": str(directivity_csv),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for case_key, case in CASES.items():
        selected = _select_stages(_catalog(args.root / case_key))
        case_out = args.outdir / case_key
        case_out.mkdir(exist_ok=True)
        for stage_info in STAGES:
            outputs.append(_save_story(case_out, case_key, case, stage_info, selected[stage_info[0]]))
    manifest = {
        "scope": "separate breakup source, FEM near-field, and HK far-field stories; no trend curves",
        "normalization": {
            "source": "phase-aligned and independently normalized",
            "near_field": "instantaneous pressure phase-aligned to the coil; clipped at physical-node 98.5 percentile",
            "far_field": "relative magnitude with a -30 dB visual floor",
        },
        "outputs": outputs,
    }
    (args.outdir / "acoustic_impact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"stories": len(outputs), "outdir": str(args.outdir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
