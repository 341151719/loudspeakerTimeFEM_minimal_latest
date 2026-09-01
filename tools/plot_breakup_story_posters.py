#!/usr/bin/env python3
"""Create presentation-grade breakup story posters and separate stage panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import meshio
import numpy as np


FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if FONT.exists():
    font_manager.fontManager.addfont(FONT)
    mpl.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
mpl.rcParams.update({"axes.unicode_minus": False, "svg.fonttype": "none"})

BG = "#071018"
PANEL_BG = "#0a1621"
INK = "#f4f1e8"
MUTED = "#91a2b1"
GOLD = "#ffd166"
PHASE_CMAP = LinearSegmentedColormap.from_list(
    "cinematic_phase",
    ["#00b8ff", "#07568a", "#09131d", "#d87320", "#ff4d3d"],
    N=512,
)
PHASE_CMAP.set_bad(BG)

CASES = {
    "soft_damped": {
        "title": "柔软 · 高阻尼",
        "params": "有效刚度 0.75×   /   损耗 2.5×",
        "tagline": "柔顺扩散｜更早进入多环分割",
    },
    "baseline": {
        "title": "生产基准",
        "params": "有效刚度 1.0×   /   损耗 1.0×",
        "tagline": "平衡参考｜分割由单环逐级展开",
    },
    "stiff_ringing": {
        "title": "高刚度 · 低阻尼",
        "params": "有效刚度 2.5×   /   损耗 0.075×",
        "tagline": "刚性延迟｜分割更晚但边界更锐利",
    },
}

STAGES = [
    ("piston", "01", "整体活塞", "PISTON", "整片振膜同相推进"),
    ("onset", "02", "初始分割", "ONSET", "第一条相位翻转边界出现"),
    ("developed", "03", "充分分割", "BREAKUP", "多重反相环带全面展开"),
]


def _profile(vtu: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = meshio.read(vtu)
    blocks = []
    for index, block in enumerate(mesh.cells):
        if block.type == "triangle6":
            tags = np.asarray(mesh.cell_data["domain_id"][index], dtype=int)
            blocks.append(np.asarray(block.data[tags == 21], dtype=int))
    triangles = np.vstack([block for block in blocks if len(block)])
    nodes = np.unique(triangles)
    radius = np.asarray(mesh.points[nodes, 0], dtype=float)
    uz = (
        np.asarray(mesh.point_data["u_z_m_real"], dtype=float)[nodes]
        + 1j * np.asarray(mesh.point_data["u_z_m_imag"], dtype=float)[nodes]
    )
    inner = radius <= np.quantile(radius, 0.12)
    phase = np.angle(np.mean(uz[inner]))
    signed = np.real(uz * np.exp(-1j * phase))
    rounded = np.round(radius, 7)
    unique_r = np.unique(rounded)
    values = np.array([np.mean(signed[rounded == value]) for value in unique_r])
    order = np.argsort(unique_r)
    unique_r, values = unique_r[order], values[order]
    return unique_r, values / max(float(np.max(np.abs(values))), 1e-30)


def _node_radii(radius: np.ndarray, values: np.ndarray) -> list[float]:
    nodes = []
    for index in np.flatnonzero(values[1:] * values[:-1] < 0):
        a, b = values[index], values[index + 1]
        fraction = abs(a) / max(abs(a) + abs(b), 1e-30)
        nodes.append(float(radius[index] + fraction * (radius[index + 1] - radius[index])))
    return nodes


def _catalog(case_root: Path) -> list[dict]:
    records = []
    for directory in sorted(case_root.joinpath("checkpoints").iterdir(), key=lambda p: float(p.name[:-2])):
        freq = float(directory.name[:-2])
        vtu = directory / f"solid_{freq:g}Hz.vtu"
        radius, values = _profile(vtu)
        records.append({
            "frequency_Hz": freq,
            "radius_m": radius,
            "values": values,
            "nodes": _node_radii(radius, values),
            "source": str(vtu),
        })
    return records


def _select_stages(records: list[dict]) -> dict[str, dict]:
    onset_index = next((index for index, item in enumerate(records) if len(item["nodes"]) >= 1), len(records) - 1)
    piston_index = max(0, onset_index - 1)
    developed_index = max(range(onset_index, len(records)), key=lambda index: len(records[index]["nodes"]))
    return {
        "piston": records[piston_index],
        "onset": records[onset_index],
        "developed": records[developed_index],
    }


def _disc_field(radius: np.ndarray, values: np.ndarray):
    rmin, rmax = float(radius.min()), float(radius.max())
    grid = np.linspace(-rmax, rmax, 701)
    xx, yy = np.meshgrid(grid, grid)
    rr = np.sqrt(xx**2 + yy**2)
    field = np.interp(rr, radius, values, left=np.nan, right=np.nan)
    field[(rr < rmin) | (rr > rmax)] = np.nan
    return xx, yy, field, rmin, rmax


def _draw_disc(ax, record: dict, *, callout: bool) -> None:
    radius, values = record["radius_m"], record["values"]
    xx, yy, field, rmin, rmax = _disc_field(radius, values)
    extent = 1000 * np.array([xx.min(), xx.max(), yy.min(), yy.max()])
    ax.imshow(field, origin="lower", extent=extent, cmap=PHASE_CMAP, vmin=-1, vmax=1, interpolation="bicubic")
    for node in record["nodes"]:
        for width, alpha in ((9, 0.055), (4, 0.16), (1.15, 0.95)):
            ax.add_patch(plt.Circle((0, 0), 1000 * node, fill=False, color=GOLD, lw=width, alpha=alpha))
    for edge in (rmin, rmax):
        ax.add_patch(plt.Circle((0, 0), 1000 * edge, fill=False, color="#cfe7f3", lw=0.8, alpha=0.45))
    if callout and np.min(values) < -0.08:
        index = int(np.argmin(values))
        target_r = 1000 * radius[index]
        angle = np.deg2rad(-34)
        target = (target_r * np.cos(angle), target_r * np.sin(angle))
        text = (0.62 * 1000 * rmax, -0.76 * 1000 * rmax)
        ax.annotate(
            "反相破裂区",
            xy=target,
            xytext=text,
            color=GOLD,
            fontsize=11,
            fontweight="bold",
            ha="center",
            arrowprops={"arrowstyle": "-|>", "color": GOLD, "lw": 1.2, "connectionstyle": "arc3,rad=-0.18"},
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#101c27", "edgecolor": GOLD, "alpha": 0.92},
        )
    ax.set_xlim(-1000 * rmax * 1.04, 1000 * rmax * 1.04)
    ax.set_ylim(-1000 * rmax * 1.04, 1000 * rmax * 1.04)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(PANEL_BG)


def _save_stage_panel(outdir: Path, case_key: str, case: dict, stage_info, record: dict) -> None:
    stage_key, number, title, english, description = stage_info
    fig = plt.figure(figsize=(10.8, 10.8), facecolor=BG)
    ax = fig.add_axes([0.07, 0.13, 0.86, 0.72], facecolor=PANEL_BG)
    _draw_disc(ax, record, callout=stage_key != "piston")
    fig.text(0.07, 0.94, f"{number}  {english}", color=GOLD, fontsize=13, fontweight="bold")
    fig.text(0.07, 0.885, title, color=INK, fontsize=28, fontweight="bold")
    fig.text(0.93, 0.90, f"{record['frequency_Hz']/1000:g} kHz", color=INK, fontsize=25, fontweight="bold", ha="right")
    fig.text(0.07, 0.065, case["title"], color=INK, fontsize=16, fontweight="bold")
    fig.text(0.93, 0.065, description, color=MUTED, fontsize=12, ha="right")
    stem = f"{case_key}_{stage_key}_{record['frequency_Hz']:g}Hz"
    fig.savefig(outdir / f"{stem}.png", dpi=180, facecolor=BG)
    fig.savefig(outdir / f"{stem}.svg", facecolor=BG)
    plt.close(fig)


def _save_case_poster(outdir: Path, case_key: str, case: dict, selected: dict[str, dict]) -> None:
    fig = plt.figure(figsize=(19.2, 10.8), facecolor=BG)
    fig.text(0.045, 0.925, case["title"], color=INK, fontsize=30, fontweight="bold")
    fig.text(0.045, 0.875, case["params"], color=GOLD, fontsize=14, fontweight="bold")
    fig.text(0.955, 0.925, "BREAKUP STORY", color="#294252", fontsize=17, fontweight="bold", ha="right")
    fig.text(0.955, 0.875, case["tagline"], color=MUTED, fontsize=13, ha="right")

    lefts = [0.035, 0.355, 0.675]
    for left, stage_info in zip(lefts, STAGES):
        stage_key, number, title, english, description = stage_info
        record = selected[stage_key]
        ax = fig.add_axes([left, 0.17, 0.29, 0.62], facecolor=PANEL_BG)
        _draw_disc(ax, record, callout=stage_key != "piston")
        fig.text(left, 0.815, f"{number}  {english}", color=GOLD, fontsize=10.5, fontweight="bold")
        fig.text(left, 0.775, title, color=INK, fontsize=18, fontweight="bold")
        fig.text(left + 0.29, 0.78, f"{record['frequency_Hz']/1000:g} kHz", color=INK, fontsize=16, fontweight="bold", ha="right")
        fig.text(left, 0.12, description, color=MUTED, fontsize=10.5)

    legend = fig.add_axes([0.36, 0.045, 0.28, 0.018])
    legend.imshow(np.linspace(-1, 1, 512)[None, :], aspect="auto", cmap=PHASE_CMAP, vmin=-1, vmax=1)
    legend.axis("off")
    fig.text(0.345, 0.047, "反相", color="#25bfff", fontsize=9, ha="right")
    fig.text(0.655, 0.047, "同相", color="#ff6558", fontsize=9)
    fig.text(0.5, 0.085, "金色边界标记相位翻转与分割区域", color=GOLD, fontsize=9.5, ha="center")
    fig.savefig(outdir / f"poster_{case_key}.png", dpi=200, facecolor=BG)
    fig.savefig(outdir / f"poster_{case_key}.svg", facecolor=BG)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    panels = args.outdir / "stage_panels"
    panels.mkdir(parents=True, exist_ok=True)
    manifest = {"design": "three separate 16:9 case posters plus nine separate square stage panels", "cases": {}}
    for case_key, case in CASES.items():
        records = _catalog(args.root / case_key)
        selected = _select_stages(records)
        _save_case_poster(args.outdir, case_key, case, selected)
        for stage_info in STAGES:
            _save_stage_panel(panels, case_key, case, stage_info, selected[stage_info[0]])
        manifest["cases"][case_key] = {
            "title": case["title"],
            "parameters": case["params"],
            "stages": {
                key: {"frequency_Hz": value["frequency_Hz"], "radial_nodes": len(value["nodes"]), "source": value["source"]}
                for key, value in selected.items()
            },
        }
    (args.outdir / "visualization_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
