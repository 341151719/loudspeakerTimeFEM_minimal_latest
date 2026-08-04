#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import TwoSlopeNorm
import meshio
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
OUT = PROJECT / "visualizations/fullfield_frequency"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
PHASE_FRAMES = 18

WORKSPACE_SOURCES = {
    100: WORKSPACE / "20_ANALYSIS/runs/validation6/checkpoints/100Hz/acoustic_100Hz.vtu",
    600: WORKSPACE / "20_ANALYSIS/runs/validation6/checkpoints/600Hz/acoustic_600Hz.vtu",
    630: WORKSPACE / "20_ANALYSIS/runs/validation6/checkpoints/630Hz/acoustic_630Hz.vtu",
    1000: WORKSPACE / "20_ANALYSIS/runs/validation6/checkpoints/1000Hz/acoustic_1000Hz.vtu",
    6300: WORKSPACE / "20_ANALYSIS/runs/validation6/checkpoints/6300Hz/acoustic_6300Hz.vtu",
    8000: WORKSPACE
    / "20_ANALYSIS/runs/stage34_directivity_15k/native_refined2_500Hz/checkpoints/8000Hz/acoustic_8000Hz.vtu",
    10000: WORKSPACE
    / "20_ANALYSIS/runs/stage34_directivity_15k/native_refined2_500Hz/checkpoints/10000Hz/acoustic_10000Hz.vtu",
    12000: WORKSPACE
    / "20_ANALYSIS/runs/stage34_directivity_15k/native_refined2_500Hz/checkpoints/12000Hz/acoustic_12000Hz.vtu",
    13500: WORKSPACE
    / "20_ANALYSIS/runs/stage34_directivity_15k/native_refined2_500Hz/checkpoints/13500Hz/acoustic_13500Hz.vtu",
    15000: WORKSPACE
    / "20_ANALYSIS/runs/stage34_directivity_15k/native_refined2_500Hz/checkpoints/15000Hz/acoustic_15000Hz.vtu",
}


def source_for(frequency: int) -> Path:
    bundled = PROJECT / f"inputs/reference_fields/frequency_fullfield/acoustic_{frequency}Hz.vtu"
    return bundled if bundled.is_file() else WORKSPACE_SOURCES[frequency]


def style() -> None:
    from matplotlib import font_manager

    font_manager.fontManager.addfont(FONT)
    name = font_manager.FontProperties(fname=FONT).get_name()
    plt.rcParams.update(
        {
            "font.family": name,
            "axes.unicode_minus": False,
            "figure.facecolor": "#07101d",
            "axes.facecolor": "#0b1728",
            "axes.edgecolor": "#71849d",
            "axes.labelcolor": "#eef3fa",
            "xtick.color": "#cbd6e5",
            "ytick.color": "#cbd6e5",
            "text.color": "#f7f9fc",
        }
    )


def physical_triangles(mesh: meshio.Mesh) -> np.ndarray:
    triangles: list[np.ndarray] = []
    for cell_block in mesh.cells:
        raw_cells = np.asarray(cell_block.data)
        corner_count = 3 if cell_block.type in {"triangle", "triangle6"} else 0
        if corner_count == 0:
            continue
        centroids = np.mean(mesh.points[raw_cells[:, :corner_count], :2], axis=1)
        # Some mixed P1/P2 exports reuse `is_PML` as an order/profile marker.
        # The production geometry has a spherical physical/PML interface at
        # R=165 mm, so geometry is the reliable cross-profile contract.
        keep = np.linalg.norm(centroids, axis=1) <= 0.165 + 1e-7
        cells = raw_cells[keep]
        if cell_block.type == "triangle":
            triangles.append(cells[:, :3])
        elif cell_block.type == "triangle6":
            # VTK/meshio order: v0,v1,v2,m01,m12,m20. Subdivide so the
            # exported P2 midside pressure participates in the visualization.
            triangles.extend(
                [
                    cells[:, [0, 3, 5]],
                    cells[:, [3, 1, 4]],
                    cells[:, [5, 4, 2]],
                    cells[:, [3, 4, 5]],
                ]
            )
    return np.vstack(triangles)


def load_field(frequency: int) -> dict:
    source = source_for(frequency)
    mesh = meshio.read(source)
    points = np.asarray(mesh.points[:, :2], float)
    pressure = np.asarray(mesh.point_data["p_Pa_peak_real"]) + 1j * np.asarray(
        mesh.point_data["p_Pa_peak_imag"]
    )
    triangles = physical_triangles(mesh)
    used = np.unique(triangles)
    scale = float(np.quantile(np.abs(pressure[used]), 0.995))
    return {
        "frequency_Hz": frequency,
        "source": str(source),
        "points": points,
        "triangles": triangles,
        "pressure": pressure,
        "scale_Pa": max(scale, 1e-12),
        "physical_cells": int(len(triangles)),
    }


def decorate(ax, frequency: int, phase_deg: float, scale: float) -> None:
    theta = np.linspace(0, np.pi, 240)
    ax.plot(165 * np.sin(theta), 165 * np.cos(theta), "--", color="#aab8ca", lw=0.65)
    ax.plot(-165 * np.sin(theta), 165 * np.cos(theta), "--", color="#aab8ca", lw=0.65)
    ax.axvline(0, color="#dbe4f0", lw=0.55, alpha=0.45)
    ax.set_aspect("equal")
    ax.set(xlim=(-168, 168), ylim=(-168, 168))
    ax.set_xlabel("镜像径向坐标 / mm")
    ax.set_ylabel("轴向坐标 z / mm")
    ax.set_title(
        f"{frequency:g} Hz 复声压全场   相位 {phase_deg:5.1f}°\n"
        f"物理声域；固定色标 ±{scale:.3g} Pa（99.5% 分位裁剪）"
    )


def draw_phase_field(ax, field: dict, phase: float) -> None:
    points = field["points"]
    triangles = field["triangles"]
    pressure = np.real(field["pressure"] * np.exp(1j * phase))
    scale = field["scale_Pa"]
    norm = TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale)
    for sign in (-1.0, 1.0):
        tri = mtri.Triangulation(
            sign * points[:, 0] * 1e3, points[:, 1] * 1e3, triangles
        )
        ax.tripcolor(tri, pressure, shading="gouraud", cmap="RdBu_r", norm=norm)


def single_frequency_gif(field: dict) -> dict:
    frequency = field["frequency_Hz"]
    fig, ax = plt.subplots(figsize=(7.6, 7.2))

    def draw(frame: int):
        ax.clear()
        phase = 2.0 * np.pi * frame / PHASE_FRAMES
        draw_phase_field(ax, field, phase)
        decorate(ax, frequency, np.degrees(phase), field["scale_Pa"])
        ax.text(
            0.02,
            0.025,
            "红/蓝：瞬时复声压实部\n虚线：R=165 mm 物理域/PML 接口",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="#07101d", alpha=0.8, edgecolor="#53677f"),
        )

    animation = FuncAnimation(fig, draw, frames=PHASE_FRAMES, interval=90)
    path = OUT / f"phase_fullfield_{frequency:05d}Hz.gif"
    animation.save(path, writer=PillowWriter(fps=11), dpi=100)
    plt.close(fig)
    return {
        "file": path.name,
        "frequency_Hz": frequency,
        "frames": PHASE_FRAMES,
        "scale_Pa": field["scale_Pa"],
        "physical_triangles": field["physical_cells"],
    }


def comparison_gif(fields: list[dict], filename: str, title: str) -> dict:
    count = len(fields)
    fig, axes = plt.subplots(1, count, figsize=(6.2 * count, 6.3), squeeze=False)
    axes = axes[0]

    def draw(frame: int):
        phase = 2.0 * np.pi * frame / PHASE_FRAMES
        for ax, field in zip(axes, fields):
            ax.clear()
            draw_phase_field(ax, field, phase)
            decorate(
                ax,
                field["frequency_Hz"],
                np.degrees(phase),
                field["scale_Pa"],
            )
        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=(0, 0, 1, 0.94))

    animation = FuncAnimation(fig, draw, frames=PHASE_FRAMES, interval=100)
    path = OUT / filename
    animation.save(path, writer=PillowWriter(fps=10), dpi=90)
    plt.close(fig)
    return {
        "file": path.name,
        "frequencies_Hz": [field["frequency_Hz"] for field in fields],
        "frames": PHASE_FRAMES,
    }


def magnitude_gallery_gif(fields: list[dict]) -> dict:
    fig, ax = plt.subplots(figsize=(7.8, 7.3))
    repeat = 5
    frames = len(fields) * repeat

    def draw(frame: int):
        ax.clear()
        field = fields[frame // repeat]
        points = field["points"]
        triangles = field["triangles"]
        magnitude = 20.0 * np.log10(
            np.maximum(np.abs(field["pressure"]) / field["scale_Pa"], 1e-3)
        )
        for sign in (-1.0, 1.0):
            tri = mtri.Triangulation(
                sign * points[:, 0] * 1e3, points[:, 1] * 1e3, triangles
            )
            ax.tripcolor(
                tri,
                magnitude,
                shading="gouraud",
                cmap="magma",
                vmin=-40,
                vmax=0,
            )
        theta = np.linspace(0, np.pi, 240)
        ax.plot(165 * np.sin(theta), 165 * np.cos(theta), "--", color="#c2cede", lw=0.65)
        ax.plot(-165 * np.sin(theta), 165 * np.cos(theta), "--", color="#c2cede", lw=0.65)
        ax.set_aspect("equal")
        ax.set(xlim=(-168, 168), ylim=(-168, 168))
        ax.set_xlabel("镜像径向坐标 / mm")
        ax.set_ylabel("轴向坐标 z / mm")
        ax.set_title(
            f"宽频声压模态画廊：{field['frequency_Hz']:g} Hz\n"
            "局部幅值 / 该频率 99.5% 分位，显示范围 -40…0 dB"
        )
        ax.text(
            0.02,
            0.025,
            "频率间分别归一化：用于比较空间形状，不比较绝对声压",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="#07101d", alpha=0.82, edgecolor="#53677f"),
        )

    animation = FuncAnimation(fig, draw, frames=frames, interval=180)
    path = OUT / "wideband_fullfield_mode_gallery_100Hz_15kHz.gif"
    animation.save(path, writer=PillowWriter(fps=6), dpi=100)
    plt.close(fig)
    return {
        "file": path.name,
        "frequencies_Hz": [field["frequency_Hz"] for field in fields],
        "frames": frames,
        "normalization": "per-frequency 99.5-percentile, -40 to 0 dB",
    }


def main() -> int:
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    fields = [load_field(frequency) for frequency in SOURCES]
    outputs = [single_frequency_gif(field) for field in fields]
    outputs.append(
        comparison_gif(
            [next(f for f in fields if f["frequency_Hz"] == value) for value in (600, 630)],
            "compare_cavity_mode_600_vs_630Hz.gif",
            "后腔奇异频带两侧：600 Hz 与 630 Hz 的相位全场",
        )
    )
    outputs.append(
        comparison_gif(
            [
                next(f for f in fields if f["frequency_Hz"] == value)
                for value in (8000, 12000, 15000)
            ],
            "compare_high_order_fields_8k_12k_15kHz.gif",
            "高阶声场与指向性演化：8 / 12 / 15 kHz",
        )
    )
    outputs.append(magnitude_gallery_gif(fields))
    manifest = {
        "status": "completed",
        "field_definition": "Re(p_peak * exp(i phase)); physical cells only; axisymmetric mirror display",
        "pml_policy": "PML cells excluded; dashed line marks R=165 mm physical/PML interface",
        "animations": outputs,
        "sources": {str(key): str(value) for key, value in SOURCES.items()},
    }
    (OUT / "FULLFIELD_GIF_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
