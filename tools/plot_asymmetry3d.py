#!/usr/bin/env python3
"""Run and visualize the reduced-order non-axisymmetric 3-D analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from loudspeaker_time_fem.asymmetry3d import analyze, export_analysis


BG = "#07121a"
PANEL = "#0b1b26"
INK = "#edf5f3"
MUTED = "#8ca5b3"
GOLD = "#ffc857"
FIELD = LinearSegmentedColormap.from_list(
    "acoustic_phase", ["#073a62", "#00b7e8", "#08131b", "#ff9e2c", "#ff4f42"], N=256
)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "axes.edgecolor": "#345365",
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "font.family": "DejaVu Sans",
        }
    )


def _phase_aligned(field: np.ndarray) -> np.ndarray:
    values = np.asarray(field, complex)
    reference = np.sum(values)
    phase = np.angle(reference) if abs(reference) else 0.0
    signed = np.real(values * np.exp(-1j * phase))
    scale = np.quantile(np.abs(signed), 0.99)
    return np.clip(signed / max(float(scale), 1e-30), -1.0, 1.0)


def _save_source_story(result: dict, out: Path) -> Path:
    source = result["source"]
    velocity = result["velocity_m_s_peak"].reshape(source.radial_points, source.azimuthal_points)
    transmission = result["basket_transmission"].reshape(source.radial_points, source.azimuthal_points)
    field = _phase_aligned(velocity)
    phi = source.azimuth_rad.reshape(source.radial_points, source.azimuthal_points)
    radius = source.radius_m.reshape(source.radial_points, source.azimuthal_points) * 1000
    phi = np.column_stack([phi, np.full(source.radial_points, 2.0 * np.pi)])
    radius = np.column_stack([radius, radius[:, :1]])
    field = np.column_stack([field, field[:, :1]])
    transmission = np.column_stack([transmission, transmission[:, :1]])

    fig = plt.figure(figsize=(19.2, 10.8), facecolor=BG)
    fig.text(0.05, 0.92, "NON-AXISYMMETRIC SOURCE", fontsize=28, fontweight="bold")
    fig.text(0.05, 0.87, "Rocking, circumferential breakup and basket shadowing", color=MUTED, fontsize=14)
    ax1 = fig.add_axes([0.05, 0.13, 0.42, 0.67], projection="polar")
    ax2 = fig.add_axes([0.53, 0.13, 0.42, 0.67], projection="polar")
    for ax in (ax1, ax2):
        ax.set_facecolor(PANEL)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.grid(color="#7595a8", alpha=0.16)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_ylim(float(np.min(radius)), float(np.max(radius)))
    ax1.contourf(phi, radius, field, levels=np.linspace(-1, 1, 81), cmap=FIELD, vmin=-1, vmax=1)
    ax2.contourf(phi, radius, transmission, levels=np.linspace(0, 1, 51), cmap="cividis", vmin=0, vmax=1)
    ax1.set_title("Diaphragm phase topology\nred/blue = opposite motion", pad=25, fontsize=16)
    ax2.set_title("Rear basket transmission\nspokes couple circumferential orders", pad=25, fontsize=16)
    metrics = result["metrics"]
    fig.text(
        0.5,
        0.055,
        f"dominant m={metrics['dominant_order']}   |   rocking tilt={metrics['rocking_tilt_magnitude_rad_peak']:.3e} rad peak   |   m>=2 fraction={metrics['higher_order_breakup_fraction']:.1%}",
        ha="center",
        color=GOLD,
        fontsize=12,
    )
    path = out / "01_source_rocking_breakup_basket.png"
    fig.savefig(path, dpi=200, facecolor=BG)
    plt.close(fig)
    return path


def _save_near_field(result: dict, out: Path, side: str) -> Path:
    front = side == "front"
    exterior_rear = side == "enclosure_rear"
    pressure = (
        result["front_pressure_Pa_peak"]
        if front
        else result["rear_exterior_pressure_Pa_peak"]
        if exterior_rear
        else result["rear_pressure_Pa_peak"]
    )
    signed = _phase_aligned(pressure)
    x = result["near_x_m"] * 1000
    y = result["near_y_m"] * 1000
    fig, ax = plt.subplots(figsize=(19.2, 10.8), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.contourf(x, y, signed, levels=np.linspace(-1, 1, 101), cmap=FIELD, vmin=-1, vmax=1)
    ax.contour(x, y, signed, levels=[0], colors=[GOLD], linewidths=0.7, alpha=0.65)
    ax.set_aspect("equal")
    ax.set_xlabel("x / mm")
    ax.set_ylabel("y / mm")
    ax.set_title(
        (
            "DIAPHRAGM FRONT NEAR FIELD"
            if front
            else "ENCLOSURE REAR EXTERIOR FIELD"
            if exterior_rear
            else "DIAPHRAGM REAR NEAR FIELD"
        )
        + "\nphase-aligned instantaneous pressure topology",
        loc="left",
        fontsize=24,
        pad=20,
    )
    note = (
        "front radiation: source asymmetry and circumferential breakup"
        if front
        else "finite enclosure reflection and edge diffraction behind the back panel"
        if exterior_rear
        else "rear radiation immediately behind the diaphragm: basket shadowing only"
    )
    fig.text(0.5, 0.04, note, ha="center", color=GOLD, fontsize=12)
    path = out / (
        "02_near_field_front.png"
        if front
        else "04_enclosure_rear_exterior.png"
        if exterior_rear
        else "03_near_field_rear.png"
    )
    fig.savefig(path, dpi=200, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_far_field(result: dict, out: Path) -> Path:
    theta = result["far_polar_rad"]
    phi = result["far_azimuth_rad"]
    db = np.clip(result["far_relative_dB"], -30.0, 0.0)
    radius = (db + 30.0) / 30.0
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    colours = plt.get_cmap("turbo")((db + 30.0) / 30.0)
    fig = plt.figure(figsize=(19.2, 10.8), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d", facecolor=BG)
    ax.plot_surface(x, y, z, facecolors=colours, rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=24, azim=-52)
    ax.set_title("FULL-SPHERE FAR FIELD\nnon-axisymmetric lobes, nulls and front/rear imbalance", loc="left", fontsize=24, pad=18)
    fig.text(0.5, 0.045, "surface radius: relative level with a -30 dB visual floor", ha="center", color=GOLD, fontsize=12)
    path = out / "05_far_field_3d.png"
    fig.savefig(path, dpi=200, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.outdir.mkdir(parents=True, exist_ok=True)
    _style()
    result = analyze(config)
    summary = export_analysis(result, args.outdir)
    figures = [
        _save_source_story(result, args.outdir),
        _save_near_field(result, args.outdir, "front"),
        _save_near_field(result, args.outdir, "rear"),
    ]
    if result["rear_exterior_pressure_Pa_peak"] is not None:
        figures.append(_save_near_field(result, args.outdir, "enclosure_rear"))
    figures.append(_save_far_field(result, args.outdir))
    summary["figures"] = [path.name for path in figures]
    (args.outdir / "asymmetry3d_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"outdir": str(args.outdir), "figures": len(figures)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
