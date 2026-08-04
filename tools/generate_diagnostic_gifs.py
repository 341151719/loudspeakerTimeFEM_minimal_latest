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
import numpy as np
import pandas as pd

from loudspeaker_time_fem.config import load_config
from loudspeaker_time_fem.model import build_transient_model


MAIN = Path(__file__).resolve().parents[1]
RUN = MAIN / "runs/transient_70Hz_nonlinear_comsol_physical_abc"
OUT = MAIN / "visualizations"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def style() -> None:
    from matplotlib import font_manager

    font_manager.fontManager.addfont(FONT)
    name = font_manager.FontProperties(fname=FONT).get_name()
    plt.rcParams.update(
        {
            "font.family": name,
            "axes.unicode_minus": False,
            "figure.facecolor": "#08111f",
            "axes.facecolor": "#0d192b",
            "axes.edgecolor": "#6f8199",
            "axes.labelcolor": "#e8eef7",
            "xtick.color": "#c9d4e4",
            "ytick.color": "#c9d4e4",
            "text.color": "#f4f7fb",
            "grid.color": "#52657d",
            "grid.alpha": 0.25,
        }
    )


def save(animation: FuncAnimation, path: Path, fps: int = 12) -> None:
    animation.save(path, writer=PillowWriter(fps=fps), dpi=105)


def periodic_four_snapshot(values: np.ndarray, phase: float) -> np.ndarray:
    coefficient = np.fft.fft(values, axis=0) / values.shape[0]
    orders = np.fft.fftfreq(values.shape[0], d=1.0 / values.shape[0])
    return np.real(
        np.sum(
            coefficient
            * np.exp(1j * orders[:, None] * phase).reshape(
                (values.shape[0],) + (1,) * (values.ndim - 1)
            ),
            axis=0,
        )
    )


def coupled_field_gif(model) -> dict:
    data = np.load(RUN / "field_snapshots.npz")
    pressure = data["pressure_Pa"]
    displacement = data["solid_displacement_m"]
    ap = data["acoustic_points_rz_m"]
    sp = data["solid_points_rz_m"]
    amap = model.acoustic.acoustic_node_map
    atri = np.asarray(
        [
            [amap[int(node)] for node in triangle]
            for triangle in model.acoustic.triangles_global
        ]
    )
    stri = np.asarray(model.solid.triangles6[:, :3], dtype=int)
    vmax = float(np.quantile(np.abs(pressure), 0.995))
    frames = 40
    fig, ax = plt.subplots(figsize=(8.4, 7.2))

    def draw(index: int):
        ax.clear()
        phase = 2.0 * np.pi * index / frames
        p = periodic_four_snapshot(pressure, phase)
        u = periodic_four_snapshot(displacement, phase)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        for sign in (-1.0, 1.0):
            tri = mtri.Triangulation(sign * ap[:, 0] * 1e3, ap[:, 1] * 1e3, atri)
            ax.tripcolor(tri, p, shading="gouraud", cmap="RdBu_r", norm=norm)
            deformed = sp + 12.0 * u
            solid_tri = mtri.Triangulation(
                sign * deformed[:, 0] * 1e3, deformed[:, 1] * 1e3, stri
            )
            ax.triplot(solid_tri, color="#ffd166", lw=0.42, alpha=0.72)
        ax.axvline(0, color="#d5dfed", lw=0.6, alpha=0.5)
        ax.set_aspect("equal")
        ax.set_xlim(-118, 118)
        ax.set_ylim(-118, 118)
        ax.set_xlabel("径向镜像坐标 / mm")
        ax.set_ylabel("轴向坐标 z / mm")
        ax.set_title(
            f"70 Hz 非线性声—固耦合响应   相位 {index * 360 / frames:5.1f}°\n"
            f"声压色标 ±{vmax:.1f} Pa；结构位移放大 12×"
        )
        ax.text(
            0.02,
            0.02,
            "红/蓝：瞬时声压    黄线：变形结构\n四相位全场快照作周期 Fourier 插值",
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            bbox=dict(boxstyle="round", facecolor="#08111f", alpha=0.78, edgecolor="#52657d"),
        )

    animation = FuncAnimation(fig, draw, frames=frames, interval=80)
    path = OUT / "01_nonlinear_coupled_field_70Hz.gif"
    save(animation, path)
    plt.close(fig)
    return {"file": path.name, "frames": frames, "pressure_scale_Pa": vmax}


def nonlinear_state_gif() -> dict:
    data = pd.read_csv(RUN / "all_probes_timeseries.csv")
    frequency = 70.0
    use = data.time_s >= data.time_s.max() - 1.0 / frequency - 1e-10
    d = data.loc[use].reset_index(drop=True)
    frames = 48
    indices = np.linspace(0, len(d) - 1, frames, dtype=int)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))

    def draw(frame: int):
        for ax in axes.ravel():
            ax.clear()
            ax.grid(True)
        j = indices[frame]
        phase = frame * 360.0 / frames
        x = d.coil_displacement_m.to_numpy() * 1e3
        current = d.current_A.to_numpy()
        bl = d.dynamic_BL_N_A.to_numpy()
        inductance = d.incremental_inductance_H.to_numpy() * 1e3
        time_ms = (d.time_s - d.time_s.iloc[0]).to_numpy() * 1e3
        axes[0, 0].plot(x, current, color="#67d5ff", lw=2)
        axes[0, 0].scatter(x[j], current[j], s=70, color="#ff5d73", zorder=5)
        axes[0, 0].set(xlabel="音圈位移 / mm", ylabel="电流 / A", title="机电状态回线")
        axes[0, 1].plot(x, bl, color="#ffd166", lw=2)
        axes[0, 1].scatter(x[j], bl[j], s=70, color="#ff5d73", zorder=5)
        axes[0, 1].set(xlabel="音圈位移 / mm", ylabel="动态 BL / N·A⁻¹", title="BL(x) 非线性")
        axes[1, 0].plot(current, inductance, color="#b892ff", lw=2)
        axes[1, 0].scatter(current[j], inductance[j], s=70, color="#ff5d73", zorder=5)
        axes[1, 0].set(xlabel="电流 / A", ylabel="增量电感 / mH", title="Linc(i) 磁非线性")
        axes[1, 1].plot(time_ms, x, label="位移 / mm", color="#67d5ff")
        axes[1, 1].plot(time_ms, current, label="电流 / A", color="#ffb45c")
        axes[1, 1].axvline(time_ms[j], color="#ff5d73", lw=2)
        axes[1, 1].set(xlabel="最后一周期时间 / ms", title="相位同步")
        axes[1, 1].legend(loc="upper right", fontsize=8)
        fig.suptitle(
            f"70 Hz 动态 BL—非线性磁场状态环   相位 {phase:5.1f}°",
            fontsize=15,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))

    animation = FuncAnimation(fig, draw, frames=frames, interval=75)
    path = OUT / "02_nonlinear_BL_magnetic_cycle.gif"
    save(animation, path)
    plt.close(fig)
    return {
        "file": path.name,
        "frames": frames,
        "BL_range_N_per_A": [float(d.dynamic_BL_N_A.min()), float(d.dynamic_BL_N_A.max())],
    }


def harmonic_build_gif() -> dict:
    data = pd.read_csv(RUN / "all_probes_timeseries.csv")
    frequency = 70.0
    use = data.time_s >= data.time_s.max() - 1.0 / frequency - 1e-10
    t = data.loc[use, "time_s"].to_numpy()
    y = data.loc[use, "p_axis_near_0p10m_Pa"].to_numpy()
    phase_data = 2.0 * np.pi * frequency * t
    design = [np.ones_like(t)]
    for order in range(1, 11):
        design.extend([np.sin(order * phase_data), np.cos(order * phase_data)])
    coefficient, *_ = np.linalg.lstsq(np.column_stack(design), y, rcond=None)
    harmonic = np.array(
        [coefficient[2 * n] - 1j * coefficient[2 * n - 1] for n in range(1, 11)]
    )
    phase = np.linspace(0, 2.0 * np.pi, 361)
    measured_phase = np.mod(phase_data, 2.0 * np.pi)
    frames = 10
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2))

    def draw(frame: int):
        for ax in axes:
            ax.clear()
            ax.grid(True)
        included = frame + 1
        reconstructed = np.full_like(phase, coefficient[0])
        for order in range(1, included + 1):
            reconstructed += np.real(harmonic[order - 1] * np.exp(1j * order * phase))
        axes[0].scatter(
            np.degrees(measured_phase),
            y,
            s=8,
            alpha=0.3,
            color="#a8bbd4",
            label="非线性时域样本",
        )
        axes[0].plot(np.degrees(phase), reconstructed, color="#ff6b7d", lw=2.2)
        axes[0].set(
            xlim=(0, 360),
            xlabel="基频相位 / °",
            ylabel="近轴声压 / Pa",
            title=f"重构波形：H1–H{included}",
        )
        ratio = np.abs(harmonic) / abs(harmonic[0]) * 100
        colors = ["#67d5ff" if n <= included else "#253a54" for n in range(1, 11)]
        axes[1].bar(np.arange(1, 11), ratio, color=colors)
        axes[1].set_yscale("log")
        axes[1].set(
            xticks=np.arange(1, 11),
            xlabel="谐波阶次",
            ylabel="相对 H1 / %",
            title="非线性谐波模态",
            ylim=(1e-4, 150),
        )
        axes[1].axvline(3, color="#ffd166", ls="--", lw=1, alpha=0.7)
        fig.suptitle(
            f"70 Hz 非线性响应的逐阶谐波构成   当前加入 H{included}",
            fontsize=15,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))

    animation = FuncAnimation(fig, draw, frames=frames, interval=100)
    path = OUT / "03_nonlinear_harmonic_modes_H1_H10.gif"
    save(animation, path, fps=2.5)
    plt.close(fig)
    thd = float(np.linalg.norm(harmonic[1:]) / abs(harmonic[0]))
    return {"file": path.name, "frames": frames, "near_axis_THD_percent": 100 * thd}


def anomaly_frequency_gif() -> dict:
    p1 = pd.read_csv(
        ROOT / "20_ANALYSIS/runs/stage34_directivity_15k/no_nra_p1_580_640/native_sweep_metrics.csv"
    )
    p2 = pd.read_csv(
        ROOT / "20_ANALYSIS/runs/stage34_directivity_15k/no_nra_p2_580_640/native_sweep_metrics.csv"
    )
    f = p1.freq_Hz.to_numpy()
    frames = len(f)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.2), sharex=True)
    peak_p1 = float(f[np.argmax(p1.domain8_mean_abs_pressure_Pa)])
    peak_p2 = float(f[np.argmax(p2.domain8_mean_abs_pressure_Pa)])

    def draw(index: int):
        for ax in axes:
            ax.clear()
            ax.grid(True)
            ax.axvline(605.0, color="#ff5d73", ls="--", lw=1.5, label="COMSOL 参考峰 605 Hz")
            ax.axvline(f[index], color="#f7f9fc", lw=1.5)
        axes[0].plot(f, p1.domain8_mean_abs_pressure_Pa, color="#67d5ff", label="P1 腔域 8")
        axes[0].plot(f, p2.domain8_mean_abs_pressure_Pa, color="#ffd166", label="P2 腔域 8")
        axes[0].scatter(
            [f[index], f[index]],
            [
                p1.domain8_mean_abs_pressure_Pa.iloc[index],
                p2.domain8_mean_abs_pressure_Pa.iloc[index],
            ],
            c=["#67d5ff", "#ffd166"],
            s=65,
        )
        axes[0].set_ylabel("腔内平均 |p| / Pa")
        axes[0].set_title(
            f"后腔尖锐共振：当前 {f[index]:.0f} Hz\n"
            f"P1 峰 {peak_p1:.0f} Hz；P2 峰 {peak_p2:.0f} Hz；COMSOL 约 605 Hz"
        )
        axes[0].legend(fontsize=8, loc="upper left")
        axes[1].plot(f, p1.axis_SPL_dB_RMS, color="#67d5ff", label="P1 轴向 SPL")
        axes[1].plot(f, p2.axis_SPL_dB_RMS, color="#ffd166", label="P2 轴向 SPL")
        axes[1].scatter(
            [f[index], f[index]],
            [p1.axis_SPL_dB_RMS.iloc[index], p2.axis_SPL_dB_RMS.iloc[index]],
            c=["#67d5ff", "#ffd166"],
            s=65,
        )
        axes[1].set(xlabel="频率 / Hz", ylabel="轴向 SPL / dB RMS", xlim=(580, 640))
        axes[1].legend(fontsize=8)
        axes[1].text(
            0.01,
            0.04,
            "该峰对声学离散敏感：应解释为 605–612 Hz 奇异频带，而非唯一精确点",
            transform=axes[1].transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="#08111f", alpha=0.82, edgecolor="#52657d"),
        )
        fig.tight_layout()

    animation = FuncAnimation(fig, draw, frames=frames, interval=80)
    path = OUT / "04_anomalous_cavity_resonance_580_640Hz.gif"
    save(animation, path)
    plt.close(fig)
    return {
        "file": path.name,
        "frames": frames,
        "P1_peak_Hz": peak_p1,
        "P2_peak_Hz": peak_p2,
        "COMSOL_reference_peak_Hz": 605.0,
    }


def eigenmode_gallery_gif() -> dict:
    data = np.load(ROOT / "20_ANALYSIS/runs/merged_stage33_eigen40/p2_eigenmodes.npz")
    points = data["points_rz_m"]
    triangles = data["triangles6"][:, :3]
    vectors = np.real(data["vectors_full"]).reshape(len(points), 2, -1)
    frequencies = data["frequencies_Hz"]
    selected = [0, 1, 2, 3]
    phase_frames = 14
    frames = len(selected) * phase_frames
    fig, ax = plt.subplots(figsize=(8.4, 6.2))

    def draw(frame: int):
        ax.clear()
        mode_position = frame // phase_frames
        mode = selected[mode_position]
        phase = 2.0 * np.pi * (frame % phase_frames) / phase_frames
        vector = vectors[:, :, mode]
        scale = 0.006 / max(np.linalg.norm(vector, axis=1).max(), 1e-30)
        deformed = points + scale * np.cos(phase) * vector
        color = vector[:, 1] / max(np.abs(vector[:, 1]).max(), 1e-30) * np.cos(phase)
        tri = mtri.Triangulation(deformed[:, 0] * 1e3, deformed[:, 1] * 1e3, triangles)
        ax.tripcolor(
            tri,
            color,
            shading="gouraud",
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1),
        )
        ax.triplot(tri, color="#dfe8f5", lw=0.3, alpha=0.45)
        ax.set_aspect("equal")
        ax.set(xlabel="r / mm", ylabel="z / mm")
        ax.set_title(
            f"结构特征模态 {mode + 1}：{frequencies[mode]:.2f} Hz\n"
            f"归一化振型，相位 {360 * (frame % phase_frames) / phase_frames:5.1f}°"
        )
        ax.text(
            0.02,
            0.03,
            "颜色：归一化轴向位移\n形变显示幅度统一为约 6 mm，不代表真实响应幅值",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="#08111f", alpha=0.82, edgecolor="#52657d"),
        )
        ax.set_xlim(-2, 88)
        ax.set_ylim(-70, 15)

    animation = FuncAnimation(fig, draw, frames=frames, interval=90)
    path = OUT / "05_structural_eigenmode_gallery.gif"
    save(animation, path)
    plt.close(fig)
    return {
        "file": path.name,
        "frames": frames,
        "mode_frequencies_Hz": [float(frequencies[index]) for index in selected],
    }


def main() -> int:
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    config, path = load_config(
        MAIN / "configs/transient_70Hz_nonlinear_comsol_physical_abc.json"
    )
    model = build_transient_model(config, path)
    results = [
        coupled_field_gif(model),
        nonlinear_state_gif(),
        harmonic_build_gif(),
        anomaly_frequency_gif(),
        eigenmode_gallery_gif(),
    ]
    manifest = {
        "status": "completed",
        "source_policy": "existing computed FEM/COMSOL-derived data only; no synthetic physics fields",
        "animations": results,
    }
    (OUT / "GIF_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
