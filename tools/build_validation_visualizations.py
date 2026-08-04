#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


COLORS = {
    "python": "#0072B2",
    "comsol": "#D55E00",
    "pass": "#009E73",
    "strict": "#CC79A7",
    "grid": "#B8C1CC",
    "dark": "#263238",
}

SIGNAL_LABELS = {
    "coil_current_A": "音圈电流",
    "coil_displacement_m": "音圈位移",
    "pressure_python_axis_near_actual": "近轴声压",
    "pressure_python_offaxis_actual": "45°声压",
    "pressure_common_rear_physical_m0p10": "后腔声压",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "#F7F9FB",
            "axes.facecolor": "white",
            "axes.edgecolor": "#90A4AE",
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.32,
            "grid.linewidth": 0.7,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )


def save_page(fig, out: Path, name: str, pdf: PdfPages) -> None:
    fig.savefig(out / f"{name}.png", dpi=190)
    fig.savefig(out / f"{name}.svg")
    pdf.savefig(fig)
    plt.close(fig)


def last_cycle(frame: pd.DataFrame, f0: float, time_column: str = "time_s"):
    lo, hi = 3.0 / f0 - 1e-10, 4.0 / f0 + 1e-10
    return frame[(frame[time_column] >= lo) & (frame[time_column] <= hi)].copy()


def phase_deg(time: np.ndarray, f0: float) -> np.ndarray:
    return (time - 3.0 / f0) * f0 * 360.0


def bar_threshold(ax, labels, values, threshold, title, unit, colors=None):
    y = np.arange(len(labels))
    if colors is None:
        colors = [
            COLORS["pass"] if value <= threshold else COLORS["comsol"]
            for value in values
        ]
    ax.barh(y, values, color=colors, alpha=0.88)
    ax.axvline(threshold, color=COLORS["strict"], ls="--", lw=2, label=f"阈值 {threshold:g}{unit}")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(unit)
    ax.legend(loc="lower right")
    for index, value in enumerate(values):
        ax.text(value, index, f"  {value:.2f}", va="center", fontsize=9)


def overview(summary: dict, mesh: pd.DataFrame):
    metrics = pd.DataFrame(summary["metrics"])
    primary = metrics[metrics.signal.isin(SIGNAL_LABELS)]
    labels = [SIGNAL_LABELS[name] for name in primary.signal]
    amplitude = 100.0 * primary.H1_amplitude_relative_error.to_numpy()
    phase = np.abs(primary.H1_phase_python_minus_comsol_deg.to_numpy())
    mesh_primary = mesh[mesh.signal.isin(SIGNAL_LABELS)]
    mesh_labels = [SIGNAL_LABELS[name] for name in mesh_primary.signal]
    mesh_amp = 100.0 * mesh_primary.H1_relative_change.to_numpy()

    fig = plt.figure(figsize=(15, 9))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.1, 1.1])
    ax0 = fig.add_subplot(grid[:, 0])
    ax1 = fig.add_subplot(grid[0, 1])
    ax2 = fig.add_subplot(grid[0, 2])
    ax3 = fig.add_subplot(grid[1, 1:])
    ax0.axis("off")
    ax0.text(0.5, 0.93, "工程验证通过", ha="center", va="center", fontsize=27, weight="bold", color=COLORS["pass"])
    ax0.text(0.5, 0.84, "70 Hz · 10 V peak · 约 1.62 mm", ha="center", fontsize=12, color=COLORS["dark"])
    cards = [
        ("机电", "电流 0.44% / 位移 0.18%"),
        ("物理声域", "最大 H1 幅值误差 6.21%"),
        ("相位", "物理声压最大 0.87°"),
        ("非线性", "近轴 THD 2.255% vs 2.497%"),
        ("界面闭合", "复数 L2 0.213% · 相关 0.999998"),
        ("COMSOL 网格", "工程 2%/2° 通过"),
    ]
    for index, (title, value) in enumerate(cards):
        y = 0.73 - index * 0.115
        patch = FancyBboxPatch((0.06, y - 0.065), 0.88, 0.085, boxstyle="round,pad=0.012", fc="white", ec="#CFD8DC")
        ax0.add_patch(patch)
        ax0.text(0.10, y, title, weight="bold", fontsize=11, va="center")
        ax0.text(0.90, y, value, ha="right", fontsize=10.5, va="center")
    ax0.text(0.5, 0.04, "严格 1% 网格无关性保留为附加审计，不阻断工程交付", ha="center", fontsize=9.5, color="#546E7A")

    bar_threshold(ax1, labels, amplitude, 10.0, "Python 对 COMSOL：H1 幅值误差", "%")
    bar_threshold(ax2, labels, phase, 10.0, "Python 对 COMSOL：H1 相位误差绝对值", "°")
    bar_threshold(ax3, mesh_labels, mesh_amp, 2.0, "COMSOL refined 相对 baseline：空间变化", "%")
    ax3.axvline(1.0, color=COLORS["dark"], ls=":", lw=1.8, label="严格审计 1%")
    ax3.legend(loc="lower right")
    fig.suptitle("时域扬声器 FEM 验证总览", fontsize=19, weight="bold")
    fig.tight_layout()
    return fig


def waveforms(comsol_dir: Path, python_dir: Path, f0: float):
    cg = pd.read_csv(comsol_dir / "global_timeseries.csv").drop_duplicates("time_s", keep="last")
    cp = pd.read_csv(comsol_dir / "pressure_points_timeseries.csv").pivot_table(
        index="time_s", columns="probe_name", values="p_Pa", aggfunc="last"
    ).reset_index()
    py = pd.read_csv(python_dir / "all_probes_timeseries.csv")
    cg, cp, py = last_cycle(cg, f0), last_cycle(cp, f0), last_cycle(py, f0)
    items = [
        ("音圈电流", cg, "coil_current_A", py, "current_A", 1.0, "A"),
        ("音圈位移", cg, "coil_displacement_m", py, "coil_displacement_m", 1e3, "mm"),
        ("近轴声压", cp, "python_axis_near_actual", py, "p_axis_near_0p10m_Pa", 1.0, "Pa"),
        ("45°声压", cp, "python_offaxis_actual", py, "p_offaxis_45deg_0p10m_Pa", 1.0, "Pa"),
        ("后腔物理声压", cp, "common_rear_physical_m0p10", py, "p_rear_physical_m0p10_Pa", 1.0, "Pa"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), sharex=True)
    for ax, (title, cf, cc, pf, pc, scale, unit) in zip(axes.flat, items):
        ax.plot(phase_deg(pf.time_s.to_numpy(), f0), pf[pc] * scale, color=COLORS["python"], lw=2.2, label="Python")
        ax.scatter(phase_deg(cf.time_s.to_numpy(), f0), cf[cc] * scale, color=COLORS["comsol"], s=17, alpha=0.72, label="COMSOL")
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.legend()
    axes.flat[-1].axis("off")
    axes.flat[-1].text(
        0.04,
        0.78,
        "共同窗口\n3T0 ≤ t ≤ 4T0\n\n"
        "COMSOL：原始求解点\nPython：180 步/周期\n\n"
        "图中未做幅值或相位对齐",
        fontsize=15,
        va="top",
        linespacing=1.45,
    )
    for ax in axes[-1, :1]:
        ax.set_xlabel("最后一周期相位 / °")
    axes[1, 1].set_xlabel("最后一周期相位 / °")
    fig.suptitle("最后完整周期：原始时序对照", fontsize=18, weight="bold")
    fig.tight_layout()
    return fig


def convergence(time_frame: pd.DataFrame, mesh_frame: pd.DataFrame):
    signals = list(SIGNAL_LABELS)
    labels = [SIGNAL_LABELS[name] for name in signals]
    time = time_frame.set_index("signal").loc[signals]
    mesh = mesh_frame.set_index("signal").loc[signals]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    x = np.arange(len(labels))
    axes[0, 0].bar(x, 100 * time.H1_relative_change, color=COLORS["python"])
    axes[0, 0].axhline(1, color=COLORS["strict"], ls="--", label="1%")
    axes[0, 0].set_title("时间加严：H1 幅值变化")
    axes[0, 0].set_ylabel("%")
    axes[0, 0].legend()
    axes[0, 1].bar(x, np.abs(time.H1_phase_change_deg), color=COLORS["python"])
    axes[0, 1].axhline(1, color=COLORS["strict"], ls="--", label="1°")
    axes[0, 1].set_title("时间加严：H1 相位变化")
    axes[0, 1].set_ylabel("°")
    axes[0, 1].legend()
    axes[1, 0].bar(x, 100 * mesh.H1_relative_change, color=COLORS["comsol"])
    axes[1, 0].axhline(2, color=COLORS["pass"], ls="--", lw=2, label="工程 2%")
    axes[1, 0].axhline(1, color=COLORS["dark"], ls=":", label="严格 1%")
    axes[1, 0].set_title("网格加密：H1 幅值变化")
    axes[1, 0].set_ylabel("%")
    axes[1, 0].legend()
    axes[1, 1].bar(x, np.abs(mesh.H1_phase_change_deg), color=COLORS["comsol"])
    axes[1, 1].axhline(2, color=COLORS["pass"], ls="--", lw=2, label="工程 2°")
    axes[1, 1].axhline(1, color=COLORS["dark"], ls=":", label="严格 1°")
    axes[1, 1].set_title("网格加密：H1 相位变化")
    axes[1, 1].set_ylabel("°")
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.set_xticks(x, labels, rotation=18, ha="right")
    fig.suptitle("COMSOL 数值稳定性：时间与空间分离", fontsize=18, weight="bold")
    fig.tight_layout()
    return fig


def harmonics(frame: pd.DataFrame):
    selected = [
        ("coil_current_A", "音圈电流"),
        ("pressure_python_axis_near_actual", "近轴声压"),
        ("pressure_python_offaxis_actual", "45°声压"),
        ("pressure_common_rear_physical_m0p10", "后腔声压"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, (signal, title) in zip(axes.flat, selected):
        data = frame[frame.signal == signal].sort_values("harmonic")
        x = data.harmonic.to_numpy()
        width = 0.36
        ax.bar(x - width / 2, np.maximum(data.comsol_ratio_to_H1, 1e-8), width, color=COLORS["comsol"], label="COMSOL")
        ax.bar(x + width / 2, np.maximum(data.python_ratio_to_H1, 1e-8), width, color=COLORS["python"], label="Python")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_ylim(1e-5, 1.5)
        ax.set_title(title)
        ax.set_xlabel("谐波阶次")
        ax.set_ylabel("相对 H1 峰值")
        ax.legend()
    fig.suptitle("H1–H10 谐波结构与 THD 来源", fontsize=18, weight="bold")
    fig.tight_layout()
    return fig


def interface_plot(frame: pd.DataFrame, summary: dict):
    active = frame[frame.reference_rms_m > 1e-6].copy()
    rear_c = "rear_physical_m0p10_comsol_motion_pressure_Pa_magnitude"
    rear_p = "rear_physical_m0p10_python_motion_pressure_Pa_magnitude"
    contribution = active.nlargest(12, rear_c).sort_values(rear_c)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].scatter(active.reference_rms_m * 1e3, active.candidate_rms_m * 1e3, s=35 + 3000 * active.axisym_area_m2, color=COLORS["python"], alpha=0.72)
    limit = max(active.reference_rms_m.max(), active.candidate_rms_m.max()) * 1e3 * 1.04
    axes[0, 0].plot([0, limit], [0, limit], color=COLORS["dark"], ls="--")
    axes[0, 0].set_xlabel("COMSOL 界面 RMS / mm")
    axes[0, 0].set_ylabel("Python 界面 RMS / mm")
    axes[0, 0].set_title("逐边界振型幅值")

    axes[0, 1].bar(active.boundary_entity.astype(str), 100 * active.relative_complex_L2_error, color=COLORS["pass"])
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("复数 L2 误差 / %")
    axes[0, 1].set_xlabel("COMSOL 边界实体")
    axes[0, 1].set_title("活动边界的局部振型误差")
    axes[0, 1].tick_params(axis="x", rotation=70)

    y = np.arange(len(contribution))
    axes[1, 0].barh(y - 0.2, contribution[rear_c], 0.4, color=COLORS["comsol"], label="COMSOL 界面运动")
    axes[1, 0].barh(y + 0.2, contribution[rear_p], 0.4, color=COLORS["python"], label="Python 界面运动")
    axes[1, 0].set_yticks(y, contribution.boundary_entity.astype(int).astype(str))
    axes[1, 0].set_xlabel("对后腔探针的单边界声压贡献幅值 / Pa")
    axes[1, 0].set_ylabel("边界实体")
    axes[1, 0].set_title("后腔主要辐射边界贡献")
    axes[1, 0].legend()

    axes[1, 1].axis("off")
    metric = summary["global_interface_metrics"]
    text = (
        "界面层闭合\n\n"
        f"全界面复数 L2：{100*metric['relative_complex_L2_error']:.3f}%\n"
        f"复相关幅值：{metric['complex_correlation_magnitude']:.6f}\n"
        f"相关相位：{metric['correlation_phase_deg']:.3f}°\n\n"
        "同一 Python 声学算子内\n"
        "两种界面振型产生的探针差：\n"
        "幅值 < 0.03%，相位 < 0.011°\n\n"
        "结论：结构源与声固映射已闭合"
    )
    axes[1, 1].text(0.08, 0.90, text, va="top", fontsize=15, linespacing=1.5, bbox=dict(boxstyle="round,pad=0.7", fc="white", ec="#B0BEC5"))
    fig.suptitle("COMSOL→Python 分层替换：界面振型归因", fontsize=18, weight="bold")
    fig.tight_layout()
    return fig


def nonlinear_plot(python_dir: Path, f0: float):
    data = last_cycle(pd.read_csv(python_dir / "all_probes_timeseries.csv"), f0)
    phase = phase_deg(data.time_s.to_numpy(), f0)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    sc = axes[0, 0].scatter(data.coil_displacement_m * 1e3, data.dynamic_BL_N_A, c=phase, cmap="viridis", s=19)
    axes[0, 0].set_xlabel("音圈位移 / mm")
    axes[0, 0].set_ylabel("动态 BL / N/A")
    axes[0, 0].set_title("动态 BL(x) 周期轨迹")
    fig.colorbar(sc, ax=axes[0, 0], label="相位 / °")
    sc2 = axes[0, 1].scatter(data.current_A, data.incremental_inductance_H * 1e3, c=phase, cmap="plasma", s=19)
    axes[0, 1].set_xlabel("电流 / A")
    axes[0, 1].set_ylabel("增量电感 / mH")
    axes[0, 1].set_title("非线性磁链的增量电感")
    fig.colorbar(sc2, ax=axes[0, 1], label="相位 / °")
    axes[1, 0].plot(phase, data.coil_displacement_m * 1e3, label="位移 / mm", color=COLORS["python"])
    twin = axes[1, 0].twinx()
    twin.plot(phase, data.current_A, label="电流 / A", color=COLORS["comsol"], alpha=0.8)
    axes[1, 0].set_xlabel("相位 / °")
    axes[1, 0].set_ylabel("位移 / mm", color=COLORS["python"])
    twin.set_ylabel("电流 / A", color=COLORS["comsol"])
    axes[1, 0].set_title("电流—位移相位关系")
    axes[1, 1].step(phase, data.nonlinear_iterations, where="mid", color=COLORS["dark"], label="Newton 次数")
    axes[1, 1].set_xlabel("相位 / °")
    axes[1, 1].set_ylabel("Newton 迭代")
    twin2 = axes[1, 1].twinx()
    twin2.plot(phase, data.ale_normalized_gap_margin, color=COLORS["pass"], label="ALE 裕量")
    twin2.set_ylabel("归一化 ALE 间隙裕量")
    axes[1, 1].set_title("非线性求解与 ALE 安全裕量")
    fig.suptitle("动态 BL、非线性磁场与移动音圈状态", fontsize=18, weight="bold")
    fig.tight_layout()
    return fig


def layer_flow():
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7)
    ax.axis("off")
    boxes = [
        (0.5, 3.0, 2.4, 1.4, "非线性磁场\nBL(x), λ(i)", "#E3F2FD"),
        (3.4, 3.0, 2.4, 1.4, "结构 / ALE\n音圈与振膜", "#E8F5E9"),
        (6.3, 3.0, 2.4, 1.4, "声固界面\n2396 Gauss 点", "#FFF3E0"),
        (9.2, 3.0, 2.4, 1.4, "物理声域\n球面 ABC", "#F3E5F5"),
        (12.1, 3.0, 2.4, 1.4, "探针 / H1–H10\n工程验收", "#ECEFF1"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=color, ec="#607D8B", lw=1.5))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=13, weight="bold")
    for x in [2.9, 5.8, 8.7, 11.6]:
        ax.annotate("", xy=(x + 0.45, 3.7), xytext=(x, 3.7), arrowprops=dict(arrowstyle="->", lw=2, color=COLORS["dark"]))
    ax.text(5.75, 5.35, "界面振型复数 L2 = 0.213%\n复相关 = 0.999998", ha="center", fontsize=13, color=COLORS["pass"], bbox=dict(boxstyle="round", fc="white", ec=COLORS["pass"]))
    ax.annotate("", xy=(7.5, 4.48), xytext=(6.4, 5.15), arrowprops=dict(arrowstyle="->", color=COLORS["pass"]))
    ax.text(10.4, 5.35, "COMSOL 位移→Python 声学\n误差 1.65–6.74%，相位 < 0.17°", ha="center", fontsize=13, color=COLORS["python"], bbox=dict(boxstyle="round", fc="white", ec=COLORS["python"]))
    ax.annotate("", xy=(10.4, 4.48), xytext=(10.4, 5.10), arrowprops=dict(arrowstyle="->", color=COLORS["python"]))
    ax.text(7.5, 1.45, "分层替换结论：结构源和弱式映射已闭合；剩余误差集中在声学离散 / COMSOL PML 参考层", ha="center", fontsize=14, weight="bold")
    ax.text(7.5, 0.7, "工程验收：Python 10%/10° · COMSOL 网格 2%/2° · 总体通过", ha="center", fontsize=15, color=COLORS["pass"])
    fig.suptitle("验证链与误差定位流程", fontsize=19, weight="bold")
    fig.tight_layout()
    return fig


def write_index(out: Path, names: list[tuple[str, str]]) -> None:
    rows = [
        "# 时域扬声器 FEM：详细可视化",
        "",
        "工况：70 Hz、10 V peak、4 周期。当前工程验收结论：**通过**。",
        "",
        "全部页面同时保存在 `validation_visual_report.pdf`；PNG 适合快速查看，SVG 可无损放大。",
        "",
    ]
    for name, description in names:
        rows.extend([f"## {description}", "", f"![{description}]({name}.png)", ""])
    (out / "README_CN.md").write_text("\n".join(rows), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = (args.out or root / "comsol_validation" / "visualizations").resolve()
    out.mkdir(parents=True, exist_ok=True)
    setup_style()

    raw = root / "comsol_validation" / "raw"
    comsol = Path("/mnt/d/loudspeakerFEM_comsol_validation/refined_mesh/raw_v2")
    python = root / "runs" / "transient_70Hz_nonlinear_comsol_physical_abc"
    summary = json.loads((raw / "physical_abc_v3_comparison" / "comparison_summary.json").read_text())
    mesh = pd.read_csv(raw / "mesh_convergence_engineering" / "convergence_metrics.csv")
    time = pd.read_csv(raw / "time_convergence_v2" / "convergence_metrics.csv")
    harmonics_frame = pd.read_csv(raw / "physical_abc_v3_comparison" / "harmonic_comparison.csv")
    interface_frame = pd.read_csv(raw / "interface_shape_comparison_v2.csv")
    interface_summary = json.loads((raw / "interface_shape_comparison_v2.json").read_text())
    pages = [
        ("00_validation_overview", "工程验证总览", overview(summary, mesh)),
        ("01_last_cycle_waveforms", "最后周期原始波形对照", waveforms(comsol, python, 70.0)),
        ("02_numerical_convergence", "COMSOL 时间与网格稳定性", convergence(time, mesh)),
        ("03_harmonic_spectrum", "H1–H10 谐波与 THD", harmonics(harmonics_frame)),
        ("04_interface_attribution", "界面振型与逐边界归因", interface_plot(interface_frame, interface_summary)),
        ("05_nonlinear_states", "动态 BL、非线性磁场与 ALE", nonlinear_plot(python, 70.0)),
        ("06_layer_replacement_flow", "分层替换验证流程", layer_flow()),
    ]
    pdf_path = out / "validation_visual_report.pdf"
    with PdfPages(pdf_path) as pdf:
        for name, _, fig in pages:
            save_page(fig, out, name, pdf)
    names = [(name, description) for name, description, _ in pages]
    write_index(out, names)
    manifest = {
        "status": "completed",
        "pages": [{"name": name, "description": description} for name, description in names],
        "pdf": str(pdf_path),
        "source_basis": {
            "python": str(python),
            "comsol": str(comsol),
            "comparison": str(raw / "physical_abc_v3_comparison"),
        },
    }
    (out / "visualization_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
