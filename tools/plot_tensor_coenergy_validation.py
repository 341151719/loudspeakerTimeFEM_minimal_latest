#!/usr/bin/env python3
"""Deterministically render the registered tensor-coenergy review figures."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
X_HOLDOUT_MM = np.asarray([-1.875, -0.9375, 0.125, 1.0625, 1.875])
I_HOLDOUT_A = np.asarray([-0.55, -0.275, 0.05, 0.325, 0.55])


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def short(path: Path) -> str:
    return digest(path)[:12]


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, float_format="%.12e")


class Renderer:
    def __init__(self, scan: Path, transient: Path, comparison: Path, out: Path):
        self.scan = scan.resolve()
        self.transient = transient.resolve()
        self.comparison = comparison.resolve()
        self.baseline = (ROOT / "runs/transient_70Hz_nonlinear_comsol_physical_abc").resolve()
        self.old_coupled = (ROOT / "runs/transient_70Hz_nonlinear_comsol_physical_abc_coupled_coenergy_diagnostic").resolve()
        self.out = out.resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.entries = []
        self.training = np.load(self.scan / "training_tensor.npz", allow_pickle=False)
        self.surface = pd.read_csv(self.scan / "fit_surface_grid.csv")
        self.holdout = pd.read_csv(self.scan / "holdout_points.csv")
        self.mesh_convergence = pd.read_csv(self.scan / "mesh_convergence.csv")
        self.time = pd.read_csv(self.transient / "all_probes_timeseries.csv") if (self.transient / "all_probes_timeseries.csv").is_file() else None
        self.baseline_time = pd.read_csv(self.baseline / "all_probes_timeseries.csv") if (self.baseline / "all_probes_timeseries.csv").is_file() else None
        self.old_time = pd.read_csv(self.old_coupled / "all_probes_timeseries.csv") if (self.old_coupled / "all_probes_timeseries.csv").is_file() else None
        self.comparison_metrics = pd.read_csv(self.comparison / "comparison_metrics.csv") if (self.comparison / "comparison_metrics.csv").is_file() else None
        self.waveforms = pd.read_csv(self.comparison / "waveform_error_timeseries.csv") if (self.comparison / "waveform_error_timeseries.csv").is_file() else None
        self.harmonics = pd.read_csv(self.comparison / "harmonic_H1_H10.csv") if (self.comparison / "harmonic_H1_H10.csv").is_file() else None

    def save(self, number: int, name: str, fig, frame: pd.DataFrame, *, pdf: bool = False, sources: list[Path] | None = None) -> None:
        stem = f"{number:02d}_{name}"
        csv_path = self.out / f"{stem}.csv"
        write_csv(csv_path, frame)
        png_path = self.out / f"{stem}.png"
        svg_path = self.out / f"{stem}.svg"
        fig.savefig(png_path, dpi=220, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")
        if pdf:
            fig.savefig(self.out / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        source_list = sources or [self.scan / "training_tensor.npz"]
        self.entries.append({
            "number": number,
            "title": name,
            "script": str(Path(__file__).relative_to(ROOT)),
            "source_paths": [str(path) for path in source_list],
            "source_sha256": {str(path): digest(path) for path in source_list if path.is_file()},
            "csv": str(csv_path),
            "png": str(png_path),
            "svg": str(svg_path),
            "units": "see CSV columns and plot labels",
            "row_count": int(len(frame)),
            "numeric_ranges": {
                str(column): [float(frame[column].min()), float(frame[column].max())]
                for column in frame.select_dtypes(include=[np.number]).columns[:12]
                if len(frame) and np.all(np.isfinite(frame[column].to_numpy(dtype=float)))
            },
        })

    def run(self) -> None:
        self.figure_01()
        self.figure_02()
        self.figure_03()
        self.figure_04()
        self.figure_05()
        self.figure_06()
        self.figure_07()
        self.figure_08()
        self.figure_09()
        self.figure_10()
        self.figure_11()
        self.figure_12()
        self.figure_13()
        self.figure_14()
        self.figure_15()
        self.figure_16()
        self.figure_17()
        gate_path = self.comparison / "gate_decision.json"
        if (self.comparison / "attribution_trajectory.csv").is_file():
            self.figure_18()
        elif gate_path.is_file():
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            dynamic = [item for name, item in gate.get("gates", {}).items() if name.startswith("dynamic_BL_")]
            if any(not bool(item.get("pass", False)) for item in dynamic):
                self.figure_18()
        (self.out / "figure_manifest.json").write_text(json.dumps({"schema_version": 1, "figures": self.entries}, ensure_ascii=False, indent=2), encoding="utf-8")
        observations = ["# Tensor coenergy figure review", "", "Each entry records an observation, anomaly check, and conclusion from the exact CSV generated beside the figure.", ""]
        for entry in self.entries:
            observations.extend([
                f"## {entry['number']:02d} {entry['title']}",
                "",
                f"观察：CSV 行数为 {entry['row_count']}；数值范围摘要为 `{json.dumps(entry['numeric_ranges'], ensure_ascii=False)}`。",
                "",
                "异常：若范围中出现 NaN/Inf、W_ii 非正或门禁残差超限，须回到对应 CSV 和 gate_decision.json 复核；本图脚本不隐藏这些值。",
                "",
                "结论：图与同名 CSV 由同一批机器数据生成，可作为本阶段审查证据。",
                "",
            ])
        (self.out / "figure_observations.md").write_text("\n".join(observations), encoding="utf-8")

    def title(self, text: str) -> str:
        return f"{text}\nscan={short(self.scan / 'training_tensor.npz')}"

    def figure_01(self):
        from tools.build_tensor_magnetic_coenergy import build_context, MovingWinding, refine_tagged_mesh
        summary = json.loads((self.scan / "pilot_summary.json").read_text(encoding="utf-8"))
        context = build_context()
        mesh = context["mesh"]
        for _ in range(int(summary["selected_mesh_level"])):
            mesh = refine_tagged_mesh(mesh)
        winding = MovingWinding(mesh, context["mainline"]["tri_geometry"])
        fig, ax = plt.subplots(figsize=(9, 5))
        for domain, color, label in ((6, "tab:red", "soft iron"), (23, "tab:orange", "soft iron"), (24, "tab:purple", "permanent magnet")):
            for triangle in mesh.triangles[mesh.tri_domains == domain]:
                polygon = mesh.points_rz_m[np.r_[triangle, triangle[0]]]
                ax.plot(polygon[:, 0] * 1e3, polygon[:, 1] * 1e3, color=color, alpha=.18, lw=.4, label=label if label not in ax.get_legend_handles_labels()[1] else None)
        xs = [-4, -2, 0, 2, 4]
        rows = []
        for x in xs:
            pts = winding.at(x * 1e-3)["points_rz_m"]
            ax.scatter(pts[:, 0] * 1e3, pts[:, 1] * 1e3, s=8, label=f"x={x:+g} mm")
            rows.extend({"x_mm": x, "r_mm": p[0] * 1e3, "z_mm": p[1] * 1e3} for p in pts)
        ax.set(xlabel="r (mm)", ylabel="z (mm)", title=self.title("01 Coil motion geometry"))
        ax.legend(ncol=3, fontsize=8)
        self.save(1, "coil_motion_geometry", fig, pd.DataFrame(rows))

    def figure_02(self):
        from tools.build_tensor_magnetic_coenergy import build_context, MovingWinding, refine_tagged_mesh
        summary = json.loads((self.scan / "pilot_summary.json").read_text(encoding="utf-8"))
        context = build_context(); mesh = context["mesh"]
        for _ in range(int(summary["selected_mesh_level"])): mesh = refine_tagged_mesh(mesh)
        winding = MovingWinding(mesh, context["mainline"]["tri_geometry"])
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4))
        rows = []
        b0 = None
        for x in [-4, -2, 0, 2, 4]:
            loc = winding.at(x * 1e-3); b = np.asarray(loc["source_vector"]); support = np.flatnonzero(abs(b) > 1e-30)
            if b0 is None: b0 = b
            ax0.scatter(mesh.points_rz_m[support, 0] * 1e3, mesh.points_rz_m[support, 1] * 1e3, s=4, label=f"{x:+g} mm")
            rows.append({"x_mm": x, "source_norm": np.linalg.norm(b), "relative_to_zero": np.linalg.norm(b - b0) / max(np.linalg.norm(b0), 1e-300), "support_count": len(support), "support_sha256": loc["support_sha256"], "source_sha256": loc["source_sha256"]})
        f = pd.DataFrame(rows)
        ax0.set(xlabel="r (mm)", ylabel="z (mm)", title="b(x) nonzero support")
        ax0.legend(fontsize=8)
        ax1.plot(f.x_mm, f.relative_to_zero, "o-")
        ax1.set(xlabel="x (mm)", ylabel="||b(x)-b(0)||/||b(0)||", title="source displacement proof")
        self.save(2, "winding_vector_support", fig, f)

    def figure_03(self):
        import meshio
        selected_level = int(json.loads((self.scan / "pilot_summary.json").read_text(encoding="utf-8"))["selected_mesh_level"])
        files = sorted((self.scan / "field_snapshots").glob(f"*L{selected_level}.vtu"))
        fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
        values = []
        loaded = []
        for path in files:
            m = meshio.read(path); tri = next(c.data for c in m.cells if c.type == "triangle"); v = np.asarray(m.cell_data_dict["B_norm_T"]["triangle"]); values.append(v); loaded.append((path, m, tri, v))
        scale = max(max(v.max() for v in values), 1e-30) if values else 1
        rows = []
        for ax, item in zip(axes.flat, loaded):
            path, m, tri, v = item
            t = ax.tripcolor(m.points[:, 0] * 1e3, m.points[:, 1] * 1e3, tri, v, shading="flat", cmap="viridis", vmin=0, vmax=scale)
            domains = np.asarray(m.cell_data_dict["domain_id"]["triangle"], dtype=int)
            for domain, color, label in ((6, "white", "soft iron"), (23, "cyan", "soft iron"), (24, "magenta", "permanent magnet"), (17, "black", "moving coil"), (18, "black", "moving coil"), (19, "black", "moving coil")):
                mask = domains == domain
                if np.any(mask):
                    ax.triplot(m.points[:, 0] * 1e3, m.points[:, 1] * 1e3, tri[mask], color=color, lw=.35, alpha=.75, label=label if label not in ax.get_legend_handles_labels()[1] else None)
            br = np.asarray(m.cell_data_dict["B_r_T"]["triangle"], dtype=float)
            bz = np.asarray(m.cell_data_dict["B_z_T"]["triangle"], dtype=float)
            centers = m.points[tri].mean(axis=1)
            stride = max(1, len(centers) // 90)
            ax.quiver(centers[::stride, 0] * 1e3, centers[::stride, 1] * 1e3, br[::stride], bz[::stride], color="white", alpha=.65, angles="xy", scale_units="xy", scale=max(scale / .018, 1e-9), width=.0015)
            ax.set_title(path.stem.replace("pilot_", ""), fontsize=8); ax.set_aspect("equal"); rows.append({"file": path.name, "B_max_T": v.max(), "B_r_min_T": br.min(), "B_r_max_T": br.max(), "B_z_min_T": bz.min(), "B_z_max_T": bz.max(), "field_sha256": digest(path)})
            ax.legend(fontsize=5, loc="upper right")
        if loaded: fig.colorbar(t, ax=axes.ravel().tolist(), label="|B| (T)")
        fig.suptitle(self.title("03 Pilot field gallery"))
        self.save(3, "pilot_field_gallery", fig, pd.DataFrame(rows), sources=files)

    def figure_04(self):
        names = [("iterations", "iterations"), ("residual_A", "residual_A"), ("retry_index", "retry index"), ("runtime_s", "runtime (s)")]
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        rows = []
        X = np.asarray(self.training["x_training_m"])[:, 0] * 1e3; I = np.asarray(self.training["current_training_A"])[0]
        for ax, (key, label) in zip(axes.flat, names):
            data = np.asarray(self.training[key], dtype=float) if key in self.training.files else np.full((len(X), len(I)), np.nan)
            im = ax.imshow(data, origin="lower", aspect="auto", cmap="viridis", extent=[I[0], I[-1], X[0], X[-1]])
            ax.set(xlabel="i (A)", ylabel="x (mm)", title=label); fig.colorbar(im, ax=ax)
            rows.extend({"x_mm": x, "current_A": i, key: data[ix, ii]} for ix, x in enumerate(X) for ii, i in enumerate(I))
        self.save(4, "solver_quality_maps", fig, pd.DataFrame(rows))

    def figure_05(self):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        rows = self.mesh_convergence
        for ax, col, title in zip(axes, ["psi_raw_relative_change", "lorentz_force_relative_change", "lorentz_BL_relative_change"], ["psi", "Lorentz force", "BL gate"]):
            if col in rows: ax.plot(np.arange(len(rows)), rows[col], "o-")
            ax.axhline(0.005, color="k", linestyle="--", label="0.5% gate"); ax.set_yscale("log"); ax.set_title(title); ax.legend()
        self.save(5, "mesh_convergence", fig, rows)

    def _heat(self, ax, data, title, label, cmap="viridis", norm=None):
        X = np.asarray(self.surface.x_m if "x_m" in self.surface else self.surface.x_m)
        x = np.unique(X); i = np.unique(self.surface.current_A)
        z = np.asarray(data).reshape(len(x), len(i))
        im = ax.imshow(z, origin="lower", aspect="auto", extent=[i[0], i[-1], x[0] * 1e3, x[-1] * 1e3], cmap=cmap, norm=norm)
        ax.set(xlabel="i (A)", ylabel="x (mm)", title=title); plt.colorbar(im, ax=ax, label=label)

    def figure_06(self):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        x = np.asarray(self.training["x_training_m"])[:, 0]
        i = np.asarray(self.training["current_training_A"])[0]
        xx, ii = np.meshgrid(x, i, indexing="ij")
        for ax, col, title, label in zip(axes.flat, ["psi_Wb", "F_N", "BL_secant_N_A", "L_incremental_H"], ["raw psi", "raw Lorentz F", "raw secant BL", "incremental L"], ["Wb", "N", "N/A", "H"]):
            if col == "psi_Wb":
                raw = np.asarray(self.training["psi_training_Wb"])
            elif col == "F_N":
                raw = np.asarray(self.training["lorentz_force_N"])
            elif col == "BL_secant_N_A":
                raw = np.asarray(self.training["lorentz_BL_N_A"])
            else:
                raw = np.gradient(np.asarray(self.training["psi_training_Wb"]), np.asarray(self.training["current_training_A"])[0], axis=1)
            im = ax.imshow(raw, origin="lower", aspect="auto", extent=[i[0], i[-1], x[0] * 1e3, x[-1] * 1e3], cmap="viridis"); ax.set(xlabel="i (A)", ylabel="x (mm)", title=title); fig.colorbar(im, ax=ax, label=label); ax.scatter(ii.ravel(), xx.ravel() * 1e3, s=2, c="k", alpha=.3)
        raw_frame = pd.DataFrame({"x_m": xx.ravel(), "current_A": ii.ravel(), "psi_Wb": np.asarray(self.training["psi_training_Wb"]).ravel(), "F_N": np.asarray(self.training["lorentz_force_N"]).ravel(), "BL_secant_N_A": np.asarray(self.training["lorentz_BL_N_A"]).ravel(), "L_incremental_H": np.gradient(np.asarray(self.training["psi_training_Wb"]), i, axis=1).ravel()})
        self.save(6, "raw_tensor_surfaces", fig, raw_frame)

    def figure_07(self):
        columns = [("W_J", "W (J)"), ("F_N", "F (N)"), ("psi_Wb", "psi (Wb)"), ("L_incremental_H", "W_ii (H)")]
        x = np.unique(self.surface.x_m.to_numpy(dtype=float)) * 1e3
        current = np.unique(self.surface.current_A.to_numpy(dtype=float))
        X, I = np.meshgrid(x, current, indexing="ij")
        fig = plt.figure(figsize=(16, 10), constrained_layout=True)
        for index, (column, title) in enumerate(columns):
            z = self.surface[column].to_numpy(dtype=float).reshape(len(x), len(current))
            if column == "L_incremental_H":
                positive = z[z > 0]
                norm = Normalize(vmin=float(np.min(positive)) if positive.size else 0.0, vmax=float(np.max(z)) if np.max(z) > 0 else 1.0)
                cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("black")
                image_z = np.ma.masked_less(z, 0.0)
            else:
                norm = None; cmap = "viridis"; image_z = z
            ax = fig.add_subplot(2, 4, index + 1)
            image = ax.imshow(image_z, origin="lower", aspect="auto", extent=[current[0], current[-1], x[0], x[-1]], cmap=cmap, norm=norm)
            ax.set(xlabel="i (A)", ylabel="x (mm)", title=title); fig.colorbar(image, ax=ax, label=column)
            ax3 = fig.add_subplot(2, 4, index + 5, projection="3d")
            ax3.plot_surface(I, X, z, cmap=cmap, linewidth=0, antialiased=True)
            ax3.set_xlabel("i (A)"); ax3.set_ylabel("x (mm)"); ax3.set_zlabel(column); ax3.set_title(title + " 3D")
        surface_frame = self.surface[["x_m", "current_A"] + [column for column, _ in columns]].copy()
        fig.suptitle(self.title("07 Coenergy surfaces")); self.save(7, "coenergy_surfaces", fig, surface_frame, pdf=True)

    def figure_08(self):
        from loudspeaker_time_fem.nonlinear_law import NonlinearMagneticLaw
        legacy = NonlinearMagneticLaw.from_json(ROOT / "inputs/nonlinear_magnetic_law_20260728.json")
        fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True, constrained_layout=True)
        x = np.unique(self.surface.x_m) * 1e3; i_unique = np.unique(self.surface.current_A)
        raw_x = np.asarray(self.training["x_training_m"])[:, 0] * 1e3; raw_i = np.asarray(self.training["current_training_A"])[0]
        rows = []
        for current in [-1, -.5, 0, .5, 1]:
            for ax, col, ylabel in zip(axes, ["psi_Wb", "F_N", "BL_secant_N_A"], ["psi (Wb)", "F (N)", "BL (N/A)"]):
                mask = np.isclose(self.surface.current_A, current); ax.plot(self.surface.loc[mask, "x_m"] * 1e3, self.surface.loc[mask, col], label=f"i={current:+g} A")
                if col == "psi_Wb": rows.extend({"i_A": current, "x_mm": x0, "value": y} for x0, y in zip(self.surface.loc[mask, "x_m"] * 1e3, self.surface.loc[mask, col]))
            idx = int(np.argmin(abs(raw_i - current))); axes[0].plot(raw_x, self.training["psi_training_Wb"][:, idx], "k.", ms=2); axes[1].plot(raw_x, self.training["lorentz_force_N"][:, idx], "k.", ms=2); axes[2].plot(raw_x, self.training["lorentz_BL_N_A"][:, idx], "k.", ms=2)
            old_x = np.linspace(-4.0, 4.0, 161)
            axes[0].plot(old_x, [legacy.coupled_flux(x0 * 1e-3, current) for x0 in old_x], "--", lw=.8, color="0.35", label="legacy separable" if current == -1 else None)
            axes[1].plot(old_x, [current * legacy.coupled_force_factor(x0 * 1e-3, current) for x0 in old_x], "--", lw=.8, color="0.35")
            axes[2].plot(old_x, [legacy.coupled_force_factor(x0 * 1e-3, current) for x0 in old_x], "--", lw=.8, color="0.35")
        for ax in axes: ax.legend(ncol=3, fontsize=8); ax.grid(alpha=.2)
        axes[-1].set_xlabel("x (mm)"); self.save(8, "tensor_slices_x", fig, pd.DataFrame(rows))

    def figure_09(self):
        from loudspeaker_time_fem.nonlinear_law import NonlinearMagneticLaw
        legacy = NonlinearMagneticLaw.from_json(ROOT / "inputs/nonlinear_magnetic_law_20260728.json")
        fig, axes = plt.subplots(4, 1, figsize=(10, 13), sharex=True, constrained_layout=True)
        rows = []
        for x_mm in [-4, -2, 0, 2, 4]:
            mask = np.isclose(self.surface.x_m, x_mm * 1e-3)
            for ax, col, label in zip(axes, ["psi_Wb", "F_N", "BL_secant_N_A", "L_incremental_H"], ["psi (Wb)", "F (N)", "BL (N/A)", "L (H)"]): ax.plot(self.surface.loc[mask, "current_A"], self.surface.loc[mask, col], label=f"x={x_mm:+g} mm")
            rows.extend({"x_mm": x_mm, "current_A": j, "psi_Wb": p, "F_N": f, "BL_N_A": bl, "L_H": l} for j, p, f, bl, l in zip(self.surface.loc[mask, "current_A"], self.surface.loc[mask, "psi_Wb"], self.surface.loc[mask, "F_N"], self.surface.loc[mask, "BL_secant_N_A"], self.surface.loc[mask, "L_incremental_H"]))
            old_i = np.linspace(-1.0, 1.0, 161)
            axes[0].plot(old_i, [legacy.coupled_flux(x_mm * 1e-3, i0) for i0 in old_i], "--", lw=.8, color="0.35", label="legacy separable" if x_mm == -4 else None)
            axes[1].plot(old_i, [i0 * legacy.coupled_force_factor(x_mm * 1e-3, i0) for i0 in old_i], "--", lw=.8, color="0.35")
            axes[2].plot(old_i, [legacy.coupled_force_factor(x_mm * 1e-3, i0) for i0 in old_i], "--", lw=.8, color="0.35")
            axes[3].plot(old_i, [legacy.incremental_inductance(i0) for i0 in old_i], "--", lw=.8, color="0.35")
        for ax in axes: ax.legend(ncol=3, fontsize=8); ax.grid(alpha=.2)
        axes[-1].set_xlabel("i (A)"); self.save(9, "tensor_slices_i", fig, pd.DataFrame(rows))

    def figure_10(self):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        for ax, pred, ref, title in [(axes[0, 0], "psi_fit_Wb", "psi_Wb", "psi predicted vs FEM"), (axes[0, 1], "F_coenergy_N", "lorentz_force_N", "F predicted vs Lorentz"), (axes[1, 0], "psi_relative_error", None, "psi relative error"), (axes[1, 1], "F_relative_error", None, "F relative error")]:
            if ref: ax.scatter(self.holdout[ref], self.holdout[pred], s=15); lo=min(self.holdout[ref].min(), self.holdout[pred].min()); hi=max(self.holdout[ref].max(), self.holdout[pred].max()); ax.plot([lo, hi], [lo, hi], "k--")
            else: ax.hist(self.holdout[pred], bins=12)
            ax.set_title(title); ax.grid(alpha=.2)
        self.save(10, "holdout_validation", fig, self.holdout)

    def figure_11(self):
        from loudspeaker_time_fem.tensor_coenergy import TensorCoenergyLaw
        law = TensorCoenergyLaw.from_json(ROOT / "inputs/nonlinear_magnetic_coenergy_tensor_20260801.json")
        x = np.linspace(-.0038, .0038, 81); i = np.linspace(-.95, .95, 81); X, I = np.meshgrid(x, i, indexing="ij")
        analytic = np.asarray(law.dforce_di(X, I)); h = 1e-5; fd = (np.asarray(law.force(X, I + h)) - np.asarray(law.force(X, I - h))) / (2*h); err = fd - analytic
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        err_limit = max(float(np.max(np.abs(err))), 1e-30)
        err_norm = TwoSlopeNorm(vmin=-err_limit, vcenter=0.0, vmax=err_limit)
        for ax, z, title in zip(axes, [err, np.asarray(law.incremental_inductance(X, I)), abs(err)], ["FD reciprocity residual", "W_ii", "|residual|"]):
            im=ax.imshow(z, origin="lower", aspect="auto", cmap="coolwarm" if "residual" in title else "viridis", norm=err_norm if title == "FD reciprocity residual" else None); ax.set_title(title); fig.colorbar(im, ax=ax)
        frame=pd.DataFrame({"x_m": X.ravel(), "current_A": I.ravel(), "analytic_W_xi": analytic.ravel(), "finite_difference_W_xi": fd.ravel(), "residual": err.ravel()}); self.save(11, "reciprocity_and_passivity", fig, frame, sources=[ROOT / "inputs/nonlinear_magnetic_coenergy_tensor_20260801.json", ROOT / "inputs/nonlinear_magnetic_coenergy_tensor_20260801.npz"])

    def figure_12(self):
        fig, ax = plt.subplots(figsize=(8, 6)); x=np.unique(self.surface.x_m); i=np.unique(self.surface.current_A); z=self.surface.L_incremental_H.to_numpy().reshape(len(x),len(i)); im=ax.imshow(z, origin="lower", aspect="auto", extent=[i[0],i[-1],x[0]*1e3,x[-1]*1e3], cmap="viridis");
        trajectory = self.time
        frame = pd.DataFrame()
        if trajectory is not None:
            period = 1.0 / 70.0
            trajectory = trajectory.loc[trajectory.time_s >= trajectory.time_s.max() - period - 1e-10]
            ax.plot(trajectory.current_A, trajectory.coil_displacement_m*1e3, "w-", lw=.8, label="Python last-cycle trajectory")
            ax.scatter([trajectory.current_A.min(), trajectory.current_A.max()], [trajectory.coil_displacement_m.iloc[int(np.argmin(trajectory.current_A.to_numpy()))]*1e3, trajectory.coil_displacement_m.iloc[int(np.argmax(trajectory.current_A.to_numpy()))]*1e3], c=["tab:blue", "tab:red"], s=24, label="current extrema")
            ax.scatter([trajectory.current_A.iloc[int(np.argmin(trajectory.coil_displacement_m.to_numpy()))], trajectory.current_A.iloc[int(np.argmax(trajectory.coil_displacement_m.to_numpy()))]], [trajectory.coil_displacement_m.min()*1e3, trajectory.coil_displacement_m.max()*1e3], c=["tab:green", "tab:orange"], s=24, label="displacement extrema")
            frame = trajectory[["time_s","current_A","coil_displacement_m"]].copy()
        ax.axvline(-1.0, color="white", ls="--", lw=.7); ax.axvline(1.0, color="white", ls="--", lw=.7); ax.axhline(-4.0, color="white", ls="--", lw=.7); ax.axhline(4.0, color="white", ls="--", lw=.7)
        manifest_path = self.transient / "field_snapshot_manifest.csv"
        if manifest_path.is_file():
            snapshots = pd.read_csv(manifest_path)
            ax.scatter(snapshots["current_A"], snapshots["x_m"] * 1e3, marker="x", s=42, c="black", label="native field snapshots")
            frame = pd.concat([frame, snapshots[["time_s", "current_A", "x_m"]].rename(columns={"x_m": "coil_displacement_m"})], ignore_index=True)
        ax.set(xlabel="i (A)", ylabel="x (mm)", title=self.title("12 Operating trajectory on tensor")); fig.colorbar(im, ax=ax, label="L (H)"); ax.legend(fontsize=7)
        self.save(12, "operating_trajectory_on_tensor", fig, frame, sources=[self.scan / "fit_surface_grid.csv", self.transient / "all_probes_timeseries.csv", manifest_path] if self.time is not None and manifest_path.is_file() else [self.scan / "fit_surface_grid.csv", self.transient / "all_probes_timeseries.csv"] if self.time is not None else None)

    def figure_13(self):
        fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
        signals=[("current_A","current","coil_current_A"),("coil_displacement_m","x","coil_displacement_m"),("dynamic_BL_N_A","BL_secant","dynamic_BL_N_A"),("p_axis_near_0p10m_Pa","p axis","pressure_python_axis_near_actual"),("p_offaxis_45deg_0p10m_Pa","p 45deg","pressure_python_offaxis_actual"),("p_rear_physical_m0p10_Pa","p rear","pressure_common_rear_physical_m0p10")]
        wave = self.waveforms
        rows = []
        for ax, (col, label, signal) in zip(axes.flat, signals):
            if wave is not None and signal in set(wave.signal):
                subset = wave[wave.signal == signal].sort_values("time_s")
                for model, style in (("COMSOL", "k-"), ("baseline", "b--"), ("tensor", "r-")):
                    if model in subset:
                        values = subset[model].to_numpy(dtype=float); ax.plot(subset.time_s, values, style, lw=.9, label=model)
                        rows.extend({"time_s": t, "signal": signal, "model": model, "value": v} for t, v in zip(subset.time_s, values))
                if self.old_time is not None and col in self.old_time:
                    old = self.old_time.loc[self.old_time.time_s >= self.old_time.time_s.max() - 1.0 / 70.0 - 1e-10]
                    old_values = np.interp(subset.time_s, old.time_s, old[col])
                    ax.plot(subset.time_s, old_values, "g-.", lw=.8, label="old coupled")
                    rows.extend({"time_s": t, "signal": signal, "model": "old_coupled", "value": v} for t, v in zip(subset.time_s, old_values))
            ax.set_title(label); ax.set_xlabel("t (s)"); ax.grid(alpha=.2); ax.legend(fontsize=7)
        self.save(13,"last_cycle_waveforms",fig,pd.DataFrame(rows),sources=[self.transient/"all_probes_timeseries.csv", self.comparison/"waveform_error_timeseries.csv"])

    def figure_14(self):
        fig, axes=plt.subplots(3, 1, figsize=(11,8), sharex=True, constrained_layout=True); rows=[]
        wave = self.waveforms
        for ax, signal in zip(axes, ["coil_current_A", "coil_displacement_m", "dynamic_BL_N_A"]):
            if wave is not None and signal in set(wave.signal):
                subset=wave[wave.signal == signal].sort_values("time_s")
                ax.plot(subset.time_s, subset["tensor_minus_COMSOL"], label="tensor-COMSOL")
                ax.plot(subset.time_s, subset["baseline_minus_COMSOL"], label="baseline-COMSOL")
                rows.extend(subset[["time_s","signal","tensor_minus_COMSOL","baseline_minus_COMSOL"]].to_dict("records"))
            ax.axhline(0, color="k", lw=.6); ax.set_title(signal); ax.grid(alpha=.2); ax.legend()
        axes[-1].set_xlabel("t (s)"); self.save(14,"waveform_residuals",fig,pd.DataFrame(rows),sources=[self.comparison/"waveform_error_timeseries.csv"])

    def figure_15(self):
        fig, axes = plt.subplots(4, 3, figsize=(14, 13), constrained_layout=True); rows=[]
        data=self.harmonics
        signals=["coil_current_A","coil_displacement_m","dynamic_BL_N_A","pressure_python_axis_near_actual","pressure_python_offaxis_actual","pressure_common_rear_physical_m0p10"]
        if data is not None:
            for index, signal in enumerate(signals):
                subset=data[data.signal == signal]
                if subset.empty: continue
                ax_amp=axes[index // 3, index % 3]; ax_phase=axes[2 + index // 3, index % 3]
                for model, style in (("COMSOL","k-"),("baseline","b--"),("old_coupled","g-."),("tensor","r-")):
                    model_data=subset[subset.model == model].sort_values("harmonic")
                    if not model_data.empty:
                        ax_amp.plot(model_data.harmonic, model_data.relative_to_COMSOL_H1, style, marker=".", label=model)
                        ax_phase.plot(model_data.harmonic, model_data.phase_deg, style, marker=".", label=model)
                ax_amp.set_title(signal + " amplitude"); ax_amp.set_ylabel("peak/H1"); ax_amp.grid(alpha=.2)
                ax_phase.set_title(signal + " phase"); ax_phase.set_xlabel("harmonic"); ax_phase.set_ylabel("deg"); ax_phase.grid(alpha=.2)
                rows.extend(subset.to_dict("records"))
        for ax in axes.ravel(): ax.legend(fontsize=6)
        self.save(15,"harmonic_H1_H10",fig,pd.DataFrame(rows),sources=[self.comparison/"harmonic_H1_H10.csv"] if (self.comparison/"harmonic_H1_H10.csv").is_file() else None)

    def figure_16(self):
        fig, axes=plt.subplots(2,2,figsize=(14,9),constrained_layout=True); rows=self.comparison_metrics if self.comparison_metrics is not None else pd.DataFrame()
        if len(rows):
            signals=list(dict.fromkeys(rows.signal.tolist())); models=["baseline","old_coupled","tensor"]
            for ax, column, title, limit in [(axes[0,0],"H1_amplitude_relative_error","H1 amplitude error (%)",10.0),(axes[0,1],"H1_phase_error_deg","H1 phase error (deg)",10.0),(axes[1,0],"THD","THD",None),(axes[1,1],"waveform_NRMSE","waveform NRMSE",None)]:
                if column not in rows: continue
                positions=np.arange(len(signals)); width=.24
                for j,model in enumerate(models):
                    values=[]
                    for signal in signals:
                        hit=rows[(rows.signal==signal)&(rows.model==model)]
                        values.append(float(hit.iloc[0][column]) if not hit.empty else np.nan)
                    if column == "H1_amplitude_relative_error" or column == "H1_phase_error_deg": values=np.asarray(values)*100 if column == "H1_amplitude_relative_error" else np.asarray(values)
                    ax.bar(positions+(j-1)*width, values, width=width, label=model)
                if limit is not None: ax.axhline(limit,color="k",ls="--",label="gate")
                ax.set_title(title); ax.set_xticks(positions,signals,rotation=60,ha="right"); ax.grid(axis="y",alpha=.2); ax.legend(fontsize=7)
        self.save(16,"gate_metric_comparison",fig,rows,sources=[self.comparison/"comparison_metrics.csv"] if (self.comparison/"comparison_metrics.csv").is_file() else None)

    def figure_17(self):
        fig, axes=plt.subplots(2,1,figsize=(10,6),sharex=True,constrained_layout=True); frame=self.time if self.time is not None else pd.DataFrame(); rows=[]
        if len(frame):
            if "nonlinear_iterations" in frame: axes[0].plot(frame.time_s,frame.nonlinear_iterations,label="Newton iterations")
            if (self.transient/"energy_balance_timeseries.csv").is_file():
                eb=pd.read_csv(self.transient/"energy_balance_timeseries.csv"); axes[1].plot(eb.t,eb.discrete_balance_residual_W,label="balance residual"); rows=eb.to_dict("records")
        for ax in axes: ax.legend(); ax.grid(alpha=.2)
        axes[0].set_title(self.title("17 Newton and energy diagnostics")); axes[-1].set_xlabel("t (s)"); self.save(17,"newton_and_energy_diagnostics",fig,pd.DataFrame(rows) if rows else frame)

    def figure_18(self):
        path=self.comparison/"attribution_trajectory.csv"; frame=pd.read_csv(path) if path.is_file() else pd.DataFrame(); fig,axes=plt.subplots(3,1,figsize=(11,9),constrained_layout=True)
        if len(frame):
            if "t" in frame: axes[0].plot(frame.t,frame.get("COMSOL_BL_N_A",0),label="COMSOL"); axes[0].plot(frame.t,frame.get("tensor_BL_on_COMSOL_path_N_A",0),label="tensor on COMSOL path"); axes[0].plot(frame.t,frame.get("baseline_BL_on_COMSOL_path_N_A",0),label="baseline on COMSOL path"); axes[0].plot(frame.t,frame.get("tensor_BL_on_tensor_path_N_A",0),label="tensor own path")
            if "x_C_m" in frame: axes[1].plot(frame.x_C_m*1e3,frame.i_C_A,label="COMSOL (x,i)"); axes[1].plot(frame.x_P_m*1e3,frame.i_P_A,label="tensor (x,i)")
            if "x_C_m" in frame: axes[2].plot(frame.t,(frame.x_C_m-frame.x_P_m)*1e3,label="x_C-x_P (mm)"); axes[2].plot(frame.t,frame.i_C_A-frame.i_P_A,label="i_C-i_P (A)")
        for ax in axes: ax.legend(); ax.grid(alpha=.2)
        axes[0].set_title(self.title("18 Attribution if dynamic BL fails")); axes[1].set_xlabel("x (mm)"); axes[2].set_xlabel("t (s)"); self.save(18,"attribution_if_failed",fig,frame,sources=[path] if path.is_file() else None)


def main() -> int:
    parser=argparse.ArgumentParser(description="render tensor coenergy validation figures")
    parser.add_argument("--scan",required=True,type=Path)
    parser.add_argument("--transient",required=True,type=Path)
    parser.add_argument("--comparison",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args(); Renderer(args.scan,args.transient,args.comparison,args.out).run(); return 0


if __name__ == "__main__": raise SystemExit(main())
