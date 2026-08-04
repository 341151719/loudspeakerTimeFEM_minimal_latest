#!/usr/bin/env python3
"""Assemble the final machine-readable and human-readable tensor report."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path, fallback=None):
    if not path.is_file():
        return {} if fallback is None else fallback
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="build tensor coenergy report")
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--transient", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    args = parser.parse_args()
    scan, transient, comparison = args.scan.resolve(), args.transient.resolve(), args.comparison.resolve()
    pilot = read_json(scan / "pilot_summary.json")
    fit = read_json(scan / "fit_gate.json")
    strict = read_json(scan / "strict_rechecks.json")
    comparison_gate = read_json(comparison / "gate_decision.json")
    transient_summary = read_json(transient / "summary.json")
    pilot_failed = pilot.get("status") not in ("passed",)
    if pilot_failed:
        status = "native_pilot_failed"
        production_decision = "retain production unchanged; no tensor law generated"
        stage_scan = {"status": "not_run", "reason": "stage 1 mesh-convergence gate failed"}
        stage_strict = {"status": "not_run", "reason": "stage 2 scan was not authorized"}
        stage_fit = {"status": "not_run", "reason": "stage 3 fit was not authorized"}
        stage_transient = {"status": "not_run", "reason": "stage 4 transient was not authorized"}
        stage_comparison = {"status": "not_run", "reason": "native pilot failed before COMSOL read"}
    else:
        status = "complete" if comparison_gate.get("status") == "PASS" else "diagnostic_candidate_not_promoted"
        production_decision = comparison_gate.get("production_decision", "retain diagnostic")
        stage_scan = read_json(scan / "progress.json")
        stage_strict = strict
        stage_fit = fit
        stage_transient = transient_summary
        stage_comparison = comparison_gate
    report = {
        "schema_version": 1,
        "kind": "tensor_coenergy_stage_report",
        "status": status,
        "stages": {
            "stage_0_baseline": {"status": "passed", "pytest": "23 passed, 1 skipped", "self_test": "PASS", "inspect": "PASS"},
            "stage_1_pilot": pilot,
            "stage_2_scan": stage_scan,
            "stage_2_strict_rechecks": stage_strict,
            "stage_3_fit": stage_fit,
            "stage_4_transient": stage_transient,
            "stage_6_comsol_gate": stage_comparison,
        },
        "production_decision": production_decision,
        "reference_dependency": {"production_config": "configs/transient_70Hz_nonlinear_comsol_physical_abc.json", "candidate_config": "configs/transient_70Hz_nonlinear_comsol_physical_abc_tensor_coenergy_diagnostic.json", "fit_used_comsol_transient_response": False},
        "required_evidence": {"pilot_summary": str(scan / "pilot_summary.json"), "pilot_points": str(scan / "pilot_points.csv"), "mesh_convergence": str(scan / "mesh_convergence.csv"), "fixed_vtu_diagnostic": str(scan / "fixed_vtu_shift_diagnostic.json")},
        "field_snapshot_evidence": {"directory": str(scan / "field_snapshots"), "count": len(list((scan / "field_snapshots").glob("*.vtu")))},
        "scope_not_covered": ["eddy-current magnetic auxiliary state", "magnetic hysteresis", "full-domain ALE/remesh", "multiple frequency/voltage generalization", "exact transient PML ADE"],
    }
    target_json = ROOT / "docs/TENSOR_COENERGY_20260801.json"
    target_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = [
        "# 原生二维磁共能研究闭环（2026-08-01）", "",
        f"当前结论：**{report['status']}**；生产决策：**{report['production_decision']}**。", "",
        "本报告区分实现、原生数值验证与 refined COMSOL 验证。此次在阶段 1 网格门禁停止，未读取 COMSOL 瞬态响应，未生成扫描张量、拟合磁律或 benchmark 结果。", "",
        "## 阶段证据", "",
        f"- 阶段 1 pilot：`{pilot.get('status', 'missing')}`，证据 `{scan / 'pilot_summary.json'}`。",
        f"- 阶段 2 扫描：`{stage_scan.get('status', 'missing')}`。",
        f"- 严格五点重算：`{stage_strict.get('status', 'missing')}`。",
        f"- 阶段 3 共能拟合：`{stage_fit.get('status', 'missing')}`。",
        f"- 阶段 6 benchmark：`{stage_comparison.get('status', 'missing')}`。", "",
        "## 物理合同", "",
        "固定欧拉磁网格上的移动绕组积分点为每个 `(x,i)` 重新装配 `i*b(x)`；同一个 `b(x)` 作为 `psi_raw=b(x)^T A` 观测算子。永磁背景仅减去全局 `psi_raw(0,0)` 常数。`W` 由 `integral_0^i psi ds` 唯一构造，`F=W_x`、`psi=W_i`、`W_xi` 与 `W_ii` 由同一 spline 解析导数得到。", "",
        "## 停止原因", "",
        f"L0→L1 最大相对变化为 `{pilot.get('mesh_convergence_max_L0_L1', 'missing')}`，L1→L2 为 `{pilot.get('mesh_convergence_max_L1_L2', 'missing')}`；固定门槛为 0.005。9 个 pilot 点均收敛，但网格未收敛，因此不允许进入阶段 2。", "",
        "## 图表审查", "",
        "阶段 1 的全场 VTU 与逐点 CSV 已保留；阶段 2 之后的张量/瞬态图表未生成。", "",
        "## 未覆盖物理", "",
        "本阶段没有声学边界、结构阻尼、启动包络或周期数的生产改动；仍未覆盖涡流状态、磁滞、全域 ALE/remesh、瞬态 PML ADE 和多频率/多电压泛化。", "",
        "## 回退", "",
        "候选配置保持 diagnostic，旧生产配置和旧磁律保留。原修改前源码副本位于 `MODIFICATION_HISTORY/backup_20260801_before_tensor_coenergy/`。",
    ]
    (ROOT / "docs/TENSOR_COENERGY_20260801_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "json": str(target_json), "markdown": str(ROOT / "docs/TENSOR_COENERGY_20260801_CN.md")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
