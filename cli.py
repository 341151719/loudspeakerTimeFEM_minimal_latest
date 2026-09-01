#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import time

from loudspeaker_time_fem.config import assert_native_production_config, load_config
from loudspeaker_time_fem.export import export_result
from loudspeaker_time_fem.model import build_transient_model
from loudspeaker_time_fem.nonlinear_solver import solve_nonlinear_transient
from loudspeaker_time_fem.solver import solve_transient


ROOT = Path(__file__).resolve().parent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="轴对称扬声器时域 FEM")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="装配模型并输出矩阵/探针摘要")
    inspect.add_argument("--config", default=str(ROOT / "configs/transient_70Hz.json"))
    inspect.add_argument(
        "--allow-reference-diagnostic", action="store_true",
        help="允许装配明确标记为 reference_identified 的诊断配置",
    )
    run = sub.add_parser("run", help="单次求解并批量导出全部结果")
    run.add_argument("--config", default=str(ROOT / "configs/transient_70Hz.json"))
    run.add_argument("--outdir", default=str(ROOT / "runs/transient_70Hz"))
    run.add_argument(
        "--allow-reference-diagnostic", action="store_true",
        help="允许运行明确标记为 reference_identified 的诊断配置",
    )
    run.add_argument(
        "--scratch-root",
        default=None,
        help="临时计算根目录；建议指向 WSL Linux 文件系统，完成后一次性回写 outdir",
    )
    asymmetry = sub.add_parser(
        "asymmetry3d",
        help="运行周向模态、摇摆、盆架遮挡与三维辐射诊断",
    )
    asymmetry.add_argument(
        "--config", default=str(ROOT / "configs/asymmetry3d_diagnostic.json")
    )
    asymmetry.add_argument(
        "--outdir", default=str(ROOT / "runs/asymmetry3d_diagnostic")
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "asymmetry3d":
        from loudspeaker_time_fem.asymmetry3d import analyze, export_analysis

        config_path = Path(args.config).resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        summary = export_analysis(analyze(config), Path(args.outdir).resolve())
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    config, config_path = load_config(args.config)
    if not args.allow_reference_diagnostic:
        assert_native_production_config(config)
    t0 = time.perf_counter()
    model = build_transient_model(config, config_path)
    if args.command == "inspect":
        print(json.dumps(model.metadata, ensure_ascii=False, indent=2))
        return 0

    scratch_parent = Path(args.scratch_root).resolve() if args.scratch_root else None
    if scratch_parent:
        scratch_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="loudspeaker-time-", dir=scratch_parent) as tmp:
        result = (
            solve_nonlinear_transient(model)
            if model.nonlinear_law is not None
            else solve_transient(model)
        )
        summary = export_result(model, result, tmp)
        outdir = Path(args.outdir).resolve()
        if outdir.exists():
            backup = outdir.with_name(outdir.name + ".previous")
            if backup.exists():
                shutil.rmtree(backup)
            outdir.replace(backup)
        shutil.copytree(tmp, outdir)
    summary["wall_seconds_including_build_export"] = time.perf_counter() - t0
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
