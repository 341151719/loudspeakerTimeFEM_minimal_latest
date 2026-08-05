# loudspeakerTimeFEM minimal latest

二维轴对称扬声器时域多物理 FEM 的最小独立源码包，面向 Python 求解、COMSOL 离线验证和后续 AI 分析。

## Start here

完整的中文交接说明在 [`README_CN.md`](README_CN.md)。它记录了当前生产配置、物理合同、验证边界、张量共能 pilot 的停止点以及下一阶段的推荐路线。版本冻结日期为 **2026-08-01**。

这个仓库不包含大型历史运行目录、图片、已求解的 COMSOL MPH 文件或 Python 虚拟环境；`inputs/` 中保留的是开箱运行所需的网格、几何、静磁场和磁律输入。精选的小型 benchmark 汇总位于 [`benchmarks/`](benchmarks/)，并明确标注哪些结论通过、未闭合或仅供诊断。

## What this project contains

- `src/loudspeaker_time_fem/`：时域结构、电路、声学、非线性磁律和求解器；
- `configs/`：生产配置与隔离的诊断配置；当前生产入口为
  `configs/transient_70Hz_nonlinear_comsol_physical_abc.json`；
- `inputs/`：生产网格、频域主线输入和 `nonlinear_magnetic_law_20260728.json`；
- `comsol_validation/`：COMSOL Java 导出和离线对比工具；
- `tests/`、`self_test.py`：独立性和数值代码检查；
- `tools/`：磁律、张量共能 pilot、审计和报告工具。

COMSOL 许可证和已求解模型不在仓库中。默认 Python 主链不在运行时读取 COMSOL 结果；COMSOL 文件只用于独立 benchmark。

## Curated benchmarks

[`benchmarks/README.md`](benchmarks/README.md) and [`benchmarks/README_CN.md`](benchmarks/README_CN.md)
describe the public-safe benchmark snapshot. It includes the 70 Hz refined comparison,
time/mesh convergence, interface isolation, FFT cross-check, boundary diagnostics, and the
failed tensor-co-energy pilot. Raw COMSOL models and long time series remain external inputs.

## Installation and first checks

建议在 Linux 或 WSL 中执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

PYTHONPATH=src python self_test.py
PYTHONPATH=src python cli.py inspect \
  --config configs/transient_70Hz_nonlinear_comsol_physical_abc.json
PYTHONPATH=src python -m pytest -q
```

正式 70 Hz 基线可能耗时较长；首次验证应复制配置到临时位置并缩短周期数，不要修改生产配置，也不要把结果写入 `inputs/`。正式运行时输出到新的 `runs/` 或项目外目录：

```bash
PYTHONPATH=src python cli.py run \
  --config configs/transient_70Hz_nonlinear_comsol_physical_abc.json \
  --outdir runs/transient_70Hz_nonlinear_comsol_physical_abc \
  --scratch-root /tmp/loudspeaker-time-fem-runs
```

## Current status and limitations

- 当前生产磁律仍是已验证的可分形式，生产配置没有切换到张量共能诊断分支；
- 张量共能 pilot 的 0.5% 网格门槛尚未通过，因此不能把 pilot 结果描述为生产模型或网格收敛结论；
- `README_CN.md` 中的误差数字是历史完整项目的证据摘要，不代表解压后已经重新运行；
- 少量历史诊断脚本保留了旧 `/mnt/...` 默认路径，运行这些脚本时必须显式指定本机路径；主生产入口使用仓库内的 `inputs/` 和 `configs/`。

如需修改物理模型、配置或数值内核，请先阅读 `README_CN.md` 的“新 AI 必须先接受的结论”“下一阶段的唯一推荐路线”和“禁止事项”。
