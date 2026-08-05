# 精选 benchmark 记录

本目录是从 `loudspeakerTimeFEM_codex_full_20260802` 选出的、适合公开仓库保存的
小型 benchmark 快照。它保存可审计的指标、判定和必要的逐点汇总，不把开发机环境
或 COMSOL 大型结果文件直接提交到 GitHub。

这些结果是历史运行的归档证据，不是 GitHub Actions 自动测试，也不能替代重新运行。
Python 的配置、求解器和对比脚本仍在仓库主体中；需要重新生成 COMSOL 参考数据时，
必须在本地具备 COMSOL 6.3、原始 MPH 和对应的导出时序。

## 选择原则

- 保留能够说明模型正确性、数值稳定性或物理层归因的 benchmark；
- 同时保留“未闭合”和“拒绝进入生产”的结果，避免只展示成功案例；
- 只提交小型 CSV/JSON 和说明，不提交 `.mph`、`.venv`、日志、长时序、VTU 场快照、
  GIF 或缓存；
- 清理了 Windows/WSL 开发机绝对路径，公开文件只引用仓库内相对路径或明确的
  “未随仓库提供”的外部输入。

## Benchmark 清单

| ID | 内容 | 结论 | 入口 |
|---|---|---|---|
| `comsol_refined_70hz` | 70 Hz、10 V peak、4 周期；Python 对 refined COMSOL 的电流、位移、声压、THD 和动态 BL 对照 | 已注册工程物理判据通过；PML 内后向点只作诊断 | `results/comsol_refined_70hz/` |
| `comsol_time_convergence_70hz` | COMSOL baseline 与 tighter time stepping | 主量变化满足 1%/1° 时间门槛 | `results/comsol_time_convergence_70hz/` |
| `comsol_mesh_convergence_70hz` | COMSOL baseline 与 refined mesh | 严格 1%/1° 空间门槛未完全闭合；保留作为限制证据 | `results/comsol_mesh_convergence_70hz/` |
| `interface_motion_substitution_70hz` | 用 COMSOL 界面位移驱动同一个 Python 声学算子 | 复数界面 L2 误差约 0.213%，用于隔离声学层 | `results/interface_motion_substitution_70hz/` |
| `native_fft_crosscheck_70hz` | COMSOL 原生 FFT 与外部 H1–H10 最小二乘 THD | 差约 0.0055 个百分点 | `results/native_fft_crosscheck_70hz/` |
| `spherical_dtn_diagnostic_70hz` | 频域精确球面 DtN 的 `lmax` 消融 | 仅诊断；高阶项没有形成生产边界 | `results/spherical_dtn_diagnostic_70hz/` |
| `tensor_coenergy_pilot_20260801` | 原生二维磁共能 9 点、三层网格 pilot | 9/9 点求解收敛，但网格门禁失败，未生成张量磁律 | `results/tensor_coenergy_pilot_20260801/` |
| `multistage_decisions_20260729` | 候选改动的接受/拒绝记录 | 生产默认保持不变 | `decisions/multistage_decisions_20260729.json` |

## 重点结论

70 Hz 主对照的公开汇总来自同一个最后完整周期窗口 `3*T0 <= t < 4*T0`。最终
refined 对照中的代表性结果为：电流 H1 幅值误差 0.44% / 相位 -1.40°，位移
0.18% / 0.61°，近轴声压 3.44% / 0.55°，45°声压 1.14% / 0.58°，后腔物理点
6.21% / 0.87°；近轴 THD 为 Python 2.255%、COMSOL 2.497%。动态 BL 的 H1
误差仍约 23.7% / -15.1°，所以“工程验证通过”不等于所有派生量都达到严格研究
指标。

时间收敛是通过项；空间收敛的严格记录明确为未闭合：近轴和后腔物理点的 H1
变化分别约 1.04% 和 1.67%。这两个结果不能被压缩成“完全网格无关”。

张量共能 pilot 的最大层间变化为 L0→L1 `1.206572%`、L1→L2 `0.854859%`，
固定门槛为 `0.5%`。因此它被保留为失败的研究 benchmark，生产配置和生产磁律
没有切换。

## 数据字段约定

- `H1_amplitude_relative_error`、`H1_relative_change` 等字段以小数保存：`0.01`
  表示 1%；
- `THD` 字段以小数保存：`0.025` 表示 2.5%；
- `THD_absolute_percentage_point_error` 是百分点：`0.5` 表示 0.5 个百分点；
- 相位字段的单位是度；长度、电流、压力等单位写在字段名中；
- `summary.json` 是机器可读的判定摘要，CSV 是对应的逐信号或逐边界数据。

## 如何复核

先运行项目自身的独立检查：

```bash
PYTHONPATH=src python self_test.py
PYTHONPATH=src python -m pytest -q
```

然后可用标准库检查快照格式：

```bash
python -m json.tool benchmarks/manifest.json >/dev/null
find benchmarks -name '*.json' -print0 \
  | xargs -0 -n1 python -m json.tool >/dev/null
```

公开快照不包含 COMSOL 原始求解所需的 `solved.mph`、长时序或许可证，因此不能仅
靠本目录重跑外部验证。外部重跑应使用 `comsol_validation/` 中的导出/比较工具和
相同的探针、时间窗及判定门槛；不要把历史结果写回 `inputs/`。
