# Agent 导航图：loudspeakerTimeFEM

本页是仓库的任务路由索引，不替代 [`README_CN.md`](../README_CN.md) 的物理合同、状态门槛和验收顺序。两仓库关系见 [`PROJECT_RELATIONSHIP_CN.md`](PROJECT_RELATIONSHIP_CN.md)。

## 项目边界

这是二维轴对称扬声器**时域** FEM。正式生产交接说明记录的主配置是 `configs/transient_70Hz_nonlinear_comsol_physical_abc.json`，生产磁律是 `inputs/nonlinear_magnetic_law_20260728.json`。CLI 在 `cli.py`，不是 `loudspeakerFEM` 的扫频 CLI。

## 按任务定位

| 任务 | 首要入口 | 继续阅读 |
|---|---|---|
| 检查装配或运行时域算例 | `cli.py inspect` / `cli.py run` | `configs/`、`src/loudspeaker_time_fem/config.py` |
| 总体装配与共享结构/声学装配 | `src/loudspeaker_time_fem/model.py` | `inputs/frequency_mainline/src/`、`inputs/frequency_mainline/best_model/` |
| 线性或非线性时域积分 | `src/loudspeaker_time_fem/solver.py`、`nonlinear_solver.py` | `tests/test_core.py`、`tests/test_nonlinear_jacobian.py` |
| 磁律、悬挂 Kms ROM、共能 | `nonlinear_law.py`、`suspension_rom.py`、`tensor_coenergy.py` | `configs/diagnostic_*.json`、`configs/*diagnostic.json`、`tools/build_tensor_*.py` |
| 辐射边界和原生声学网格 | `spherical_nrbc.py`、`native_acoustic.py`、`comsol_mesh.py` | `inputs/comsol_transient_mesh.mphtxt`、`tests/test_comsol_mesh.py` |
| COMSOL 独立验证 | `comsol_validation/` | `README_CN.md` 的 benchmark 口径；不可将结果用作 Python 生产校正 |
| 源码包完整性和独立性 | `self_test.py`、`tools/audit_reference_dependencies.py` | `tests/`、`README_CN.md` 第 2、4、10 节 |

## 频域依赖：不要误判成自动同步

时域模型通过 `src/loudspeaker_time_fem/config.py` 解析 `base_mainline`，再由 `model.py` 把 `inputs/frequency_mainline/src` 和 `inputs/frequency_mainline/best_model` 加入导入路径。正常打包运行优先使用仓库内副本。旁边的 `loudspeakerFEM` 克隆只是另一份工作树，不会自动覆盖或更新这里的快照。

若任务修改共享频域代码：先读 [`PROJECT_RELATIONSHIP_CN.md`](PROJECT_RELATIONSHIP_CN.md)，逐文件比较两份实现，确认哪一份是目标与依赖；如果两边都需要，分别修改、分别检查、分别提交。不得把整个频域仓库复制进 `inputs/frequency_mainline/` 来代替审查。

## 必须保留的数值状态

- 默认生产磁律为可分形式。张量共能当前是诊断分支；九个 pilot 点求解收敛不代表网格收敛，固定 0.5% 门槛未通过前不得升级。
- 保持球面时域辐射边界的曲率项 `p/R`；COMSOL 对照不是 Python 运行时依赖。
- 新结果写入唯一的新 `runs/` 路径或项目外目录，不写入 `inputs/`。CLI 会轮换已存在的输出目录，详见根目录 `AGENTS.md`。
- 完整验收次序和不得宣称的结论见 `README_CN.md` 第 4、7、8、10 节。
