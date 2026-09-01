# loudspeakerTimeFEM 最小独立交接包

版本冻结日期：2026-08-01  
交付性质：可独立安装、装配和运行 Python FEM；不包含历史运行结果、图片或 COMSOL 已解模型。

## 1. 新 AI 必须先接受的结论

这是扬声器二维轴对称时域 FEM 的当前生产源码包。解压后的目录是独立项目，不允许依赖原始 `loudspeakerTimeFEM`、相邻 `loudspeakerFEM` 或开发机绝对路径。Python 第三方库通过 pip 安装；Python 运行必需、不能由 pip 获得的网格、几何、静磁场和磁律已放入 `inputs/`。

当前生产入口是：

```text
configs/transient_70Hz_nonlinear_comsol_physical_abc.json
```

当前生产磁律是：

```text
inputs/nonlinear_magnetic_law_20260728.json
```

2026-08-01 新增了二维张量磁共能实现，但其三层网格 pilot 未达到 0.5% 门槛，所以它只保留为诊断代码，不能启用为生产磁律。不得生成一个虚假的张量 JSON 来绕过保护，也不得把“9/9 点求解收敛”误写成“网格收敛”。

## 2. 包含与排除范围

本包包含：

- `src/loudspeaker_time_fem/`：时域模型、求解器、非线性磁律、声学、COMSOL 网格读取和张量共能代码；
- `inputs/frequency_mainline/`：时域模型实际导入的频域结构/声学代码及其运行输入；
- `inputs/comsol_transient_mesh.mphtxt`：生产声学网格；
- `inputs/nonlinear_magnetic_law_20260728.json`：当前生产非线性磁律；
- `configs/`：生产配置和隔离的诊断配置；
- `tests/`、`self_test.py`：代码与独立性检查；
- `tools/`：磁律构造、张量共能 pilot/拟合/报告、审计及诊断工具源码；
- `comsol_validation/` 中的 Python/Java 导出与对比程序，但不带对比数据；
- 本文件，它是包内唯一的项目说明文档。

本包明确不包含：

- `runs/`、历史 CSV/NPZ/JSON 结果、checkpoint；
- PNG/JPG/GIF、VTK/VTU 可视化输出（频域子模型运行必需的三个静磁 VTU 除外）；
- COMSOL solved MPH、COMSOL 运行日志及历史 benchmark 导出数据；
- `.venv`、`site-packages`、pip wheel、缓存、`__pycache__`、egg-info；
- 修改历史、旧报告、旧压缩包和备份副本。

`inputs/frequency_mainline/inputs/.../*.vtu` 是 Python 默认生产链的静磁初值/偏置场输入，不是本次附带的扫频或瞬态完成结果；删除它们会破坏开箱运行。

## 3. 环境和安装

要求 Python 3.11 或更高。建议在 WSL/Linux 中从解压目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

若系统上的 `gmsh` wheel 缺少 OpenGL 共享库，Ubuntu/WSL 常见补充为 `libglu1-mesa`。默认生产运行读取现成网格，不启动 COMSOL，也不需要 COMSOL 许可证。

## 4. 解压后的强制验收顺序

任何新 AI 必须按顺序执行，前一步失败就停止，不得跳过后宣称可运行：

```bash
source .venv/bin/activate
PYTHONPATH=src python self_test.py
PYTHONPATH=src python cli.py inspect --config configs/transient_70Hz_nonlinear_comsol_physical_abc.json
PYTHONPATH=src python -m pytest -q
```

随后做一个真实但缩短的 Python 时域冒烟算例。不要改生产配置；复制配置到临时目录，把周期数和每周期步数减小，再运行。若只验证正式基线，使用：

```bash
PYTHONPATH=src python cli.py run \
  --config configs/transient_70Hz_nonlinear_comsol_physical_abc.json \
  --outdir runs/transient_70Hz_nonlinear_comsol_physical_abc \
  --scratch-root /tmp/loudspeaker-time-fem-runs
```

正式基线是 70 Hz、10 V peak、4 周期、180 steps/period，可能耗时较长。输出只能写入新生成的 `runs/` 或项目外目录，不能写回 `inputs/`。

## 5. 当前生产物理合同

生产模型的核心合同如下，修改时不得遗漏：

- 二维轴对称 P2 结构；
- COMSOL 原生物理声域网格上的 P1 压力声学；
- 电路、结构、声学采用单体/低秩 Newton 耦合；
- 移动线圈采用接口级 ALE 降阶坐标映射，不是全域 remesh；
- 外边界是一阶球面时域辐射条件 `dp/dr + p/R + (1/c) dp/dt = 0`，曲率项 `p/R` 不得删除；
- `BL(x)` 与 `lambda(i)` 来自原生静磁场扫描；
- 生产磁共能仍采用已验证的可分形式；
- CLI 默认拒绝 `reference_identified` 配置，只有明确诊断才可传 `--allow-reference-diagnostic`。

COMSOL benchmark 曾包含非线性 B-H、感应电流、Lorentz 力、几何非线性、moving mesh/remesh、瞬态压力声学和 PML。Python 模型并未逐项等价实现这些全部物理，因此不得把工程一致性写成全物理等价。

## 6. 已验证生产基线

历史完整项目在 70 Hz、10 V peak、4 周期下，对 refined COMSOL 得到：

- 电流 H1 幅值误差约 0.44%，相位约 -1.40°；
- 位移 H1 幅值误差约 0.18%，相位约 +0.61°；
- 前轴约 0.10 m 声压 H1 误差约 3.44%，相位约 +0.55°；
- 45°约 0.10 m 声压 H1 误差约 1.14%，相位约 +0.58°；
- 后腔物理点声压 H1 误差约 6.21%，相位约 +0.87°；
- 近轴 THD：Python 约 2.255%，COMSOL 约 2.497%；
- 界面振型复数 L2 误差约 0.213%，复相关约 0.999998。

这些数值是历史证据摘要，本最小包没有携带原始对比数据，不能仅凭本文件重新声称已经复算。工程 10%/10°口径通过；严格 1% 空间网格无关性未完全闭合。动态 BL H1 约 23.7%/-15.1°，波形 NRMSE 约 1.096%，严格研究指标未通过。

## 7. 2026-08-01 张量共能成果与停止点

已经实现：

- `src/loudspeaker_time_fem/tensor_coenergy.py`：共能面读取、样条和解析导数；
- `src/loudspeaker_time_fem/nonlinear_solver.py`：隔离的 tensor 分支及 Jacobian；
- `tools/build_tensor_magnetic_coenergy.py`：固定欧拉网格的 `(x,i)` pilot、扫描与拟合流程；
- `tools/build_tensor_comparison.py`、`build_tensor_report.py`、绘图/场导出工具；
- `tests/test_tensor_coenergy.py`、`test_nonlinear_jacobian.py`；
- `configs/...tensor_coenergy_diagnostic.json`：仅诊断配置。

物理定义固定为：每个 `(x,i)` 在固定欧拉磁网格上重新装配移动绕组源 `i*b(x)`；同一 `b(x)` 用于 `psi_raw=b(x)^T A`；永磁背景只减去全局常数 `psi_raw(0,0)`；`W'(x,i)=integral_0^i psi(x,s) ds`；力 `F=dW'/dx`、磁链 `psi=dW'/di`、交叉导数和增量电感均来自同一共能面解析导数。

pilot 的 9/9 个点均求解收敛，但：

- L0→L1 最大相对变化：1.206572%；
- L1→L2 最大相对变化：0.854859%；
- 固定门槛：0.5%。

所以状态是 `native_pilot_failed / failed_mesh_not_converged`。未运行 513 点正式扫描、未拟合张量磁律、未运行张量瞬态、未做 refined COMSOL benchmark。生产配置和生产磁律保持不变。尤其要注意：失败主要来自 Lorentz/BL 后处理的收敛；L1→L2 的磁链最大变化约 0.356%，不能用这个较小数字掩盖力泛函不收敛。

## 8. 下一阶段的唯一推荐路线

新 AI 不要立即扩大扫描。按以下次序处理：

1. 先使 pilot 证据可审计：每一层、每一点保存绝对 `psi`、`F/BL`、残差、迭代数、网格统计和场快照；不得只保留层间百分比。
2. 对磁力泛函做收敛改进：优先比较能量/虚功一致的力、连续梯度恢复或局部加密，不允许用 COMSOL 数值校正原生力。
3. 固定原 pilot 九点和 0.5% 门槛重跑 L0/L1/L2。只有全部指定指标通过才能进入正式扫描。
4. 通过后生成完整 `(x,i)` 张量；检查对称性、单调性、增量电感正性、混合偏导一致性、节点重构误差和边界外推保护。
5. 再启用诊断配置运行完整瞬态；同时保留旧生产基线做 A/B。
6. 最后才使用独立 refined COMSOL 数据比较电流、位移、多个声压探针、THD、动态 BL 和能量残差。
7. 只有预先规定的指标整体净改善且无回归，才允许把 tensor 配置升级为生产；否则回退。

## 9. 图、数据与报告的硬性要求

本包没有附带图和历史数据，但下一轮研究必须重新导出：

- 三层网格与材料/线圈/永磁体标记图；
- 九点 pilot 的 `psi`、力、残差和层间变化表；
- 代表点的 `A_phi`、`B_r`、`B_z`、`|B|`、能量密度场；
- 完整共能面、磁链面、力面、增量电感面及切片；
- 混合偏导闭合误差、拟合残差和外推触发位置；
- 时域电流/位移/压力/BL 波形、H1、THD、NRMSE、能量收支；
- Python 与 COMSOL 的同坐标、同单位、同时间窗比较。

所有图必须有单位、色标、网格/配置标识；所有汇总数必须能追溯到逐点 CSV/JSON。报告不得只贴漂亮图片，也不得只有汇总表而没有绝对原始值。

## 10. 三维非轴对称诊断线

项目现已加入第一层三维非轴对称分析：

- `src/loudspeaker_time_fem/asymmetry3d.py`：周向 Fourier 模态、摇摆倾角、周向高阶分割能量、盆架角向透射/模态耦合、Rayleigh 三维辐射和有限箱体 Kirchhoff 散射；
- `configs/asymmetry3d_diagnostic.json`：包含 `m=0/1/3/5`、五辐条盆架和偏置矩形箱体的诊断示例；
- `tools/plot_asymmetry3d.py`：分别输出振膜/盆架、膜前近场、膜后近场、箱体背板外场和全空间远场；
- `cli.py asymmetry3d`：输出机器可读 NPZ 与 JSON。

运行机器数据分析：

```bash
PYTHONPATH=src python cli.py asymmetry3d \
  --config configs/asymmetry3d_diagnostic.json \
  --outdir runs/asymmetry3d_diagnostic
```

运行独立可视化：

```bash
PYTHONPATH=src python tools/plot_asymmetry3d.py \
  --config configs/asymmetry3d_diagnostic.json \
  --outdir runs/asymmetry3d_diagnostic_visuals
```

必须准确理解其物理层级：当前版本对给定的周向模态速度做三维声辐射分析，可以表示 `m=1` 摇摆、`m>=2` 花瓣分割、非对称盆架后向遮挡和非对称箱体的有限面散射；它还没有从材料、悬边和音圈三维网格自动求出这些模态幅值。箱体使用高频 Kirchhoff physical-optics 近似，不是已收敛的全波 BEM/FEM。局部几何非线性屈曲、接触、裂纹扩展及大冲程下的自发对称破缺仍需要第二层三维结构/声学体网格求解器，不能由本诊断配置宣称已经预测。

## 11. 禁止事项与完成定义

禁止：改生产配置做试验；读取历史 COMSOL 数据作为运行时校正；把诊断辨识参数称为预测；只看求解器残差而忽略网格收敛；用单频/单点改善替代完整回归；覆盖唯一 COMSOL canonical 文件；在输入目录写结果。

一个修改只有同时满足以下条件才算完成：源码和配置隔离清楚；单元测试通过；模型能从本目录独立装配和运行；数值门禁通过；图和机器可读数据齐全；与旧生产基线完成回归；已知限制如实记录；未引入原项目绝对路径或隐藏 COMSOL 运行时依赖。
