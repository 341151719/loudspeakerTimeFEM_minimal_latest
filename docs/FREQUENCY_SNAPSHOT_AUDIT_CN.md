# 内置频域快照变更审计

审计基线：频域仓库 `ebc97e26f672a3aabf847617751529b55540b1d4`，本时域仓库 `f1c2f5b6d28d76df86bbb86c1efe4a4c7123fade`。频域仓库的[逐文件清单](https://github.com/341151719/loudspeakerFEM_minimal_latest/blob/main/docs/TIME_SNAPSHOT_AUDIT_CN.md)由 `tools/audit_time_snapshot.py` 生成。

`src/loudspeaker_time_fem/model.py` 直接导入快照中的 `axisym_magnetics`、`stage4C_acoustic_structure` 和 `p2_axisym_solid`。其静态导入闭包有 7 个模块；其中只有 `p2_axisym_solid.py` 与当前频域仓库不同。人工核对差异后，差别在 `complex_stiffness()` 的可配置频率过渡参数；当前时域 `model.py` 只调用该文件的 `build_p2_solid`、`assemble_p2_G` 和 `assemble_lorentz_force`，没有调用 `complex_stiffness()`。因此本次不复制频域文件，保留现有时域快照行为。

频域本次新增的扫频续跑代码只供频域 CLI 使用，不在时域导入闭包内。以后若改动落在这 7 个依赖模块内，应重新生成审计、检查函数级调用并在两个仓库分别提交；不要整目录覆盖 `inputs/frequency_mainline/`。
