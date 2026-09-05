# Phase 2.8r1 审查修正版

本目录是当前修正入口。原 phase2_8 脚本、配置及 outputs/phase2_8 均保留为历史版本；新增共享函数使用 v2 名称，避免改变历史调用语义。协议见 [PROTOCOL.md](PROTOCOL.md)。

## 复现

在仓库根目录使用现有 .venv（PowerShell）：

```powershell
& .\.venv\Scripts\python.exe experiments/phase2_8_r1/24_information_decomposition.py
& .\.venv\Scripts\python.exe experiments/phase2_8_r1/25_kernel_bridge.py
& .\.venv\Scripts\python.exe experiments/phase2_8_r1/verify_results.py
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Task25 正式命令运行 16/32/64 相位的完整参数选择及留出评价，32 为主结果。--quick 写独立 outputs/phase2_8_r1_quick；不能替代正式结果。Task24 --output-root 仅允许在对应 r1 输出目录之下，不能覆盖历史版本。

## 产物

- predictability_oof.csv：19,232 条样本×目标分量×模型×CV 记录，含原始 dataset_index、fold、分组、observed/predicted/train_null、score_scale。不是 19,232 个独立实验。Pl 四个 ILR 分量分别记录；joint 输出原始坐标，score_scale 重建其标准化评分。
- predictability_spectrum_folds.csv：原 Q²/alpha + 标量 R² 对照；folds/ 中包含主样本和 in-box 的 train/test 索引。
- predictability_spectrum.png / hatch_contribution_spectrum.png：统一顺序的目标谱图和 src/proc paired hatch contribution。delta 是逐折差的 median，不是两个 median 相减。
- candidate_simulations.csv：全部候选×条件的原始分布、物理有效性、最小场值和最小裕量。
- candidate_selection.csv：每一 LOGO 折全部候选的组级 median、训练组名单、有效性与并列标记；测试响应与测试有效性不参与选择。
- kernel_bridge_levels.csv：全部 7×5 条留出预测，包括 raw q 与评分 q、失败标志；物理失败保留并按 invalid class 计分，不能删掉。
- summary/gsl28_b_evaluation.json：门槛、物理失败、并列最优集合。physical_invalid 是模型不能通过当前物理检查，不是程序崩溃。
- phase_grid_sensitivity.csv 和 phase_sensitivity/16、64：各相位数独立重新选择参数和评价。
- acceptance_checks.json：从 OOF 重建 Q²、历史主统计一致、真实反例被拦截等检查。
- frozen_integrity.json：本次修改前后 33 个旧 formal/回归产物 SHA256 对照。

## 修正后结果

32 相位组级 TV：L1=0.34375，L2=0.357143，L3a=0.34375，L3b=0.428571。L2/L3a 未通过改进门槛；L3b 有 1 条留出物理失败，标记 physical_invalid。16/32/64 下均未获得可用的模型改进证据。旧“负 γ 唯一正向信号”不再是当前结论；每折都存在跨正负号的并列最优候选，不能从 tie-break 负号识别机制。

Task24 原 360 行折级 Q² 保持一致，新增 OOF 与 R² 不更改主统计结论。样本仍为 discovery 数据，不能称独立 confirmation。后续 effect maps/error atlas/P/T separability/PDE 与设计验证见任务说明中的 JMPT 无新增实验路线，不在本次修正中虚报为完成。
