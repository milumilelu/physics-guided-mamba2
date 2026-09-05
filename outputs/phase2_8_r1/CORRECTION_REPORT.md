# Phase 2.8r1 审查修正与验收

日期：2026-09-05。基于 HEAD cfa5542 的工作区修正。修正范围是审查列出的六项实现/契约/表述问题，以及后续路线冲突；没有把尚未执行的科研扩展标记完成。旧 phase2_8 脚本与配置保留，当前入口为 experiments/phase2_8_r1。

## 六项修正

| 审查项 | 处理 | 验证 |
|---|---|---|
| 正去除区域物理约束漏检 | 新增 physical_validity_relative_v2：所有像素 z≥min(base,0)−0.01 µm；同样检查留出场 | dataset_index=177、γ=−0.5 的 −4.808 µm 反例被正确拒绝；失败留出不删样本 |
| L3a 半周期采样 | 非零 c 的 phase grid 覆盖 2h，训练/留出一致 | 16/32/64 完整候选选择与评价均执行 |
| mean/median 偏离 | 恢复训练 kernel-group 平均损失的 median；评价仍为留出组均值 | 新增对抗测试，保存所有候选、训练组、并列最优集 |
| realization 过强结论 | README 改为单对重复不足以判断可控性；四输入匹配明确不含 h | exact repeat 限定五参数及独立 source |
| 缺 OOF/R² | 保存原坐标 observed/predicted/train-null、分组、原 dataset_index、评分尺度、标量 R²；补 in-box fold artifact | 19,232 分量记录，360 折 Q² 重建最大差约 1.13e−15 |
| array transfer DC 错误 | 新增 array_transfer_v2 的有限复指数求和；旧 API 仅历史复现使用 | DC/整数 kh 返回 N²，单/双线与近共振测试通过 |

## 正式科学结果

32 相位下组级 TV_cond（越低越好）：

| 模型 | 旧 formal | r1 | r1 判定 |
|---|---:|---:|---|
| L1 | 0.343750 | 0.343750 | 基线 |
| L2 | 0.357143 | 0.357143 | not_achieved |
| L3a | 0.571429 | 0.343750 | not_achieved，未优于基线 |
| L3b | 0.285714 | 0.428571 | physical_invalid，1/7 留出失败 |

L3b 的失败不是修复后继续漏检：该条件被留出时，训练可行候选仍可能在新条件失败，修正版现在识别并保留此失败，按 invalid class 计分。不能利用测试条件再挑选 γ 使结果变好。每一折的 γ 并列最优集合都跨正负号，选出的负值来自既定 tie-break，不能解释为负相互作用符号被识别。16/32/64 相位均无有效的模型改进证据。

Task24 的原 360 行折级 Q² 与当前结果差为 0；D/Sq/ILR/A2/entropy 的主统计结论保持。新图按照 D→A→P→T 排列，并补 src/proc hatch contribution 图。

## 验收

- 全量 unittest：247/247 通过，含新增 11 项反例和合同测试。日志见 unittest.log。
- verify_results.py：PASS，见 acceptance_checks.json。核对 OOF 唯一性、逐折覆盖、R²、Q² 重建、历史数值一致、训练/留出隔离、失败计分、真实反例与相位敏感性。
- 修正前后旧 outputs/phase2_7 与 outputs/phase2_8 共 33 个文件 SHA256 一致，见 frozen_integrity.json。未重写历史 formal 结果。
- 完成 Task24 formal、Task25 quick 与 formal 全网格；不依赖 quick 结果作正式判定。

## 后续边界

README 已把新增 repeatability matrix 改为 future work，当前路线为不新增实验的论文统计收敛。Route P effect maps、统一 error atlas、P/T residual/conditional analyses、PDE solver benchmark 与 morphology-by-design 均是待实际执行的研究任务；本次只补齐其统一 OOF 基础并登记执行和声明边界，不称已通过物理应用验证。详见任务说明/JMPT_无新增实验路线_20260905.md。

修正版没有宣称可投稿或独立 confirmation；现有 200 ROI 仍是经过多轮探索的 discovery 数据。
