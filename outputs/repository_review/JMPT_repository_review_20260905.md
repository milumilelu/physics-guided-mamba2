# 仓库审查：聊天记录路线与当前实现对照

日期：2026-09-05。审查基准：本地工作区；HEAD cfa5542ffef41fb2c9f0d098fd150c0f21757f40。审查两份用户粘贴记录、README、Phase 2.7/2.8 契约、共享实现、formal CSV/JSON 及测试。未修改科研代码或冻结结果；本报告不评定期刊录用概率，未核查聊天记录引用的外部文献。

## 结论

统计主线已推进到 2.8 formal，旧聊天的“HEAD 8c1e039 / 2.8 草案 / 2.7r2 待办”已过时。现有数据支持在所考察模型、DOE 和 8–16 µm 带内，方向组织具有较强 hatch 预测贡献、完整谱组成保留其他参数的预测贡献。当前还不能升级为已验证的独立设计通道、微观机制或 physics-informed inverse design。

2.8B 存在需要修正并重新评估的实现/契约问题。建议保留历史产物，新增版本修正；不要为了获得正结果修改门槛。

## 按优先级排列的发现

### P1：L3b 物理约束漏掉烧蚀区域中的负去除深度

位置：experiments/phase2_8/25_kernel_bridge.py:322–329；src/forward_models.py:155–163。

当前只对 base <= 0 的像素检查 z-base；base > 0 的区域不检查 z 是否变负。按实际 81-line library、7 个 usable 条件、formal 的 32 个相位与已选 gamma=-0.5 复算，在 dataset_index=177 的正 base 区域中 min(z)=-4.808000044 µm。它不是基底噪声负谷。dataset_index=164 还出现 -0.145965454 µm（主要位于非正 base 区域）。当前 JSON 的 L3b 排除列表为空。

一个独立正值 Gaussian toy 同样复现：min(base)=17.9544，min(z)=-44.0190 µm，base<=0 像素为空，当前检查直接通过。因此这不是仅文档不一致。不能把现有 L3b 正向变化解释为已经符合物理限制的相邻轨竞争。

建议：新版本明确基底噪声容差与正去除区域中的符号/幅度约束；训练候选和留出预测均登记物理有效性，失败不可静默删样本。重新计算 L3b 及门槛。正文把冻结绝对判据改为校正判据的附录确实存在，但该修订并不能解决上述漏检。

### P1：2.8 L3a 的相位积分只覆盖半个 period-2 周期

位置：experiments/phase2_8/25_kernel_bridge.py:245–246、258、312；对照 experiments/phase2_7/23_single_track_envelope.py:201–203、331–334。

2.8 对所有模型统一 phi=j*h/n_phases。L3a 的 a_n=1+c*(-1)^n 在 c!=0 时具有 2h 的空间周期；2.7 对 period-2 明确使用 2h 相位区间。2.8 因而没有完整边缘化奇偶相位，且不能作为相同相位协议的 legacy continuity control。

建议：按模型周期定义相位区间，训练与测试同改；复算 L3a，并报告 16/32/64 相位敏感性。影响大小尚未重跑全网格量化。

### P2：Task 25 参数选择的 mean 与冻结 median 不一致

位置：experiments/phase2_8/25_kernel_bridge.py:332、337；研究任务说明:251；落地细则:16。

两份冻结说明要求 median TV_cond 最小、并列取小值；代码使用训练 rows 的 mean。mean 与 median 在当前仅 6 个训练条件时可以选择不同参数。未发现针对这项改变的明确勘误；不能仅凭已输出 formal 标签认定合同一致。

建议：先明确新版本估计目标；保留当前 mean 结果并列出偏离，而不是静默重写历史。重新评价时记录每个候选得分与并列最优集合。所有折选择 gamma=-0.5 是边界解，而 tie 取数值小值本身偏向负边界，单看参数一致性不能证明负号可辨识。

### P2：README 对 phase realization 作出超出数据的结论

位置：README.md:38；24_information_decomposition.py:317–332；gsl28_a_evaluation.json 的 realization_diagnostic。

README 写“波峰落点不受工艺控制（描述性）”。实际完全相同五参数的独立来源重复仅 1 对，距离处随机配对第 32.43 百分位。另 11 个 same_condition_key 对并未要求 hatch 相同，不能统称同工艺重复。一次重复对加随机配对分位不能支持“工艺不控制”的一般性否定结论。

建议改为：现有单对 exact repeat 未显示明显更相似的 phase realization；重复样本不足，无法判断工艺对 realization 的可控程度。保留探索性诊断，不据此推导不可约随机性。

### P2：统一统计产物尚缺已承诺对照列及新路线所需样本级 OOF

位置：24_information_decomposition.py:241–246；研究任务说明:140；落地细则:13。

fold CSV 实际只有 target, variant, model, fold, skill_q2, alpha, n_train, n_test，没有契约承诺的标量 sklearn R² 对照列。run_cv 虽计算 pred，却只保留折分数，不写样本级预测。已有 Phase 2.5 composition OOF 不能替代 2.8 全通道统一 OOF；缺少这些产物，后续 unified error atlas、P/T cross-fitted residual coupling 需要重新跑模型。

建议新增带 dataset_index、split、target、model、observed、predicted、train-null 的 OOF 表与 R² 对照；保持 Q² 为 primary。summary 的 skill 为各折 median，delta_h 为逐折差的 median，两者不能通过“两个 median 相减”互相替代。

### P2：array_transfer 在 DC 返回错误结果

位置：src/forward_models.py:166–173。

实测 array_transfer([0],4,20) 返回 [0]；直接定义 |sum(exp(-i*2*pi*k*n*h))|² 在 k=0 应为 20²=400。分母添加 1e-300 不能处理 Dirichlet 核的可去奇点。

当前 Task25 筛掉 DC，因此未证实这改变现有 gate，但公共 API 在零频/共振附近不稳，会影响后续谱传递模型复用。建议用稳定 sinc 比值或有限复指数求和，验证 k=0 和整数 k*h。旧 frozen 语义按仓库版本化约定保留。

## 已完成，不应再列作原始阻塞项

- 2.7r2 已落地：weighted TV、h×session 内 DOE unit bootstrap、own-profile q、稳定区 flags、16/32/64 敏感性、陈旧 simulation CSV 清理及 formal-contract tests。formal TV_C=0.615135，TV_P2_LOHO=0.529043，delta=0.086092，MODEL_INADEQUATE。相位敏感性列是 c_global 版本，不应混为 LOHO 主判据。
- 统一 D/A/P/T Ridge 谱表已完成；D 与 Sq 分类已修正；“五个正交分量”已移除。
- canonical src 迁移已经完成，旧“不要先做大重构”不再是当前执行待办。已有回归报告记载 Task22/23 的 7 个产物 EXACT、冻结文件未变；本次未重新执行整个 golden pipeline。
- 功率已按仓库记录升级为用户确认物镜后独立实测值，仪器/日期缺失仍有登记。不能沿用旧聊天说它仍只能是 proxy；亦不能把仓库登记视为本次独立测量验证。f 与 pulse energy 的耦合限制保留。

## 当前统一结果（各折 Q² 中位数；delta 为 paired-fold 中位数）

| target | src full | src delta_h | proc full | proc delta_h |
|---|---:|---:|---:|---:|
| D | 0.552 | 0.090 | 0.577 | 0.108 |
| Sq | 0.160 | 0.018 | 0.157 | 0.017 |
| 完整 ILR composition | 0.308 | 0.175 | 0.339 | 0.199 |
| A2 8–16 | 0.641 | 0.636 | 0.552 | 0.623 |
| angular entropy 8–16 | 0.662 | 0.644 | 0.645 | 0.631 |

这些数值与所讨论的方向/组成预测贡献不对称一致；尚无跨通道不对称的正式不确定性评估、统一 P/T 非冗余分析或逆向设计验证。G28-A VALID 是协议完成/QA 标签，不是聊天中建议的全部科学假设均已通过。

## 与新论文路线的差距

1. Task24：主体已完成；补统一样本级 OOF、标量 R² 对照、各 ILR/scalar 解释性输出、统一 spline/ET sensitivity 和 hatch contribution 图。
2. Route P effect laws：未找到聊天要求的 cross-fitted ILR response/ALE effect maps。现有 process_map 主要是 CV、模型比较和 importance，不能视为该任务完成。尤其 N=4→5 与 session 混杂、f 与 E_p 耦合，effect 应明确为预测关联。
3. Error atlas：Phase2.5 版本存在，但未见 D/Sq/ILR/A2/entropy 在同一 2.8 OOF 协议下统一实现。
4. P/T separability：未见新路线要求的双向 conditional additions 与 cross-fitted residual coupling。不同 hatch delta 不等于已证明可独立设定两个目标。
5. Physics-informed application：无 Phase2.9/PINN PDE solver、独立数值 benchmark、单轨 held-out calibration/validation、dense virtual DOE、多目标设计和 retrospective evaluation 资产。现有 measured-kernel bridge 与独立虚拟增强脚本均不能替代这些层级。
6. 当前 README/冻结路线仍把新增 repeatability matrix 作为下一阶段，和聊天提出“不补实验”不一致。应另写版本化路线，保留新增实验方案为 future work。
7. 当前 200 条已用于多轮 discovery。现在再划留出集可做限制明确的 retrospective/internal validation，但不能消除此前对目标定义和模型选择的知情，不能称未见过的独立 confirmation。nearest existing 条件还需预定距离阈值并报告参数偏差。

## 建议执行顺序

先修 2.8B 约束、相位与估计目标契约，并降低 README 过强表述；继而补齐统一 OOF、effect maps、error atlas 和 P/T statistical separability；再冻结 effective-physics 的方程、参数来源、识别范围与 validation gate。solver / single-track held-out validation 未通过，则不进入正文 virtual design。同步整理 claim ledger，把 experimental/statistical/机制相容解释/假设分级。

不需要重做已经封账的 2.7r2，也不建议现在引入 Mamba 或全面再次重构。论文主线应先收敛于观察到的工艺—形貌预测结构；应用验证是否可进入正文，取决于实际结果。

## 本次验证记录

- .venv Python：Phase2.7 formal-contract 专项 7/7 通过。
- 独立诊断脚本执行成功：读取实际 profile library 并检查 7 条件 × 32 相位的 L3b 场；复现 dataset_index=177 的负去除深度。
- array_transfer DC 反例及实际 CSV 列检查执行成功。
- 全量 unittest discover 已尝试，长时间未完成且无输出，主动停止；本次不宣称全量测试通过。未重跑昂贵的全部 formal 模型网格，也未更改冻结产物。

## 后续处理（2026-09-05）

上述六项实现/契约/表述问题已在独立 2.8r1 中修正并重算。处理结果见 [修正验收报告](../phase2_8_r1/CORRECTION_REPORT.md)。旧报告与 formal 结果保留作历史记录；以 r1 当前结果为准。论文扩展分析单独列入无新增实验路线，不标为本次已完成。
