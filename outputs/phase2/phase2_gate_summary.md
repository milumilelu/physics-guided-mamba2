# Phase 2 gate summary（2B gate，2026-09-03）

> 状态：**PHASE2_GATE_CLOSED — 无预设路线触发；结论为"谱组成可预测、幅度不可预测"，下一步优先级交给 repeatability matrix。**
> 依据：Phase 2A gate（`local_structure/phase2A_gate_answers.md`，CLOSED）→ 04–07 主 CV（5940 折行）→ 08 local probe（520 行，同 held-out 比较）→ 09 敏感性（6 臂 × 7 targets × {A,R} × {ridge,extratrees} × 5 折，含 gate 要求的 exclude-artifact-yes 臂）→ 03 阈值扰动（P5/P95、P10/P90、P15/P85 within-null）。
> 所有 R² 均为 exploratory cross-validated explainability estimate (n=200)（细则 §18）；所有路线判据为预设方向性阈值，非通过/失败判据（细则 §17）。

## 0. 路线判读（细则 §17，全部为"未触发"）

| 路线 | 预设触发条件 | 实测 | 判定 |
|---|---|---|---|
| **N** nonlinear mapping priority | ΔR²(ET−Ridge) ≥ +0.1 的 primary target ≥ 半数 | **2/7**（depth +0.37 [5/5 折]、8–16 RMS +0.15 [3/5]；32–64 −0.15 [0/5]、≥64 −0.14 [0/5] ET 反而更差） | **未触发** |
| **S** scale-dependent predictability | 形貌口径 R²(≥64)−R²(8–16) ≥ +0.2 且跨 09 三臂方向一致 | Ridge −0.01 / ET −0.19；DoG 臂同向（Ridge 0.100 vs 0.151，仍反向） | **未触发** |
| **H** local heterogeneity / regime candidate | 同 held-out 批上 Skill_local − Skill_global ≥ +0.10 且跨折一致、跨 session 重复 | 08 几乎处处为负（ET 总中位 −0.064、Ridge −0.048；唯一的正格子 consensus_half×{Sq,32–64,8–16}≤+0.03） | **未触发** |
| **U** unresolved branching / replicate-needed | within-null 下 Type II p ≤ 0.01 且 T_λ 一致，且过 P15/P85 扰动 | 2A：全部 12 组 Type II p>0.17、formal-only 全不显著；扰动后 Type II p 仍 ≥0.21；唯一 T_λ 单点（phys 16–32，p=0.011@P10，0.057@P5，0.027@P15）不跨 null 复现 | **未触发** |

## 1. 细则 §29 十二问逐条回答

1. **高 leverage 是否主要由 artifact 引起？** 否（2A gate PASS_WITH_FLAGS：yes=3/uncertain=9/no=16；存在非 repair 高杠杆反例 #152/#167）。09 exclude-artifact-yes 臂显示剔除 3 个 yes 样本后多数 band-RMS R² **上升**（ET input A：8–16 +0.11、32–64 +0.13；input R：+0.13/+0.17）——可预测性不是 artifact 伪影， suspect 样本反而在稀释它。
2. **是否存在 process-near/morphology-far 的真实候选？** 个案存在、总体无过量（2A gate Q4；threshold perturbation 后仍如此）。
3. **exact-repeat 49/50 处于什么量级？** total 残差距离 0.287 µm，位于 ordinary 对的 **P11**；但带级并不特别近（8–16 P38、16–32 P89、32–64 P146、≥64 P25；`sentinel_multiscale_table.csv`）。单对、不构成 universal noise floor——且它自己就说明"总量接近"与"带级接近"不是一回事。
4. **哪个空间尺度最容易被工艺预测？** 没有哪一个**波段的绝对幅度**可被稳定预测（RMS 全部 ≤0.24，且 formal-only 下塌到 ≈0）。真正可预测的是**谱组成**：8–16 µm 能量分数 R² 0.42–0.63（Ridge/ET，input A）。
5. **哪个空间尺度最难预测？** 总幅度 Sq（≤0.08）、各带 PC 模态（family-D 全部 ≤0.07，且跨折 PC1–3 对齐 47–84°）、pit_density（≤0）。
6. **重参数化坐标是否改善泛化？** ET 上对 Sq +0.10、16–32 +0.18、32–64 +0.20（均 4/5 折同号），depth −0.13；CV-A/CV-B 差距多数 |gap|<0.1（E_8_16_frac ET +0.20 例外，说明该目标泛化部分依赖相邻 design condition）。定性为"更契合 inductive bias"的探索性效应（细则 §0.2），不写"物理信息增加"。
7. **非线性模型是否显著优于线性？** 仅 depth（+0.37，5/5）；形貌 target 上 ET ≤ Ridge 为主。n=200 不足以支撑"非线性映射优先"。
8. **local/regime 模型是否有稳定优势？** 没有（08：同 held-out 上 delta_skill 几乎处处为负）。盲评审定的"形貌家族"不构成可学习的局部 process→morphology 映射。
9. **session effect 是否影响结论？** 分目标：depth、Sq、**E_8_16_frac**（formal-only 0.63/0.44，stable）不受影响；**band RMS 绝对幅度受影响大**（formal-only 下 8–16 −0.21 ET / −0.16 Ridge，strong；其余 moderate）——band 幅度的弱可预测性部分依赖 pass/supplement 的跨 session 结构，必须带此警告引用。
10. **raw/repaired 是否影响结论？** 不影响：repaired 臂全部 |ΔR²| ≤ 0.03（stable）。
11. **当前数据更支持 deterministic nonlinear / scale-dependent / regime-specific / hidden-variable 中的哪一种？** **都不是**——四条预设路线全部未触发。数据定位出的可复现事实是：工艺对"谱能量分布"（8–16 µm 份额）有可泛化的控制（跨 formal-only/artifact/leverage/repaired 全部稳健），对"绝对形貌幅度"（Sq、带 RMS、带模态）没有可泛化控制。
12. **下一轮实验最值得增加什么？** **repeatability matrix（v2 §25.1）优先级提升**。理由：Sq/band-RMS 的近零 R² 存在两种不可分辨解释——(a) 工艺不控制它们，(b) 工艺控制但 exact-repeat 噪声底相对条件间散布过大；n=200 且仅 1 个 exact-repeat 无法区分。5–10 条件 × 3–5 重复即可把两者分开；若可行，同槽 N=1–4 纵向测量（v2 §25.2）价值更高。

## 2. 对 Phase 3 的建议（由数据决定，非由架构决定）

按规划 §24 的路线分叉：**当前不走 3A/3B/3C/3D 中的任何一条建模路线**。Phase 2 的产出把问题从"形貌很复杂"收缩为"复杂性来自哪里"的两个候选：(i) 谱组成受控（可泛化、稳健），(ii) 幅度类不可控或被重复性噪声淹没（不可分辨）。区分 (i)/(b) 的唯一途径是 v2 §25.1 的重复实验矩阵；在此之前不应开始任何 surrogate / multi-scale representation / regime / hidden-state 建模。

