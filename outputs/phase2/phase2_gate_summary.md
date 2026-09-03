# Phase 2 gate summary（2B gate，2026-09-03，rev2 含审查修正）

> 状态：**PHASE2_GATE_CLOSED — 四条预设路线均未触发；登记新观察"谱组成受控 / 幅度未解析"；下一步优先级交给 repeatability matrix。**
> 依据：Phase 2A gate（`local_structure/phase2A_gate_answers.md`，CLOSED）→ 04–07 主 CV（5940 折行）→ 08 local probe（520 行，同 held-out 比较）→ 09 敏感性（6 臂 × 7 targets × {A,R} × {ridge,extratrees} × 5 折，含 gate 要求的 exclude-artifact-yes 臂）→ 03 阈值扰动（P5/P95、P10/P90、P15/P85 within-null）。
> rev2 修正（2026-09-03 外部审查）：sentinel 百分位单位、Route N 计数 3/7、local probe 措辞与正向格子、R 集合定义（reduced derived feature set，非 lossless reparameterization）、E_frac 稳健性措辞。
> 所有 R² 均为 exploratory cross-validated explainability estimate (n=200)（细则 §18）；所有路线判据为预设方向性阈值，非通过/失败判据（细则 §17）。

## 0. 路线判读（细则 §17，四条预设路线均为"未触发"）

| 路线 | 预设触发条件 | 实测 | 判定 |
|---|---|---|---|
| **N** nonlinear mapping priority | 折配对 ΔR²(ET−Ridge) ≥ +0.1 的 primary target ≥ 半数（≥4/7） | **3/7**：depth +0.365 [5/5 折]、E_8–16_frac **+0.193 [5/5]**、8–16 RMS +0.154 [3/5]；32–64 −0.154 [0/5]、≥64 −0.138 [0/5] ET 反而更差 | **未触发**（3/7 < 4/7） |
| **S** scale-dependent predictability | 形貌口径 R²(≥64)−R²(8–16) ≥ +0.2 且跨 09 臂方向一致 | Ridge −0.01 / ET −0.19；DoG 臂同向（Ridge：≥64 0.100 < 8–16 0.151，仍反向） | **未触发** |
| **H** local heterogeneity / regime candidate | 某层 ΔSkill_local−global ≥ +0.10、跨折一致、Dummy 不解释、层内跨 session 重复 | 总体中位为负（ET −0.064、Ridge −0.048）；但存在个别正向格子（见 §2） | **未触发**（方向不一致、局部训练集仅 ~12 行） |
| **U** unresolved branching / replicate-needed | within-null 下 Type II p ≤ 0.01 且 T_λ 一致，且过 P15/P85 扰动 | 2A：12 组 Type II p>0.17、formal-only 全不显著；扰动后 Type II p ≥ 0.21；唯一 T_λ 单点（phys 16–32：p=0.011@P10、0.057@P5、0.027@P15）不跨 null 复现 | **未触发** |

## 0b. 登记新观察（第五条，不属于预设 Route）

$$\boxed{\text{spectral-allocation control / amplitude-unresolved（specific-scale selectivity at 8–16 µm）}}$$

- 工艺对"residual 能量分配到哪个空间尺度"有可泛化控制（E_8–16_frac R² 0.42–0.63），对"总幅度有多大"（Sq ≤0.08）与"具体空间 realization"（band PC ≤0.07）没有。
- 这是 **specific-scale selectivity**，不是 Route S 那种"波长越大越好预测"的单调关系。
- 机制线索（描述性，不赋物理）：E_λ ≈ (RMS_λ/Sq)²，即 RMS_λ = Sq·√E_λ——E_8–16 可预测而 Sq 不可预测时，RMS_8–16 必然难预测；两者不矛盾，反而互相印证。
- family-D 的失败（跨折 PC1–3 对齐 47–84°、全部 PC R² ≤0.07）独立支持同一结论："能量落在哪个尺度"是稳定 observable，"具体长成哪个模态/相位"不是。与盲评看到的 periodic stripe 家族呼应，但 E_frac 不含方向信息，不能直接说工艺控制 stripe——directional PSD 是 Phase 2.5 的连接点。

## 1. 细则 §29 十二问逐条回答

1. **高 leverage 是否主要由 artifact 引起？** 否（2A gate PASS_WITH_FLAGS：yes=3/uncertain=9/no=16）。09 exclude-artifact-yes 臂：剔除 3 个 yes 样本后多数 band-RMS R² **上升**（ET input A：8–16 +0.11、32–64 +0.13；input R：+0.13/+0.17）——可预测性不是 artifact 伪影，suspect 样本反而在稀释它。
2. **是否存在 process-near/morphology-far 的真实候选？** 个案存在、总体无过量（2A Q4；阈值扰动后 Type II p 仍 ≥0.21）。注意：R 集合是有损压缩（(N,h)→N/(vh)），可能人为制造 process-near——phys 空间那个 16–32 T_λ 单点信号因此要更加谨慎对待。
3. **exact-repeat 49/50 处于什么量级？（rev2 修正）** `sentinel_pct_of_ordinary` 是**百分位（0–100 单位，`np.mean(d_ord < sent_d)×100`）**：total **P0.11**、8–16 **P0.38**、16–32 **P0.89**、32–64 **P1.46**、≥64 **P0.25**——全部低于 ordinary 对的 P5。**49/50 在 total 与全部四个带上都异常接近**（v2 §13 的 total 0.287 µm 快照与此一致）。修正后的含义：至少在这个登记工艺条件下，多尺度 residual morphology 的重复性非常好；真正未知的是 **σ_repeat(u,λ) 是否随工艺条件剧烈变化**——这正是 repeatability matrix 要回答的问题（动机由此更新，必要性不变）。
4. **哪个空间尺度最容易被工艺预测？** 没有哪个波段的**绝对幅度**可稳定预测（RMS ≤0.24 且 formal-only 塌到 ≈0）。可预测的是**谱组成**：E_8–16_frac 0.42–0.63（unseen source）/ 0.40–0.42（unseen process，Ridge/ET 0.398/0.424）。
5. **哪个最难预测？** Sq（≤0.08）、各带 PC 模态（≤0.07）、pit_density（≤0）。
6. **R 集合是否改善泛化？（rev2 改口径）** R 是 **physics-motivated reduced derived feature set**（有损：E_p↔f 双射，但 (N,h) 被压成 N/(vh) 一个自由度；I(g(u);Y) ≤ I(u;Y)）。实测 E_8–16_frac：A 0.626/0.418（ET/Ridge）≫ R 0.366/0.036——单独使用 R 丢失 (N,h) 区分，**不是** A 的等价坐标系；而 C=A+derived 0.639/0.452 略优于 A——derived 组合作为**补充**对简单模型略有 inductive-bias 价值。后续信息性对比是 **A vs C**，不是 A vs R。ET 上 R 对 Sq +0.10、16–32 +0.18、32–64 +0.20（4/5 折）的增益也应在 C 口径下复核。
7. **非线性是否显著优于线性？** 3/7 primary target 达 +0.1（depth、E_frac、8–16 RMS），其中两个 5/5 折同号；但 32–64/≥64 上 ET 更差。n=200 不足以支撑"非线性映射优先"。
8. **local/regime 模型是否有稳定优势？** **在被测的启发式分层（depth/Sq 四分位、consensus 中位）下没有稳定优势**：总体中位 delta_skill 为负；但存在个别正向格子——最大 depth Q3 × ≥64 µm RMS（ridge）中位 **+0.478 [4/5 折]**、depth Q3 × 32–64（ET）+0.234 [3/5]、sq_q4 Q4 × ≥64（ET）+0.124 [3/5] 等。它们方向不一致（同层其他 target 为负）、局部训练集仅 ~12 行，且**盲评的 phenotype 家族（periodic stripe / long-wave multi-lobe）从未被直接用作分层**——因此只能说"启发式分层下无稳定优势"，depth Q3 × 粗带格子登记为候选（与盲评长波家族呼应），留待 phenotype 分层复检。
9. **session effect 是否影响结论？** 分目标：depth、Sq、**E_8–16_frac**（formal-only 0.633/0.443，stable）不受影响；**band RMS 绝对幅度受影响大**（formal-only 8–16：ET −0.21 / Ridge −0.16，strong）——band 幅度可预测性部分依赖 pass/supplement 跨 session 结构，引用必须带此警告。
10. **raw/repaired 是否影响结论？** 不影响：repaired 臂全部 |ΔR²| ≤ 0.03（stable）。
11. **更支持 deterministic nonlinear / scale-dependent / regime-specific / hidden-variable 中的哪一种？** 都不是——四条预设路线全部未触发。可复现事实收缩为：工艺可泛化控制谱能量分布（8–16 份额），不控制绝对幅度与空间 realization。
12. **下一轮实验最值得增加什么？** **repeatability matrix（v2 §25.1）优先级提升（动机 rev2 更新）**：49/50 已证明该条件下多尺度重复性极好，所以 Sq/band-RMS 近零 R² 的两种解释——(a) 工艺不控制幅度、(b) σ_repeat(u,λ) 强烈依赖条件——只能靠 6–8 个条件 × 3–5 exact repeats 区分（覆盖 ordinary / periodic-stripe 高 E8 / long-wave 高 LOCO / intermediate）。若可行，同槽 N=1–4 纵向测量（v2 §25.2）价值更高。

## 2. E_8–16_frac 的稳健性（措辞 rev2 校准）

按 09 的 |ΔR²| 分桶，该目标并非每个格子都是 literal "stable"（exclude-artifact ET −0.053、minus-top5 ET −0.100 为 moderate），准确表述是：**signal survives every tested sensitivity arm and remains qualitatively robust**——formal-only 0.633/0.443、exclude-artifact 0.574/0.408、minus-top5 0.526/0.401，无任何一臂将其消灭或反向。

## 3. 对 Phase 3 的建议

按规划 §24：**当前不走 3A/3B/3C/3D 任何建模路线**。Phase 2 把问题收缩为："为什么工艺能稳定控制 residual 的尺度组成，却不能稳定决定总幅度与空间 realization？" 建议顺序：**Phase 2.5（小，纯现有数据）**——谱组成 compositional 分析（补 E_<8 = 1−ΣE_λ≥8，ILR 坐标 grouped CV，回答"抬升 8–16 份额"还是"从哪些尺度系统性转移"；加 directional PSD/anisotropy 连接 periodic-stripe 家族）→ **repeatability matrix**（区分幅度不可控 vs 条件依赖噪声底）→ 之后再议 Phase 3。

