# Phase 2.8 gate 评估（Task 24 + Task 25 formal）

> 状态：**formal 完成（2026-09-05）**。上位协议：`任务说明/Phase2.8_...md` **v2.1 FROZEN** + `experiments/phase2_8/Phase2.8_落地执行细则.md` FROZEN。运行环境 `.venv`（sklearn 1.7.2）。WP1 结构收敛（8 模块 + golden 回归 7/7 EXACT）完成后执行。

---

## 1. Task 24（2.8A）— G28-A = **VALID**

九条件全过（`summary/gsl28_a_evaluation.json`）：common intersection n=200；folds artifact SHA256 入档（src_gkf=GroupKFold(shared_height_source_id)/proc_gkf=GroupKFold(cv_process_group)，双 GKF 历史）；preprocessing fold-internal；α 用 target-native scorer（标量 MSE / Aitchison Q² / joint 标准化 multi-MSE）；dummy |Q²|<1e-9；三模型×全 target 完成；raw/repaired N/A 已登记（矩形只有一个注册高度场，200 行验证 max diff 3.8e-6 µm）；coverage 200/200；无历史分数拼接。

### 1.1 Predictability Spectrum（src_gkf，Q² train-mean null，逐折中位）

| target | M_full | M_h | M_-h | Δ_h | 折方向 |
|---|---|---|---|---|---|
| O_θ:entropy | **0.662** | 0.582 | 0.015 | **0.644** | 5/5 |
| O_θ joint (std, sec.) | 0.661 | 0.598 | 0.012 | 0.645 | 5/5 |
| O_θ:A2 | 0.641 | 0.615 | 0.005 | **0.637** | 5/5 |
| D（深度） | **0.552** | 0.044 | 0.501 | 0.090 | 5/5 |
| P_λ（Aitchison） | 0.308 | 0.145 | 0.145 | **0.175** | 5/5 |
| A（幅度） | 0.161 | 0.001 | 0.142 | 0.018 | 3/5 |
| O_θ:A2 (16–32 带， sens.) | 0.143 | 0.168 | 0.002 | 0.141 | 5/5 |

proc_gkf / in-box 101 敏感性同向（in-box P_λ full 0.363，复现 2.7 in-box 量级；数字见 `predictability_spectrum.csv`）。

**发现（描述性，附逐折方向证据；纵轴是 cross-validated normalized predictive skill，不是信息量）**：

1. **层级分化存在且方向一致**：方向组织（O_θ，0.64–0.66）≫ 尺度组成（P_λ，0.31）≫ 幅度（A，0.16）；Δ_h 的层级（O_θ ≈ 0.64 ≫ P_λ 0.175 ≫ A 0.018）与 Route T/P 的既有结论在统一协议下复现。
2. **新结果：深度 D 可预测（0.55）但几乎不由 hatch 承载**（M_-h 仍 0.501，Δ_h 仅 0.090）——去除深度由 τ/f/N/v 共同控制。这是统一协议下第一次把 D 与 P(λ)/O_θ 放在同一协议里比较得到的新分层事实。
3. **h 主导严格限于 8–16 µm 基频带**：16–32 带 O_θ 技能塌缩到 ~0.14。
4. **数据发现（登记）**：phase-1 manifest 的 D/A 统计早于 v1 场配准（200/200 行均匀 rel~3e-5 漂移，corr=1.0）。通道 D/A 改由冻结场重算（与谱通道同源）；manifest 旧列保留审计。建议 Phase 3 前统一重建 manifest 统计列。

### 1.2 realization diagnostic（φ，描述性，不入 Gate）

exact-repeat 对（dataset 48/49，唯一同 u 对）d_φ=0.982，处 random-ordinary 对的第 **32 百分位**；same-condition-key 对均值 d_φ=0.987 vs ordinary 0.983。**同工艺不产生更可复现的 Fourier-phase realization**——波峰落点不受工艺控制的描述性证据，与层级框架的 φ 层语义一致。

---

## 2. Task 25（2.8B）— G28-B1 三比较均未达成；G28-B2 双轴分化显著

Population：**19 候选条件 → 13 条件有线匹配（=2.7r1 3A 的 13）→ 7 可用行 / 7 kernel groups**（kernel 库由 2.7r2 修正后的共享提取路径重建——`src.data.build_line_profile_library`：plateau membership FLAGS + 视场外 0 深度；本表数字为 r2-kernel 重算值，与 r1-kernel 首算的差异仅 L2 微移 0.353→0.357，其余各级不变）（6 行因 λ_peak 不可观测合法剔除，逐行登记；无 profile_unsuitable——81/81 线 profile suitable）。子集观察类别 = 5×m1 + 2×m2，全部 h∈{6,8,10}。n=7 是登记在案的硬限制。

### 2.1 主指标 TV_cond（out-of-group，逐 condition-matched）

| level | TV_cond | ΔTV_cond vs L1 | 95% CI（kernel-group paired） | B1 |
|---|---|---|---|---|
| L0 kernel-only | 0.857 | — | — | — |
| L1 linear | 0.344 | — | — | — |
| L2 饱和 | 0.357 | −0.013 | [−0.040, 0.000] | not achieved |
| L3a alternating | 0.571 | −0.228 | [−0.563, 0.000] | not achieved |
| **L3b pairwise** | **0.286** | **+0.058** | **[0.000, 0.138]** | **not achieved**（CI 下界 = 0，未严格 > 0；Bonferroni 98.33% 下界亦 0） |

参数（LOGO_kernel，训练组 only）：L2 D_sat\* = 45.25 µm（全部组——近似线性，与 L2≈L1 一致）；L3a c\* ∈ {0（5 组）, 0.2（2 组）}；L3b γ\* = −0.5 µm⁻¹（全部组，网格负端点）。guard（校正量判据，tol=0.01 µm）：L2 逐组剔 1 个（小 D_sat）、L3a 逐组剔 4 个（大 c）、L3b 无剔除；候选永不 clip。

**登记（多重比较）**：三个 B1 比较为同一 model-family exploration，CI 不作独立 confirmatory 解释。**唯一正向信号是 L3b**（邻轨 overlap cross-term，负 γ = 相邻轨去除相互竞争）：Δ=0.058 达到 0.05 门槛但 bootstrap CI 下界为 0——在 n=7 下"未达成"是功效问题而非否定证据。L3b 进入 Phase 3 mechanism confirmation set 的候选清单；L3a 在本子集上明确反向（period-2 有害）。

### 2.2 G28-B2（pooled-TV legacy adequacy reference）

| level | TV_pooled | legacy 分级 |
|---|---|---|
| L0 | 0.714 | MODEL_INADEQUATE（legacy reference） |
| L1 | 0.286 | partial（legacy reference） |
| L2 | 0.286 | partial（legacy reference） |
| **L3a** | **0.000** | strong reproduction（legacy reference） |
| L3b | 0.286 | partial（legacy reference） |

**方法论发现（本 formal 最重要的一课）**：L3a 的 pooled TV = 0（聚合类别分布与观测完全一致）而 TV_cond 恶化到 0.571——聚合指标在 condition-specific 预测任务上可以完全失真（A/B 互换不惩罚）。这是外审 F5 把主指标改为 condition-matched TV 的实证演示；pooled TV 自此只作 2.7 连续性参照，不作 adequacy 主张。

### 2.3 描述性（不进 Gate）

Spearman(O(h), observed class) = 0.00；Spearman(r_pred, r_observed) = −0.12（n=7，无关联证据）；L1→L2→L3b 单调性 = 0.344 → 0.357 → 0.286（L3b 单调改善，L2 平）。

---

## 3. 实现级修订登记（formal 结果产生之前冻结）

1. **F6 guard 语义细化**：绝对 total-field min(z)≥−tol 判据数学上不可行（实测 profile 含 ~0.1 µm 噪声负谷；F(s)<s 对 s<0 严格成立，任何有限参数都会被排除）。冻结为**校正量判据**：min_{s≤0}[z−s] ≥ −tol，tol=0.01 µm。→ 建议 v2.1 勘误注记。
2. **kernel profile NaN 净化**：横向全段视场外位置固定 0 深度（与 synth_field left/right=0 同约定）。
3. `forward_model_simulation.csv` 为冻结目录旧版遗留，现行 Task 23 不产出（golden 回归目标以现行代码为准）。

## 4. 下一步（承接 v2.1 §5）

1. 第一批新实验 = **repeatability matrix**（设计见 `experiments/phase2_8/repeatability_matrix_design.md`；混合方案备选）；
2. 2.8B 后冻结 mechanism confirmation set（20–30 conditions），经 `src/confirmation.py` 锁定流程揭盲，L3b（负 γ cross-term）进入预注册候选；
3. manifest D/A 统计列与 v1 配准场的 3e-5 漂移建议在 Phase 3 前统一重建；
4. 方向 provenance / 双人盲标 / discovery-confirmation 分离三风险继续开放。
