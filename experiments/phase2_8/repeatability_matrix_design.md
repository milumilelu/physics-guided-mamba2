# Repeatability Matrix 设计（第一批新实验）— DRAFT_FOR_REVIEW

> 状态：设计登记（Phase 2.8 v2.1 §5.2/§5.4；WP4 产物）。Phase 2.8 保持 discovery-only，本设计不进入任何 2.8 分析。
> 日期：2026-09-05。上位文件：`任务说明/Phase2.8_...md` v2.1 FROZEN §5。

## 0. 为什么第一批做 repeatability 而不是 confirmation

当前最大未知是 **Var(H|u) 到底多大**，不是 L2/L3 哪个模型好。若 σ_repeat(u) 与当前 OOF 误差同量级，则"低可预测性 = 随机性"假设成立，机制 confirmation 的样本量设计与模型改进都要重新定价；若 σ_repeat(u) 远小于 OOF 误差，则 missing variables 是主因。这一批直接回答：**低可预测性是随机性，还是 missing variables**——全项目当前最值钱的新实验。

## 1. 设计方案 A（首选）：纯 repeatability matrix

- **6–8 conditions × 3–5 independent repeats**（建议 8×4 = 32 次加工/测量，与混合方案同预算）。
- 每个 condition 一次装卡内加工 1 个矩形 + 1 条单线（同 batch 同时给 2.8B 的 kernel 实测提供新条件下的 g(x)）。
- repeats 独立 = 不同装卡/不同时间/重新对焦，间隔随机化以打散时间漂移。

### 1.1 Condition 选取判据（从 discovery 200 中选，选择过程必须留痕）

| # | 判据 | 目的 |
|---|---|---|
| 1 | Route T（A2_8_16）预测最高 condition | 覆盖 h 主导端 |
| 2 | Route T 预测最低 condition | 覆盖低可预测端 |
| 3 | Route P（P_λ）预测最高 condition | 多因素端 |
| 4 | Route P 预测最低 condition | |
| 5 | 高 Sq condition | 幅度端 |
| 6 | 低 Sq condition | |
| 7 | error hotspot condition（OOF 误差最高，Phase 2.5 error_atlas） | 直接测 hotspot 是否为随机性 |
| 8 | stripe phenotype / ordinary morphology 各一（目视型态对照） | 型态定性覆盖 |

（条件间保持 u 尽量分散；同一 (τ,f,N,v) 只出现一次。）

### 1.2 分析（预注册，测量前冻结）

- 每通道重复方差 σ²_repeat(u) vs discovery OOF MSE 对比：**repeatability ratio** ρ(u) = σ²_repeat / MSE_OOF。
- 判读（预注册）：ρ ≥ ~0.5 → 随机性主导；ρ ≤ ~0.2 → missing variables 主导；中间 → 混合，列出候选缺失变量。
- 全部通道（D/A/P_λ/O_θ）+ 2.8B 观测量（λ_peak class 的重复稳定性、单线 W50/g(x) 的重复性）同时报告。
- 本批数据**不得**用于任何模型选择/阈值调整（discovery-only 边界延续）。

## 2. 设计方案 B（预算只允许一批时）：混合设计

- 6 anchor conditions × 3 repeats = 18（估 σ_repeat(u)）；
- + 14 个新 confirmation conditions（预注册预测后揭盲）；
- 总 32 次加工/测量；
- **两部分分析预先分开**：anchor 部分只做重复性，confirmation 部分只做锁定预测评估，不得交叉选样。

## 3. 第二批（2.8B 完成后）：mechanism confirmation set

- 20–30 个新 unique conditions；
- 预注册预测：P_λ、O_θ、λ/h class、L1/L2/L3 相对排序、overlap descriptor 方向效应；
- 经 `src/confirmation.py` 的 fit → predict → **write_lock**（测量前落盘）→ 一次性 `evaluate_locked_predictions` 揭盲。

## 4. 工程与记录要求

- 每次加工记录：u 五元组 + 装卡 ID + 时间戳 + 操作者 + 功率实测（P_obj 进 `POWER_REGISTRY` 口径）；
- 测量与 discovery 相同管线（80×80 µm @ 0.5 µm/px；单线 0.278657 µm/px）；
- 数据入库后新增 `session_role = confirmation_batch1`（或 repeatability/mixed），discovery 200 不变。

## 5. 决策点

| 项 | 待定 | 建议默认 |
|---|---|---|
| 方案 A vs B | 预算 | A（若一批预算 ≥ 32） |
| repeats 数 | 装卡成本 | 4 |
| 条件数 | 覆盖 vs 重复 | 8 |
| 揭盲规则 | — | 锁定预测测量前落盘，SHA256 登记，单次评估 |
