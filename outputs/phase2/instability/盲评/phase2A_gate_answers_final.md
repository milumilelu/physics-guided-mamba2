# Phase 2A Gate 人工结论（视觉盲评 + unblind）

> Reviewer: GPT-5.6 Sol (visual audit)  
> Review set: 28 个 selected ROI（AUDIT-01...AUDIT-28），其中 AUDIT-27/28 为 exact-repeat sentinel。  
> 状态：**PHASE2A_GATE_CLOSED**。28 个 selected ROI 的 blind/unblind 人工审计已完成；`neighborhood_summary.csv` 已补齐 Type-II count、连续近邻统计量 `T_lambda`、within-session / global permutation null 及 formal-only 复核，因此四条 Phase 2A gate 均可正式作答。

## Review 摘要

- unblind artifact flag: **yes=3 / uncertain=9 / no=16**。
- 主要视觉表型计数（多标签，不能相加为 28）：
  - low-frequency waviness: 18
  - anisotropic texture: 11
  - periodic stripe: 6
  - edge contamination: 9
  - localized collapse: 8
  - multi-lobe morphology: 7
  - large pit: 5
  - repair-driven feature: 3
  - large-area dropout: 2

---

## 1. 高 leverage 是否主要由 artifact 驱动？

**人工结论：否，不是“主要由 artifact 驱动”；但存在一个不能忽略的 artifact-sensitive 子集。Phase 2B 不需要整体暂停，但必须保留 artifact sensitivity。**

证据：

- 28 个盲评样本揭盲后只有 3 个被明确标成 `yes`，另有 9 个因 ROI 边界/plane/部分 repair 保持 `uncertain`，其余 16 个没有可见 artifact 主导证据。
- global total LOCO 最高的一组并非都与 repair 同步：
  - `sample_065 / AUDIT-04`：global LOCO rank 1，repair 约 0.19%，plane RMSE 约 0.24 um；强阶跃无法由 repair 解释，但边界侵入仍需复核。
  - `sample_152 / AUDIT-24`：global LOCO 约 rank 5，repair=0；是明显的非 repair 高杠杆例子。
  - `sample_023 / AUDIT-16`：global LOCO 约 rank 7，repair 约 0.5%，主多叶瓣结构远大于 repair 区。
  - `sample_167 / AUDIT-14`：global LOCO 约 rank 9，repair=0，表现为周期条纹 + 长波梯度。
- 同时也确实存在明确 artifact-sensitive 高杠杆样本：`sample_037 / AUDIT-08`、`sample_082 / AUDIT-17` 的 repair 区成片并与极端结构空间重合；`sample_149 / AUDIT-13` 的 plane RMSE 约 0.78 um 且主结构贴边。

**Gate 决策：PASS_WITH_FLAGS。** 不回退整个 preprocessing；但 Phase 2B 必须至少报告 `all samples` 与 `exclude artifact=yes`（以及原细则规定的 raw/repaired、top-LOCO sensitivity）。`uncertain` 样本不应直接删除。

---

## 2. 高 leverage 是否集中在某类真实形貌结构？

**人工结论：是，存在清楚的形貌家族倾向，但目前不足以称为离散 processing regime。**

最明显的两类为：

1. **大尺度/长波、边界相连或多叶瓣的强形貌**：global LOCO 前列大量属于 `low-frequency waviness`、`localized collapse`、`multi-lobe morphology` / 边界阶跃类（例如 AUDIT-04、06、09、16、19、20、24）。
2. **规则方向性周期条纹**：AUDIT-02、12、14、23、25、26 形成视觉上高度一致的 `periodic stripe; anisotropic texture` 家族；其中若干是 DCT 8–16 的 scale-specific leverage 样本，而 global LOCO 可以很低。

这说明 **global leverage 与 scale-specific leverage 不是同一个问题**：global total 更容易被强长波/大尺度形貌支配，而 8–16 um band 可以突出规则条纹。

**Gate 决策：YES_AS_MORPHOLOGY_FAMILIES。** 后续可以做 local/regime probe，但名称保持描述性（如 R1/R2），不能在此阶段命名为热积累、相变、脆裂等物理机制。

---

## 3. 高 leverage 是否只是连续幅度尾部？

**人工结论：否。高 Sq/大振幅与 leverage 有明显共现，但不是充分条件，也不是唯一解释。空间组织与尺度组成同样重要。**

证据：

- 多个 global high-leverage 表面确实同时具有较大 Sq，例如 `sample_065`（Sq 约 12.6 um）、`sample_019`（约 9.1 um）、`sample_023`（约 6.8 um）、`sample_022`（约 6.5 um）。因此 amplitude tail 是重要因素。
- 但存在关键反例：`sample_152 / AUDIT-24` 的 global LOCO 约 rank 5，而 Sq 仅约 **0.75 um**、repair=0；它主要表现为长波空间梯度/局部坑，而不是极高总体振幅。
- `sample_110 / AUDIT-05` 和 `sample_098 / AUDIT-11` 的 large pit 很显眼，但 global LOCO 并不高，说明“局部坑很深/很显眼”也不自动等于 global leverage。
- 周期条纹样本可在 DCT 8–16 上高 leverage，但 total-field LOCO 很低（例如 AUDIT-26），再次说明 leverage 取决于观察尺度与空间组织。

**Gate 决策：NO_NOT_AMPLITUDE_ONLY。** Phase 2B 应继续按 spatial scale 分解 target，不应只用 Sq/depth 或一个全局 anomaly amplitude 来解释异质性。

---

## 4. process-near / morphology-far（Type II）是否真实存在？

**正式结论：作为单个候选 pair，Type II 存在；但当前数据没有证据表明它们在总体上显著多于置换 null。换言之，当前不支持“存在统计上过量的 process-near / morphology-far branching”。**

### 4.1 Type-II count 置换检验

执行细则的正式判据是：工艺距离位于 ordinary pair 的 P10 以内，同时 morphology distance 位于 P90 以上；再将观察到的 Type-II 数量与 1000 次置换 null 比较。

| process space | morphology metric | Type-II / near pairs | p(Type-II), within-session | p(Type-II), global | formal-only Type-II | formal-only p | T_lambda | p(T), within-session |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | D_morph_total_um | 115/1989 (5.8%) | 0.992 | 0.991 | 81 | 0.945 | 1.040 | 1.000 |
| raw | D_morph_DCT_8_16_um | 175/1989 (8.8%) | 0.825 | 0.778 | 123 | 0.778 | 0.553 | 0.999 |
| raw | D_morph_DCT_16_32_um | 129/1989 (6.5%) | 0.993 | 0.982 | 91 | 0.907 | 0.439 | 0.999 |
| raw | D_morph_DCT_32_64_um | 129/1989 (6.5%) | 0.986 | 0.982 | 95 | 0.943 | 0.306 | 1.000 |
| raw | D_morph_DCT_64_inf_um | 107/1989 (5.4%) | 0.999 | 0.995 | 72 | 0.958 | 0.451 | 1.000 |
| raw | D_morph_desc | 173/1989 (8.7%) | 0.827 | 0.790 | 38 | 0.807 | 5.201 | 0.999 |
| phys | D_morph_total_um | 146/1986 (7.4%) | 0.718 | 0.825 | 58 | 0.750 | 1.408 | 0.074 |
| phys | D_morph_DCT_8_16_um | 169/1986 (8.5%) | 0.437 | 0.716 | 85 | 0.590 | 0.659 | 0.101 |
| phys | D_morph_DCT_16_32_um | 196/1986 (9.9%) | 0.388 | 0.513 | 80 | 0.394 | 0.640 | 0.010 |
| phys | D_morph_DCT_32_64_um | 124/1986 (6.2%) | 0.788 | 0.911 | 63 | 0.772 | 0.512 | 0.069 |
| phys | D_morph_DCT_64_inf_um | 146/1986 (7.4%) | 0.742 | 0.824 | 53 | 0.702 | 0.612 | 0.160 |
| phys | D_morph_desc | 275/1986 (13.8%) | 0.173 | 0.079 | 11 | 0.935 | 6.594 | 0.170 |

关键结果：

- **所有 12 个 `p_perm_type2_within` 都 > 0.17**；最小值为 phys + descriptor distance 的 **0.173**。
- **所有 global-null Type-II p 值也 > 0.05**；最小值仍为 phys + descriptor distance，**p = 0.079**。
- **formal-only 复核全部不显著**；Type-II count 的最小 p 为 **0.394**。
- raw process space 的 Type-II count p 值普遍非常高（约 0.78–1.00）。按当前单侧定义，这不是“异常多的近工艺远形貌”，反而说明观察到的极端 Type-II 数量没有超过随机重排预期。
- phys space 中 Type-II 占 near pairs 的比例可达到约 6–14%，但**比例本身不是显著性**；在相应 null 下并没有形成统计过量。

因此，之前 round2 中看到的 `proc_near_morph_far_top10` 样本应该继续保留为**个案审计对象**，但不能由这些 top pairs 推导出总体 branching。

### 4.2 连续近邻统计量 `T_lambda`

这里还有一个值得保留、但不能升级为主结论的局部信号：

- phys process space、DCT 16–32 µm：
  - `T_lambda = 0.640`
  - within-session permutation **p = 0.00999**
- 但同一结果：
  - global permutation **p = 0.130**
  - formal-only within-session **p = 0.166**

所以这个信号**没有跨 null / formal-only 复现**。当前只能登记为：

> phys-reparameterized 邻域在 16–32 µm 尺度上存在一个探索性的局部 morphology-separation 信号。

不能写成：

> 16–32 µm 尺度发生了显著 branching。

### 4.3 与 exact-repeat sentinel 的关系

AUDIT-27/28 的 exact-repeat sentinel 仍然远近一致，可作为“同一登记工艺点下形貌可以非常接近”的一个实例；但只有一个 exact-repeat condition，因此它不能把所有 Type-II candidate 自动解释成 hidden variable，也不能充当 universal noise floor。

### 4.4 Gate 决策

**Gate 4 = NOT SUPPORTED AS A POPULATION-LEVEL EXCESS。**

更精确地说：

- individual Type-II candidates exist；
- no robust excess of Type-II pairs over the permutation null。

因此当前 Phase 2A **不触发 hidden-variable / stochastic-branching 优先路线**。这也不等于证明 process→morphology 是 deterministic；它只说明在当前工艺距离定义、P10/P90 阈值和现有样本量下，没有发现“近工艺却极远形貌”在总体上异常富集。

---

# Phase 2A 总体状态

**Phase 2A gate 正式关闭：允许进入 Phase 2B。**

四条 gate 的结论为：

1. **Artifact-driven? — NO, with flagged subset.**  
   高 leverage 不是主要由 preprocessing artifact 驱动，但存在 3 个明确 `artifact=yes` 和 9 个 `uncertain` 样本，需要在后续 sensitivity 中保留。

2. **存在真实形貌家族？— YES, descriptive morphology families.**  
   最清楚的是大尺度/长波高 leverage 家族，以及 scale-specific 的 `periodic stripe; anisotropic texture` 家族；当前不命名为物理 regime。

3. **只是连续幅度尾部？— NO.**  
   Sq/振幅是重要因素，但不足以解释 leverage；空间组织与尺度组成同样重要。

4. **Type-II 是否总体显著富集？— NO ROBUST EVIDENCE.**  
   个别 process-near/morphology-far pair 存在，但 Type-II count 在 within-session、global 和 formal-only 检验中均未形成稳健统计过量。phys / DCT 16–32 的 `T_lambda` within-session p≈0.010 是探索性局部信号，但未被 global-null 或 formal-only 复现。

## 对 Phase 2B 的直接含义

当前最值得优先检验的不是 stochastic branching，而是：

**scale-resolved nonlinear / condition-dependent process→morphology mapping**

Phase 2B 应继续：

- 比较不同空间尺度的 cross-validated explainability；
- 比较 raw controls 与 physics-motivated reparameterization；
- 比较线性与非线性模型；
- 将 `artifact=yes` 样本作为 sensitivity arm，而不是直接删除；
- 将 `uncertain` 样本保留在主分析，并通过 raw/repaired / exclude-flagged sensitivity 检查；
- 对 16–32 µm phys-neighborhood 的局部信号保留专项观察，但不预注册为“branching 已成立”。

当前结果**不支持**：

- 把形貌家族直接称为离散物理 regime；
- 把 Type-II candidate 解释为 stochastic branching；
- 把“不显著 Type-II excess”解释为已经证明 deterministic mapping；
- 把 49/50 sentinel 当成 universal noise floor。

## 路线优先级更新

1. **P1：scale-resolved process explainability**
2. **P1：linear vs nonlinear mapping**
3. **P2：local morphology structure / conditional model**
4. **P2：artifact-sensitive robustness**
5. **P3：repeatability matrix**
6. **暂不优先：hidden-variable / stochastic branching**

如果 Phase 2B 后出现“简单/非线性工艺模型仍无法解释某些稳定 morphology families”的结果，再把 repeatability matrix 和 hidden-state 假设提高优先级。
