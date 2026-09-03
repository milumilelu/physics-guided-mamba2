# Phase 2.5 Gate 评估 rev3(2026-09-03,review-fix 之后)

> formal 结果 commit = df10dba;review-fix commit = 本提交(修 spline 管道、敏感性索引、加权角熵、repaired 臂、Task 13 schema、provenance 字段,并新增 p8↔A2 桥)。config 阈值在 Task 12 运行前已冻结(86ecd28),无 post-hoc gate。
> 所有 R²/Q² 均为 exploratory CV estimate (n=200);细则状态已转 FROZEN_EXECUTED。

## Gate 判定(rev2 的两个 INVALID 支路已重算)

| Gate | 判据 | 实测(rev2 修正后) | 判定 |
|---|---|---|---|
| **G1** robust spectral allocation control | 双 GKF Q² 中位 ≥0.20、≥4/5 折为正、formal-only >0.10 | **src_gkf:ET 0.422 / Ridge 0.313 / spline 0.361;proc_gkf:0.309/0.341/0.311;全部 5/5 折为正;formal-only 0.327/0.322;exclude-artifact 0.444/0.336;minus-top5 0.376/0.285;repaired 0.412/0.311**(dummy 严格 =0,索引修复已验证) | **ROBUST(敏感性全面加固)** |
| **G2a** phenotype validation | 盲评 stripe 分离,perm p ≤0.05 | **AUROC 0.970、rank-biserial +0.94、p=0.0003(n_pos=6)**(不受 rev2 影响) | **VALIDATED** |
| **G2b** process predictability of A2_8_16 | grouped CV R² ≥0.20 双 GKF 同向 | **src 0.506/0.510、proc 0.536/0.495(ET/Ridge,5/5)** | **SUPPORTED** |
| **G3a** derived feature gain | dQ2(C−A) ≥0.05 双 GKF | Ridge +0.040 [5/5] / +0.030;spline src +0.153 但 spline 总体弱;ET ≈0 | **NOT 触发** |
| **G3b** mechanism bridge | dQ2_mech ≥0.05 双 GKF ≥4/5 折 | **+0.031 [4/5] / +0.032 [4/5]**,方向为正;provenance 降级为 historical_calibration_provenance=unknown | **NOT SUPPORTED**(登记正向趋势) |
| **G4** pass redistribution | ≥2 step 全局 p_exact ≤0.05 | **仅 N1→2 p=0.008**;2→3 p=0.112、3→4 p=0.241、N5→6 p=0.453;z3/z4 Holm=0.0505 | **NOT 触发**(N1→2 探索性) |
| **G5** error localization | Moran p ≤0.05 + 聚集 | **Sq:I=0.24/0.30,p≤0.0002,Jaccard 0.60,Spearman 0.51(model-robust);A2/depth 亦聚集;composition I<0,p≈0.65–0.88 不聚集;coverage 相关仅中等(ρ 0.13–0.31),非单纯 DOE 稀疏** | **Sq = LOCALIZED(model-robust)** |

## rev2 修正带来的三个实质变化

1. **spline 复活**:管道内选 alpha + 一致标准化后,composition Q² src 0.361(5/5)。此前"spline failed ⇒ 交互必然重要"的推断撤回:ET−spline 的增益收缩到约 +0.06,交互作用只是**温和**存在。
2. **敏感性从"存疑"变成"加固"**:索引修复(dummy 严格归零)后,exclude-artifact 0.444/0.336 与 minus-top5 0.376/0.285 都高于主 CV——artifact 样本与 top-LOCO 样本反而在轻微稀释信号;repaired 0.412/0.311 确认 raw/repaired 稳健(S4 补齐)。
3. **加权角熵成为最强方向观测量**:修复(按 PSD power 加权)后,angular_entropy_8_16 的 src_gkf R² 0.620/0.607/0.593(ET/Ridge/spline,5/5),proc_gkf 0.51–0.64——超过 A2 本身,且三个模型一致。

## P ↔ T 桥(rev2 新增,回答"是否同一物理链")

- 原始相关:ρ(p_8–16, A2_8–16) = **+0.553**——中度相关,不是同一指标的重复。
- 条件化诊断(ridge,src_gkf,折配对):
  - A2 ~ u + p8:dR2 = **+0.024**(5/5)——给定组成后,工艺对各向异性仍保留几乎全部预测力(基线 0.510);
  - p8 ~ u + A2:dR2 = **+0.050**(5/5)——给定各向异性后,组成可预测性也基本保留(基线 0.423)。
- **结论:Route P 与 Route T 是两个相关(ρ≈0.55)但实质上相互独立的工艺受控属性**——"能量分配到哪个尺度"与"该尺度上的取向强度"各自携带可泛化的工艺信息。二者是否源于同一物理链(如条纹形成同时决定局部能量与取向)仍需机理对照,但统计上不是简单的冗余投影。

## Task 13 方向措辞修正(rev2)

N1→2 的 p_exact=0.008 成立,但 Δz3=−0.47、Δz4=−0.62 按 ILR 定义意味着 **16–32 相对 ≥32、32–64 相对 ≥64 的份额下降**——即倾向**向较粗/长波尺度重新分配**(Δz1 亦微负,非 fineward),且 z3/z4 的 Holm=0.0505 略高于 0.05。因此只写:multivariate shift 显著;coordinate 方向为 descriptive/exploratory。G4 = NOT TRIGGERED 不变。rev2 之前的"移向更细尺度"表述作废。

## 规划 §41 十二问简答(rev3)

1. 是——组合 Q² 0.31–0.42 vs 单带绝对 RMS ≤0.24。
2. 是——centroid/N_eff/entropy 同向可预测,属完整谱重分配结构。
3. 最可解释的 balance:z2(<8 vs 8–16;src ET 中位 ~0.60)与 z1(见 ilr_balance_predictability.png)。
4. A 与 C 都稳健;C 增量 +0.03–0.04 低于阈值。
5. 非线性/交互:ET−Ridge 温和(约 +0.06–0.11),ET−spline 约 +0.06——存在但非决定性。
6. 8–16 能量与方向性中度相关(ρ=0.553)且**条件化后各自保留预测力**——两个相关而独立的受控属性(桥分析)。
7. 盲评 stripe 表型可被 A2_8_16 客观区分(VALIDATED)。
8. 仅 N1→2 一步显著(p=0.008),G4 不满足。
9. N5→6 无一致信号(p=0.453)。
10. mechanism 特征增量 +0.031(4/5 折),低于阈值;provenance 的历史标定来源未明,已如实登记。
11. Sq 误差显著局域化且跨模型稳健;composition 误差不局域化;coverage 相关中等,非单纯稀疏问题。
12. 下一阶段:①stripe/hatch 几何与 A2/entropy 的机理对照(联系 Route T);②Sq 局域化失效区域的重复性采样(G5 方向);③二者都不需要新模型架构。

## 边界重申

exploratory CV estimate (n=200);DC/均值偏移独立于谱分配;stripe 验证集为 enriched selection;N4→5 从未分析;§40 禁止语言全部适用。G5 措辞按审查收紧:"Sq prediction error shows significant local autocorrelation in process space; no such localization was detected for composition error"——不推广为"幅度可预测性全局均匀"。
