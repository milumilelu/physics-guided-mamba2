# Phase 2.5 Gate 评估(2026-09-03,formal 参数)

> 全部数字来自冻结 config(commit 先于 Task 12 运行)下的 formal 输出;R²/Q² 均为 exploratory CV estimate (n=200)。

## Gate 判定

| Gate | 判据 | 实测 | 判定 |
|---|---|---|---|
| **G1** robust spectral allocation control | src_gkf 与 proc_gkf 的 Q²_Aitchison 中位 ≥0.20;各 ≥4/5 折为正;formal-only 中位 >0.10 | **src_gkf:ET 0.422 / Ridge 0.313;proc_gkf:ET 0.309 / Ridge 0.341(全部 5/5 折为正);formal-only:ET 0.327 / Ridge 0.322**;exclude-artifact 0.211/0.173、minus-top5 0.078/0.096 仍为正 | **ROBUST** |
| **G2a** phenotype validation | 盲评 stripe 样本 A2_8_16 分离,perm p ≤0.05 | **AUROC 0.970、rank-biserial +0.94、p=0.0003(n_pos=6)** | **VALIDATED** |
| **G2b** process predictability of A2_8_16 | grouped CV R² ≥0.20,双 GKF 同向 | **src_gkf ET 0.506/Ridge 0.510;proc_gkf ET 0.536/Ridge 0.495(全部 5/5)** | **SUPPORTED** |
| **G3a** derived feature gain | dQ2(C−A) 中位 ≥0.05 且双 GKF ≥4/5 折 | Ridge src +0.040 [5/5]、proc +0.030;ET ≈0 | **NOT 触发** |
| **G3b** mechanism bridge | dQ2_mech 中位 ≥0.05 且双 GKF ≥4/5 折 | **+0.031 [4/5] / +0.032 [4/5]**——方向一致但低于阈值 | **NOT SUPPORTED**(正向趋势登记) |
| **G4** systematic pass redistribution | ≥2 个 step 全局 p_exact ≤0.05 | **仅 N1→2:p=0.008**(T=0.809);2→3 p=0.112;3→4 p=0.241;N5→6 p=0.453;z3/z4 的 Holm=0.0505 | **NOT 触发**(N1→2 登记为探索性) |
| **G5** prediction-error localization | 主要 unresolved target Moran p ≤0.05 且 hotspot 聚集 | **Sq:I=0.24/0.30,p≤0.0002,Jaccard 0.60,Spearman 0.51(model-robust);composition:I<0,p≈0.65–0.88(不聚集)** | **Sq = LOCALIZED;composition 不聚集** |

## 路线判定

- **Route P(spectral allocation control)= 触发(G1 ROBUST)**。
- **Route T(directional texture formation)= 触发(G1+G2a+G2b)**:工艺可泛化预测 8–16 µm 带的各向异性强度(A2 R²≈0.5),且该指标与盲评 stripe 表型在 AUROC 0.97 上一致。注意 E_frac 本身无方向信息,是 A2 把两者连起来的。
- Route M(mechanism bridge)未触发(dQ2 +0.031,低于 0.05 阈值,方向为正——登记)。
- Route P-N(pass redistribution)未触发(仅 N1→2 一步 p=0.008;1→2 步的 z3/z4 均值为负——首次重复加工把能量从 ≥32 µm 移向更细尺度——属探索性,不作 dynamics 解释)。
- Route E(Sq 的 localized unresolved)触发:Sq 预测误差在 process space 聚集且跨模型稳健,而 composition 误差不聚集——**幅度不可预测是局域化的,谱组成可预测性是全局的**,这个反差是 Phase 2.5 最有信息量的结果之一。

## 规划 §41 十二问简答

1. 五段成分(composition)整体 Q² 0.31–0.42,远高于任何单带绝对 RMS(≤0.24)——是。
2. E_8–16 是完整谱重分配的一部分:centroid R² 0.36–0.42、N_eff 0.17–0.25、entropy 0.13–0.18 同向可预测。
3. 最可解释的 balance:z1(fine vs coarse)与 z2(见 ilr_balance_predictability.png;R² 逐折中位)。
4. A 与 C 都稳健;C 的增量 dQ2 仅 +0.03–0.04(低于 0.05 阈值)——derived 特征可有可无。
5. 非线性增益:composition 上 ET ≈ Ridge(spline 失败);A2_8_16 上 Ridge 已 0.51,ET 0.51——方向纹理可预测性不需要复杂模型。
6. 8–16 能量与方向性相关:G2a AUROC 0.97 支持。
7. 盲评 stripe 表型可被 A2_8_16 客观区分(VALIDATED)。
8. pass count:仅 N1→2 一步呈共同重分配(p=0.008),不满足 G4 的两步要求。
9. N5→6 与 N1–4 无描述性一致(p=0.453)。
10. mechanism 特征 dQ2 +0.031/4–5 折——增量方向为正、量级不足。
11. Sq 的误差聚集(Moran p≤0.0002)且跨模型稳健;composition 误差不聚集。
12. 下一阶段:spectral allocation + directional texture 的机理对照(联系 stripe 与 hatch/scan 几何),以及 Sq 局域化误差的因果审计(优先于 repeatability matrix 之外新增任何建模)。

## 边界重申

所有结果为 exploratory CV estimate(n=200);composition 的 DC/均值偏移独立于谱分配(dc_offset_frac 单列);stripe 验证集为 enriched selection;N4→5 从未分析;禁止 §40 语言。
