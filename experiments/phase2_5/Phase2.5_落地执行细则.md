# Phase 2.5 落地执行细则（how）

> 状态：**FROZEN_EXECUTED**（2026-09-03）。提交链：config/gate 预冻结 = `86ecd28`（先于 Task 12 运行，无 post-hoc gate）；formal 结果 = `df10dba`；外部审查二（Route P/T 保留、3 实现问题 + 2 解释问题）后的 review-fix = 本提交（修 spline 管道内选 alpha、敏感性 arm-local 索引、加权角熵、repaired 臂补齐 S4、Task 13 schema、provenance 字段降级，新增 p8↔A2 桥）。Gate 终判见 `outputs/phase2_5/summary/phase2_5_gate_eval.md`（rev3）：G1 ROBUST、G2a VALIDATED、G2b SUPPORTED（Route P+T 触发）、G3a/G3b/G4 未触发、G5 Sq 局域化。
> 上位规划：同目录 `Phase2.5_形貌谱组成与工艺控制机制_执行细则.md`——它定 what/why（科学问题、Task 10–14、Gate G1–G5、语言边界）；本文件定 how：逐脚本的输入、约定、公式、QA 断言、输出 schema、测试与运行预算。两文件冲突时以本文件为准并回写上位规划。
> 事实基线：`现有数据基础说明_v2.md`；`experiments/phase1_5/Phase1.5_本细则.md`；`experiments/phase2/Phase2_执行细则.md`（含 §0 收紧点 rev3）与 `outputs/phase2/phase2_gate_summary.md`（rev2）。
> 继承原则：height_raw 为唯一主证据；residual 沿用冻结定义 R = H − per-sample valid-median；不重新定义 plane/depth/ROI/repair/朝向/DCT 带不改变（上位规划 §3.2）；49/50 永远只叫 repeatability sentinel；N4→5 因 session 混杂禁止分析；pass 数据只称 cross-sectional pseudo-trajectories。

---

## 0. 相对上位规划的收紧点（差异决策登记）

以下决策在对照仓库核查后做出（核查数据见各条括注），改动任何一条必须回写本节：

1. **成分定义双轨制（本细则最重要的决策；rev2 修正代数）**。核查确认 1.5-05 第 102 行 `var_R = np.maximum(np.mean(R ** 2, ...), 1e-300)`——**`var_R` 名不副实，它是第二矩 `M₂ = mean(R²)`**，不是中心化方差；`E_b = mean(R_b²)/M₂`、`Sq = √M₂`（第 102–105 行）。设 μ = mean(R)、**`dc_offset_frac = r_DC = μ²/M₂ ∈ [0,1]`**（即 C_DC²/Σ_all C²，Parseval 下 C_DC²/N_px = μ²）。五段 clean 成分取 non-DC 系数能量份额 `p_b = Σ_{k∈band,k≠DC}C² / Σ_{k≠DC}C²`。正确的对账恒等式（全部 1e-8 断言）：
   - 三个 DC-free 带：`E_b^frozen = (1 − r_DC)·p_b`（b ∈ 8–16/16–32/32–64）；
   - ≥64 带（mask 含 DC）：`E_≥64^frozen = r_DC + (1 − r_DC)·p_≥64`。
   - **我 rev1 的两处错误作废**：`mean_offset_frac = μ²/Var(R)` 实为 `r_DC/(1−r_DC)`（DC-to-nonDC **ratio**，不是 fraction，已更名 `dc_to_non_dc_ratio` 作为可选附加列）；"有限带 clean 份额 == frozen 值"也不成立（差 (1−r_DC) 因子）。
   - 双轨语义：对账轨复刻 frozen 约定（provenance/连续性）；ILR 只用 clean non-DC 五段（DC/均值偏移不进谱分配）；`dc_offset_frac` 独立为 **"DC/均值偏移类描述符"**（median-centering 后残留的 DC 贡献，与高度分布不对称相关但不等同于 skewness）；可选附加列 `dc_to_non_dc_ratio = r_DC/(1−r_DC)`。
   - 实测：四带份额 min 0.00216（零值分支不触发）；`dc_offset_frac` 量级 median ≈0.006、max ≈0.235（1.5-05 的 median 去中位残差仍可有非零均值）。
2. **零值处理分支不会触发**（实测四带份额 min 0.00216，无任何 <1e-6）：按上位规划 §6.1 走"禁止 pseudocount"路径；replacement 代码仍实现并留单测，但正式运行必须零替换，`replacement_used` 列全 False。
3. **径向谱分箱范围**：DCT λ 实际范围 **0.712–160 µm**，上位规划的 24 bins 未给下界；本细则冻结 `lambda_lo_um = 0.7`、`lambda_hi_um = 160.0`（geomspace 25 边），QA 断言 non-DC 未覆盖能量 < 1e-9。
4. **径向谱与 broad 成分的约定差异**：径向谱按 non-DC 系数能量（与成分轨一致）；broad 成分对账轨含 DC-in-≥64 的冻结约定。两套都输出但**不得混用**；所有跨表对比只用成分轨。
5. **盲评 stripe 标签解析**：盲评表 `blind_morphology_pattern` 用**空格分隔**（"periodic stripe;anisotropic texture"），不是下划线；解析规则 = 子串匹配 `"periodic stripe"`（实测 5–6/28 阳性）。G2 的 AUROC/置换检验据此实现，并强制"enriched selection,不估计总体 prevalence"注记（上位规划 §8.5）。
6. **G2 功效预告**：阳性仅 ~5–6 个，AUROC 置换检验功效有限；G2 未达标只能写 "not established at this audit size"，不写"方向纹理不受工艺控制"。
7. **14B 的 OOF 主模型钉死为 Ridge**：n=200 下 ET 在弱信号 target 上折间方差大（Sq 的 ET R² 0.034 < Ridge 0.078）；error atlas 主口径 = Ridge OOF，ET OOF 作为 sensitivity 行并列输出。05 的教训（family-D 标签泄漏循环变量）要求 OOF 组装用显式 fold 列，禁止位置隐含。
8. **Task 12 的 pre-freeze 规则落地**：primary targets、G1/G2/G3 阈值已在 config 冻结；**执行顺序 = 先 commit config（含阈值）再跑 12**，禁止看到结果后改阈值。
9. **quick 隔离沿用 Phase 2 修订**：`--quick` 下 `load_config` 把输出根改写为 `outputs/phase2_5_quick/`；quick 链需先在 quick 根重跑 10/11（确定性拷贝）；quick 产物不得被任何结论引用。
10. **Task 12 的 R 集合降级为 sensitivity**（上位规划 §3.4 已定）：主对比 **A vs C**；R 只在 input_comparison 附表出现。
11. **Moran 权重**：kNN(5) 二值对称图、行标准化；I = (n/S₀)·(zᵀWz)/(zᵀz)，z = 中心化误差；置换 10000 次（exact Monte-Carlo p = (1+b)/(1+10000)）。
12. **SplineTransformer 可用性**：sklearn 1.7.2 已验证可用（degree=3, n_knots=4, include_bias=False）；spline 管道 = StandardScaler → SplineTransformer → Ridge(fold-internal alpha)，spline 基函数只在训练折拟合。
13. **14A provenance 三状态（2026-09-03 审查二决定）**：`APPLICABLE` / `NOT_APPLICABLE` / `REDUNDANT_WITH_C`（label-free process-only 特征存在但只是 C 已用的同批代数组合 → REDUNDANT_WITH_C，不单独开 bridge）。14A **不是** 14B 的前置 gate；机制特征即使是 process controls 的确定函数也没问题——检验的是 mechanistic transformation 是否提供更好的 inductive bias。
14. **G2 拆分（同上决定）**：G2a = phenotype validation（AUROC/rank-biserial/perm p/n_pos/n_neg，状态 VALIDATED / INCONCLUSIVE_AT_AUDIT_SIZE / NOT_ALIGNED）；G2b = 全 200 的 process predictability（grouped CV）。两状态独立报告，允许 "G2a=INCONCLUSIVE_AT_AUDIT_SIZE 且 G2b=PROCESS_PREDICTABLE" 的组合措辞；p>0.05 禁止写成"metric 无效/工艺不控制方向纹理"。
15. **14B "unresolved" 加跨模型保险（同上决定）**：Ridge = primary localization map；hotspot 称 "model-robust unresolved" 必须在 ET sensitivity 下误差图型持续（`error_spearman_ridge_et` + `hotspot_jaccard_ridge_et` 两列量化）；Task 12 显示某 target ET−Ridge ≥0.1 且 ≥4/5 折同向时，该 target 的 Ridge hotspot 标 `linear-baseline hotspot`，不得直接称 unresolved。
16. **exact sign-flip 的 p 值（同上决定）**：全枚举不是 Monte-Carlo，`p_exact = #{T_null ≥ T_obs}/2^B`，**不加** (1+b)/(1+M) 修正；且观测到的全 +1 符号组态本身在枚举空间内，故 p ≥ 1/2^B 自动成立、p=0 不可能出现。

## 1. 通用约定

- **种子**：`random_seed: 20260903`；偏移登记：10 无随机、11 无随机、12 gss=seed+100/200、ExtraTrees=seed+700+fold、置换 importance=seed+800、13 无随机（exact 枚举）、14 Moran 置换=seed+900、error-atlas 其余无随机。
- **单位**：长度 µm；λ：µm；角度：deg（image frame）；能量份额：无量纲；谱质心：µm。
- **脚本骨架**：每脚本 `EXPECTED` 清单 + `_lib.require` 硬断言 + 分步 `log`；输出只写 `outputs/phase2_5/<子目录>/`；PNG/PDF/log 按 .gitignore 处理，CSV/JSON/NPZ-cache 按各节标注入库与否。
- **复用**：`experiments/phase2_5/_lib.py` 用 importlib 以独立模块名加载 `phase2/_lib.py`（其内部已加载 1.5 `_lib` 为 `l15`），禁止复制实现；phase2_5/_lib.py 只新增：五段成分/ILR/逆 ILR/Aitchison 距离、径向谱分箱、FFT 方向谱与 A2、sign-flip 枚举、Moran I。phase2 的 grouped split/契约校验/kNN 直接调 `p2.*`。
- **manifest**：读 `outputs/phase2/manifest/phase2_manifest.csv`（`p2.read_manifest`，含 LOCO 回填列），禁止重新构建；CV 分组键与契约校验（GKF partition / GSS 逐 split）完全复用 `p2.gkf_splits / gss_splits / check_*`。
- **依赖**：无新增（sklearn 1.7.2 已验证 SplineTransformer）。
- **commit 协议**：每个 Task formal 完成即 commit（中文信息）；Task 12 运行前先单独 commit config（§0.8）。

## 2. 数据契约

- 高度：`l15.load_frozen(cfg)`（cfg 复用 phase2 的两个路径键）→ `R`（200,160,160，float64，无 NaN）。
- manifest：`p2.read_manifest(cfg, require_loco=True)`；49/50 同 `cv_process_group` 由 phase2 QA 断言保证。
- 盲评标签：`outputs/phase2/instability/盲评/instability_manual_review_completed.csv`；stripe 阳性 = `blind_morphology_pattern` 含子串 `"periodic stripe"`；仅 Task 11 的验证节使用，Task 12 一律用全 200 自动 metric（上位规划 §8.5 的红线，单测锚定）。
- artifact-yes 名单：由 09 同款解析（unblind_artifact_suspected=="yes" → [37,149,82]），保持与 Phase 2 一致。

## 3. Task 10 — `10_build_spectral_composition.py`

1. 对账轨：调 `l15.dct_band_fields(R, 0.5, [[8,16],[16,32],[32,64],[64,1e9]])` 重算四带场，`E_b = mean(R_b²)/var(R)` 逐样本；与 `morphology_descriptors.csv` 的四个 `E_*_frac` 对账 `atol ≤ 1e-8`，失败 abort（上位规划 §5.2 STOP 条款）。
2. 成分轨：`C = dctn(R, axes=(1,2), norm="ortho")`；按 `l15.dct_lambda_grid` 的 λ mask（k≠DC）计算 `p_b = Σ_{k∈band,k≠DC}C²/Σ_{k≠DC}C²`（b ∈ {<8, 8–16, 16–32, 32–64, ≥64}）；断言 Σp=1（atol 1e-12）、p>0；计算 `dc_offset_frac = μ²/M₂`（μ=mean(R)，M₂=mean(R²)，均逐样本）。**对账 QA（rev2 恒等式，全部 1e-8）**：`E_b^frozen == (1−r_DC)·p_b`（b ∈ 8–16/16–32/32–64）与 `E_≥64^frozen == r_DC + (1−r_DC)·p_≥64`，r_DC = dc_offset_frac；另与 frozen CSV 直采值对账（同 1e-8）。
3. ILR：按上位规划 §6 的 Z1–Z4 闭式公式（系数 √(6/5)、√(1/2)、√(2/3)、√(1/2)）；断言逆 ILR 闭合（‖ILR⁻¹(ILR(p))−p‖∞ < 1e-12）与 Z 基正交性（单测覆盖）。
4. 径向谱：non-DC 模式按 λ geomspace(0.7, 160, 25) 分 24 bin；每样本每 bin 存 `lambda_lo/hi/geo_um, n_modes, energy, energy_fraction, energy_density_per_loglambda`；Σq=1；`low_mode_count = n_modes < 20` 标记；未覆盖能量 QA < 1e-9。
5. 描述符：`spectral_centroid_log_um`、`geometric centroid`、`spectral_entropy = H/log B`、`effective_band_number = exp(−Σq log q)`、`lambda_peak_um + peak_low_mode_count`。
6. 幅度—成分一致性（rev2 分 frozen/clean 两套列）：frozen 恒等式 `RMS_b == Sq·√E_b^frozen` 全带成立（1e-8，同一 M₂ 分母的代数恒等）；clean 恒等式 `RMS_{b,nonDC} == Sq·√((1−r_DC)·p_b)` 对三个有限带成立、≥64 带为 `Sq·√(r_DC+(1−r_DC)p_≥64 − r_DC)` 形式（即从 frozen 中剥出偏移），两套列都进 `amplitude_vs_fraction_consistency.csv`。该恒等式正是"fraction 可预测而 RMS 不可预测可以共存"的代数原因（§0b 第五观察）。
7. 输出（上位规划 §7.7 全清单）+ `dct_reconciliation.csv`（对账轨逐带残差）+ `radial_spectrum_matrix.npz`（**gitignored cache**，.gitignore 加行）。README 注明双轨约定。

## 4. Task 11 — `11_directional_spectrum.py`

1. 预处理（上位规划 §8.3 原样）：仅去 DC → 2D separable Hann → FFT → PSD=|F|² → 除以 Σw² 窗能归一。禁止重新去趋势。
2. λ 轴：`k = fftfreq(160, d=0.5)`（cycles/µm），λ=1/|k|，与 DCT 网格同一定义；带 mask 用 [8,16)/[16,32)/[32,64)/[64,∞)。
3. 指标：`A2`（二阶角动量，36 θ bin）、`theta_k_deg`（波矢角）、`theta_stripe_deg = theta_k+90 (mod 180)`、`angular_entropy`；每 band 一套。
4. 验证节：28 盲评样本，`stripe` 阳性 vs 其余的 `A2_8_16` AUROC（rank 法）+ 10000 次置换 p + rank-biserial；输出强制注记 "audit set is enriched selection"。
5. 方向框架：无 scan/hatch 相对位姿 metadata → 只能解释为 image-frame orientation（上位规划 §8.6）；README 必须写明。
6. 输出：上位规划 §8.7 全清单。

## 5. Task 12 — `12_spectral_process_map.py`（主实验）

- **targets**：P1 = ILR z1–z4（multivariate，Aitchison Q² 为主指标）；P2 = `A2_8_16, angular_entropy_8_16`（Task 11 QA 通过为前提）；secondary = centroid/entropy/N_eff/A2_16_32/A2_32_64；reference（不入 gate）= depth、Sq、rms_DCT_8_16。
- **inputs**：A（raw 5）与 C（A + 4 derived，`_proxy` 名保留）；R 只进 sensitivity 附表。
- **models**：dummy / ridge（fold-internal alpha，复用 `e05._select_alpha` 的内层标准化协议）/ spline（§0.12）/ extratrees（500, min_samples_leaf=2）。
- **CV**：主 = src_gkf + proc_gkf（5 折，`p2.check_*_contract`）；敏感性 = src_gss/proc_gss + formal_only + exclude_artifact_yes + minus_top5（选样逻辑与 phase2-09 逐字一致）+ repaired（仅 Sq/band RMS/spectrum，depth 保留 raw authority）。
- **指标**（上位规划 §12 全套）：`Q2_Aitchison`（dummy=fold-train z 均值）、`d_A` 的 median/Q25/Q75、五个 `MAE_p_*`（逆 ILR 后）、`R2_z1..z4`；禁止只挑最好 balance 代表整体。
- **A vs C**：折配对 `ΔQ²_{C−A}` 逐 (target, variant) 全变体输出（Phase 2-06 的教训：CSV 全 variant，图只画主 variant）。
- **feature interpretation 门槛**：median Q²/R² > 0.10 才做 permutation importance（n_repeats=10，seed+800）；`Q²_ET−Q²_spline` 在两个主 variant 均稳定为正才解释 interaction（优先对按上位规划 §14）。
- **OOF 契约**：`composition_oof_predictions.csv` / `directional_oof_predictions.csv` 每行 = (dataset_index, fold, cv_variant, model, input_set, z1..z4 预测, p 预测, 标量目标预测)；主口径 = src_gkf + Ridge（§0.7），ET 行并列；**每个样本在每个 (variant, model, input) 下恰好出现一次**（单测 #15）。
- 输出：上位规划 §15 全清单。运行前 commit config（§0.8）。

## 6. Task 13 — `13_pseudopass_spectral_redistribution.py`

- 样本集：pass_main = `base_condition_group` ∈ T01..T15 且 session_role=="pass_main"（15×N1–4）；supplement check = 映射后 T 组的 N∈{5,6}（10 组）。**任何 N4→5 组合直接 `require` 拒绝**（单测 #18）。
- Δz：z(N+1)−z(N)，主 steps 1→2/2→3/3→4，独立 5→6；每 step 样本量 = 组数（15 或 10）。
- 全局 exact sign-flip：T = ‖mean_b Δz_b‖；枚举 2^15/2^10 个符号向量（numpy 位运算实现，单测 #19/#20 锚定 32768/1024）；**p_exact = #{T_null ≥ T_obs}/2^B（rev2：全枚举不加 (1+b)/(1+M) Monte-Carlo 修正；观测到的全 +1 组态本身在枚举空间内，故 p ≥ 1/2^B 自动成立、p=0 不可能出现）**；coordinate-wise z1..z4 同 step 内 Holm 校正。
- 与 depth 关联：Δd vs Δz 的 Spearman，只作 cross-sectional association；图形标题强制 "cross-sectional pseudo-trajectory"。
- 输出：上位规划 §20 全清单。

## 7. Task 14A/14B — `14_mechanism_bridge_error_atlas.py`

- **14A provenance audit（rev2 三状态）**：扫描 `experiments/mechanism_virtual_augmentation/`，生成 `mechanism_feature_provenance.csv`（上位规划 §22 全字段 + 最终状态列）：`APPLICABLE`（有机制含义且 `depends_on_measured_morphology=false ∧ was_fitted_using_labels=false` 的额外 mechanistic summaries）/ `NOT_APPLICABLE`（无合格特征）/ `REDUNDANT_WITH_C`（label-free process-only 但只是 Task 12 的 C 已用同批代数组合，无新增科学问题）。14A **不是** 14B 的前置 gate；若非 APPLICABLE → `mechanism_bridge_summary.csv` 记对应状态，14B 独立运行。APPLICABLE 时检验的也只是"mechanistic transformation 是否提供更好的 inductive bias"。
- **14A bridge**：M0: z~A vs M1: z~[A, m(u)]，同 src_gkf/proc_gkf 折；`ΔQ²_mech` 配对输出；即使 >0 也只写 "mechanism-informed covariates provide incremental explanatory information"。
- **14B error atlas**（OOF = src_gkf + Ridge 主口径，ET sensitivity）：标量误差 `e/ IQR(y_train(fold))`；composition 误差 `d_A`；process coverage（`p2.knn_median_distance`，k=5/10）与 Spearman(e, d_proc)；Moran I（§0.11）；diagnostics 表（error vs Sq/depth/LOCO/consensus/repair/plane/session_role/spectral_entropy）；hotspot = top 10% OOF error 的 audit 标签 + 近邻表。**跨模型保险（rev2）**：输出 `error_spearman_ridge_et`（逐 target 样本级误差 Spearman）与 `hotspot_jaccard_ridge_et`（top-10% hotspot Jaccard）；hotspot 称 **"model-robust unresolved"** 必须在 ET sensitivity 下误差图型持续；Task 12 显示 ET−Ridge ≥0.1 且 ≥4/5 折同向的 target，其 Ridge hotspot 一律标 `linear-baseline hotspot`，不得直接称 unresolved。所有 diagnostics 只用于定位 unresolved regions，禁止 hidden state / stochasticity 语言。

## 8. `phase2_5_config.yaml` 草案

```yaml
random_seed: 20260903

paths:                                # 复用 phase2 冻结输入
  dataset_npz: outputs/rectangle_registration/manual_internal_roi_v1/dataset/stable_roi_80um_dataset.npz
  exploration_manifest: outputs/phase1_minimal/exploration_manifest.csv
  phase2_manifest: outputs/phase2/manifest/phase2_manifest.csv
  output_root: outputs/phase2_5

scales:                               # 与 1.5 完全一致，禁止改动
  pixel_um: 0.5
  sigmas_px: [2, 4, 8, 16]
  dct_bands_um: [[8, 16], [16, 32], [32, 64], [64, 1.0e9]]

spectrum:
  radial_log_bins: 24
  lambda_lo_um: 0.7                   # DCT λ 实际下界 0.712（核查值）
  lambda_hi_um: 160.0
  radial_bin_sensitivity: [16, 24, 32]
  low_mode_count_threshold: 20
  reconciliation_atol: 1.0e-8

composition:
  zero_threshold: 1.0e-10
  replacement_delta: 1.0e-6           # 本数据集不会触发（min 0.00216）
  replacement_sensitivity: [1.0e-8, 1.0e-6, 1.0e-5]
  replacement_stop_fraction: 0.05

directional:
  theta_bins: 36
  hann_window: true
  bands_um: [[8, 16], [16, 32], [32, 64], [64, 1.0e9]]
  blind_stripe_token: "periodic stripe"   # 空格分隔（解析核查值）
  permutation_n: 10000

input_sets:
  A: [pulse_duration_fs, frequency_kHz, hatch_spacing_um, pass_count, velocity_mm_s]
  C_extra: [pulse_energy_proxy_uJ, scan_spacing_um,
            areal_pulse_density_per_mm2, areal_dose_proxy_J_per_mm2]

targets:
  primary_multivariate: ilr_z1_z4
  primary_directional: [A2_8_16, angular_entropy_8_16]
  secondary: [spectral_centroid_log_um, spectral_entropy,
              effective_band_number, A2_16_32, A2_32_64]
  reference: [median_depth_um, Sq_um, rms_DCT_8_16_um]

cv:
  n_splits: 5
  gss_repeats: 5

models:
  ridge_alpha_grid: [0.01, 0.1, 1, 10, 100]
  spline: {degree: 3, n_knots: 4}
  extratrees: {n_estimators: 500, min_samples_leaf: 2}
  oof_primary_model: ridge            # §0.7

gates:                                # 上位规划 §32 冻结；先 commit 再跑 12
  G1_multivariate_q2: 0.20
  G1_formal_only_q2: 0.10
  G1_balance_r2: 0.20
  G2a_validation_p: 0.05              # rev2：G2 拆分为 phenotype validation…
  G2b_directional_r2: 0.20            # …与 full-200 process predictability
  G3_delta_q2: 0.05
  G4_step_p: 0.05
  G5_moran_p: 0.05
  hotspot_quantile: 0.90

pass:
  exact_sign_flip: true

error_atlas:
  knn_k: [5, 10]
  moran_permutations: 10000

sentinel: {session: zro2_120_formal, processing_orders: [49, 50]}
plot: {dpi: 150, diverging_cmap: RdBu_r}
```

## 9. `tests/test_phase2_5_lib.py`（22 项落地）

上位规划 §36 逐条映射（全部 CI-safe；importlib 加载 phase2_5/_lib，模块名 `phase2_5_lib`）：

1. `test_frozen_fractions_reconcile`（对账轨 1e-8，需真实数据；CI 无 outputs 时 skip）
2. `test_five_part_sums_to_one`
3. `test_no_pseudocount_without_zero`
4. `test_ilr_inverse_roundtrip`
5. `test_aitchison_distance_is_ilr_euclidean`
6. `test_z_basis_orthonormal`（**rev2 修正**：Z1–Z4 的 4×5 系数矩阵 A 满足 **AAᵀ = I₄**（行正交）；AᵀA 是 simplex contrast 子空间上的 rank-4 投影，不等于 I₅——rev1 写反了）
7. `test_radial_energy_sums_to_one`
8. `test_fft_window_normalization_finite`
9. `test_isotropic_field_low_A2`（白噪声合成场 A2 < 0.2）
10. `test_vertical_stripe_orientation`（合成竖条纹 → 波矢水平 → stripe 竖直，θ 关系 90°）
11. `test_gkf_source_contract`（复用 p2 断言）
12. `test_gkf_process_contract`
13. `test_sentinel_same_process_group`（复用 phase2 断言）
14. `test_mechanism_features_no_morphology_dependency`（provenance 表扫描）
15. `test_oof_rows_unique_per_sample`
16. `test_pass_bases_15` / 17. `test_supplement_bases_10`
18. `test_n4_to_5_refuses`（require 触发）
19. `test_signflip_15_equals_32768` / 20. `test_signflip_10_equals_1024`
21. `test_blind_labels_validation_only`（Task 12 的 target 表不得含盲评列——结构断言）
22. `test_low_mode_count_flagged`

## 10. 输出树 / 11. 运行顺序与预算

按上位规划 §4/§7.7/§8.7/§15/§20/§29 的目录与文件名执行；补充：`radial_spectrum_matrix.npz` 与一切 `*.npz` cache 加 gitignore；PNG/PDF/log 不入库。

```text
10 composition   ~1 min   （对账断言为 STOP 门）
11 directional   ~2 min   （stripe 验证 + A2 QA）
12 process map   ~15–25 min（formal；先 commit config）
13 pseudopass    <1 min   （exact 枚举）
14 bridge+atlas  ~15 min  （Moran 10000 置换 + provenance 扫描）
```

每步 formal 后 commit；quick 冒烟先行不产生 commit。

## 12. Gate 落地判读（G1–G5 → 数据文件映射）

- **G1**：`process_map/cv_summary.csv` 的 `Q2_Aitchison`（src_gkf 与 proc_gkf 中位 ≥0.20、各 ≥4/5 折为正、formal_only 中位 >0.10）；未达多变量线时按 balance `R2_z1..z4` 走 PARTIAL。
- **G2（rev2 拆两态）**：**G2a phenotype validation** — `directional_spectrum/stripe_validation.csv`（AUROC、rank-biserial、perm p、n_pos/n_neg；状态 VALIDATED / INCONCLUSIVE_AT_AUDIT_SIZE / NOT_ALIGNED，p>0.05 ≠ metric 无效）；**G2b process predictability** — `cv_summary.csv` 的 `A2_8_16` grouped R² ≥0.20 双 GKF 同向。两态独立报告，组合矩阵（如 G2a=INCONCLUSIVE + G2b=PROCESS_PREDICTABLE）按上位规划 §8.5 措辞解释。
- **G3a/G3b**：`input_comparison.csv` 的 `ΔQ²_{C−A}` / `mechanism_bridge_summary.csv` 的 `ΔQ²_mech`（中位 ≥0.05 且双 GKF ≥4/5 折同号）；G3b 的 provenance 为三状态（APPLICABLE / NOT_APPLICABLE / REDUNDANT_WITH_C），NOT_APPLICABLE ≠ 机制错误。
- **G4**：`pseudopass/pass_step_global_test.csv` + `pass_step_coordinate_tests.csv`（≥2 个 step 全局 p_exact ≤0.05 且同一 balance 跨 step 方向一致、Holm ≤0.05）。
- **G5**：`error_atlas/error_moran_test.csv`（p ≤0.05）+ `error_hotspots.csv` 的 process-space 聚集；结合 `error_density_association.csv` 区分 coverage-driven 与 dense-region unresolved；**hotspot 称 unresolved 须过跨模型保险（§7 rev2），否则标 linear-baseline hotspot**。
- 结论语言严格执行上位规划 §39/§40；Route P/T/M/P-N/E 名称保持数据驱动。

## 13. 待确认问题 → 已决定（2026-09-03 审查二）

| 项目 | 决定 |
|---|---|
| 14A mechanism bridge | 接受 NOT_APPLICABLE 可能性；provenance 三状态（APPLICABLE / NOT_APPLICABLE / REDUNDANT_WITH_C）；14A 不是 14B 的前置 gate，14B 无条件独立运行 |
| G2 audit 功效 | 接受 audit-size 限制措辞；G2 拆为 G2a（phenotype validation）+ G2b（full-200 process predictability），两态独立报告（§0.14、§12） |
| DC/均值偏移 | 支持 DC 独立、不进 ILR；rev1 的 `mean_offset_frac` 公式作废，改为 `dc_offset_frac = μ²/M₂`，对账恒等式按 §0.1/§3 四条重建（1e-8） |
| 14B 主模型 | Ridge 主口径 + ET sensitivity；"unresolved" 须过跨模型保险（error Spearman + hotspot Jaccard），否则标 linear-baseline hotspot |

另修两处技术点：ILR 正交单测改为 **AAᵀ = I₄**；exact sign-flip 改 **p_exact = #{T_null ≥ T_obs}/2^B**（无 +1 修正）。三项硬条件（DC normalization、ILR 单测、exact p）完成后，本细则进入正式实现阶段。
