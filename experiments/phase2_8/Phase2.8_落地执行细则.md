# Phase 2.8 落地执行细则（how）

> 状态：**FROZEN（随上位 v2.1，2026-09-04）**。上位规划 = `任务说明/Phase2.8_信息分解与单轨kernel多轨桥_研究任务说明.md` **v2.1 FROZEN**——全部科学定义、公式、门槛、判定顺序以该文件为准，本文件只登记 how：文件路径、config 键、QA 断言、运行顺序。两文件冲突时以上位规划为准。
> 事实基线：Phase 2.7 gate 终判 rev2（2.7r1 封账）+ `outputs/phase2_6/summary/RUNTIME_ENVIRONMENT.md`（强制 `.venv`，sklearn 1.7.2）。
> 执行顺序强制：**WP0 → WP1 → WP2 → WP3 → WP4**；WP1 未完成前禁止运行 Task 24/25 formal。

---

## 0. 收紧点（差异决策登记）

1. **脚本编号 24/25**（延续 21–23）；config = `experiments/phase2_8/phase2_8_config.yaml`。随机 seed 基 = `20260904`（沿 2.7 模式），offsets：`proc_gss_sensitivity 300`（仅 shuffle 型 sensitivity variant 使用；主协议双 GKF 均确定性、无 seed）、`bootstrap_b 900`（Task 25 kernel-group bootstrap）、`descriptor_perm 400`（描述性关联置换，若用）。
2. **folds 冻结 artifact（F1：恢复历史语义）**：`outputs/phase2_8/folds/fold_assignments.csv`，列 = `variant, fold, dataset_index, role∈{train,test}`（**完整 split 落盘**，不再只记 test assignment）。variant ∈ {src_gkf, proc_gkf}，语义与 Phase 2/2.5/2.7 完全一致（已核 Task 22 L65–66）：`src_gkf = GroupKFold(5, groups=shared_height_source_id)`（160 簇）、`proc_gkf = GroupKFold(5, groups=cv_process_group)`（134 组）——两者均为确定性 GroupKFold、无 shuffle、无 seed；shuffle 型如需要必须叫 `proc_gss_sensitivity`（GroupShuffleSplit，random_state=meta+300），**禁叫 proc_gkf**。由 Task 24 脚本生成一次，SHA256 写入 gate_eval；此后任何脚本只读不写。
3. **外层 skill 定义统一为 Q² 约定**（null = train-mean）：1 − Σ_test(y−ŷ)²/Σ_test(y−ȳ_train)²。这与 Phase 2.7 Task 22 用的 sklearn `r2_score`（test-mean 分母）存在定义差——2.8A 的数字**不与 0.505/0.595 等历史数直接同列比较**；标量目标附 `r2_score` 对照列仅为连续性目视。Phase 2.7 产物不回写。
4. **O_θ joint 目标** = (A2_8_16, angular_entropy_8_16) 原始坐标 Euclidean（两坐标均在 [0,1]，不另标准化）；分量 R² 作对照列。inner scorer 同为多变量 MSE。
5. **φ（F2：Moran proxy 废弃）**：Moran's I 不能代表 φ——I ~ RᵀWR 是二阶空间统计量，近似平移不变的邻接算子下由功率分布 \|R̂(k)\|² 主导（重新混入 P(λ)/O_θ），且 25600 像素节点的稠密 W 矩阵不可行。改为独立 **realization diagnostic**（上位 §2.8）：phase-only field `Q_i(k)=R̂_i(k)/(|R̂_i(k)|+ε)` → `q_i=F⁻¹[Q_i]`；shift-invariant pairwise `s_φ(i,j)=max_Δ corr(q_i, q_j(·+Δ))`、`d_φ=1−s_φ`（Δ 搜索范围 \|Δ\| ≤ 4 px，冻结入 config）；只报三个描述性问题（process-near pairs vs ordinary pairs、exact repeat 49/50 的 d_φ percentile、近邻工艺的可复现 realization）。不回归、不入谱表、不入 G28-A。实现入 `src/spectrum.py`（`phase_only_field/shift_invariant_phase_distance`）。
6. **Task 25 参数选择协议（F4：留出单位 = kernel identity）**：`kernel_group = (τ, f, N, v, 单线身份)`——同一 measured kernel 的全部 rectangle rows（可跨 h）**同组同留出**；协议名 **LOGO_kernel**（leave-one-kernel-condition-out；2.7 的 LOHO = leave-one-hatch-level-out，不同名不同义，禁止混叫）。全局参数（D_sat\*、c\*、γ\*）= LOGO_kernel median **TV_cond** 最小（tie 取小值）；per-h 参数作 sensitivity。c 网格沿用 2.7 `g27_3.c_grid`；`gamma_per_um` 网格 `linspace(−0.5, 0.5, 21)` **µm⁻¹**（交叉项 [g·g]=µm² ⇒ [γ]=µm⁻¹）；D_sat 网格 `geomspace(1, 64, 13)` µm。**physical guard（F6）**：训练 kernel group 模拟中出现 `z < −1e-9 µm` 的参数候选标 `physical_invalid` 并**排除出选择**（被排除格点逐个登记）；**禁 post-hoc clip**——clip 本身是又一个未登记的非线性。全部入 config，formal 后禁改。
7. **exact-match 子集（F3：候选 ≠ 可用）**：`n_candidate_exact_match = 19`（`direct_bridge_exact_match.csv` 行数）只是候选；经 estimable / QA / suitable profile / valid observation 过滤后的 `n_usable_kernel_groups` **程序化判定并在 gate_eval 报告**，**不设预写下限**（2.7r1 3A 同源登记为 13 个 exact-match conditions，量级参照）。执行时与 `lambda_over_hatch.csv` 按 dataset_index 合并取 observed class；observed class = INVALID/缺失的条件剔除并逐条登记。
8. **raw/repaired sensitivity**：raw 输入路径在 config `paths.raw_dataset_npz` 固定（执行时核对 Phase 1.5 raw 冻结产物；若不存在，该项记 N/A + 原因，G28-A 条件 7 以"N/A 已登记"方式满足）。
9. **文档修改（WP0）**逐处见 §2；冻结 CSV（phase2_manifest.csv）不改写。
10. **预冻结提交先于一切 formal**（沿 2.7 规则）：WP2/WP3 的 config + 脚本先 commit，此后门槛/网格/判定顺序禁改；formal 产物单独 commit。
11. **G28-B1 多重比较登记（P1）**：L2−L1 / L3a−L1 / L3b−L1 为**同一 model-family exploration 的三个比较**，CI 不解释为相互独立的 confirmatory 95% coverage；gate_eval 与 JSON 同时报告 **98.33% Bonferroni 式 simultaneous CI** 作 sensitivity。Phase 2.8 discovery-only，不做严格族错误率检验。

---

## 1. 数据契约（输入 → 用途 → QA）

| 输入 | 用途 | QA 断言 |
|---|---|---|
| `outputs/rectangle_registration/manual_internal_roi_v1/dataset/stable_roi_80um_dataset.npz` | R/H/V 场（200×160×160）+ `load_frozen` 重建 | shape/顺序/valid 覆盖（沿用 l15 断言） |
| `outputs/phase1_minimal/exploration_manifest.csv`、`outputs/phase2/manifest/phase2_manifest.csv` | D=`median_depth_um`、A=`residual_Sq_um`、u 五列、分组变量 | 200 行；`shared_height_source_id` nunique==160；`cv_process_group` nunique==134；两 manifest 的 D/A 逐元素一致 |
| `outputs/phase2_5/spectral_composition/spectral_composition.csv` + `ilr_coordinates.csv` | P_λ targets | 200 行；五行组分和 = 1（±容差）；ILR parity vs `src.composition` |
| `outputs/phase2_5/directional_spectrum/directional_metrics.csv` | O_θ targets（band=8_16；16_32 为 sensitivity） | 800 行（200×4 band）；A2∈[0,1]、H_θ∈[0,1] |
| `outputs/phase2_6/scale_bridge/lambda_over_hatch.csv` | observed class（Task 25） | 200 行；class 与 `assign_class` 一致 |
| `outputs/phase2_6/scale_bridge/direct_bridge_exact_match.csv` | exact-match 条件表（Task 25 primary） | 19 候选行；hatch_values 非空；`n_usable_kernel_groups` 程序化判定并报告（13 量级参照，不设预写下限） |
| `outputs/phase2_6/single_line/single_line_geometry.csv` + `geometry_qa_labels.csv` + `single_line_manifest.csv` + `annotations/single_line_view_manifest.csv` | g(x) 提取（Task 25，沿 Task 23 路径） | population==81（estimable ∧ qa≠reject） |
| `氧化锆/120组直线.cag` | 单线高度场现读 | `CagHeightReader` 用后 close |
| `outputs/phase2_6/scale_bridge/../../phase2_6 配置 bridge.box` | in-box 101 sensitivity | 子集 n==101；子集内重生成 splits + 契约校验 |

A 通道 parity：`sqrt(mean(R²))`（valid mask 上）与 manifest `residual_Sq_um` 逐元素一致（rtol 1e-9）；不一致 → STOP，查 Phase 1 定义，不允许静默用其一。

---

## 2. WP0 — 文档与 provenance 同步（先行，1 个 commit）

1. `README.md` 三处：
   - "矩形 ROI 来自 160 个独立 measurement（…）" → 改为 200 ROI（120 formal + 60 pass_main + 20 pass_supplement）/ 160 unique height-source / 134 cv_process_group 表述；
   - 核心结论 `P(m=2|h) 随 h 单调递减` → 加"block permutation 后 h-dependence 不显著（p=0.4103），descriptive only"；
   - 已知风险"功率 provenance … 保持 proxy" → 改为登记项（P_obj=5.3333 W post-objective canonical；instrument metadata unavailable；`_proxy` 旧列保留；f↔E_p 耦合提醒保留）。
2. `outputs/phase2/manifest/README.md`：Power note 改写——P=5.3333 W 已按外审 §1 更新决定升级 canonical；`power_measurement_version=PENDING_REGISTRATION` 指"仓库登记前"的历史状态，登记落点为 `src/provenance.py`；冻结 CSV 不改写。
3. `现有数据基础说明_v2.md` §11 末尾追加"登记状态更新（2026-09-04）"块：manifest 已登记 measured_power_W/source/version；外审确认物理可信 → canonical；instrument metadata unavailable；E_p/D_E canonical 派生见 `src/provenance.py`；f↔E_p 耦合提醒原文保留。
4. 验证：`grep` 断言 README 无 "160 个独立"、无 "P(m=2|h) 随 h 单调递减"单独句、无 "保持 proxy" 风险行。

## 3. WP1 — src/ 八模块迁移（每模块 1 个 commit + golden tests）

### 3.1 迁移顺序与内容

| 序 | 模块 | 迁入函数（来源） | 测试文件 |
|---|---|---|---|
| 1 | `src/data.py` | l15 `load_frozen`；包一层 `src.io_npz/io_cag` reader 门面 | `tests/test_src_data.py` |
| 2 | `src/provenance.py` | p2 `pulse_energy_proxy_uJ/scan_spacing_um/areal_pulse_density/areal_dose_proxy_j_mm2`（legacy 派生）+ `log/require` + **新** `POWER_REGISTRY`、`canonical_power_columns(man)`、`assert_canonical_power_parity(man)` | `tests/test_src_provenance.py` |

执行修订（2026-09-05 登记）：① 实际迁移顺序 provenance **先于** data（`load_frozen` 依赖 `require/log`，依赖序要求）；② `load_config/output_dir` **保持 phase-local**——各 phase 的 CLI 适配器语义不同（l15 用 `--config`+strict parse，p27 用 `--quick`+parse_known_args；output_root 约定也不同），统一会改动冻结脚本的 CLI 行为，不属共享科学实现，不迁移；③ 迁移中对 `_lib` 文件补 `sys.path`（REPO）插入以保证直接 import 场景可用。
| 3 | `src/cv.py` | p2 `gkf_splits/gss_splits/_group_sets/check_gkf_contract/check_gss_contract`；p26 `make_ridge_alpha_grid/make_ridge/ridge_alpha_inner_gkf`（语义冻结为 `_v1`）+ **新** `select_alpha_inner(X, y, groups, *, scorer, n_splits=5, grid=None)`（scorer ∈ {mse, multi_mse, aitchison_ilr_q2}） | `tests/test_src_cv.py` |
| 4 | `src/composition.py` | p25 `five_part_composition/frozen_band_fractions/apply_zero_replacement/ilr_matrix(ILR_A)/ilr_transform/ilr_inverse/aitchison_distance` | `tests/test_src_composition.py` |
| 5 | `src/spectrum.py` | p25 `radial_spectrum/spectrum_descriptors/directional_band_metrics`；l15 `dct_lambda_grid` | `tests/test_src_spectrum.py` |
| 6 | `src/geometry.py` | p26 `axis_frame/_pixel_indices/sample_profiles/lateral_positions/detect_online_flags/line_extent/stable_region/section_positions/_run_boundaries/section_features/aggregate_line/lambda_star_4_32/lambda_peak_4_32/in_box_mask/condition_key/shuffle_h_by_block/scan_plateau_features/plateau_stable_run/reconcile_stable_region/FragmentedStableRegion`；p27 `assign_class/q_distribution/profile_suitable` | `tests/test_src_geometry.py` |
| 7 | `src/statistics.py` | p25 `sign_matrix/exact_signflip_test/require_no_n4_to_5/knn_row_standardized_graph/moran_i/moran_permutation_p`；l15 `cluster_lists/boot_draw/build_resample_bank/boot_angles*/loco_angles`（cluster bootstrap 族）；p27 `tv/tv_perm_p/logistic_slope` | `tests/test_src_statistics.py` |
| 8 | `src/forward_models.py` | p27 `synth_field/field_class/hann_projection/cycles_level` + **新** `array_transfer(|Ã_array(k)|²)`、`overlap_descriptor(g, h)`（§3.4 定义）、`saturate(s, D_sat)`、`pairwise_interaction_field(g, h, phi, gamma)` | `tests/test_src_forward_models.py` |

每个迁入函数：canonical 实现 → 与旧 `_lib` 版本做 parity test（固定 fixture，逐元素 rtol 0）→ 旧 `_lib` 该函数体替换为 `from src.xxx import ...` re-export → 全测试 → commit。**parity 不过不许 re-export。**

### 3.2 golden anchors（`tests/test_golden_anchors.py`）

- 静态层：断言 `outputs/phase2_7/summary/*.json` 的冻结值（上位规划 §4.3 表）未被任何运行改动——每次全测试都跑，防"顺手重跑覆盖"。
- 重生成层（F8：frozen 只读）：`scripts/40_refactor_golden_regression.py`——以冻结输入把 Task 22/23 管线**重跑进 scratch**（`outputs/phase2_8/_regression_scratch/`；经 config 输出根覆写实现，**不写任何 frozen 路径**），然后 **scratch vs frozen** 做 SHA256/JSON 数值清单比对（不依赖 git 状态；不一致 → 报告差异并 fail）。**禁止"覆盖冻结产物 → 比对 → git checkout 恢复"流程**——中途异常或未被 git 跟踪的文件都可能污染封账产物。实现注：22/23 脚本输出路径来自 `p27.output_dir(cfg)`；scratch 模式通过复制 config 并覆写输出根实现；若 `output_dir` 不受 config 控制，则以最小 diff 给两脚本加 `--output-root` 参数（默认行为不变，改动由 golden 层覆盖验证）。跑通后产出 `outputs/phase2_8/refactor_regression_report.md`。

### 3.3 版本化纪律

frozen 语义函数禁止原地修改。2.8A 的 target-native inner scorer 是**新函数**（`select_alpha_inner`），不改 `ridge_alpha_inner_gkf`；未来任何 metric 变体走 `_v2` 新名。

---

## 4. WP2 — Task 24：`24_information_decomposition.py`

### 4.1 config 键（phase2_8_config.yaml 新增节）

```yaml
task24:
  targets:
    D:       {column: median_depth_um, type: scalar}
    A:       {column: residual_Sq_um, type: scalar}
    Pl:      {ilr: z1_z4, type: composition}
    Ot_A2:   {column: A2_8_16, type: scalar, band_sensitivity: 16_32}
    Ot_ent:  {column: angular_entropy_8_16, type: scalar, band_sensitivity: 16_32}
    Ot_joint: {columns: [A2_8_16, angular_entropy_8_16], standardize: fold_internal_train, type: multi_secondary}
  models: {full: [tau,f,h,N,v], h_only: [h], minus_h: [tau,f,N,v]}
  ridge: {alpha_grid_logspace: [-3, 3, 13], inner_splits: 5}
  folds: {src_gkf: shared_height_source_id, proc_gkf: cv_process_group, splitter: GroupKFold5}
  populations: {primary: common_intersection, in_box: bridge.box, raw_repaired: optional}
  realization_diagnostic: {phase_only_eps: 1.0e-12, max_shift_px: 4, repeat_pair: [49, 50]}
```

### 4.2 运行步骤

1. 载入冻结输入（§1 表）→ 构建 primary target 矩阵（D、A、ILR z1–z4、A2_8_16、angular_entropy_8_16）+ A-parity 断言 → 求 common intersection（登记 n）。
2. 生成 folds artifact（src_gkf=GroupKFold(5, shared_height_source_id) + proc_gkf=GroupKFold(5, cv_process_group)，含 train/test role 列，双 `check_gkf_contract`）→ SHA256 登记。
3. 逐 target × 逐 variant × 逐模型（M_full/M_h/M_-h/dummy）：inner `select_alpha_inner`（target-native：A2/entropy 与 D/A 同为标量 MSE，P_λ 为 Aitchison Q²）→ fold-internal StandardScaler+Ridge → 外层 Q²（标量附 r2_score 对照列；O_θ joint standardized Q² 作 secondary summary，坐标每训练折内标准化）→ 逐折明细入 `predictability_spectrum_folds.csv`。
4. realization diagnostic（§0 收紧点 5）独立计算 → `realization_diagnostic.csv` + JSON（不入谱表、不入 G28-A）。
5. 汇总：逐折中位 → `predictability_spectrum.csv`；图（Q² 主轴 + Δ_h 副轴标注，图注含上位 §2.4 解释边界语句）→ `.png`；契约断言 + coverage + folds 哈希 → `summary/gsl28_a_evaluation.json`。
6. G28-A 九条件逐条判定 → VALID/INVALID。

### 4.3 QA 断言（全部写进脚本，fail 即停）

- manifest 200 行 / 160 height-source / 134 cv-group；intersection n 登记且五 target 一致；
- folds：src_gkf/proc_gkf 均 GroupKFold(5)，分组变量分别为 `shared_height_source_id` / `cv_process_group`，过 `check_gkf_contract`；artifact 含 role 列且哈希与登记一致（重跑时）；
- O_θ joint secondary：坐标标准化只用训练折统计量（断言无 test 泄漏）；
- dummy：|skill| < 1e-9（Q² 约定下逐折精确为 0，容差防浮点）；
- 组分目标：五行和为 1；ILR 逆变换 roundtrip parity；
- coverage 表无空缺；无任何从历史 CSV 汇总直接读取 skill 的代码路径（审计注释）。

---

## 5. WP3 — Task 25：`25_kernel_bridge.py`

### 5.1 config 键

```yaml
task25:
  population: exact_match            # primary；surrogate 推迟（Phase 3，需 nested CV）
  holdout_unit: kernel_group         # (tau,f,N,v,single-line identity)；同 kernel 全部 rows 同留出
  selection_protocol: LOGO_kernel    # leave-one-kernel-condition-out（2.7 LOHO=leave-one-hatch-out，禁混名）
  sign_convention: removal_positive  # QA: median(g) > 0，否则冻结翻转+登记
  phase_grid: 32
  field_pipeline: {pixel_um: 0.5, roi_um: 80.0, bins: 24, window_um: [4,32]}
  levels:
    L0:  kernel_only
    L1:  {a_n: 1}
    L2:  {family: D_sat(1-exp(-s/D_sat)), grid_um: geomspace(1,64,13), select: global_LOGO_kernel}
    L3a: {c_grid: [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], select: global_LOGO_kernel}
    L3b: {gamma_per_um: linspace(-0.5,0.5,21),   # 量纲 µm^-1（[g·g]=µm²）
          physical_guard: {z_min_um: -1.0e-9, rule: exclude_candidate_no_clip},
          select: global_LOGO_kernel}
  metrics: {primary: tv_cond_out_of_group, legacy_reference: tv_pooled_5class,
            secondary: [tv_spectral_24bin]}
  g28b:
    b1: {delta_min: 0.05, bootstrap: {B: 2000, ci: 0.95, unit: kernel_group, paired: true},
         bonferroni_sensitivity: 0.9833, multiplicity_note: same_model_family_exploration}
    b2: {strong: 0.20, partial: 0.30, label: pooled_tv_legacy_reference}
```

### 5.2 运行步骤

1. 重建 81 线 profile library（复用 Task 23 提取路径，走 `src.geometry`）；断言 population==81、符号约定 QA。
2. exact-match 合并 → primary 条件表（`n_candidate=19` 登记；`n_usable_kernel_groups` 程序化判定并报告，不设预写下限；observed class 缺失者剔除并逐条登记）。
3. 逐条件 × 逐 level（32 相位）：`src.forward_models` 合成场 → `field_class` → q_pred(cond)；L0 用 g 自身 λ_peak→class。L3b 逐 γ_per_um 候选跑 physical guard（训练 kernel group 模拟 z<−tol → 剔除该候选并登记）。
4. TV 指标：primary = out-of-group **TV_cond**（= 1 − mean_i q_i(y_i)）；legacy reference = **TV_pooled**；secondary = 24-bin 谱 TV；全局参数 LOGO_kernel 选择（只用训练 kernel group）；逐级表入 `kernel_bridge_levels.csv`（含逐 condition、逐 h、逐 kernel_group、逐 level）。
5. B1：ΔTV_cond{L1→Lj} + kernel-group paired bootstrap CI（seed=meta+900；另报 98.33% Bonferroni simultaneous CI + multiplicity 登记）；B2：TV_pooled legacy 分级（pooled_tv_legacy_reference 标签）；组合判定（含 IMPROVED_BUT_INADEQUATE）→ `summary/gsl28_b_evaluation.json`。
6. 描述性附件：O(h)、r_pred 与 observed class/A2 的 Spearman（exact-match 子集；不做 CV claims）；L1→L2→L3 单调性表。

### 5.3 QA 断言

- profile library 数与 Task 23 一致；median(g)>0；D_sat/c/γ 选择仅用训练 kernel_group（日志打印选择用 group 清单与被 physical guard 排除的候选）；
- **kernel_group 完整性**：同一 (τ,f,N,v,单线身份) 的全部 rectangle rows 在所有 LOGO_kernel 折中同侧（断言无跨折泄漏）；
- 每个模拟场过 `field_class` 同管线（禁第二套谱实现）；全库无 post-hoc clip 代码路径；
- L3a 在 32-phase × 自家条件下的行为与 2.7 period-2 家族定性一致（sanity，非数值锚——数值锚仍归 §3.2 重生成层）；
- bootstrap 重采单位 = kernel_group，paired delta，非 ROI 重复行。

---

## 6. WP4 — gate_eval 与 confirmation 骨架（1–2 个 commit）

1. `outputs/phase2_8/summary/phase2_8_gate_eval.md`：G28-A 九条件表、G28-B1（TV_cond + Bonferroni sensitivity + multiplicity 登记）/B2（pooled-TV legacy reference）双轴表、Predictability Spectrum 终表（D/A/P_λ/O_θ:A2/O_θ:entropy + joint secondary）、realization diagnostic 结果、`n_candidate/n_usable_kernel_groups` 登记、发现节（仅当出现稳定排序才写，附逐折方向证据）、与 2.7 数字的关系说明（split 语义一致 + Q² 定义差声明）。
2. `src/confirmation.py` 骨架：`fit(discovery_manifest) / predict(confirmation_manifest) / evaluate_locked_predictions(...)` + locked-prediction JSON schema（model version、config hash、feature order、timestamp）；单测走通 mini fixture；**2.8 脚本零调用**。
3. `experiments/phase2_8/repeatability_matrix_design.md`：第一批 6–8 cond × 3–5 repeat 设计草案（condition 选取判据：Route T/P 高低、Sq 高低、error hotspot、stripe phenotype、ordinary；混合设计备选 6×3+14）；明确"分析与 discovery 预分离"。

## 7. 提交计划（commit 序列）

| # | 内容 | 触发 |
|---|---|---|
| 1 | WP0 文档同步（README/manifest README/§11） | 立即 |
| 2–9 | WP1 八模块（每模块：src + parity tests + 旧 `_lib` re-export） | 每模块全测试绿 |
| 10 | golden anchors + regression script + 回归报告 | 重生成层通过 |
| 11 | WP2 预冻结（config + 24 脚本 + folds artifact 说明） | formal 之前 |
| 12 | WP2 formal（spectrum CSV/PNG/JSON + gate 九条件） | 预冻结后 |
| 13 | WP3 预冻结（config + 25 脚本） | formal 之前 |
| 14 | WP3 formal（levels CSV/JSON + B1/B2） | 预冻结后 |
| 15 | WP4 gate_eval + confirmation 骨架 + repeatability 设计 | 收尾 |

## 8. 运行环境与命令

```bash
# 一律 .venv（sklearn 1.7.2；不同版本会导致 Ridge 内层 α 翻转）
.venv/Scripts/python.exe -m unittest discover tests          # 每次提交前
.venv/Scripts/python.exe experiments/phase2_8/24_information_decomposition.py
.venv/Scripts/python.exe experiments/phase2_8/25_kernel_bridge.py
.venv/Scripts/python.exe scripts/40_refactor_golden_regression.py
```
