# Phase 2 执行细则：形貌失稳审计与尺度分辨工艺可解释性

> 状态：**DRAFT_FOR_REVIEW v2**（v1 经 2026-09-03 外部审查，本版已吸收全部 6 项关键修改与 5 项工程修改；仍未冻结）
> 上位规划：`任务说明/Phase2_任务规划说明.md`——规划定 what/why；本文件定 how：逐脚本的输入、公式、输出 schema、验收与运行预算。
> 事实基线：`现有数据基础说明_v2.md`（2026-09-01，下称 v2）；`experiments/phase1_5/Phase1.5_本细则.md`（§8 冻结结论，commit 454f345）。
> 继承原则：height_raw 为唯一主证据；不预设离散物理机制命名；PCA 维数不解释为物理维数；模型残差不直接解释为随机性；不引入 Mamba/深度网络；不删除高 leverage 样本后报告"更好"的结果；49/50 永远只叫 repeatability sentinel。

---

## 0. 相对规划说明的收紧点（差异决策登记）

规划说明留有实现自由度之处，本细则做如下具体化。**改动任何一条都必须回写本节**：

1. **功率与能量坐标（proxy 化）**：三张设计表均无功率列（已核对）。按 v2 §11 登记实测物镜后功率 `measured_power_W = 5.3333`（软件标称 10 W 仅备注）。**在功率独立测量记录补登记之前**，能量/剂量列一律命名 `pulse_energy_proxy_uJ`、`areal_dose_proxy_J_per_mm2`，标注 provisional；补齐后才允许去 `_proxy` 后缀（§19.1）。
2. **"physics coordinates" 更名**：Input Set B 全部派生量是 raw 五参数的确定性函数，不增加信息内容（I(u_raw; Y) 不变）。全套文档与代码将 Set B 称为 **physics-motivated reparameterized coordinates**；其检验问题是"某种重参数化是否更契合简单模型的 inductive bias"，而不是"加入了更多物理信息"。
3. **f 与 E_p 完全耦合**（v2 §11）：Set B 中 E_p 与 frequency 严格单调等价；Set B 的重参数化价值在 Δx、n_A、D_E 的组合方式。全文禁止解读"f 或 E_p 的独立效应"（§18）。
4. **CV-B 分组新增 `cv_process_group`**：formal 按**五元组** (τ, f, h, N, v) 分组（49/50 强制同组，杜绝 exact-repeat 跨 train/test）；pass_main/supplement 按 base (τ, f, h, v) 分组并绑定 N=1..6（134 组）。CV-A = 泛化到 unseen physical source/surface；CV-B = 泛化到 unseen process condition（§2.2、§7.3）。`base_condition_group` 保留原义（轨迹结构），仅用于 08 与描述。
5. **base_condition_group 合并 T/S**：已核实 S01–S10 与 pass_main 的 T 组四元组 (τ, f, h, v) 10/10 重合，合并后 10 组覆盖 N=1..6、5 组覆盖 N=1..4。
6. **N≥5 ⟺ supplement session**（v2 §10.2）：轨迹式解释一律限 N=1–4；supplement 参与 CV 与 instability 清单，但 pass 演化类表述必须带 session 混杂声明。
7. **CV 折契约按 split 类型区分**（§7.3）：GroupKFold 要求 test 组两两不交且并集=全部组；GroupShuffleSplit 只要求 split 内 train∩test=∅，并报告每 group 入 test 次数。v1 的统一断言是错误的，已废弃。
8. **Type II 判据与 null 重构**（§5）：主 null = **within-session_role 行置换**（保持 session 构成）；global 行置换降为敏感性 null；另报 formal-only Type II。二值计数之外增设连续统计量 `T_λ = median[D_morph | D_proc ≤ P10]`（降低对 P90 二值阈值的依赖）。阈值 P10/P90 仍用于展示，Route D 判读需通过 P15/P85 扰动（§17）。
9. **family-D（band PC score）降级为 secondary representation diagnostic**（§7.5）：主结论建立在 band RMS/energy、Sq、depth 与显式 descriptor 上；family-D 增加跨折模态对齐输出 `fold_pc_alignment.csv`，只有 within-fold bootstrap 与 between-fold 对齐双稳定时才可引用。
10. **Route 改名并降低判读强度**（§17）：A → **nonlinear mapping priority**（"deterministic" 必须等 repeatability matrix）；B 的触发只比较 morphology 波段（depth 仅作参考 panel，永不参与触发）；D → **unresolved morphology branching / replicate-needed**（不区分 hidden variable 与坐标不充分——那是补实验后的问题）。
11. **band-definition sensitivity 用 Difference-of-Gaussians**（§11）：G2/G4/G8/G16 是累计低通，与互斥 DCT 带不可比；改用 DoG 近似带 (G4−G2, G8−G4, G16−G8, R−G16)，其 −3dB 波长对应 (≈7.5–15.1, 15.1–30.2, 30.2–60.3, ≥60.3 µm)，与 DCT 带逐带对应。
12. **A_consensus 的谱项改为四带最异常 rank**：`spectral_rank_min = min(rank_E_DCT_8_16, …, rank_E_DCT_64_inf)`，替代 v1 固定用 ≥64 µm 能量分数（避免天然偏向低频极端样本）。
13. **top-N 产物更名**：`instability_selected.csv`（实际行数 20–30），不再叫 top20。
14. **人工审计两轮盲评**（§4）：第一轮只看形貌图/repair mask/PSD，匿名编号（AUDIT-xx），不显示工艺参数、LOCO rank 与样本身份；第二轮 unblind。防止确认偏差。
15. **band_fields.npz 为本地可重建 cache，不入 git**（§6、§14，.gitignore 显式加行）；入库的只有 target 表、config 与 summary。
16. **05 规模修正与 family-D 缓存**（§7.6）：outer fits = 21×3×3×4×5 = 3780（非 ~7500）；family-D = 12×3×3×4×5 = 2160；family-D 的折内 PCA/bootstrap 按 (variant, fold, band) 只算一次并缓存，不随 input/model 重算。
17. **quick 协议与输出隔离**（2026-09-03 审查后修订）：`--quick` 下 `load_config` 把 `output_root` 改写为 `outputs/phase2_quick/`，quick 产物与 formal 根完全隔离，杜绝"CSV 来自 quick、PNG 来自 formal"的混合产物；quick 链需要先在 quick 根重跑 01/04（内容与 formal 相同的确定性拷贝，代价秒级）；无 ExtraTrees 的 quick 运行不产出 by-model 图，EXPECTED 相应放宽。quick 产物只做冒烟，不得被任何结论文件引用。
18. **LOCO 全量复算口径**：1.5 仅存每 (subset, scale) top-5 且用 k=1（PC1）口径；Phase 2A 对全 200 样本重算 PC1 与 PC1–3 双口径，以 1.5 的 global/total rank-1（cluster `zro2_120_formal:m066`）作一致性哨兵（§3）。

## 1. 通用约定

- **种子**：`random_seed: 20260903`；脚本内全部随机性 = seed + 固定偏移，偏移写在各脚本 docstring（03 的 within-null = seed+300、global-null = seed+400）。
- **单位表**：长度 µm；τ：fs；f：kHz；v：mm/s；h：µm；N：无量纲；P：W；E_p：µJ；D_E：J/mm²；角度：deg；距离：µm（descriptor 空间距离为 robust-z 无量纲）。
- **脚本骨架**：每脚本定义 `EXPECTED` 输出清单，结束前 `require` 全部存在（同 1.5 惯例）；关键计数/行数用 `_lib.require` 硬断言；分步 `_lib.log`。
- **复用**：`experiments/phase2/_lib.py` 通过 `importlib` 以独立模块名加载 `experiments/phase1_5/_lib.py` 为 `l15`（避免与 phase2 自身 `_lib` 的 sys.modules 冲突），**禁止复制其实现**。phase2/_lib.py 仅新增：自有 `load_config`（默认读 `phase2_config.yaml`）、manifest 构建与校验、§2 派生坐标、kNN、grouped split 与契约校验、fold-internal PCA 封装、DoG 带。
- **依赖**：requirements.txt 现有包已覆盖，无新增依赖。
- **输出**：只写 `outputs/phase2/<子目录>/`；PNG/PDF/log 由现有 .gitignore 处理，CSV/JSON 入库（`band_fields.npz` 除外，§0.15）。
- **不动既有产物**：不改 `src/`、不改 1.5 脚本与输出；Phase 2 全部代码在 `experiments/phase2/`。

## 2. phase2_manifest（数据契约）

文件：`outputs/phase2/manifest/phase2_manifest.csv`，200 行 × 约 41 列。
由 `experiments/phase2/_lib.load_phase2_manifest(cfg)` 构建并写盘：纯 join、无随机性、幂等。**01 首跑负责生成并回填 LOCO 两列；03–09 只读**（读取时 require 列齐全，缺失则提示先跑 01）。

### 2.1 原样继承列（来源 `outputs/phase1_minimal/exploration_manifest.csv`，23 列）

`dataset_index, session_id, measurement_id, sample_id, shared_height_source_id, roi_within_measurement, session_role, design_group, processing_order, x_position_um, y_position_um, compressor_steps, pulse_duration_fs, pulse_duration_calibration_id, pulse_duration_calibration_version, frequency_kHz, hatch_spacing_um, pass_count, velocity_mm_s, valid_fraction, repair_fraction, median_depth_um, residual_Sq_um`

### 2.2 新增列

| 列 | 来源 / 公式 | 单位 |
|---|---|---|
| `plane_rmse_um`, `plane_status` | `config/frozen/measurement_planes_160.csv` 按 (session_id, measurement_id) join（utf-8-sig 读） | µm / — |
| `measured_power_W` | v2 §11 实测物镜后功率 = 5.3333（provisional） | W |
| `nominal_software_power_W` | v2 §11 = 10（仅备注） | W |
| `power_measurement_source` | `"experiment_background_v2_s11_no_independent_record"` | — |
| `power_measurement_version` | 待补（用户登记日期/记录号后回填） | — |
| `constant_power_assumption` | true | — |
| `pulse_energy_proxy_uJ` | `1000 * measured_power_W / frequency_kHz`（provisional，§0.1） | µJ |
| `scan_spacing_um` | `velocity_mm_s / frequency_kHz` | µm |
| `areal_pulse_density_per_mm2` | `1e6 * N * f / (v * h)` | pulses/mm² |
| `areal_dose_proxy_J_per_mm2` | `1000 * P * N / (v * h)`（provisional，§0.1） | J/mm² |
| `quad_key` | `f"{τ:g}:{f:g}:{h:g}:{v:g}"`（四维碰撞审计用） | — |
| `base_condition_group` | design_group ∈ T01..T15 原样；S01..S10 按四元组映射到对应 T 组；formal → `f"F{processing_order}"` | — |
| `cv_process_group` | formal：`"FQ:" + quad_key + f"|N{pass_count:g}"`（五元组，49/50 强制同组）；pass_main/supplement：`"BASE:" + base_condition_group`。共 119 + 15 = **134 组** | — |
| `phase1_global_loco_rank`, `phase1_global_loco_angle_deg` | 01 回填（total 场 PC1 口径） | — |

S→T 映射（已核实，写死在 phase2/_lib.py 并由单测锚定）：
`S01→T01, S02→T02, S03→T06, S04→T07, S05→T08, S06→T10, S07→T12, S08→T13, S09→T14, S10→T15`。

### 2.3 单位推导（写进 phase2/_lib.py docstring，单测锚定）

- Δx[µm] = v[mm/s] / f[kHz]。锚点：(11, 2) → 5.5；(20, 40) → 0.5。
- E_p[µJ] = 1000·P[W] / f[kHz]。锚点：(5.3333, 2) → 2666.65；(5.3333, 200) → 26.6665。
- n_A[pulses/mm²] = 1e6·N·f / (v·h)。锚点：(N=2, f=2, v=9, h=6)（49/50 条件）→ ≈ 74074.07。
- D_E[J/mm²] = 1000·P·N / (v·h)。锚点：(5.3333, N=2, v=9, h=6) → ≈ 197.90。

### 2.4 QA 断言（构建时 require）

200 行；(session_id, sample_id) 唯一；shared source 160 个；design_group 非空恰 80 行；base_condition_group 135 组、cv_process_group 134 组，每行恰属一组；5 个 raw 工艺列 + 4 个派生坐标列无 NaN；49/50 的 quad_key 与 cv_process_group 相同；`session_role` 计数 = {formal:120, pass_main:60, pass_supplement:20}；`valid_fraction.min() == 1.0`。

## 3. `01_instability_inventory.py`（Phase 2A-1）

### 3.1 输入

phase2_manifest（本脚本构建）、NPZ（经 `l15.load_frozen(cfg)`）、`outputs/phase1_5/morphology_descriptors.csv`、`outputs/phase1_5/loco_top5_influencers.csv`（仅对账）、`outputs/cone_repair_inventory/cone_repair_artifact_table.csv`。

### 3.2 步骤

1. 构建/读取 phase2_manifest（§2）。
2. **全量 LOCO**：fields = `[total, DCT_8_16, DCT_16_32, DCT_32_64, DCT_64_inf]`（`l15.multiscale_fields`）；clusters = `l15.cluster_lists(man)`（160 个 shared source 整簇，与 1.5 同键）。每 field 两遍：`k=1`（PC1 角）与 `k=3`（PC1–3 子空间最大角）。LOCO 结果按**簇**记录，逐样本列取该行所属簇的角度（双槽共享源两行同值，如实反映测量单位）。
   **一致性哨兵**：k=1 口径下 total 场 rank1..5 的 cluster_id 序列必须与 1.5 `loco_top5_influencers.csv` 的 (subset=global, scale=total) 完全一致，否则 require fail。预计 < 10 min。
3. **amplitude 块**：descriptors 已有 9 列；新增 `peak_to_valley_p98p2_um`（valid 像素 R 的 P98−P2，主口径）、`R_max_minus_min_um`（备注口径）、`deepest_negative_residual_um = -min(R)`。
4. **spectral 块**：4 band 的 `rms_*_um` 与 `E_*_frac`（descriptors）；新增 `band_PC1_score_audit/band_PC2_score_audit` ×4 band（全 200 样本 `l15.gram_pca(X, 2)` 投影）。**audit 专用，禁止用作 CV target**（§7.4）。
5. **leverage 块**：10 列 LOCO（5 field × 2 口径）+ 回填 manifest 两列。
6. **isolation 块**：
   - descriptor 空间 14 列（§12 config `descriptor_cols`）；robust 标准化 `(x − median)/IQR`（任一列 IQR=0 → require fail）。
   - kNN 欧氏、自排除：k ∈ {5, 10} → `D_morph_k5, D_morph_k10`（robust-z 单位）。
   - process 空间：raw = 5 维 z-score `(τ, f, h, N, v)`；phys/reparam = 5 维 z-score `(τ, pulse_energy_proxy_uJ, scan_spacing_um, areal_pulse_density_per_mm2, areal_dose_proxy_J_per_mm2)`。各算 k=5 → `D_proc_raw_k5, D_proc_phys_k5`。
7. **artifact 块**：`repair_fraction, valid_fraction`（manifest）、`repair_pixel_count, repair_largest_component_px`（直接从冻结 repair_mask 逐样本计算；**注意 `outputs/cone_repair_inventory` 只覆盖 15 个单线 pilot 组，与 200 样本主数据集无关，不得 join**——该 join 键断言在实现时已验证并拦截）、`plane_rmse_um`。ROI 边界距离：**不构造**（200 个 ROI 同为中央 80×80 µm；README 注明不适用）。
8. **rank 与 consensus**：降序 rank（1=最极端，ties 取 average）：`rank_Sq, rank_loco_total_pc1, rank_D_morph_k10, rank_pit_density`，以及 4 个 `rank_E_DCT_*`；`spectral_rank_min = min(四个 rank_E)`（§0.12）；`A_consensus = median(rank_Sq, rank_loco_total_pc1, rank_D_morph_k10, rank_pit_density, spectral_rank_min)`。**只用于排序，不解释为物理量**。
9. **top-N 选择**（selection_reason 多标签 `|` 分隔）：LOCO total PC1 top10 ∪ 每 band PC1 top5（4×5）∪ Sq top10 ∪ D_morph_k10 top10 ∪ `D_morph_k10/D_proc_phys_k5` 比值 top10 ∪ sentinel 49/50（强制）∪ `repair_fraction>0` 打标签（不自动入池）。去重后**cap 30**（超限先丢仅凭比值入选者、再丢 morph_isolation、再丢 sq_top10 尾部；sentinel 与 LOCO total top5 永久保留）。输出 `instability_selected.csv`（§0.13 更名）。
10. 输出（§14）：`instability_inventory.csv`、`instability_selected.csv`、`loco_full.csv`（field, k, cluster_id, members, rank, loco_angle_deg；1600 行）、manifest 回写、`README.md`（字段字典 + 不适用字段说明）。

### 3.3 验收

对账哨兵通过；inventory 数值列零 NaN；`loco_full.csv` 行数 = 5×2×160 = 1600；README 齐备。

## 4. `02_instability_montage.py`（Phase 2A-2，两轮盲评）

- **第一轮（盲评）**：对 `instability_selected.csv` 每样本出 `round1/AUDIT-<seq:02d>_blind.png`（3×3 panel：A absolute H、B residual R、C–F 四 DCT band、G repair mask、H 中心行列 profile、I 径向 PSD）。**匿名编号**（按 consensus 降序分配 AUDIT-01..，不显示样本身份/工艺/LOCO）；每 panel 自身 valid 像素 P2–P98 色标。
- **第二轮（unblind）**：`round2/sample_<ddd>_unblind.png`（4×3 = 12 panel，增加 J process 近邻、K morphology 近邻、L 元信息文本：工艺五参数 + 派生坐标 + depth/Sq + 5 场 LOCO + repair/valid/plane/cone）。J/K 用 k=5 近邻（phys 工艺空间 / descriptor 空间），标注距离。
- **fixed group scale**：round2 的 B–F panel 另提供全 selected 集合联合 P2–P98 色标版本（`_groupscale` 后缀），两类色标不得混用（规划 §7.1）。
- Panel I PSD：1.5 的 `dct_lambda_grid` 径向 log 分箱（24 bin，2–160 µm）均方，单位 µm²。
- 汇总 `instability_montage_round1_blind.pdf`（盲评页序）。
- `instability_manual_review.csv` 模板：`dataset_index, anon_code, reviewer, blind_morphology_pattern, blind_artifact_suspected, blind_notes, unblind_artifact_suspected, artifact_reason, morphology_pattern, confidence, notes`。morphology_pattern 取规划 §7.2 的 9 项 checklist，多选 `;` 分隔。**02 不做任何自动分类**。
- **门**：两轮审计未完成前 2A gate 不关闭；04/05 允许 quick 冒烟，但不得冻结任何 2B 结论。

## 5. `03_local_neighborhood_audit.py`（Phase 2A-3）

- 全 200 样本，k=5 邻域背景；D_morph 六口径：total R、4 band（`l15.pairwise_rmse_from_gram`，µm，与 1.5-04 同定义）+ descriptor robust-z 欧氏。ordinary 对 mask：`l15.ordinary_pair_mask`（剔 40 共享源对 + sentinel 对；mask 只依赖样本结构，置换时保持不变）。
- **判据展示**：per 工艺空间 `near = D_proc ≤ P10(ordinary 对 D_proc)`；per 口径 `far = D_morph ≥ P90(ordinary 对 D_morph)`；四类 Type I–IV 逐对判定。
- **连续统计量（主读数）**：`T_λ = median[ D_morph | D_proc ≤ P10 ]`，per (空间 × 口径)。
- **双 null**（1000 次，quick 100）：
  - **Null-1（主）within-session_role**：process 行在 session_role 块内置换（120/60/20 各自内部打乱），保持 session 构成（§0.8，v2 §10.2）。
  - **Null-2（敏感性）global**：200 行整体置换。
  - 每次置换重算 D_proc → P10 阈值 → T_λ 与 Type II 计数（P90 阈值固定为观测 ordinary 对的分位，D_morph 不随置换变化）。
  - `p_perm = (1 + #{null ≥ obs}) / (1 + n_perm)`。
- **formal-only 口径**：Type II 计数与 T_λ 限制在两端均为 formal 样本的对，三口径（within-null、global-null、formal-only）方向必须并列呈现。
- 输出：`neighborhood_pairs.csv`（**宽表** 19859 行：i, j, D_proc_raw, D_proc_phys, 6 个 D_morph, D_over_sentinel ×5 场口径, 12 个 type 列）、`process_near_morph_far_pairs.csv`（Type II 全行）、`process_far_morph_near_pairs.csv`（Type III）、`neighborhood_summary.csv`（per 空间×口径：n_near、双阈值、T_λ、TypeI–IV 计数、p_perm_count、p_perm_T、formal-only 计数与 p）、`phase2A_gate_answers.md` 模板（§16 四问自动填数，结论人工写）。
- `D_over_sentinel = D_morph / D_sentinel(λ)`，D_sentinel 取 1.5 `sentinel_multiscale_table.csv`（不重算）；descriptor 口径留空。展示处必须带注："只有一个 exact-repeat condition，不是全局噪声标准差，不得称 universal noise floor"。

## 6. `04_build_multiscale_targets.py`（Phase 2B-1）

- `multiscale_targets.csv`：200 行宽表，21 个 target：A(1) `median_depth_um`；B(12) descriptors 十项 + `peak_to_valley_p98p2_um, deepest_negative_residual_um`；C(8) `rms_DCT_*_um, E_DCT_*_frac`。任一 target NaN → require fail。
- `band_fields.npz`（5 × (200,160,160) float32，≈98 MiB）：**本地可重建 cache，显式 gitignore，不入库**（§0.15）；04 重跑时覆盖重建。
- `targets_manifest.json`：每个 target_id → {family, definition, unit, source, nan_policy, notes}；family D 声明 fold-internal（§7.5）。
- repaired 版本 target 在 09 现场重算，不双份冻结。

## 7. `05_process_explainability_cv.py`（Phase 2B 核心）

### 7.1 输入空间（§0.2 更名）

- **A**（raw）：(τ, f, h, N, v)。
- **R**（physics-motivated reparameterized）：(τ, pulse_energy_proxy_uJ, scan_spacing_um, areal_pulse_density_per_mm2, areal_dose_proxy_J_per_mm2)。不放 f（与 E_p 严格耦合）。
- **C**（hybrid）：A ∪ R（10 维；共线警告同 v1）。

### 7.2 模型（第一批）

`DummyRegressor(mean)`、`Ridge`（alpha ∈ {0.01,0.1,1,10,100}，**折内选**：train 折内 GroupKFold(3)（groups 同 variant）取 R² 中位最优后重拟合）、`ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, random_state=seed+fold)`。pipeline = StandardScaler → model。

### 7.3 CV 协议（§0.4、§0.7 修订）

- **CV-A**（unseen physical source/surface）：groups = `shared_height_source_id`（160 组）。
- **CV-B**（unseen process condition）：groups = `cv_process_group`（134 组，49/50 强制同组）。
- variant：`gkf` = GroupKFold(5)（确定性主结果）与 `gss` = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=seed+100+i)（稳定性重复）；共 4 个 variant：`A_gkf, A_gss, R_gkf, R_gss`。
- **契约断言（按 split 类型区分）**：
  - 所有 split：`train_groups ∩ test_groups = ∅`；
  - `gkf`：不同 fold 的 test 组两两不交，且并集 = 全部组；
  - `gss`：**不要求**跨 split 不交或并集覆盖；额外输出每 group 被置入 test 的次数（`gss_test_counts` 旁路文件）。

### 7.4 family-D（secondary representation diagnostic，§0.6/§0.9）

- 每 train 折 × 每 band：`l15.gram_pca(X_band_train, 3)` → y_train = train 投影、y_test = test 中心化后投影；折内 cluster bootstrap（B=200/quick 20）算每 PC θ_q50 → `target_flag`（>40° 为 unstable_pc）。
- **跨折模态对齐**：输出 `fold_pc_alignment.csv`——每 band 内所有 fold 对 (a,b)：`theta_pc1_deg` = θ(PC1^(a), PC1^(b))、`theta_pc123_deg` = θ(span PC1:3^(a), span PC1:3^(b))（用 `l15.principal_angles`）。
- **引用规则**：仅当 within-fold bootstrap 与 between-fold 对齐双稳定（θ 中位数分别 <40° / <45°）时，band PC1 的 R² 才可作为"较清晰的 observable"引用；否则 family-D 只作辅助诊断。**Phase 2B 主结论优先建立在 band RMS/energy、Sq、depth 与显式 descriptor 上。**

### 7.5 指标与输出

- 折级 R²、MAE、RMSE、Spearman ρ；汇总 median/Q10/Q25/Q75/Q90。
- `cv_fold_results.csv`（长表：target_id, family, input_set, model, cv_variant, fold, n_train, n_test, R2, MAE, RMSE, spearman_rho；family-D 行附 band, pc_index, theta_boot_q50_deg, target_flag）、`cv_summary.csv`、`gss_test_counts.csv`、`fold_pc_alignment.csv`。

### 7.6 规模与缓存（§0.16 修正）

- outer fits = 21×3×3×4×**5** = **3780**（4 variant 各 5 个 outer split）；family-D = 12×3×3×4×5 = 2160。Ridge 折内 alpha 选择与 family-D bootstrap 增量可观。
- **family-D 的折内 PCA/bootstrap 按 (variant, fold, band) 只算一次并缓存**（字典传递），不随 input/model 重算。
- 预算：全量 ≈ 1–1.5 h。quick：model = {Dummy, Ridge}、variant = {A_gkf}、B=20 → 分钟级。

## 8. `06_physics_coordinate_comparison.py`

- 折配对差：`ΔR²_reparam = R²(R) − R²(A)`、`ΔR²_nonlin = R²(ExtraTrees) − R²(Ridge)`，逐 (target, variant)。
- 报告折中位数、[Q25,Q75]、符号一致折数/总折数（探索性方向一致性，不做 p 值包装）。
- 输出：`raw_vs_reparam_coordinates.csv`、`ridge_vs_tree.csv`、图 6/7。解释逻辑照规划 §18.1，但"物理坐标改进泛化"一律表述为"**重参数化更契合模型 inductive bias**"（§0.2）。

## 9. `07_scale_predictability_summary.py`

- `scale_predictability_summary.csv`：A_gkf 每 (target, input, model) 折间 R² 分位。
- 图 5 主图：横轴 = band 下界 (8, 16, 32, 64 µm) log 轴；系列 = `rms_DCT_*`（4 条）+ depth、Sq 参考横线；panel 按 model。by_input 图：ΔR²_reparam(λ)；by_model 图：ΔR²_nonlin(λ)。
- 曲线解读必须同时引用 CV-A 与 CV-B 差异（差距大 → 泛化主要依赖相邻 design condition，规划 §15.3 措辞保留）。

## 10. `08_local_regime_probe.py`（§0 增补：同 held-out 比较）

- **触发条件**：仅当 2A gate 判定"高 leverage 来自真实形貌结构（非 artifact 主导）"后运行。
- **比较设计（v1 的 R²_local − R²_global 已废弃）**：
  - 每个 outer fold（CV-A gkf）内：global model 用**全部** train 折拟合；local model 仅用 train 折中**与 held-out 样本同 stratum** 的部分拟合；两者预测**完全相同的一批 held-out 样本**。
  - 同批样本上比较 `MAE_global,r vs MAE_local,r`，并报 `Skill_r = 1 − MAE_model,r / MAE_dummy,r`（dummy = 同 stratum 训练数据均值，也在**同一 held-out 批**上评估）。
  - 不再使用 R²_local − R²_global（层内 Var(Y|r) 变小会使该差值虚假为正）。
- **outcome-defined stratum 禁令**：depth 四分位不用于评判 depth target；Sq 四分位不用于评判 Sq target；只用于评判其他 morphology target。
- 分层变量：`median_depth_um` 四分位、`Sq_um` 四分位、`A_consensus` 上/下半。
- targets = {median_depth_um, Sq_um, 4×rms_DCT}（跨分层评判时剔除与分层同源的 target）。
- 输出：`local_vs_global.csv`（stratum, target, model, fold, MAE_global, MAE_local, MAE_dummy, Skill_global, Skill_local, delta_skill）、`local_regime_probe.png`。

## 11. `09_sensitivity_checks.py`

四臂，均跑主 target 子集 {median_depth_um, Sq_um, **E_DCT_8_16_frac**, 4×rms_DCT}（7 个；E_DCT_8_16_frac 于 2026-09-03 加入：它是 05 中最强的形貌 target（R² 0.42–0.63），2B gate 必须覆盖其敏感性）× input {A, R} × model {Ridge, ExtraTrees} × CV-A gkf：

1. **raw vs repaired**：target 重算自 height_repaired 残差（band_fields 现场重生成，不落盘冻结）。
2. **formal-only**（n=120，groups=120 单例）。
3. **exclude top leverage**：minus LOCO total PC1 top-1 / top-5（**禁止**把"去掉后更好"解释为应删除样本）。
4. **band-definition 交叉（DoG，§0.11）**：targets = 4 个 Difference-of-Gaussians 近似带 std：`std_DoG_8_16_um`(G4−G2)、`std_DoG_16_32_um`(G8−G4)、`std_DoG_32_64_um`(G16−G8)、`std_DoG_64_inf_um`(R−G16)，σ 取 scales.sigmas_px，−3dB 对应 (≈7.5–15.1, 15.1–30.2, 30.2–60.3, ≥60.3 µm)，与 DCT 带逐带对应。G 累计低通 std **不再**作为 band 交叉证据。

输出：`sensitivity_summary.csv`（arm, target, model, R2_med_base, R2_med_arm, ΔR², sensitivity = stable(<0.05)/moderate(0.05–0.15)/strong(>0.15)，仅描述性）。

## 12. `phase2_config.yaml` 草案

```yaml
# Phase 2 shared configuration (frozen inputs; audit + explainability, low-model-assumption).
random_seed: 20260903

paths:
  dataset_npz: outputs/rectangle_registration/manual_internal_roi_v1/dataset/stable_roi_80um_dataset.npz
  exploration_manifest: outputs/phase1_minimal/exploration_manifest.csv
  output_root: outputs/phase2

power:                               # v2 §11; proxy until measurement record registered
  measured_power_W: 5.3333
  nominal_software_power_W: 10.0
  source: "experiment_background_v2_s11_no_independent_record"
  constant_power_assumption: true

scales:                              # 与 1.5 完全一致，禁止改动
  pixel_um: 0.5
  sigmas_px: [2, 4, 8, 16]
  dct_bands_um: [[8, 16], [16, 32], [32, 64], [64, 1.0e9]]

instability:
  top_n: 20
  top_cap: 30
  knn_k: [5, 10]

neighborhood:
  k: 5
  proc_near_quantile: 0.10
  morph_far_quantile: 0.90
  type2_permutations: 1000
  type2_permutations_quick: 100

bootstrap:
  cluster_key: shared_height_source_id
  n_replicates: 200                  # 05 family-D 折内稳定性
  n_replicates_quick: 20

cv:
  n_splits: 5
  gss_repeats: 5

targets:
  primary_subset: [median_depth_um, Sq_um, rms_DCT_8_16_um, rms_DCT_16_32_um,
                   rms_DCT_32_64_um, rms_DCT_64_inf_um]

models:
  ridge_alpha_grid: [0.01, 0.1, 1, 10, 100]
  extratrees: {n_estimators: 500, min_samples_leaf: 2}

sentinel:                            # 与 1.5 相同，禁止改动
  session: zro2_120_formal
  processing_orders: [49, 50]

plot: {dpi: 150, diverging_cmap: RdBu_r}
```

## 13. `tests/test_phase2_lib.py`（CI-safe，合成数据为主）

1. `test_pulse_energy_units` / `test_scan_spacing_units` / `test_areal_density_and_dose`：§2.3 锚点。
2. `test_power_provenance_required`：manifest power 列与 source 非空（防"未登记先用"）。
3. `test_base_condition_group_merge`：S→T 映射正确；base 组数 135、cv_process_group 组数 134；每行恰一组。
4. `test_cv_process_groups_sentinel`：49/50 同 `cv_process_group`（exact-repeat 不跨折的结构保证）。
5. `test_gkf_contract`：test 组两两不交 + 并集=全部 + 每折 train∩test=∅。
6. `test_gss_contract`：每 split train∩test=∅；**不**要求跨 split 不交；返回每 group 入 test 次数。
7. `test_fold_internal_pca_no_leak`：折内 comps 与直接 gram_pca(X_train) 全等；test 投影 = 中心化后投影。
8. `test_pc_alignment_identical_and_orthogonal`：相同子空间 θ≈0；正交子空间 θ≈90°。
9. `test_band_sum_is_masked_reconstruction`：4 band 场之和 == idctn(C·(λ≥8))。
10. `test_dog_band_localization`：单一波长合成场（λ=10 µm）的 std 集中在 DoG_8_16。
11. `test_knn_self_exclusion`。
12. `test_loco_outlier_max`（合成）。
13. `test_sentinel_normalization_value`：D_over_sentinel 手算对账。
14. `test_consensus_uses_min_band_rank`：谱项 = 四带 rank 最小值。
15. `test_process_near_morph_level`：T_λ 统计量手算对账。
16. `test_targets_align_manifest`（04 实现后启用）。

目标：全套 `python -m unittest discover -s tests` 通过。

## 14. 输出树

```text
outputs/phase2/
  manifest/
    phase2_manifest.csv
    README.md
  instability/
    instability_inventory.csv
    instability_selected.csv
    loco_full.csv
    instability_manual_review.csv        # 02 生成模板,两轮人工填写
    phase2A_gate_answers.md              # 03 生成模板,人工填结论
    README.md
    round1/ AUDIT-xx_blind.png
    round2/ sample_xxx_unblind.png [+ _groupscale.png]
    instability_montage_round1_blind.pdf
  multiscale_targets/
    multiscale_targets.csv
    targets_manifest.json
    band_fields.npz                      # 本地 cache,gitignored,不入库
  process_explainability/
    cv_fold_results.csv
    cv_summary.csv
    gss_test_counts.csv
    fold_pc_alignment.csv
    raw_vs_reparam_coordinates.csv
    ridge_vs_tree.csv
    scale_predictability_summary.csv
    permutation_importance.csv           # 第二批
    ale/                                 # 第二批
  local_structure/
    local_vs_global.csv
  sensitivity/
    sensitivity_summary.csv
```

## 15. 运行顺序与时间预算（formal 参数，§0.16 修正）

```text
01 inventory      ~8 min    （全量 LOCO + 对账哨兵）
02 montage        ~5 min    （两轮盲评模板;其后人工审计 = 2A gate 必要条件）
03 neighborhood   ~5 min    （双 null × 1000 次置换,D_morph 预计算）
── 2A gate 人工关闭 ──
04 targets        ~5 min    （band_fields.npz 为本地 cache）
05 cv             ~1–1.5 h  （3780 outer fits + family-D 缓存后 2160）
06/07             <5 min
08 local probe    ~30 min   （视 2A gate;同 held-out 比较）
09 sensitivity    ~45 min   （4 臂 × 主 target 子集,DoG 臂含 Gaussian 场重建）
```

每步 formal 完成后 `git commit`（中文信息，惯例同 AGENTS 约定）；quick 冒烟不产生 commit。

## 16. Phase 2A / 2B 验收门

**2A gate**（03 完成后、04 正式运行前）：

1. 高 leverage 是否 artifact 驱动？→ 两轮盲评 manual_review.csv + cone/plane/repair 诊断列。若 artifact 主导：**先修 preprocessing（回到 registration 流程），2B 暂停**（规划 §10.A）。
2. 高 leverage 是否集中在某类真实形貌结构？→ 盲评 pattern 分布 + gate_answers.md 第 2 条。
3. 是否只是连续幅度尾部？→ leverage 与 Sq/peak_to_valley 的 rank 关系 + Type I/IV 占比。
4. process-near/morphology-far 是否真实存在？→ **T_λ 与 Type II 在 within-session null 下的 p_perm（主）**、global null 与 formal-only 口径方向、P15/P85 扰动稳健性。

gate_answers.md 四条**必须**有非空人工结论 + 证据文件/行，gate 才算关闭。

**2A gate provenance（2026-09-03 修订）**：视觉审计允许 AI 辅助执行，但 gate 文件必须记录 reviewer 身份、盲评→揭盲流程与用户确认；用户对结论承担最终责任，gate 文件落盘于 `outputs/phase2/local_structure/phase2A_gate_answers.md`（canonical；盲评原始记录归档于 `outputs/phase2/instability/盲评/`）。本次 gate：reviewer = GPT-5.6 Sol（两轮盲评，AUDIT-01..28），用户于 2026-09-03 接受其结论并指示进入 2B，故记为 CLOSED（AI-assisted audit, user-accepted）；如需严格人工口径，可随时对 round1 盲评页重新审计，结论以重审为准。

**2B gate**（09 完成后）：R²(λ) 结构 + sensitivity 稳定性 + CV-A/CV-B 差距 + formal-only 一致性；输出 `outputs/phase2/phase2_gate_summary.md`（规划 §29 十二问逐条作答，每条附证据文件与列名）。

## 17. 路线决策门（§0.10 降强度后）

预设的**方向性阈值**（非通过/失败判据），全部读自上表文件；只允许得出 §0.10 列出的四类候选结论之一，不得升级为机制结论：

- **Route N（nonlinear mapping priority）**：within-session null 下 Type II p_perm > 0.05 且 T_λ null 方向一致；ΔR²_nonlin 折中位 ≥ 0.1 的 target 覆盖 ≥ 半数主 target；09 各臂方向一致。**只支持"非线性映射优先"，不支持 "deterministic"**——后者必须等 repeatability matrix（v2 §25.1）。
- **Route S（scale-dependent predictability）**：**仅 morphology**：R²_CV(≥64 µm) − R²_CV(8–16 µm) ≥ 0.2（ExtraTrees、A_gkf），或 macro/fine 带 R² 中位差 ≥ 0.2；且 09 的 raw/repaired、formal-only、DoG 三臂方向一致。**depth 只作参考 panel，永不参与触发**。
- **Route H（local heterogeneity / regime candidate）**：2A gate 第 2 条成立 + 08 在同 held-out 批上 Skill_local − Skill_global ≥ 0.10 且跨折方向一致 + 该层样本跨 session 重复出现。
- **Route U（unresolved morphology branching / replicate-needed）**：within-session null 下 Type II p_perm ≤ 0.01 且 T_λ 一致，且对 P10/P90 → P15/P85 扰动稳健；下一步 = v2 §25.1 repeatability matrix，之后才允许讨论 missing-coordinates vs repeat-level branching 的分叉。
- 多条可同时触发；由 phase2_gate_summary.md 并列陈述，不做单一强制路由。

## 18. 结论措辞边界

继承规划 §30/§31 与 v2 §17 全部条目，另加：

- Input Set R 表述为 "physics-motivated reparameterized coordinates"；禁止"增加了物理信息/新物理坐标"类语言（§0.2）。
- 禁止解读 f 或 E_p 的"独立效应"；变量重要性图中两者 rank 互换不作为发现。
- 能量/剂量列在功率 provenance 补齐前必须带 `proxy` 与 provisional 标注。
- 所有 R² 表述为 "exploratory cross-validated explainability estimate (n=200)"；禁止"验证了模型/建立了预测模型"（v2 §17）。
- 距离表述用 "sentinel-normalized morphology distance"；禁止 "noise-normalized / universal noise floor"。
- N=5–6 相关观察必须带 "session 与 pass count 完全混杂（v2 §10.2）" 声明；轨迹表述延续 1.5：cross-sectional pseudo-trajectories。
- 延续 Phase 1.5 撤回令：不写 "pass reversal / oscillation"。
- cluster/regime 名称保持数据驱动（R1/R2…），直至外部物理证据支持机制命名。
- 禁止由 Route 结果直接推出 "deterministic / stochastic / hidden state"（§17）。

## 19. 待确认问题（评审时决定）

1. **功率登记**：power_measurement_version（测量日期/记录号）待补；补齐前 E_p/D_E 列带 `proxy` 后缀（§0.1）。
2. **Type II 阈值**：P10/P90 展示阈值是否接受（Route U 已内置 P15/P85 扰动检验）。
3. **GPR/GAM 第二批**：建议 05 主结果后决定。
4. **两轮盲评**：reviewer 与时间安排；两轮完成前 2B 只允许 quick 冒烟。
5. **T/S 组合并**：若认为 supplement 装夹/漂移差异足以拆开 T/S，`base_condition_group`/`cv_process_group` 的 pass 侧改回 T、S 分组（组数 145），需同步改 §2.2 与单测锚点。
