# Phase 2.6 落地执行细则（how）

> 状态：**FROZEN_EXECUTED**（2026-09-04）。v1 外审结论：总体研究设计成立，但存在 6 项 freeze blocker（M0 对账不可同时成立 / blind QA 破盲 / censored 线污染 G-SL1 / exact-match 应升级为 direct bridge / G-SL3 混淆变量变换与机制 / Aitchison Q² 误用于 scalar target）与 2 项强建议（H2 改用有效 λ_peak、shuffled-h 保持 DOE block 结构）。**v2 已全部落实（§0 rev2 变更记录）；v2 外审复核通过后随预冻结提交生效 FROZEN_EXECUTED——预冻结提交 = 本提交：本细则 + `phase2_6_config.yaml` + `15–20` 脚本（15/16 为完整实现，17–20 为冻结骨架）+ `tests/test_phase2_6_lib.py`（17 项：15 过 + 2 个 SkipTest 锚点）。此后宽度定义、λ\* 窗口/guard、G-SL1~G-SL3 门槛、种子禁止改动（§0.14）。**
> 上位规划：`任务说明/Phase2.6_单线扫描尺度溯源_研究任务说明.md`（DRAFT_FOR_REVIEW）——它定 what/why（RQ1–RQ3、H1/H2/H3、Task SL-01~SL-05、Gate G-SL1~G-SL4、8 个必答问题）；本文件定 how。两文件冲突时以本文件为准并回写上位规划；**v2 对上位规划的实质性回写项**：G-SL3 语义改为 Geometry-compression Gate（§0.13）、新增 SL-03a exact-match direct bridge（§0.17）、H2 主证据改为有效 λ_peak（§0.18）、H2 命名弃用 harmonic（§14）。
> 事实基线：`现有数据基础说明_v2.md`（§4 单线 DOE、§11 功率、§12 弓字形填充）；`experiments/phase2_5/Phase2.5_落地执行细则.md`（FROZEN_EXECUTED）与 `outputs/phase2_5/summary/phase2_5_gate_eval.md`（rev3：G1 ROBUST、G2a VALIDATED、G2b SUPPORTED → Route P+T 触发）；`outputs/zro2_single_line_pilot/pilot_protocol.json`（单线 pilot 冻结约定）。
> 继承原则：Phase 2.5 的 height_raw 主证据、冻结平面、DCT 五段带（8–16 µm = λ∈[8,16)）、ILR z1–z4、src_gkf/proc_gkf CV 契约一律**原样复用、不重定义**；单线 pilot 的 removal-depth 符号约定（`removal depth = robust reference plane − measured height`）与阈值约定（`D > 4×1.4826×MAD`）直接继承；N4→5 依旧禁止任何 step 分析；单线数据不得替代 Phase 2.5 的 200 面形貌主数据。

---

## 0. 相对上位规划的收紧点（差异决策登记，rev2）

以下每条都在对照仓库核查后做出（核查日期 2026-09-04，核查证据见括注）。改动任何一条必须回写本节。
**rev2 变更记录**：重写 §0.7（blind montage 禁带 8–16 信息）、§0.13（G-SL3 改 Geometry-compression）、§0.16（M0 对账拆分）、新增 §0.17–§0.20（direct bridge / H2 主证据 / shuffle block 结构 / 免旋转直接采样）；§0.3 加入 `width_identifiability`。

1. **跨数据集条件覆盖是本阶段最大的硬约束（决定 Ŵ 生成方式）**。矩形 200 样本（`outputs/phase2/manifest/phase2_manifest.csv`）与单线 120 条（`氧化锆/氧化锆_line_design.csv`）的 (τ,f,v,N) **精确匹配只有 19 个条件 / 20 个样本**（10%）。矩形网格更大：τ ∈ {223,500,1000,2000,4000,**6000**}、f ∈ {2,5,10,20,40,**50,100,200**}、v ∈ {**3**,5,…,25}、N ∈ {1..5,**6**}；单线网格 τ ∈ {223..4000}、f ∈ {2..40}、v ∈ {5..25}、N ∈ {1..5}。按单线盒（τ∈[223,4000] ∧ f∈[2,40] ∧ v∈[5,25] ∧ N∈[1,5]，闭区间）判定，**101/200 在盒内、99/200 在盒外**（24 个 τ=6000、72 个 f>40、24 个 v=3、10 个 N=6，集合有交叠）。因此：
   - `Ŵ_line` 的主口径 = **用单线数据拟合的工艺模型预测**（Task 17 → Task 18），不是查表；
   - 每个矩形样本登记 `bridge_coverage ∈ {exact_match, in_box_pred, out_of_box}`；**primary bridge 只用盒内 101（含 exact_match 20）**，盒外 99 只进 extrapolated sensitivity 臂并显式标注；
   - exact_match 子集同时承担两项角色：**(a)** Ŵ 模型 vs 单线实测查表的一致性 QA（无外插）；**(b)** 升级为独立的 direct measured-W bridge（SL-03a，§0.17）——这是测量→测量的直接证据，优先级高于一切模型代理证据。
2. **`lambda_star` 是新造量，Phase 2.5 没有对应列**（全仓核查：仅有 `spectral_centroid_um / spectral_centroid_log_um / lambda_peak_um`）。冻结两个受限波长量：
   - **λ\*_4–32（centroid 型，H1/H3 主口径）**：λ\* = exp( Σ_{b: λ_geo,b ∈ [4,32)} E_b · ln λ_geo,b / Σ E_b )，b 为 `outputs/phase2_5/spectral_composition/radial_spectrum_long.csv` 的 24 个 geomspace bin（E_b = energy 列，non-DC）；**guard**：Σ_{b∈[4,32)} E_b / Σ_all E_b ≥ 0.10，否则 NA（`lambda_star_valid=False`，样本退出比值检验，计数入 QA）。
   - **λ_peak_4–32（peak 型，H2 主口径，§0.18）**：取窗口内 E 最大的 bin 的 λ_geo；有效性双条件：该 bin `n_modes ≥ 20` **且** `E_peak / Σ_{b∈[4,32)} E_b ≥ 0.20`（峰 bin 须持有窗口能量 ≥20%，否则"宽谱无峰"不得冒充周期证据）；无效 → NA（`lambda_peak_valid=False`）。
   - 全局 `spectral_centroid_um`（既有列）作为第二 sensitivity。禁止把 λ\* 或 λ_peak 称为"物理光栅波长"；与 W_line、h 的比较只支持"尺度一致性"语言。
3. **单线横向视场仅 17.83 µm（64 px × 0.278657 µm/px），截面宽度 censoring 是本批数据最大的统计风险，必须显式建模**。冻结规则：逐截面逐 q 记 `censored_q`（d≥q 的 run 触及剖面 v 边界或 v 点落在原始 FOV 外，§0.20）；线级新增三态字段：
   - `width_identifiability = estimable`：`n_sections_used ≥ 20` 且 `censored_frac_W50 ≤ 0.5`；
   - `right_censored`：`censored_frac_W50 > 0.5` → 该线 W50_obs 只作为**下界**（W_true > W50_obs）报告，禁止把剩余 uncensored 截面的 median 当作完整线宽；
   - `insufficient_sections`：`n_sections_used < 20`。
   `right_censored` / `insufficient_sections` 线**不得进入 G-SL1 的任何主统计量**（pooled median、带内比例、W_eq 一致性），只进 lower-bound 报告与 `width_identifiability_summary.csv`（含按工艺条件的 censored 分布）。理由：真实 W=20 µm 的线在 17.83 µm 视场下大多数截面 censored，偶发较窄截面会给出假性"带内"读数，构成对 8–16 的选择偏差。（pilot `qc_rules.edge_clipped` 为 review 级，本条硬化为可计算规则。）
4. **阈值宽度的离散实现冻结为"最长连续 run + 子像素插值"**。threshold crossing 可出现多个区间（侧壁锯齿、局部台地），冻结：W_q = d_n ≥ q 的**最长连续 run** 长度（px × 0.278657 µm，两端线性插值到阈值）；辅列 `n_runs_q`、`total_qualified_width_q` 全量保存，禁止只报合计。W20 ≥ W50 ≥ W80 作为硬 QA。
5. **高度版本双轨**：raw 为主证据（`CagHeightReader` 原始 z + 冻结平面 → removal depth）；cone-repaired 为 sensitivity（cone 参数沿用 pilot 冻结：`half_window_px=12, seed_sigma=6, grow_sigma=1.5, min_seed_depth_um=0.8, max_span_px=36`，经 `src/conical_dropout.py` 对全 120 组现算）。G-SL1 以 raw 判定；若 repaired 臂方向不一致（estimable 线带内比例差 > 0.10），G-SL1 降注 `raw-repaired divergent`，不得静默。
   **rev2 补注（2026-09-04，实施前登记）**：pilot 的锥修复实现已随删除脚本离开工作区，其参数形状（`half_window_px`/`max_span_px`）与现存 `ConicalDropoutConfig` 不同构；按**阈值公式同构**映射实施（pilot 实测阈值 seed=max(0.8, 6×noise)、grow=max(0.35, 1.5×noise) 与现模块公式一致）：`seed_sigma=6.0, grow_sigma=1.5, minimum_seed_depth_um=0.8, maximum_span_um=36 px×0.278657≈10.0 µm`；`boundary_protection_um` 由模块默认 5.0（为 80 µm ROI 设计）覆盖为 **1.0**（17.8 µm 窄 FOV 适配）；其余字段取模块默认。以 `outputs/cone_repair_inventory/inventory_config.json` + 15 组记录为对账参考。另冻结：**extent 与截面位置只由 raw 臂定义，repaired 臂在同一 (s, v) 采样点重提特征**（修复不得重定义几何）。
6. **几何提取不依赖进行中的单线盲标**。`annotations/single_line_range_annotation.csv` 仅 20/120 complete（2026-09-04），不得作为数据依赖；线轴用 `annotations/single_line_view_manifest.csv` 的冻结 `theta_line_deg` 与 `orientation_center_x/y_um` 锚点（`orientation_confident=False` 时降级为 `low_confidence` 标注，不弃样）。盲标全部完成后仅作 QA 交叉核对（`long_axis_um` vs 检测线长），不作数据源。
7. **人工 QA 双盲，montage 严禁携带目标带信息（rev2 重写）**。人工标签三值制 `usable / uncertain / reject_geometry`；QA montage 六面板只含几何信息：绝对高度 | 校正轮廓 | W20/50/80 标注 | W_eq | **W50 截面分布（纯分布，无 8 µm/16 µm 参考线、无带阴影、无任何带标记）** | mask/artifact。**任何**展示"宽度 vs 8–16 带"的图/列（`W_line_distribution_vs_band.*`）都是科学结果图，只能在人工 QA 全部完成并提交后由 Task 17 另行生成，禁止提前进入标注流程或 montage 目录。标注者不得知晓哪些线宽接近 8–16 µm（上位规划 §8 红线；测试 T22 锚定 montage spec 无带引用）。
8. **SL-05 方向 provenance 判定：预期 NOT_APPLICABLE**。核查 `现有数据基础说明_v2.md` §12：只记录"矩形区域采用弓字形填充"，**无逐样本 scan 方向、无填充轴（x/y）记录**；`annotations/session_geometry.csv` 的 `theta_session_deg`（−0.45°~−0.70°，d4=identity）是图像旋转约定，不是扫描方向；frozen 配置（`manual_registration_200.csv`、`measurement_planes_160.csv`）均无方向字段。单线侧线方向≈0°（−0.43°~−0.77°）但**起点/终点符号无记录**，且单线无 hatch。因此 Task 20 按"provenance 缺失"路径运行：G-SL4 = **NOT_APPLICABLE**，仅输出 image-frame 的 `theta_stripe(8_16)` 0°/90° 聚集 **descriptive** 检查，并强制注记"不得作为 scan/hatch 对齐证据"。若日后人工确认填充轴，须回写本节并升级 conditional arm，不得在 formal 后追加。
9. **f 与单脉冲能量完全耦合的解释约束**（v2 §11：P 固定 → f 效应与 Ep=P/f 不可分离）。Task 17 响应曲线与一切文字结论中，f 的效应一律写作 "f (Ep-coupled)"，禁止单独归因为"重复率效应"或"脉冲能量效应"。
10. **功率 provenance 弱但两数据集取同一实测值**。单线 `power_w=5.333`（pilot protocol；v2 §11 measured post-objective 5.3333 W）；矩形 `measured_power_W=5.3333` 但 `power_measurement_source=experiment_background_v2_s11_no_independent_record`、`power_measurement_version=PENDING_REGISTRATION`。同功率条件成立（同一来源数值），桥不被阻断；但 manifest 如实登记来源字段，gate 文档结论必须复述"功率无独立测量记录"的弱点。
11. **`氧化锆/72组单脉冲直线.cag` 全程排除**：无设计表、无参数记录 → 无 provenance，按上位规划 §4 红线不得进入任何定量分析；manifest 阶段仅登记一条排除说明。
12. **脚本编号 15–20，`_lib` 链式加载**。延续 phase 系列连续编号（Phase 2 = 01–09，2.5 = 10–14，2.6 = 15–20）；与 `scripts/33/34`（单线标注器）是不同命名空间，勿混淆。`experiments/phase2_6/_lib.py` 用 importlib 以独立模块名加载 `phase2_5/_lib.py`（后者已链式加载 phase2 与 phase1_5），**禁止复制实现**；phase2_6/_lib 只新增：轴向直接采样、宽度函数、λ\*/λ_peak 计算、盒内判定、block 结构置换。
13. **G-SL3 语义重定义（rev2 核心变更）：从"Ŵ 增量 = 机制"改为 Geometry-compression Gate**。理由：Ŵ = g(log₁₀τ, f, v, N)（Task 17 特征含 log τ），若 M1 = Y~[u, Ŵ] 比 M0 = Y~u 好，增量可能只是 Ŵ 偷带入了 log τ 非线性基，而非"线宽是几何中介变量"。冻结三层处理：
   - **M0b 变换对照**：Y ~ [u, log₁₀τ]（把生成 Ŵ 用到的全部 primitive basis 加入但不加 Ŵ）。Ridge 下 Ŵ 恰为 M0b 基的线性组合 → **ΔR²(M1−M0b) ≈ 0 是数学必然**；该量只作 confirmatory log（|median Δ| ≤ 0.02 预期，超限查实现，作 warning 不作 gate）。
   - **M1−M0 降级改名**：ΔR²(M1−M0) 一律称 **LOW-CAPACITY REPRESENTATION GAIN**，descriptive 报告，**不得**作为 overlap/width 机制证据，不得写入任何 gate 判定语。
   - **G-SL3 主判定 = Geometry-compression Gate**：比较 Y~u（5 个原始工艺量）与 **M_GEO: Y~[Ŵ, h, Ŵ/h]**（仅 3 个几何量）。retention_k = fold-paired median of CVperf(M_GEO)/CVperf(M0)（retention 仅在 M0 median perf ≥ 0.10 的 target 上定义，否则该 target 记 `retention_undefined` 并注明）。SUPPORTED = composition 的 Q² retention ≥ 0.80 **且** scalar 主 target（p_8_16、A2_8_16、angular_entropy_8_16）retention 的 median ≥ 0.80（均 src_gkf、盒内 101），且 ≥4/5 折 retention ≥ 0.60；proc_gkf retention ≥ 0.70 否则降 PARTIAL；retention ≥ 0.90 另记 strong tier。ilr_z2 与 λ\* 的 retention 一并报告但不进主判定。判读语言冻结："五维工艺关系可压缩为单轨宽度–hatch overlap 几何"，**不是**"加入 Ŵ 提高了预测"。
14. **预冻结协议**（对齐 Phase 2.5 §0.8）：本细则 + `phase2_6_config.yaml`（含全部宽度定义、λ\*/λ_peak 定义、G-SL1~G-SL4 门槛、种子）在**任何 formal 运行之前**一起 commit（DRAFT_FOR_REVIEW → FROZEN_EXECUTED）；此后禁止改动宽度定义、λ 窗口、guard、门槛（上位规划 §20.12、§21）。工作区当前有未提交改动（annotations 单线标注进行中），冻结提交只允许包含 phase2_6 的 config + 细则 + 脚本骨架，不得混入标注改动。
    **偏差登记（2026-09-04 仓库审计后回写，供 gate 文档逐字复述）**：预冻结提交 `2daa611` 之后、Task 16 formal 期间，`phase2_6_config.yaml` 发生过实质改动（`git diff 2daa611 HEAD`）。冻结红线本身（宽度定义 `thresholds_q/primary`、λ 窗口与 guard、G-SL1~G-SL4 门槛数值、种子）**未被触碰**；改动落在两处：
       (i) `stable_region` 由 central-70% 改为深度台地最长段 + 碎片守卫（见 §0.15 rev2 补注 (a)，属预处理/选区规则，会改变哪些截面进入宽度统计）；
       (ii) `pilot_reconcile_min_agreement: 0.90` 的 **abort 门被降级为 informational 清单**（见 §0.15 rev2 补注 (b)）。
       其中 (ii) 是**预注册 abort 条件在看到首次 formal 结果（对账一致率 0.70–0.97）之后被放宽**，属实质性偏差。其理由（pilot 精确算法随删除脚本丢失、边界不可局部复现，改用 qualifying-only 布点 + 碎片守卫 + 人工三值 QA 三条结构性机制作污染保证）论证充分且已登记，但**记录时机在 formal 首跑之后**这一事实不得隐去。`phase2_6_gate_eval.md` 终判必须显式声明本条，并说明放行依据是对账之外的结构性机制而非对账本身。
15. **pilot 15 组仅作协议与对账源，formal 数据全 120 现算**。仓内无持久化单线高度缓存，全部经 `io_cag.CagHeightReader` 现读；Task 16 的稳定区判定必须与 pilot 的 `included_in_stable_region`（`outputs/zro2_single_line_pilot_cut_only_stable_region/pilot_longitudinal_profiles.csv`，15 组）做逐截面对账，一致率 ≥ 0.90，否则 abort 修实现，禁止带病放行。
    **rev2 补注（2026-09-04，formal 首次运行后、任何 gate 使用前登记——属对账 QA 与稳定区实现修正，不动 G-SL1~G-SL3 科学门槛）**：首次 formal 对账一致率仅 0.70–0.97，诊断显示 pilot 旗标不是固定中心占比裁剪，而是**逐线数据驱动的稳定裁剪**：排除段是**深度坡/过渡区**（组 116 端部 depth_p95 ≈5 vs 台地 15 µm；组 104 浅坡 ≈3.8/7.0 vs 台地 10.8 µm）外加个别深而窄段（组 104 [−75,−57] 深度 11.9 但绝对阈值宽度 7.04 < 台地 7.68）；其精确算法已随删除脚本丢失。原 frozen central-70% 会把浅坡宽度（系统性偏窄）送进 Ŵ 模型与 G-SL1——正是外审第 3 条担心的选择偏差方向。据此：
    (a) **稳定区判据修正**为有原理的自有规则（§4.1 rev2）。**2026-09-04 审计回写：本条原文与最终实现不一致，以下为实跑口径（与 `phase2_6_config.yaml::single_line.stable_region` 一致），原文的数值保留在括注中作变更留痕。** 实现：`stable = 最长连续段 {on-line ∧ depth_p95 ≥ `**`depth_frac=0.50`**` × `**`P90`**`(depth_p95 | on-line)}`，外加碎片守卫 `len(run) ≥ max(min_stable_len_um=60 µm, min_stable_frac=0.50 × on-line 跨度)`，否则判 `FragmentedStableRegion → insufficient_sections`。三项与原文的偏离及理由：
       - 深度门槛 **0.80 → 0.50**，参考分位 **median → P90**（`ref_quantile: 0.90`）：median 会被斜坡段系统性拉低，P90 才稳健代表台地深度；
       - **绝对阈值宽度条件已撤销**（`width_band_frac: null`，原文为 `≥ 0.95·median(绝对阈值宽度|on-line)`）：规模化时该条件经锥坑分裂的 run 振荡把数十条健康线切碎，得不偿失；组 104 的"深而窄"分歧降级为 informational 清单项，由深度规则 + 碎片守卫兜底；
       - 桥接间隙 **≤2 µm → `gap_merge_um=10 µm`**：5 点内部浅凹造成 6 µm 间隙，曾把组 5 型健康线切成 104+86 两段；真实端部瞬态是 20–40 µm 块，10 µm 仍挡得住。
       central-70% 作为"深度坡"证据的 pilot `central_fraction=0.7` 记录保留在案，不再作为几何规则。宽度报告值仍是冻结的相对 d_n 族，绝对阈值宽度只用于选稳定区。
    (b) **对账降级为纯清单（终稿）**：逐步诊断证明 pilot 排除边界不可局部复现（其两段式 crater 管线已删除），且在管道对齐下滞留计数的"硬/软"分级随参考分位定义（online vs kept）漂移——继续把对账作为 abort 门会变成定义边界的 chasing。**有约束力的污染保证是对账之外的结构性机制**：(i) 截面只落在 qualifying 位置（dp ≥ 0.5·P90(on-line)，桥接凹陷内不布截面→部分消融材料上按构造无截面）；(ii) 碎片守卫（run < max(60 µm, 0.5×在线跨度) → `FragmentedStableRegion` → insufficient_sections + 人工 QA；首跑实证拦截 36 条病理/浅弱线，其中组 48 为离散模式双坑线）；(iii) 人工三值 QA 标签。pilot 对账输出（互相关预对齐 ±30 µm + 严重度分级 `n_hard_invaded`（dp<0.25·ref_kept）/带边清单 + precision/recall/agreement）全部保留在 `stable_region_reconciliation.csv` 供审阅，不再 abort；config 键改为 `pilot_reconcile_hard_depth_ratio: 0.25`（分级定义）。桥接阈值 10 µm（5 点内部浅凹的间隙为 6 µm，曾把组 5 型健康线切成 104+86 两段；真实端部瞬态 20–40 µm 仍挡住）。
    (c) W50 对 pilot `W_line_um` 的对账保持提示级（pilot 为绝对阈值宽度，本阶段报告值为相对 d_n 阈值宽度）。
16. **M0 对账拆分（rev2：v1 的"盒内 101 重新分折 + 复现全 200 数值"在数学上不能同时成立）**。拆为两条独立轨道：
   - **M0_RECON_FULL200（纯 QA）**：用与 Phase 2.5 Task 12 完全相同的 200 样本、相同 groups（`shared_height_source_id`）、相同 input set A、相同 target 集、相同特征变换重跑 M0（Ridge, src_gkf, 5 折），与 `outputs/phase2_5/process_map/cv_fold_results.csv` 对应行 Δ ≤ 0.005；失败 abort 修实现。它只证明"实现无漂移"，不产生任何科学结论。
   - **M0_PRIMARY_INBOX101（正式科学基线）**：盒内 101 子集上重新生成 splits 的 M0；**不与 Phase 2.5 数值强行对账**（样本数与 fold 组成本就不同）；G-SL3 的 retention 分母用的是它。
17. **SL-03a：exact-match direct bridge（rev2 新增；本阶段最强的直接物理证据）**。对 20 个 exact-match 矩形样本 / **19 个独立工艺条件**，绕过 Ŵ，直接构造 r_W_direct = λ\* / W_line_measured（W_line_measured = 对应单线在 Task 16 的线级 median_W50）。实测重复条件核查（2026-09-04）：唯一重复为 `dataset_index 54（zro2_120_formal, 加工顺序 55）`与 `156（zro2_60_pass, BASE:T12）`，共享 (2000 fs, 10 kHz, N4, 5 mm/s) 但 **h 不同（10 vs 8 µm）**→ 该条件两条 λ\* 取均值入条件级统计并记录 spread 与 h 差异；非 49/50 sentinel。统计单位 = 19 个条件（Spearman/median/IQR/P(|r−1|≤0.25) 均按 n=19）。**rev2 补注（2026-09-04，仓库审计后回写：`W_lower_bound` 语义不足，新增 `W_unavailable`）**：原句只写了"≠ estimable → `W_lower_bound`"，但 `W_lower_bound` 的"真值 > 观测值"方向性**只对 `right_censored` 成立**。Task 16 formal 实测：`width_identifiability` = estimable 84 / insufficient_sections 36 / **right_censored 0**，且实测 censoring 比例极低（W20 0.00%、W50 0.01%、W80 0.03%；W20 最大 12.33 µm < 17.83 µm 视场）—— §0.3 预注册的最大风险（视场截断）**没有发生**，真实瓶颈是碎片守卫。因此 direct bridge 遇到的非 estimable **全部是 `insufficient_sections`**，这类条件**根本没有宽度估计，不是下界**。冻结两态：
   - `right_censored` → `W_lower_bound`（r_W_direct 按"≤真实值"语义报告，单独列出，不混入主 median 的 robustness 声明）；
   - `insufficient_sections` → **`W_unavailable`**（无条件级 r_W_direct，**不得**赋予任何方向性陈述；统计时从分母剔除，单独列出并报告缺失率）。
   实测 direct bridge 可用性（2026-09-04 审计）：19 个条件中 **13 个可用**（线 2,5,27,55,56,65,70,89,94,105,109,112,116；其中仅 27、70 人工标签为 `usable`，其余 11 条 `uncertain`），**1 个 estimable 但人工 `reject_geometry`（线 63）按 §5 排除**，**5 个 `W_unavailable`（线 10,75,83,90,92）→ 缺失率 5/19 = 26%**。终判与 gate 文档必须显式复述该 26% 缺失率，并按其打折陈述 direct 臂的权重，不得只报 n=13 的统计量而隐去分母 19。**证据优先级冻结：exact-match direct > in-box predicted > out-of-box sensitivity**；H1 终判时 direct 证据权重最高——但权重高不等于覆盖足，缺失率与权重须一并报告。输出 `scale_bridge/direct_bridge_exact_match.csv` + 核心科学图（λ\* vs W_measured 散点 + r_W_direct 分布）。
18. **H2 主证据 = 有效 λ_peak_4–32，不是 centroid（rev2）**。centroid 回答"有效尺度在哪"，不回答"是否存在周期"；宽谱的 centroid 可以落在 12 µm 而 12 µm 处并无峰。因此 H1/H3 主口径用 λ\*_4–32（centroid 型），**H2（hatch 相关周期）主口径用 λ_peak_4–32**（有效性双条件见 §0.2：n_modes ≥ 20 且峰 bin 持窗内能量 ≥ 0.20）；centroid 版 r_h 作为 sensitivity 并排报告，不替代主判定。λ_peak 无效率与无效样本清单入 QA（若 valid 比例 < 0.5，G-SL2 最高 PARTIAL 并注明"峰证据覆盖不足"）。
19. **shuffled-h null 必须保持 DOE block 结构（rev2）**。200 样本不是 200 个可完全交换的设计点：formal 120、pass 60 = 15 base × N1–4、supplement 20 = 10 base × N2。逐行 shuffle 会把同 base 的 N1–4 当成独立 h 指派，人为扩大有效样本量。冻结：置换单位 = **unique (session_id, base_condition_group)**（实测：formal 120 个单行单位、pass 15 个、supplement 10 个；h 在单位内恒定已验证）；每次置换在各 session 块内部重排"单位 → h"指派，单位全体行一起带走；10,000 次（seed+800）；p = (1 + #{A_null ≥ A_obs})/(1 + 10000)。peak 与 centroid 两个统计量各做各自 null。
20. **几何提取 = 原坐标轴向直接采样，禁止整幅旋正（rev2）**。长轴 285 µm × 0.7° ≈ 3.5 µm 横向漂移，占 17.8 µm FOV 约 20%，在固定 1024×64 canvas 内整幅旋转会引入新裁剪与插值假背景。冻结：对每个 s，直接在**原始图像坐标**采样垂直剖面 (x,y) = (x₀,y₀) + s·t̂ + v·n̂，其中 θ = `theta_line_deg`（image frame，+x 右 +y 下），t̂ = (cosθ, sinθ)，n̂ = (−sinθ, cosθ)，锚 (x₀,y₀) = `orientation_center_x/y_um`，v_j = (j − 31.5)·0.278657（j = 0..63）；采样用 `map_coordinates(order=1)`，**落在原始 FOV 外的 v 点直接记 NaN → 计入该截面该 q 的 censoring**（censoring 判定因此更严格、无插值假背景）。整幅旋正只允许出现在 QA 可视化（若使用必须同时输出 `rotated_valid_footprint`，且不参与任何数值计算）。

---

## 1. 通用约定

- **种子**：`random_seed: 20260904`。偏移登记沿用 Phase 2.5 惯例：GSS = seed+100（src）/ seed+200（proc）；ExtraTrees = seed+700+fold；置换（shuffled-h、orientation null）= seed+800；其余无随机。
- **单位**：长度 µm；波长 λ µm；角度 deg（image frame，+x=图像右，+y=图像下）；能量份额无量纲；τ fs、f kHz、v mm/s、N 次、P W。
- **脚本骨架**：每脚本头部 `EXPECTED` 清单 + `_lib.require` 硬断言（`AssertionError: HARD ASSERTION FAILED: ...`）+ 分步 `log`；输出只写 `outputs/phase2_6/<子目录>/`；PNG/montage 与 log 按 .gitignore 处理；CSV/JSON 入库。
- **quick 隔离**：`--quick` 时 `load_config` 把输出根改写为 `outputs/phase2_6_quick/`（复用 p25.load_config 行为）；quick 只跑 15/16 的 15 个 pilot 组 + 17/18/19 的子样，quick 产物不得被任何结论引用。
- **复用链**：`phase2_6/_lib.py` →（importlib）`phase2_5/_lib.py`（`p25`）→ `phase2/_lib.py`（`p2`）→ `phase1_5/_lib.py`（`l15`）。Phase 2.5 特征文件一律**只读直采**（p_8_16、ilr_z*、A2_8_16、angular_entropy_8_16、centroid 等禁止重算覆盖；测试 T17 锚定）。
- **编码警示**：`氧化锆/氧化锆_line_design.csv` 为 gb18030；annotations 下 CSV 带 BOM；读入时显式声明。
- **commit 协议**：每个 Task formal 完成即 commit（中文信息）；细则+config 的预冻结提交先于一切 formal（§0.14）。

## 2. 数据契约

### 2.1 单线侧（120 条，session `zro2_120_line`）

| 项 | 值 / 来源 |
|---|---|
| 高度源 | `氧化锆/120组直线.cag` → `src.io_cag.CagHeightReader`，每 path 一张 **1024×64** 高度图，`dx_um=dy_um=0.278657`（容器 `x_pitch_pm*1e-6`），z LSB 1 nm，`valid_mask` 全 True（120 组 `valid_pixel_ratio=1.0`） |
| 设计表 | `氧化锆/氧化锆_line_design.csv`（gb18030），列 `加工顺序,脉宽_fs,频率_kHz,重复扫描次数,速度_mm/s`；120 行，τ∈{223,500,1000,2000,4000}，f∈{2,5,10,20,40}，v∈{5,10,15,20,25}，N∈{1..5}；**120 个条件互不重复** |
| ID 映射 | `加工顺序 = CAG Path`（user-confirmed，冻结于 `pilot_protocol.json` 的 mapping 字段）；manifest 逐行带 `mapping_provenance` |
| 冻结平面 | `annotations/single_line_view_manifest.csv`（120/120）：`plane_a/b/c, plane_rmse_um, sigma_ref_um, theta_line_deg, orientation_center_x/y_um, orientation_confident, crop_*` |
| removal depth | `D(x,y) = plane(x,y) − z(x,y)`（pilot 冻结 height_sign）；z_bg ≡ 0（平面校正后背景） |
| 检测阈值 | `D > 4 × 1.4826 × sigma_ref_um`（sigma_ref_um 取 view manifest 冻结值 = pilot 的 reference-residual MAD，不重算） |
| hatch | **不存在**（单线 DOE 无 h）→ 所有 λ/h 量对单线为 NA |
| pilot 对账源 | `outputs/zro2_single_line_pilot*/`（15 组：13,19,33,34,43,44,48,51,60,68,94,95,101,104,116） |

### 2.2 矩形侧（Phase 2.5 复用，只读）

| 项 | 值 / 来源 |
|---|---|
| manifest | `outputs/phase2/manifest/phase2_manifest.csv`（200 行，39 列；`hatch_spacing_um ∈ {2,4,6,8,10}`；`pulse_duration_fs/frequency_kHz/pass_count/velocity_mm_s`；`cv_process_group`、`shared_height_source_id`、`base_condition_group`、`session_role`） |
| 组成 | `outputs/phase2_5/spectral_composition/spectral_composition.csv`：`p_lt8, p_8_16, p_16_32, p_32_64, p_64_inf`（p_8_16 ∈ 0.00244–0.9126） |
| ILR | `outputs/phase2_5/spectral_composition/ilr_coordinates.csv`：`ilr_z1..z4`（z2 = <8 µm vs 8–16 µm balance；解读前用 `p25.ILR_A` 核对符号约定） |
| 方向 | `outputs/phase2_5/directional_spectrum/directional_metrics.csv`（长表 `dataset_index,band,A2,theta_k_deg,theta_stripe_deg,angular_entropy`；band=8_16 行即 Route T 特征源） |
| 描述符 | `outputs/phase2_5/spectral_composition/spectrum_descriptor_summary.csv`：`spectral_centroid_log_um, spectral_centroid_um, spectral_entropy, effective_band_number, lambda_peak_um` |
| λ\*/λ_peak 源 | `outputs/phase2_5/spectral_composition/radial_spectrum_long.csv`（24 bins，geomspace 0.7–160 µm；`lambda_geo_um, energy, energy_fraction, n_modes, low_mode_count`） |
| CV | `p25.gkf_splits / gss_splits / check_gkf_contract`；src_gkf 分组 = `shared_height_source_id`（160 唯一源），proc_gkf 分组 = `cv_process_group`；n_splits=5；**桥接子集（盒内 101）上重新生成 splits 并照跑契约** |
| 对账/对齐 | `outputs/phase2_5/process_map/cv_fold_results.csv`（仅 M0_RECON_FULL200 用，§0.16） |
| shuffle 单位 | unique (`session_id`, `base_condition_group`)：formal 120 单行单位 / pass 15 base / supplement 10 base；h 在单位内恒定（已验证） |

## 3. Task SL-01a — `15_build_single_line_manifest.py`

输出：`outputs/phase2_6/single_line/single_line_manifest.csv`（120 行）+ `outputs/phase2_6/single_line/manifest_provenance.json`。

字段映射（上位规划 §4 的 16 个必备字段 → 仓库来源）：

| 字段 | 来源 |
|---|---|
| `single_line_id` | 加工顺序 1..120（= CAG Path） |
| `source_file` | `氧化锆/120组直线.cag` + `cag_path` 序号 |
| `session_id` / `measurement_id` | `zro2_120_line` / `m{加工顺序:03d}`（对齐 view manifest 命名） |
| `pulse_duration_fs, frequency_kHz, velocity_mm_s, pass_count` | 设计表（gb18030 读入） |
| `power_W_or_proxy` | 5.333（pilot protocol）+ `power_source_note`（v2 §11 无独立记录） |
| `pixel_size_um` | 0.278657（`io_cag` 容器头，逐组读取后断言 120 组一致） |
| `line_scan_direction` | `image-frame line axis ≈ 0°（theta_line_deg）；start/end sign unknown`（显式 NA 语义） |
| `measurement_orientation` | `theta_line_deg` + `orientation_confident` |
| `processing_date_or_batch` | `20260528_single_line_batch`（线索级，来自 CAG 内嵌 VK4 路径时间戳；登记 `date_confidence=filename_only`） |
| `height_data_type` | `absolute_height_raw_primary / cone_repaired_sensitivity` |
| `background_correction_status` | `frozen_plane_from_view_manifest` + `plane_rmse_um` |
| `valid_mask_status` | `all_valid`（逐组断言 ratio=1.0） |
| `hatch_spacing_um` | NA（显式） |
| `notes` + 扩展列 | `edge_clipped_risk, cone_repair_available, mapping_provenance, exclusion_note(72组单脉冲)` |

核查清单落实（上位规划 §4 九项 → manifest 布尔/注记列）：同一激光系统（`provenance_same_system=true`，依据 v2 §11 同一实测功率 + 同一 VK4/CAG 测量链；附功率弱点注记）；单位一致（fs/kHz/mm·s⁻¹ 原生）；pixel size 可信（容器头）；完整槽截面与裁剪截断（Task 16 QC 回填 `geometry_qc` 列）；重复位置（无 → `replicates=1`）；背景平面（frozen RMSE 列）；scan direction 可恢复性（`partial`）。

QA 断言：120 行、`single_line_id` 1..120 严格唯一、四因素网格值域与 §2.1 一致、与 view manifest 全量 join 成功、`72组单脉冲直线.cag` 排除说明存在。

## 4. Task SL-01b — `16_extract_single_line_geometry.py`

### 4.1 预处理（rev2：原坐标轴向直接采样，无整幅旋转）

1. 读 CAG → raw z + valid_mask；`D = plane − z`（平面系数取 view manifest 冻结值，**不重拟合**）。
2. 轴系定义：θ = `theta_line_deg`；t̂ = (cosθ, sinθ)；n̂ = (−sinθ, cosθ)；锚 (x₀,y₀) = `orientation_center_x/y_um`（view manifest 冻结值）。
3. 剖面采样：对候选 s（沿 t̂，步长 0.278657 µm 粗扫 → 2.0 µm 正式步长）取 v_j = (j−31.5)·0.278657，j=0..63；(x,y) = (x₀,y₀) + s·t̂ + v_j·n̂，`map_coordinates(order=1)` 从 D 采样；**FOV 外（x∉[0,285.3448] 或 y∉[0,17.8340]，或非有限）→ NaN**。
4. 线检测/端点（pilot 约定）：逐 s 剖面 groove 存在 = 该剖面 max D > 4×1.4826×`sigma_ref_um`；连续 ≥ `min_profile_points=8` 个 s 视为线体；线体范围 [s_start, s_end]，L = s_end − s_start。
5. 稳定区：**central 70%**，s ∈ [s_start + 0.15L, s_start + 0.85L]（与 pilot `central_fraction=0.7` 一致）。**rev2 修正**：改为深度台地最长连续段规则（见 §0.15 rev2 补注 (a)）；central-70% 不再作几何规则。
6. 正式截面：稳定区内步长 **2.0 µm**（预期每线 ≈ 70 个截面）。censoring 判定直接来自步骤 3 的 NaN/贴边情况，无插值假背景。整幅旋正仅限 QA 可视化且必须输出 `rotated_valid_footprint`（§0.20），不进数值。

### 4.2 宽度定义（冻结，逐截面）

- `D_max(s) = max_v D(s,v)`（NaN 忽略）；`d_n(x) = D / D_max`（z_bg ≡ 0）。
- `W_q(s)` = d_n ≥ q 的最长连续 run（q ∈ {0.2, 0.5, 0.8}；px × 0.278657，两端对阈值线性插值）；辅列 `n_runs_q, total_qualified_width_q, censored_q`（run 触及剖面 v 端点或任一端 v 点为 NaN/FOV 外）。
- `A_remove(s) = Σ_v [D]_+ · 0.278657`（µm²，NaN 计 0 并记 `valid_v_count`）；`W_eq(s) = A_remove / D_max`（µm）。
- `W_affected(s)`（次要）：|D| > δ_aff 的最长 run，`δ_aff = max(0.10 µm, 3×plane_rmse_um)`（预注册；只作 descriptor，不进 gate）。
- 硬 QA：`W20 ≥ W50 ≥ W80`（atol 1e-6）、`W_eq > 0` 且有限、`d_n ≤ 1 + 1e-9`（d_n 允许负，min 值入列）。

### 4.3 线级聚合与 width_identifiability

- 逐线：`median/IQR/P10/P90`（仅 uncensored 截面）、`CV_W`、`n_sections_used/n_sections_total`、`censored_frac_W50`、`D_max 分布`、`edge_clipped`（u 方向线端贴场边）、`theta_line_deg`、`L_detected`。
- **`width_identifiability`**（§0.3 三态）逐线判定并写入输出；`right_censored` 线的 W50_obs 一律带 `lower_bound=True` 语义。
- 输出：

```text
cross_section_widths.csv   长表：single_line_id, s_um, D_max_um, W20/50/80_um, n_runs_*, total_width_*, censored_*, W_eq_um, W_affected_um, max_depth_um, left/right_slope, edge_asymmetry, ridge_left/right_um, profile_skewness
single_line_geometry.csv   线级：聚合宽度 + CV_W + width_identifiability + QC 标志 + theta_line_deg + L_detected
width_identifiability_summary.csv   三态计数 + 按工艺条件（τ,f,v,N）的 censored 分布
qa_montages/group_<id>_qa.png  六面板（blind）：绝对高度 | 校正轮廓 | W20/50/80 标注 | W_eq | W50 截面分布（无 8/16 参考线/阴影） | mask/artifact
geometry_qa_labels.csv      人工三值标签（annotator, single_line_id, label, timestamp_utc, comment）
```

### 4.4 pilot 对账与人工流程

- 15 个 pilot 组：稳定区逐截面对账一致率 ≥ 0.90（abort 条件，§0.15）；W50 对账仅提示不作门槛（实现差异记录）。
- cone-repaired sensitivity 臂同流程重跑（§0.5），输出 `*_repaired` 后缀列。
- 人工标签流程：blind montage 全 120 组（§0.7 面板规格，测试 T22）→ 三值标注（可分批）→ 回填 `geometry_qa_labels.csv`。**标注全部完成前，禁止生成任何 vs-8–16-band 的科学图**；完成后由 Task 17 生成 `W_line_distribution_vs_band.csv/.png`。

## 5. Task SL-02 — `17_line_width_process_model.py`

- 样本集（rev2）：
  - **G-SL1 gate 总体 = `width_identifiability == estimable` 的线**（且人工标签 ≠ reject_geometry）；
  - Ŵ 模型训练主集 = estimable；sensitivity 臂 = estimable ∪ uncertain（uncertain 只进 sensitivity，不进 G-SL1）；
  - `right_censored` / `insufficient_sections` 不进任何训练数值（censored 以 lower-bound 形式进 `width_identifiability_summary.csv`）。
- 目标：线级 `median_W50`（raw 主；repaired 敏感性）。
- 特征：主集 u_line = [log₁₀τ, f, v, N]（标准化在训练折内）；C-extra（descriptive）：`pulse_pitch_um = v/f`（mm/s ÷ kHz ≡ µm/pulse）、`E_line_J_mm = P·N/v`、`单线输入能量`（沿用 pilot 派生列定义，禁止新造第二套）。
- 模型：Ridge（主，fold-internal α ∈ logspace(−3,3,13)，内层 GKF(5) 选 α，同 Phase 2.5 spline 管道惯例）；Spline 管道（StandardScaler→SplineTransformer(degree=3,n_knots=4,include_bias=False)→Ridge，基函数只在训练折拟合）；ExtraTrees（sensitivity，seed+700+fold）。
- CV：GKF(5)，groups = `single_line_id`（120 条线互为独立条件，无重复条件 → 天然无组重叠）+ `p25.check_gkf_contract`；GSS(seed+100) 敏感性。
- 输出：`line_width_process_model.csv`（fold 级：fold, model, R2, MAE, alpha）；`W_line_response_curves.csv`（Ridge 单变量响应曲线：W50 vs 各因素，f 列标注 Ep-coupled）；`W_line_distribution_vs_band.csv/.png`（**仅人工 QA 完成后生成**，§0.7）。
- **G-SL1 判定**（raw 臂；总体 = estimable 线）：
  1. estimable 线全部 uncensored 截面 W50 的 pooled median ∈ [8,16) µm；
  2. ≥ 50% 的 estimable 线 per-line median W50 ∈ [8,16)；
  3. estimable 线 pooled median W_eq 同在 [8,16)（同方向）。
  三条全满足且 `n_estimable ≥ 60` → SUPPORTED；三条全满足但 `n_estimable < 60` → 最高 PARTIAL（视场代表性保护）；恰两条 → PARTIAL；≤1 条 → NOT_SUPPORTED。repaired 臂方向不一致（estimable 线带内比例差 > 0.10）→ 加注降级。同时报告 `n_estimable / n_right_censored / n_insufficient` 与 censored-by-process 表。

## 6. Task SL-03 — `18_scale_bridge_model_compare.py`

### 6.1 Ŵ 生成与桥接表

- 用 Task 17 的 Ridge（estimable 主集）refit → 对 200 个矩形样本预测 `W_hat_um`（预测特征 = manifest 的 τ,f,v,N，**无任何 morphology 列**，测试 T10 锚定）。
- `bridge_coverage`：`exact_match`（(τ,f,v,N) 命中单线 DOE 行，20 个）/ `in_box_pred`（盒内非精确，81 个）/ `out_of_box`（99 个）。
- 输出 `scale_bridge/morphology_scale_match.csv`：`dataset_index, tau, f, v, N, hatch_spacing_um, W_hat_um, W_hat_lookup_um(exact_match 才有), bridge_coverage, in_box, eta_h = W_hat/h, lambda_star_4_32, lambda_star_valid, lambda_peak_4_32, lambda_peak_valid, r_W = λ*/Ŵ, r_h = λ*/h, r_h_peak = λ_peak/h, d_int, d_int_peak, |λ*−Ŵ|, |λ*−h|, |λ*−2h|`。
- λ\* 与 λ_peak 从 `radial_spectrum_long.csv` 现算（§0.2/§0.18 定义 + guard）；全局 `spectral_centroid_um` 进 sensitivity 列。

### 6.2 SL-03a：exact-match direct bridge（§0.17，测量→测量）

- 20 样本 / 19 条件；重复条件（54 formal 与 156 pass-T12）条件级聚合（λ\* 取均值、记录 spread 与 h 差异）。
- `r_W_direct = λ* / W_line_measured`（W_line_measured = 对应单线线级 median_W50；按 §0.17 rev2 补注两态：单线 `right_censored` → `W_lower_bound`、`insufficient_sections` → `W_unavailable`，后者不得赋予方向性并从统计分母剔除；实测 19 条件中 13 estimable / 5 W_unavailable / 1 estimable-but-reject（线 63，按 §5 排除）→ 缺失率 26% 须一并报告）。
- 统计（n=19）：median(r_W_direct)、IQR、P(|r−1| ≤ 0.25)、Spearman(λ\*, W_measured)；centroid 版为主，λ_peak 版作 sensitivity 列。
- 输出：`scale_bridge/direct_bridge_exact_match.csv` + 科学图（QA 完成后）：λ\* vs W_measured 散点（标注 19 条件与 8–16 带）+ r_W_direct 分布。
- **证据优先级（冻结）**：exact-match direct > in-box predicted > out-of-box sensitivity；gate 文档与 H1 终判必须按此排序引用证据。

### 6.3 模型比较 M0/M0b/M1/M2/M3/M_GEO（§0.13 重定义）

- targets 与 metric（rev2）：`p_8_16`（scalar R²）、`ilr_z2`（scalar R²）、**`ilr_z1_z4`（multivariate，Q²_Aitchison）**、`A2_8_16`（R²）、`angular_entropy_8_16`（R²）、`lambda_star_4_32`（R²，valid 子集）。**Q² 只属于完整 composition（ilr_z1_z4），禁止用于任何 scalar target。**`Q2_Aitchison` 定义勘误（2026-09-04 审计后对齐）：**采用 Phase 2.5 `12_spectral_process_map._q2_aitchison` 的 ILR 坐标空间定义**——`1 − Σ(z_test−z_pred)² / Σ(z_test−mean(z_train))²`（参考均值 = 训练折 z 均值，同一 Aitchison 几何、无需 compose 回成分空间）；原"ilr_inverse + aitchison_distance"表述作废。
- 模型（主口径 Ridge，线性基）：
  - M0：Y~u（τ,f,h,N,v，5 列）——正式基线（M0_PRIMARY_INBOX101）；
  - M0b：Y~[u, log₁₀τ]——变换对照（含 Ŵ 全部 primitive basis、不含 Ŵ）；confirmatory 检查 Δ(M1−M0b) 中位 |·| ≤ 0.02（warning 级，超限查实现）；
  - M1：Y~[u, Ŵ]——Δ(M1−M0) 只称 **LOW-CAPACITY REPRESENTATION GAIN**（descriptive，禁入 gate）；
  - M2：Y~h；M3：Y~[u, Ŵ, Ŵ/h]；
  - **M_GEO：Y~[Ŵ, h, Ŵ/h]**（仅 3 个几何量）——G-SL3 主角。
- **G-SL3 = Geometry-compression Gate**（§0.13）：retention_k = fold-paired median of CVperf(M_GEO)/CVperf(M0)；retention 仅在 M0 median perf ≥ 0.10 的 target 上定义（否则 `retention_undefined` 注明）。SUPPORTED = composition Q² retention ≥ 0.80 ∧ scalar 主 target（p_8_16、A2_8_16、angular_entropy_8_16）retention median ≥ 0.80（src_gkf，盒内 101），且 ≥4/5 折 retention ≥ 0.60；proc_gkf retention ≥ 0.70 否则 PARTIAL；≥0.90 记 strong tier。ilr_z2 / λ\* retention 报告不进主判定。判读语："五维工艺关系可压缩为单轨宽度–hatch overlap 几何"。
- **M0_RECON_FULL200（脚本第 0 步，纯 QA，§0.16）**：全 200、Phase 2.5 同协议（input A、Ridge、src_gkf 全 200 splits、Task 12 同 target 集）复跑，与 `cv_fold_results.csv` 对应行 Δ ≤ 0.005，输出 `model_compare/m0_reconciliation.csv`；失败 abort。
- CV：盒内 101 子集重新生成 src_gkf（groups=shared_height_source_id 子集化）与 proc_gkf（groups=cv_process_group 子集化），`check_gkf_contract` 照跑；GSS 敏感性（seed+100/+200）。
- sensitivity 臂：全 200（含 out_of_box，标注 extrapolated）；`exclude_artifact`（[37,149,82]）；`minus_top5`；Spline/ET（compression 参考，注记 §0.13：ET 下 Δ≈0 不构成对几何尺度解释的反证）。
- 输出：`model_compare/width_bridge_cv.csv`、`overlap_bridge_cv.csv`（fold 级 + 汇总，含全部 M0/M0b/M1/M2/M3/M_GEO 行与 retention 列）、`oof_predictions.csv`（显式 fold 列，每样本每 (variant,model,input) 恰一行——沿用 Phase 2.5 OOF 唯一性契约）、`m0_reconciliation.csv`。

## 7. Task SL-04 — `19_lambda_ratio_test.py`

- 口径分工（§0.18）：**H2 主证据 = λ_peak_4–32**（validity：bin n_modes ≥ 20 ∧ 峰 bin 持窗内能量 ≥ 0.20）；H1/H3 主口径 = λ\*_4–32（centroid）；两者各自进 ratio 表。
- `r_h / r_h_peak = λ*/λ_peak ÷ h`：**全 200**（h 恒有值）；单线（h=NA）不进此表（测试 T14）。
- `r_W = λ*/Ŵ`：盒内 101（primary）+ 全 200 extrapolated 臂；λ_peak 版作 sensitivity 列。
- **G-SL2（peak 主口径）**：在 λ_peak_valid 子集上，`A_obs = #{d_int_peak ≤ 0.25}/n_valid_peak`（d_int_peak = min_{m∈{1,2,3}} |λ_peak/h − m|）；shuffled-h null（§0.19 block 结构：单位 = unique(session_id, base_condition_group)，session 内重排单位→h，10,000 次，seed+800）→ `p = (1 + #{A_null ≥ A_obs})/(1+10000)`。SUPPORTED = A_obs ≥ 0.40 **且** p ≤ 0.05；仅其一 → PARTIAL；均否 → NOT_SUPPORTED。若 λ_peak_valid 比例 < 0.5 → 最高 PARTIAL 并注记"峰证据覆盖不足"。centroid 版 r_h 的同型统计并排输出（sensitivity，不替代主判定）。
- H1 侧（不设无约束 clustering 证据，上位规划 §13）：median/IQR(r_W)、`#{|r_W−1| ≤ 0.25}`、Spearman(λ\*, Ŵ)；**H1 终判的证据排序按 §0.17：direct > in-box predicted > out-of-box**。
- 输出：`scale_bridge/lambda_over_hatch.csv`、`lambda_over_width.csv`、`shuffled_h_null.csv`（A_null 分布 + p，peak 与 centroid 各一套，含置换单位数：120+15+10）。

## 8. Task SL-05 — `20_orientation_provenance_check.py`

- `provenance_valid = False`（§0.8 核查结论，config 登记）→ scan/hatch-relative Δθ **不计算**，G-SL4 = `NOT_APPLICABLE`（测试 T16 锚定：脚本不得输出 scan-relative 角）。
- descriptive 臂（明确非证据）：`theta_stripe_8_16`（directional_metrics band=8_16）对 0°/90° 的聚集：`#{|mod(θ,90°)| ≤ 10°}/200`，对照均匀角置换 null（seed+800）；输出 `orientation/stripe_scan_alignment.csv` + `orientation_provenance.json`（记录 v2 §12 弓字形、无逐样本轴、单线无 hatch、起终点符号未知四条事实）。
- 单线侧 `theta_line_deg` 分布（−0.43°~−0.77°）已在 manifest，不重复分析。

## 9. config（`experiments/phase2_6/phase2_6_config.yaml`，随细则一起冻结）

```yaml
meta:        {description, random_seed: 20260904, quick_output_root: outputs/phase2_6_quick}
paths:
  line_cag: 氧化锆/120组直线.cag
  line_design_csv: 氧化锆/氧化锆_line_design.csv   # gb18030
  line_view_manifest: annotations/single_line_view_manifest.csv
  phase2_manifest: outputs/phase2/manifest/phase2_manifest.csv
  p25_spectral_csv: outputs/phase2_5/spectral_composition/spectral_composition.csv
  p25_ilr_csv: outputs/phase2_5/spectral_composition/ilr_coordinates.csv
  p25_directional_csv: outputs/phase2_5/directional_spectrum/directional_metrics.csv
  p25_radial_long_csv: outputs/phase2_5/spectral_composition/radial_spectrum_long.csv
  p25_descriptor_csv: outputs/phase2_5/spectral_composition/spectrum_descriptor_summary.csv
  p25_cv_fold_csv: outputs/phase2_5/process_map/cv_fold_results.csv
single_line:
  pixel_um: 0.278657          # 断言与容器头一致
  power_w: 5.333
  plane_source: view_manifest_frozen   # 禁止重拟合
  geometry_method: axis_direct_sampling  # 禁止整幅旋正进数值（§0.20）
  axis: {theta_source: theta_line_deg, anchor_source: orientation_center_x_y, v_px: 64, v_step_um: 0.278657}
  detection_threshold_k: 4.0           # D > k*1.4826*sigma_ref_um（sigma_ref 冻结值）
  min_profile_points: 8
  stable_region: {fraction: 0.70, pad_low: 0.15, pad_high: 0.85}
  cross_section_step_um: 2.0
  min_sections: 20
  whole_map_rotation: qa_visualization_only   # 必须输出 rotated_valid_footprint
widths:
  thresholds_q: [0.2, 0.5, 0.8]
  primary: W50
  weq_from_positive_part: true
  affected_delta_um: 0.10             # 实际取 max(0.10, 3*plane_rmse_um)
  height_arm_primary: raw             # repaired = sensitivity
  identifiability:
    states: [estimable, right_censored, insufficient_sections]
    estimable: {min_sections: 20, max_censored_frac_W50: 0.5}
    censored_semantics: lower_bound
qa_montage:
  blind: true
  forbidden_elements: [band_8_um, band_16_um, band_shading, any_vs_8_16_reference]
  panels: [abs_height, corrected_profile, W20_50_80_marks, weq, W50_distribution_no_band, mask_artifact]
lambda_star:
  centroid_window_um: [4.0, 32.0)
  min_band_energy_fraction: 0.10
  peak:
    window_um: [4.0, 32.0)
    n_modes_min: 20
    min_peak_energy_share_in_window: 0.20
  sensitivity: [global_centroid]
bridge:
  box: {tau_fs: [223, 4000], f_khz: [2, 40], v_mm_s: [5, 25], pass: [1, 5]}
  coverage_states: [exact_match, in_box_pred, out_of_box]
  exact_match_repeat_handling: condition_level_aggregate   # 19 条件；重复=54/156(2000,10,4,5;h 10/8)
  direct_bridge: {stat_unit: condition, n_conditions: 19, evidence_priority: [exact_match_direct, in_box_predicted, out_of_box]}
model_compare:
  targets:
    - {name: p_8_16, metric: R2}
    - {name: ilr_z2, metric: R2}
    - {name: ilr_z1_z4, metric: Q2_Aitchison}
    - {name: A2_8_16, metric: R2}
    - {name: angular_entropy_8_16, metric: R2}
    - {name: lambda_star_4_32, metric: R2}
  aitchison_scope: full_composition_only    # 禁止用于 scalar target
  models: [M0_u, M0b_u_plus_logtau, M1_u_plus_What, M2_h, M3_u_What_What_over_h, M_GEO_What_h_eta]
  transform_control: M0b                    # Δ(M1−M0b)≈0 confirmatory（warning 级，阈 0.02）
  delta_M1_M0_label: LOW_CAPACITY_REPRESENTATION_GAIN   # descriptive，禁入 gate
  cv: {subset: in_box_101, src_group: shared_height_source_id, proc_group: cv_process_group, n_splits: 5}
  retention: {min_supported: 0.80, strong_tier: 0.90, fold_min: 0.60, min_folds: 4, m0_perf_floor: 0.10, proc_min_agree: 0.70}
  m0_recon: {dataset: full200, tolerance: 0.005, abort_on_fail: true}
  sensitivity: [spline, extratrees, exclude_artifact, minus_top5, full200_extrapolated]
  alpha_grid_logspace: [-3, 3, 13]
ratio_test:
  d_int_tolerance: 0.25
  a_obs_min: 0.40
  n_perm: 10000
  shuffle_unit: [session_id, base_condition_group]   # formal 120 单行 / pass 15 base / supplement 10 base
  primary_stat: lambda_peak                # H2 主证据；centroid 版并排 sensitivity
  peak_valid_fraction_floor: 0.5
gates:
  gsl1: {population: estimable, band_um: [8, 16), min_line_fraction: 0.50, min_estimable_for_supported: 60, weq_same_direction: true, repaired_divergence_note_above: 0.10}
  gsl2: {a_obs_min: 0.40, p_max: 0.05, stat: lambda_peak}
  gsl3: {type: geometry_compression, retention_min: 0.80, strong_tier: 0.90, fold_min: 0.60, min_folds: 4, proc_min_agree: 0.70}
  gsl4: {provenance_valid: false}      # §0.8；如人工确认填充轴须回写 §0.8 再改
orientation:
  image_frame_cluster_deg: 10.0        # descriptive only
```

## 10. 测试（`tests/test_phase2_6_lib.py`，unittest，沿用 `require` 硬断言风格；缺冻结输入时 SkipTest）

- T1 `test_cag_pixel_pitch_and_field`：dx=dy=0.278657±1e-6；视场 285.3448×17.8340 µm。
- T2 `test_design_table_grid`：120 行、四因素值域、加工顺序唯一、无重复条件。
- T3 `test_manifest_fields_complete`：16 必备字段非空或显式 NA；排除说明存在。
- T4 `test_width_ordering`：W20 ≥ W50 ≥ W80（随机抽 30 截面 + formal CSV 全量）。
- T5 `test_weq_positive_finite`：W_eq > 0 且有限。
- T6 `test_dn_bounds`：d_n ≤ 1+1e-9；负 min 有记录。
- T7 `test_min_sections`：estimable 线 `n_sections_used ≥ 20`（按定义）。
- T8 `test_identifiability_gate_population`：三态判定正确；`right_censored`/`insufficient_sections` 线不出现在 G-SL1 主统计输入（pooled median / 带内比例 / W_eq 一致性的输入行集合 = estimable）。
- T9 `test_grouped_cv_line_contract`：SL-02 CV 按单线分组，`check_gkf_contract` 通过；train/test 无同 `single_line_id`。
- T10 `test_what_hat_feature_whitelist`：Ŵ 预测特征列 ⊆ {τ,f,v,N}（+登记的 C-extra），无任何 morphology 列。
- T11 `test_inbox_definition`：盒内判定复现 101/200（冻结数字）。
- T12 `test_lambda_star_and_peak_validity`：λ\* ∈ [4,32) 或 NA（guard 0.10）；λ_peak 有效双条件（n_modes ≥ 20、峰能量份额 ≥ 0.20），构造宽谱合成 case → peak NA。
- T13 `test_lambda_star_recompute`：抽 3 样本从 radial_spectrum_long 现算与输出一致（1e-9）。
- T14 `test_ratio_table_exclusions`：h=NA 行不出现在 r_h 表；out_of_box 不出现在 primary r_W 表。
- T15 `test_shuffle_null_block_structure`：置换后 h 在 (session_id, base_condition_group) 单位内恒定、单位数（120/15/10）不变、置换按 session 内进行；固定 seed 复现 A_null 分布与 p。
- T16 `test_orientation_na_gate`：provenance_valid=false 时无 scan-relative 输出、G-SL4=NOT_APPLICABLE。
- T17 `test_p816_reuse_not_recomputed`：桥接表 p_8_16 与 Phase 2.5 CSV 逐行一致（直采）。
- T18 `test_no_pass_step_analysis`：`outputs/phase2_6` 无任何 N-step/pseudo-pass 产物（负面断言）。
- T19 `test_m0_recon_full200`：M0_RECON_FULL200 与 Phase 2.5 `cv_fold_results.csv` 对应行 Δ ≤ 0.005（纯 QA 轨道）。
- T20 `test_m0_primary_inbox_no_forced_match`：M0_PRIMARY_INBOX101 在 101 子集上生成 splits 并通过契约、产出结果；**不存在**与 Phase 2.5 数值的相等性断言。
- T21 `test_direct_bridge_conditions`：direct bridge 统计单位 = 19 条件；重复条件（54/156）已聚合且记录 h 差异；单线 `right_censored` 标 `W_lower_bound`、`insufficient_sections` 标 `W_unavailable`（无方向性、剔除分母）；r 值有限。
- T22 `test_blind_montage_spec`：montage 配置/代码不含任何 8 µm/16 µm 带参考（负向断言）；`W_line_distribution_vs_band` 仅在 `geometry_qa_labels.csv` 全量完成后的产物目录中出现。

## 11. 输出树

```text
outputs/phase2_6/
  single_line/
    single_line_manifest.csv        manifest_provenance.json
    single_line_geometry.csv        cross_section_widths.csv
    width_identifiability_summary.csv
    geometry_qa_labels.csv          qa_montages/*.png
  scale_bridge/
    morphology_scale_match.csv      direct_bridge_exact_match.csv
    lambda_over_width.csv           lambda_over_hatch.csv
    overlap_metrics.csv             shuffled_h_null.csv
  model_compare/
    line_width_process_model.csv    W_line_response_curves.csv
    W_line_distribution_vs_band.csv (+.png，仅 QA 完成后)
    width_bridge_cv.csv             overlap_bridge_cv.csv
    m0_reconciliation.csv           oof_predictions.csv
  orientation/
    stripe_scan_alignment.csv       orientation_provenance.json
  summary/
    phase2_6_gate_eval.md
```

## 12. 运行顺序与预算

1. 预冻结提交：本细则（v2）+ `phase2_6_config.yaml` + 15–20 脚本骨架（**先于一切 formal**；只含 phase2_6 文件，不混入标注改动）→ 状态升级 FROZEN_EXECUTED。
2. `15` manifest → commit。3. `16` geometry（raw + repaired 两臂，**blind** QA montage）→ 人工三值标注（可异步，17 先用 estimable 预集 quick）→ 冻结宽度 → commit。此时回答上位规划 §21.4"单线宽度到底是多少"。4. `17` 宽度工艺模型（G-SL1，estimable 总体）→ commit；QA 全量完成后补 `W_line_distribution_vs_band` 科学图。5. `18` SL-03a direct bridge + Ŵ 桥 + M0_RECON（QA）→ M0–M_GEO 比较（G-SL3）→ commit。6. `19` λ_peak/centroid 比值检验（G-SL2，block-structured null）→ commit。7. `20` orientation（G-SL4/NA）→ commit。8. `summary/phase2_6_gate_eval.md` 终判（证据排序：direct > in-box predicted > out-of-box）→ commit。
预算：CAG 现读 120 组（分钟级）；几何/宽度/模型全部 CPU（单机分钟–十分钟级）；无训练大数据。

## 13. Gate → 文件映射 → 必答 8 问题

| Gate | 判定文件 | 回答的问题（上位规划 §22） |
|---|---|---|
| G-SL1（总体=estimable） | `line_width_process_model.csv` + `W_line_distribution_vs_band.csv` + `width_identifiability_summary.csv` | Q1 单线宽度真实范围（censored 以 lower-bound 并入解读）；Q2 是否覆盖 8–16 主分布；Q3 主要受控变量（response curves，f 标 Ep-coupled） |
| SL-03a direct bridge | `direct_bridge_exact_match.csv` | Q4 的最直接证据（测量→测量，19 条件）；H1 终判的最高优先级证据 |
| G-SL2（λ_peak 主口径） | `lambda_over_hatch.csv` + `shuffled_h_null.csv` | Q4 λ 更接近 W、h 还是 integer-multiple scale（与 G-SL3 合并读；hatch-related periodic，非 harmonic） |
| G-SL3（Geometry-compression） | `width_bridge_cv.csv` + `overlap_bridge_cv.csv`（retention 列） | Q5 W/h 是否优于单独 W 或 h；Q6 Route P/T 是否都随 overlap 几何改变；判读语="五维工艺关系压缩为单轨宽度–hatch overlap 几何" |
| G-SL4 | `orientation_provenance.json`（预期 NA） | Q7 方向对应（无 provenance → 显式不可判定，不得暗示） |
| 终判 | `phase2_6_gate_eval.md` | Q8 intrinsic track scale / hatch-related periodic scale / overlap composite scale（按上位规划 §17 的 A–D 判读矩阵映射 + §0.17 证据优先级，禁止为保留 H1 强选） |

## 14. 语言边界（沿用上位规划 §3，执行期强制）

不得把整数比称为"谐波机制"；**全文弃用 harmonic 一词**——2h/3h 统一称 "hatch-related periodic / integer-multiple scale（multi-line / envelope scale）"，严格 Fourier 意义下 h 的 harmonic 是 h/n 而非 2h/3h；不得把 W_line 与 Fourier wavelength 混同；不得以 nominal spot 直推加工宽度；不得用单线数据替代 Phase 2.5 面形貌证据；不得把结果命名为热积累/相变/脆性断裂；G-SL4 不得在无 provenance 时给出任何 scan 对齐措辞；ΔR²(M1−M0) 只能以 LOW-CAPACITY REPRESENTATION GAIN 名义出现，不得写成机制证据。

## 15. 已决定 / 开放

- **已决定（v2）**：§0 全部 20 条；脚本编号 15–20；种子 20260904；盒内 101 primary + exact-match 19 条件 direct bridge；λ\*（centroid，H1/H3）与 λ_peak（H2）双口径及各自 guard；W50 最长 run 实现；原坐标轴向直接采样（无整幅旋正）；central-70% 稳定区；width_identifiability 三态及 G-SL1 estimable 总体（n≥60 保护）；M0_RECON_FULL200 / M0_PRIMARY_INBOX101 拆分；G-SL3 = Geometry-compression（retention ≥0.80，strong 0.90）；Aitchison Q² 仅限 ilr_z1_z4；shuffled-h block 结构（120/15/10 单位）；blind montage 禁带 8–16 信息；证据优先级 direct > predicted > out-of-box；G-SL1/2/3 门槛数值；G-SL4 路径。
- **2026-09-04 审计回写（已落入正文）**：§0.14 增列冻结后 config 改动的偏差登记；§0.15 补注 (a) 按实跑 config 回写（depth_frac 0.50 / P90 / 宽度条件撤销 / gap_merge 10 µm / 碎片守卫）；§0.17 新增 `W_unavailable` 态（`insufficient_sections` 不是下界，direct bridge 实测缺失率 5/19 = 26%）。
- **2026-09-04 实施状态**：Task 15/16 已 formal 完成并含人工三值 QA（120/120：usable 18 / uncertain 78 / reject 24）；Task 17/18 代码完整但**尚未 formal 运行**；**Task 19（SL-04，G-SL2）与 Task 20（SL-05，G-SL4）已于本日由冻结骨架落地为完整实现**（19 走 λ_peak 主口径 + block-structured shuffled-h null，20 走 provenance_valid=false 的 NOT_APPLICABLE 路径 + image-frame descriptive 臂），并补齐单测 T14/T16/T18/T22。**上述实现未经本机执行验证**（审计环境无 numpy/pandas/scipy，包索引不可用），首次 formal 运行须以 `--quick` 先行冒烟。
- **G-SL1 预读（2026-09-04，由 Task 16 产物按冻结口径直接算得，非 formal 判定；formal 归 Task 17）**：总体 = estimable 且人工标签 ≠ reject，n = 81（≥60，视场代表性保护不适用）。pooled W50 median = **5.776 µm**（带内 2.7%）✗；per-line median W50 带内 **1/81 = 1.2%** ✗；pooled W_eq median = **5.857 µm**（带内 1.5%）✗ ⇒ **G-SL1 三条判据 0/3，指向 NOT_SUPPORTED**。预注册旁支口径 **W20 pooled median = 8.161 µm（P10 6.36 / P90 10.08），带内 53.8%** —— 8–16 µm 更接近槽的**外缘/近阈值宽度**而非半深宽度。W20 属 §5 预注册保存的口径，可作 pre-registered sensitivity 并列报告；**严禁**据此回改 `W_line = W50` 主定义或将 W20 提升为 G-SL1 主判据（上位规划 §21、本细则 §0.14）。据此，H1 终判的落点主要在 **G-SL2**（Task 19）。
- **开放（不阻塞冻结）**：① ~~人工 QA 标注完成时间~~（**已于 2026-09-04 完成 120/120**，vs-band 科学图的生成前置条件已满足）；② 若外审要求 conditional 方向臂，须先人工登记填充轴并回写 §0.8；③ `power_measurement_version=PENDING_REGISTRATION` 的登记（v2 §11 要求），完成前 gate 文档保留弱点注记；④ ~~Task 17/18/19/20 的首次 formal 运行与 commit~~（**已于 2026-09-04 完成**，formal 统一使用 `.venv`——运行环境约定与复现锚点见 `outputs/phase2_6/summary/RUNTIME_ENVIRONMENT.md`）。
- **2026-09-04 封账登记（Phase 2.6 关账，报告层修正、不重跑主结果）**：① **adaptive refinement 正式入账**——§0.15 稳定区规则在 Task 16 formal 期间经历多轮迭代（central-70% → 深度台地 0.8·median → 0.5·P90 + 碎片守卫 + qualifying-only 截面），全部发生在任何 gate 统计计算之前，迭代轨迹与config 终态已回写 §0.15 补注 (a)(b)；② Task 17 variant 混报修复（primary 数据的 GSS 折曾误标 variant → 现显式 primary_gkf / primary_gss / sensitivity_gkf / usable_only_gkf）并重跑 formal——G-SL1 全部数值逐位不变（pooled W50 5.776 / 带内 1.2% / W_eq 5.857）；③ 新增 **usable-only sensitivity**（n=18：W50 带内 0.0%、W_eq 带内 0.0%——保守下界与主口径同向，写入 gsl1_evaluation.json）；④ gate 终判文档升 rev2（表述收紧 + 上述登记），判定维持 B。Phase 2.6 关账，后续进入 Phase 2.7。
