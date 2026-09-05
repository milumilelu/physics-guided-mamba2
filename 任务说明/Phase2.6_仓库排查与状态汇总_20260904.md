# Phase 2.6 仓库排查与任务状态汇总

> 核查日期：2026-09-04
> 核查对象：`任务说明/Phase2.6_单线扫描尺度溯源_研究任务说明.md`（上位规划，what/why）
> 对照基准：`experiments/phase2_6/Phase2.6_落地执行细则.md`（FROZEN_EXECUTED，how；冲突时以细则为准）
> 核查性质：**只读审计**。未运行任何 Task 17–20，未生成任何科学结果图/表。
> 结论摘要：**SL-01（15/16）已完成并含人工 QA；SL-02/SL-03（17/18）代码就绪但未运行；
> SL-04/SL-05（19/20）仍为冻结骨架。G-SL1 预读为 NOT_SUPPORTED（0/3 判据）。**

---

## 0. 一句话状态

| Task | 脚本 | 代码状态 | 是否已 formal 运行 | 输出 |
|---|---|---|---|---|
| SL-01a manifest | `15_build_single_line_manifest.py` (221 行) | 完整实现 | ✅ 已跑 | 120×41 CSV + provenance.json |
| SL-01b geometry | `16_extract_single_line_geometry.py` (426 行) | 完整实现 | ✅ 已跑（终稿 a469206） | geometry 120×58、截面 14102 行、montage 120 张 |
| — 人工三值 QA | — | — | ✅ **120/120 完成** | `geometry_qa_labels.csv` + provenance.json |
| SL-02 宽度工艺模型 | `17_line_width_process_model.py` (310 行) | 完整实现 | ❌ **未运行** | — |
| SL-03 scale bridge | `18_scale_bridge_model_compare.py` (519 行) | 完整实现 | ❌ **未运行** | — |
| SL-04 比值检验 | `19_lambda_ratio_test.py` (40 → **约 400 行**) | **本次审计后落地为完整实现** | ❌ 待 18 | — |
| SL-05 方向 provenance | `20_orientation_provenance_check.py` (37 → **约 190 行**) | **本次审计后落地为完整实现** | ❌ 可独立运行 | — |

语法校验：15/16/17/18/19/20/`_lib.py` 全部通过 `py_compile`。
`NotImplementedError` 计数：17 = 0、18 = 0、**19 = 0（完整实现）**、20 = 1。
> Task 20 的这 1 处 `NotImplementedError` 是**故意保留的守卫**：仅当有人把
> `gates.gsl4.provenance_valid` 翻成 true 而 §0.8 未回写时才会触发，用来强制
> "先人工登记填充轴再开 conditional 臂"的冻结约定；正常路径（provenance_valid=false）
> 已完整实现并会输出 G-SL4 = NOT_APPLICABLE。

**本次审计已完成的整改（详见 §9）**：实现 Task 19/20、补齐单测 T14/T16/T18/T22（测试数 23 → **27**）、
并按 §4.1–§4.3 回写细则（`W_unavailable` 语义、§0.15(a) 实跑口径、§0.14 偏差登记）。

Git 链：`2daa611`（预冻结）→ `65ded0e`（15 formal）→ `7e99bf9`（16 formal）→
`a469206`（16 终稿：qualifying-only 截面 + 碎片守卫 + pilot 对账降级）。

---

## 1. 输出树对账（细则 §11 / 上位规划 §18）

```text
outputs/phase2_6/
  single_line/
    ✅ single_line_manifest.csv          120 行 × 41 列
    ✅ manifest_provenance.json          （九项核查全落列）
    ✅ single_line_geometry.csv          120 行 × 58 列（raw + _rep 双臂）
    ✅ cross_section_widths.csv          14102 行 × 28 列
    ❌ width_identifiability_summary.csv 【缺失，Task 17 的 EXPECTED 之一】
    ✅ geometry_qa_labels.csv            120/120（usable 18 / uncertain 78 / reject 24）
    ✅ qa_montages/group_*.png           120 张
    ✅ stable_region_reconciliation.csv  （对账清单，已降级为 informational）
  scale_bridge/     ❌ 整目录缺失（6 个文件全缺）
  model_compare/    ❌ 整目录缺失（7 个文件全缺）
  orientation/      ❌ 整目录缺失（2 个文件全缺）
  summary/          ❌ 整目录缺失（gate_eval.md + gsl1/gsl3 json）
```

缺失明细（17 项）：
`width_identifiability_summary.csv`、`line_width_process_model.csv`、
`morphology_scale_match.csv`、`direct_bridge_exact_match.csv`、`lambda_over_width.csv`、
`lambda_over_hatch.csv`、`overlap_metrics.csv`、`shuffled_h_null.csv`、
`W_line_response_curves.csv`、`W_line_distribution_vs_band.csv(.png)`、
`width_bridge_cv.csv`、`overlap_bridge_cv.csv`、`m0_reconciliation.csv`、
`oof_predictions.csv`、`stripe_scan_alignment.csv`、`orientation_provenance.json`、
`phase2_6_gate_eval.md`、`gsl1_evaluation.json`、`gsl3_evaluation.json`。

上位规划 §19 的 7 张「最低核心图」**一张都还没生成**（归属于 17/18/19）。

---

## 2. 逐条对账上位规划章节

### §4 数据 provenance manifest —— ✅ 通过

`single_line_manifest.csv` 120 行，上位规划要求的 16 必备字段 **17/17 存在且 120/120 非空**
（唯一例外：`notes` 列 0/120 非空，但该列本为自由文本，非阻断）。

manifest_provenance.json 的九项核查落列情况：

| 核查项 | 结论 |
|---|---|
| 同一激光系统 | `same_laser_system: true`（同实测功率 + 同 VK4/CAG 测量链） |
| 功率可换算 | `power_condition_convertible: true`（两侧同 5.333 W），但**无独立测量记录**（v2 §11，`PENDING_REGISTRATION`） |
| 单位一致 | `units_native_consistent: true` |
| 像素尺寸可信 | 0.278657 µm（CAG 容器头，逐组断言一致） |
| 完整槽截面 | `checked_in_task_16` |
| 裁剪截断 | `checked_in_task_16` |
| 重复位置 | `repeated_positions: 0` |
| 背景平面 | `frozen_plane_from_view_manifest`（冻结，未重拟合） |
| scan direction 可恢复 | `partial: axis yes, sign no` |

`氧化锆/72组单脉冲直线.cag` 已按红线排除并登记（无设计表 → 无 provenance）。

### §5 单线宽度定义 —— ✅ 通过（W1/W2/W3 三类齐全）

| 冻结口径 | 列 | 状态 |
|---|---|---|
| W1 threshold width | `W20_um / W50_um / W80_um` + `n_runs_* / total_width_* / censored_*` | ✅ |
| W2 equivalent-area | `W_eq_um`（+ `A_remove_um2`） | ✅ |
| W3 affected width | `W_affected_um`（`δ_aff = max(0.10 µm, 3×plane_rmse)`，仅 descriptor） | ✅ |

硬 QA `W20 ≥ W50 ≥ W80`、`W_eq > 0` 有对应单测（WidthTests）。

### §6 附加几何描述符 —— ✅ 9/9 齐全

`D_max_um`(=max_depth)、`A_remove_um2`(=cross_section_area)、`left_slope`、`right_slope`、
`edge_asymmetry`、`ridge_left_um`、`ridge_right_um`、`ridge_separation_um`、`profile_skewness`。

### §7 多截面测量 —— ✅（但 30% 线不满足）

estimable 线 `n_sections_used` 实测范围 39–102（要求 ≥20）。

### §8 QA montage 与盲标 —— ✅ 通过

- 120/120 montage 已生成，`plt.subplots(2,3)` 六面板，与细则 §4.3 面板规格一致。
- **T22 负向断言通过**：`16_extract_single_line_geometry.py` 全文无 `8.0` / `16.0` 字面量、
  无 band shading、无 8–16 参考线。（出现的 "band-edge" 指稳定区 qualifying band，与形貌带无关。）
- 人工标签三值制已落地：usable 18 / uncertain 78 / reject_geometry 24。
- provenance json 如实记录 annotator 自述为 GPT（AI 辅助标注），盲性保持。

### §15 CV 防泄漏 —— ⚠️ 部分

- 单线侧 `check_gkf_contract` + `single_line_id` 分组的实现在 `_lib.py` 中，但**对应单测 T9
  （grouped_cv_line_contract）尚未激活**（依赖 Task 17 产物）。
- Phase 2.5 侧 `src_gkf/proc_gkf` 契约沿用，未改动。

### §20 最低测试（12 条）—— 见 §6 表

---

## 3. 关键科学读数：G-SL1 预读（**尚未 formal 判定**）

以下全部由 Task 16 **现有产物**按冻结口径直接算得，属预读；formal 判定仍归 Task 17
（`summary/gsl1_evaluation.json`）。

### 3.1 可辨识性分布（与预注册风险模型不符）

```text
width_identifiability (raw 臂):   estimable 84  |  insufficient_sections 36  |  right_censored 0
width_identifiability (rep 臂):   estimable 84  |  insufficient_sections 36  |  right_censored 0
实测 censoring 比例（全 120 线 raw）:  W20 0.00%  W50 0.01%  W80 0.03%
```

细则 §0.3 把「17.83 µm 横向视场导致线宽截断」预注册为**本批数据最大的统计风险**，并为此
设计了 `right_censored` 三态与 lower-bound 语义。实测该风险**没有发生**（W20 最大仅 12.33 µm，
远小于 17.83 µm 视场）。真实瓶颈是**碎片守卫**：36/120（30%）被判 `insufficient_sections`。

### 3.2 G-SL1 三条判据（总体 = estimable 且 qa_label ≠ reject_geometry，n = 81）

| 判据 | 冻结阈值 | 实测 | 结果 |
|---|---|---|---|
| 1. pooled W50 median ∈ [8,16) | in band | **5.776 µm**（P10 4.19 / P90 7.34，min 1.48 / max 9.43，带内 2.7%） | ✗ |
| 2. ≥50% 线 per-line median W50 ∈ [8,16) | ≥0.50 | **1/81 = 1.2%** | ✗ |
| 3. pooled W_eq median 同在 [8,16) | in band | **5.857 µm**（带内 1.5%） | ✗ |

`n_estimable = 81 ≥ 60` ⇒ 细则 §5 的「视场代表性保护」（三条全中但 n<60 才降 PARTIAL）**不适用**。

> **G-SL1 = NOT_SUPPORTED（0/3）**，即 H1「8–16 µm ≈ 单轨有效横向加工尺度」在冻结的
> **W50 主口径下不成立**。

### 3.3 预注册旁支口径：W20 恰好落在带内（重要，但不得回改定义）

```text
W20   pooled median = 8.161 µm   P10 6.358  P90 10.075   带内比例 53.8%
W50   pooled median = 5.776 µm   P10 4.193  P90  7.338   带内比例  2.7%
W80   pooled median = 3.269 µm   P10 1.898  P90  4.612   带内比例  0.0%
W_eq  pooled median = 5.857 µm   P10 4.482  P90  7.251   带内比例  1.5%
```

**读法**：8–16 µm 更接近槽的**外缘/近阈值横向尺度**（W20，即 d_n ≥ 0.2 的宽度），
而不是**半深宽度**（W50）。这是一个真实的、可解释的几何差异，且 W20 是上位规划 §5 与细则 §5
**预注册保存**的三个 threshold width 之一 —— 因此它可作为 pre-registered sensitivity 报告，
**不是**事后改口径。

**红线提醒**（上位规划 §21、细则 §0.14）：`W_line = W50` 的定义在看到任何结果之前已冻结，
**禁止**因 W20 落在带内而回改主定义或把 W20 提升为 G-SL1 主判据。正确做法是：
G-SL1 按 W50 报 NOT_SUPPORTED，同时把 W20 结果作为预注册 sensitivity 并列写入 gate eval。

---

## 4. 需要人盯的 4 个问题

### 4.1【中】冻结后仍改过 config 与细则（2daa611 → a469206）

细则 §0.14 声明「此后禁止改动宽度定义、λ 窗口、guard、门槛」。实际 diff：

| 改动 | 性质 |
|---|---|
| `stable_region`：central 70% → 深度台地最长连续段 | **预处理/选区规则**（影响哪些截面进宽度统计） |
| `pilot_reconcile_min_agreement: 0.90` 的 **abort 门降级**为 informational 清单 | **预注册 abort 条件被放宽** |
| 新增 `gap_merge_um / min_stable_len_um / min_stable_frac` | 碎片守卫 |

严格说，**未触碰**字面红线（宽度定义 `thresholds_q/primary`、λ 窗口与 guard、G-SL1~G-SL4 门槛数值
均未改）。但「首次 formal 对账一致率仅 0.70–0.97 → 把 0.90 abort 门降级」是**看到结果后放宽预注册
abort 条件**，属实质性偏差，应在 `phase2_6_gate_eval.md` 中显式声明并说明理由。

细则 §0.15 rev2 补注 (b) 已自行登记了这一改动及其理由（pilot 精确算法随删除脚本丢失、边界不可
局部复现），并改用三条结构性机制（qualifying-only 布点 / 碎片守卫 / 人工三值 QA）作为污染保证，
论证是充分的 —— 问题在于**记录时机**（formal 首跑之后），需在终稿中如实复述。

### 4.2【中】细则 §0.15(a) 与 config 漂移（文档未回写）

细则 §0.15 rev2 补注 (a) 仍描述：

```text
depth_p95 ≥ 0.80·median(depth_p95|on-line)  ∧  绝对阈值宽度 ≥ 0.95·median(...)  ∧  间隙 ≤2 µm 桥接
```

而 `phase2_6_config.yaml` 实际实现为：

```yaml
depth_frac: 0.50            # 不是 0.80
ref_quantile: 0.90          # 用 P90，不是 median
width_band_frac: null       # REVOKED：宽度条件在规模化时把大量线切碎，已撤销
gap_merge_um: 10.0          # 不是 2 µm（细则 §0.15(b) 正文提到 10 µm，(a) 未同步）
min_stable_len_um: 60.0     # 碎片守卫，(b) 有描述，(a) 无
min_stable_frac: 0.50
```

⇒ **实现与文档不一致**。建议把 (a) 回写为 config 实际值，并注明宽度条件被撤销的理由。
（`width_band_frac` 的撤销理由已写在 config 注释里，但未回写细则。）

### 4.3【高】SL-03a direct bridge 严重退化 + 细则语义缺口

细则 §0.17 把 exact-match direct bridge 定为「本阶段最强的直接物理证据」，证据优先级最高。
实测可行性：

```text
exact_match 矩形样本 = 20，独立条件 = 19（唯一重复：2000fs/10kHz/5mm·s⁻¹/N4，h = 10 vs 8 µm）
```

19 个条件对应单线的状态：

| 状态 | 条件数 | 线号 |
|---|---|---|
| estimable 且 qa ≠ reject → **可用于 direct bridge** | **13** | 2,5,27,55,56,65,70,89,94,105,109,112,116 |
| estimable 但 qa = reject_geometry → §5 排除 | 1 | 63 |
| `insufficient_sections` → **无宽度可用** | **5** | 10, 75, 83, 90, 92 |

13 条可用中，仅 2 条人工标签为 `usable`（线 27、70），其余 11 条为 `uncertain`。

**细则语义缺口**：§0.17 只写了

> 若对应单线 `width_identifiability ≠ estimable`，该条件标 `W_lower_bound`

但 `W_lower_bound` 语义只对 `right_censored`（真值 > 观测值）成立；本次出现的非 estimable 状态
**全部是 `insufficient_sections`**（实测 right_censored = 0），这类条件**根本没有宽度估计，不是下界**。
按现行文字把 5 个条件标成 `W_lower_bound` 会给出错误的方向性陈述。
⇒ 需补登记：`insufficient_sections` 条件标 `W_unavailable`（无条件级统计，单独列出并报告缺失率 5/19 = 26%）。

**另一个读数**：13 条可用条件的 median W50 实测仅 **4.75–6.65 µm**，全部 < 8 µm，
⇒ `r_W_direct = λ*/W_measured` 预计显著 > 1，与 §3.2 的 G-SL1 NOT_SUPPORTED 方向一致。

### 4.4【中】盒内 101 已复现，但 in-box 内仍有 26% 条件无 direct 宽度

`bridge.box` 冻结的 101/200 已复现（实测 = 101，与冻结数字一致，✅）。
但注意盒内 101 与 direct bridge 的 19 条件是不同子集：direct bridge 的高缺失率（26%）会削弱
「测量→测量」这条最高优先级证据臂，终判时须按缺失率打折陈述。

---

## 5. 单测覆盖对账（细则 §10 的 T1–T22）

实际 `tests/test_phase2_6_lib.py`：428 行，**23 个 test 方法**（预冻结提交时为 17 项：15 过 + 2 SkipTest 锚点）。

> **更新（本次审计后续整改）**：T14 / T16 / T18 / T22 已补齐（新增 `Task19Task20Tests` 类），
> 测试数 23 → **27**。下方"实现情况"列已同步更新。所有新测试均带阳性对照验证，
> 且在其依赖的 Task 产物缺失时走 `SkipTest`（与文件既有风格一致）。

| 细则 T# | 名称 | 实现情况 |
|---|---|---|
| T1 | cag_pixel_pitch_and_field | ✅ `test_cag_pitch_and_field_frozen` |
| T2 | design_table_grid | ✅ `test_design_table_grid_frozen` |
| T3 | manifest_fields_complete | ✅ `test_manifest_exists` |
| T4 | width_ordering (W20≥W50≥W80) | ✅ 含于 `test_section_features_analytic_groove` |
| T5 | weq_positive_finite | ✅ 含于同上 |
| T6 | dn_bounds | ✅ 含于同上 |
| T7 | min_sections | ✅ `test_aggregate_identifiability_states` |
| T8 | identifiability_gate_population | ✅ `test_aggregate_identifiability_states` + `test_geometry_outputs_states` |
| T9 | grouped_cv_line_contract | ❌ 待 Task 17 |
| T10 | what_hat_feature_whitelist | ❌ 待 Task 18 |
| T11 | inbox_definition (101/200) | ✅ `test_in_box_coverage_frozen_101` |
| T12 | lambda_star_and_peak_validity | ✅ `test_lambda_star_window_and_guard` + `test_peak_gates` |
| T13 | lambda_star_recompute | ❌ 待 Task 18 产物 |
| T14 | ratio_table_exclusions | ✅ **本次补齐**（h=NA 不入 r_h 表；out_of_box 不入 primary r_W 臂）|
| T15 | shuffle_null_block_structure | ✅ `test_shuffle_block_structure` |
| T16 | orientation_na_gate | ✅ **本次补齐**（provenance_valid 必须为 false + 输出列禁含 scan/hatch-relative）|
| T17 | p816_reuse_not_recomputed | ❌ 待 Task 18 |
| T18 | no_pass_step_analysis | ✅ **本次补齐**（负面断言：rglob 扫 `outputs/phase2_6` 无 N4→5 产物）|
| T19 | m0_recon_full200 | ❌ 待 Task 18 |
| T20 | m0_primary_inbox_no_forced_match | ❌ 待 Task 18 |
| T21 | direct_bridge_conditions | ❌ 待 Task 18 |
| T22 | blind_montage_spec | ✅ **本次补齐**（config 去注释后查 8/16；脚本查单词边界的 8.0/16.0 与 BAND；vs-band 图以 120/120 QA 为前提）|

额外覆盖（细则未列，实现已加）：plateau 稳定区 5 项、extent 检测 2 项、pilot 对账互相关对齐 1 项、
轴向帧正交性 2 项。

> ⚠️ **本次审计未执行测试**。原因：审计环境的托管 Python 无 numpy/pandas/scipy/sklearn/matplotlib，
> 且沙箱内的包索引不可用（`No matching distribution found for scikit-learn`），无法补齐依赖；
> 仓库脚本实际运行于 cpython-312 环境（pycache 证据），本次未定位到该解释器。
> 上表为**静态代码对账**（源码 + 产物数据双向核验），测试实际通过与否需在正式环境中
> 执行 `python -m unittest tests/test_phase2_6_lib.py -v` 确认。
> 注：审计过程为**只读**，未修改 `outputs/phase2_6/` 下任何既有产物。

---

## 6. 上位规划 §20「最低测试」12 条对账

| # | 要求 | 状态 |
|---|---|---|
| 1 | 横向坐标单位为 µm | ✅ 0.278657 µm/px，容器头断言 |
| 2 | W20 ≥ W50 ≥ W80 | ✅ 单测 + 实现 |
| 3 | W_eq > 0 | ✅ 单测 + 实现 |
| 4 | 多横截面不产生 train/test 泄漏 | ⚠️ 实现有 `check_gkf_contract`，单测 T9 待 Task 17 |
| 5 | `single_line_id` grouped CV 正确 | ⚠️ 同 #4 |
| 6 | 8–16 band 复用 Phase 2.5 | ⚠️ 单测 T17 未激活（待 Task 18）；实现侧为只读直采 |
| 7 | `lambda_star` 不从低 mode 粗 bin 强行取峰 | ✅ `peak_n_modes_min: 20` + 峰能量份额 ≥ 0.20 双条件 |
| 8 | single-line 中 hatch=NA 不参与 λ/h | ✅ 实现 + 单测 T14 已补齐 |
| 9 | N4→5 不进 Phase 2.5 bridge | ✅ 单测 T18（负面断言）已补齐，实测无违禁产物 |
| 10 | scan-relative angle 仅 provenance_valid=true 时算 | ⚠️ `gsl4.provenance_valid: false` 已冻结，单测 T16 待 Task 20 |
| 11 | W_hat 不使用 Phase 2.5 morphology label | ⚠️ 单测 T10 待 Task 18（特征白名单 = τ,f,v,N） |
| 12 | H1/H2/H3 gate threshold 在 formal 前冻结 | ✅ `gates:` 段已冻结（见 §4.1 的改动声明） |

---

## 7. 上位规划 §22「必须回答的 8 个问题」当前可答性

| # | 问题 | 可答性 |
|---|---|---|
| 1 | 单线有效加工宽度真实范围？ | ✅ **已可答**：W50 median 5.78 µm（P10–P90: 4.19–7.34）；W20 median 8.16 µm（6.36–10.08） |
| 2 | 8–16 µm 是否覆盖单线宽度主分布？ | ✅ **已可答（否定）**：W50 带内 2.7%、W_eq 1.5%；仅 W20 带内 53.8% |
| 3 | 单线宽度主要受控变量？ | ❌ 待 Task 17（`W_line_response_curves.csv`；f 须标 Ep-coupled） |
| 4 | λ\* 更接近 W / h / 2h？ | ❌ 待 Task 19（G-SL2，λ_peak 主口径 + block-structured shuffled-h null） |
| 5 | W/h 是否优于单独 W 或 h？ | ❌ 待 Task 18（G-SL3 = Geometry-compression，retention ≥ 0.80） |
| 6 | Route P 与 T 是否都随 overlap 几何改变？ | ❌ 待 Task 18（p_8_16 / A2_8_16 / angular_entropy_8_16 retention） |
| 7 | 条纹方向与 scan/hatch 有稳定几何对应？ | ⏸ 预期 **NOT_APPLICABLE**（v2 §12 无逐样本填充轴、单线无 hatch、起终点符号未知） |
| 8 | 最终解释为 intrinsic / hatch periodic / overlap composite？ | ❌ 待 G-SL2、G-SL3 合并判读 |

按上位规划 §17 判读矩阵的当前走向：**G-SL1 = NO**。若 G-SL2（hatch 相关周期尺度）成立，
则倾向判读 **B：8–16 µm 主要反映 hatch line array 的周期/整数倍尺度结构**。
但细则 §14 的语言红线要求：不得称其为 "harmonic"，须写 "hatch-related periodic / integer-multiple
scale（multi-line / envelope scale）"。

---

## 8. 建议的下一步（按细则 §21 执行顺序）

1. **先跑 Task 17**（宽度工艺模型 + formal G-SL1）。它会产出缺失的
   `width_identifiability_summary.csv`、`line_width_process_model.csv`、
   `W_line_response_curves.csv`，以及人工 QA 已完成前提下才允许生成的
   `W_line_distribution_vs_band.*`（§0.7 的前置条件现已满足）。
   → 这一步正式回答上位规划 §21.4「单线宽度到底是多少」。
2. **跑 Task 17 前先补登记** §4.3 的 `W_unavailable` 语义（insufficient_sections 不是下界）。
3. **跑 Task 18**：M0_RECON_FULL200（纯 QA，Δ ≤ 0.005，失败 abort）→ SL-03a direct bridge →
   Ŵ 桥接 → M0–M_GEO 比较（G-SL3）。注意 direct bridge 只有 13/19 条件可用。
4. **实现 Task 19**（仍为骨架）：G-SL2 用 **λ_peak_4–32** 主口径 + block-structured shuffled-h null
   （单位 = unique(session_id, base_condition_group)，10,000 次，seed+800）。这是当前最关键的
   未实现环节 —— G-SL1 已倾向 NO，终判的落点主要在 G-SL2。
5. **实现 Task 20**（仍为骨架）：provenance_valid = false → G-SL4 = NOT_APPLICABLE，
   仅输出 image-frame descriptive 检查。
6. **回写** §4.1/§4.2 的文档漂移（细则 §0.15(a) 对齐 config），并在
   `summary/phase2_6_gate_eval.md` 中声明 §4.1 的 abort 门放宽与 §4.3 的 direct bridge 缺失率。

---

## 9. 本次审计的整改动作（2026-09-04 同日完成）

排查发现问题后，就地完成了三类整改。**全部改动限于代码与文档，未生成任何科学结果
产物**，`outputs/phase2_6/` 下除既有的 QA 标注文件外无任何新增（`git status` 已确认）。

### 9.1 Task 19（SL-04 / G-SL2）由冻结骨架落地为完整实现

`19_lambda_ratio_test.py`：40 行 → 约 400 行，`NotImplementedError` 归零。要点：

- **H2 主口径 = `lambda_peak_4_32`**，复用 `_lib.lambda_peak_4_32`（双条件：bin
  `n_modes ≥ 20` 且峰 bin 持窗内能量 ≥ 0.20）；centroid 版 `r_h` 作**并排 sensitivity**，
  不替代主判定（§0.18）。
- **T13 对账**：读入 `radial_spectrum_long.csv` 现算 λ\* 与 λ_peak，与 Task 18 写入
  `morphology_scale_match.csv` 的列断言 `|Δ| ≤ 1e-9`，不符直接 `require` 失败。
- **block-structured shuffled-h null**（§0.19）：置换单位 = `unique(session_id,
  base_condition_group)`，10,000 次、seed = 20260904+800 = **20261204**，调用冻结的
  `p26.shuffle_h_by_block`；p = (1+#{A_null ≥ A_obs})/(1+n_perm)。peak 与 centroid 各一套。
- **T14 硬断言**：矩形侧 h 永不 NA；`out_of_box` 样本泄漏进 primary r_W 臂即失败。
- **G-SL2 判定阶梯**：A_obs ≥ 0.40 ∧ p ≤ 0.05 → SUPPORTED；仅其一 → PARTIAL；均否 →
  NOT_SUPPORTED；λ_peak valid 比例 < 0.5 时**封顶 PARTIAL**（"宽谱无峰"不得冒充周期证据）。
- 附带产出 `overlap_metrics.csv`（η_h 与 p_8_16 / A2_8_16 / angular_entropy_8_16 / ilr_z2
  的 Spearman，盒内 101）供核心图 6 使用；H1 侧 r_W 统计以 `h1_side_informational`
  记录并显式标注**不是机制证据**（§0.17 证据优先级 direct > predicted > out-of-box）。
- 语言红线：JSON 内写入 `interpretation_note`，重申禁用 harmonic、须与 G-SL3 合并读。

### 9.2 Task 20（SL-05 / G-SL4）由冻结骨架落地为完整实现

`20_orientation_provenance_check.py`：37 行 → 约 190 行。要点：

- `provenance_valid = false` 路径完整实现：**不计算任何 scan/hatch-relative Δθ**，
  G-SL4 = `NOT_APPLICABLE`，并写出 `orientation_provenance.json` 逐条登记四条事实
  （弓字形填充无逐样本轴、`theta_session_deg` −0.70..−0.45° 是图像旋转约定、
  frozen 配置无方向字段、单线无 hatch 且起终点符号未知）。
- **保留 1 处 `NotImplementedError` 作为守卫**：若有人把 `provenance_valid` 翻成 true
  而 §0.8 未回写，脚本先 `require` 失败再抛错，强制"先登记填充轴再开 conditional 臂"。
- image-frame descriptive 臂：`theta_stripe_8_16` 对 90° 整数倍的聚集比例，对照
  `[0,180)` 均匀角置换 null（10,000 次，seed+800），并在 JSON 中写入
  `DESCRIPTIVE_ONLY__NOT_EVIDENCE` 与措辞约束。
- **T16 负向断言**：输出列名含 `scan_relative/hatch_relative/delta_theta/...` 任一
  即 `require` 失败。
- 顺带核实：实测 `theta_stripe_deg`(8_16) 中位数 **90.46°**、范围 3.95–176.38°
  （200 样本）—— 但该读数**仅登记于 descriptive 臂，不得作为 scan/hatch 对齐证据**。

### 9.3 补齐单测 T14 / T16 / T18 / T22（测试数 23 → 27）

新增 `Task19Task20Tests` 类，沿用既有风格（依赖产物缺失时 `SkipTest`）：

| 测试 | 内容 | 当前状态 |
|---|---|---|
| T14 | h=NA 不入 r_h 表；out_of_box 不入 primary r_W 臂 | SkipTest（待 Task 19 产物） |
| T16 | provenance_valid 必须为 false；输出列禁含 scan/hatch-relative | 配置断言**已生效通过**；产物部分 SkipTest |
| T18 | 负面断言：`outputs/phase2_6` 无 N4→5/pseudo-pass 产物 | **已运行通过**（扫描 129 文件，0 违禁） |
| T22 | montage 盲性：config 去注释后无 8/16；脚本无 `8.0`/`16.0`/`BAND`；vs-band 图以 120/120 QA 为前提 | **已运行通过**（含阳性对照验证） |

### 9.4 细则回写（三处）

1. **§0.17** 新增 `W_unavailable` 态：`W_lower_bound` 的方向性只对 `right_censored`
   成立，而实测非 estimable **全部是 `insufficient_sections`**（right_censored = 0），
   这类条件无宽度估计、不是下界；并写入 direct bridge 实测可用性 13/19（缺失 5 个，26%）。
2. **§0.15 rev2 补注 (a)** 按实跑 config 回写：`depth_frac 0.80→0.50`、`median→P90`、
   宽度条件撤销、间隙 `2 µm→10 µm`、新增碎片守卫；原文数值保留作变更留痕。
3. **§0.14** 增列**偏差登记**：冻结后 config 的两处改动（stable_region、
   `pilot_reconcile_min_agreement` 的 abort 门降级）如实登记，并声明
   "记录时机在 formal 首跑之后"不得隐去，终判须显式复述。
   同时 §15 增加"实施状态"与"G-SL1 预读"两条，把 §3 的读数固化进冻结文档。

### 9.5 环境限制与未验证项

- 审计环境**无 numpy / pandas / scipy / sklearn / matplotlib**，且沙箱包索引不可用
  （`No matching distribution found for pandas`），故 **Task 15–20 与单测均未能实际执行**。
- 因此 9.1/9.2 的实现与 9.3 的测试**只做了静态校验**：语法编译通过、算法用纯 Python
  等价实现验证（d_int 边界、G-SL2 判定阶梯五分支、p 值公式、方向 null 解析期望
  `4c/180 = 0.2222` 与蒙特卡洛一致）、新增断言均带阳性对照。
- **首次 formal 运行须以 `--quick` 先行冒烟**，再用完整口径跑（Task 19 的 10,000 次
  置换是耗时主体）。

> ⚠️ 排查中发现一个环境陷阱，供后续复现：**Git Bash（MSYS）会把传给 python 的
> 反斜杠转成正斜杠**（`\s` → `/s`），导致 shell 内联的正则静默失配、产生假阴性。
> 本次的验证脚本全部改用 `chr(92)` 构造反斜杠后才得到可信结果。

## 附：本次核查用到的事实基线

- 单线视场：285.3448 × 17.8340 µm（1024 × 64 px @ 0.278657 µm/px），120 组 `valid_pixel_ratio = 1.0`
- 单线 DOE：120 个互不重复条件，τ∈{223,500,1000,2000,4000} fs、f∈{2,5,10,20,40} kHz、
  v∈{5,10,15,20,25} mm/s、N∈1..5
- 矩形侧 200 样本：盒内 101 / 盒外 99；exact_match 20 样本 / 19 条件
- `provenance json` 已记录：annotator 自述 GPT（AI 辅助标注），盲性保持，120 条标签
