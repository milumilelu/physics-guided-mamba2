# Phase 2.7 落地执行细则（how）

> 状态：**FROZEN（2026-09-04）**。上位规划 = `任务说明/Phase2.7_单轨谱包络与hatch阵列尺度选择_研究任务说明.md` **v2.1 FROZEN**（`776bf7b`）——全部科学定义、公式、门槛、判定顺序以该文件为准，本文件只登记 how：文件路径、config 键、QA 断言、运行顺序。两文件冲突时以上位规划为准。
> 事实基线：Phase 2.6 gate 终判 rev2（`8189fc7`）与 `summary/RUNTIME_ENVIRONMENT.md`（强制 `.venv`）。

## 0. 收紧点（差异决策登记）

1. **脚本编号 21–23**（延续 15–20）；`experiments/phase2_7/_lib.py` 以 importlib 链式加载 `phase2_6/_lib.py`（`p26`，其内部已加载 p25/p2/l15），禁止复制实现；Task 12 的 ILR 空间 Q²（`q2_aitchison_ilr`）在本 `_lib` 重声明（与 `18_scale_bridge_model_compare.py` 同一定义，来源 Phase 2.5 `12_` 脚本），provenance 注明。
2. **m/OUT 指派单一实现**：`_lib.assign_class(r)` 返回 `INVALID(-1)/OUT(0)/m1(1)/m2(2)/m3(3)`；Task 21（四分类，peak-valid 条件）与 Task 23（五分类，含 INVALID）共用，禁止第二套。
3. **Task 22 总体 = 全 200**（无 Ŵ 参与）；in-box 101 为 sensitivity 并列（splits 在子集内重新生成 + 契约校验）。Ridge/α 沿用 `p26.ridge_alpha_inner_gkf`（与 Task 18 M0_u 同协议，保证 Δ 与 2.6 结果可比）。
4. **Task 23 总体 = estimable ∧ qa≠reject（81）**；usable-only（18）作 sensitivity。profile 边界判据：`max(|g_边3px|) ≤ 0.15·D_max` 否则标记 `UNSUITABLE_FOR_SYNTHESIS`（不硬零延拓）。
5. **模拟规模冻结**：c 网格扫描（LOHO）= 16 phases × 27 线子采样（81 线等距取 27）；终值 q_M,h(c*) 与 c=0 = 32 phases × 81 线全 library。3A guard 用自家线 × 32 phases × {c=0, 全局 c*}。
6. **c_guard（3A 用）**= LOHO 五折 c* 的众数（tie 取小 c）；登记为非 in-sample 量。
7. **随机性登记**：seed = 20260904；TV 置换/LOHO/bootstrap/phase-grid 均为 deterministic（无随机采样——phase 是 deterministic grid，bootstrap 是重采 DOE units，LOHO 是穷举网格）；bootstrap B=2000，replicate 内重跑 LOHO。
8. 预冻结提交先于一切 formal；此后门槛/区间/判定顺序禁改。

## 1. 数据契约

全部输入为已冻结产物（见上位规划 §5 表）；单线高度经 `src.io_cag.CagHeightReader` 现读，冻结平面/轴/稳定区复用 Task 16 机制（`p26.sample_profiles / line_extent / plateau_stable_run / lateral_positions / axis_frame / scan_plateau_features`）。矩形 targets 直采 Phase 2.5 CSV。模拟谱管线 = `p25.radial_spectrum(R, 0.5, 24, 0.7, 160.0)` → 组 long DF → `p26.lambda_peak_4_32(window=(4,32), n_modes_min=20, share_min=0.20)` → `assign_class`。

## 2. Task 21 — `21_peak_selection_decomposition.py`

- 输入：`outputs/phase2_6/scale_bridge/lambda_over_hatch.csv`（200 行）。
- 四分类（peak-valid 条件）：q_h 按 h（5 水平）；总体 conditional；C_family 总体/逐 h。
- null：`p26.shuffle_h_by_block` B=10000（seed+800+perm）；重算 r → 类 → q^(b)_h（按 shuffled h 分层）；TV permutation p 按 v2.1 公式（池化中心 q̄_{0,h}）。
- H_DEPENDENT：family 内 logistic I(m=2)~h（sklearn，无正则），斜率置换 p（B=2000，unit-level permute h）。
- 判定：v2.1 顺序（coverage 优先 → TV/p 前提 → MIXED 先于 DOMINANT）。
- 输出：`peak_selection_m.csv`（逐样本）、`family_coverage.csv`、`shuffled_h_null_tv.csv`、`summary/gsl27_2_evaluation.json`。

## 3. Task 22 — `22_hatch_ablation.py`

- 三模型矩阵（`M_{-h}`/`M_h`/`M_full`）× 4 target × {src_gkf, proc_gkf}（全 200）+ {src_gkf in-box 101}（sensitivity）。
- fold-paired ΔR²_h / ΔQ²_h；retention_h。
- 判定：src 主门槛（Route T 双 target median Δ≥0.05 ∧ 4/5 折）+ proc cap（双 target median Δ>0 ∧ ≥3/5 折）。
- 输出：`hatch_ablation_cv.csv`、`summary/gsl27_1_evaluation.json`。

## 4. Task 23 — `23_single_track_envelope.py`

- (a) 81 线包络：重采样稳定截面 → 逐截面 Hann 投影 $S_g(k)$ 在候选 $k=1/(mh)$ 直接求值 → 线级/条件级平均；可测性三级表（HIGH≥2 / LOW [1.2,2) / UNMEASURABLE<1.2 cycles）；profile 边界判据（§0.4）。
- (b) 3A 对照：13 条件 × ρ_m + $d_i$ guard（期望加权定义，双样本条件 54/156）。
- (c) 3B 模拟：2D 场（§0.5 规模）→ 同管线 → 五分类 q_M,h(c)；LOHO（10 点 c 网格）→ held-out q_M,alt；constant = c=0 列。
- 判定：v2.1 顺序（MODEL_INADEQUATE → NOT_SUPPORTED → SUPPORTED/PARTIAL → d_i guard cap）+ bootstrap（B=2000，DOE units only，replicate 内 LOHO）。
- 输出：`single_track_envelope.csv`、`envelope_selection_compare.csv`、`forward_model_simulation.csv`、`bootstrap_delta_tv.csv`、`summary/gsl27_3_evaluation.json`。

## 3. config（`experiments/phase2_7/phase2_7_config.yaml`，随本细则冻结）

```yaml
meta: {random_seed: 20260904, quick_output_root: outputs/phase2_7_quick}
paths:  # 全部指向 phase2_6/phase2_5 冻结产物（同 2.6 config 的 paths 表 + cag/view manifest）
classes: {intervals: {m1: [0.75,1.25], m2: [1.75,2.25], m3: [2.75,3.25]}}
g27_2: {c_family_min: 0.70, tv_min: 0.15, p_max: 0.05, mixed_gap: 0.15, mixed_p2_min: 0.25,
        dominant_min: 0.50, low_n_family: 8, n_perm_tv: 10000, n_perm_logistic: 2000,
        h_dependent_slope: negative}
g27_1: {delta_min: 0.05, min_folds_positive: 4, route_t: [A2_8_16, angular_entropy_8_16],
        route_p: [p_8_16, ilr_z1_z4], proc_cap: {median_delta_gt: 0, min_folds_positive: 3}}
g27_3: {c_grid: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        phases_scan: 16, lines_scan: 27, phases_final: 32, lines_final: 81,
        phase_grid_sensitivity: [16, 32, 64],
        cycles: {high: 2.0, low_min: 1.2}, edge_frac_max: 0.15,
        tv: {delta_min: 0.10, period2_max: 0.20, inadequate: 0.30},
        bootstrap: {B: 2000, ci: 0.95}, h_consistency: {h_eval: [4,6,8,10], min_n_obs: 8, min_wins: 3, min_evaluable: 3},
        d_guard: {n_eval_min: 8, contradiction_frac: 0.3333}}
seeds: {tv_perm: 800, logistic_perm: 850, bootstrap: 900}
```

## 4. 测试（`tests/test_phase2_7_lib.py`，unittest；16 项对应上位规划 §7 清单）

区间边界/OUT/无 tie 负向断言；两层分布与 C_family；coverage 优先于 NO_DOMINANT；TV p 公式复现；互斥标签（0.55/0.45 → MIXED）；shuffled-h 单位数复现；H_DEPENDENT 构造数据 → YES；消融 fold-paired；包络 Parseval 抽查 + 可测性表逐格 + 边界判据；解析 comb λ>h = 0（仅解析函数）+ finite 模拟允许非零（负向断言）；同管线（2D 场走 `p25.radial_spectrum`，负向断言无第二实现）；phase 域（period2 ⊂ [0,2h)）；五分类含 INVALID；verdict 全分支互斥；LOHO/bootstrap 语义；d_i 双样本加权；总体 81 排除 reject；输出 grep 无 "harmonic"。

## 5. 输出树与运行顺序

见上位规划 §8；顺序 = 预冻结 commit（本细则 + config + 脚本 + 测试）→ `21` → `22` → `23` → `summary/phase2_7_gate_eval.md`，逐 Task commit。预算全部分钟级（Task 23 模拟 ~4 万场 160×160 DCT，约 3–5 分钟）。
