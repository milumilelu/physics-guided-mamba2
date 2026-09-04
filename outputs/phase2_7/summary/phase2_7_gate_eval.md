# Phase 2.7 gate 评估（rev1）

> 状态：formal 完成（2026-09-04）。任务说明 v2.1 FROZEN（`776bf7b`）；预冻结（`4959a2e`）；formal（`07801cc`）。运行环境强制 `.venv`（`RUNTIME_ENVIRONMENT.md`）。

## 1. Gate 结果

| Gate | 判定 | 关键数值 | 证据文件 |
|---|---|---|---|
| G27-1（hatch unique contribution） | **SUPPORTED** | Route T 双 target src median ΔR²_h：A2 0.571 / 角熵 0.638（均 ≥0.05，≥4/5 折正）；proc Δ 0.734/0.739（一致性 cap 不触发）；Route P Δ 0.165/0.296（远小于 Route T） | `summary/gsl27_1_evaluation.json`、`hatch_ablation/hatch_ablation_cv.csv` |
| G27-2（m 分解 + hatch periodicity） | **DOMINANT_m=1 + H_DEPENDENT=YES** | C_family 0.904（≥0.70）；TV_w 0.297、p=0.0001；conditional P(m)= {m1 0.628, m2 0.340, m3 0.032}；**P(m=2\|h) 单调递减**：h=4→0.545, h=6→0.471, h=8→0.257, h=10→0.250；logistic slope −0.332, p=0.0005 | `summary/gsl27_2_evaluation.json`、`peak_selection/family_coverage.csv` |
| G27-3（envelope × array model） | **MODEL_INADEQUATE** | TV_w(constant) 0.707 / TV_w(period2/LOHO) 0.652，双双 >0.30；ΔTV=0.055 方向为正（CI 下界 0.054>0）但绝对拟合均不足 | `summary/gsl27_3_evaluation.json`、`envelope/forward_model_simulation.csv` |
| G27-4（direction alignment） | **NOT_APPLICABLE**（Phase 2.6 G-SL4 承接） | 无方向 provenance | — |

## 2. 科学发现

**G27-1 确立 Route T hatch 主导**：hatch 单独（M_h）对 A2_8_16 的 R²=0.486、角熵 0.546，与全工艺 M_full（0.505/0.596）相当；去掉 h 后塌方（M_{-h} = −0.066/−0.043）→ ΔR²_h 0.571/0.638，proc 同向。**Route T 的预测能力几乎完全由 hatch spacing 单变量承载**。

**Route P 仍是多因素问题**：composition Q² 的 ΔR²_h = 0.165（h 贡献存在但不足），p_8_16 ΔR²_h = 0.296——去 h 后 M_{-h} 保留 0.127/0.148，说明 τ/f/N/v 对 Route P 有独立贡献。

**G27-2 确立 m 分解结构**：全局 P(m) = m1 62.8% / m2 34.0% / m3 3.2%（DOMINANT_m=1）。但 **H_DEPENDENT=YES**（slope −0.332，p=0.0005）揭示了一个单调 h 趋势：

| h | C_family | P(OUT) | P(m=1) | P(m=2) | P(m=3) |
|---|---|---|---|---|---|
| 2 | 0.00 | 1.00 | 0 | 0 | 0 |
| 4 | 0.73 | 0.27 | 0 | **0.55** | 0.18 |
| 6 | 0.94 | 0.06 | 0.41 | **0.47** | 0.06 |
| 8 | 0.97 | 0.03 | **0.71** | 0.26 | 0 |
| 10 | 1.00 | 0 | **0.75** | 0.25 | 0 |

**h=2 全族外**（λ_peak≈18.6 µm，10/5=9.3h）；**h=4 偏向 2h/3h**；**h=6 竞争**（m1≈m2）；**h=8/10 收敛 m=1**。h=2 的 C_family=0 是"槽完全熔并"的几何必然（h=2 < W50=5.8 → 相邻槽不可分辨）。

**G27-3 = MODEL_INADEQUATE**：finite-array 模拟（含 period-2 组织）的 TV_w 双双 >0.30。ΔTV=0.055 方向为正且 bootstrap CI 下界 >0（period-2 优于 constant，排序支持），但两个模型的绝对拟合均不足以称"重现观测"。可能原因：单轨 FOV（17.8 µm）只能覆盖 1–2 个周期的横向频谱测量（h≥8 的 m=2 不可测）；一维截面模型压缩了 2D 谱结构；实际表面还有材料非线性（h<W 的熔并、重铸）。

## 3. 判读与下一步

**Phase 2.7 的贡献**：将 Phase 2.6 的"hatch-related scale"收敛为一个具体的 h 依赖机制图景——h < W50 时槽完全熔并（h=2 全族外）；h ≈ W50 时竞争态（h=6 的 m1/m2 接近均分）；h > W50 时 hatch 周期主导（h=8/10 的 m1 占优）。ΔR²_h 数据独立确认 Route T 完全由 hatch 几何承载。

**进入 Phase 3 的条件已成熟**：Route T 主问题 = "hatch spacing 如何通过单轨响应的空间频谱选择出最终的表面方向纹理与尺度分配"；Route P 主问题 = "多因素（含 h 但不限于 h）如何决定完整谱组成"。两条线在 Phase 3 可分别建模或联合建模（shared representation + task-specific heads），且都有明确的物理锚点。

**弱点登记**：① 单轨 FOV 限制了 m≥2 的包络测量（h≥8 时 2h 不可测）；② 1D 截面模型压缩了 2D 谱结构；③ 模拟未含材料非线性（熔并/重铸/氧化动力学）；④ AI 辅助 QA 标注（GPT）可能存在系统性偏好。
