# Phase 2.6 gate 评估（rev2，封账版）

> 状态：**封账（2026-09-04）**。rev1 判定全部维持；rev2 仅做报告层修正（不重跑主结果，G-SL1/2/3 结论不变）：① 终判表述从"8–16 µm = hatch"收紧为"**8–16 µm 方向纹理具有强 hatch-related scale 证据**"（G-SL2 证明的是 λ_peak/h 聚集于 {1,2,3}，未分解到具体 m）；② 登记 formal 后稳定区规则发生过 **adaptive refinement**（central-70% → 深度台地最长连续段 + 碎片守卫 + qualifying-only 截面，经多轮实跑迭代定型——非预注册一步到位，已回写 §0.15 补注并留全程轨迹）；③ Task 17 variant 混报修复（GSS 折曾误标 variant）与 **usable-only sensitivity** 补充（n=18：W50 带内 0.0%、W_eq 带内 0.0%——保守下界与主口径同向）；④ Q5 回答改述为"**geometry compression 不成立**"（原文"劣于"暗示 h 差，实际 scalar retention 1.037 不劣于全工艺，是 composition 保持率 0.592 不达 0.80）。
> 事实链：config/门槛预冻结于细则 v2（`2daa611`）；formal 链 Task 15–20 + 审计修复（`65ded0e`→`8e646a1`）；封账修正（`8189fc7` 之后、本提交）。运行环境强制 `.venv`（见 `summary/RUNTIME_ENVIRONMENT.md`——conda 解释器下 α 翻转致对账超差，复现必读）。

## 1. Gate 结果

| Gate | 判定 | 关键数值 | 证据文件 |
|---|---|---|---|
| G-SL1（line-width scale alignment） | **NOT_SUPPORTED**（0/3） | pooled section W50 median **5.78 µm**（不在 [8,16)）；线级带内比例 **1.2%**（1/81，<50%）；pooled section W_eq median **5.86 µm**（不在带内）；n_estimable 81；repaired 臂带内比例差 0.000（无 divergent 降级） | `summary/gsl1_evaluation.json`、`model_compare/W_line_distribution_vs_band.csv`、`scale_bridge/width_identifiability_summary.csv` |
| SL-03a（direct bridge，最高优先级证据） | 对 H1 不利 | 19 条件（13 estimable / 5 W_unavailable / 1 rejected_by_qa，缺失率 26%）；**median r_W_direct = 2.090**（IQR ≈ [1.79, 2.73]）；P(\|r−1\|≤0.25) = **0/13**；Spearman(λ\*, W_measured) = 0.253 | `scale_bridge/direct_bridge_exact_match.csv` |
| G-SL2（hatch integer-multiple scale） | **SUPPORTED** | 有效 λ_peak 104/200（52%）；**A_obs = 0.904** vs block-structured null 中位 0.471，**p = 0.0001**（10,000 次置换，单位 = unique(session_id, base_condition_group) = 120/15/10）；centroid sensitivity A_obs = 0.408 / p = 0.021（并排不替代主判定） | `summary/gsl2_evaluation.json`、`scale_bridge/shuffled_h_null.csv`、`scale_bridge/lambda_over_hatch.csv` |
| G-SL3（Geometry-compression） | **NOT_SUPPORTED** | composition retention **0.592**（< 0.80；proc_gkf 0.490）；scalar median retention 1.037；n_folds≥0.60 = 2/5 | `summary/gsl3_evaluation.json`、`model_compare/width_bridge_cv.csv` |
| G-SL4（direction alignment） | **NOT_APPLICABLE** | provenance_valid = false（v2 §12 仅"弓字形"、无逐样本轴；单线无 hatch、起终点符号未知）；image-frame 0°/90° 聚集 117/200（p=0.0001）**仅 descriptive，非证据** | `orientation/orientation_provenance.json` |

附：Ŵ（单线宽度工艺模型）预测力弱（primary Ridge median R² = −0.006，.venv 口径）——单线宽度本身由工艺参数解释有限，但这不影响 G-SL1 的宽度分布判定；W_hat 仅用于桥接坐标（r_W sensitivity），按 §0.13 不作为机制证据。

## 2. 终判（上位规划 §17 矩阵）

**G-SL1 NO + G-SL2 YES → 判定 B（rev2 表述收紧）：8–16 µm 方向纹理具有强 hatch-related scale 证据**——λ_peak/h 强聚集于 {1,2,3}（A_obs 0.904，p=0.0001），且该聚集不能用 DOE 结构性伪影解释（block-structured null）；但 {1,2,3} 内部的主导 m 尚未分解（P(m|h) 初步显示 m=2 份额随 h 单调下降：h=4 约 0.75 → h=10 约 0.25），"8–16 µm = hatch"的简单等式不成立，精确机制留给 Phase 2.7。

支撑链（按 §0.17 证据优先级）：

1. **direct bridge（测量→测量，最高权重）**：λ\* ≈ 2.09 × W_line_measured，P(|r−1|≤0.25) = 0——H1（λ\* ≈ 单线宽度）在最强证据上被直接否定。同时 median r_W_direct ≈ 2 意味着 2×W_line（≈ 11.6 µm）落在 8–16 带内——这是一个**探索性观察**（λ 与单线宽度的 2 倍尺度联系），不改变 B 判定，登记为后续研究入口。
2. **G-SL2（主判定）**：λ_peak/h 在 1/2/3 附近的聚集率 0.904，显著高于保持 DOE block 结构的 shuffled-h null（0.471，p=0.0001）——整数倍 hatch 阵列结构成立；缺失率与覆盖（valid 52%）已并排报告。
3. **G-SL3 不支持**：加入 W/h 后 composition 可预测性的保持率不足（0.592）——"五维工艺关系可压缩为宽度–overlap 几何"的更强命题**不成立**；scalar 臂 retention 1.037 只说明几何量对方向纹理不劣于全工艺，不构成机制证据。
4. **G-SL4 不可判定**：无方向 provenance，不暗示任何 scan 对齐。

语言边界（§14）执行：全文以 "hatch-related periodic / integer-multiple scale（multi-line / envelope scale）" 表述，未使用 harmonic；未命名热/相变/脆性机制；W_line 与 Fourier 波长未混同。

## 3. 必答 8 问题（上位规划 §22）

1. **单线有效宽度真实范围**：estimable 线 pooled W50 = 5.78 µm（线级 3.8–8.1 µm），W_eq = 5.86 µm；W20 = 8.04 µm（含过渡带的受影响宽度）。视场截断未发生（W20 max 12.33 < 17.83 µm；censoring ≈ 0）。
2. **8–16 µm 是否覆盖单线宽度主分布**：否（W50 带内 1.2%，W_eq 0%）。
3. **宽度的主要受控变量**：预测力弱（R² ≈ 0，f 为 Ep-coupled 不可分离）——单线宽度对 (τ,f,v,N) 的响应弱于预期，宽度更多受局部/材料因素调制（response curves 见 `W_line_response_curves.csv`，描述性）。
4. **λ 更接近 W、h 还是 2h**：λ_peak/h 聚集 1/2/3（G-SL2 SUPPORTED）；λ\*/W_line ≈ 2.09（direct bridge）→ **更接近 h 的整数倍尺度；与 2×W_line 的重合为探索性观察**。
5. **geometry compression 是否成立**（rev2 改述）：不成立——3 个几何量 [Ŵ, h, Ŵ/h] 只保留 composition 可预测性的 59%（retention 0.592 < 0.80，G-SL3 NOT_SUPPORTED）；注意这不是"W/h 劣于 h"——scalar 臂 retention 1.037 说明几何量对 A2/角熵**不劣于**全工艺，失败 specifically 发生在完整谱组成上（Route P 仍需多因素）。另外实测 M2_h（h 单独）对 A2/角熵的 src_gkf 中位 R² 为 0.694/0.676，与全工艺 M0（0.663/0.648）相当——Route T 的 hatch 主导性是 Phase 2.7 G27-1 的正式检验对象。
6. **Route P/T 是否随 overlap 几何改变**：几何量（Ŵ,h,Ŵ/h）对四 target 的 Spearman 显示方向性关联（A2 −0.836、角熵 +0.861、p_8_16 −0.630、ilr_z2 +0.564），但压缩保持率不足 → 关联存在、可压缩性不成立。
7. **条纹方向与 scan/hatch 对应**：不可判定（无 provenance，G-SL4 = NA）；image-frame 0/90 聚集 117/200 仅为 descriptive。
8. **最终解释**（rev2 收紧）：8–16 µm 方向纹理具有**强 hatch-related scale 证据**（判定 B），但主导尺度（h vs 2h vs 混合，及 P(m|h) 的 h 依赖）未分解；单轨本征宽度（≈5.8 µm）不在带内已确证；overlap-composite 压缩命题未获支持。理论要点（Phase 2.7 动机）：常数幅值 h-线阵的傅里叶梳齿位于 λ=h/j ≤ h，**不可能产生 λ=2h 峰**——实测 34% 的 m=2 份额因此要求两线周期组织（a_n 奇偶交替/two-line unit）或材料响应展宽（h < W 的熔并），这正是 G27-3 的证伪结构。

## 4. 权力与弱点（必读）

- direct bridge 缺失率 5/19 = 26%（W_unavailable，无方向性；另 1 条件 rejected_by_qa）——direct 统计基于 n=13 条件，权重与缺失率必须一并引用。
- λ_peak valid 比例 52%——G-SL2 的峰证据覆盖过半但非全量；centroid sensitivity（A_obs 0.408 / p 0.021）方向一致但弱。
- 功率 provenance 弱（5.3333 W 实测值，无独立记录；PENDING_REGISTRATION）——f 与 Ep 完全耦合，所有 f 效应表述均为 "f (Ep-coupled)"。
- G-SL1 总体 n=81（estimable ∧ 非人工否决）；36 条 insufficient 线（多为碎片守卫拦截）不进任何 gate 统计。
- 单线 QA 标注由 AI 辅助完成（annotator=GPT，盲态保持），provenance 已如实登记；若需双人独立标注复核，现有标注可作为第一盲评轮次。
