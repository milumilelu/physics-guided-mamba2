# Phase 2.7 研究任务说明：单轨谱包络 — hatch 阵列尺度选择机制

> 建议路径：`experiments/phase2_7/Phase2.7_单轨谱包络与hatch阵列尺度选择_落地执行细则.md`（配套执行细则随冻结产出）
> 建议状态：`DRAFT_FOR_REVIEW`
> 阶段定位：**短平快的机制判别阶段**——不进入预测建模，只回答三个决定论文主线的科学问题：h 对 Route T 的独立贡献、主导尺度的 m 分解、以及"单轨谱包络 × line-array"forward model 能否解释观测的峰选择。

---

## 0. 背景：Phase 2.6 把问题收窄到了什么

Phase 2.6 封账事实（全部已 formal 验证）：

1. **单轨本征宽度不在带内**：estimable 线 pooled W50 = 5.78 µm（3.8–8.1），W_eq = 5.86 µm，带内比例 1.2%/0.0% → G-SL1 NOT_SUPPORTED。"8–16 µm = 单线宽度"被否定。usable-only（n=18）同向（带内 0.0%）。
2. **λ_peak/h 强聚集于 {1,2,3}**：A_obs = 0.904（valid 104/200），block-structured shuffled-h null 中位仅 0.471，p = 0.0001 → G-SL2 SUPPORTED。但 {1,2,3} 内部主导 m 未分解。
3. **P(m|h) 初步结构**（本次核查，formal 归 Task 21）：m=2 份额随 h 单调下降——h=4: 0.75 → h=6: 0.53 → h=8: 0.26 → h=10: 0.25；h=4 另有 m=3（λ=11.85≈3h）；h=2 的峰在 λ≈18.6 µm（族外）。逐 h 峰值双族结构：h=6 → 6.01/11.85；h=8 → 7.54/14.86；h=10 → 9.45/18.64。
4. **geometry compression 不成立**：composition retention 0.592（proc 0.490）→ G-SL3 NOT_SUPPORTED；但 scalar retention 1.037 且 **M2_h（h 单独）对 A2/角熵的 src_gkf R²（0.694/0.676）不低于全工艺 M0（0.663/0.648）**——Route T 的 hatch 主导性是显式检验对象（G27-1），Route P 仍是多因素问题。
5. **direct bridge（最高权重证据）**：λ\* ≈ 2.09 × W_line_measured（13 条件，P(|r−1|≤0.25)=0/13）——"几何宽度"不是合适的频谱描述子；λ\* ≈ 2×W50 ≈ 11.6 µm 落带内，是待解释的探索性观察。

**核心科学问题**：

$$
\boxed{\text{单轨本身产生怎样的空间频谱，hatch 阵列又怎样从中选择/组织出最终表面尺度？}}
$$

## 1. 理论骨架与关键洞察（G27-3 的证伪结构）

Forward model：

$$
z(x)=\sum_{n} a_n\, g(x-nh),\qquad Z(k)=G(k)\sum_n a_n e^{-iknh},
$$

其中 $G(k)=|\mathcal F\{g\}|^2$ 是**单轨材料响应的频谱包络**（可直接从单线测量），$\sum_n a_n e^{-iknh}$ 是**线阵几何与线间幅值调制**（不可直接观测，只能通过模型比较推断）。

**关键数学事实**：常数幅值（$a_n\equiv a$）的 h-线阵，其阵因子是梳齿位于 $k=j/h$ 的 Dirichlet 核——对应波长 **λ = h/j ≤ h**。因此：

> **常数 h-阵列在数学上不可能产生 λ = 2h 的谱峰。**

而实测 m=2（λ≈2h）份额高达 34%（h=4 时 0.75、h=6 时 0.53）。所以数据**已经强制**要求以下二者之一（或兼有）：

- **(a) 幅值调制的周期加倍**：$a_n$ 存在奇偶交替/两线单元（two-line unit）→ 阵因子梳齿移到 $k=j/(2h)$ → 波长族 {2h, h, 2h/3, …}，同时解释 λ≈h 与 λ≈2h 两族峰；
- **(b) 材料响应展宽**：$h < W_{50}$ 的条件下相邻槽熔并/饱和，线性叠加失效（h=4 < 5.8 µm 正是 m=2 占优的区段）。

由于无 scan/hatch 方向 provenance（G-SL4 = NA），(a) 只能表述为 **two-line / period-doubled spatial organization**，禁止归因弓字形扫描。

**Phase 2.7 的判决逻辑**：若单轨包络 $G(k)$ 在候选频率处平滑（无锐利选择），则最终表面的尺度选择必须来自阵列因子侧——m=2 的存在就把"two-line 组织"变成几乎唯一的线性解释；若 $G(k)$ 本身在特定 $m/h$ 处有结构，则单轨材料响应直接参与尺度选择。这用 measurement→measurement 的直接比较即可判别，不需要代理模型。

## 2. 三件连续的事（Task 21–23）与三个 Gate

### Task 21 — G27-2：P(m|h) 分解（script `21_peak_selection_decomposition.py`）

对全部 200 矩形样本中 λ_peak valid 的子集（104），计算：

$$
m(\text{sample}) = \operatorname{argmin}_{m\in\{1,2,3\}} \left|\frac{\lambda_{\rm peak}}{h} - m\right|,\quad
\text{valid 若 } d_{\rm int}\le 0.25,
$$

输出 $P(m=1|h), P(m=2|h), P(m=3|h)$：总体、**按 h 水平**（2/4/6/8/10）、**按 session**（formal/pass/supplement）。族外样本（如 h=2 的 λ≈18.6 µm）单列，不进 P(m) 分母。对照组：block-structured shuffled-h null（复用 §0.19 置换机制）给出 $P_{\rm null}(m)$ 与 TV 距离置换 p 值。

**Gate G27-2**：
- `DOMINANT_m`：若存在 $m^*$ 使 $P(m^*) \ge 0.50$ 且观测分布与 null 的 TV 距离 ≥ 0.15（置换 p ≤ 0.05）→ 报告 "DOMINANT_m=$m^*$"；
- `MIXED`：前两名 P 相差 < 0.15 → 报告 "MIXED（h 与 2h 竞争）"；
- 其余 → `NO_DOMINANT`。
- 附带检验：$P(m=2|h)$ 对 h 的单调趋势（Spearman( h, m=2 份额 ) + 逐 h CI）——这是"熔并/两线单元随 h 演化"的证据。

### Task 22 — G27-1：hatch 的 unique contribution（script `22_hatch_ablation.py`）

干净消融三元组（Ridge，Phase 2.5/2.6 同 CV 契约）：

$$
M_{-h}: Y\sim[\tau,f,N,v],\qquad M_h: Y\sim h,\qquad M_{\rm full}: Y\sim[\tau,f,h,N,v],
$$

$$
\Delta R_h^2 = R^2(M_{\rm full}) - R^2(M_{-h}),\qquad
\text{retention}_h = R^2(M_{-h})/R^2(M_{\rm full}).
$$

- targets：`A2_8_16`、`angular_entropy_8_16`（Route T）；`p_8_16`、`ilr_z1_z4`（Route P，Q² 用 ILR 空间定义）。
- 总体：**全 200**（无 Ŵ 参与，不涉及跨数据集外插，沿用 Phase 2.5 CV 契约；in-box 101 作 sensitivity 并列）——与 2.6 的"盒内 101 primary"不冲突（那是 Ŵ 桥的约束）。
- src-GKF / proc-GKF 双 CV；fold-paired Δ。

**Gate G27-1**：src_gkf 下 Route T 两个 target 的 median ΔR²_h ≥ 0.05 且 ≥4/5 折 Δ>0 → **SUPPORTED（Route T hatch 主导）**；仅一个 target 达标 → PARTIAL；Route P 的 Δ 与 contrast（Δ_T − Δ_P）为描述性报告。预期：Δ(Route T) 大而 retention 低，Δ(Route P) 小——正式确立"Route T hatch 主导、Route P 多因素"。

### Task 23 — G27-3：单轨谱包络直接测量 + forward model 判别（script `23_single_track_envelope.py`）

**(a) 包络测量（measurement→measurement，不做代理）**：对每条 estimable 单线（84 条；usable-only n=18 作敏感性），逐稳定截面 $g(v)$（64 px @ 0.278657 µm）加 Hann 窗做周期图，按线平均得 $G_{\rm line}(k)$；在候选频率 $k=1/h,\,1/(2h),\,1/(3h)$ 处读值。**可测性判据**（预注册）：候选波长 λ=m·h 需 ≤ FOV/1.2 ≈ 14.9 µm（cycles ≥ 1.2）→ h=2 全可测；h=4 可测 m=1,2（12 µm 边缘，标记 low-confidence）；h=6 可测 m=1（2h=12 low）；h=8/10 仅 m=1。低置信格单独列出，不进 flatness 统计。

**(b) 选择函数比较**：对 13 个 measured-W exact-match 条件（先 measurement→measurement），比较
$$
\rho_{m} = \frac{G_{\rm line}(1/(mh))}{G_{\rm line}(1/h)}
$$
与矩形侧谱能量在 λ≈mh 处的份额（Phase 2.5 `radial_spectrum_long` 同窗读数）——检验"矩形在 λ=2h 的能量超出"是否伴随"单轨包络在 1/(2h) 的系统性抬升"（(b) 材料展宽解释）还是包络平坦（(a) 阵列侧解释）。

**(c) forward model 数值判别**：频域合成 $Z(k)=G_{\rm line}(k)\cdot A(k)$，$a_n$ 三模型：`constant`（梳齿 λ=h/j）、`alternating`（两线周期，梳齿 λ=2h/j）、`random`（宽瓣）；每模型对 h∈{2,…,10} 生成合成谱取 λ_peak → 预测 $P_{\rm model}(m|h)$，与 Task 21 观测分布比 TV 距离。

**Gate G27-3**：
- `SUPPORTED`：TV(alternating) ≤ TV(constant) − 0.10 且 TV(alternating) ≤ 0.25——"两线周期组织 + 单轨包络"重现观测峰选择；
- `PARTIAL`：仅排序成立（alternating < constant 但余量不足）；
- `NOT_SUPPORTED`：alternating 无优势——指向材料非线性（h<W 熔并）为 m=2 的主因，登记为 Phase 3 的材料响应问题。

## 3'. 三个 Gate 压缩后的主线判读

$$
\boxed{\text{G27-1：h 是否对 Route T 提供独立且稳定的增量解释？}}\quad
\boxed{\text{G27-2：主导峰到底是 }h,\;2h,\text{ 还是混合？}}\quad
\boxed{\text{G27-3：单轨谱包络 × line-array 模型能否解释观测 peak selection？}}
$$

三分支终判：
- G27-1 ✓ + G27-2 DOMINANT_m + G27-3 SUPPORTED → 项目主问题正式收敛为：**"超快激光单轨材料响应的空间频谱如何被 hatch 填充阵列选择和调制，从而形成氧化锆表面的尺度特异谱分配与方向纹理？"**——Phase 3 围绕该结构做定量建模；
- G27-3 = NOT_SUPPORTED（材料非线性主因）→ 主问题改为"单轨/邻轨材料响应非线性"引导的 Phase 3；
- G27-2 = MIXED 且 G27-3 不判别 → 保持"尺度特异谱分配"的现象学主线，Phase 3 先做可观测性扩展（方向 provenance 补测实验）。

## 4. 明确不做什么

- 不进入 Phase 3 预测建模（Mamba/深度模型等）；
- 不重跑 Phase 2.6 主结果（G-SL1/2/3 冻结）；
- 不将 m 分解结果称为"谐波机制"（语言边界沿用 §14：integer-multiple / two-line / period-doubled）；
- 不将 two-line 组织归因弓字形扫描（方向 provenance 缺失）；
- 不对 h=2 的族外峰（λ≈18.6 µm）强行纳入 P(m)（单列描述）；
- 不用 W50 作为单轨频谱描述子（2.6 已证不合适；用完整包络 G(k)）；
- 不在 13 个 exact-match 条件之外做包络—矩形因果陈述（跨条件比较属 Task 23b 的描述性外推）。

## 5. 数据与依赖（全部已冻结，无新测量）

| 输入 | 来源 | 用于 |
|---|---|---|
| λ_peak/λ\*、validity、r_h 表 | `outputs/phase2_6/scale_bridge/lambda_over_hatch.csv` | Task 21 |
| block shuffle 机制 | `_lib.shuffle_h_by_block`（120/15/10 单位） | Task 21 null |
| manifest（CV 契约列） | `outputs/phase2/manifest/phase2_manifest.csv` | Task 22 |
| 矩形 targets（A2/角熵/p_8_16/ilr） | Phase 2.5 spectral/directional CSV | Task 22 |
| 单线高度（重新采样剖面） | `氧化锆/120组直线.cag` + `_lib.sample_profiles` + 冻结平面/轴 | Task 23 |
| estimable 线清单与稳定区 | `single_line/single_line_geometry.csv` + `geometry_qa_labels.csv` | Task 23 |
| 矩形径向谱能量（选择函数对照） | `outputs/phase2_5/spectral_composition/radial_spectrum_long.csv` | Task 23b |
| 13 exact-match 条件映射 | `scale_bridge/direct_bridge_exact_match.csv` | Task 23b |

## 6. 运行环境与预算

强制 `.venv`（见 `outputs/phase2_6/summary/RUNTIME_ENVIRONMENT.md`）。预算：Task 21 置换 10,000×3 组（分钟级）；Task 22 为 4 target × 3 模型 × 4 CV 变体（分钟级）；Task 23 含 84 线 × ~90 截面的周期图 + 3×5 合成谱组（分钟级）。无训练大数据。

## 7. 最低测试（`tests/test_phase2_7_lib.py`）

1. m 指派：d_int ≤ 0.25 边界、族外样本剔除、并列取小 m（预注册）；
2. P(m|h) 按session/h 分层行列完整（5 h × 3 session 无缺格）；
3. shuffled-h null：单位数 120/15/10 不变、固定 seed 复现、TV 距离 ∈ [0,1]；
4. 消融 Δ：fold-paired（同一 split 下 M_full − M_{-h}）、契约校验、Route P/T 分组正确；
5. 包络：Hann 周期能量守恒（Parseval 抽查）、候选 k 读数落在 DFT 网格插值范围内、可测性判据（cycles ≥ 1.2）逐格登记；
6. forward model：constant 模型在 λ>h 处功率为零（硬断言——理论事实的代码锚）、alternating 在 λ=2h 有峰、TV ∈ [0,1]；
7. 语言边界负面断言：全部输出文件 grep 不到 "harmonic"。

## 8. 输出树

```text
outputs/phase2_7/
  peak_selection/
    peak_selection_m.csv            P(m|h) 总体/按h/按session
    gsl27_2_evaluation.json
  hatch_ablation/
    hatch_ablation_cv.csv           M_{-h}/M_h/M_full × 4 target × 4 CV
    gsl27_1_evaluation.json
  envelope/
    single_track_envelope.csv       84 线 × 候选 k 读数 + 可测性/置信标记
    envelope_selection_compare.csv  13 条件 measurement→measurement 对照
    forward_model_simulation.csv    3 a_n 模型 × h 的预测 P(m|h)
    gsl27_3_evaluation.json
  summary/
    phase2_7_gate_eval.md
```

## 9. 执行顺序

1. Phase 2.6 封账（**已完成**，`b77ec38`）；2. 细则评审冻结；3. `21` → `22` → `23` 逐 Task commit；4. `phase2_7_gate_eval.md` 终判 → commit。预计一个工作日内完成全部 formal。
