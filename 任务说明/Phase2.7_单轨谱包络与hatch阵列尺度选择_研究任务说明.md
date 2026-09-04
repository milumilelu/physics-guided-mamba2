# Phase 2.7 研究任务说明：单轨谱包络 — hatch 阵列尺度选择机制

> 建议路径：`experiments/phase2_7/Phase2.7_单轨谱包络与hatch阵列尺度选择_落地执行细则.md`（配套执行细则随冻结产出）
> 状态：**FROZEN（2026-09-04，v2.1 收口后经外审确认直接冻结）**
> 阶段定位：**短平快的机制判别阶段**——不进入预测建模，只回答三个决定论文主线的科学问题：h 对 Route T 的独立贡献（G27-1）、主导尺度的 m 分解（G27-2）、以及"单轨谱包络 × line-array"observation model 能否解释观测的峰选择（G27-3）。

> **v2.1 收口（2026-09-04 外审，九项冻结）**：
> 1. G27-2 状态优先级钉死：**C_family < 0.70 → `INSUFFICIENT_FAMILY_COVERAGE`（优先）**；仅当 coverage 合格而 TV_w < 0.15 或 p_perm > 0.05 时才 `NO_DOMINANT`。
> 2. TV permutation p 唯一公式：$\bar q_{0,h}=\frac1B\sum_b q_h^{(b)}$；$T_{\rm obs}=\sum_h w_h TV(q^{obs}_h,\bar q_{0,h})$；$T_b=\sum_h w_h TV(q^{(b)}_h,\bar q_{0,h})$；$p=\frac{1+\#\{T_b\ge T_{\rm obs}\}}{1+B}$。
> 3. **period-2 phase 边缘化用 [0,2h)**（基本周期 2h，[0,h) 会漏 parity 边缘化）；constant 用 [0,h)；均 32 个均匀 deterministic offsets。
> 4. **G27-3 的 TV 改五分类** $q_h^{obs}=[P(INVALID),P(OUT),P(1),P(2),P(3)]$，分母 = 该 h 的**全部**矩形样本——模型必须同时复现峰有效率与峰选择；Task 21 保持四分类（peak-valid 条件分布）不变。
> 5. synthetic line-array 做成 **160×160 二维场**（$\sum_n a_n g(x-nh)$ 沿线方向复制），**直接调用 Phase 2.5/2.6 同一套 2D residual→radial spectrum→peak validity 代码**；取消一切"1D 类比"规则；单轨 profile 边界处理冻结：背景校正后在实测支持（17.83 µm）外为 0，两侧未回背景的 profile 标记不适合 forward synthesis。
> 6. verdict 判定顺序固定：**MODEL_INADEQUATE → NOT_SUPPORTED → SUPPORTED/PARTIAL → direct-guard cap**（先判双模型 TV_w>0.30；再 ΔTV≤0；再四条 SUPPORTED 条件；最后 exact-match guard）。
> 7. bootstrap **只重采 DOE units**（unit=(session_id, base_condition_group)，h×session 层内有放回），每 replicate 内重跑 LOHO c*；32-phase 是 deterministic marginalization 不参与 bootstrap，另做 16/32/64 phase-grid sensitivity。
> 8. exact-match contradiction 可单测定义：$d_i=q_{P2,i}(c_i)-q_{C,i}(c_i)$（各条件由 constant/period2 的 finite simulation 得类别概率）；n_eval≥8 且 $\#\{d_i<0\}/n_{\rm eval}>1/3$ → 封顶 PARTIAL。
> 9. h-level consistency：$H_{\rm eval}=\{h\in\{4,6,8,10\}:n_{\rm obs}(h)\ge8\}$，要求 $|H_{\rm eval}|\ge3$ 且 $\#\{h\in H_{\rm eval}: TV_h(P2)<TV_h(C)\}\ge3$；h=2 仍进全局 TV_w（OUT 结构也是数据），只是不承担 h-level 票。G27-1 加 proc-GKF 一致性 cap（src 主门槛过但 proc median Δ≤0 或 <3/5 折 → 封顶 PARTIAL）。
> 三个 TV 数值（0.10 / 0.20 / 0.30）**冻结不变**——关键是要让它们面对同一个、完整、可复现的 observation distribution。
> **v2 修订记录（四 blocker，v2.1 在其上收口）**：
> ① **P(m) 互斥区间化**：m=1: r∈[0.75,1.25]；m=2: r∈[1.75,2.25]；m=3: r∈[2.75,3.25]；其余 OUT。删除 argmin/"并列取小 m"（tol=0.25 下合法样本无并列，r=1.5 本身 OUT）。
> ② **OUT 进 primary distribution**：主分布 q_h=[P(OUT|h), P(1|h), P(2|h), P(3|h)]，分母 = 全部 peak-valid 样本；conditional P(m|family,h) 为解释性第二层；新增 family-coverage guard（C_family,all ≥ 0.70，逐 h n_family < 8 → LOW_N）；DOMINANT/MIXED 改为严格互斥的排序规则（先判 MIXED）；TV 改为四分类加权 TV_w。
> ③ **G27-3 改 finite-array observation model**：常数阵列"不可能产生 λ>h"仅对理想无限周期阵列成立；有限 N、80 µm 有限 ROI、窗口/去趋势、逐线形貌差异使 λ>h 处有 Dirichlet 旁瓣/泄漏功率。删除实际模型"λ>h 功率为零"的硬断言（保留为仅限解析 infinite-comb 的理论 sanity test）；G27-3 的真问题改为"**period-2 组织是否比 realistic finite constant array 更能解释观测 m 分布**"。
> ④ **G27-3 门槛收紧与状态补全**：TV_w(alternating) ≤ 0.20（原 0.25）；DOE-block bootstrap 95% CI 下界 > 0；period-2 参数 c = 冻结的 LOHO 选择（禁止 in-sample 挑选）；新增 `MODEL_INADEQUATE` 状态（不得从 NOT_SUPPORTED 直接推出材料非线性）；新增 exact-match consistency guard（measurement→measurement 证据可把 verdict 封顶 PARTIAL）；G27-3 拆分为 3A（13 条件 own measured envelope，primary measurement→measurement）与 3B（population simulation，secondary）。

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

## 1. 理论骨架与记号（rev2 修正）

Forward model（复幅值记号，避免 power/amplitude 混用）：

$$
\tilde G(k)=\mathcal F\{g(x)\},\qquad
\tilde A(k)=\sum_{n} a_n\, e^{-i2\pi k n h},\qquad
\tilde Z(k)=\tilde G(k)\,\tilde A(k),
$$

$$
S_Z(k)=|\tilde G(k)|^2\,|\tilde A(k)|^2 .
$$

实测单轨谱包络即 $S_g(k)=|\tilde G(k)|^2$（Task 23 直接测量）；$\tilde A(k)$ 不可直接观测，只能通过模型比较推断。

**理论边界（rev2 收紧，审计 blocker ③）**：

- **解析极限（仅作 sanity）**：理想**无限**周期阵列（$N\to\infty$，$a_n$ 常数）的阵因子趋于频率 comb，谱线位于 $k=j/h$，即 $\lambda=h/j\le h$。此陈述只用于解析 sanity test 与直觉引导。
- **有限观测**：真实数据是有限条线（$N\sim 80/h$）、80 µm 有限 ROI、窗口/去趋势、逐线形貌差异——阵因子是 **Dirichlet 核**而非 comb，观察窗卷积进一步展宽，**λ>h 处可以有非零旁瓣/泄漏功率**。因此有限常数阵列与 period-2 阵列的差异是**程度问题**（TV 距离），不是"有/无"问题。
- **G27-3 的真问题**由此改写为：**period-2 幅值组织（$a_n=1+c(-1)^n$）是否比 realistic finite constant array 更好地复现实测 m 分布？**——不允许把 λ=2h 峰的存在预先等价于 period doubling。

## 2. 三件连续的事（Task 21–23）与三个 Gate

### Task 21 — G27-2：P(m|h) 分解（script `21_peak_selection_decomposition.py`）

**m/OUT 互斥区间指派（rev2，无 tie-breaking）**：对每个 λ_peak valid 样本，令 $r=\lambda_{\rm peak}/h$：

$$
m=1:\ r\in[0.75,1.25];\quad m=2:\ r\in[1.75,2.25];\quad m=3:\ r\in[2.75,3.25];\quad \text{其余 } m=\mathrm{OUT}.
$$

区间互斥（1.25 < 1.75 < 2.25 < 2.75），不存在并列；r=1.5 本身为 OUT。

**两层分布**：

- **primary q_h**（分母 = 全部 peak-valid 样本）：
  $$q_h=[P(\mathrm{OUT}|h),\,P(1|h),\,P(2|h),\,P(3|h)];$$
- **conditional**（解释性第二层）：$P(m\,|\,m\in\{1,2,3\},h)$——"一旦属于 family，更像 h/2h/3h 中哪一个"。

**family coverage guard（v2.1：优先级钉死）**：$C_{\rm family} = 1-P(\mathrm{OUT})$；总体 $C_{\rm family,all}< 0.70$ → **`INSUFFICIENT_FAMILY_COVERAGE`（优先于一切其他标签，不落到 NO_DOMINANT）**；$C_{\rm family,all}\ge 0.70$ 才允许全局 DOMINANT/MIXED 标签；逐 h $n_{\rm family}(h)<8$ → 该 h 只报 `LOW_N`，不赋 local dominant 标签。

**null 与 TV（四分类加权）**：block-structured shuffled-h null（单位 = unique(session_id, base_condition_group) = 120/15/10，10,000 次，seed+800）给出 $q_{\rm null,h}$；

$$
TV_h=\tfrac12\sum_{c\in\{\mathrm{OUT},1,2,3\}}\left|q_{\mathrm{obs},h}(c)-q_{\mathrm{null},h}(c)\right|,\qquad
TV_w=\sum_h \frac{n_h}{N}\,TV_h .
$$

**G27-2 判定（v2.1：状态优先级钉死）**：$C_{\rm family,all}<0.70$ → **`INSUFFICIENT_FAMILY_COVERAGE`（优先，不落到 NO_DOMINANT）**；coverage 合格但 $TV_w<0.15$ 或 $p_{\rm perm}>0.05$ → `NO_DOMINANT`。置换 p 用 v2.1 唯一公式：$\bar q_{0,h}=\frac1B\sum_b q_h^{(b)}$、$T_{\rm obs}=\sum_h w_h\,TV(q^{obs}_h,\bar q_{0,h})$、$T_b=\sum_h w_h\,TV(q^{(b)}_h,\bar q_{0,h})$、$p=\frac{1+\#\{T_b\ge T_{\rm obs}\}}{1+B}$。过前提后对 conditional 分布 $P(m|\mathrm{family})$ 排序 $p_{(1)}\ge p_{(2)}\ge p_{(3)}$：

- 若 $p_{(1)}-p_{(2)}<0.15$ 且 $p_{(2)}\ge 0.25$ → **MIXED**；
- 否则若 $p_{(1)}\ge 0.50$ 且 $p_{(1)}-p_{(2)}\ge 0.15$ → **DOMINANT_m**；
- 否则 → **NO_DOMINANT**。

**H_DEPENDENT 独立标记（不与上述互斥）**：sample-level logistic $I(m=2)\sim h$（family 内），斜率 p 值用与 DOE block 一致的置换（unit-level permute h → 重算斜率）；斜率 < 0 且 p ≤ 0.05 → `H_DEPENDENT = YES`。聚合 Spearman（5 点）降为 descriptive。预期形态：GLOBAL = MIXED 与 H_DEPENDENT = YES 并存——"h 小时 2h 组织占优、h 增大后 h-scale 组织渐占优"比全局单 m 主导更有科研价值。

**输出**：$P(m)$ 总体/按 h/按 session（formal/pass/supplement）、逐 h 峰值双族清单、$P_{\rm null}(m)$、TV 表、G27-2 JSON。

### Task 22 — G27-1：hatch 的 unique contribution（script `22_hatch_ablation.py`）

（rev1 → rev2 无改动）干净消融三元组（Ridge，Phase 2.5/2.6 同 CV 契约）：

$$
M_{-h}: Y\sim[\tau,f,N,v],\qquad M_h: Y\sim h,\qquad M_{\rm full}: Y\sim[\tau,f,h,N,v],
$$

$$
\Delta R_h^2 = R^2(M_{\rm full}) - R^2(M_{-h}),\qquad
\text{retention}_h = R^2(M_{-h})/R^2(M_{\rm full}).
$$

- targets：`A2_8_16`、`angular_entropy_8_16`（Route T）；`p_8_16`、`ilr_z1_z4`（Route P，Q² 用 ILR 空间定义）。
- 总体：**全 200**（无 Ŵ 参与，不涉及跨数据集外插，沿用 Phase 2.5 CV 契约；in-box 101 作 sensitivity 并列）。
- src-GKF / proc-GKF 双 CV；fold-paired Δ。

**Gate G27-1**：src_gkf 下 Route T 两个 target 的 median ΔR²_h ≥ 0.05 且 ≥4/5 折 Δ>0 → 主门槛过；**v2.1 加 proc-GKF 一致性 cap**：两 target 均须 proc median ΔR²_h > 0 且 ≥3/5 折 Δ>0，否则封顶 **PARTIAL**（防"hatch 主导"是 source-group split 特异结果）。仅一个 target 达 src 主门槛 → PARTIAL。Route P 的 Δ 与 contrast（Δ_T − Δ_P）为描述性报告。预期：Δ(Route T) 大而 retention 低，Δ(Route P) 小——正式确立"Route T hatch 主导、Route P 多因素"。

### Task 23 — G27-3：单轨谱包络测量 + finite-array observation model（script `23_single_track_envelope.py`）

**记号与两轨拆分（审计 #17）**：

- **G27-3A（primary，measurement→measurement）**：仅用 13 个 exact-match 条件，每个矩形条件配**它自己实测的** $S_g(k)$；
- **G27-3B（secondary，population）**：81 条单线包络（estimable ∧ qa≠reject，与 Phase 2.6 总体一致——审计 #18：3 条 estimable-but-reject 线不回 primary）作为 empirical library 与每个 h 组合生成 population 分布；**不得表述为 exact-condition mechanism proof**。

**(a) 包络测量（Hann 连续 Fourier 投影，审计 #10）**：对每条入组单线，逐稳定截面 $g_j$（64 px @ 0.278657 µm，坐标 $x_j$）计算

$$
S_g(k)=\Big|\sum_j w_j g_j\, e^{-i2\pi k x_j}\Big|^2,\qquad w_j=\text{Hann},
$$

**直接在 $k=1/(mh)$ 处求值**（不取 nearest FFT bin）；按线/按条件平均得 $S_g(k)$。zero-padding 只用于画图，不得解释为分辨率提升。

**可测性三级（审计 #9，预注册表）**：$N_{\rm cycles}=L/\lambda$，$L\approx17.83$ µm；HIGH：≥2；LOW：[1.2, 2)；UNMEASURABLE：<1.2。

| h | m=1 | m=2 | m=3 |
|---|---|---|---|
| 2 | high (2) | high (4) | high (6) |
| 4 | high (4) | high (8) | **low (12)** |
| 6 | high (6) | **low (12)** | unavailable (18) |
| 8 | high (8) | unavailable (16) | unavailable (24) |
| 10 | **low (10)** | unavailable (20) | unavailable (30) |

（括号内为 λ=m·h；rev1 的"h=4 可测 m=1,2（12 µm 边缘）"系笔误——12 µm 是 m=3。）

**(b) 选择函数比较（G27-3A）**：对 13 个 measured-W 条件（h 可测子集），比较

$$
\rho_m = S_g(1/(mh))\,/\,S_g(1/h)
$$

与矩形侧径向谱在 λ≈mh 窗的能量份额——检验"矩形在 λ=2h 的能量超出"是否伴随单轨包络在 1/(2h) 的系统性抬升（材料展宽解释）还是包络平坦（阵列侧解释）。**consistency guard（审计 #16；v2.1 可单测定义）**：可评估条件 n ≥ 8 时，对条件 i 定义 $d_i=q_{P2,i}(c_i)-q_{C,i}(c_i)$（$c_i$ = 该条件观测类别；双样本条件 54/156 用观测类分布加权期望 $\sum_c q_{\rm obs,i}(c)\,[q_{P2,i}(c)-q_{C,i}(c)]$；类别概率由 constant/period2 的 finite simulation 给出）；$d_i<0$ 为 direct contradiction；$\#\{d_i<0\}/n_{\rm eval}>1/3$ → G27-3 终判封顶 PARTIAL（measurement→measurement 证据优先）。

**(c) finite-array observation model（G27-3B，审计 #3/#12；v2.1 收口）**：模拟必须复现真实观测过程——

1. **160×160 二维场**（0.5 µm、80 µm ROI）：$z(x,y)=\sum_{n=0}^{N-1}a_n\,g(x-nh-\phi)$ 沿线方向复制，$N\sim 80/h$；
2. **phase 边缘化按基本周期**：constant 用 $\phi\sim U[0,h)$，period2 用 $\phi\sim U[0,2h)$（[0,h) 会漏 parity 边缘化）；均 32 个均匀 deterministic offsets；16/32/64 phase-grid sensitivity 另做；
3. 使用实测单轨 profile（3A 条件用自家实测线；3B 从 81 线 library 均匀子采样）；**profile 边界冻结**：背景校正后在实测支持（17.83 µm）外为 0，两侧未回背景 → 标记不适合 synthesis；
4. 同样的 DC/residual convention（减 median）；
5. **直接调用 Phase 2.5/2.6 同一套 2D residual→radial spectrum→peak validity 代码**（`p25.radial_spectrum`，同 geomspace 0.7–160/24 bins 与 mode-count/峰份额 guard）——**无任何"1D 类比"第二实现**；
6. 最后做 m/OUT 互斥区间指派（与 Task 21 共用同一实现）。

模型族（审计 #11，冻结）：

$$
a_n = 1 + c\,(-1)^n,\qquad c\in[0,\,0.9];
$$

**c 由 leave-one-h-out 选择**（每次用 4 个 h 水平上 TV 最小的 $c^*$，预测剩余一个 h；五次循环汇总 held-out $P_{\rm alt}(m|h)$）——constant 即 $c=0$，同 LOHO 折评估，保证 alternating 的优势不是 in-sample 拟合优势。`random` a_n 降为 secondary negative control，不参与主 Gate。

**G27-3 判定（v2.1：五分类加权 TV + 固定 verdict 顺序）**：G27-3 的 TV 用**五分类** $q_h^{obs}=[P(INVALID),P(OUT),P(1),P(2),P(3)]$，分母 = 该 h 的**全部**矩形样本（模型生成同样五分类：模拟中 peak 无效 → INVALID 类）；Task 21 保持四分类不变——Task 21 问"已有清晰峰时属于哪族"，Task 23 问"模型能否复现整个峰观测过程"。对每模型 $M\in\{\mathrm{constant},\mathrm{period2}\}$，

$$
TV_h(M)=\tfrac12\sum_{c\in\{\mathrm{INVALID},\mathrm{OUT},1,2,3\}}\left|q_{\mathrm{obs},h}(c)-q_{M,h}(c)\right|,\qquad
TV_w(M)=\sum_h\frac{n_h}{N}TV_h(M),
$$

（同时报告未加权 $TV_{\rm macro}=\frac15\sum_h TV_h$。）$\Delta TV = TV_w(\mathrm{constant})-TV_w(\mathrm{period2})$。

- **判定顺序（v2.1，互斥）**：
  1. **MODEL_INADEQUATE 先判**：$TV_w(\mathrm{constant})>0.30$ 且 $TV_w(\mathrm{period2})>0.30$ → 立即结束（当前线性阵列模型族不足；材料非线性是候选解释之一，未经独立证据不得升级）；
  2. $\Delta TV\le 0$ 且至少一个模型 $TV_w\le 0.30$ → **NOT_SUPPORTED**；
  3. 否则查四条 SUPPORTED 条件：$\Delta TV\ge 0.10$、$TV_w(\mathrm{period2})\le 0.20$、DOE-block bootstrap 95% CI of $\Delta TV$ 下界 > 0、h-level 一致性（$H_{\rm eval}=\{h\in\{4,6,8,10\}:n_{\rm obs}(h)\ge8\}$ 且 $|H_{\rm eval}|\ge3$，$\#\{h\in H_{\rm eval}:TV_h(P2)<TV_h(C)\}\ge3$；h=2 进全局 TV_w 不投票）——全过 → **SUPPORTED**，否则 **PARTIAL**；
  4. 最后 exact-match consistency guard：n_eval≥8 且 $\#\{d_i<0\}/n_{\rm eval}>1/3$ → **SUPPORTED 封顶 PARTIAL**。

bootstrap 细节（v2.1 冻结）：B = 2000；**只重采 DOE units**（unit = (session_id, base_condition_group)，h×session 层内有放回），每 replicate 内重跑 LOHO c*；**phase 边缘化始终用冻结的 32-point deterministic grid**（nuisance 积分变量不作为随机样本；16/32/64 phase-grid sensitivity 另行报告）。$\Delta TV$ 分布 → CI 与 $p_{\rm boot}$。

## 3'. 三个 Gate 压缩后的主线判读

$$
\boxed{\text{G27-1：h 是否对 Route T 提供独立且稳定的增量解释？}}\quad
\boxed{\text{G27-2：主导峰到底是 }h,\;2h,\text{ 还是混合？}}\quad
\boxed{\text{G27-3：单轨谱包络 × line-array 模型能否解释观测 peak selection？}}
$$

三分支终判（rev2 更新）：
- G27-1 ✓ + G27-2 = DOMINANT_m 或 MIXED（附 H_DEPENDENT 标记）+ G27-3 SUPPORTED → 项目主问题正式收敛为：**"超快激光单轨材料响应的空间频谱如何被 hatch 填充阵列选择和调制，从而形成氧化锆表面的尺度特异谱分配与方向纹理？"**——Phase 3 围绕该结构做定量建模；
- G27-3 = MODEL_INADEQUATE 或 NOT_SUPPORTED → 主问题改为"单轨/邻轨材料响应（含 h<W 熔并非线性）"引导的 Phase 3，但**措辞停留在候选机制**；
- G27-2 = MIXED 且 H_DEPENDENT = YES → 即使全局混合，"m=2 份额随 h 演化"本身就是 Phase 3 的定量目标；
- 可观测性缺口（方向 provenance）在任何分支下都登记为 Phase 3 的实验补测项。

## 4. 明确不做什么

- 不进入 Phase 3 预测建模（Mamba/深度模型等）；
- 不重跑 Phase 2.6 主结果（G-SL1/2/3 冻结）；
- 不将 m 分解结果称为"谐波机制"（语言边界沿用 §14：integer-multiple / two-line / period-doubled）；
- 不将 two-line 组织归因弓字形扫描（方向 provenance 缺失，G-SL4 = NA）；
- **不得从 G27-3 = NOT_SUPPORTED 直接推出"材料非线性主因"**（MODEL_INADEQUATE 承接；升级需独立证据）；
- **不得把有限阵列旁瓣/泄漏或窗效应解释为 period doubling 的证据**（反向亦然——period-2 判定只来自模型 TV 比较）；
- population 模拟（G27-3B）**不得表述为 exact-condition mechanism proof**（那是 G27-3A 的角色）；
- 不对 h=2 的族外峰（λ≈18.6 µm）强行纳入 P(m)（OUT 单列描述）；
- 不用 W50 作为单轨频谱描述子（2.6 已证不合适；用完整包络 $S_g(k)$）；
- 不在 13 个 exact-match 条件之外做包络—矩形因果陈述（跨条件比较属 3B 的 population 描述）。

## 5. 数据与依赖（全部已冻结，无新测量）

| 输入 | 来源 | 用于 |
|---|---|---|
| λ_peak/λ\*、validity、r_h 表 | `outputs/phase2_6/scale_bridge/lambda_over_hatch.csv` | Task 21 |
| block shuffle 机制 | `_lib.shuffle_h_by_block`（120/15/10 单位） | Task 21 null |
| manifest（CV 契约列） | `outputs/phase2/manifest/phase2_manifest.csv` | Task 22 |
| 矩形 targets（A2/角熵/p_8_16/ilr） | Phase 2.5 spectral/directional CSV | Task 22 |
| 单线高度（重新采样剖面） | `氧化锆/120组直线.cag` + `_lib.sample_profiles` + 冻结平面/轴 | Task 23 |
| estimable ∧ ≠reject 线清单与稳定区 | `single_line/single_line_geometry.csv` + `geometry_qa_labels.csv`（81 条；usable-only n=18 sensitivity） | Task 23 |
| 矩形径向谱能量（选择函数对照） | `outputs/phase2_5/spectral_composition/radial_spectrum_long.csv` | Task 23b |
| 13 exact-match 条件映射 | `scale_bridge/direct_bridge_exact_match.csv` | Task 23a |

## 6. 运行环境与预算

强制 `.venv`（见 `outputs/phase2_6/summary/RUNTIME_ENVIRONMENT.md`）。预算：Task 21 置换 10,000×{peak,centroid} + logistic 置换（分钟级）；Task 22 为 4 target × 3 模型 × 4 CV 变体（分钟级）；Task 23 含 81 线 × ~90 截面投影 + LOHO 5 折 × c 网格 × 32 相位 × 5 h 的合成谱组 + bootstrap 2000（十分钟级，可分块）。无训练大数据。

## 7. 最低测试（`tests/test_phase2_7_lib.py`）

1. **区间指派**：r ∈ {0.75, 1.25, 1.75, 2.25, 2.75, 3.25} 边界包含性；r=1.5 → OUT；r=9.32 → OUT；**不存在 tie 分支**（负向断言：实现中无 argmin-tie 逻辑）；
2. **两层分布**：q_h 四类和为 1、分母 = 全部 peak-valid；conditional 分母 = family 内；C_family 计算正确；<0.70 → INSUFFICIENT_FAMILY_COVERAGE **优先于 NO_DOMINANT**（构造 coverage 合格但 TV 弱的 case → NO_DOMINANT）；
3. **TV permutation p**：固定 seed 复现 $\bar q_{0,h}$、$T_{\rm obs}$、p（公式 v2.1 唯一实现）；
3. **互斥标签**：构造 (p_(1),p_(2)) 组合验证 MIXED 先于 DOMINANT 判定（0.55/0.45 → MIXED 非 DOMINANT）；前提不满足 → NO_DOMINANT；
4. **shuffled-h null**：单位数 120/15/10 不变、固定 seed 复现、TV ∈ [0,1]、四分类含 OUT；
5. **H_DEPENDENT**：logistic 斜率符号与置换 p 复现；构造单调数据 → YES；
6. **消融 Δ**：fold-paired（同一 split 下 M_full − M_{-h}）、契约校验、Route P/T 分组正确；
7. **包络投影**：Hann 投影在 k=1/(mh) 处直接求值（无 nearest-bin）；Parseval 抽查；可测性表逐格等于 §2 预注册表（h=4/m3=low、h=6/m2=low、h=8/m2=unavailable、h=10/m1=low）；
8. **理论 sanity（仅解析）**：理想 infinite comb 的解析函数在 λ>h 处功率为 0——**该断言只作用于解析 comb 函数单元测试**；对 finite-array 模拟**不得**作此断言（负向断言：模拟在 λ>h 处允许非零功率）；
9. **同管道约束**：模拟谱走与实测完全相同的 bin/peak/validity/m-指派实现（单一来源，不允许 23 重写第二套峰检测）；**2D 场经 `p25.radial_spectrum` 全管线**（负向断言：无 1D n_modes 第二实现）；
10. **phase 边缘化**：constant offsets ⊂ [0,h)、period2 offsets ⊂ [0,2h)（负向断言：period2 不得只用 [0,h)）；16/32/64 grid sensitivity 输出存在；
11. **五分类**：G27-3 的 q 为五类（INVALID 进分母），与 Task 21 四分类共存不冲突；
12. **verdict 顺序**：构造 q 表覆盖 MODEL_INADEQUATE 先判 / NOT_SUPPORTED / SUPPORTED / PARTIAL / exact-match 封顶全分支（互斥）；
13. **LOHO**：c* 只用训练 h 选择；held-out 预测聚合；c=0 路径 = constant；bootstrap 只重采 DOE units、phase grid 固定（负向断言：bootstrap 不重抽 phase）；
14. **d_i guard**：单样本条件 $d_i=q_{P2,i}(c_i)-q_{C,i}(c_i)$；双样本条件（54/156）用观测分布加权期望；>1/3 反向 → 封顶 PARTIAL；
15. **总体口径**：Task 23 primary = estimable ∧ ≠reject（81），estimable-reject 3 条被排除（负向断言）；
16. **语言边界负面断言**：全部输出文件 grep 不到 "harmonic"。

## 8. 输出树

```text
outputs/phase2_7/
  peak_selection/
    peak_selection_m.csv            逐样本 m/OUT 指派 + q_h/conditional 两层分布 + 按 h/session
    family_coverage.csv             C_family 总体与逐 h
    shuffled_h_null_tv.csv          四分类 TV 的 null 分布
    gsl27_2_evaluation.json
  hatch_ablation/
    hatch_ablation_cv.csv           M_{-h}/M_h/M_full × 4 target × 4 CV
    gsl27_1_evaluation.json
  envelope/
    single_track_envelope.csv       入组线 × 候选 k 读数 + 三级可测性标记
    envelope_selection_compare.csv  13 条件 measurement→measurement 对照（3A）
    forward_model_simulation.csv    LOHO held-out 预测 q_M,h（constant/period2）
    bootstrap_delta_tv.csv          ΔTV bootstrap 分布
    gsl27_3_evaluation.json
  summary/
    phase2_7_gate_eval.md
```

## 9. 执行顺序

1. Phase 2.6 封账（**已完成**，`b77ec38`）；2. 本说明 v2 评审冻结；3. 落地执行细则 + `_lib2_7`/脚本 21–23 + 测试 → 预冻结 commit；4. `21` → `22` → `23` 逐 Task commit；5. `phase2_7_gate_eval.md` 终判 → commit。预计一个工作日内完成全部 formal。
