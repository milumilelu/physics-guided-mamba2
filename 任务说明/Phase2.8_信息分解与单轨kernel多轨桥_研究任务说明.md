# Phase 2.8 研究任务说明：层级信息通道可预测性与单轨 kernel → 多轨桥

> 版本：**v2.1 FREEZE-FIX → FROZEN（2026-09-04）**——v2 获外审"接近可冻结"判断后，按终审 8 处 freeze-fix（F1–F8）修订；外审已预批准"完成这八处后即可冻结、无需再概念层重审"，故 **v2.1 即冻结版**。此后门槛/网格/协议/判定顺序禁改，执行自 WP1 起（§6）。
> 路径：`experiments/phase2_8/`（Task 24 + 25）
> 分工：本文件 = what/why（科学定义、协议、Gate）；`experiments/phase2_8/Phase2.8_落地执行细则.md` = how（文件路径、config 键、QA 断言、运行顺序）。两文件冲突时以本文件为准。
> 前置：Phase 2.6/2.7 封账（2.7r1）；**WP1 结构收敛完成后才允许 Task 24/25 formal**——Phase 2.8 是重构后第一个新 phase，公共实现一律走 `src/`，禁止新增 phase-local `_lib` 重复实现。

---

## 0. v1 → v2 修订登记（外审 blocker 落实表）

| # | v1 问题（外审条款） | v2 处置 | 落点 |
|---|---|---|---|
| B1 | "五个正交信息分量"数学上不成立（φ 与 A/P(λ)/P(θ) 重叠；residual_Sq_um 误归 D） | 改为**层级谱分解**（hierarchical information channels）：H → {D, A, P(λ), O_θ(λ), φ(k_x,k_y)}；全文放弃"正交"表述 | §2.1 |
| B2 | φ(x,y) = residual field 本身与幅度/谱信息重复 | φ 重定义为 **Fourier phase** arg F[R]（空间 realization 信息）；不可标量回归，谱表中以描述性 proxy 呈现 | §2.1/§2.3 |
| B3 | A2/角熵被直接称为完整 P(θ\|λ) | 更名 **O_θ(λ) directional organization descriptors**；完整 P(θ\|λ)（circular/compositional 分析）留待升级 | §2.1 |
| B4 | 示例表混用 in-box 101 与 full-200 数值（0.362/0.663 来自 in-box，0.147/0.181/0.486 来自 full-200） | 示例数值全部改 **TBD**；Task 24 全 target 在统一协议下**完整重算**，禁拼历史数字 | §2.3 |
| B5 | 无统一 population/folds 契约 | primary = 四通道 **common intersection**；outer folds 冻结 artifact 全通道共用；另报 maximal-coverage sensitivity | §2.2 |
| B6 | 只对 P/T 做 hatch ablation；ILR inner-α 按第一坐标选 | 全通道统一 **M_full / M_h / M_-h**；inner selection 改 **target-native scorer**（Aitchison Q² 等）；Phase 2.7 不回写 | §2.2 |
| B7 | 谱域公式多乘一个 S_g | 修正为 **S_z(k) = S_g(k)·\|Ã_array(k)\|²** | §3.3 |
| B8 | L2 的 "max / or clip" 不可冻结 | 冻结**单调饱和族** F_β(s) = D_sat(1−e^(−s/D_sat))；D_sat 网格 + LOHO 选取，无 "or" | §3.2 |
| B9 | L3 = period-2，不是真正邻轨交互 | 拆 **L3a**（legacy alternating，continuity control）与 **L3b**（pairwise cross-term Σg_n + γΣg_n·g_{n+1}） | §3.2 |
| B10 | g(x) 来源未冻结 | primary = **measured g(x) + exact process match**（报告 n）；surrogate 预测 kernel 推迟且需 nested CV | §3.1 |
| B11 | "三级逐步证伪"实为四级 | 改为 **4 tier / 5 model**（L0, L1, L2, L3a, L3b） | §3.2 |
| B12 | G28-B 单调判据过强（0.10 门槛 > 2.7 实测 ΔTV=0.0699）又过松（TV 0.70→0.55 仍极差） | 双轴：**B1 MODEL_CLASS_IMPROVEMENT**（ΔTV ≥ 0.05 + bootstrap CI 下界 > 0）+ **B2 absolute adequacy**（沿用 2.7 的 0.20/0.30 语言）；允许 IMPROVED_BUT_INADEQUATE | §3.6 |
| B13 | O(h) 分母受有效 support 影响 | 归一化 overlap（common support、baseline-corrected）+ 符号约定冻结 | §3.4 |
| B14 | "信息分层"结论先写结果；"信息量"表述越界 | 语言规则：**cross-validated normalized predictive skill**；结论改假设检验式 | §2.4/§2.7 |
| B15 | Task 25 写"走 2.7 `_lib`"与"禁止 phase-local `_lib`"自相矛盾 | 新增 `src/forward_models.py`；Phase 2.8 只 `from src.forward_models import ...` | §4.1 |
| B16 | 结构收敛"六模块"实列七个；thin re-export 有复现风险 | **8 个 canonical 模块** + 六步迁移协议 + **golden scientific regression tests** + frozen 函数版本化 | §4 |
| B17 | 功率 provenance / "160 独立 measurement" / P(m=2\|h) 证据等级三处文档错误 | WP0 文档同步：P_obj canonical 化、200/160/134 计数修正、P(m=2\|h) 降级 descriptive | §1 + WP0 |

**v2.1 freeze-fix 登记表（外审终审 F1–F8）**：

| # | v2 问题（外审终审） | v2.1 处置 | 落点 |
|---|---|---|---|
| F1 | `src_gkf/proc_gkf` 语义写错（历史实为双 GroupKFold：src=`shared_height_source_id`，proc=`cv_process_group`；已核 Task 22 L65–66） | 恢复历史语义；shuffle 型如需要必须新命名 `proc_gss_sensitivity`；fold artifact 增 `role` 列（完整 split 落盘） | §2.2 |
| F2 | Moran's I 不能代表 φ（I ~ RᵀWR 是二阶空间统计量，被功率分布 \|R̂(k)\|² 主导，重新混入 P(λ)/O_θ；25600 节点稠密 W 不可行） | φ 撤出谱表；改独立 **realization diagnostic**（phase-only field + shift-invariant pairwise distance），不回归 | §2.8 |
| F3 | 19 候选 ≠ 可用数（2.7r1 3A 登记 13 个 exact-match conditions）；"预期 ≥15"是预写结果 | `n_candidate_exact_match=19` 与 `n_usable_kernel_groups` 分开；usable 程序化判定并报告，不设预写下限 | §3.1 |
| F4 | holdout unit 必须是 measured kernel identity（同一 g_i(x) 桥接的多个 rows 分开 train/test 会乐观） | `kernel_group = (τ,f,N,v,单线身份)`，同 kernel 全部 rows 同留出；协议更名 **LOGO_kernel**（2.7 LOHO=leave-one-hatch-out，禁止混名） | §3.2 |
| F5 | pooled TV 对 condition-specific 预测太钝（A/B 互换预测不惩罚） | B1 主指标改 out-of-group **TV_cond** = 1 − (1/M)Σq_i(y_i)；pooled TV 降为 **pooled-TV legacy adequacy reference** | §3.5/§3.6 |
| F6 | γ 缺量纲（[γ]=µm⁻¹）；γ<0 可产生负去除深度 | 更名 `gamma_per_um` ∈ [−0.5, 0.5] µm⁻¹；**physical guard**：训练模拟 z < −tol 的候选标 physical-invalid 剔除，禁 post-hoc clip | §3.2 |
| F7 | O_θ"范围相同"≠"统计尺度相同"，joint Euclidean 会被大方差坐标主导 | primary 拆 **A2 / angular_entropy 两行标量 Q²**（与 2.7 continuity 对接）；joint standardized Q²（fold-internal 标准化）仅作 secondary | §2.2/§2.3 |
| F8 | golden regression 不得覆盖冻结产物（异常中断/未跟踪文件会污染封账产物） | rerun 写 `outputs/phase2_8/_regression_scratch/`，**scratch vs frozen** 比对；frozen 永远只读 | §4.4 |

**功率项已关闭**：外审终审批准 v2 的 canonical 化处理（含 f↔E_p 耦合提醒保留），无冲突残留。

---

## 1. Provenance 与文档基线（WP0，先行）

### 1.1 功率：P_obj = 5.3333 W 升级为 canonical 物理输入

| 项 | 决定 |
|---|---|
| P = 5.3333 W | **正式实测量，canonical physical input**——物镜后独立实测平均功率（post-objective independently measured average power），测量物理可信 |
| pulse energy / areal dose | 升级为正式 derived quantities：E_p = P_obj / f；D_E = P_obj·N / (v·h)；canonical 列名 `pulse_energy_uJ`、`areal_dose_J_per_mm2` |
| 历史 `_proxy` 字段 | 保留（不删除、不覆盖），供 Phase 2–2.7 复现链使用；Phase 2.8 起只用 canonical 字段 |
| README 风险项 | 删除"功率 provenance 不可信"，改为**测量元数据登记项** |

原则：**measurement is physically trusted ≠ measurement provenance metadata is perfectly complete**。仪器型号/测量日期若无记录，登记 `instrument metadata unavailable`——元数据不完整不构成把测量值降为 proxy 的理由。

登记落点（WP0/WP1 执行）：

1. `src/provenance.py`：`POWER_REGISTRY` 单一事实源（value 5.3333 W；type post_objective_average；trust user_confirmed；instrument/date unavailable；source_doc `现有数据基础说明_v2.md` §11）；canonical 列由 registry 派生，并带与 `_proxy` 旧列**逐元素恒等**的 parity 断言。
2. `README.md`：风险项替换 + ROI 计数修正 + P(m=2|h) 措辞修正（已随本 v2 提交）。
3. `outputs/phase2/manifest/README.md`：provisional 表述改为"已升级 canonical；`_proxy` 旧列保留；冻结的 phase2 manifest CSV 不改写"。
4. `现有数据基础说明_v2.md` §11：追加登记状态更新块（不改写历史叙述）。

**保留的辨识性提醒（与功率可信度正交，不得删除）**：DOE 中 P 恒定 ⇒ f 与 E_p 完全耦合，现有数据不能区分 repetition-rate effect 与 pulse-energy effect。正式措辞一律为 **frequency / pulse-energy coupled effect**。

### 1.2 计数表述修正

- 矩形主数据 = **200 个 ROI / 实验记录**（120 formal + 60 pass_main + 20 pass_supplement），对应 **160 个唯一 height-source**（`shared_height_source_id`）与 **134 个 `cv_process_group`**。
- 禁用"160 个独立 measurement"类表述：物理测量独立性与统计独立性是两个概念；统计独立性由分组变量单独定义。
- Phase 2.8 grouped-CV 分组变量沿用 `cv_process_group`（primary）；如需更保守的 sensitivity 可用 `shared_height_source_id`（160 簇）并登记。

### 1.3 P(m=2|h) 证据等级修正

m=2 份额逐 h 描述（0 / 0.545 / 0.471 / 0.257 / 0.25）确呈下降趋势，但 block permutation（每次 permute 重算 class/family）后 **h-dependence 不显著（logistic slope −0.332，p = 0.4103，H_DEPENDENT = NO）**。该结论只能以 descriptive observation 出现，不得与 A_obs = 0.904（p = 0.0001）置于同一证据等级。

### 1.4 仍开放的三风险（不因 WP0 关闭）

1. **方向 provenance** 缺失（G-SL4 / G27-4 NOT_APPLICABLE 承接）；
2. **单线 QA 为 AI 辅助标注**——若 2.8B 结果进入论文核心机制证据，需补双人独立盲标；
3. **discovery/confirmation 未分离**——Phase 2.8 保持 discovery-only（§5）。

---

## 2. Task 24 — Phase 2.8A：层级信息通道的可预测性

### 2.1 层级谱分解（hierarchical spectral factorization）

对每个 ROI 的冻结残差形貌场，按信息层级分解：

```text
H(x,y) → { D,  A,  P(λ),  O_θ(λ),  φ(k_x,k_y) }
```

| 通道 | 定义 | target 形态 |
|---|---|---|
| **D** — 宏观深度（material removal） | 标量 `median_depth_um` | 标量 |
| **A** — 形貌幅度（amplitude） | A = Sq(R) = sqrt(mean_Ω R²)，与 manifest `residual_Sq_um` 做逐元素 parity 断言 | 标量 |
| **P(λ)** — 尺度组成（scale allocation） | 五段带能量组分 [p_lt8, p_8_16, p_16_32, p_32_64, p_64_inf] 的 ILR 坐标 z1–z4 | 4 维组分 |
| **O_θ(λ)** — 方向组织描述子（directional organization descriptors） | 8–16 µm 带的 (A2, angular_entropy) | 2 维 |
| **φ(k_x,k_y)** — 空间 realization（specific realization） | φ = arg F[R]（Fourier phase）；评估见 §2.8 realization diagnostic（**不进谱表**） | 场（不可标量化） |

层级语义：**D（去除了多少材料）→ A（剩下来的起伏多大）→ P(λ)（能量怎么分配到尺度）→ O_θ（起伏在方向上组织得多强）→ φ（波峰具体落在哪里）**。相邻层不宣称正交或统计独立——这是层级分解（hierarchical channels / hierarchical spectral factorization），不是正交分解。

两条更名纪律：

1. v1 的 D 行曾包含 `residual_Sq_um`——错误（幅度信息），v2 归 A；
2. A2/角熵是方向分布的**低维 summary statistics**，在构造角度直方图并做 circular/compositional 分析之前，一律称 O_θ(λ)，不称 P(θ|λ)。

### 2.2 统一可预测性协议（apples-to-apples 契约）

对四个 primary 通道（D, A, P_λ, O_θ）完全一致地执行：

1. **Population**：全 200 ROI；primary 拟合样本集 = 全部 primary targets（D、A、P_λ、A2_8_16、angular_entropy_8_16）的 **common intersection**（程序化求交并登记 n；预期 200/200）。另报各 target 的 maximal-coverage sensitivity（本数据集预期与 primary 相同，仍须登记）。
2. **Outer folds（F1：恢复历史语义，与 Phase 2/2.5/2.7 完全一致的双 GroupKFold）**：

   ```text
   src_gkf  = GroupKFold(5), groups = shared_height_source_id   (160 簇)
   proc_gkf = GroupKFold(5), groups = cv_process_group          (134 组)
   ```

   两者均为确定性 GroupKFold（无 shuffle、无 seed）。如需 shuffle 型 variant，必须新命名 **`proc_gss_sensitivity`**（GroupShuffleSplit，seed 冻结），**不得叫 proc_gkf**。folds 各生成**一次**，写入冻结 artifact `outputs/phase2_8/folds/fold_assignments.csv`，列 = `variant, fold, dataset_index, role∈{train,test}`（完整 split 落盘，不再只记 test assignment）；所有通道共用同一份索引；过 `check_gkf_contract`。

   修正理由：v2 误写 src_gkf=`cv_process_group`、proc_gkf=GroupShuffleSplit。若沿用，Q²_2.7 vs Q²_2.8 的差异将混入 split 算法变化，无法归因于 R²→Q² 与 target-native α 两个已登记的协议差。
3. **三模型统一**（每个通道都跑，不再只对 P/T 做 ablation）：
   - M_full：u = [τ, f, h, N, v]
   - M_h：u = [h]
   - M_-h：u = [τ, f, N, v]
4. **模型族**：Ridge（fold-internal StandardScaler）× α 网格 logspace(−3, 3, 13) × inner GKF(5)（训练 unit 内）。
5. **Target-native inner scorer**（`src/cv.py` 新实现；**不回写** Phase 2.7 的 `ridge_alpha_inner_gkf`）：
   - 标量（D、A、A2_8_16、angular_entropy_8_16 四个标量 target）：inner mean MSE；
   - 组分（P_λ 的 ILR z1–z4）：inner **Aitchison Q²** 最大化（等价 ILR 空间 MSE 最小化）；
   - joint secondary（O_θ 二维联合）：inner 多变量 Euclidean MSE，坐标在**每个训练折内**标准化。

   内层 objective 与外层 skill 同族——修复 v1 沿用"ILR 第一坐标选 α"的 objective 错配。
6. **外层 skill 定义**：Q² 约定 = 1 − Σ_test(y−ŷ)² / Σ_test(y−ȳ_train)²（null = train-mean，与 `q2_aitchison_ilr` 同约定）。标量目标同时附 sklearn `r2_score` 对照列（仅为与 2.7 历史数字的连续性对照，不作 primary）；P_λ 报 Aitchison Q²；**O_θ primary 拆为 A2 与 angular_entropy 两个标量行**（与 Phase 2.7 continuity 直接对接）；joint standardized Q²（fold-internal 标准化，两维等权、无泄漏）仅作 channel summary 的 secondary。
7. **Dummy baseline**：train-mean（组分目标 = closed geometric-mean composition）逐折预测；断言 dummy skill ≈ 0（数值容差）。
8. **Sensitivity**：(a) in-box 101（沿用 phase2_6 冻结 box；子集内 `src_gkf_inbox` = GroupKFold(5) on `shared_height_source_id`，同 Task 22 语义，重新生成 + 契约校验）；(b) raw/repaired 输入对（复用 Phase 2.5 raw/repaired 数据路径，config 固定并断言 200 ROI 对齐；若 raw 冻结数据缺失则记 N/A 并登记原因，不静默跳过）。
9. **Coverage**：每 target 报 n、n_folds、逐折 n_train/n_test、NaN 明细。
10. **禁令**：不得从 Phase 2/2.5/2.7 的历史汇总中拼任何数字进谱表——每个数都出自本协议的本次运行，可溯源到逐折 CSV。

### 2.3 输出：Predictability Spectrum（示例值一律 TBD）

```text
target            full skill    h-only    no-h skill   Δ_h     metric
D                 TBD           TBD       TBD          TBD     Q² (scalar)
A                 TBD           TBD       TBD          TBD     Q² (scalar)
P_λ               TBD           TBD       TBD          TBD     Q² (Aitchison, ILR z1–z4)
O_θ : A2          TBD           TBD       TBD          TBD     Q² (scalar)
O_θ : entropy     TBD           TBD       TBD          TBD     Q² (scalar)
(O_θ joint)       TBD           TBD       TBD          TBD     Q² (standardized 2-D; secondary summary)
```

- variants：src_gkf 为 primary 列；proc_gkf / in-box / raw-repaired 为一致性列。
- Δ_h = fold-paired [skill(M_full) − skill(M_-h)] 的逐折中位；h-only 列回答"h 单独能解释多少"，与 Δ_h（"h 的 unique contribution"）是两个不同问题，都报。
- **φ 不设谱表行（F2）**：Moran's I proxy 被外审驳回——I ~ RᵀWR 是二阶空间统计量，近似平移不变的邻接算子下由功率分布 |R̂(k)|² 主导，会重新混入 P(λ)/O_θ，不能代表 specific realization；且 160×160=25600 像素节点的稠密 W 矩阵不可行。φ 的处理移入 §2.8 realization diagnostic，**不回归**。
- 输出物：`outputs/phase2_8/predictability_spectrum.csv` + `predictability_spectrum.png` + `summary/gsl28_a_evaluation.json`（含协议契约断言结果与 coverage）。

### 2.4 解释边界（图注与正文强制）

- 纵轴语义 = **cross-validated normalized predictive skill**（1 − prediction loss / null loss），**不是**信息量、不是 Shannon mutual information；不同 target 空间与 loss 之间不传递大小关系。
- 允许的措辞：同一 target、同一 metric 内比较模型间差异与 Δ_h；跨通道排序只能作为描述性观察并注明 metric 不可比性。
- "Predictability Spectrum" 名称保留。

### 2.5 新描述符晋升规则（2.8 内只做描述性评估）

O(h) 与谱域 overlap 峰选择（§3.3–3.4）是**候选机制描述符**。关键约束：O(h) 依赖被测 g(x)，而 exact-match 只有 §3.1 的小子集——**2.8 内不存在可支撑 full-200 grouped-CV 的 O(h) 预测子**。因此：

- 2.8 内：仅在 exact-match 子集上做描述性关联（如 Spearman(O(h), A2 / m class)），不做 CV 提升 claims；
- 晋升"机制描述符"推迟到 Phase 3 confirmation 数据到位后：届时新条件下实测 g(x)，grouped-CV 中 Δ ≥ 0.05 且折方向一致才升级。

### 2.6 实施 + G28-A

- 脚本：`experiments/phase2_8/24_information_decomposition.py`；全部模型/管线调用 `src/`（`src.cv` / `src.composition` / `src.spectrum` / `src.data` / `src.provenance`），禁止 phase-local 重复实现。
- D / A 的 src/proc CV 若 Phase 2 没有按本协议跑过，本次补齐——这正是统一重算的一部分。
- **G28-A = VALID / INVALID（QA completion gate；描述性输出，无 SUPPORTED/NOT）**。VALID 须同时满足：

  1. primary targets 同一 population（common intersection，n 登记；realization diagnostic 不在 primary 内）；
  2. outer folds 完全一致（同一冻结 artifact，哈希登记）；
  3. preprocessing 全部 fold-internal；
  4. α 用 target-native inner scorer；
  5. dummy baseline 正常（≈ 0）；
  6. M_full / M_h / M_-h 三模型全通道完成；
  7. raw/repaired sensitivity 完成（或 N/A 理由登记）；
  8. 每 target 报告 coverage；
  9. 无任何历史分数拼接（审计：谱表每个数可溯源到本次运行的逐折 CSV）。

### 2.7 语言规则（防 narrative lock-in）

- 任务目标句：**检验不同形貌信息通道是否存在稳定的 predictive-skill hierarchy**——假设检验式，不预写结论。
- v1 的"这张图把结论统一为：工艺对形貌的控制力具有明确的信息分层"（运行前写好结果）废止。
- 若计算后确出现稳定排序（如 S_θ > S_P > S_A），才允许在 gate_eval 的"发现"节写排序，并附逐折方向一致性证据。

### 2.8 Realization diagnostic（φ(k_x,k_y)：不回归、不进谱表、不入 G28-A）

φ = arg F[R] 不可标量回归，v2.1 起 φ 的评估独立于 Predictability Spectrum：

1. **Phase-only field**：Q_i(k) = R̂_i(k) / (|R̂_i(k)| + ε)，q_i(x,y) = F⁻¹[Q_i]——保留相位、抹平幅度；
2. **Shift-invariant pairwise distance**（消除 ROI 小幅平移的全局 phase ramp；Δ 搜索范围冻结入 config，如 |Δ| ≤ 4 px）：

   ```text
   s_φ(i,j) = max_{Δx,Δy} corr(q_i, q_j(·+Δ)),   d_φ(i,j) = 1 − s_φ(i,j)
   ```

3. **只问三个描述性问题**：
   - process-near pairs 的 d_φ 是否小于 ordinary pairs？
   - exact repeat（数据集内 49/50 对）的 d_φ 处于什么 percentile？
   - 相同/相近工艺是否具有可复现的 spatial realization？

这直接对应"工艺有没有决定具体波峰长在哪里"，且不与 P(λ)/O_θ 的信息混合。输出 `outputs/phase2_8/realization_diagnostic.csv` + JSON；实现入 `src/spectrum.py`（`phase_only_field` / `shift_invariant_phase_distance`）。

---

## 3. Task 25 — Phase 2.8B：measured kernel + array geometry + minimal interaction

**核心问题**：不再用 W50 压缩单轨信息，保留完整截面 profile g(x)，与 hatch spacing h 一起——

```text
measured single-track kernel + array geometry + minimal nonlinear interaction
究竟能解释多少矩形加工中的尺度选择？
```

### 3.1 g(x) 来源冻结（本 task 最重要的事前决定）

核心问题：对某个 rectangle condition，**用哪一条 measured g(x)**？

- **Primary mechanism bridge**：只用 **measured single-line g(x) + exact process match**——(τ, f, N, v) 与 rectangle condition 精确匹配；h 来自 rectangle 设计表（单线 DOE 无 h，h 只进入阵列几何）。数据基础 = `outputs/phase2_6/scale_bridge/direct_bridge_exact_match.csv`。
- **计数语义（F3）**：`n_candidate_exact_match = 19`（候选条件数，多数 n=1）**≠** 可用数；经 estimable / QA / suitable profile / valid observation 过滤后，`n_usable_kernel_groups` 由程序判定并在 gate_eval 报告——2.7r1 的 3A 登记为 **13 个 exact-match conditions（own envelope）**，可用数预期在该量级，但**不预写任何下限**（v2 的"预期仍 ≥15"删除，避免 formal 跑出 13 时出现假性的"未达协议预期"）。
- **primary 中禁止使用预测出来的 g(x)**。
- **Secondary（推迟，不在本次 formal）**：扩大到 in-box 需先建 (τ,f,N,v) → g(x) 的 functional surrogate 且必须 nested CV——"predicted kernel → predicted rectangle" 两层模型误差混合会使机制解释失效。登记为 Phase 3 可选项。
- g(x) 提取路径与 Task 23 完全一致（冻结平面/轴框/稳定区 + 逐线 stable-region mean profile），实现走 `src/geometry.py`；population 判据同 Task 23（estimable ∧ qa ≠ reject）。
- **符号约定**：removal-depth positive（g ≥ 0 = 去除深度）。QA 断言 median(g) > 0；若存储符号相反，做一次冻结符号翻转并登记。

### 3.2 模型族：4 tier / 5 model（L3 拆分）

| tier | model | 定义 | 角色 |
|---|---|---|---|
| L0 | kernel-only | 单轨谱 S_g(k) 自身的 λ_peak → r = λ_g/h → class | 单轨基线（2.6 direct bridge 的模型化） |
| L1 | linear array | z(x) = Σ_n g(x−n·h−φ)，a_n = 1 | 理想 h-阵列（复现 G27-3 的 constant 臂） |
| L2 | pointwise saturation | s(x) = Σ_n g(x−n·h−φ)；z_L2 = F_β(s)，F_β(s) = D_sat·(1−e^(−s/D_sat)) | 逐点非线性饱和（相邻槽熔并的最低阶近似） |
| L3a | legacy alternating | a_n = 1 + c·(−1)^n | **Phase 2.7 continuity control**（period-2 家族） |
| L3b | pairwise interaction | z = Σ_n g_n + γ·Σ_n g_n·g_{n+1}，g_n(x) = g(x−n·h−φ) | **首次显式加入邻轨 overlap cross-term**（最低阶非线性相邻轨相互作用项） |

冻结细则：

- **留出单位（F4）**：`kernel_group = (τ, f, N, v, 单线身份)`——同一 measured kernel 桥接的全部 rectangle rows（可跨 h）**必须同组同留出**，否则 g_i(x) 已在训练集出现、测试集又拿同一 g_i(x) 预测另一 h，会系统性乐观。协议名 **leave-one-kernel-condition-out（LOGO_kernel）**；Phase 2.7 的 LOHO 是 leave-one-**hatch-level**-out，两者不同名不同义，禁止混用。
- **参数选择**：全局参数（D_sat\*、c\*、γ\*）= **LOGO_kernel** median TV_cond 最小化（tie 取小值），只用训练 kernel group；评价 group 的观测 class/spectrum 不得进入选择。per-h 参数作 sensitivity。
- **L2 无 "or"**：饱和族唯一；D_sat 网格 = geomspace(1, 64, 13) µm。F_β 对 s ≥ 0 单调递增、饱和于 D_sat——与 removal-positive 约定相容。
- **L3a** 的 c 网格沿用 Phase 2.7 `g27_3.c_grid`（0.0–0.9 步 0.1）——模型族与 G27-3 同族可比（选择协议为本 task 的 LOGO_kernel，与 2.7 的 LOHO 不同 population，不作数值对表）。它**不是**新的邻轨交互主张，只是 continuity control；若 L3b 也解释不了，应升级模型族，而不是把 period-2 包装成邻轨作用。
- **L3b（F6：量纲 + 物理 guard）**：交叉项量纲分析——[g_n·g_{n+1}] = µm²，故 **γ 的量纲为 µm⁻¹**，参数更名 **`gamma_per_um`**，网格 = linspace(−0.5, 0.5, 21) µm⁻¹（对称，允许符号发现）。**physical guard**：若某 γ 候选在训练 kernel group 的模拟中出现 z(x) < −tol（tol = 1e-9 µm，冻结），该候选标记 `physical_invalid` 并**排除出选择**（负去除深度无物理意义）；**禁止 post-hoc clip**——clip 本身是又一个未登记的非线性。被排除候选逐格登记。
- 相位：φ 网格 32 点（同 Task 23 final）；模拟场 → `field_class`（同管线：radial_spectrum 24 bin → lambda_peak_4_32 → assign_class 五类）。
- **全部 forward model 走 `src/forward_models.py`**（迁移自 2.7 `_lib`：`synth_field` / `field_class` / array construction / phase marginalization / observation operator wrappers）。

### 3.3 谱域公式（修正 v1 错误）

线性叠加阵列的谱（S_g(k) = |g̃(k)|²）：

```text
S_z(k) = S_g(k) · |Ã_array(k)|²,   Ã_array(k) = Σ_n e^(−iknh)
```

**不是** S_g²·|Ã|²（v1 多乘一个 S_g，冻结前必须修——本条已修）。

overlap 峰选择描述符：k\* = argmax_k S_g(k)·|Ã_array(k; h)|² → λ_pred = 2π/k\* → r_pred = λ_pred/h。

### 3.4 O(h)：归一化 overlap 描述符

```text
O(h) = ∫_{Ω_h} g̃(x)·g̃(x−h) dx / sqrt( ∫_{Ω_h} g̃²(x) dx · ∫_{Ω_h} g̃²(x−h) dx )
```

- g̃ = baseline-corrected 单轨 profile；Ω_h = 共同有效 support（边缘 3 px 剔除，与 `profile_suitable` 同族判据）；
- 分母对称归一化，避免 v1 定义受有效 support 影响导致的不对称；
- 值域 [−1, 1]；removal-positive 约定下 O(h) > 0 表示相邻轨去除区重叠。
- O(h) 与 r_pred 都**不进 Gate**——按 §2.5 的描述性/晋升规则处理。

### 3.5 TV 目标与指标

- **Primary（F5：condition-matched TV）**——2.8B 是 **condition-specific prediction**（g_i(x) + h_i → q_i(m)），population-level 的 pooled TV 对它太钝：模型把 condition A 预测成 B、B 预测成 A 时，pooled 类别比例可以完全正确而逐 condition 全错。主指标：

  ```text
  TV_cond = (1/M) Σ_i TV(q_i^pred, e_{y_i}) = 1 − (1/M) Σ_i q_i(y_i)
  ```

  （observation 为 one-hot，故 TV(q_i, e_{y_i}) = 1 − q_i(y_i)——语义即"模型平均给真实 observed class 分配了多少概率"）。q_i^pred 必须是 **out-of-group** 的：参数选择经 LOGO_kernel 排除该 kernel group。
- **Legacy continuity diagnostic（F5）**：**TV_pooled** = TV(mean_cond q_pred, empirical q_obs) 继续报告，0.20/0.30 阈值作为 **pooled-TV legacy adequacy reference**——与 2.7 衔接的参照，不宣称跨 population/跨预测任务的同一 "model adequacy"。
- **Secondary（连续谱 TV）**：24-bin radial energy_fraction 的 TV（模拟场谱 vs ROI 实测谱，逐 condition 平均）——比五分类更敏感，作趋势对照。
- Primary 规模小（候选 19、可用数程序判定，§3.1）是**登记在案的事实**：不声称总体代表性；per-h 分解仅在 n_h ≥ 3 的 h 层报告。

### 3.6 G28-B：双轴判定

**B1 — MODEL_CLASS_IMPROVEMENT（相对改善，主轴）**：对 Lj ∈ {L2, L3a, L3b} vs L1，在 **out-of-group TV_cond** 上：

```text
ΔTV_cond{L1→Lj} = TV_cond(L1) − TV_cond(Lj) ≥ 0.05
且 kernel-group bootstrap（B=2000，重采 kernel group）paired-delta 95% CI 下界 > 0
```

成立时登记 **"model class improvement"**——只说明该模型族更好，**不等于机制成立**。

**多重比较登记（P1，随冻结入档）**：L2−L1 / L3a−L1 / L3b−L1 是**同一 model-family exploration 的三个比较**，其 CI 不解释为相互独立的 confirmatory 95% coverage；同时报告 **98.33% Bonferroni 式 simultaneous CI**（三比较族 overall ≈ 95%）作 sensitivity。Phase 2.8 是 discovery-only，不在此做严格族错误率检验。

**B2 — pooled-TV legacy adequacy reference（连续性诊断轴）**：继续报告 TV_pooled，沿用 Phase 2.7 阈值语言：

```text
TV ≤ 0.20        → strong reproduction（legacy reference）
0.20 < TV ≤ 0.30 → partial（legacy reference）
TV > 0.30        → MODEL_INADEQUATE（legacy reference）
```

**不得把两个不同 population/预测任务下的 0.30 说成同一种 "model adequacy"**——正式措辞一律为 pooled-TV legacy adequacy reference。

两轴组合允许 **IMPROVED_BUT_INADEQUATE**——这正是 Phase 2.7 period-2 的真实状态（ΔTV = 0.0699、CI 下界 0.0286 > 0，但 TV_w = 0.355 > 0.30），预期 L3a 在本数据上的落点与之衔接。L1→L2→L3a/L3b 的单调性作描述性报告。机制语言仍受 G27-3 约束：MODEL_INADEQUATE 不确立材料非线性，只说明当前模型族不足。

输出：`outputs/phase2_8/kernel_bridge_levels.csv` + `summary/gsl28_b_evaluation.json`。

---

## 4. 结构收敛（WP1）：src/ 八个 canonical 模块

Phase 2.8 是重构后第一个新 phase；**先收敛、后 formal**。

### 4.1 模块清单（8 个）

| 模块 | 合并来源（canonical implementation） | 内容 |
|---|---|---|
| `src/data.py` | phase1_5 `load_frozen` + `src/io_cag/io_npz` | 冻结数据加载（H/V/R/manifest） |
| `src/provenance.py` | phase2 派生列函数 + 各 `_lib.load_config/output_dir/log/require` | `POWER_REGISTRY`、E_p/D_E canonical 派生、config/运行标识 |
| `src/cv.py` | phase2 `gkf_splits/gss_splits/check_*`（**src_gkf=shared_height_source_id / proc_gkf=cv_process_group 双 GKF 语义冻结**）+ phase2_6 `make_ridge/ridge_alpha_inner_gkf`（保留为 v1 语义） | grouped CV + 契约校验 + **target-native inner scorer（新）** |
| `src/composition.py` | phase2_5 `five_part_composition/frozen_band_fractions/apply_zero_replacement/ilr_*/aitchison_distance` | 谱组成 + ILR |
| `src/spectrum.py` | phase2_5 `radial_spectrum/spectrum_descriptors/directional_band_metrics` + phase1_5 `dct_lambda_grid` + **新** `phase_only_field/shift_invariant_phase_distance` | 径向/方向谱 + realization diagnostic |
| `src/geometry.py` | phase2_6 `sample_profiles/lateral_positions/axis_frame/assign_class/plateau_stable_run/line_extent/scan_plateau_features/lambda_peak_4_32/in_box_mask/...` + phase2_7 `profile_suitable` | 单线几何 |
| `src/statistics.py` | phase2_5 `sign_matrix/exact_signflip_test/moran_*` + phase2_6/1_5 cluster bootstrap + phase2_7 `tv/tv_perm_p/logistic_slope` | 统计检验 |
| `src/forward_models.py` | phase2_7 `synth_field/field_class/hann_projection/cycles_level` + **新** `pairwise_interaction_field(γ_per_um)/saturate(D_sat)/physical_validity_guard` | 正演模拟 + 峰选择观测算子 |

（逐函数最终归属在迁移 PR 中固定；凡被 frozen phase 使用的函数，语义必须逐位一致。）

### 4.2 迁移协议（每函数六步，不满足不 re-export）

v1 的"旧 `_lib` 全改 thin re-export"被外审否决（复现风险：日后 `src` 一次语义修改会让 HEAD 上重跑旧 Phase 得到不同结果）。严格规则：

1. canonical `src` 实现；
2. old vs canonical **parity test**（固定输入逐元素/逐统计量比较，含 NaN 与边界 case）；
3. rerun 对应 frozen phase 的代表性产物；
4. 核心 JSON 数值 vs golden anchors **exact**（确定性管线）或冻结容差内一致；
5. 通过后才允许旧 `_lib` 改为 `from src.xxx import ...` 的 thin re-export；
6. **frozen canonical 函数此后禁原地改语义**——新语义一律新函数名（`q2_aitchison_v1/v2` 式版本化），杜绝"Phase 2.7 某天悄悄吃到 Phase 3 的新定义"。

### 4.3 Golden scientific regression tests

Phase 2.7 曾出现 duplicate folds / permutation 未重算 family / `field_class` 缺参——都改过科学结果。重构必须测**科学锚点**，不只测"函数能跑"：

| 锚点 | 冻结值（2.7r1） |
|---|---|
| G27-1 ΔR²_h（A2，src_gkf） | 0.6509672473528761 |
| G27-1 ΔR²_h（angular_entropy，src_gkf） | 0.6449781075749867 |
| G27-1 ILR Q² full / −h / Δ（src_gkf） | 0.31305506446560427 / 0.14779357389704195 / 0.18101258577517343 |
| G27-1 ΔR²_h（p_8_16，src_gkf） | 0.34961895113131647 |
| G27-2 C_family / TV_w / p_perm | 0.9038461538461539 / 0.29721160530020074 / 9.999000099990002e-05 |
| G27-2 H_DEPENDENT slope / p | −0.33234665899340665 / 0.4102948525737131 |
| G27-3 TV_w constant / period2 / ΔTV | 0.4245949074074074 / 0.35469112596305574 / 0.06990378144435166 |
| G27-3 bootstrap CI low / p_boot | 0.028577491181657866 / 0.001999000499750125 |

入 `tests/test_golden_anchors.py`；迁移后任一锚点不一致 → fail，直接修复，不带病前进。

### 4.4 每模块一个 commit；顺序与全量回归

迁移顺序：**data → provenance → cv → composition → spectrum → geometry → statistics → forward_models**；每 commit 跑全测试（`.venv` 下 `python -m unittest discover tests`）+ 相关 golden 锚点。全部完成后做 **scratch 重生成回归（F8：冻结产物只读）**——把代表性产物（phase2 manifest、phase2_5 spectral/directional CSV、phase2_7 Task 22/23 summary JSON）以冻结输入**重跑进 `outputs/phase2_8/_regression_scratch/`**，然后 **scratch vs frozen** 做 SHA256/数值清单比对，报告写 `outputs/phase2_8/refactor_regression_report.md`。**禁止"覆盖冻结产物 → 比对 → git checkout 恢复"流程**：中途异常或未被 git 跟踪的文件都可能污染封账产物——scientific regression test 对 frozen artifacts 永远只读。

---

## 5. Confirmation 预设计（不污染 2.8）

**决定：现在开始设计 confirmation interface；Phase 2.8 本身仍 discovery-only**——2.8 的任何模型选择、threshold、descriptor 都不得看到 confirmation outcome。

### 5.1 接口（代码骨架 + 单测；2.8 不调用）

三段式预留：

```text
fit(discovery_manifest) → predict(confirmation_manifest) → evaluate_locked_predictions(...)
```

锁定的预测在 confirmation 测量完成前写盘（JSON schema 含 model version、config hash、feature order）；揭盲只有一次。

### 5.2 第一批新实验：repeatability matrix 优先

当前最大未知是 **Var(H|u) 到底多大**——不是 L2/L3 哪个好。第一批**不做** 20–40 个全唯一 confirmation 条件，而做：

- **6–8 conditions × 3–5 independent repeats**（如 8 × 4 = 32 次加工/测量）；
- condition 覆盖：Route T 高/低、Route P 高/低、高/低 Sq、error hotspot、stripe phenotype、ordinary morphology；
- 直接回答：**低可预测性是随机性，还是 missing variables**——全项目当前最值钱的新实验。

### 5.3 第二批：mechanism confirmation set

2.8B 跑完后冻结 20–30 个新 unique conditions，预注册预测（P_λ、O_θ、λ/h class、L1/L2/L3 相对排序、overlap descriptor 方向效应），一次性揭盲——这才是真正意义上的 mechanism confirmation set。

### 5.4 预算只允许一批时：混合设计

6 anchor conditions × 3 repeats（= 18，估 σ_repeat(u)）+ 14 个新 confirmation conditions（总 32）；**分析时两部分预先分开**。

---

## 6. 执行顺序

1. **WP0**：文档与 provenance 同步（README × 3 处 / manifest README / 数据说明 §11）→ commit
2. **WP1**：src/ 八模块（六步迁移 × 8 + golden tests + 回归报告）→ 每模块一 commit
3. **WP2**：Task 24（2.8A）：folds artifact → `24_information_decomposition.py` → Predictability Spectrum → G28-A → commit
4. **WP3**：Task 25（2.8B）：`25_kernel_bridge.py` → 逐级 TV → G28-B 双轴 → commit
5. **WP4**：`phase2_8_gate_eval.md` + confirmation 接口骨架 + repeatability matrix 设计登记 → commit

Task 24/25 均为分钟级计算；WP1 完成后 formal 可一日内完成。

---

## 7. Gate 汇总

| Gate | 判定 | 性质 |
|---|---|---|
| **G28-A** | VALID / INVALID（9 条件，§2.6） | QA completion gate（描述性，无 SUPPORTED/NOT） |
| **G28-B1** | MODEL_CLASS_IMPROVEMENT 达成/未达成（L2/L3a/L3b vs L1，out-of-group TV_cond） | 相对改善（ΔTV_cond ≥ 0.05 + kernel-group paired bootstrap CI 下界 > 0；三比较为同族探索 + 98.33% Bonferroni sensitivity）；不等于机制成立 |
| **G28-B2** | strong reproduction / partial / MODEL_INADEQUATE（逐级，TV_pooled） | **pooled-TV legacy adequacy reference**（0.20/0.30 承 2.7；不作跨 population/任务 adequacy 宣称） |
| 组合 | 允许 **IMPROVED_BUT_INADEQUATE** | 与 Phase 2.7 period-2 状态衔接 |

**冻结核心（一句话）**：

- 2.8A 研究：**宏观深度 → 幅度 → 尺度组成 → 方向组织 → 空间 realization** 这几个层级信息通道的可预测性如何逐级变化（cross-validated normalized predictive skill，不宣称信息量）。
- 2.8B 研究：**measured single-track kernel + array geometry + minimal nonlinear interaction** 究竟能解释多少矩形加工中的尺度选择。
