# Phase 2.7 gate 评估（r2，2.7r2 封账版）

> 状态：**formal 完成（2026-09-04，2.7r1；2026-09-05，2.7r2）**。rev1 判定仅 G27-3 从不可接受的 MODEL_INADEQUATE 修正为 2.7r1 修复后重新计算的 MODEL_INADEQUATE（实现修复后仍然成立），G27-1/G27-2 维持。
> **2.7r2（2026-09-05，外审终审）**：G27-1/G27-2 封账接受（CLOSED）；G27-3 审出统计契约错误——LOHO 评价统计量是 macro mean 而主统计量/bootstrap 用 weighted TV，主 ΔTV 与 bootstrap CI/p 不是同一个统计量——**已按外审 6 项修复并全部重算（本文件 §2.3 = r2 数字）**。r2 后 `TV_w(C)=0.615 > 0.30` 且 `TV_w(P2)=0.529 > 0.30` 同时成立，外审接受的 `G27-3 = MODEL_INADEQUATE` 正式封账。
> 2.7r1 修复登记：① Task 22 重复行 bug（in-box 行追加到 primary rows 导致 10 折）→ 分离列表 + 去重断言；② Task 21 H_DEPENDENT 置换未重算 class/family → 每次 permute 重做 `assign_class(λ/h^(b))` 并重定义 family；③ Task 23 `field_class` 缺 h 参数（所有模拟落 OUT）→ 加 h 参数 + 五类 family rate 0%→48%；④ Task 23 其余六项（81-line/32-phase/DOE-unit bootstrap/per-replicate LOHO/d_i guard/诊断输出）全部落实。
> **2.7r2 修复登记（外审 6 项，均不改 Gate/阈值/科学问题）**：① weighted LOHO（评价统计量统一为 TV_w=Σ(n_h/N)TV_h；c\* 选择统计量不变）；② bootstrap unit = (session_id, base_condition_group)，h×session 层内重采（r1 仅 session）；③ 3A d_i 改 own-envelope measurement→measurement（自家 profile 合成 q_C/q_P2，c_global=LOHO c\* 众数；r1 借用 81 线 population q_M）；④ profile 提取用 plateau membership FLAGS（bridged shallow 不再回入 mean profile）+ 视场外横向位置 0 深度净化（上收 `src/data.build_line_profile_library`，2.8B 同源）；⑤ 16/32/64 相位敏感性正式执行（final arms；嵌套网格 stride 聚合）；⑥ formal-contract 测试 7 项（weighted-vs-macro、train-only c\*、DOE unit/strata、own-envelope、窄 kernel→constant array 恢复 m=1 正控制）；stale `forward_model_simulation.csv`（r0 遗留）删除。
> 运行环境：`.venv`（sklearn 1.7.2）。

## 1. Gate 结果（2.7r1 正式）

| Gate | 判定 | 关键数值 | 证据文件 |
|---|---|---|---|
| G27-1（hatch unique contribution） | **SUPPORTED** | Route T 双 target src **paired median ΔR²_h：A2 0.651 / 角熵 0.645**（≥0.05，**5/5 折正**）；proc ΔR²_h [0.734, 0.739]（cap 不触发）；Route P paired Δ：ILR 0.181 / p_8_16 0.350（远小于 Route T） | `summary/gsl27_1_evaluation.json` |
| G27-2（m 分解 + periodicity） | **DOMINANT_m=1 + H_DEPENDENT=NO** | C_family 0.904；TV_w 0.297、p=0.0001；conditional P(m) = {m1 0.628, m2 0.340, m3 0.032}；**H_DEPENDENT=NO**（logistic slope −0.332，置换 p=0.4103——置换重算 class/family 后不再显著） | `summary/gsl27_2_evaluation.json` |
| G27-3（envelope × array model） | **MODEL_INADEQUATE（2.7r2 重算后封账）** | TV_w(constant) **0.6151** / TV_w(period2/LOHO) **0.5290**（双双 >0.30；r2 修正 weighted 统计后）；ΔTV=0.0861 方向为正（DOE-unit bootstrap CI 下界 0.0188>0，p=0.0050）；**相位敏感性 16/32/64：ΔTV 0.111/0.116/0.118（稳定）**；own-envelope d_i：n=3 条件中 **2 个方向反对** period-2 population model（直接证据警告） | `summary/gsl27_3_evaluation.json`、`envelope/forward_model_diagnostic.csv`、`envelope/phase_grid_sensitivity.csv` |
| G27-4（direction alignment） | **NOT_APPLICABLE**（承接 Phase 2.6 G-SL4） | — | — |

## 2. 科学发现（2.7r1 正式）

### 2.1 Route T hatch 主导（G27-1 SUPPORTED）

paired median ΔR²_h 在 Route T 双 target 上为 0.651/0.645——**去掉 hatch 后 A2/角熵的预测能力几乎归零**（M_{-h} 的 R² 接近 0 或负值）。proc-GKF 同向 0.734/0.739。这确证：**在当前 Ridge + grouped-CV 消融下，h 对 8–16 µm 方向组织指标既近乎必要，也接近充分**——这是预测消融证据，不是单因素因果证明。

Route P 对比（外审措辞修正）：**完整 ILR composition 是 multi-factor**（paired ΔQ²_h 0.181，去 h 后 M_{-h} 仍保留 0.132/0.073），**但单带 p_8_16 本身对 hatch 很敏感**（ΔR²_h = 0.350）——不应笼统表述为"Route P 对 hatch 弱"；准确的说法是"完整尺度组成由多因素共同调节，而 8–16 µm 单带能量份额对 hatch 敏感"。

### 2.2 m 分解（G27-2：DOMINANT_m=1，H_DEPENDENT=NO）

conditional P(m) = {m1 62.8%, m2 34.0%, m3 3.2%}，C_family = 0.904 → `DOMINANT_m=1`。

**H_DEPENDENT 从 YES 变为 NO**：修正置换逻辑（每次 permute 后重算 class/family）后 p 从 0.0005 升至 0.4103。原因：旧置换固定了 family 成员和 is_m2 标签（它们由 λ/h 定义），permute h 只改变分母但不重新映射样本到 family——人为构造了不可能在真实数据下出现的对应关系。修正后 m=2 份额的 h 趋势不再显著于 block permutation 的 null 分布。

这仍然是一个**descriptive observation**（逐 h 描述表确认 m=2 份额从 h=4 的 0.55 递减到 h=10 的 0.25），但不足以作为正式推断（logistic slope 在置换 null 下不显著）。固定措辞（外审定稿）：**"存在 descriptive h-trend，但尚无正式统计证据支持 h-dependent family transition。"** h=2 的表述同样受限：只能说"peak-valid 样本全部 OUT / strong-overlap-compatible"，"槽完全熔并的几何必然"属机制解释，不是数据陈述。Phase 3 应重新设计 trend test。

### 2.3 Peak selection 机制（G27-3：MODEL_INADEQUATE，2.7r2 重算封账）

2.7r2 修正统计契约后（weighted LOHO / DOE-unit bootstrap / own-envelope d_i / profile 提取修正），正式数字：

- **TV_w(constant) = 0.6151，TV_w(period2/LOHO) = 0.5290**——同一 weighted 统计下双双大幅超过 0.30 的 MODEL_INADEQUATE 门槛（外审预设的接受条件成立，判定封账）；
- ΔTV = **0.0861** 方向为正（DOE-unit bootstrap CI 下界 0.0188 > 0，p = 0.0050）——period-2 组织有正向信号但 TV 仍远离 0.20 的"重现"门槛；
- **相位敏感性（16/32/64）：ΔTV = 0.111 / 0.116 / 0.118**——相位边缘化不敏感，结论对 phase-grid 稳定；
- **own-envelope d_i（直接证据）**：n=3 个可评估 exact-match 条件中 **2 个方向反对** period-2 population model（66.7% 反向；r1 的 population-borrowing 版本为 5/7）。n<8 未触发 frozen guard，但科学上与外审警告一致：**population 级的 period-2 改善没有得到 exact-match 直接证据的支持**——ΔTV 的正向信号主要是 population 分布层面，不是逐条件测量层面。

数值较 r1（0.4246/0.3547）整体上移的原因：r1 的 P2 臂是 macro mean（低估）、profile 提取修正改变了 kernel 库（stable_flags + 0 深度净化）、weighted 统计下 constant 臂本身也更如实。方向性结论（MODEL_INADEQUATE）不变，但现在是同一统计量上的干净判定。

可能原因（候选列表，非确立）：
- 单轨 FOV（17.8 µm）限制了横向频谱的覆盖——h ≥ 8 时 m=2 不可测（λ=16/20 > 14.9）；
- 1D 截面模型压缩了 2D 谱结构（真实 ROI 是 2D DCT，含沿线变化）；
- 线性叠加未含材料非线性（h < W50 时的熔并、重铸、氧化动力学）；
- 径向 bin 的 geomspace 离散化（24 bin / 2.4 倍频程）导致 λ_peak 被量化到 bin 中心——对 h=8/10 的 m=1 分派（λ_geo 7.54/9.45 vs h=8/10 → r=0.94/0.95）引入了系统性偏差。

**MODEL_INADEQUATE 的登记含义**：当前线性阵列模型族不足以解释 observed selection；邻轨非线性相互作用是候选解释之一，但尚未被独立识别。Phase 3 需要先扩展观测（方向 provenance、2D 谱测量、repeatability matrix）再回到 this question。

## 3. 论文主线（2.7r1 后定稿措辞）

> **工艺对氧化锆激光加工形貌的控制具有分层结构：hatch spacing 对方向组织具有强而近乎充分的预测控制（ΔR²_h 0.651/0.645，proc 同向），而尺度组成仍受多工艺因素共同调节（ΔR²_h 0.181/0.350；完整组成多因素，p_8_16 单带对 hatch 敏感）；简单的线性单轨阵列模型尚未充分重现观测峰选择，提示模型中仍缺失重要结构；邻轨非线性相互作用是候选解释之一，但尚未被独立识别。**

## 4. Phase 3 方向（待外审确认）

**Route T（hatch-dominated structured model）**：h-only spline/GAM + physical interaction terms（如 overlap descriptor $O(h) = \int g(x)g(x-h)\,dx / \int g^2\,dx$）；确认 h 对 A2/角熵的充分性。

**Route P（multi-factor compositional model）**：ILR/Aitchison composition 为主，raw A / hybrid C 输入集；检验 τ/f/h/N/v 联合分配各尺度能量。

**机制桥**：不再用 W50 单值，改用 **profile-overlap descriptor family**（$O(h)$、spectral overlap 等），作为候选物理描述符——在 grouped-CV 中稳定提升后再升级机制地位。

**可观测性扩展**：补方向 provenance（确认 period-2 的物理来源——弓字形 vs 材料交替）；扩展单轨 FOV 或改用 2D 谱测量。
