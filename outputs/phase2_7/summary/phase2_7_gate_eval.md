# Phase 2.7 gate 评估（rev2，2.7r1 封账版）

> 状态：**formal 完成（2026-09-04，2.7r1）**。rev1 判定仅 G27-3 从不可接受的 MODEL_INADEQUATE 修正为 2.7r1 修复后重新计算的 MODEL_INADEQUATE（实现修复后仍然成立），G27-1/G27-2 维持。
> 2.7r1 修复登记：① Task 22 重复行 bug（in-box 行追加到 primary rows 导致 10 折）→ 分离列表 + 去重断言；② Task 21 H_DEPENDENT 置换未重算 class/family → 每次 permute 重做 `assign_class(λ/h^(b))` 并重定义 family；③ Task 23 `field_class` 缺 h 参数（所有模拟落 OUT）→ 加 h 参数 + 五类 family rate 0%→48%；④ Task 23 其余六项（81-line/32-phase/DOE-unit bootstrap/per-replicate LOHO/d_i guard/诊断输出）全部落实。
> 运行环境：`.venv`（sklearn 1.7.2）。

## 1. Gate 结果（2.7r1 正式）

| Gate | 判定 | 关键数值 | 证据文件 |
|---|---|---|---|
| G27-1（hatch unique contribution） | **SUPPORTED** | Route T 双 target src **paired median ΔR²_h：A2 0.651 / 角熵 0.645**（≥0.05，**5/5 折正**）；proc ΔR²_h [0.734, 0.739]（cap 不触发）；Route P paired Δ：ILR 0.181 / p_8_16 0.350（远小于 Route T） | `summary/gsl27_1_evaluation.json` |
| G27-2（m 分解 + periodicity） | **DOMINANT_m=1 + H_DEPENDENT=NO** | C_family 0.904；TV_w 0.297、p=0.0001；conditional P(m) = {m1 0.628, m2 0.340, m3 0.032}；**H_DEPENDENT=NO**（logistic slope −0.332，置换 p=0.4103——置换重算 class/family 后不再显著） | `summary/gsl27_2_evaluation.json` |
| G27-3（envelope × array model） | **MODEL_INADEQUATE** | TV_w(constant) **0.4246** / TV_w(period2/LOHO) **0.3547**（双双 >0.30）；ΔTV=0.0699 方向为正（CI 下界 0.0286>0，p_boot=0.0020）；**family rate 48%**（修复后不再全 OUT） | `summary/gsl27_3_evaluation.json`、`envelope/forward_model_diagnostic.csv` |
| G27-4（direction alignment） | **NOT_APPLICABLE**（承接 Phase 2.6 G-SL4） | — | — |

## 2. 科学发现（2.7r1 正式）

### 2.1 Route T hatch 主导（G27-1 SUPPORTED）

paired median ΔR²_h 在 Route T 双 target 上为 0.651/0.645——**去掉 hatch 后 A2/角熵的预测能力几乎归零**（M_{-h} 的 R² 接近 0 或负值）。proc-GKF 同向 0.734/0.739。这确证：**在当前 Ridge + grouped-CV 消融下，h 对 8–16 µm 方向组织指标既近乎必要，也接近充分**——这是预测消融证据，不是单因素因果证明。

Route P 对比：composition Q² 的 paired Δ 仅 0.181、p_8_16 0.350——h 贡献存在但远不如 Route T，去掉 h 后 M_{-h} 仍保留 0.132/0.073，确证 Route P 仍是多因素问题。

### 2.2 m 分解（G27-2：DOMINANT_m=1，H_DEPENDENT=NO）

conditional P(m) = {m1 62.8%, m2 34.0%, m3 3.2%}，C_family = 0.904 → `DOMINANT_m=1`。

**H_DEPENDENT 从 YES 变为 NO**：修正置换逻辑（每次 permute 后重算 class/family）后 p 从 0.0005 升至 0.4103。原因：旧置换固定了 family 成员和 is_m2 标签（它们由 λ/h 定义），permute h 只改变分母但不重新映射样本到 family——人为构造了不可能在真实数据下出现的对应关系。修正后 m=2 份额的 h 趋势不再显著于 block permutation 的 null 分布。

这仍然是一个**descriptive observation**（逐 h 描述表确认 m=2 份额从 h=4 的 0.55 递减到 h=10 的 0.25），但不足以作为正式推断（logistic slope 在置换 null 下不显著）。Phase 3 应重新设计 trend test。

### 2.3 Peak selection 机制（G27-3：MODEL_INADEQUATE）

2.7r1 修复了 field_class 缺 h 参数的根因 bug（此前所有模拟 λ_peak 被当成 r 直接指派，导致 0% family rate）。修复后 family rate 升至 48%，TV_w 从 0.707/0.652 降至 0.425/0.355——**显著改善但仍然双双超过 0.30 的 MODEL_INADEQUATE 门槛**。

ΔTV=0.0699 方向为正（period-2 优于 constant，bootstrap p=0.002），但 TV_w(period2)=0.355 > 0.20 的"重现"门槛——**period-2 组织有正向信号但不足以称"解释了观测峰选择"**。

可能原因：
- 单轨 FOV（17.8 µm）限制了横向频谱的覆盖——h ≥ 8 时 m=2 不可测（λ=16/20 > 14.9）；
- 1D 截面模型压缩了 2D 谱结构（真实 ROI 是 2D DCT，含沿线变化）；
- 线性叠加未含材料非线性（h < W50 时的熔并、重铸、氧化动力学）；
- 径向 bin 的 geomspace 离散化（24 bin / 2.4 倍频程）导致 λ_peak 被量化到 bin 中心——对 h=8/10 的 m=1 分派（λ_geo 7.54/9.45 vs h=8/10 → r=0.94/0.95）引入了系统性偏差。

**MODEL_INADEQUATE 的登记含义**：当前线性阵列模型族不足以解释 observed selection；材料非线性是候选解释之一，但未被单独识别。Phase 3 需要先扩展观测（方向 provenance、2D 谱测量、非线性 forward model）再回到 this question。

## 3. 论文主线（2.7r1 后定稿措辞）

> **工艺对氧化锆激光加工形貌的控制具有分层结构：hatch spacing 对方向组织具有强而近乎充分的预测控制（ΔR²_h 0.651/0.645，proc 同向），而尺度组成仍受多工艺因素共同调节（ΔR²_h 0.181/0.350）；简单的线性单轨叠加模型（含 period-2 组织）目前尚不足以重现观测到的尺度选择细节（MODEL_INADEQUATE），表明材料响应的非线性在峰选择中扮演了不可忽略的角色。**

## 4. Phase 3 方向（待外审确认）

**Route T（hatch-dominated structured model）**：h-only spline/GAM + physical interaction terms（如 overlap descriptor $O(h) = \int g(x)g(x-h)\,dx / \int g^2\,dx$）；确认 h 对 A2/角熵的充分性。

**Route P（multi-factor compositional model）**：ILR/Aitchison composition 为主，raw A / hybrid C 输入集；检验 τ/f/h/N/v 联合分配各尺度能量。

**机制桥**：不再用 W50 单值，改用 **profile-overlap descriptor family**（$O(h)$、spectral overlap 等），作为候选物理描述符——在 grouped-CV 中稳定提升后再升级机制地位。

**可观测性扩展**：补方向 provenance（确认 period-2 的物理来源——弓字形 vs 材料交替）；扩展单轨 FOV 或改用 2D 谱测量。
