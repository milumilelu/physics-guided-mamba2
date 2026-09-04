# Phase 2.8 研究任务说明：信息分解与单轨 kernel → 多轨桥

> 建议路径：`experiments/phase2_8/`（Task 24 + 25）
> 建议状态：`DRAFT_FOR_REVIEW v1`
> 前置条件：Phase 2.6/2.7 封账 + **结构收敛 src/ 重构完成**（Phase 2.8 是重构后第一个新 phase——所有公共实现必须走统一 `src/`，不再允许 phase-local `_lib` 重复）

---

## 0. 为什么是这两个任务

Phase 2/2.5/2.7 的结果可以统一读作：**工艺参数对不同形貌信息分量的控制力不等**。

- Route T（A2/角熵）：hatch spacing 近乎充分（ΔR²_h 0.651/0.645）
- Route P（谱组成）：多因素共同调节（去 h 后仍有 0.127/0.148）
- 深度/幅度：Phase 2 已有数据但从未跟 P(λ)/P(θ) 统一对比

但这些结论散落在四个 phase 里，用了不同的 target、不同的 CV、不同的输入集——从未放在**同一张图**上比较。Phase 2.8A 把它们统一起来。

Phase 2.7 确立了 W50 不是合适的单轨频谱描述子，但没有回答"完整 g(x) 加上 h 能解释多少"。Phase 2.8B 用完整 kernel 逐步逼近。

## 1. Task 24 — Phase 2.8A: Morphology Information Decomposition

### 1.1 信息分量分解

将矩形表面形貌 H 分解为五个正交信息分量：

| 分量 | 定义 | 已有实现 |
|---|---|---|
| **D** — 深度 | residual_Sq_um, median_depth_um | phase2 manifest（已有） |
| **A** — 幅度 | Sq_um = √mean(R²) | Phase 2.5 spectrum_descriptors |
| **P(λ)** — 谱组成 | [p_lt8, p_8_16, p_16_32, p_32_64, p_64_inf] + ILR z1–z4 | Phase 2.5 spectral_composition |
| **P(θ\|λ)** — 方向 | A2, angular_entropy (band 8_16 等) | Phase 2.5 directional_metrics |
| **φ(x,y)** — 空间实现 | residual field 本身 | Phase 1.5 load_frozen |

### 1.2 统一可预测性比较

对每个分量 target，用**同一套 Ridge（fold-internal α）× 同一套 src_gkf/proc_gkf × 同一套 input set**（u = [τ,f,h,N,v]）跑 grouped-CV，报告 fold-paired R²/Q²。

输出就是一张 **Predictability Spectrum** 表 + 图：

```text
分量        src_gKF R²/Q²   proc_gKF   h-only   ΔR²_h
D           (已有)           (已有)     (已有)    (已有)
A           (已有)           (已有)     —        —
P(λ)        0.362            (已有)     0.147    0.181
P(θ|λ)      0.663            (已有)     0.486    0.651
φ(x,y)      无法标量回归      —          —        —
```

这张图把 Phase 2/2.5/2.7 的结论统一为一句话："**工艺对形貌的控制力具有明确的信息分层**"。

### 1.3 实施

- 脚本：`24_information_decomposition.py`
- 所有模型/管线调用统一 `src/`（结构收敛后），不再允许 phase-local 重复
- 新增：A（Sq）和 D（depth）的 src/proc CV 如果 Phase 2 没有正式跑过就补
- 输出：`outputs/phase2_8/predictability_spectrum.csv` + `.png` + JSON

### 1.4 Gate

G28-A：Predictability Spectrum 图完成且各分量 CV 契约通过。不设 SUPPORTED/NOT（这是描述性统一，不是假设检验）。

---

## 2. Task 25 — Phase 2.8B: Single-track kernel → multi-track bridge

### 2.1 核心思想

不再用 W50 压缩单轨信息，保留**完整截面 profile g(x)**，与 hatch spacing h 一起预测矩形形貌。

三级逐步证伪：

| 级别 | 模型 | 含义 |
|---|---|---|
| L0 | g(x) 本身的频谱 S_g(k) | 单轨基线 |
| L1 线性叠加 | z(x) = Σ g(x−nh)，a_n = 1 | 理想 h-阵列 |
| L2 饱和 | g_merge(x) = max(g₁,g₂) 或 clip | 相邻槽熔并 |
| L3 邻轨交互 | a_n = 1 + c(−1)ⁿ 或 a_n = f(overlap) | 幅值调制 |

每一级计算预测谱与实测矩形谱的 TV 距离（五分类：INVALID/OUT/1/2/3）。

### 2.2 新增物理描述符（候选，不是 gate）

$$
O(h) = \frac{\int g(x)\,g(x-h)\,dx}{\int g(x)^2\,dx}
$$

以及谱域 overlap：$S_g(k) \cdot S_g(k) \cdot |\tilde A_{\rm array}(k)|^2$ 的峰选择。

这些描述符**不直接进 Gate**——先看在 grouped-CV 中是否稳定提升 Route T 预测（Δ ≥ 0.05），再决定是否升级为机制描述符。

### 2.3 实施

- 脚本：`25_kernel_bridge.py`
- Level 0–3 全部走 Phase 2.7 `_lib.synth_field` + `field_class`（同管线）
- 输出：`outputs/phase2_8/kernel_bridge_levels.csv` + `summary/gsl28_b_evaluation.json`

### 2.4 Gate

G28-B：L1→L3 逐级 TV 改善的单调性报告（描述性，不设硬门槛）。如果 L2 或 L3 显著优于 L1（ΔTV ≥ 0.10），登记为"非线性交互有证据"——Phase 3 的物理锚点。

---

## 3. 前置：结构收敛（src/ 重构）

Phase 2.8 是重构后第一个新 phase——在开始之前必须完成以下合并：

| 目标模块 | 合并来源 | 内容 |
|---|---|---|
| `src/cv.py` | phase2 `_lib.gkf_splits/gss_splits/check_*` + phase2_5 再导出 | grouped CV + 契约校验 |
| `src/composition.py` | phase2_5 `_lib.five_part_composition/ilr_transform/ilr_inverse/aitchison_distance/apply_zero_replacement` | 谱组成 + ILR |
| `src/spectrum.py` | phase2_5 `_lib.radial_spectrum/spectrum_descriptors/directional_band_metrics` + `l15.dct_lambda_grid` | 径向/方向谱 |
| `src/geometry.py` | phase2_6 `_lib.sample_profiles/lateral_positions/axis_frame/assign_class/plateau_stable_run/...` | 单线几何 |
| `src/statistics.py` | phase2_5 `_lib.sign_matrix/exact_signflip_test/moran_*` + phase2_6 bootstrap | 统计检验 |
| `src/provenance.py` | phase2 `manifest` 读取 + config 路径 | 数据溯源 |
| `src/data.py` | phase1_5 `load_frozen` + `src/io_cag/io_npz` | 数据加载 |

实施方式：**不移动旧文件**（保持 Phase 1.5–2.7 的提交链可追溯），而是在 `src/` 下新建模块，将公共函数提为 canonical implementation，旧 `_lib` 改为 `from src.xxx import ...` 的 thin re-export。逐模块做，每模块一个 commit，跑全测试确认无回归。

这一步做完后，Phase 2.8 的脚本直接 `from src.cv import gkf_splits`，不再依赖任何 phase-local `_lib` 链。

---

## 4. 执行顺序

1. **结构收敛**：src/ 六模块 → 全测试回归 → commit（1–2 天）
2. **Task 24**（2.8A）：`information_decomposition.py` → Predictability Spectrum → commit
3. **Task 25**（2.8B）：`kernel_bridge.py` → 逐级 TV → commit
4. **gate_eval**：phase2_8_gate_eval.md → commit

Task 24 和 25 都是分钟级计算，一天可完成全部 formal。
