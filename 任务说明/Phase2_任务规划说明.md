# Phase 2 任务规划说明：形貌失稳审计与尺度分辨工艺可解释性

> 建议文件路径：`experiments/phase2/Phase2_任务规划说明.md`  
> 建议状态：`DRAFT_FOR_REVIEW`  
> 本阶段定位：**低模型假设的结构审计 + 可解释工艺映射**  
> 本阶段原则：**不预设离散物理机制、不把 PCA 维数解释为物理维数、不把预测残差直接解释为随机性、不提前引入 Mamba/深度神经网络。**

---

## 0. Phase 2 的出发点

Phase 1 / Phase 1.5 已经得到几个足以改变后续路线的事实。

1. **absolute height 与加工深度高度相关**  
   原始高度场 absolute PC1 的 explained variance ratio 约为 98.77%，说明整个 200 样本集合最强的变化首先是整体深度变化。

2. **去除 per-sample median depth 后，residual morphology 不再是简单的一维问题**  
   residual PC1 EVR 约为 25.12%，PC2 约为 20.13%，前多个主成分共同描述剩余形貌变化。

3. **raw / repaired 结果高度一致**  
   raw 与 repaired residual PCA 子空间差异较小，因此当前 residual morphology 的主要结构不能简单归因于锥坑修复步骤。

4. **residual morphology 具有明显的空间尺度依赖**  
   某些 DCT 空间波段的 PC1 bootstrap 稳定性明显优于全 residual 场，但 matched-null 与 LOCO 分析表明，这种稳定性还不足以证明存在单一、全局稳定的形貌流形。

5. **少数表面对 PCA / conditional PCA 具有高杠杆作用**  
   某些单样本删除可造成很大的模态旋转。因此当前最大的科学问题之一不是“如何把全部数据压进一个 latent space”，而是：
   > 这些高 leverage 样本到底是测量异常、局部伪影、连续分布尾部，还是进入了另一种真实形貌状态？

6. **简单五维工艺欧氏距离与形貌距离相关较弱**  
   但 exact-repeat sentinel 49/50 的形貌距离远低于普通样本对。  
   这提示：
   - 不能直接得出“工艺与形貌弱相关”；
   - 也不能直接得出“形貌主要随机”；
   - 更合理的工作假设是：**工艺→形貌映射可能具有强非线性、强交互、局部状态切换或条件依赖。**

7. **当前 pass 数据不支持一个与工艺无关的 universal pass dynamics**  
   15 条 pass 数据是 cross-sectional pseudo-trajectories，不是同一个槽的逐 pass longitudinal measurement。相邻 pass 的形貌增量方向跨工艺条件并不一致，因此本阶段不做统一时序动力学建模。

因此 Phase 2 不从“建立复杂模型”开始，而从两个核心问题开始：

\[
\boxed{
\text{Phase 2A：高杠杆/异常形貌究竟是什么？}
}
\]

\[
\boxed{
\text{Phase 2B：不同空间尺度的形貌，究竟能被工艺条件解释到什么程度？}
}
\]

---

# 1. Phase 2 总目标

Phase 2 的总目标不是立即得到最终预测模型，而是建立一个可以支撑后续主线选择的、可审计的证据链。

具体回答四个问题：

### Q1. 是否存在真实的 morphology instability / regime candidate？

即：

\[
u \approx u'
\]

时，是否可能出现结构上显著不同的形貌？

以及：

\[
H_i
\]

中高 leverage 样本是否集中在某些工艺区域、空间尺度或形貌描述符上。

---

### Q2. 哪些空间尺度是“工艺可解释”的？

比较：

\[
u \rightarrow Y_\lambda
\]

在不同空间尺度 \(\lambda\) 上的 out-of-sample predictive power。

目标不是追求最高 R²，而是建立：

\[
R^2(\lambda)
\]

随空间尺度变化的结构。

---

### Q3. 非线性工艺坐标是否比原始工艺坐标更能组织形貌？

比较：

\[
u_{\rm raw}
=
(\tau,f,h,N,v)
\]

与物理重参数化：

\[
u_{\rm phys}
=
(\tau,E_p,\Delta x,n_A,D_E,\ldots)
\]

是否能使形貌映射更简单、更平滑、更具泛化性。

---

### Q4. 下一阶段应该走哪条路线？

Phase 2 最终不是只输出一个模型，而是根据数据选择主线：

- deterministic nonlinear process→morphology；
- regime-specific morphology modeling；
- morphology instability / transition；
- repeatability / stochastic branching；
- hidden-state / state-dependent dynamics；
- multi-scale predictive representation。

---

# 2. Phase 2 明确不做什么

Phase 2 暂不进行以下工作：

- 不使用 Mamba；
- 不使用 Transformer；
- 不使用大规模 CNN；
- 不使用复杂 autoencoder 作为首轮证据；
- 不使用 SINDy / Koopman 作为主要分析；
- 不做“最小预测状态”正式论证；
- 不用 mutual information 直接估计 latent sufficiency；
- 不把 PCA component 数量解释为真实物理自由度；
- 不把模型 residual 直接解释为 stochasticity；
- 不把数据驱动 cluster 自动命名成：
  - brittle fracture；
  - thermal accumulation；
  - phase transition；
  - plasma regime；
  - melting regime；
  - crack-dominated regime；
- 不删除高 leverage 样本后再报告“更漂亮”的 PCA 结果；
- 不把 N=1→4 pseudo-trajectory 称为同槽 dynamics；
- 不把 N=4→5 的变化直接解释为 pass transition。

---

# 3. 数据冻结与输入

## 3.1 主高度数据

继续使用：

```text
outputs/rectangle_registration/
  manual_internal_roi_v1/
    dataset/
      stable_roi_80um_dataset.npz
```

主要数组：

```text
height_raw[200,160,160]       # 主分析
height_repaired[200,160,160]  # sensitivity
valid_mask[200,160,160]
repair_mask[200,160,160]
session_id
measurement_id
sample_id
x_um
y_um
```

主分析继续以：

```text
height_raw
```

为 authority。

`height_repaired` 只用于 sensitivity analysis。

---

## 3.2 Phase 1 / Phase 1.5 冻结输入

Phase 2 原则上不重新定义 Phase 1.5 已经生成的变量。

继续读取：

```text
outputs/phase1_minimal/
  exploration_manifest.csv
  raw_repaired_sensitivity.csv
```

以及：

```text
outputs/phase1_5/
  morphology_descriptors.csv
  scale_energy_table.csv
  scale_pca_bootstrap.csv
  conditional_pca_table.csv
  depth_window_table.csv
  pairwise_distance_summary.csv
  sentinel_multiscale_table.csv
  session_separability.csv
  pass_step_stats.csv
  loco_top5_influencers.csv
```

---

## 3.3 Phase 2 统一样本 manifest

新建：

```text
outputs/phase2/phase2_manifest.csv
```

每行对应一个真实 ROI。

建议字段：

```text
dataset_index
session_id
measurement_id
sample_id
shared_height_source_id
processing_order

x_position_um
y_position_um

compressor_steps
pulse_duration_fs
frequency_kHz
hatch_spacing_um
pass_count
velocity_mm_s

median_depth_um
residual_Sq_um
repair_fraction

is_formal
is_pass60
is_supplement

phase1_global_loco_rank
phase1_global_loco_angle_deg
```

如果加入后续 derived process coordinates，也必须保留原始 process variables。

---

# 4. Phase 2 目录结构

建议新增：

```text
experiments/
  phase2/
    Phase2_任务规划说明.md
    phase2_config.yaml
    _lib.py

    01_instability_inventory.py
    02_instability_montage.py
    03_local_neighborhood_audit.py

    04_build_multiscale_targets.py
    05_process_explainability_cv.py
    06_physics_coordinate_comparison.py
    07_scale_predictability_summary.py

    08_local_regime_probe.py
    09_sensitivity_checks.py

outputs/
  phase2/
    manifest/
    instability/
    multiscale_targets/
    process_explainability/
    local_structure/
    sensitivity/
```

Phase 2A 对应脚本 01–03。

Phase 2B 对应脚本 04–07。

08–09 为 Phase 2 收尾与路线分叉实验。

---

# 5. Phase 2A：Morphology Instability Audit

## 5.1 核心科学问题

Phase 2A 不问：

> 哪些样本是 outlier？

而问：

> **哪些样本会显著改变总体形貌结构，以及这种高 leverage 是否对应真实、可解释、可重复的 morphology state candidate？**

需要区分四种可能：

### A. measurement / preprocessing artifact

例如：

- ROI 边界问题；
- 背景平面问题；
- CAG 解码问题；
- repair mask 影响；
- 大面积 dropout；
- 局部异常值。

---

### B. 连续分布尾部

即样本虽然极端，但仍位于同一连续形貌族上。

---

### C. process-conditioned morphology branch

即某些工艺区域具有另一条形貌支路。

---

### D. stochastic / hidden-variable candidate

即工艺坐标十分接近，但真实形貌差异仍然很大，并且这种差异超过 repeatability floor。

Phase 2A 只建立 candidate，不直接做机制命名。

---

# 6. Phase 2A-1：Instability inventory

脚本：

```text
01_instability_inventory.py
```

---

## 6.1 单样本 instability feature vector

为每个样本构建：

\[
A_i
=
[
A^{\rm shape},
A^{\rm spectral},
A^{\rm leverage},
A^{\rm local},
A^{\rm artifact}
]
\]

不要第一版就压成单一 scalar score。

优先保留多维 audit table。

### morphology amplitude

包括：

```text
median_depth_um
Sq
Sa
peak_to_valley
deepest_negative_residual
pit_density
gradient_rms
laplacian_rms
```

---

### spatial-frequency features

对 DCT band：

```text
8–16 um
16–32 um
32–64 um
>=64 um
```

记录：

```text
band_energy_fraction
band_rms_um
band_PC1_score
band_PC2_score
```

---

### leverage features

至少包括：

```text
global_total_LOCO_angle
global_DCT8_16_LOCO_angle
global_DCT16_32_LOCO_angle
global_DCT32_64_LOCO_angle
global_DCT64_inf_LOCO_angle
```

如果 Phase 1.5 只保存 top-5，则 Phase 2 重新完整计算所有 200 个样本的 LOCO influence。

得到：

\[
L_i^{(\lambda)}
=
\theta
\left(
\Phi_{\rm full}^{(\lambda)},
\Phi_{-i}^{(\lambda)}
\right)
\]

其中至少对 PC1 和 PC1–3 子空间计算。

---

### local morphology isolation

定义 morphology descriptor space：

\[
m_i\in\mathbb R^p
\]

标准化后计算 kNN morphology distance：

\[
D^{morph}_{i,k}
=
\mathrm{median}_{j\in kNN(i)}
\|m_i-m_j\|
\]

建议：

```text
k = 5
k = 10
```

两套都保存。

---

### local process isolation

在 raw process space 与 physics process space 分别计算：

\[
D^{proc}_{i,k}.
\]

这样区分：

- process isolated；
- morphology isolated；
- process-near but morphology-far。

---

### artifact diagnostics

保存：

```text
repair_fraction
valid_fraction
distance_to_ROI_boundary_if_available
plane_correction_metric_if_available
dropout_component_count_if_available
```

---

## 6.2 不建议立即定义单一 anomaly score

第一轮不建议：

\[
A_i
=
w_1 Sq+w_2 LOCO+\cdots
\]

因为权重人为。

优先建立：

```text
instability_inventory.csv
```

并给每个变量 rank。

例如：

```text
rank_Sq
rank_LOCO
rank_local_morph_distance
rank_pit_density
rank_band_energy
```

然后定义一个非常保守的 consensus rank：

\[
A_i^{consensus}
=
\mathrm{median}
(
r_{Sq},
r_{LOCO},
r_{local},
r_{pit},
r_{spectral}
)
\]

只用于排序，不解释成物理量。

---

## 6.3 重点 audit 样本

默认选择：

```text
top 20
```

但必须包含：

- global LOCO top samples；
- scale-specific LOCO top samples；
- residual Sq top samples；
- local morphology isolation top samples；
- process-near/morphology-far top samples；
- exact repeat 49/50 作为低异常对照。

不要只按一个综合分数选样。

---

# 7. Phase 2A-2：Instability montage

脚本：

```text
02_instability_montage.py
```

对每个重点样本生成统一格式诊断页。

建议每个样本 1 页：

```text
Panel A: absolute H
Panel B: residual R
Panel C: DCT 8–16
Panel D: DCT 16–32
Panel E: DCT 32–64
Panel F: DCT >=64
Panel G: repair mask
Panel H: row/column profile
Panel I: radial or directional PSD
Panel J: nearest process neighbors
Panel K: nearest morphology neighbors
Panel L: process metadata + descriptor summary
```

---

## 7.1 图像尺度

必须同时提供：

### individual scale

每张图按自身 percentile scaling：

```text
P2–P98
```

用于看内部结构。

### fixed group scale

所有重点样本使用同一色标。

用于比较真实 amplitude。

两种图不能混为一谈。

---

## 7.2 视觉检查 checklist

人工审计至少记录：

```text
edge contamination?
large-area dropout?
repair-driven feature?
large pit?
ridge?
periodic stripe?
anisotropic texture?
low-frequency waviness?
localized collapse?
multi-lobe morphology?
```

但这些字段只作为形貌描述，不赋予机制。

保存：

```text
instability_manual_review.csv
```

字段建议：

```text
dataset_index
reviewer
artifact_suspected
artifact_reason
morphology_pattern
confidence
notes
```

---

# 8. Phase 2A-3：Process-neighborhood audit

脚本：

```text
03_local_neighborhood_audit.py
```

这是 Phase 2A 最关键的定量实验之一。

---

## 8.1 对每个重点样本找 process neighbors

分别在：

### raw process space

\[
(\tau,f,h,N,v)
\]

### physics process space

\[
(\tau,E_p,\Delta x,n_A,D_E)
\]

中找最近邻。

建议：

```text
k = 5
```

---

## 8.2 比较 morphology distance

得到：

\[
D_{\rm morph}(i,j)
\]

分别在：

```text
total residual
DCT 8–16
DCT 16–32
DCT 32–64
DCT >=64
descriptor space
```

计算。

---

## 8.3 建立四类 pair

### Type I

\[
D_{proc}\ll1,\quad D_{morph}\ll1
\]

解释：

> 工艺相近、形貌也相近。

---

### Type II

\[
D_{proc}\ll1,\quad D_{morph}\gg1
\]

解释：

> morphology branching / hidden-variable candidate。

这是最需要关注的。

---

### Type III

\[
D_{proc}\gg1,\quad D_{morph}\ll1
\]

解释：

> process degeneracy / morphology-equivalent candidate。

---

### Type IV

\[
D_{proc}\gg1,\quad D_{morph}\gg1
\]

普通 pair。

---

## 8.4 exact repeat floor

49/50 只能作为 sentinel。

定义：

\[
D_{\rm repeat}^{(\lambda)}
=
D_{49,50}^{(\lambda)}
\]

Phase 2 中可以展示：

\[
\frac{D_{ij}^{(\lambda)}}
{D_{\rm repeat}^{(\lambda)}}
\]

但必须标注：

> 只有一个 exact-repeat condition，因此不是全局噪声标准差，也不能称为 universal noise floor。

建议名称：

```text
sentinel-normalized morphology distance
```

而不是：

```text
noise-normalized distance
```

---

# 9. Phase 2A 输出

建议生成：

```text
outputs/phase2/instability/
  instability_inventory.csv
  instability_top20.csv
  instability_manual_review.csv

  instability_montage.pdf/png
  leverage_vs_Sq.png
  leverage_vs_local_distance.png
  process_near_morph_far_pairs.csv
  process_far_morph_near_pairs.csv

  sample_cards/
    sample_066.png
    ...
```

---

# 10. Phase 2A 验收标准

Phase 2A 完成的标准不是“找出几个异常点”。

必须回答：

### A. 高 leverage 样本是否主要由 artifact 驱动？

如果是：

> 先修 preprocessing / QA，再进入 Phase 2B。

---

### B. 高 leverage 是否集中在某类真实形貌结构？

如果是：

> 进入 regime / instability candidate 分析。

---

### C. 高 leverage 是否只来自连续幅度尾部？

如果是：

> 更适合 robust regression / nonlinear manifold，而不是 discrete regime。

---

### D. process-near/morphology-far pair 是否显著存在？

如果大量存在：

> 后续必须加强 repeatability / hidden-state 实验。

如果基本不存在：

> deterministic process→morphology 方向优先级提高。

---

# 11. Phase 2B：Scale-resolved Process Explainability

Phase 2B 的目标不是预测完整 160×160 高度图。

目标是回答：

\[
\boxed{
\text{工艺参数到底能解释哪些 morphology observables？}
}
\]

并建立：

\[
R^2(\lambda)
\]

或其他 out-of-sample performance 随空间尺度的图谱。

---

# 12. Phase 2B-1：构造多尺度 target

脚本：

```text
04_build_multiscale_targets.py
```

---

## 12.1 target family A：整体加工量

包括：

```text
median_depth_um
```

必要时增加：

```text
P10/P50/P90 depth-like statistics
```

但注意当前 80×80 µm ROI 是中心 interior ROI，不应把它解释为完整槽宽或完整槽截面几何。

---

## 12.2 target family B：surface roughness descriptors

例如：

```text
Sq
Sa
Ssk
kurtosis_excess_fisher
gradient_rms
lap_rms
acf_e_fold_lag_um
anisotropy
pit_density_per_Mpx
```

---

## 12.3 target family C：spatial-band amplitude

对：

```text
DCT 8–16 um
DCT 16–32 um
DCT 32–64 um
DCT >=64 um
```

记录：

```text
band_rms
band_energy_fraction
```

---

## 12.4 target family D：band-specific low-dimensional scores

对每个 band 单独做 PCA。

但必须遵守：

1. PCA 只在 training fold 内拟合；
2. test fold 只做 projection；
3. 不允许在完整 200 样本上先 PCA 再 cross-validation；
4. 每个 band 第一轮只使用：
   ```text
   PC1
   PC2
   PC3
   ```
5. 如果某个 band 的 PC1 bootstrap 极不稳定，则：
   - 不把其 PC score 当核心 target；
   - 只保留 descriptor / band RMS；
   - 或使用训练折 bootstrap stability 作为附加报告。

---

# 13. Phase 2B-2：工艺坐标体系

Phase 2B 至少比较三套 input representation。

---

## 13.1 Input Set A：raw controls

\[
u_A=
[
\tau,
f,
h,
N,
v
]
\]

对应：

```text
pulse_duration_fs
frequency_kHz
hatch_spacing_um
pass_count
velocity_mm_s
```

---

## 13.2 Input Set B：physics-reparameterized

在平均功率固定为 \(P\) 的前提下，增加：

### pulse energy

\[
E_p=\frac{P}{f}
\]

---

### pulse-to-pulse scan spacing

若：

- \(v\)：mm/s
- \(f\)：kHz

则注意单位统一后：

\[
\Delta x=\frac{v}{f}
\]

---

### nominal areal pulse density proxy

\[
n_A
\propto
\frac{Nf}{vh}
\]

---

### nominal areal energy dose

\[
D_E
=
\frac{PN}{vh}
\]

---

同时保留：

```text
pulse_duration_fs
```

因为脉宽不能由上述量替代。

---

## 13.3 Input Set C：hybrid

\[
u_C
=
[
u_{\rm raw},
u_{\rm phys}
]
\]

但注意 \(f\) 与 \(E_p=P/f\) 强共线。

因此：

- Ridge 可以接受；
- ordinary linear regression 需检查 condition number；
- tree model 可以保留；
- GAM 需要避免重复表达同一个自由度。

---

# 14. Phase 2B-3：模型集合

脚本：

```text
05_process_explainability_cv.py
```

首轮模型只允许：

```text
DummyRegressor
LinearRegression
Ridge
RandomForestRegressor
ExtraTreesRegressor
GaussianProcessRegressor
GAM
```

其中 GPR / GAM 如果实现成本太高，可以作为第二批。

第一批最低要求：

```text
Dummy
Ridge
ExtraTrees
```

---

# 15. Cross-validation 设计

这是 Phase 2B 的关键。

不能使用普通随机 train/test split 作为唯一结果。

---

## 15.1 shared source grouping

必须按：

```text
shared_height_source_id
```

分组。

共享来源的 ROI 不允许跨 train/test。

---

## 15.2 推荐 CV

优先：

```text
Repeated GroupKFold
```

或：

```text
GroupKFold
```

例如：

```text
n_splits = 5
```

外加多个固定随机重复的 grouped partition。

---

## 15.3 pseudo-trajectory 泄漏防范

60-pass / supplement 中，同一个 base condition 的多个 N 属于强相关设计点。

建议增加第二套更严格 split：

```text
group = base_process_condition_id
```

确保同一 base condition 的 N=1..4 不跨 train/test。

因此最终至少报告：

### CV-A

```text
shared_height_source grouped
```

### CV-B

```text
base-process-condition grouped
```

如果两者差距很大，需要明确说明模型泛化主要依赖相邻 design condition。

---

# 16. 评价指标

每个 target 保存：

```text
R2
MAE
RMSE
Spearman rho
```

其中：

- R² 作为主要 explainability 指标；
- MAE / RMSE 保留物理单位；
- Spearman 用于检查模型是否至少抓住单调排序。

报告：

```text
fold-wise value
median
Q25
Q75
Q10
Q90
```

不要只报均值。

---

# 17. 核心结果：Scale Predictability Curve

对不同尺度 target 形成：

\[
\lambda
\rightarrow
R^2_{\rm CV}
\]

例如：

```text
depth
>=64 um
32–64 um
16–32 um
8–16 um
```

如果同一 band 有多个 target：

```text
band RMS
PC1
PC2
```

则分开画。

不要混成一个人为综合分数。

---

# 18. Phase 2B-4：模型增量比较

脚本：

```text
06_physics_coordinate_comparison.py
```

比较：

\[
\Delta R^2
=
R^2(u_{\rm phys})
-
R^2(u_{\rm raw})
\]

以及：

\[
\Delta R^2_{\rm nonlinear}
=
R^2(\text{ExtraTrees})
-
R^2(\text{Ridge})
\]

---

## 18.1 解释逻辑

### 情况 A

\[
R^2_{\rm phys}
>
R^2_{\rm raw}
\]

且多折稳定：

> 物理重参数化使 process→morphology relation 更简单。

---

### 情况 B

\[
R^2_{\rm tree}
\gg
R^2_{\rm ridge}
\]

> 形貌映射具有明显非线性 / 交互。

---

### 情况 C

两者都低：

> 不能简单归因于“随机”。

需要继续检查：

- regime mixing；
- hidden variables；
- measurement variation；
- process descriptor 不充分。

---

### 情况 D

低频形貌高 R²，高频形貌低 R²：

> 形貌 predictability 具有明显空间尺度依赖。

这是 Phase 2 最值得关注的潜在结果之一。

---

# 19. Phase 2B-5：变量重要性

树模型可以输出：

```text
permutation importance
```

不建议把 impurity importance 作为主要证据。

对每个 target：

\[
Y_\lambda
\]

计算：

\[
I_j^{(\lambda)}
\]

得到：

```text
feature × spatial scale
```

热图。

例如可能出现：

```text
pass_count         -> low-frequency morphology
hatch_spacing      -> mid-frequency texture
frequency          -> certain band
pulse_duration     -> instability-sensitive descriptor
```

但只能描述统计关系，不直接命名机制。

---

# 20. Partial dependence / ALE

对最稳定 target 可以做：

```text
ALE
```

优先于传统 PDP。

原因是 process variables 可能有关联或设计约束。

重点考察：

```text
pulse_duration_fs
frequency_kHz
hatch_spacing_um
pass_count
velocity_mm_s
```

以及：

```text
pulse_energy
pulse_spacing
nominal_areal_dose
```

---

# 21. Phase 2B-6：Local model probe

脚本：

```text
08_local_regime_probe.py
```

这一部分只在 Phase 2A 发现明显异质性后启动。

不做“硬聚类然后当真”。

先比较：

### global model

\[
Y=F(u)
\]

### local model

\[
Y=F_r(u)
\]

其中 \(r\) 可以是：

- depth quantile；
- morphology amplitude range；
- process-region；
- instability score quantile。

---

## 21.1 关键检验

如果：

\[
R^2_{\rm local}
\gg
R^2_{\rm global}
\]

并且改善不是因为小样本 overfit：

> 支持 condition-dependent / local morphology representation。

如果 local model 没有稳定提升：

> 不应继续强调 discrete regime。

---

# 22. Phase 2B 输出

建议：

```text
outputs/phase2/process_explainability/
  multiscale_targets.csv

  cv_fold_results.csv
  cv_summary.csv

  raw_vs_physics_coordinates.csv
  ridge_vs_tree.csv

  scale_predictability_curve.png
  scale_predictability_by_model.png
  scale_predictability_by_input.png

  permutation_importance.csv
  permutation_importance_heatmap.png

  ale/
```

---

# 23. Sensitivity checks

脚本：

```text
09_sensitivity_checks.py
```

至少完成以下 sensitivity。

---

## 23.1 raw vs repaired

重复主要：

```text
target generation
scale predictability
instability rank
```

若结论变化明显，需要报告。

---

## 23.2 formal-only

因为 120 formal DOE 的设计结构最干净。

Phase 2B 所有 process explainability 至少额外跑：

```text
formal-only
```

避免 pass-session 数据结构主导模型。

---

## 23.3 exclude top leverage samples

可以做 sensitivity：

```text
all samples
minus top-1
minus top-5
```

但只能回答：

> 结果对高 leverage 样本是否敏感？

禁止把“去掉后性能更好”解释为应该删除这些样本。

---

## 23.4 spatial-band definition

Gaussian bands 与 DCT bands 对主要结论做交叉验证。

如果只在某一种滤波定义下成立，需要降低结论强度。

---

# 24. Phase 2 决策门

Phase 2 结束后根据结果选择 Phase 3。

---

## Route A：deterministic nonlinear mapping

触发条件：

- repeatability candidate 很好；
- process-near/morph-far pair 不多；
- ExtraTrees / GPR 明显优于 Ridge；
- 多个 morphology scale 有较高 CV performance。

进入：

```text
Phase 3A:
nonlinear process -> morphology surrogate
```

之后再考虑更复杂 representation。

---

## Route B：scale-dependent predictability

触发条件：

\[
R^2_{\rm low}
\gg
R^2_{\rm high}
\]

且 across sensitivity 稳定。

进入：

```text
Phase 3B:
multi-scale deterministic / unresolved decomposition
```

重点研究不同空间尺度的信息来源与可预测性。

注意仍不直接把 high-frequency residual 叫 stochastic。

---

## Route C：regime-specific morphology

触发条件：

- 高 leverage 样本形成可重复结构；
- local model 显著优于 global；
- 某些 process region 出现系统性 morphology branch。

进入：

```text
Phase 3C:
regime probability + within-regime morphology model
```

形式：

\[
p(r|u),
\qquad
p(H|u,r)
\]

但 regime 名称保持数据驱动，例如：

```text
R1
R2
R3
```

直到外部物理证据支持机制命名。

---

## Route D：hidden variable / stochastic branching candidate

触发条件：

- process-near/morphology-far pair 大量存在；
- 新增 repeat experiments 显示同一工艺条件存在明显多样性；
- 这种多样性不能由 measurement / preprocessing 解释。

进入：

```text
Phase 3D:
repeatability + hidden-state study
```

这时才真正值得讨论：

\[
H_{t+1}
=
F(H_t,u_t,\eta_t)
\]

以及 predictive-state 问题。

---

# 25. 推荐补充实验

Phase 2 可以使用现有数据完成，但为了区分 deterministic nonlinear 与 stochastic branching，建议同步规划下一轮物理实验。

---

## 25.1 repeatability matrix

选择：

```text
5–10 个代表工艺条件
```

每个：

```text
3–5 exact replicates
```

覆盖：

- smooth region；
- intermediate region；
- high-leverage neighborhood；
- high frequency；
- low frequency；
- high pass；
- low pass。

目标估计：

\[
\sigma_{\rm repeat}(u,\lambda)
\]

而不是只靠一个 49/50 sentinel。

---

## 25.2 interrupted pass experiment

如果未来要研究状态/记忆：

同一物理位置：

\[
H_0
\rightarrow
H_1
\rightarrow
H_2
\rightarrow
H_3
\rightarrow\cdots
\]

每 pass 后重新测量。

这比增加更多 cross-sectional pseudo-trajectories 更有价值。

---

# 26. 代码质量与可复现要求

Phase 2 继续沿用 Phase 1.5 的严格做法。

---

## 26.1 config driven

所有关键参数放入：

```text
phase2_config.yaml
```

例如：

```yaml
random_seed: 20260903

instability:
  top_n: 20
  knn_k:
    - 5
    - 10

pca:
  max_components: 6

cv:
  n_splits: 5

models:
  ridge_alpha_grid:
    - 0.01
    - 0.1
    - 1
    - 10
    - 100
```

---

## 26.2 禁止 hidden magic numbers

脚本中：

- 阈值；
- band；
- kNN k；
- CV split；
- top-N；

全部进入 config 或 constants。

---

## 26.3 tests

新增：

```text
tests/test_phase2_lib.py
```

至少测试：

1. process feature units；
2. \(E_p=P/f\)；
3. \(\Delta x=v/f\) 单位换算；
4. \(n_A\)；
5. \(D_E\)；
6. grouped CV 无 leakage；
7. training-fold PCA 无 test leakage；
8. DCT decomposition reconstruction；
9. kNN self-exclusion；
10. LOCO angle；
11. raw/repaired indexing consistency；
12. base-condition grouping。

---

# 27. 运行顺序

推荐：

```powershell
python experiments/phase2/01_instability_inventory.py
python experiments/phase2/02_instability_montage.py
python experiments/phase2/03_local_neighborhood_audit.py
```

先人工审计 Phase 2A。

确认没有 preprocessing blocker 后，再：

```powershell
python experiments/phase2/04_build_multiscale_targets.py
python experiments/phase2/05_process_explainability_cv.py
python experiments/phase2/06_physics_coordinate_comparison.py
python experiments/phase2/07_scale_predictability_summary.py
```

最后：

```powershell
python experiments/phase2/08_local_regime_probe.py
python experiments/phase2/09_sensitivity_checks.py
```

---

# 28. Phase 2 最低核心图

建议至少形成以下 8 张图。

---

## Figure 1 — Morphology instability atlas

Top leverage / top Sq / top local-isolation 样本 montage。

---

## Figure 2 — Leverage vs morphology descriptors

例如：

\[
LOCO
\text{ vs }
Sq
\]

以及：

\[
LOCO
\text{ vs }
local\ morphology\ distance.
\]

---

## Figure 3 — Process-near / morphology-far pairs

展示最典型的 pair。

---

## Figure 4 — Scale-resolved target map

不同 DCT band 的 variance / descriptor summary。

---

## Figure 5 — Scale predictability curve

核心：

\[
R^2_{\rm CV}(\lambda).
\]

---

## Figure 6 — Raw vs physics coordinate comparison

\[
\Delta R^2(\lambda).
\]

---

## Figure 7 — Linear vs nonlinear model comparison

\[
R^2_{\rm ExtraTrees}
-
R^2_{\rm Ridge}.
\]

---

## Figure 8 — Feature importance × morphology scale

展示：

\[
I_j^{(\lambda)}.
\]

---

# 29. Phase 2 最终报告需要明确回答的问题

Phase 2 结束时，报告正文必须逐条回答：

1. 高 leverage morphology 是否主要由 artifact 引起？
2. 是否存在 process-near / morphology-far 的真实候选？
3. exact-repeat 49/50 相对普通样本对处于什么量级？
4. 哪个空间尺度最容易被工艺预测？
5. 哪个空间尺度最难预测？
6. physics process coordinates 是否改善泛化？
7. nonlinear model 是否显著优于 linear model？
8. local / regime-specific model 是否有稳定优势？
9. session effect 是否影响结论？
10. raw / repaired 是否影响结论？
11. 当前数据更支持：
    - deterministic nonlinear；
    - scale-dependent predictability；
    - regime-specific；
    - hidden-variable candidate；
    中的哪一种？
12. 下一轮实验最值得增加什么数据？

---

# 30. 本阶段允许得出的结论语言

推荐使用：

> 在当前采样工艺窗口内，去除整体加工深度后的表面形貌表现出明显的尺度依赖与样本异质性。

> 某些空间尺度上的形貌特征比其他尺度更容易由加工参数解释。

> 非线性模型相对线性模型的稳定增益表明，工艺—形貌关系可能包含显著的非线性和变量交互。

> 一部分高 leverage 样本对应真实形貌极端状态候选，但当前数据不足以为这些状态赋予具体物理机制名称。

> 当前结果提示全局统一形貌表示可能不是最自然的第一模型，应优先检验条件依赖或局部表示。

---

# 31. 本阶段禁止使用的结论语言

在没有额外证据前，不使用：

> “发现了新的物理相变。”

> “证明存在两个加工机制。”

> “高频形貌是随机噪声。”

> “PCA 证明系统只有 3 个自由度。”

> “发现了最小物理状态。”

> “Mamba 能够恢复真实隐藏状态。”

> “N=3 时出现动力学 reversal。”

> “m066 是实验异常值，应删除。”

---

# 32. 最终路线判断

Phase 2 的真正作用是完成从：

\[
\text{形貌很复杂}
\]

到：

\[
\boxed{
\text{复杂性来自哪里？}
}
\]

的转换。

建议当前主线优先级为：

\[
\boxed{
\text{Morphology instability audit}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{Scale-resolved process explainability}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{global vs local / regime model comparison}
}
\]

然后再根据证据决定是否进入：

\[
\text{nonlinear surrogate}
\]

或：

\[
\text{multi-scale representation}
\]

或：

\[
\text{regime transition}
\]

或：

\[
\text{hidden-state / predictive-state dynamics}.
\]

现阶段不建议让模型架构决定科学问题；应让 Phase 2 的实验结果决定 Phase 3 的模型形式。

---

## Appendix A. Phase 2 主要仓库依据

当前方案建立在以下冻结产物上：

```text
outputs/phase1_minimal/
  exploration_manifest.csv
  raw_repaired_sensitivity.csv

outputs/phase1_5/
  scale_energy_table.csv
  scale_pca_bootstrap.csv
  conditional_pca_table.csv
  depth_window_table.csv
  pairwise_distance_summary.csv
  sentinel_multiscale_table.csv
  session_separability.csv
  pass_step_stats.csv
  morphology_descriptors.csv
  loco_top5_influencers.csv
```

以及：

```text
experiments/phase1_5/Phase1.5_本细则.md
```

Phase 2 的所有新结论必须能够追溯到：

```text
真实高度数据
→ 明确定义的 preprocessing
→ 冻结 target / feature
→ 无 leakage 的 CV
→ sensitivity analysis
→ 明确的统计边界
```

不能以模型输出本身替代实验事实。

---

## Appendix B. 建议阶段命名

建议避免把 Phase 2 命名成：

```text
predictive_state
hidden_state
Mamba
mechanism_discovery
```

更合适：

```text
Phase 2:
Morphology Instability and Scale-Resolved Process Explainability
```

中文：

```text
Phase 2：
形貌失稳审计与尺度分辨工艺可解释性
```

这样可以最大程度保持研究方向开放，并让后续路线真正由数据决定。
