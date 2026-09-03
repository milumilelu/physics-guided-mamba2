# Phase 2.5 执行细则：形貌谱组成与工艺控制机制

> 建议路径：`experiments/phase2_5/Phase2.5_形貌谱组成与工艺控制机制_执行细则.md`  
> 建议状态：`DRAFT_FOR_REVIEW`  
> 阶段定位：**在不新增物理实验的前提下，对 200 个真实 zirconia ROI 的 residual morphology 做“幅度—谱组成—方向纹理—空间实现”分层，并检验哪些形貌成分真正具有可泛化的工艺控制规律。**

---

## 0. Phase 2.5 的核心定位

Phase 2.5 不再以“预测整幅 160×160 高度图”为主要问题，也不继续强推 discrete regime、hidden state、stochastic branching 或 Mamba。

Phase 2 已经提示，不同 morphology observable 的可预测性明显不等价：

\[
\boxed{
\text{Depth}
\neq
\text{Residual amplitude}
\neq
\text{Spectral composition}
\neq
\text{Spatial realization}
}
\]

当前最值得系统验证的工作假设是：

\[
H_i(x,y)=d_i+A_iS_i(x,y)
\]

进一步写成谱形式：

\[
PSD_i(k)=A_i^2q_i(k)
\]

其中：

- \(d_i\)：中心稳定 ROI 的加工深度；
- \(A_i\)：residual morphology 总幅度，近似由 \(S_q\) 表征；
- \(q_i(k)\)：归一化空间频率能量分布；
- 方向扩展后可写为 \(q_i(k,\theta)\)；
- 具体空间位置、相位和局域 realization 不在本阶段假设为完全由工艺决定。

Phase 2.5 的核心问题为：

\[
\boxed{
\text{Which components of morphology are actually process-controlled?}
}
\]

具体拆分为：

\[
\begin{aligned}
u &\rightarrow d &&\text{整体加工深度}\\
u &\rightarrow A &&\text{residual 总幅度}\\
u &\rightarrow q(k) &&\text{空间尺度能量分配}\\
u &\rightarrow q(k,\theta) &&\text{方向性纹理}\\
u &\rightarrow \phi(x,y) &&\text{具体空间 realization}
\end{aligned}
\]

Phase 2.5 主要研究：

\[
\boxed{u\rightarrow q(k)}
\]

以及：

\[
\boxed{u\rightarrow q(k,\theta)}
\]

并分析为什么这些可预测性没有直接体现在 \(S_q\) / band RMS 上。

---

## 1. Phase 2 已冻结的前置事实

### 1.1 depth 是强工艺响应

当前 grouped-CV 中，`median_depth_um` 可由工艺参数较好预测，非线性模型明显优于线性模型。因此：

\[
d(u)
\]

是一个强 process-controlled response。

### 1.2 residual 总幅度明显更难泛化预测

以：

```text
Sq_um
rms_DCT_8_16_um
rms_DCT_16_32_um
rms_DCT_32_64_um
rms_DCT_64_inf_um
```

为代表的绝对 morphology amplitude，在 grouped-CV 下整体解释力明显弱于 depth，且部分尺度在 formal-only 下显著下降。

这意味着当前工艺参数不能稳定决定：

\[
A_i
\]

但这**不等于**：

```text
A_i 是随机噪声
```

也不等于：

```text
工艺对形貌没有作用
```

### 1.3 normalized spectral allocation 已出现强信号

`E_DCT_8_16_frac` 相比相同 band 的 absolute RMS 显示出明显更强、更稳定的 out-of-sample explainability。

工作假设：

\[
\boxed{
\text{工艺可能比总幅度更稳定地控制形貌能量的尺度分配。}
}
\]

### 1.4 当前没有 population-level Type-II branching 证据

Phase 2A 中 individual process-near / morphology-far candidate 存在，但 Type-II count 没有稳定超过 permutation null，不支持优先进入 stochastic branching / hidden-variable 主线。

### 1.5 local heuristic strata 没有稳定模型增益

Phase 2 的：

```text
depth quartile
Sq quartile
A_consensus half
```

局部模型没有相对 global model 获得稳定优势。因此本阶段不预设 discrete regime。

### 1.6 exact-repeat 49/50 是低距离 sentinel，但不是 universal noise floor

49/50 在 total residual 和各 DCT band 上都属于 ordinary pair distribution 中非常接近的样本对。

因此至少有一个登记工艺条件能够产生高度相近的 multiscale morphology。

但只有一个 exact-repeat condition，不能据此估计全局：

\[
\sigma_{\rm repeat}(u,\lambda)
\]

Phase 2.5 可以使用该 pair 作为 numerical sentinel，但不能把它称为：

```text
universal noise floor
```

---

## 2. Phase 2.5 明确不做什么

本阶段暂不：

- 训练 Mamba；
- 训练 Transformer；
- 建立 pass-level latent dynamics；
- 把 cross-sectional pass 数据称为 longitudinal dynamics；
- 证明 stochastic branching；
- 证明 hidden state；
- 根据 cluster 自动命名材料机制；
- 把 PSD 本身宣传成创新点；
- 用 synthetic morphology label 扩充 200 个真实表面；
- 将 mechanism virtual samples 混入真实形貌数据；
- 将 `E_DCT_8_16_frac` 单一结果直接解释为某一种热机制；
- 将低 \(R^2\) 直接解释成 measurement noise；
- 将高 \(R^2\) 直接解释成 causal mechanism。

本阶段“机制”只表示：

> **工艺控制机制候选 / physics-informed explanatory structure**

而不是已完成因果证明。

---

## 3. 数据与 provenance 冻结

### 3.1 主真实数据

继续使用：

```text
outputs/rectangle_registration/
  manual_internal_roi_v1/
    dataset/
      stable_roi_80um_dataset.npz
```

数组：

```text
height_raw
height_repaired
valid_mask
repair_mask
session_id
measurement_id
sample_id
x_um
y_um
```

### 3.2 residual authority

Phase 2.5 不重新定义 residual。

必须复用 Phase 1.5 / Phase 2 已冻结的 raw-height residual：

```text
R_raw
```

或通过 Phase 1.5 `_lib.py` frozen loader 获取。

禁止在 10–14 中单独改变：

- plane correction；
- depth removal；
- ROI；
- repair；
- orientation；
- DCT band 定义。

### 3.3 主 manifest

复用：

```text
outputs/phase2/manifest/phase2_manifest.csv
```

关键 grouping：

```text
shared_height_source_id
cv_process_group
base_condition_group
session_role
```

CV-A：

```text
group = shared_height_source_id
```

问题：泛化到未见过的物理 source / surface。

CV-B：

```text
group = cv_process_group
```

问题：泛化到未见过的 process condition。

49/50 必须继续在 CV-B 中同组。

### 3.4 process input sets

#### Set A — raw controls

\[
u_A=[\tau,f,h,N,v]
\]

对应：

```text
pulse_duration_fs
frequency_kHz
hatch_spacing_um
pass_count
velocity_mm_s
```

#### Set C — raw + physics-motivated derived features

\[
u_C=[u_A,\text{derived features}]
\]

可包含：

```text
pulse_energy_proxy_uJ
scan_spacing_um
areal_pulse_density_per_mm2
areal_dose_proxy_J_per_mm2
```

功率 provenance 未完整登记前，能量/剂量必须保留 `_proxy`。

#### Legacy Set R

Phase 2 中的 R 仅作为 sensitivity。

不再称为与 A 等价的 lossless reparameterization。

Phase 2.5 的主要问题为：

\[
A\quad\text{vs}\quad C
\]

而不是：

\[
A\quad\text{vs}\quad R
\]

---

## 4. 目录结构

```text
experiments/
  phase2_5/
    Phase2.5_形貌谱组成与工艺控制机制_执行细则.md
    phase2_5_config.yaml
    _lib.py

    10_build_spectral_composition.py
    11_directional_spectrum.py
    12_spectral_process_map.py
    13_pseudopass_spectral_redistribution.py
    14_mechanism_bridge_error_atlas.py

tests/
  test_phase2_5_lib.py

outputs/
  phase2_5/
    spectral_composition/
    directional_spectrum/
    process_map/
    pseudopass/
    mechanism_bridge/
    error_atlas/
    summary/
```

---

## 5. 统一数学定义

### 5.1 residual total amplitude

主 amplitude：

\[
A_i=S_{q,i}
\]

同时保留 `Sa`、`peak_to_valley`、band RMS 作为 secondary amplitude descriptors。

### 5.2 DCT broad-band composition

继续使用 Phase 1.5 的 DCT wavelength definition。

五段 composition：

\[
\mathbf p_i=
[p_{<8},p_{8-16},p_{16-32},p_{32-64},p_{\ge64}]
\]

满足：

\[
\sum_{j=1}^{5}p_{ij}=1
\]

已有：

```text
E_DCT_8_16_frac
E_DCT_16_32_frac
E_DCT_32_64_frac
E_DCT_64_inf_frac
```

新增：

\[
p_{<8}=1-\sum_{\lambda\ge8}p_\lambda
\]

正式实现不依赖 CSV round-off 相减，而从同一 DCT coefficient energy table 重算五段，并断言四个 frozen bands 与 Phase 2 对账：

```text
atol <= 1e-8
```

不能对账则 Task 10 立即 abort。

### 5.3 composition 使用 Aitchison geometry

因为：

\[
\sum_jp_j=1
\]

不能把五个 fraction 当普通独立 Euclidean targets。

---

## 6. 固定 ILR 坐标

定义：

\[
p_1=p_{<8},\quad
p_2=p_{8-16},\quad
p_3=p_{16-32},\quad
p_4=p_{32-64},\quad
p_5=p_{\ge64}
\]

### Z1：fine vs coarse

\[
z_1=
\sqrt{\frac65}
\log
\frac{(p_1p_2)^{1/2}}
{(p_3p_4p_5)^{1/3}}
\]

解释：`<16 µm` vs `>=16 µm`。

### Z2：sub-8 vs 8–16

\[
z_2=
\sqrt{\frac12}
\log\frac{p_1}{p_2}
\]

### Z3：16–32 vs coarser

\[
z_3=
\sqrt{\frac23}
\log\frac{p_3}{(p_4p_5)^{1/2}}
\]

### Z4：32–64 vs >=64

\[
z_4=
\sqrt{\frac12}
\log\frac{p_4}{p_5}
\]

### 6.1 zero handling

先报告：

```text
min(p_j)
number of exact zeros
number < 1e-10
```

若无 zero：

```text
禁止添加 pseudocount
```

若确认为 numerical zero，采用 multiplicative replacement，主值：

```text
delta = 1e-6
```

并做 `1e-8 / 1e-5` sensitivity。

若 >5% 样本需要 replacement：

```text
STOP
```

重新审查 spectrum definition。

---

# 7. Task 10 — normalized PSD / compositional spectrum

脚本：

```text
10_build_spectral_composition.py
```

## 7.1 科学问题

回答：

> residual morphology 的总幅度与 normalized spectral shape 是否真的可以分离？

以及：

> Phase 2 中 `E_DCT_8_16_frac` 的强可预测性是单 band 特例，还是完整 spectrum redistribution 的一部分？

## 7.2 broad-band composition

输出：

```text
p_lt8
p_8_16
p_16_32
p_32_64
p_64_inf

ilr_z1
ilr_z2
ilr_z3
ilr_z4
```

## 7.3 full normalized radial spectrum

主 functional spectrum 基于**冻结 DCT wavelength grid**。

所有非 DC DCT modes 按 `log(lambda)` 分箱。

建议：

```text
n_log_bins = 24
```

bin edge 必须在 config 冻结。

每个 bin 保存：

```text
lambda_lo_um
lambda_hi_um
lambda_geo_um
n_modes
energy
energy_fraction
energy_density_per_loglambda
```

每个样本：

\[
\sum_bq_{ib}=1
\]

## 7.4 low-mode-count bins

标记：

```text
low_mode_count = n_modes < 20
```

规则：

- 可以画；
- 可以被 broad composition 吸收；
- 不单独做强 pointwise inferential claim；
- 图中必须阴影标出。

## 7.5 spectrum descriptors

计算：

### log-wavelength centroid

\[
\mu_{\log\lambda}=\sum_bq_b\log\lambda_b
\]

### geometric centroid

\[
\lambda_c=\exp(\mu_{\log\lambda})
\]

### normalized entropy

\[
H=
-\frac{\sum_bq_b\log q_b}{\log B}
\]

### effective number

\[
N_{\rm eff}=
\exp\left(-\sum_bq_b\log q_b\right)
\]

### peak wavelength

```text
lambda_peak_um
peak_low_mode_count
```

只作 descriptive descriptor。

## 7.6 amplitude–composition consistency

显式验证：

\[
RMS_\lambda\approx S_q\sqrt{p_\lambda}
\]

输出：

```text
max_abs_error
median_relative_error
```

用于解释为什么 band RMS 弱预测与 fraction 强预测可同时成立。

## 7.7 输出

```text
outputs/phase2_5/spectral_composition/
  spectral_composition.csv
  ilr_coordinates.csv
  radial_spectrum_long.csv
  radial_spectrum_matrix.npz
  spectrum_descriptor_summary.csv

  amplitude_vs_fraction_consistency.csv
  dct_reconciliation.csv

  mean_normalized_spectrum.png
  spectrum_by_process_quantiles.png
```

`radial_spectrum_matrix.npz` 作为 cache，加入 gitignore。

---

# 8. Task 11 — directional spectrum

脚本：

```text
11_directional_spectrum.py
```

## 8.1 科学问题

Phase 2A 出现：

```text
periodic stripe
anisotropic texture
```

家族。

Task 11 问：

> `E_DCT_8_16_frac` 的可预测性是否部分来自方向性纹理，而不是普通 isotropic roughness？

## 8.2 directional spectrum

采用：

```text
2D FFT + separable Hann window
```

原因：提供完整 angular wave-vector information。

## 8.3 preprocessing

每个 residual map：

1. 不重新 plane detrend；
2. 不重新 polynomial correction；
3. 只移除 DC；
4. 乘二维 Hann；
5. FFT；
6. PSD = \(|F(k_x,k_y)|^2\)；
7. 用 \(\sum w^2\) 做 window energy normalization。

## 8.4 directional metrics

对：

```text
8–16 µm
16–32 µm
32–64 µm
>=64 µm
```

计算：

### second angular moment

\[
A_2=
\frac{
\left|\sum P(k,\theta)e^{i2\theta}\right|
}{
\sum P(k,\theta)
}
\]

### dominant wave-vector angle

\[
\theta_k=
\frac12
\arg\left(\sum Pe^{i2\theta}\right)
\]

### real-space stripe orientation

\[
\theta_{stripe}=\theta_k+90^\circ\pmod{180^\circ}
\]

禁止把 wave-vector angle 直接当 stripe direction。

### angular entropy

固定：

```text
n_theta_bins = 36
```

## 8.5 blind labels 用途

28 个 selected audit samples 只用于验证：

> 自动 anisotropy metric 是否与 blind `periodic stripe` 标签一致。

允许：

```text
AUROC
rank-biserial effect
permutation p-value
```

必须注明：

> audit set 是 enriched selection，不能据此估计总体 stripe prevalence。

所有 process modeling 必须基于完整 200 样本的自动 metric。

## 8.6 orientation frame

若没有可靠的 scan/hatch direction 相对于 image frame 的 metadata，则 orientation 只能解释为：

```text
image-frame orientation
```

不能解释为 scan-relative / hatch-relative。

## 8.7 输出

```text
outputs/phase2_5/directional_spectrum/
  directional_metrics.csv
  directional_spectrum_long.csv

  stripe_validation.csv
  stripe_validation.png
  anisotropy_vs_E8.png
  orientation_histogram.png
  example_directional_psd/
```

---

# 9. Task 12 — spectral process map

脚本：

```text
12_spectral_process_map.py
```

这是 Phase 2.5 主实验。

## 9.1 targets

### Primary P1 — 5-part composition

在 ILR：

\[
\mathbf z_i=[z_1,z_2,z_3,z_4]
\]

中训练。

### Primary P2 — directional 8–16 morphology

```text
A2_8_16
angular_entropy_8_16
```

前提是 Task 11 metric 通过 QA。

### Secondary

```text
spectral_centroid_log_um
spectral_entropy
effective_band_number
A2_16_32
A2_32_64
```

### Reference

```text
median_depth_um
Sq_um
rms_DCT_8_16_um
```

只作参考，不参与 spectrum route trigger。

---

## 10. 模型集合

第一批：

```text
DummyRegressor
Ridge
SplineTransformer + Ridge
ExtraTreesRegressor
```

### Ridge

沿用 Phase 2 fold-internal alpha selection。

### additive spline

建议：

```text
degree = 3
n_knots = 4
```

作为 low-capacity GAM-like model。

### ExtraTrees

```text
n_estimators = 500
min_samples_leaf = 2
```

用于测试明显 nonlinear / interaction gain。

---

## 11. CV 协议

Primary：

```text
src_gkf
proc_gkf
```

均 5 folds。

Sensitivity：

```text
src_gss
proc_gss
formal_only
exclude_artifact_yes
minus_top5_LOCO
```

必要时再做 repaired。

GSS 只要求每个 split 内 train/test groups 不重叠，不要求跨 split test sets 构成 partition。

---

## 12. composition 评价指标

### 12.1 multivariate Aitchison Q²

\[
Q^2_{Aitchison}
=
1-
\frac{
\sum_i\|\mathbf z_i-\hat{\mathbf z}_i\|_2^2
}{
\sum_i\|\mathbf z_i-\bar{\mathbf z}_{train(i)}\|_2^2
}
\]

dummy 必须使用对应 fold training mean。

### 12.2 Aitchison distance

\[
d_A(\mathbf p,\hat{\mathbf p})
=
\|ILR(\mathbf p)-ILR(\hat{\mathbf p})\|_2
\]

报告 median / Q25 / Q75。

### 12.3 inverse-ILR fraction MAE

报告：

```text
MAE_p_lt8
MAE_p_8_16
MAE_p_16_32
MAE_p_32_64
MAE_p_64_inf
```

### 12.4 balance-level R²

保存：

```text
R2_z1
R2_z2
R2_z3
R2_z4
```

但不能只挑表现最好的 balance 代表整体 composition。

---

## 13. process representation 比较

主要比较：

\[
A\quad\text{vs}\quad C
\]

定义 paired fold：

\[
\Delta Q^2_{C-A}
\]

如果 C 稳定更好，只解释为：

> derived combinations 给简单模型提供 useful inductive bias。

禁止写：

> C 增加了新的实验信息。

---

## 14. feature interpretation

只有 target grouped-CV 明显优于 dummy 后才解释 feature effect。

建议最低条件：

```text
median Q2 or R2 > 0.10
```

主 feature importance：

```text
permutation importance
```

不以 impurity importance 为主证据。

Spline model 输出单变量 response curve。

只有：

\[
Q^2_{ET}-Q^2_{spline}
\]

在两个 primary CV variant 均稳定为正时，才继续解释 interactions。

优先 interaction：

```text
frequency × velocity
hatch × velocity
pass × frequency
pass × hatch
pulse_duration × frequency
```

---

## 15. Task 12 输出

```text
outputs/phase2_5/process_map/
  cv_fold_results.csv
  cv_summary.csv
  composition_oof_predictions.csv
  directional_oof_predictions.csv

  input_comparison.csv
  nonlinear_comparison.csv
  permutation_importance.csv
  additive_response_curves.csv

  spectrum_predictability_map.png
  ilr_balance_predictability.png
  predicted_vs_true_composition.png
  process_feature_importance.png
```

---

# 16. Task 13 — pseudo-pass spectral redistribution

脚本：

```text
13_pseudopass_spectral_redistribution.py
```

## 16.1 科学问题

不再研究 whole-field pass-step direction，而研究：

\[
\mathbf p_N
\]

是否存在跨 matched process conditions 的共同谱重分配。

## 16.2 数据边界

Main pass：

```text
15 base conditions × N=1,2,3,4
```

只能称：

> matched-condition cross-sectional pseudo-trajectories

Supplement：

```text
10 base conditions × N=5,6
```

允许独立分析 N5→6。

禁止：

```text
N4→5
```

因为 session-confounded。

## 17. ILR step

\[
\Delta\mathbf z_{b,N}
=
\mathbf z_{b,N+1}-\mathbf z_{b,N}
\]

主：

```text
1→2
2→3
3→4
```

独立 check：

```text
5→6
```

## 18. Global exact sign-flip test

\[
T_N=
\left\|
\frac1B\sum_b\Delta\mathbf z_{b,N}
\right\|_2
\]

Null：

\[
\Delta z_b^{null}=s_b\Delta z_b,\quad s_b\in\{-1,+1\}
\]

N1–4：

```text
2^15 = 32768 exact flips
```

N5–6：

```text
2^10 = 1024 exact flips
```

每个 z1..z4 同时做 coordinate-wise sign-flip，同一 step 内 Holm correction。

不再使用 turning-cosine reversal language。

## 19. 与 depth 的关系

secondary：

\[
\Delta d_{b,N}
\]

vs：

\[
\Delta z_{b,N}
\]

做 Spearman。

只解释为 cross-sectional association。

## 20. 输出

```text
outputs/phase2_5/pseudopass/
  pass_composition_table.csv
  pass_ilr_step_table.csv
  pass_step_global_test.csv
  pass_step_coordinate_tests.csv
  pass_depth_spectrum_association.csv

  pass_composition_trajectory.png
  pass_ilr_shift.png
  pass_sign_consistency.png
```

所有图标题必须标记 `cross-sectional pseudo-trajectory`。

---

# 21. Task 14A — mechanism bridge

脚本：

```text
14_mechanism_bridge_error_atlas.py
```

已有：

```text
experiments/mechanism_virtual_augmentation/
```

可复用 physics/mechanism computation，但：

\[
\boxed{\text{不使用 synthetic morphology labels}}
\]

只允许对 200 个真实 process rows 计算 mechanism covariates。

## 22. mechanism provenance audit

先生成：

```text
mechanism_feature_provenance.csv
```

字段：

```text
feature_name
physical_meaning
source_code
depends_only_on_process_controls
depends_on_fixed_constants
depends_on_measured_depth
depends_on_measured_morphology
was_fitted_using_labels
fit_scope
allowed_primary
notes
```

只有：

```text
depends_on_measured_morphology = false
was_fitted_using_labels = false
```

才允许 primary。

若某 feature 使用 measured depth / morphology / residual mapping，则必须 outer-fold 内重拟合，默认 secondary。

## 23. bridge model

比较：

\[
M_0:\mathbf z\sim u_A
\]

和：

\[
M_1:\mathbf z\sim[u_A,m(u)]
\]

同一 folds，核心：

\[
\Delta Q^2_{mech}
\]

即使 >0，也只能写：

> mechanism-informed covariates provide incremental explanatory information.

不能写 E1/E2/E5 已被实验验证为真实 morphology state。

---

# 24. Task 14B — prediction error atlas

只使用 OOF predictions。

至少分析：

```text
Sq_um
ILR composition
E_DCT_8_16_frac bridge target
A2_8_16
median_depth_um reference
```

### scalar error

\[
e_i=|y_i-\hat y_i|
\]

标准化：

\[
e_i^{norm}
=
\frac{|y_i-\hat y_i|}
{IQR(y_{train(fold)})}
\]

### composition error

\[
e_i^{comp}=d_A(\mathbf p_i,\hat{\mathbf p}_i)
\]

---

## 25. process coverage density

在 standardized raw process A 中计算 leave-one-out：

```text
kNN distance
k = 5
k = 10
```

测试：

\[
Spearman(e_i,d^{proc}_{i,k})
\]

若明显正，提示 error 可能受 process-space coverage / extrapolation 影响。

---

## 26. error clustering

构造：

```text
kNN graph, k=5
```

计算 Moran's I 或等价 graph autocorrelation statistic。

Permutation：

```text
10000
```

问题：

> 高预测误差是否集中在特定 process neighborhoods？

---

## 27. error diagnostics

探索：

```text
error vs Sq
error vs depth
error vs LOCO
error vs A_consensus
error vs repair_fraction
error vs plane_rmse
error vs session_role
error vs spectral_entropy
```

只能用于定位 unresolved regions。

不能直接叫 hidden state / stochasticity。

---

## 28. error hotspot

descriptive hotspot：

```text
top 10% OOF error
```

输出：

```text
process coordinates
morphology descriptors
audit label if available
nearest process neighbors
nearest morphology neighbors
```

解释逻辑：

- sparse process region → coverage-limited candidate；
- dense region → missing-coordinate / condition-sensitive candidate；
- repair/plane flags → artifact-sensitive；
- particular spectrum family → morphology-specific failure。

---

## 29. Task 14 输出

```text
outputs/phase2_5/mechanism_bridge/
  mechanism_feature_provenance.csv
  mechanism_features_real200.csv
  mechanism_bridge_cv.csv
  mechanism_bridge_summary.csv

outputs/phase2_5/error_atlas/
  oof_error_atlas.csv
  process_density.csv
  error_density_association.csv
  error_moran_test.csv
  error_hotspots.csv

  error_process_map.png
  error_vs_density.png
  mechanism_increment.png
```

---

# 30. 主统计与 multiple testing

Primary claims 只来自：

```text
src_gkf
proc_gkf
```

共同方向。

GSS 只作稳定性，不允许某个最好的 GSS split 取代主结果。

Pass z1..z4 每 step 做 Holm correction。

Directional 多 band inference 作为一组做 Holm。

完整 24-bin \(R^2(\lambda)\) 主要作为 curve；如做逐-bin inference，采用 BH-FDR，并标记 low-mode-count bins。

---

# 31. Sensitivity arms

最低要求：

### S1 formal-only

检查 spectrum signal 是否依赖 pass/supplement 设计。

### S2 exclude artifact=yes

只去人工明确 yes，uncertain 保留。

### S3 minus top-5 LOCO

检查是否被少数 high-leverage surface 驱动。

### S4 repaired morphology

对 Sq / band RMS / spectrum 重算；depth 仍以 raw authority 为主。

### S5 radial bin count

```text
16
24
32
```

broad composition 结论不得依赖 bin count。

---

# 32. Phase 2.5 Gate

允许多个 gate 同时为 YES。

## G1 — robust spectral allocation control

如果：

1. src_gkf 与 proc_gkf 均：

\[
median\ Q^2_{Aitchison}\ge0.20
\]

2. 两套 CV 均至少 4/5 folds Q² > 0；
3. formal-only median Q² > 0.10；

则：

```text
G1 = ROBUST
```

若 multivariate 未达标，但至少一个 pre-registered ILR balance：

```text
R2 >= 0.20
4/5 folds positive
formal-only same sign
```

则：

```text
G1 = PARTIAL
```

## G2 — directional texture control

要求：

1. blind periodic-stripe set 中 `A2_8_16` 明显更高；
2. validation permutation p <= 0.05；
3. 完整 200 样本上 process→`A2_8_16` grouped-CV \(R^2\ge0.20\)，两种 GKF 同方向。

则：

```text
G2 = SUPPORTED
```

audit permutation 只验证 metric，不估计总体 prevalence。

## G3a — derived feature gain

若：

\[
median\Delta Q^2_{C-A}\ge0.05
\]

且 src_gkf、proc_gkf 均 >=4/5 folds positive：

```text
G3a = DERIVED_FEATURE_GAIN
```

## G3b — mechanism bridge

若：

\[
median\Delta Q^2_{mech}\ge0.05
\]

且两套 primary CV 均 >=4/5 folds positive：

```text
G3b = MECHANISM_BRIDGE_SUPPORTED
```

否则只是 NOT SUPPORTED，不等于机制错误。

## G4 — systematic pass spectral redistribution

若 N1→2 / N2→3 / N3→4 中至少两个 step：

```text
global exact sign-flip p <= 0.05
```

且至少一个同一 pre-registered ILR balance 在两个 step 中方向一致、Holm-adjusted p <= 0.05：

```text
G4 = SUPPORTED
```

N5→6 只作独立 check。

## G5 — prediction-error localization

若主要 unresolved target 的 OOF error：

```text
Moran permutation p <= 0.05
```

且 hotspot 在 process space 稳定聚集：

```text
G5 = LOCALIZED_FAILURE
```

再结合 error-vs-density 区分 coverage-driven 与 dense-region unresolved。

---

# 33. 可能形成的路线

### Route P — spectral allocation control

触发 G1。

### Route T — directional texture formation

触发 G2。

### Route M — mechanism-guided spectrum

触发 G3b。

### Route P-N — pass-dependent spectral redistribution

触发 G4。

名称必须保持：

```text
cross-sectional pass-dependent spectral redistribution
```

不能叫 pass dynamics。

### Route E — localized unresolved morphology

触发 G5，用于未来实验设计，不用于现在宣称 stochasticity。

---

# 34. 最低核心图

1. **Amplitude vs spectral composition**
2. **Mean normalized spectrum**
3. **Spectral predictability map \(R^2(\lambda)\)**
4. **5-part composition predicted vs observed**
5. **Directional PSD / anisotropy**
6. **A2_8_16 vs E_8_16**
7. **Process effect on ILR balances**
8. **Cross-sectional pass spectral redistribution**
9. **Mechanism feature incremental value**
10. **OOF error atlas**

---

# 35. 推荐表格

1. Phase 2.5 frozen data contract
2. Five-part composition summary
3. grouped-CV composition results
4. directional metrics predictability
5. pass exact sign-flip tests
6. mechanism bridge A vs A+M
7. error hotspots

---

# 36. Tests

新增：

```text
tests/test_phase2_5_lib.py
```

最低测试：

1. DCT broad-band fractions reproduce Phase 2 frozen fractions；
2. five-part composition sums to 1；
3. no hidden pseudocount when no zero exists；
4. ILR → inverse ILR round trip；
5. Aitchison distance == Euclidean distance in ILR；
6. Z1–Z4 basis orthonormal；
7. radial spectrum energy sum == 1；
8. FFT window normalization finite；
9. isotropic synthetic field → low \(A_2\)；
10. synthetic vertical stripe → correct 90° wave-vector / stripe relation；
11. GKF source contract；
12. GKF process contract；
13. 49/50 same process group；
14. mechanism primary features contain no measured morphology dependency；
15. OOF error row appears exactly once for GKF；
16. pass set exactly 15 matched N1–4 bases；
17. supplement exactly 10 matched N5–6 bases；
18. N4→5 analysis refuses to run；
19. B=15 exact sign flips = 32768；
20. B=10 exact sign flips = 1024；
21. blind stripe labels only used for validation subset；
22. low-mode-count bins flagged。

---

# 37. Config 建议

```yaml
random_seed: 20260903

spectrum:
  radial_log_bins: 24
  radial_bin_sensitivity: [16, 24, 32]
  low_mode_count_threshold: 20

directional:
  theta_bins: 36
  hann_window: true
  primary_band_um: [8, 16]

composition:
  zero_threshold: 1.0e-10
  replacement_delta: 1.0e-6
  replacement_sensitivity: [1.0e-8, 1.0e-6, 1.0e-5]

cv:
  n_splits: 5
  gss_repeats: 5

models:
  ridge_alpha_grid: [0.01, 0.1, 1, 10, 100]

  spline:
    degree: 3
    n_knots: 4

  extratrees:
    n_estimators: 500
    min_samples_leaf: 2

pass:
  exact_sign_flip: true

error_atlas:
  knn_k: [5, 10]
  moran_permutations: 10000
  hotspot_quantile: 0.90
```

DCT `<8` 的实际实现必须调用 frozen wavelength grid，不按 `[0,8]` 字面构造频率。

---

# 38. 执行顺序

```powershell
python experiments/phase2_5/10_build_spectral_composition.py
```

先完成 DCT reconciliation；失败则 STOP。

```powershell
python experiments/phase2_5/11_directional_spectrum.py
```

确认 anisotropy metric 与 stripe validation。

```powershell
python experiments/phase2_5/12_spectral_process_map.py
```

主结果。必须在看到结果前冻结 primary targets 与 G1/G2/G3 thresholds。

```powershell
python experiments/phase2_5/13_pseudopass_spectral_redistribution.py
```

独立回答 pass composition question。

```powershell
python experiments/phase2_5/14_mechanism_bridge_error_atlas.py
```

最后再接 mechanism features，避免 post-hoc feature engineering。

---

# 39. 允许的结果语言

> 工艺参数能够对 residual morphology 的 normalized spectral allocation 提供稳定的 out-of-sample explainability。

> 8–16 µm 相对能量份额比相同尺度的绝对 RMS 更容易由工艺条件预测。

> 结果表明 total morphology amplitude 与 spectral composition 应视为不同响应层级。

> 某些 process combinations 与方向纹理强度存在稳定统计关系。

> mechanism-informed covariates 对 measured spectral composition 提供了增量 out-of-sample explanatory value。

---

# 40. 禁止的结果语言

> 工艺完全决定了表面形貌。

> 高频部分是随机噪声。

> 8–16 µm band 就是热积累机制。

> PSD 证明发生了相变。

> 周期条纹证明存在新的加工 regime。

> exact-repeat sentinel 证明整个工艺是 deterministic。

> mechanism feature 增益证明 E1/E2/E5 是真实材料状态。

> pass composition shift 就是同一表面的动力学演化。

---

# 41. 最终必须回答的 12 个问题

1. 五段 spectral composition 是否比 absolute band RMS 更可预测？
2. `E_DCT_8_16_frac` 是单 band 特例，还是完整 spectrum redistribution 的一部分？
3. 哪个 ILR balance 最容易被工艺解释？
4. raw controls A 与 hybrid C 谁更稳健？
5. nonlinear / interaction 是否真正改善 composition prediction？
6. 8–16 µm energy 是否主要与 directionality / stripe metric 相关？
7. blind periodic-stripe phenotype 能否被自动 spectral metric 客观区分？
8. pass count 是否对应跨条件一致的 spectral redistribution？
9. N5→6 是否与 N1–4 的方向具有描述性一致性？
10. mechanism model 的 label-free covariates 是否增加真实 measured spectrum 的解释力？
11. amplitude / composition prediction error 是否集中在特定 process region？
12. 下一阶段最应该继续研究 spectrum control、directional texture、mechanism bridge，还是 coverage/missing-variable 问题？

---

# 42. Phase 2.5 完成后的路线判断

如果 G1 robust：

\[
\boxed{
\text{process-controlled multiscale spectral allocation}
}
\]

如果 G1 + G2：

\[
\boxed{
\text{spectral allocation + directional texture formation}
}
\]

优先研究 8–16 µm selective texture。

如果 G3b：

\[
\boxed{
\text{mechanism-guided spectral morphology representation}
}
\]

此时再讨论 physics-guided machine learning 才更有理由。

如果 G1 不成立，而 depth 仍强、amplitude 与 normalized spectrum 都弱：

> 当前五个 process controls 对 residual morphology 的解释空间不足。

此时未来才提高：

```text
new process observables
repeatability matrix
hidden-state candidate
```

优先级。

---

# 43. 本阶段最终科学目标

Phase 2.5 不以找到“最好的预测模型”为终点。

目标是建立：

\[
\boxed{
\text{Process}
\rightarrow
\begin{cases}
\text{Depth}\\
\text{Residual amplitude}\\
\text{Spectral allocation}\\
\text{Directional texture}\\
\text{Spatial realization}
\end{cases}
}
\]

并定量回答：

\[
\boxed{
\text{哪些 morphology observables 真正受到工艺稳定控制，哪些没有？}
}
\]

如果 Phase 2.5 成功，后续主线应由：

\[
\boxed{
\text{observable-dependent process controllability}
}
\]

决定，而不是由 Mamba、latent state 或其他模型架构决定。
