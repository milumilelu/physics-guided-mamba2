# Phase 2.6 研究任务说明：单线扫描尺度溯源与 8–16 µm 形貌特征来源验证

> 建议路径：`experiments/phase2_6/Phase2.6_单线扫描尺度溯源_研究任务说明.md`  
> 建议状态：`DRAFT_FOR_REVIEW`  
> 阶段定位：**利用已有单线扫描数据，对 Phase 2.5 中稳定出现的 8–16 µm 谱分配与方向纹理信号进行几何尺度溯源，区分“单轨有效加工宽度”“hatch 周期/谐波”“线宽—线间距 overlap 复合尺度”三类竞争解释。**

---

## 0. 背景与问题来源

Phase 2.5 已经得到两个稳定结果：

1. **Route P：谱分配控制**
   - residual morphology 的 normalized spectral composition 可由工艺参数稳定预测；
   - 其中 `<16 µm vs >=16 µm`、`<8 µm vs 8–16 µm` 是最强的 ILR balance；
   - 8–16 µm 不再是单一偶然 band，而属于稳定的谱重分配结构。

2. **Route T：方向纹理形成**
   - `A2_8_16` 与盲评 `periodic stripe` 表型高度一致；
   - 8–16 µm 的方向各向异性与角熵可由工艺参数稳定预测；
   - `p_8_16` 与 `A2_8_16` 中度相关，但条件化后两者仍保留独立工艺信息。

因此，当前核心科学问题已经从：

> “8–16 µm 是否可预测？”

转为：

> **“8–16 µm 这个形貌尺度究竟对应什么实际加工几何？”**

已有一个重要候选解释：

\[
8-16~\mu m \approx \text{单线扫描的有效横向加工尺度}
\]

但这一解释必须与至少两个竞争假设同时检验，而不能直接接受。

---

## 1. 核心研究问题

### RQ1：单线扫描的有效加工宽度是否主要落在 8–16 µm？

对每一组单线工艺条件，测量：

\[
W_{\rm line}
\]

并回答其主要分布是否落在 8–16 µm，以及 \(W_{\rm line}\) 是否随：

\[
(\tau,f,v,N)
\]

发生系统变化。

### RQ2：Phase 2.5 的 8–16 µm 特征更接近“线宽”还是“线间距/周期”？

比较：

\[
\lambda_{\rm morph}
\]

与：

\[
W_{\rm line},\;h,\;2h,\;3h
\]

之间的关系。

竞争假设：

- **H1 单轨宽度假设**：\(\lambda_{\rm morph}\approx W_{\rm line}\)
- **H2 hatch 周期/谐波假设**：\(\lambda_{\rm morph}\approx h,2h,3h\)
- **H3 overlap 复合尺度假设**：\(\lambda_{\rm morph}=F(W_{\rm line},h)\)，重点控制量
  \[
  \eta_h=\frac{W_{\rm line}}{h}
  \]

### RQ3：方向纹理 Route T 是否与单线几何尺度一致？

如果 scan/hatch orientation provenance 可恢复，检验：

\[
\Delta\theta=\theta_{\rm stripe}-\theta_{\rm scan/hatch}
\]

回答 8–16 µm 方向纹理是否与加工轨迹方向存在稳定几何对应。

若方向 provenance 不可靠，则仅保留 `image-frame orientation`，不做 scan-relative / hatch-relative 物理解释。

---

## 2. 研究假设

### H1 — Intrinsic single-track scale

若 8–16 µm 主要由单轨横向作用尺度决定，则预期：

1. \(W_{\rm line}\) 主要落在 8–16 µm；
2. 不同 hatch spacing 下，\(\lambda_{\rm morph}\) 更接近 \(W_{\rm line}\) 而不是 \(h\)；
3. 将 \(W_{\rm line}\) 引入后，`p_8_16` / `A2_8_16` 的低复杂度解释明显增强。

### H2 — Hatch-periodic scale

若 8–16 µm 主要来自线阵列周期，则预期：

\[
\lambda_{\rm morph}/h
\]

在 1、2、3 等简单整数附近聚集。

### H3 — Overlap-controlled composite scale

若单线宽度和 hatch spacing 共同决定最终纹理，则预期：

\[
\eta_h=W_{\rm line}/h
\]

比 \(W_{\rm line}\) 或 \(h\) 单独更能解释：

- `p_8_16`
- `A2_8_16`
- `angular_entropy_8_16`
- spectral centroid

但本阶段不预设离散 regime。

---

## 3. 明确不做什么

本阶段不：

- 直接把 8–16 µm 等同于单线宽度；
- 只凭 nominal spot diameter 推断真实加工宽度；
- 只用一条中央截面代表整条单线；
- 把 line width 与 Fourier wavelength 混为同一概念；
- 把整数倍关系自动解释成物理谐波机制；
- 根据少量单线数据证明因果机制；
- 把结果直接命名为热积累、相变或脆性断裂；
- 用复杂网络作为核心证据；
- 用单线数据替代 Phase 2.5 的真实面形貌数据。

---

## 4. 数据与 provenance audit

正式分析前生成：

```text
single_line_manifest.csv
```

至少包含：

```text
single_line_id
source_file
session_id
measurement_id
pulse_duration_fs
frequency_kHz
velocity_mm_s
pass_count
power_W_or_proxy
pixel_size_um
line_scan_direction
measurement_orientation
processing_date_or_batch
height_data_type
background_correction_status
valid_mask_status
notes
```

必须核查：

1. 是否与 Phase 2/2.5 同一激光系统；
2. 是否使用相同或可换算的实际功率条件；
3. pulse duration / frequency / velocity / pass 单位是否一致；
4. 横向像素尺寸是否可信；
5. 是否包含完整槽截面；
6. 是否存在裁剪导致的线宽截断；
7. 是否有多个重复位置；
8. 是否存在背景倾斜/平面问题；
9. 是否能恢复 scan direction 与 image orientation。

无法确认 provenance 的样本可以进入描述性分析，但不得进入跨数据集定量 bridge。

---

## 5. 单线宽度定义

不能只定义一个“看起来合理”的线宽。至少保留三类宽度。

### W1 — threshold width

设背景高度 \(z_{bg}\)、槽底 \(z_{min}\)，定义：

\[
d_n(x)=\frac{z_{bg}-z(x)}{z_{bg}-z_{min}}
\]

保存：

- `W20`
- `W50`
- `W80`

其中 \(W_q\) 表示 \(d_n(x)\ge q\) 的横向宽度。

主 line width 建议：

\[
W_{\rm line}=W50
\]

### W2 — equivalent-area width

\[
A_{\rm remove}=\int[z_{bg}-z(x)]_+dx
\]

\[
D_{max}=z_{bg}-z_{min}
\]

\[
W_{eq}=\frac{A_{\rm remove}}{D_{max}}
\]

用于避免复杂侧壁使 threshold width 失真。

### W3 — affected width

若存在显著 ridge / uplift / side-affected region，可定义 `W_affected`，但必须预注册阈值，且只作为 secondary descriptor。

---

## 6. 单线附加几何描述符

建议同时计算：

```text
max_depth_um
cross_section_area_um2
left_slope
right_slope
edge_asymmetry
ridge_height_left
ridge_height_right
ridge_separation
profile_skewness
```

用于判断 8–16 µm 更接近“主槽宽度”还是“槽+边缘影响区”的总横向尺度。

---

## 7. 多截面测量

若单线数据是二维高度图，不得只取中央截面。

沿扫描方向取至少：

```text
n_cross_sections >= 20
```

或所有有效横截面。

每条线保存：

\[
median(W50), IQR(W50), P10, P90
\]

以及：

\[
CV_W=IQR(W50)/median(W50)
\]

作为单线宽度稳定性描述符。

---

## 8. Task SL-01 — 单线数据审计与宽度提取

建议脚本：

```text
experiments/phase2_6/
  01_build_single_line_manifest.py
  02_extract_single_line_geometry.py
```

输出：

```text
outputs/phase2_6/single_line/
  single_line_manifest.csv
  single_line_geometry.csv
  cross_section_widths.csv
  qa_montages/
```

QA montage 至少包括：

1. absolute height；
2. background-corrected profile；
3. W20/W50/W80；
4. equivalent area；
5. 多横截面 width distribution；
6. mask / artifact 标记。

人工只填：

```text
usable
uncertain
reject_geometry
```

禁止根据是否接近 8–16 µm 决定是否保留。

---

## 9. Task SL-02 — 单线宽度的工艺依赖

研究：

\[
W_{\rm line}=F(\tau,f,v,N)
\]

若实际单线数据缺少部分变量，则按真实可用子集分析。

第一批模型：

```text
Ridge
Spline/GAM-like
ExtraTrees sensitivity
```

重点输出：

- 真实 line-width scale；
- 分布范围；
- 工艺响应曲线；
- 是否稳定落在 8–16 µm。

---

## 10. Task SL-03 — 单线尺度与 Phase 2.5 建桥

建立：

\[
\hat W_{\rm line}(u)
\]

对 Phase 2.5 的 200 个面形貌样本产生：

```text
single_line_width_proxy_um
```

注意：

> 这是由单线数据推断的单轨宽度代理，不是从面形貌中直接量得的线宽。

分析：

\[
p_{8-16} \;vs\; \hat W_{\rm line}
\]

\[
A2_{8-16} \;vs\; \hat W_{\rm line}
\]

\[
H_{\theta,8-16} \;vs\; \hat W_{\rm line}
\]

以及：

\[
\lambda_* \;vs\; \hat W_{\rm line}
\]

---

## 11. 竞争尺度变量

对每个面形貌样本构造：

```text
W_hat
h
2h
3h
W_hat / h
h / W_hat
abs(lambda_star - W_hat)
abs(lambda_star - h)
abs(lambda_star - 2h)
```

\(\lambda_*\) 主定义不建议直接使用单个 unstable peak。

优先：

- 4–32 µm 区间的 energy-weighted wavelength；
- 或 local spectral centroid。

`lambda_peak` 只作 sensitivity。

---

## 12. 关键模型比较

对：

```text
p_8_16
ilr_z2
A2_8_16
angular_entropy_8_16
```

比较：

### M0 原始工艺

\[
Y\sim u
\]

### M1 单线宽度

\[
Y\sim[u,\hat W]
\]

### M2 geometry-only hatch

\[
Y\sim h
\]

### M3 overlap

\[
Y\sim[u,\hat W,\hat W/h]
\]

核心报告：

\[
\Delta R^2_{CV}
\]

composition 可用：

\[
\Delta Q^2_{Aitchison}
\]

注意：若 \(\hat W\) 完全由 process controls 拟合，它不增加实验信息。这里检验的是“几何尺度表征是否让关系更简单/更可解释”，不是证明增加了新信息。

---

## 13. Task SL-04 — characteristic wavelength ratio test

定义：

\[
r_W=\frac{\lambda_*}{\hat W_{\rm line}}
\]

\[
r_h=\frac{\lambda_*}{h}
\]

判读：

- \(r_W\approx1\) 且 \(r_h\) 漂移 → 支持 H1；
- \(r_h\approx1,2,3\) 聚集 → 支持 H2；
- 单独 \(r_W/r_h\) 均不稳，但 \(W/h\) 能解释 P/T → 支持 H3。

不允许用无约束 clustering 作为主证据。

可以定义：

\[
d_{int}=\min_{m\in\{1,2,3\}}|r_h-m|
\]

并与 shuffled-h null 比较。

---

## 14. Task SL-05 — 方向几何验证

仅当 scan/hatch orientation provenance 有效时运行。

计算：

\[
\Delta\theta=\min(|\theta_1-\theta_2|,180^\circ-|\theta_1-\theta_2|)
\]

测试 stripe 是否聚集在：

```text
0°
90°
```

附近。

如果方向 provenance 缺失：

```text
SL-05 = NOT_APPLICABLE
```

不影响 H1/H2/H3 主判断。

---

## 15. CV 与防泄漏

如果同一条 single line 有多个横截面：

```text
必须 group by single_line_id
```

不能把不同横截面随机拆到 train/test。

Phase 2.5 端仍沿用：

```text
src_gkf
proc_gkf
```

不改变原 CV contract。

如果 \(W_hat\) 需要建模生成，必须保证对 Phase 2.5 test fold 使用的 width proxy 不从其 morphology label 获取任何信息。

---

## 16. Gate

### G-SL1 — line-width scale alignment

若：

1. `W50` 总体中位数在 8–16 µm；
2. 至少 50% 可用单线工艺条件的 median W50 位于 8–16 µm；
3. `W_eq` 同方向；

则：

```text
G-SL1 = SUPPORTED
```

否则为 `PARTIAL` 或 `NOT_SUPPORTED`。

### G-SL2 — hatch periodicity

若：

\[
\lambda_*/h
\]

明显聚集在 1/2/3 中至少一个尺度附近，且比 shuffled-h null 更集中：

```text
G-SL2 = SUPPORTED
```

### G-SL3 — overlap control

若：

\[
W/h
\]

对 Route P/T target 的 grouped-CV 或低复杂度模型提供稳定增益：

```text
G-SL3 = SUPPORTED
```

建议增益门槛在 formal 运行前冻结，例如：

\[
\Delta R^2 \ge 0.05
\]

或：

\[
\Delta Q^2 \ge 0.05
\]

### G-SL4 — direction alignment

若方向 provenance 有效且 stripe 与 scan/hatch 方向存在显著 0°/90° 聚集：

```text
G-SL4 = SUPPORTED
```

否则 `NOT_SUPPORTED` 或 `NOT_APPLICABLE`。

---

## 17. 最终判读

### A. G-SL1 YES，G-SL2 NO

倾向：

> 8–16 µm 是与单轨有效横向加工尺度一致的 intrinsic processing scale。

### B. G-SL1 NO/PARTIAL，G-SL2 YES

倾向：

> 8–16 µm 主要反映 hatch line array 的周期/谐波结构。

### C. G-SL1 YES，G-SL2 YES，G-SL3 YES

倾向：

> 单线横向作用尺度与 hatch pitch 处于相近数量级，最终 8–16 µm morphology 由 overlap geometry 共同决定。

### D. 三者均不支持

说明：

> 8–16 µm 不是简单几何尺度映射，需要进一步研究材料响应、裂纹、剥落、再沉积等复杂结构形成过程。

不能为了保留原假设强行选择 H1/H2/H3。

---

## 18. 输出

```text
outputs/phase2_6/
  single_line/
    single_line_manifest.csv
    single_line_geometry.csv
    cross_section_widths.csv

  scale_bridge/
    line_width_process_model.csv
    morphology_scale_match.csv
    lambda_over_width.csv
    lambda_over_hatch.csv
    overlap_metrics.csv

  model_compare/
    width_bridge_cv.csv
    overlap_bridge_cv.csv

  orientation/
    stripe_scan_alignment.csv

  summary/
    phase2_6_gate_eval.md
```

---

## 19. 最低核心图

1. 单线典型截面 + W20/W50/W80/W_eq；
2. W50 随工艺变量的变化；
3. 单线宽度分布与 8–16 µm band 对照；
4. \(\lambda_* vs W_{line}\)；
5. \(\lambda_*/h\) 分布并标记 1/2/3；
6. \(W/h\) vs `p_8_16` / `A2_8_16` / `angular_entropy_8_16`；
7. 若方向 provenance 有效，绘制 stripe–scan/hatch 相对角 polar histogram。

---

## 20. 最低测试

1. 横向坐标单位为 µm；
2. W20 ≥ W50 ≥ W80；
3. W_eq > 0；
4. 多横截面不产生 train/test 泄漏；
5. `single_line_id` grouped CV 正确；
6. 8–16 band 复用 Phase 2.5；
7. `lambda_star` 不从低-mode-count coarse bin 强行取峰；
8. single-line 数据中 hatch=NA 不参与 lambda/h；
9. N4→5 仍不进入 Phase 2.5 bridge；
10. scan-relative angle 仅在 provenance_valid=true 时计算；
11. W_hat 不使用 Phase 2.5 morphology label；
12. H1/H2/H3 gate threshold 在 formal 运行前冻结。

---

## 21. 建议执行顺序

1. 只做 single-line manifest + QA；
2. 冻结 W20/W50/W80/W_eq；
3. 完成全部单线 geometry extraction；
4. **先回答“单线宽度到底是多少”**；
5. 再与 Phase 2.5 的 `p_8_16` / `A2_8_16` / `angular_entropy_8_16` 建桥；
6. 最后比较 W、h、2h、W/h 四类解释。

禁止看到 8–16 结果后再改 width definition。

---

## 22. 本阶段最终必须回答的 8 个问题

1. 单线有效加工宽度的真实范围是多少？
2. 8–16 µm 是否覆盖单线宽度的主要分布？
3. 单线宽度主要由哪些工艺变量控制？
4. 面形貌 characteristic wavelength 更接近 W、h 还是 2h？
5. W/h 是否比单独 W 或 h 更能解释谱组成？
6. Route P 与 Route T 是否都随 single-line overlap geometry 改变？
7. 条纹方向是否与 scan/hatch direction 存在稳定几何对应？
8. 最终 8–16 µm 应解释为 intrinsic track scale、hatch periodicity，还是 overlap composite scale？

---

## 23. 推荐的后续主问题表述

如果 Phase 2.6 支持几何尺度桥接，可以将项目主问题进一步收敛为：

> **超快激光单轨横向作用尺度与线间填充几何如何共同决定氧化锆加工表面的多尺度谱能量分配与方向纹理形成？**

数学链条：

\[
(\tau,f,v,N)\rightarrow W_{line}
\]

随后：

\[
(W_{line},h,W_{line}/h)
\rightarrow
[q(\lambda),q(\lambda,\theta)]
\]

这将把 Phase 2.5 的“统计可预测性”推进到：

\[
\boxed{\text{具体加工几何尺度解释}}
\]

而不是继续停留在黑箱 process→morphology mapping。
