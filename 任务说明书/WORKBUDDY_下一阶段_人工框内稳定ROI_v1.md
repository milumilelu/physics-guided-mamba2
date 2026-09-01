# WorkBuddy 下一阶段：人工框内稳定 ROI v1

制定日期：2026-09-01  
方法标识：`manual_internal_roi_v1`  
证据等级：Level 3  
状态：`FAST_ROUTE_IMPLEMENTED`（200/200 已导出；人工审批文件保持 PENDING）

> **2026-09-01 快速实施修订**：本任务已由 Codex 接手。为尽快形成可用预处理
> 数据，不再等待逐阶段人工审批。先对全体样本比较 60–120 µm 嵌套中心区域，
> 80×80 µm 在保留面积与边缘稳定性之间形成拐点（边缘/中心梯度比中位数
> 1.007、P90 1.124；边缘-中心高度差 P90 为 0.462 µm），因此正式统一 ROI
> 冻结为 80×80 µm、0.5 µm/pixel。实现入口为
> `scripts/22_extract_stable_roi_fast.py`，一键入口仍为
> `scripts/32_run_manual_internal_roi_v1.py`。以下完整工作包保留为方法审计和后续
> 增强路线；快速路线输出 raw/repaired/valid/repair mask、profiles、metrics 和 QA。

## 1. 本阶段到底做什么

全部 200 个矩形槽的真实加工边界已经由同一位标注者完成。下一阶段不再重新
寻找加工区，也不再要求每槽都能导出以中心为基准的 260 µm 公共视场，而是：

```text
冻结人工四边
  → 每幅 measurement 用所有加工框之外的表面拟合同一个背景平面
  → 在框内识别并审计共聚焦锥坑伪影
  → 从四边向内测量边界影响距离
  → 冻结全体样本共用的稳定 ROI
  → 同时导出原始稳定 ROI、修复稳定 ROI、mask 和 QA
```

本阶段只建立稳定形貌输入，不提取最终形貌特征，不做 PCA、预测模型、虚拟
数据增强或机理解释。

旧 `manual_v1` 260 µm Phase A 结果是方法演进证据，必须继续保持
`BLOCKED`，不得删除、覆盖或改写成 PASS。新方法是正式的替代分支，不是假装
旧门禁已经通过。

## 2. 固定事实与输入合同

### 2.1 数据规模

- `zro2_120_formal`：120 个 measurement，120 个槽；
- `zro2_60_pass`：30 个 measurement，每次两个槽，共 60 个槽；
- `zro2_20_supplement`：10 个 measurement，每次两个槽，共 20 个槽；
- 合计 160 个 measurement、200 个槽；
- paired measurement 中槽的 sample id 与数据名数字一致，按 `+x=右、+y=下`
  解释左右顺序。

### 2.2 冻结输入

权威输入只能是：

```text
config/height_source_manifest.csv
config/manual_orientation.yaml
config/manual_registration_v1.yaml
outputs/rectangle_registration/registration/manual_v1/
  manual_four_edge_validation_frozen.csv
```

要求：

- 使用 `annotator_a_*` 字段；不得构造 annotator B；
- frozen CSV 必须恰有 200 个唯一 `(session_id, sample_id)`，且均为 complete；
- 在任何计算前记录上述文件的 SHA-256；运行过程中发现哈希变化立即 STOP；
- session 角度、D4 和坐标方向沿用已冻结值；
- 人工四边是实际形貌边界，保留真实物理宽高，不强制缩放为 200×200 µm；
- v2–v7 的中心、region/edge 分数仅作历史 QA，不参与主计算；
- CAG/官方 CSV、人工标注和旧输出全部只读。

## 3. 关键科学边界

### 3.1 背景平面按 measurement 拟合

同一个 paired measurement 的两个槽共享原始高度图，因此只能拟合一个背景
平面。将该 measurement 的全部人工框逆变换为原始坐标四边形，各向外膨胀
固定缓冲后取并集，排除该并集，再用剩余真实未加工表面拟合稳健平面。

锥坑修复不得参与背景平面拟合。若加工框外没有足够真实参考面，则该
measurement 的绝对深度状态为 STOP；禁止用框内表面补平或按槽分别拟合。

### 3.2 锥坑修复必须适配矩形面加工

仓库现有 `extract_zro2_single_line.py::repair_conical_dropouts` 是单线加工算法，
假定激光线沿图像列方向，并通过单线 corridor 限制修复。它及其历史专利路径
不能直接作为本阶段实现。

本阶段必须实现矩形面加工专用版本，并满足：

1. 只在人工框内部寻找候选，边界保护带内禁止修复；
2. 候选必须是紧凑的局部向下失真，不得把连续扫描沟槽、周期纹理、边缘陡壁
   或大面积低区当成锥坑；
3. 候选判定使用二维形状约束和局部稳健基面，不能依赖固定图像行列方向；
4. 只能向上替换已接受候选，且每个像素都保留 repair mask；
5. 原始调平图永远保留，修复图是并行派生物，不能覆盖原始图；
6. 主 margin 计算在原始图上进行，并把 repair mask 像素视为缺失。这样修复
   模型不会反过来决定 stable ROI；
7. 同时在修复图上计算 sensitivity margin，报告 raw/masked 与 repaired 差异；
8. 未通过人工锥坑门禁前，禁止全量采用修复高度进入下游特征。

这一步的目标是消除已识别的测量失真，不是把真实加工表面“修平”。修复量
属于模型估计，证据等级仍是 Level 3。

### 3.3 stable ROI 的精确定义

对样本 `i` 的四边影响距离记为
`m_i,left/right/top/bottom`。对每个方向，在总体与样本数不少于 8 的预定义
工艺分层内计算 Q90；取所有合格分层 Q90 的最大值，再向上取整到 10 µm：

```text
M_left  = ceil10(max(Q90_overall, Q90_each_eligible_stratum))
M_right = ...
M_top   = ...
M_bottom= ...
Mx = max(M_left, M_right)
My = max(M_top, M_bottom)
```

所有样本共用、关于人工中心对称的 primary ROI 半宽为：

```text
A90 = floor10(min_i(manual_width_i / 2)  - Mx)
B90 = floor10(min_i(manual_height_i / 2) - My)
ROI90 = [-A90, A90] × [-B90, B90]
```

Q95 sensitivity ROI 同理。若 `A<=0`、`B<=0`，或所得 ROI 不在任一人工框内部，
立即 BLOCKED。禁止逐样本缩框、插值 margin 或静默删除难例。

## 4. 新增冻结配置

新增 `config/manual_internal_roi_v1.yaml`。第一次正式候选检测前写入并记录
SHA-256；看到全量结果后不得修改同一个方法标识下的阈值。

最低配置合同：

```yaml
method: manual_internal_roi_v1
evidence_level: 3

inputs:
  height_manifest: config/height_source_manifest.csv
  annotation_csv: outputs/rectangle_registration/registration/manual_v1/manual_four_edge_validation_frozen.csv
  annotator: annotator_a

measurement_level_plane:
  processing_box_buffer_um: 20.0
  minimum_reference_valid_fraction: 0.20
  minimum_reference_x_span_fraction: 0.60
  minimum_reference_y_span_fraction: 0.60
  minimum_reference_quadrants: 3
  robust_sigma_low: 3.0
  robust_sigma_high: 3.0
  max_iterations: 8

conical_dropout:
  enabled: true
  mode: audit_then_apply
  boundary_protection_um: 15.0
  preserve_raw: true
  emit_repair_mask: true
  primary_margin_uses_repaired_height: false
  require_manual_approval: true

edge_profiles:
  smoothing_sigma_um: 1.0
  corner_exclusion_um: 30.0
  center_reference_halfwidth_um: 20.0
  minimum_tangential_valid_um: 80.0
  sampling_step_um: 0.5

primary_margin:
  signal: raw_height_with_repair_candidates_masked
  criterion: gradient_mad_3x_persistence_5um
  gradient_sigma_threshold: 3.0
  minimum_persistence_um: 5.0

sensitivity_margin:
  signal: repaired_height
  criterion: gradient_mad_5x_persistence_8um
  gradient_sigma_threshold: 5.0
  minimum_persistence_um: 8.0

stable_roi:
  primary_coverage_quantile: 0.90
  sensitivity_coverage_quantile: 0.95
  rounding_um: 10.0
  symmetric_about_manual_center: true
  shared_across_all_samples: true
  minimum_valid_margin_fraction_overall: 0.90
  minimum_valid_margin_fraction_each_session: 0.90
  minimum_stratum_size: 8

forbid:
  - overwrite_raw_height
  - reuse_single_line_cone_repair_without_validation
  - repair_true_scan_texture
  - samplewise_roi_shrink
  - silent_sample_drop
  - margin_imputation
  - automatic_method_fallback
  - use_internal_surface_as_unprocessed_absolute_reference
```

锥坑检测的尺寸、形状、阈值参数不能照抄单线算法。先由 WP-C0 用实际矩形
数据生成诊断，再冻结到该 YAML。冻结前输出必须使用 `pilot` 标签，不能混入
正式结果。

## 5. 实施工作包

### WP-I0：输入冻结与新路线 manifest

新增 `scripts/22_freeze_manual_internal_roi_inputs.py`：

- 校验 160 measurement / 200 槽映射；
- 校验 paired 左右槽 id 与 manifest；
- 校验 200 条人工标注完整性和哈希；
- 生成只属于新路线的 append-only stage manifest；
- 记录 Python、依赖锁、Git 状态和全部权威输入哈希。

不得修改旧 `manual_v1` manifest 或 Phase A 审批文件。

### WP-I1：测量级 final plane

新增 `scripts/23_fit_measurement_level_plane.py`。

每个 measurement：读取权威高度与原始 valid mask；将全部人工框逆变换为原始
坐标四边形并膨胀 20 µm；在排除区之外稳健拟合一次平面；检查有效比例、x/y
跨度、象限数、残差尺度和 retained fraction。paired measurement 的两个槽必须
引用同一个 `plane_id` 和同一组系数。

输出：

```text
outputs/rectangle_registration/manual_internal_roi_v1/
  metrics/measurement_level_plane.csv
  planes/<measurement_key>.npz
  qa/measurement_level_plane/
```

失败必须显式记录。平面失败的 measurement 仍进入相对形貌/有效性审计，但
不得输出绝对深度特征。

### WP-C0：矩形面锥坑适配与人工门禁

新增通用模块 `src/conical_dropout.py`、单元测试和
`scripts/24_audit_conical_dropout.py`。

先按固定规则抽取覆盖三个 session、不同深度/纹理/边界状态的 pilot；对每个
候选输出原始图、候选 mask、局部基面、修复图、差值图、横纵剖面以及组件表。
至少报告：组件面积、物理长宽、长宽比、圆度/紧凑度、最大/均值修正量、距人工
边界距离、修复像素比例和是否与周期沟槽相连。

生成：

```text
outputs/rectangle_registration/manual_internal_roi_v1/
  cone_repair_pilot/
  CONE_REPAIR_APPROVAL.md
```

脚本只能写 `Status: PENDING` 或 `BLOCKED`。研究者确认没有系统性填平真实纹理
并手工改为 APPROVED 后，才允许全量修复；否则 runner 必须在此停止。

### WP-I2：全量锥坑检测与双轨高度

新增 `scripts/25_apply_conical_dropout.py`。

对 200 个槽保存：

```text
height_levelled_raw
height_levelled_repaired
valid_mask
repair_mask
component_table
repair_metrics
```

每个派生物记录源高度、人工标注、plane、锥坑配置及审批文件哈希。全量统计若
越过冻结的异常门限，只能 BLOCKED，不能运行后调参重跑同一方法版本。

### WP-I3：框内四边 profiles

新增 `scripts/26_extract_manual_edge_profiles.py`。

以人工中心为局部原点，不缩放人工框：

- 左：`d=u-left_u`；右：`d=right_u-u`；
- 上：`d=v-top_v`；下：`d=bottom_v-v`；
- 沿边使用排除角部后的中央连续有效段；
- 同时保存 raw/masked 与 repaired profile；
- 保存绝对高度、中心相对高度、平滑高度、法向梯度、有效样本数、valid 覆盖率
  和 repair-mask 覆盖率。

不得生成或依赖 260 µm `H_reg`。

### WP-I4：每边影响距离

新增 `scripts/27_estimate_internal_edge_margins.py`。

中心 `±20 µm` 只用于估计框内稳定梯度噪声，不得称作未加工参考面。由边界
向内寻找梯度回到冻结阈值内并持续规定距离的第一个位置。输出四边 primary、
sensitivity margin，以及状态、失败原因、profile 有效率和锥坑覆盖率。

单个尖峰不能决定 margin；无法检测时标记 REVIEW/STOP，不插值、不借用其他
样本值。

### WP-I5：解析全体统一 stable ROI

新增 `scripts/28_resolve_shared_stable_roi.py`，严格执行第 3.3 节公式。

预定义分层只能来自已有实验设计字段：session、pulse width、frequency、hatch、
pass、velocity。报告总体和所有分层的样本量、有效率、Q90/Q95 与置信区间；
不得事后挑选有利分层。输出 primary Q90 与 sensitivity Q95 两套冻结定义。

### WP-I6：直接导出 stable ROI

新增 `scripts/29_export_internal_stable_roi.py`。

直接从 measurement-level 高度按人工中心和统一 ROI 旋正、mask-aware 重采样。
每个槽同时导出：

```text
registered/H_stable_q90_raw/
registered/H_stable_q90_repaired/
registered/H_stable_q95_raw/
registered/H_stable_q95_repaired/
registered/masks/
metrics/stable_roi_metrics.csv
```

其中 raw 产品必须同时附带 `repair_mask`，便于下游选择排除而不是填补。20 补充
pass 直接进入此链，不以 260 µm `H_reg` 为前置条件。

### WP-I7：QA、审计与停止点

新增：

```text
scripts/30_generate_internal_roi_qa.py
scripts/31_final_manual_internal_roi_audit.py
scripts/32_run_manual_internal_roi_v1.py
```

每槽 QA 至少显示人工框、四条 profile、primary/sensitivity margin、统一 Q90/Q95
ROI、raw/repaired stable height、valid mask、repair mask、差值图和 plane 状态。

统一 ROI 边界仍超过冻结梯度阈值的样本必须保留，stable feature 状态记为
missing；不得局部修补或按样本缩框。报告 missingness 是否集中在工艺层，任一
工艺层 invalid 比例超过 30% 必须告警。

## 6. 一键执行顺序

`scripts/32_run_manual_internal_roi_v1.py --fresh-manifest` 必须按以下顺序执行：

```text
固定 Python 环境检查
→ 全部单元测试
→ WP-I0 输入冻结
→ WP-I1 measurement plane
→ WP-C0 锥坑 pilot 与审批检查
→ WP-I2 全量锥坑双轨高度
→ WP-I3 profiles
→ WP-I4 margins
→ WP-I5 shared ROI
→ WP-I6 export
→ WP-I7 QA 与最终审计
```

`--fresh-manifest` 只能原子重建
`outputs/rectangle_registration/manual_internal_roi_v1/run_manifest.json`，不得删除或
覆盖旧路线输出。任一步非零退出立即停止，后续阶段不得继续。

第一次执行预计停在锥坑人工审批；批准后从同一冻结配置继续。最终必须停止在：

```text
outputs/rectangle_registration/manual_internal_roi_v1/
  STABLE_ROI_APPROVAL.md
Status: PENDING 或 BLOCKED
```

脚本不得写 APPROVED/PASS。批准前不得进入形貌特征提取或预测建模。

## 7. 最终验收清单

- 200/200 人工框完整，权威哈希与运行前一致；
- 160/160 measurement 均有明确 plane 状态；
- paired measurement 共用同一个 plane；
- 锥坑算法不是未经验证的单线算法复用；
- 原始高度未被覆盖，raw/repaired/valid/repair mask 一一对应；
- 四边 profile 坐标方向和 µm 距离正确；
- primary margin 未使用修复后的高度决定 ROI；
- 阈值和 Q90/Q95 规则未在看到全量结果后修改；
- Q90/Q95 ROI 位于全部人工框内部；
- 没有逐样本 ROI、margin 插值、自动 fallback 或静默删样本；
- 20 补充 pass 全部进入同一链；
- NPZ、CSV、QA、审批文件和 manifest 哈希闭合；
- 自动测试覆盖平面共享、坐标逆变换、锥坑防误修、mask 传播、分位数公式、
  paired 映射和 runner 停止行为。

## 8. 必须写入最终报告的限制

- 人工边界和锥坑判定均为 Level 3，不是绝对真值；
- margin 是形貌 profile 恢复稳定的位置，不是热影响区的绝对物理边界；
- 修复高度是局部模型估计，必须与原始高度和 repair mask 一起解释；
- 框内平缓不等于没有真实扫描纹理，周期沟槽不得为了“看起来平”而滤除；
- measurement 外部参考不足时，不能报告可靠绝对去除深度；
- 本阶段没有实施或验证虚拟数据增强。
