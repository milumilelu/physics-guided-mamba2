# Codex 任务说明：超快激光矩形加工区全局配准与稳定 ROI 提取

## 1. 任务背景

本项目处理超快激光矩形区域加工后的激光共聚焦三维表面形貌数据。每个实验条件均为一次性加工、一次性测量。矩形理论加工尺寸为 **200 μm × 200 μm**，扫描采用**弓字形（boustrophedon）填充**。

已知实验与设备特征：

- 各实验的矩形加工区域理论尺寸固定为 200 μm × 200 μm。
- 同一批实验加工与共聚焦测量的整体装夹/测量旋转偏置理论上基本固定。
- 测量文件之间仍可能存在少量平移偏差，因此矩形中心应允许逐文件微调。
- 弓字形扫描在两端换向处存在加速/减速，局部真实扫描速度低于设定速度且不可知，因此对应两侧常出现更深的边缘沟槽。
- 受软件限制，层与层之间可能偶发：一次额外矩形边框扫描，或上/下某一侧额外一次扫描。
- 软件缺少“关激光空走”功能，因此上述边界曝光属于不可完全控制的轨迹伪影。
- 后续科学分析的主要对象是**中央恒速、受控扫描区域**，边界异常区不应混入材料本征响应分析。
- 共聚焦原始文件可能是 Keyence VK-X 系列导出的高度 CSV。示例文件为 `1_高度.csv`。
- 示例文件对应设计表 `120正式_design记录表_0516.csv` 中的第 1 条实验。

本任务的首要目标不是直接进行机器学习或形貌 PCA，而是建立**可靠、统一、可审计的几何配准与 ROI 标准化流程**。

---

## 2. 总体目标

建立一个两阶段流程：

\[
\text{全部文件}
\rightarrow
\text{全局几何标定}
\rightarrow
\text{逐文件受约束配准}
\rightarrow
\text{统一 200\times200\,\mu m 坐标}
\rightarrow
\text{全局边缘影响分析}
\rightarrow
\text{统一中央稳定 ROI}
\]

必须严格遵守以下原则：

1. **能全局确定的参数，不允许每个文件单独重新拟合。**
2. **浅加工样本不能依赖单一高度阈值识别。**
3. **矩形理论尺寸 200 μm × 200 μm 作为强先验，而不是要求真实形貌边界必须严格等于 200 μm。**
4. **每个正式样本必须最终落在统一物理坐标系中。**
5. **正式统计阶段所有样本必须使用同一个物理尺寸的中央稳定 ROI。**
6. **任何边界额外深沟不能默认解释成材料本征形貌。**
7. **第一阶段配准完成后必须先生成 QA 总览，人工检查通过后，再进入第二阶段稳定 ROI 自动确定。**

---

# 3. 本任务分阶段执行

## Phase A：全局矩形几何标定与 200×200 μm 标准化配准

这是当前优先级最高的阶段。

### Phase A 完成后必须暂停

不要直接继续做：

- PCA
- autoencoder
- PSD
- 形貌压缩
- 工艺参数回归
- 槽深/槽宽建模
- 机器学习

Phase A 结束后先输出 QA 结果供人工检查。

只有在配准确认可靠后，才执行 Phase B。

---

## Phase B：全局边缘影响分析与统一中央稳定 ROI

在所有样本完成可靠配准后：

1. 自动识别扫描长线方向 / 换向边方向；
2. 估计四个边的边缘影响距离；
3. 全局统计后确定一个固定物理尺寸的中央稳定 ROI；
4. 所有正式分析统一从该 ROI 提取。

---

# 4. 输入文件

## 4.1 实验设计表

至少包括：

- `120正式_design记录表_0516.csv`
- 后续可能还有其它材料对应设计表

应实现独立的 design-table parser，避免与高度文件解析耦合。

实验编号与高度文件命名应可建立一一对应关系，例如：

```text
1_高度.csv  -> design row 1
2_高度.csv  -> design row 2
...
```

如果实际命名规则不同，应通过配置文件指定，不要在代码中写死。

---

## 4.2 共聚焦高度文件

示例：

```text
1_高度.csv
```

需要自动解析：

- 高度矩阵
- X/Y 像素数
- XY 像素间距
- 单位
- 可能存在的元数据
- 无效点 / 缺失值

### 强制要求

不要假定：

```text
height == 0
```

就是无效值。

只有在原始格式明确标记无效数据时才转为 NaN。

---

# 5. 坐标和符号约定

统一使用物理坐标，单位 μm。

标准矩形坐标系定义：

\[
x,y\in[-100,100]~\mu m
\]

最终所有样本均需重采样到统一 canonical grid。

建议：

```text
x: -100 -> +100 μm
y: -100 -> +100 μm
```

坐标原点：理论矩形中心。

## 扫描方向定义

最终应统一规定：

```text
y = 弓字形长扫描方向
x = hatch / 相邻扫描线方向
```

若实际方向相反，则在全局方向识别后统一旋转/翻转。

换向减速产生明显深沟的两条边应最终对应：

```text
y = -100 μm
y = +100 μm
```

---

# 6. Phase A 详细算法要求

## 6.1 原始高度解析

实现函数：

```python
parse_vk_csv(path) -> HeightMap
```

建议返回数据结构：

```python
@dataclass
class HeightMap:
    z: np.ndarray
    dx_um: float
    dy_um: float
    x_um: np.ndarray
    y_um: np.ndarray
    metadata: dict
```

要求：

- 尽量自动识别编码；
- 对 Keyence CSV 的非标准头部进行健壮处理；
- 记录原始数据尺寸；
- 原始文件绝不覆盖。

---

## 6.2 表面基准校正

每张图先做 robust plane leveling：

\[
z_{\rm leveled}=z_{\rm raw}-(ax+by+c)
\]

### 不允许

直接普通最小二乘拟合整张图，因为深槽会拉偏平面。

### 推荐流程

1. 初次 robust plane fit；
2. 计算 residual；
3. 排除明显低于上表面的区域；
4. 再次拟合；
5. 迭代至参数稳定。

可使用：

- Huber regression
- RANSAC
- iterative sigma clipping
- IRLS

优先可解释、稳定的方法，不需要复杂神经网络。

输出：

```text
plane_a
plane_b
plane_c
plane_fit_rmse
surface_inlier_fraction
```

---

# 7. 高 SNR calibration samples 自动选择

全局旋转角不应由浅加工样本估计。

先为所有文件计算加工对比度指标，例如：

\[
C_i=Q_{50}(z_i)-Q_{05}(z_i)
\]

也可以组合：

- negative-tail amplitude
- inside/outside contrast
- connected low-region area

自动选出高对比度样本，例如 top 20%–30%，但比例必须写进配置文件。

输出：

```text
calibration_sample_ids.csv
```

并记录每个样本的 contrast score。

---

# 8. 初步矩形检测：仅用于全局标定

对于高 SNR calibration samples，可用较自由的方法估计：

\[
(c_{x,i},c_{y,i},\theta_i,L_{x,i},L_{y,i})
\]

可采用：

- 高度阈值
- 连通域
- morphology
- contour
- minimum-area rectangle
- Hough / edge fitting

### 重要限制

`minAreaRect` 等方法只允许用于**全局标定阶段**。

最终正式配准不得让每个文件完全自由拟合 angle/width/height，否则浅槽会产生随机姿态误差。

---

# 9. 全局旋转偏置估计

理论上同一批样本存在共同旋转：

\[
\theta_{\rm global}
\]

由于 200×200 μm 为正方形，必须处理 90° 对称性。

所有角度统一折算到：

\[
[-45^\circ,45^\circ)
\]

使用 robust 统计，例如 weighted median。

权重可综合：

- 矩形边缘清晰度
- 面积与 200×200 μm 一致程度
- 拟合残差
- 连通域完整度
- inside/outside contrast

输出：

```text
theta_global_deg
theta_mad_deg
theta_sample_distribution.csv
```

若角度离散明显超过合理范围，应报警，不得自动强行平均。

建议 QA 阈值先设为：

```text
MAD > 0.3°  -> warning
MAD > 0.8°  -> hard review
```

具体阈值配置化。

---

# 10. 200×200 μm 是强几何先验，不是硬形貌边界

必须明确区分：

```text
theoretical machining rectangle
```

和：

```text
observed modified / ablated footprint
```

因加减速、边框扫描、额外曝光，实际深度变化范围允许超出：

\[
[-100,100]\times[-100,100]~\mu m
\]

因此：

- 不允许强迫形貌边缘严格位于 ±100 μm；
- ±100 μm 只用于定义理论加工坐标；
- 边缘影响分析在 Phase B 单独处理。

---

# 11. 每个文件最终只允许局部优化中心位置

一旦获得：

```text
theta_global
rectangle_size = 200 × 200 μm
```

单文件正式配准只优化：

\[
(c_{x,i},c_{y,i})
\]

即：

```text
angle = fixed
width = fixed
height = fixed
center = variable
```

中心搜索应使用全局先验：

\[
(c_x,c_y)\sim \text{batch prior}
\]

高 SNR 样本可先估计中心分布：median center offset + MAD/std。

浅槽只允许在有限搜索窗口内移动。

---

# 12. 浅加工矩形定位：禁止仅依赖绝对高度阈值

实现一个受约束的 rectangle template score。

对于候选中心 \((c_x,c_y)\)，放置固定 200×200 μm、固定 \(\theta_{\rm global}\) 的理论矩形。

综合以下得分：

\[
S=w_1S_{\rm region}+w_2S_{\rm edge}+w_3S_{\rm center-prior}
\]

## 12.1 Region contrast score

比较理论矩形内部与外部参考面，可采用 robust median：

\[
S_{\rm region}=
\frac{\operatorname{median}(z_{\rm outside})-\operatorname{median}(z_{\rm inside})}
{\sigma_{\rm surface}}
\]

## 12.2 Edge score

不要要求边缘精确位于 ±100 μm。

在理论边界两侧设置 tolerance band，例如 ±10–15 μm，计算法向梯度响应：

\[
|\nabla z\cdot n|
\]

目的：利用加工边缘大致位置，同时允许边缘深沟外扩、减速区和额外 contour scan。

## 12.3 Center prior

对中心偏离 batch-global median 进行惩罚，例如：

\[
S_{\rm prior}=-\left[
\left(\frac{c_x-\bar c_x}{\sigma_x}\right)^2+
\left(\frac{c_y-\bar c_y}{\sigma_y}\right)^2
\right]
\]

---

# 13. Canonical 200×200 μm 重采样

每个文件完成中心定位后，旋转到统一方向并重采样。

目标：

\[
H_i^{200}(x,y),\quad x,y\in[-100,100]~\mu m
\]

### 网格要求

优先保持接近原始像素间距。

对于示例文件，pixel size 约 0.344174 μm，200 μm 对应约 581 个采样间隔，因此可采用约 582×582，但不要写死 582。

实现方式：根据全数据的像素分辨率自动确定统一 grid spacing。

若全批次分辨率完全一致，可直接使用对应 canonical shape。

若存在不同分辨率：

- 选定统一物理步长；
- 所有样本统一插值；
- 记录原始和目标步长。

插值建议 linear/cubic，禁止高阶过拟合插值。

---

# 14. 正方形 90° 方向歧义处理

200×200 μm 无法单靠矩形几何判断扫描长线方向。

应通过全局形貌统计识别两对边：edge pair A / edge pair B。

由于弓字形长线两端存在减速/换向，通常其中一对边具有更显著的额外深度、深沟、梯度或边缘能量。

将该方向定义为 scan-long direction = y。

最终必须保证所有样本：

```text
turning edges -> y = ±100 μm
```

如无法可靠判断，输出 warning 并等待人工确认。

---

# 15. Phase A QA 输出

Phase A 必须为每个样本生成可视化 QA。

建议单张 QA 图包含：

1. raw height map
2. leveled height map
3. 理论 200×200 μm 矩形叠加
4. canonical 200×200 μm map
5. 中心位置与旋转信息
6. registration score
7. low-confidence warning

额外生成全数据 montage：

```text
registration_montage.png
```

要求：所有样本采用统一且可比的色标策略，标注 sample id，异常样本可一眼识别。

输出 `registration_metrics.csv`，至少包含：

```text
sample_id
center_x_um
center_y_um
theta_global_deg
registration_score
region_score
edge_score
center_prior_score
plane_rmse
contrast_score
warning_flag
```

---

# 16. Phase A 验收标准

Phase A 只有满足以下条件后才可进入 Phase B。

## 几何一致性

所有样本 canonical map 中：理论矩形中心一致、边缘整体位置一致、不出现明显逐文件旋转抖动。

## 中心定位

高 SNR 样本的自动中心与自由拟合中心差异应小。

建议：

```text
median center error < 2 μm
95th percentile < 5 μm
```

若达不到，先修正配准。

## 角度稳定

全局角度应由高 SNR 样本一致支持。

## 浅槽成功率

浅层样本不能因阈值不足大量定位失败。

## 人工检查

必须人工查看 `registration_montage.png` 并确认 PASS，之后才允许 Phase B。

---

# 17. Phase B：中央稳定区的全局确定

在所有样本统一配准后，分析边缘轨迹影响。

不能让每个样本单独选择不同大小 ROI。

最终必须得到：

\[
L_x^{stable},L_y^{stable}
\]

且所有正式统计统一使用该尺寸。

---

# 18. 四边边缘影响距离

对每个样本估计：

\[
m_{i,\rm top},m_{i,\rm bottom},m_{i,\rm left},m_{i,\rm right}
\]

其中 m 表示从理论矩形边缘向内需要剔除的距离。

原因：

- 上下可能受换向减速影响；
- 四边可能受额外边框扫描影响；
- 上/下某一边可能偶发额外单边扫描。

因此不能只分析两个边。

---

# 19. 一维全局边缘影响 profile

对 canonical map 定义去除深度：

\[
D_i(x,y)=-z_i(x,y)
\]

或根据数据符号自动确定，并在 metadata 中记录。

对于 y 方向：

\[
p_i(y)=\operatorname{median}_{|x|<x_0}D_i(x,y)
\]

初始可取：

```text
x0 = 50–60 μm
```

但必须配置化。

使用 median 而不是 mean，以降低 debris、isolated pits、spikes 的影响。

x 方向同理。

---

# 20. 不同深度样本必须归一化后再做全局边缘统计

不同样本绝对深度差异可能非常大。

禁止直接 ensemble-average absolute depth profile。

建议构造 shape-normalized profile：

\[
\tilde p_i(s)=\frac{p_i(s)-p_{i,\rm center}}{A_i}
\]

其中 \(A_i\) 使用 robust amplitude，例如 Q90-Q10 或中央深度尺度。

同时必须保留原始绝对 profile，归一化仅用于确定边缘影响距离。

---

# 21. 边缘影响起点判据

不能使用简单固定阈值，例如 depth > 2 μm。

推荐基于局部梯度或曲率的 robust 判据。

例如：

\[
g_i(s)=\left|\frac{dp_i}{ds}\right|
\]

用中央区域估计正常背景：

\[
g_0=\operatorname{median}(g)
\]

\[
\sigma_g=1.4826\,MAD(g)
\]

当：

\[
g_i(s)>g_0+3\sigma_g
\]

且异常连续长度超过 5 μm，才认定进入 boundary transition。

所有阈值必须配置化，并建议额外保留 alternative criterion 做敏感性分析。

---

# 22. 统一稳定 ROI 的确定原则

不要使用：

\[
\min_i r_i
\]

因为偶发额外扫描会让所有样本 ROI 被极端异常值缩小。

建议采用覆盖率分位数，例如：

\[
r_{\rm global}=Q_{0.10}(r_i)
\]

表示统一 ROI 对约 90% 样本安全。

建议默认 coverage = 90%，并允许 95% 敏感性分析。

剩余异常样本只做 flag，而不是继续缩小全局 ROI。

---

# 23. ROI 最终尺寸必须取整为便于复现的物理尺寸

假设算法给出：

```text
Lx = 163.4 μm
Ly = 104.7 μm
```

不要直接使用这种数字。

应向下保守取整，例如：

```text
160 × 100 μm
```

最终：

\[
ROI=[-L_x/2,L_x/2]\times[-L_y/2,L_y/2]
\]

所有样本完全一致。

---

# 24. Phase B QA

输出：

```text
boundary_influence_metrics.csv
```

至少包含：

```text
sample_id
margin_top_um
margin_bottom_um
margin_left_um
margin_right_um
stable_x_um
stable_y_um
boundary_warning
```

输出全局图：

```text
boundary_margin_distributions.png
stable_roi_selection.png
stable_roi_montage.png
```

`stable_roi_selection.png` 应清楚展示四边 margin 分布、90%/95% 分位线，以及最终取整后的 ROI 尺寸。

---

# 25. 后续形貌提取接口预留

本任务不要求现在完成形貌建模，但代码结构应预留：

```python
extract_stable_roi(...)
extract_mean_profile(...)
extract_mad_profile(...)
compute_macro_micro(...)
compute_pca_features(...)
```

正式下游输入统一为：

\[
H_i^{stable}(x,y)
\]

而不是原始图。

---

# 26. 明确禁止事项

1. 禁止对每个文件独立 `minAreaRect -> rotate -> crop` 后直接作为最终配准结果。
2. 禁止每个样本自己决定不同的 stable ROI 尺寸。
3. 禁止使用单一绝对深度阈值识别所有浅槽。
4. 禁止在配准之前做强平滑、低通滤波或 PCA。
5. 禁止 PCA 前对每张高度图单独 min-max 到 [0,1]，这会消除真实绝对深度信息。
6. 禁止把边缘深沟自动解释成材料响应。
7. 禁止把同一张高度图中的大量 patch 当作彼此独立的实验样本；后续训练/验证必须按完整实验条件/完整加工面划分，避免 pseudo-replication。

---

# 27. 推荐代码结构

```text
project/
├── config/
│   └── processing.yaml
├── src/
│   ├── io_vk.py
│   ├── leveling.py
│   ├── global_calibration.py
│   ├── registration.py
│   ├── canonical_grid.py
│   ├── scan_axis.py
│   ├── boundary_analysis.py
│   ├── stable_roi.py
│   ├── qa.py
│   └── utils.py
├── scripts/
│   ├── 01_calibrate_geometry.py
│   ├── 02_register_all.py
│   ├── 03_generate_registration_qa.py
│   ├── 04_estimate_boundary_effect.py
│   └── 05_select_stable_roi.py
├── outputs/
│   ├── calibration/
│   ├── registered/
│   ├── qa/
│   ├── metrics/
│   └── stable_roi/
└── tests/
```

---

# 28. 配置文件必须包含

示例：

```yaml
nominal_rectangle_um: [200.0, 200.0]

calibration:
  top_fraction: 0.25
  angle_warning_mad_deg: 0.3
  angle_review_mad_deg: 0.8

registration:
  edge_band_um: 15.0
  center_search_um: 30.0
  interpolation: linear

canonical_grid:
  size_um: [200.0, 200.0]
  pixel_um: auto

boundary:
  profile_halfwidth_um: 60.0
  center_reference_halfwidth_um: 20.0
  gradient_sigma_threshold: 3.0
  min_persistence_um: 5.0
  coverage_quantile: 0.10

qa:
  save_individual: true
  save_montage: true
```

不要把关键参数散落硬编码在脚本中。

---

# 29. 可复现性要求

每次运行必须记录：

```text
timestamp
git commit hash
config file
input file list
software versions
global angle
canonical pixel spacing
stable ROI size
warnings
```

输出：

```text
run_manifest.json
```

若随机算法存在，固定 random seed。

---

# 30. 异常处理

遇到以下情况不得静默继续：

- 高度文件无法解析；
- 像素间距缺失；
- 矩形中心搜索结果触及搜索边界；
- registration score 明显偏低；
- 全局角度分布多峰；
- 扫描方向无法可靠判断；
- canonical ROI 超出原始有效测量范围；
- 缺失值比例过高。

必须输出 warning + flag + QA 图，而不是自动猜测。

---

# 31. 第一阶段最终交付物

Phase A 结束时至少应得到：

```text
outputs/
├── calibration/
│   ├── global_geometry.json
│   └── calibration_sample_ids.csv
├── registered/
│   ├── 001.npy / csv
│   ├── 002.npy / csv
│   └── ...
├── metrics/
│   └── registration_metrics.csv
├── qa/
│   ├── registration_montage.png
│   └── individual/
└── run_manifest.json
```

`global_geometry.json` 至少包含：

```json
{
  "nominal_size_um": [200.0, 200.0],
  "theta_global_deg": null,
  "scan_axis": null,
  "canonical_pixel_um": null,
  "center_prior": {
    "x_um": null,
    "y_um": null,
    "sigma_x_um": null,
    "sigma_y_um": null
  }
}
```

---

# 32. Codex 执行顺序

严格按以下顺序：

### Step 1
只用示例 `1_高度.csv` 打通解析、平面校正、基础可视化。

### Step 2
批量读取全部高度文件，只计算基础 metadata 和 contrast score。

### Step 3
选择高 SNR calibration samples。

### Step 4
估计全局旋转角与中心 prior。

### Step 5
固定 200×200 μm + 固定全局角度，逐文件优化中心。

### Step 6
统一重采样为 canonical 200×200 μm 高度图。

### Step 7
生成完整 registration QA。

### Step 8
**停止。等待人工确认。**

### Step 9
人工确认后，才进行扫描方向识别和边界影响统计。

### Step 10
确定全局 stable ROI。

### Step 11
生成 stable ROI QA。

---

# 33. 科学解释边界

本任务只解决：

\[
\text{测量坐标}
\rightarrow
\text{统一加工坐标}
\rightarrow
\text{统一稳定区}
\]

不得在此阶段声称：

- 边缘深沟属于材料本征烧蚀机制；
- 当前形貌已经是充分状态；
- 存在 hidden material memory；
- PCA 模态具有某种具体物理机制；
- 某个加工参数导致某种材料机制。

这些属于后续科学建模问题。

---

# 34. 最终目标

得到两个标准数据对象。

## 完整理论加工区

\[
H_i^{200}(x,y)
\]

用于：

- QA
- 边缘轨迹伪影研究
- 完整形貌存档

## 中央稳定加工区

\[
H_i^{stable}(x,y)
\]

用于后续：

- 槽深/槽宽/体积
- 平均截面
- roughness
- PSD
- PCA / functional PCA
- 形貌压缩
- 工艺参数重参数化
- predictive-state 分析

核心原则：

> **先建立统一、可靠、物理上可解释的坐标与 ROI，再讨论任何“压缩”或机器学习问题。**
