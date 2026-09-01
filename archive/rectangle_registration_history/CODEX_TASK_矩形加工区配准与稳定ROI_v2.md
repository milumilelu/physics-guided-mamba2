# Codex 任务说明 v2：矩形加工区 session 级配准与统一稳定 ROI

## 0. 任务定位

本任务建立超快激光矩形加工形貌的可审计预处理流程，只解决：

[
	ext{测量坐标}
ightarrow
	ext{统一加工坐标}
ightarrow
	ext{统一稳定分析区}
]

本任务不进行 PCA、形貌压缩、机器学习、工艺回归或材料机理解释。

理论加工尺寸为 200 μm × 200 μm。这里的 200 μm 统一定义为：

> nominal programmed machining region，即加工软件给出的名义扫描区域。

除非设备轨迹文件能够进一步证明，否则不得将其表述为最外侧扫描中心线范围、光学曝光足迹或实际改性边界。实际形貌允许超出理论边界。

---

## 1. 不可绕过的方法学修正

v2 强制执行以下规则：

1. 数据对象必须分为 `H_raw -> H_reg -> H_200 -> H_stable`。
2. `H_reg` 必须保留理论加工区外侧参考表面；Phase B 只能使用 `H_reg`。
3. 没有真实高度矩阵或经过交叉验证的 CAG 解码器时，Phase 0 必须停止。
4. 全局姿态按 measurement/mounting session 估计，不得默认全数据共用一个角度。
5. 正方形的旋转象限和镜像，即 D4 变换，必须在 Phase A 内按 session 固定。
6. 单文件正式配准只优化平移中心，不允许自由优化角度和矩形尺寸。
7. center prior 只限制搜索域，不进入主配准目标函数。
8. 没有独立物理参考或人工标注时，不得声称绝对中心误差。
9. 边缘污染统一定义为向内剔除距离 `boundary_margin_um`；90%覆盖使用 `Q0.90`。
10. 固定 ROI 对仍受边界污染的样本必须标记无效，不得静默修复或删除。
11. 高度符号按格式或 session 固定，不允许逐文件自动翻转。
12. 高度和有效性 mask 必须共同变换；禁止跨越大缺失区或向原视场外推。

---

## 2. 标准数据对象

### 2.1 原始高度图 H_raw

原始测量坐标中的高度与有效性 mask：

```text
height_raw
valid_mask_raw
dx_um, dy_um
metadata
```

原始文件只读，绝不覆盖。

### 2.2 大视场配准图 H_reg

完成初步基准校正、session 级连续旋转、session 级 D4 变换、逐文件受约束平移、统一物理网格重采样和最终外部参考面校正后的对象。

`H_reg` 必须同时保存：

```text
height
valid_mask
x_um, y_um
source_transform
leveling_metadata
```

目标优先为：

[
x,yin[-150,150]~mu m
]

即 300 μm × 300 μm，但不得硬编码。实际尺寸由变换后所有样本有效视场的公共交集决定：

[
L_{reg}=min(300~mu m,L_{common})
]

硬性要求：

[
L_{reg}ge260~mu m
]

若某个 session 达不到要求，必须 flag 并停止其 Phase B；禁止外推补足。

### 2.3 理论加工区 H_200

从 `H_reg` 确定性裁取：

[
H_{200}=H_{reg}|_{[-100,100]	imes[-100,100]}
]

它用于完整名义加工区存档、QA 和边缘轨迹伪影研究，不用于估计外侧参考面。

### 2.4 中央稳定区 H_stable

Phase B 基于合格 `H_reg` 估计统一边缘污染距离后，从 `H_200` 裁取固定物理尺寸 ROI：

[
H_{stable}=H_{200}|_{ROI_{stable}}
]

只有 `H_stable` 可以进入后续主要材料响应分析。

---

## 3. 阶段和门禁

```text
Phase 0   数据可用性、样本映射与 session 定义
Phase A1  输入验证、解析与 coarse leveling
Phase A2  session 级角度和 D4 方向标定
Phase A3  固定姿态下的逐文件平移配准
Phase A4  公共视场计算与 H_reg 重采样
Phase A5  final leveling 与 registration QA
STOP      人工确认 Phase A
Phase B1  四边影响距离估计
Phase B2  工艺参数分层覆盖诊断
Phase B3  统一 stable ROI 确定
Phase B4  invalid/missing 协议执行
Phase B5  stable ROI QA
```

未经人工明确确认 Phase A PASS，不得执行 Phase B。

---

## 4. Phase 0：数据可用性检查

### 4.1 必需输入

至少需要：

- 实验设计表；
- 每个样本的真实高度矩阵；
- 样本编号与设计表的确定性映射；
- measurement/mounting session 定义；
- 像素间距和高度单位；
- 原始无效点定义。

当前仓库若不存在 `*_高度.csv`，不得假定附件、历史会话或其他目录中的文件可用。

### 4.2 CSV 输入

Keyence 高度 CSV 使用独立解析器：

```python
parse_vk_csv(path) -> HeightMap
```

解析器必须处理非标准头部、编码、单位、像素间距、无效值和高度矩阵尺寸。不得假定 `height == 0` 表示无效。

### 4.3 CAG 输入

允许直接解析 CAG，但每一种 CAG 采集系列必须先完成独立可行性验证：

1. 选择至少 3 个覆盖浅、中、深形貌的样本；
2. 从 Keyence 软件导出对应高度 CSV；
3. 比较 CAG 解码与 CSV 的尺寸、像素间距、有效 mask 和高度数值；
4. 保存逐像素差异统计和 QA 图；
5. 只有误差满足预注册容差后，才允许批量使用该 CAG 解码器。

已有其他 CAG 文件上的成功解码不能自动证明当前矩形加工 CAG 可用。

### 4.4 样本映射

映射必须依据显式 `sample_id` 或配置规则，不得依据 DataFrame 当前行号隐式连接。

输出：

```text
input_inventory.csv
sample_design_mapping.csv
session_manifest.csv
phase0_validation.json
```

任一以下情况必须 STOP：

- 没有可验证高度矩阵；
- 样本与设计表不是一一对应；
- 像素间距或单位缺失；
- session 无法定义；
- CAG 解码尚未通过对应 CSV 交叉验证。

---

## 5. Session 定义

session 定义为一次连续的 measurement/mounting coordinate system。以下任一事件均应新建 session：

- 重新装夹；
- 样品旋转；
- 显微镜坐标重置；
- 共聚焦采集方向改变；
- 无法证明姿态连续的独立采集文件。

每个候选 CAG 或独立批量导出目录默认视为独立 session，除非元数据和姿态统计共同支持合并。

所有全局几何参数写成：

```text
theta_session_deg
d4_transform_session
center_search_prior_session
```

不得使用未经分组的唯一 `theta_global`。

---

## 6. 坐标与高度符号约定

统一单位为 μm。

最终坐标约定：

```text
y = 弓字形长扫描方向
x = hatch / 相邻扫描线方向
```

换向边对应 `y = ±100 μm`。正负方向不得按每个样本“哪边更深”决定。

推荐 session 级确定性约定：完成连续旋转和轴识别后，与原始显微镜图像 +Y 方向最接近的长扫描方向定义为 canonical +y。同一 session 的全部样本应用完全相同的 D4 变换。

高度符号必须按输入格式或 session 固定。例如经格式验证后规定：

```text
z_positive = upward
D = z_reference - z
```

先用 3–5 个明显深槽人工核验。单个文件若呈现相反符号，必须 flag 为解析或导出异常，不得自动翻转。

---

## 7. Phase A1：解析与 coarse leveling

### 7.1 HeightMap

```python
@dataclass
class HeightMap:
    z: np.ndarray
    valid_mask: np.ndarray
    dx_um: float
    dy_um: float
    x_um: np.ndarray
    y_um: np.ndarray
    metadata: dict
```

### 7.2 Coarse leveling

配准前不得直接对全图做普通最小二乘平面拟合。

优先使用图像外围 reference frame。中央排除区必须配置化，默认可从 320 μm × 320 μm 开始，但只能在实际视场足够大时使用。

外围区域使用可解释的 robust plane fit，例如 IRLS、Huber 或固定规则的非对称 sigma clipping：

[
z_{coarse}=z_{raw}-(ax+by+c)
]

输出：

```text
coarse_plane_a
coarse_plane_b
coarse_plane_c
coarse_plane_rmse
coarse_reference_fraction
coarse_leveling_warning
```

如果外围有效参考面积不足、空间分布不覆盖多个象限或拟合残差异常，必须停止自动处理。

---

## 8. Calibration samples：高 SNR 且覆盖工艺空间

不得简单选择最深或 contrast 最高的 top 25%。高 SNR 往往与特定工艺条件相关，会把形貌偏差写入全局先验。

流程：

1. 为全部文件计算不依赖单一阈值的 contrast diagnostics；
2. 按主要工艺参数对设计空间分层；
3. 每个有足够样本的 strata 内选择高 SNR 样本；
4. 检查 calibration set 是否覆盖主要脉宽、频率、速度、pass 和 hatch 水平；
5. 选择规则和比例写入配置，运行后不得按结果调整。

可用诊断包括：

```text
Q50(z) - Q05(z)
negative-tail amplitude
inside/outside candidate contrast
edge energy
connected modified-area fraction
valid-data fraction
```

输出：

```text
calibration_sample_ids.csv
calibration_coverage_by_process.csv
```

---

## 9. Phase A2：session 级连续角度与 D4 方向

### 9.1 初步自由拟合仅用于标定

只对 calibration samples 允许自由估计：

[
(c_x,c_y,	heta,L_x,L_y)
]

可使用阈值、连通域、边缘拟合、Hough 或 minimum-area rectangle。该结果不是正式单文件配准，更不是中心真值。

### 9.2 连续旋转角

正方形角度折算到：

[
[-45^circ,45^circ)
]

在每个 session 内使用预先规定的稳健统计估计 `theta_session`。权重只能来自边缘完整度、拟合残差、面积一致性和 contrast 等预定义质量指标。

输出：

```text
theta_session_deg
theta_session_mad_deg
theta_sample_distribution.csv
angle_multimodality_warning
```

建议 QA 门限配置化：

```text
MAD > 0.3 deg -> warning
MAD > 0.8 deg -> hard review
```

出现明显多峰时不得强行平均，应检查 session 划分。

### 9.3 D4 象限和镜像

200 μm 正方形具有 90°旋转和镜像歧义。必须使用 session pooled morphology、扫描轨迹元数据或人工确认确定：

[
T_{session}^{D4}
]

该变换必须在输出任何 canonical map 前固定。禁止每个样本自行选择旋转象限或翻转。

若长扫描方向无法可靠识别：

```text
d4_status = requires_manual_confirmation
STOP before H_reg export
```

---

## 10. Phase A3：受约束平移配准

固定：

```text
angle = theta_session
d4_transform = session fixed
nominal size = 200 μm × 200 μm
```

单文件只优化：

[
(c_{x,i},c_{y,i})
]

### 10.1 搜索域

center prior 只用于定义宽松搜索窗口，不直接加入目标函数。初始优先使用仪器采集几何，例如 image center ±40–50 μm。

只有证明高 SNR 中心估计与主要工艺参数无明显系统关系后，才允许使用 session median center 缩小搜索域。

搜索最优点触及边界时：

```text
center_search_boundary_hit = True
registration_status = review
```

### 10.2 无量纲配准分数

主目标只包含 region 和 edge 两项：

[
S=w_rS_{region}+w_eS_{edge}
]

两项必须先无量纲化。

区域项示例：

[
S_{region}=
rac{median(z_{outside})-median(z_{inside})}
{1.4826,MAD(z_{outside})}
]

边缘项在理论边界两侧的固定 tolerance band 内计算法向梯度，并使用外部参考梯度尺度归一化。不得要求观察边缘严格位于 ±100 μm。

默认权重固定为：

```text
(w_region, w_edge) = (0.5, 0.5)
```

必须同时运行预注册敏感性组合：

```text
(0.25, 0.75)
(0.50, 0.50)
(0.75, 0.25)
```

若不同权重导致中心位置跨度超过配置阈值，例如 2–3 μm：

```text
registration_unstable = True
```

不得事后选择视觉上最好的权重作为主结果。

---

## 11. Phase A4：公共视场与 mask-aware 重采样

### 11.1 公共视场

先将每个样本的有效视场边界按最终旋转和平移变换到 canonical 坐标，再计算每个 session 的公共有效正方形尺寸 `L_common`。

记录：

```text
common_fov_um
registered_fov_um
external_reference_width_um
```

如果个别样本导致公共视场异常缩小，不得静默删除。应输出样本级覆盖诊断，由人工决定是排除该测量失败样本还是终止该 session。

### 11.2 Canonical grid

统一物理步长根据全数据原始分辨率确定，不写死 582×582。若分辨率不同，应使用预注册规则，例如不细于最粗原始步长，避免伪造空间分辨率。

禁止高阶过拟合插值。主分析使用 linear；其他插值只能作为敏感性分析。

### 11.3 NaN 与 mask

高度与有效 mask 必须使用同一变换。可采用 normalized interpolation：

[
z' = rac{I(zM)}{I(M)}
]

只有在：

[
I(M)>eta
]

且目标点距真实有效数据不超过预注册距离时，目标高度才有效。否则保持 NaN。

必须满足：

- 不向原始视场外推；
- 不跨越大面积缺失区；
- 保存变换后的 valid mask；
- 保存每个输出对象的 valid fraction。

---

## 12. Phase A5：final leveling

完成配准后，使用 `H_reg` 中理论加工区之外的明确外部参考面重新拟合最终平面。

配置示例：

```text
final_exclusion_halfwidth_um: 120
minimum_reference_frame_width_um: 10
minimum_reference_valid_fraction: 0.20
```

排除区必须覆盖名义加工区及合理边缘曝光缓冲，但不得大到使最小 `H_reg` 无参考面。

最终参考点必须满足：

- 有效面积达到阈值；
- 覆盖足够的 x/y 空间跨度；
- 至少覆盖多个象限；
- 不存在明显大面积外部加工或缺失；
- robust plane residual 通过 QA。

失败时：

```text
final_leveling_failed = True
registration_status = failed
```

此时不得生成可供下游使用的 `H_200` 或 `H_stable`。

---

## 13. 配准精度的证据等级

### Level 1：独立物理参考

包括扫描轨迹坐标、scanner position、共聚焦 stage coordinate、fiducial 或已知样品标记。只有该等级允许报告 absolute registration error。

### Level 2：独立人工标注

至少两位独立标注者对预先选定的高质量样本标注中心、边和方向，并报告 inter-rater uncertainty。算法误差必须相对于冻结的人工参考评估。

### Level 3：纯形貌配准

只能报告：

- 重复性；
- 参数扰动稳定性；
- 不同算法一致性；
- registration 后跨样本残余边缘离散；
- 人工 QA 通过率。

没有 Level 1 或 Level 2 参考时，删除“median center error < 2 μm”之类绝对精度结论。

---

## 14. Phase A QA 与验收

每个样本输出两类 QA 色标。

### 14.1 Absolute-scale QA

全批次或 session 使用统一绝对色标，用于比较真实深度。

### 14.2 Local-contrast QA

每个样本使用固定分位规则，例如 Q1–Q99，用于检查浅槽定位、缺失区和边缘。图上必须明确标记：

```text
LOCAL CONTRAST — NOT COMPARABLE IN ABSOLUTE DEPTH
```

每张 QA 至少显示：

1. raw height；
2. coarse-leveled height；
3. 理论 ±100 μm 边界；
4. `H_reg` 边界；
5. `H_200`；
6. valid mask；
7. 中心、session angle、D4 变换；
8. region/edge score；
9. 权重敏感性位移；
10. warning 和失败状态。

输出：

```text
registration_metrics.csv
registration_montage_absolute.png
registration_montage_local_contrast.png
qa/registration_individual/
```

`registration_metrics.csv` 至少包括：

```text
sample_id
session_id
center_x_um
center_y_um
theta_session_deg
d4_transform_session
registration_score
region_score
edge_score
weight_sensitivity_shift_um
center_search_boundary_hit
coarse_plane_rmse
final_plane_rmse
contrast_score
registered_valid_fraction
registration_status
warning_flags
```

Phase A PASS 必须同时满足：

- session 角度分布单峰且稳定；
- D4 方向已由证据或人工确认固定；
- `L_reg >= 260 μm`；
- 无大量搜索边界命中；
- 浅加工样本没有系统性失败；
- 权重敏感性在预注册容差内；
- final leveling 具有足够外部参考面；
- 人工检查两套 montage 并明确确认 PASS。

---

## 15. Phase B1：四边边缘影响分析

Phase B 只能读取通过 Phase A 的 `H_reg`，不得从 `H_200` 估计边缘影响。

统一去除深度：

[
D=z_{reference}-z
]

四个边分别估计向内污染距离：

```text
margin_left_um
margin_right_um
margin_top_um
margin_bottom_um
```

### 15.1 一维 profiles

示例：

[
p_y(y)=median_{|x|<x_0}D(x,y)
]

[
p_x(x)=median_{|y|<y_0}D(x,y)
]

`x0/y0` 必须配置化。必须保存绝对 profile 和用于形状比较的归一化 profile。

归一化示例：

[
	ilde p_i(s)=rac{p_i(s)-p_{i,center}}{A_i}
]

其中 `A_i` 使用预注册的 robust amplitude。归一化仅用于定位边界形状，不替代绝对深度。

### 15.2 边界转变判据

不得使用固定绝对深度阈值。主判据使用中心稳定区估计的梯度背景：

[
g(s)=left|rac{dp}{ds}ight|
]

[
g_{thr}=median(g_{center})+kcdot1.4826MAD(g_{center})
]

只有异常连续长度超过 `min_persistence_um` 才视为边缘污染。阈值、平滑规则和连续长度均写入配置，并设置至少一个预注册 alternative criterion 做敏感性分析。

算法从理论边界向中心搜索，输出“污染消失并持续恢复稳定”的最内侧位置。禁止用单个尖峰决定 margin。

---

## 16. Phase B2：工艺参数分层诊断

边缘影响距离可能依赖脉宽、频率、速度、pass 和 hatch。必须检查：

```text
boundary_margin ~ pulse_width + frequency + velocity + pass + hatch
```

该分析只判断统一 ROI 是否对某些工艺层级系统失效，不作因果解释。

至少输出：

- 总体 margin 分布；
- 各主要离散工艺水平的 margin 分布；
- 各 strata 的目标 ROI coverage；
- invalid 与工艺参数的关联表。

若样本量允许，统一 margin 使用各主要 strata 的保守分位数：

[
m^*=max_g Q_{0.90}(m_imid g)
]

strata 定义和最小样本量必须预注册。样本不足时不得伪装成可靠分层分位数，应报告不确定性并使用总体方案。

---

## 17. Phase B3：统一 stable ROI

全程只使用 `boundary_margin_um`，不再引入含义相反的 `r_i`。

总体90%覆盖定义：

[
m_L^*=Q_{0.90}(m_L),quad
m_R^*=Q_{0.90}(m_R)
]

[
m_T^*=Q_{0.90}(m_T),quad
m_B^*=Q_{0.90}(m_B)
]

为了得到统一且居中的 ROI：

[
m_x^*=max(m_L^*,m_R^*)
]

[
m_y^*=max(m_T^*,m_B^*)
]

[
ROI_x=[-100+m_x^*,100-m_x^*]
]

[
ROI_y=[-100+m_y^*,100-m_y^*]
]

若执行工艺分层方案，上述各边分位数替换为对应 strata 中最保守的 `Q0.90`。

最终宽度向下保守取整到配置规定的 5 或 10 μm 整倍数。所有正式样本使用完全相同、关于原点对称的物理 ROI。

必须同时计算 95% coverage 敏感性 ROI，但不得根据结果选择更有利的 ROI。

---

## 18. Phase B4：无效样本与缺失协议

固定 ROI 确定后，重新检查每个样本在该 ROI 内是否仍有明显边界污染。

若存在：

```text
stable_roi_valid = False
```

该样本：

- 不进入依赖 `H_stable` 的 primary morphology analysis；
- 对应 stable-ROI 特征保存为 missing；
- 不插值、不局部修补、不为该样本单独缩小 ROI；
- 继续保留在 registration、boundary artifact 和完整形貌存档中。

必须分析 invalid 是否集中于特定工艺条件。如果 missingness 与工艺参数明显相关，必须明确报告 selection bias 风险，并用 95% coverage ROI 重复下游敏感性分析。

不得静默删除 invalid 样本。

---

## 19. Phase B QA 与验收

输出：

```text
boundary_influence_metrics.csv
boundary_profiles_absolute.csv
boundary_profiles_normalized.csv
boundary_margin_distributions.png
boundary_margin_by_process.png
stable_roi_selection.png
stable_roi_montage_absolute.png
stable_roi_montage_local_contrast.png
```

`boundary_influence_metrics.csv` 至少包含：

```text
sample_id
session_id
margin_left_um
margin_right_um
margin_top_um
margin_bottom_um
margin_method
margin_sensitivity_um
stable_roi_valid
boundary_warning_flags
```

`stable_roi_selection.png` 必须展示：

- 四边 margin 分布；
- 总体和分层 Q0.90；
- Q0.95 敏感性线；
- 取整前后 ROI 尺寸；
- 每个主要工艺层级的实际 coverage；
- invalid 样本数量。

Phase B PASS 必须满足：

- 主判据与 alternative criterion 不产生不可接受的 ROI 漂移；
- 主要工艺层级 coverage 接近预注册目标；
- invalid 样本及其 missingness 已完整报告；
- 90%与95% ROI 均已冻结并保存；
- 人工检查 stable ROI montage 并确认 PASS。

---

## 20. 配置文件

建议 `config/rectangle_registration.yaml`：

```yaml
random_seed: 42
nominal_programmed_region_um: [200.0, 200.0]

input:
  format: vk_csv  # vk_csv or validated_cag
  sample_id_pattern: "configured_explicitly"
  height_sign: upward

sessions:
  manifest: config/session_manifest.csv

coarse_leveling:
  central_exclusion_um: [320.0, 320.0]
  method: asymmetric_sigma_clip
  minimum_reference_valid_fraction: 0.20

calibration:
  selection_fraction: 0.25
  stratify_by: [pulse_width, frequency, velocity, pass, hatch]
  angle_warning_mad_deg: 0.3
  angle_review_mad_deg: 0.8

registration:
  nominal_edge_band_um: 15.0
  center_search_halfwidth_um: 50.0
  primary_weights: [0.5, 0.5]
  sensitivity_weights:
    - [0.25, 0.75]
    - [0.5, 0.5]
    - [0.75, 0.25]
  unstable_shift_um: 3.0

registered_canvas:
  preferred_size_um: [300.0, 300.0]
  minimum_size_um: [260.0, 260.0]
  pixel_um: auto_not_finer_than_coarsest_input
  interpolation: linear
  minimum_interpolated_mask_weight: 0.99
  maximum_gap_um: auto_from_input_pixel_size
  extrapolation: forbidden

final_leveling:
  exclusion_halfwidth_um: 120.0
  minimum_reference_frame_width_um: 10.0
  minimum_reference_valid_fraction: 0.20

boundary:
  profile_halfwidth_um: 60.0
  center_reference_halfwidth_um: 20.0
  gradient_sigma_threshold: 3.0
  minimum_persistence_um: 5.0
  primary_coverage_quantile: 0.90
  sensitivity_coverage_quantile: 0.95
  roi_rounding_um: 10.0

qa:
  save_individual: true
  save_montage: true
  absolute_color_scale: global
  local_color_percentiles: [1.0, 99.0]
```

配置中的 `auto` 必须在运行 manifest 中解析为具体数值，不能只保存字符串。

---

## 21. 推荐代码结构

```text
config/
├── rectangle_registration.yaml
└── session_manifest.csv
src/
├── io_vk.py
├── io_cag.py
├── data_contracts.py
├── leveling.py
├── calibration_selection.py
├── session_geometry.py
├── registration.py
├── canonical_grid.py
├── boundary_analysis.py
├── stable_roi.py
├── qa.py
└── provenance.py
scripts/
├── 00_validate_inputs.py
├── 01_calibrate_sessions.py
├── 02_register_all.py
├── 03_generate_registration_qa.py
├── 04_estimate_boundary_effects.py
├── 05_select_stable_roi.py
└── 06_generate_stable_roi_qa.py
tests/
├── data/
├── test_vk_parser.py
├── test_cag_csv_equivalence.py
├── test_transforms.py
├── test_masked_interpolation.py
├── test_registration_synthetic_transform.py
└── test_margin_quantile.py
```

测试可以使用少量合成数据验证坐标变换、mask 插值和已知平移恢复，但 parser 与正式端到端验收必须使用真实测量文件。

---

## 22. 可复现性与 run manifest

每次运行输出 `run_manifest.json`，至少记录：

```text
timestamp
git_commit_hash
git_worktree_dirty
python_version
package_versions
config_path
config_sha256
resolved_auto_parameters
input_file_list
input_file_sha256
sample_design_mapping
session_definitions
random_seed
theta_by_session
d4_transform_by_session
registered_canvas_size_um
canonical_pixel_um
stable_roi_primary_um
stable_roi_sensitivity_um
warnings
manual_approval_status
```

Phase A 与 Phase B 的人工批准记录必须分别保存，不能只写一个最终 PASS。

---

## 23. 异常处理

以下情况不得静默继续：

- 高度文件无法解析；
- 单位或像素间距缺失；
- CAG 尚未通过同系列 CSV 交叉验证；
- sample-design 映射不唯一；
- session 不明确；
- coarse/final leveling 参考面不足；
- session 角度多峰；
- D4 方向无法确定；
- 中心搜索命中边界；
- 配准对权重敏感；
- `H_reg` 小于 260 μm；
- 目标网格超出原始有效视场；
- 缺失值比例超过阈值；
- 边缘判据敏感性过高；
- 某主要工艺层级 coverage 明显不足。

每个异常必须产生结构化 flag、可读 warning 和对应 QA 图。

---

## 24. 明确禁止事项

1. 禁止逐文件自由 `minAreaRect -> rotate -> crop` 作为正式结果。
2. 禁止在 `H_200` 上估计边缘外侧参考和污染距离。
3. 禁止每个样本自行决定角度、D4 镜像或 stable ROI 尺寸。
4. 禁止依赖单一绝对高度阈值定位全部浅加工样本。
5. 禁止把自由形貌拟合当作独立中心真值。
6. 禁止将 center prior 与未归一化 score 任意线性相加。
7. 禁止逐样本自动翻转高度符号。
8. 禁止向原始 FOV 外推或跨大缺失区插值。
9. 禁止使用 `Q0.10(boundary_margin)` 构造90%安全 ROI。
10. 禁止静默删除 fixed ROI 中仍受污染的样本。
11. 禁止配准前强平滑、PCA 或自动编码器处理。
12. 禁止 PCA 前逐图 min-max 到 [0,1]。
13. 禁止把边缘深沟自动解释为材料本征机制。
14. 禁止把同一加工面的 patches 当作独立实验样本。
15. 禁止在查看测试结果后调整主配准权重、阈值或 coverage 分位数。

---

## 25. Phase A 交付物

```text
outputs/rectangle_registration/
├── phase0/
│   ├── input_inventory.csv
│   ├── sample_design_mapping.csv
│   ├── session_manifest_resolved.csv
│   └── phase0_validation.json
├── calibration/
│   ├── session_geometry.json
│   ├── calibration_sample_ids.csv
│   ├── calibration_coverage_by_process.csv
│   └── theta_sample_distribution.csv
├── registered/
│   ├── H_reg/
│   ├── H_200/
│   └── masks/
├── metrics/
│   └── registration_metrics.csv
├── qa/
│   ├── registration_montage_absolute.png
│   ├── registration_montage_local_contrast.png
│   └── registration_individual/
├── run_manifest.json
└── PHASE_A_APPROVAL.md
```

Phase A 完成后必须停止，等待 `PHASE_A_APPROVAL.md` 被人工明确标记为 PASS。

---

## 26. Phase B 交付物

```text
outputs/rectangle_registration/
├── boundary/
│   ├── boundary_influence_metrics.csv
│   ├── boundary_profiles_absolute.csv
│   ├── boundary_profiles_normalized.csv
│   └── boundary_coverage_by_process.csv
├── stable_roi/
│   ├── H_stable/
│   ├── masks/
│   ├── stable_roi_definition.json
│   └── stable_roi_validity.csv
├── qa/
│   ├── boundary_margin_distributions.png
│   ├── boundary_margin_by_process.png
│   ├── stable_roi_selection.png
│   ├── stable_roi_montage_absolute.png
│   └── stable_roi_montage_local_contrast.png
└── PHASE_B_APPROVAL.md
```

---

## 27. Codex 执行顺序

### Step 0

执行 Phase 0，只检查数据、映射、格式和 session。任何硬性依赖未闭合即 STOP。

### Step 1

用至少一个真实高度文件打通 parser、mask、单位和 coarse leveling。若使用 CAG，先完成同系列 CAG–CSV 数值等价验证。

### Step 2

批量读取全部文件，只输出 metadata、有效比例和 contrast diagnostics。

### Step 3

按工艺空间分层选择 calibration samples。

### Step 4

逐 session 估计连续角度，识别并固定 D4 方向。

### Step 5

固定姿态和200 μm名义尺寸，仅优化逐文件中心平移，并执行权重敏感性分析。

### Step 6

计算公共有效视场，生成带 mask 的 `H_reg`，再执行 final leveling 和派生 `H_200`。

### Step 7

生成完整 Phase A QA 和 manifest。

### Step 8

**停止，等待人工确认 Phase A。**

### Step 9

仅对获批 `H_reg` 执行四边边缘影响估计。

### Step 10

执行总体和工艺分层 coverage 诊断，冻结90%主 ROI 和95%敏感性 ROI。

### Step 11

应用 invalid/missing 协议，生成 `H_stable` 和 Phase B QA。

### Step 12

**停止，等待人工确认 Phase B。**

---

## 28. 最终科学表述边界

本任务完成后只允许声称：

- 建立了 session 一致的形貌配准坐标；
- 保存了理论加工区外部参考面；
- 使用预注册规则估计了边缘轨迹污染距离；
- 为正式分析定义了统一物理尺寸 stable ROI；
- 完整记录了无效样本与选择偏差风险。

不得据此声称：

- 已获得扫描器绝对中心真值；
- 边缘深沟属于材料本征烧蚀机制；
- 当前形貌是充分状态；
- 存在 hidden material memory；
- PCA 模态对应具体材料机制；
- 任一工艺参数对材料机制具有因果效应。

核心原则：

> 先保留外部参考、固定 session 坐标和 D4 方向，再定义统一 ROI；没有独立真值时，只报告可证明的配准重复性，不夸大为绝对精度。

