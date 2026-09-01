# WorkBuddy 执行方案：矩形加工区配准与统一稳定 ROI

> 基准日期：2026-08-30  
> 基准提交：`b31a83f`（原文写 `a3c20d5`，2026-08-30 修订时已过时，重新锚定）  
> 上位方法学规范：`CODEX_TASK_矩形加工区配准与稳定ROI_v2.md`  
> 本文件作用：把 v2 规范转换为可逐项执行、可验收、可中止的工程任务。

## 0. 修订记录

### 2026-08-30 第一次修订（WP0 实况核对后）

执行 WP0 时核对仓库实况，发现原文有 4 处与事实冲突，已就地修正。修正点集中在本文件
§2.1、§2.2、§2.3、§2.5、§3、§5、§6.2、§6.3、§9.4、§10：

| 编号 | 原文 | 修正为 | 依据 |
|---|---|---|---|
| R1 | 基准提交 `a3c20d5` | `b31a83f` | `git rev-parse --short HEAD` |
| R2 | `csv文件/60pass组` 只有 1 个 CSV，缺 29 个 | 30 个 CSV 已齐，但全部为解码器派生 | 目录实测 |
| R3 | paired 映射恒为 `左槽=2m-1，右槽=2m` | slot→sample 必须读 CAG 数据名，禁止公式推导 | 60pass 有 12/30 个数据名回折 |
| R4 | 文件名/数量规则未排除旧命名目录 | 显式排除 `csv文件/_待清理_*` | 该目录含 99 个待删旧名文件 |

### 2026-08-30 第二次修订（WP2 实现后）

| 编号 | 原文 | 修正为 | 依据 |
|---|---|---|---|
| R5 | `height float32` | `height float64` | WP3 等价门禁按 0.0000005 μm 绝对差判高度一致；float32 在 100 μm 量程上的分辨率只有 0.000006 μm，比容差大一个数量级，会把真实分歧变成舍入争论。磁盘 129 GiB 可用，160 张图 float64 约 4 GB |

**R3 仍是本方案最关键的修正。** 若按原文的 `left=2m-1, right=2m` 推导，
`60pass组` 会有 12 个 measurement 的左右样本整组串位，且不会报错。

## 1. 总目标与执行边界

建立下面这条可审计流水线：

```text
CAG / KEYENCE CSV
  -> H_raw + valid_mask_raw
  -> session 姿态标定
  -> 单矩形受约束平移配准
  -> H_reg + valid_mask_reg
  -> H_200
  -> 人工批准 Phase A
  -> 四边污染距离
  -> 统一 H_stable
  -> 人工批准 Phase B
```

本任务只做矩形加工区的解析、配准、稳定 ROI 和 QA，不做 PCA、Mamba、工艺回归、虚拟数据增强或机理解释。

执行时遵守以下优先级：

1. 本文件中的仓库实况和执行顺序；
2. `CODEX_TASK_矩形加工区配准与稳定ROI_v2.md` 的方法学约束；
3. 现有代码只能作为可复用素材，不能因为已经存在就默认正确。

任何硬门禁失败时必须停止后续阶段，保存结构化失败原因；不得把 STOP 降级成 warning。

---

## 2. 当前仓库实况

### 2.1 已确认可用

> 2026-08-30 复核后更新（R1）。

- 当前提交为 `b31a83f`；工作区除本方案文件外干净。
- 固定环境为 CPython 3.12.13，虚拟环境 `.venv/Scripts/python.exe` 可用。
- `scripts/verify_environment.py` 检查结果为 `environment=OK`。
- 核心包版本已固定：NumPy 2.3.5、Pandas 3.0.1、SciPy 1.18.1、Matplotlib 3.10.7、scikit-learn 1.7.2、Pillow 12.3.0。
- PyTorch 未安装，但本任务不需要 PyTorch。
- pytest 未安装；本任务优先使用标准库 `unittest`，不要临时引入未固定依赖。
- 三个 CAG 和三份设计表均存在。
- 设计表与 measurement 的映射规则已由用户确认，写入 `config/session_manifest.csv`。
- 图像规格：2048×1536，XY pitch 0.344174 μm，Z 量化 100 pm，
  物理 FOV 约 705 μm × 529 μm；无效像素哨兵 `raw >= 0xFF000000`。
  以上为前置工作实测结论，WP2 仍需在每个 session 上独立复核。

### 2.2 固定的 session 与样本映射

| session_id | CAG | 设计样本数 | measurement 数 | 每次测量矩形数 | 映射 |
|---|---|---:|---:|---:|---|
| `zro2_120_formal` | `氧化锆/pass实验数据/120正式.cag` | 120 | 120 | 1 | `sample_id = measurement_id`，无歧义 |
| `zro2_60_pass` | `氧化锆/pass实验数据/60Pass组.cag` | 60 | 30 | 2 | **读 CAG 数据名**，禁止推导 |
| `zro2_20_supplement` | `氧化锆/pass实验数据/20补充pass.cag` | 20 | 10 | 2 | **读 CAG 数据名**，禁止推导 |

对双矩形 measurement，文件名 `1 2_高度.csv` 表示同一张测量图中含样本 1 与样本 2 两个矩形；
它不是两个高度文件，也不能把整张图直接当成一个样本。

### 2.2.1 paired 的 slot 与 sample 必须分开表述（R3）

paired measurement 中存在两个不同概念，本方案全文必须区分：

```text
slot      = 图像中的物理位置（沿分离轴的第 1 个 / 第 2 个矩形）
sample_id = 设计表中的加工顺序号
```

**slot → sample_id 的映射只能来自 CAG 容器内的 `MeasurementDataMap / FileItemAccessor`
数据名**（即 `export_height_csv.py --naming data` 读取的那个字段），
禁止用 `left=2m-1, right=2m` 之类公式推导。

原因：`60Pass组.cag` 是蛇形扫描，回折处数据名会交换次序。已实测确认 12/30 个
measurement 的样本号不是 `(奇数, 偶数)` 顺序：

```text
14 13   16 15   18 17   20 19   22 21   24 23
38 37   40 39   42 41   44 43   46 45   48 47
```

`20补充pass` 的 10 个 measurement 数据名均为 `(奇数, 偶数)`，
但这只是巧合，**不得据此认为公式可用**。

### 2.2.2 两个矩形的分离轴尚未确定

设计表中同一 measurement 的两个样本，其 `中心Y` 相差 400–500 μm
（`20补充pass` 为 500 μm，`60Pass组` 为 400–500 μm），`中心X` 基本不变。
因此两矩形是沿**载台 Y 轴**分离的。

但载台 Y 轴对应图像的哪一个轴，取决于该 session 的显微镜采集方向，
属于 WP7 的 D4 标定结果。因此：

- WP5 之前只称 `slot_1 / slot_2`（按数据名 token 顺序），不称"左/右"；
- WP7 确定 D4 后，才能把 `slot_1 / slot_2` 落成图像中的具体方位；
- 若 WP8 检测到两矩形中心的实测次序与数据名 token 次序不一致，
  该 measurement 标记 `slot_assignment_conflict=True` 并 STOP，不得静默按位置重排。

### 2.3 当前高度 CSV 数量

> 2026-08-30 复核后更新（R2、R4）。

| 目录 | 当前数量 | 期望数量 | 状态 |
|---|---:|---:|---|
| `csv文件/120正式` | 120 | 120 | 数量完整；全部 `cag_decoder_derived`，来源与 mask 证据待登记 |
| `csv文件/20补充pass` | 10 | 10 | 数量完整；**含用户手工官方导出**，是唯一候选独立 fixture |
| `csv文件/60pass组` | 30 | 30 | 数量已齐（原缺 29 个已补齐）；全部 `cag_decoder_derived` |
| `csv文件/_待清理_旧索引命名_120正式` | 99 | 0 | **待删除的旧补零命名目录，必须排除** |

`csv文件/60pass组/60Pass组_主.xlsx` 不能自动视为高度矩阵或等价性证据。先识别其来源和内容，再决定是否登记；不得仅凭扩展名使用。

`csv文件/_待清理_旧索引命名_120正式/` 是此前 `--naming index` 导出的旧补零文件
（`001_高度.csv`…`099_高度.csv`），已确认被 `--naming data` 的正确命名取代，
等待用户确认后删除。WP1 的文件名/数量校验必须显式排除任何以 `_待清理_` 开头的目录，
否则 99 个文件会被判为 extra files 而触发 STOP。

### 2.4 当前实质性缺口

1. `export_height_csv.py` 的 `read_height(..., fill_invalid=True)` 默认将 CAG 无效哨兵点用 8 邻域中位数填补，孤立时填 0；这违反 `H_raw + valid_mask_raw` 数据契约。
2. 导出器只写高度 CSV，不写有效 mask；后续无法区分实测点和填补点。
3. 导出器文档声称与官方导出逐像素一致，但仓库没有冻结的 fixture、测试、差异表或 QA 图支持该结论。
4. 由同一个 CAG 解码器生成的 CSV 不能反过来作为该解码器的独立验证真值。
5. `scripts/00_validate_inputs.py` 会发现所有 `*_高度.csv`，但把它们统一标记为 `unassigned`，不核对 session、measurement、命名、来源或数量。
6. Phase 0 验证器无论 CSV 是否存在都会加入 `cag_csv_equivalence_not_established`；尚无可被机器校验的 PASS 证据格式。
7. 已跟踪的 `PHASE0_RESULT.md`、`phase0_validation.json` 和 `run_manifest.json` 是 CSV 生成前的旧结果，仍显示 STOP，且 manifest 指向旧提交 `85c0fea`。
8. `src/`、正式解析器、配准模块、Phase A/B 脚本和测试尚未建立。

在以上缺口关闭前，不允许开始批量 Phase A。

### 2.5 设计表字段映射（新增）

三份设计表字段名完全一致，均为 GBK 编码 CSV：

```text
加工顺序, 中心X, 中心Y, 脉宽, 频率, 线间距, 次数, 速度
```

到 v2 规范中工艺参数的规范化映射固定如下，不得在运行时按列名猜测：

| 设计表原字段名 | 规范化键 | 含义 | v2 中的角色 |
|---|---|---|---|
| `加工顺序` | `sample_id` | 加工顺序号 | 样本主键，1-based |
| `中心X` | `stage_center_x_mm` | 载台 X 坐标（mm） | **中心先验 / 潜在 Level-1 参考** |
| `中心Y` | `stage_center_y_mm` | 载台 Y 坐标（mm） | **中心先验 / 潜在 Level-1 参考** |
| `脉宽` | `pulse_width` | 脉冲宽度 | 分层变量 |
| `频率` | `frequency` | 重复频率 | 分层变量 |
| `线间距` | `hatch` | 扫描线间距（mm，0.002–0.01） | 分层变量 |
| `次数` | `pass` | 扫描次数 | 分层变量 |
| `速度` | `velocity` | 扫描速度 | 分层变量 |

行数核对：120 正式 121 行（含表头）、60pass 61 行、20 补充 21 行，与期望设计样本数一致。

#### 2.5.1 中心先验的坐标等级必须单独判定

`中心X / 中心Y` 是加工软件给出的名义中心位置，属于**独立于形貌的先验信息**。
但它能否作为 v2 §13 的 Level-1（独立物理参考，可报绝对配准误差）使用，
取决于是否存在载台坐标到显微图像坐标的已验证转换。

在 WP7 完成 session 姿态标定之前：

- 只能把它当作**搜索域先验**，用于缩小 center search window；
- 不得用它计算并报告"绝对中心误差"；
- 不得在配准目标函数中加入与它的偏移惩罚项。

WP7 之后若证明载台坐标与图像坐标存在稳定仿射关系，才允许升级证据等级，
并在 run manifest 中记录该判定依据。

---

## 3. 全局不可违反的工程规则

1. 原始 CAG、官方导出 CSV 和设计表只读，不覆盖、不改名、不原地修复。
2. 所有路径从配置或命令行读取，不新增硬编码绝对路径。
3. `H_raw` 中无效点必须为 `NaN`，同时保存独立布尔 `valid_mask_raw`；不得填补后伪装为有效测量。
4. 高度和 mask 始终共同执行裁剪、D4、旋转、平移和重采样。
5. 120 正式与两个 paired session 分开建立解码等价证据，不能用一个系列的结果替代另一个系列。
6. paired measurement 必须生成两个 sample-level 记录，并保留共同的 `measurement_id`、`slot`（`slot_1` / `slot_2`）和由数据名解析出的 `sample_id`。
7. 两个矩形的中心搜索窗必须按 slot 分离且不得重叠；不得允许两个样本匹配到同一矩形。
8. **slot → sample_id 只能读 CAG 数据名，禁止用 `2m-1 / 2m` 或任何公式推导**（见 §2.2.1）。
9. **扫描输入目录时必须排除 `_待清理_*` 目录**（见 §2.3）。
10. session 角度和 D4 在 session 内固定；正式单样本只优化中心平移。
11. `H_reg` 必须保留名义 200 μm 区域外的参考面，且边长不得小于 260 μm。
12. Phase A 未人工 PASS，不执行 Phase B；Phase B 不得反向调参优化 Phase A。
13. 生成数据可被 `.gitignore` 忽略，但关键配置、数值指标、JSON、Markdown 和测试必须纳入版本控制。
14. 所有自动选择规则必须在观察最终结果前写入配置并记录哈希。
15. 设计表的 `中心X / 中心Y` 在 WP7 完成 session 姿态标定前只能作搜索域先验，不得用于报告绝对配准误差（见 §2.5.1）。

---

## 4. 目标代码与产物结构

执行完成后至少形成：

```text
config/
├── rectangle_registration.yaml
├── session_manifest.csv
├── height_source_manifest.csv
└── manual_orientation.yaml

src/
├── data_contracts.py
├── io_cag.py
├── io_vk_csv.py
├── provenance.py
├── leveling.py
├── calibration_selection.py
├── session_geometry.py
├── registration.py
├── canonical_grid.py
├── boundary_analysis.py
├── stable_roi.py
└── qa.py

scripts/
├── 00_validate_inputs.py
├── 00b_validate_cag_csv_equivalence.py
├── 01_inventory_height_maps.py
├── 02_calibrate_sessions.py
├── 03_register_all.py
├── 04_generate_registration_qa.py
├── 05_estimate_boundary_effects.py
├── 06_select_stable_roi.py
└── 07_generate_stable_roi_qa.py

tests/
├── data/
├── test_io_cag.py
├── test_io_vk_csv.py
├── test_mapping.py
├── test_transforms.py
├── test_masked_interpolation.py
├── test_registration_synthetic.py
└── test_margin_quantile.py
```

不要一次写完整流水线再统一测试。严格按下面的工作包顺序推进。

---

## 5. WP0：冻结基线与运行规范

### 工作内容

1. 记录执行开始时的 Git 提交和工作区状态。基准提交为 `b31a83f`，不是 `a3c20d5`。
2. 运行环境检查，不安装 PyTorch。
3. 创建 `config/rectangle_registration.yaml`，把 v2 中所有参数写成具体键。
4. 所有 `auto` 参数允许在源配置中存在，但运行前必须解析成数值并写入 run manifest。
5. 随机数固定为 42。
6. 确认 `config/session_manifest.csv` 已存在且映射规则与 §2.2 一致（该文件已由前置工作建立，不需要重建）。

### 命令

```powershell
git status --short
git rev-parse HEAD
.\.venv\Scripts\python.exe scripts\verify_environment.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

最后一条在测试尚未创建时可暂时报告“0 tests”，WP2 后必须实际通过测试。

### 验收

- 不修改现有原始输入。
- 环境检查通过。
- 配置可被 Python 解析。
- run manifest 记录 Python、依赖、Git、配置 SHA-256 和 dirty 状态。

---

## 6. WP1：建立高度来源清单并补齐 60Pass

### 6.1 新建来源清单

新建 `config/height_source_manifest.csv`，每个 measurement 一行，至少包含：

```text
session_id
measurement_id
n_rectangles
slot_1_sample_id
slot_2_sample_id
cag_data_name
cag_path
csv_path
csv_source_type
csv_export_tool
csv_export_timestamp
csv_sha256
expected_width
expected_height
expected_dx_um
expected_dy_um
provenance_status
```

`slot_1_sample_id` / `slot_2_sample_id` 直接来自 `cag_data_name` 的两个 token
（单矩形时 `n_rectangles=1`，`slot_2_sample_id` 留空）。
**不得**在此处重排成升序，也不得补一列"推导出的期望样本号"来校验它们。

`csv_source_type` 只能使用：

- `keyence_official_export`：由 KEYENCE 软件独立导出；可作为等价 fixture。
- `cag_decoder_derived`：由仓库解码器生成；可作为处理输入，但不能作为独立验证真值。
- `unknown`：来源无法证明；不能作为等价 fixture。

不得通过文件名外观猜测来源。没有证据时写 `unknown`。

### 6.2 文件名与数量规则

- 120 正式：measurement 1–120；文件名 `1_高度.csv` … `120_高度.csv`（**不补零**）。
- 20 补充：measurement 1–10；标签 `1 2`、`3 4`、…、`19 20`。
- 60Pass：measurement 1–30；标签覆盖 30 对样本，**但顺序不保证为 `(奇数, 偶数)`**（见 §2.2.1）。
- 用自然数解析文件名，不使用字符串字典序。
- 检查重复 ID、缺号、多余文件、空文件和无法解析名称。
- **扫描前排除 `_待清理_*` 目录**，其内的旧补零命名文件不参与任何校验。
- paired 文件名解析出的两个 token 只写入 `slot_1_sample_id` / `slot_2_sample_id`；
  不得在解析阶段重排成升序，也不得校验"第一个 token 必须是奇数"。

### 6.3 补齐 60Pass 的原则

> 2026-08-30 状态更新：原缺的 29 个 CSV 已由 `export_height_csv.py` 补齐，
> `csv文件/60pass组` 现有 30 个 CSV。但补齐动作**不等于**缺口关闭。

已补齐的 29 个文件存在两个必须先处理的问题：

1. 它们是用旧的默认 `fill_invalid=True` 路径生成的，无效哨兵点已被 8 邻域中位数填补，
   违反 `H_raw + valid_mask_raw` 契约。**WP2 完成 mask 修正后必须重新导出并覆盖这些派生文件**
   （它们是派生数据，不是原始输入，允许覆盖；官方导出文件永不覆盖）。
2. 文件名为 CAG 数据名，其中 12 个是回折次序（§2.2.1）。登记时必须逐文件记录解析结果，
   不得默认 `slot_1 < slot_2`。

> **G9 已关闭（2026-08-30）**：不需要重新导出。
>
> 全量核查结果：三个容器全部 **160 个 measurement 的无效像素总数 = 0**
> （`session_decode_probe.json`，`--groups 0`）。`fill_invalid` 只改写
> `raw >= 0xFF000000` 的像素，一张图上都不存在，因此旧脚本的补填是**空操作**，
> 产物与新的 `preserve_nan` 路径逐字节相同。
>
> 佐证：60pass 全部 30 个 CSV 与新导出 **30/30 SHA-256 完全一致**
> （`phase0/derived_csv_parity_60pass.json`）。§6.3 第 1 条的覆盖动作因此取消。
> 120 正式的 120 个派生 CSV 同理（同源、同样零无效点）。
>
> 注意：这**不是**“mask 语义已验证”。由于数据中根本不存在无效点，
> WP3 §8.3 的“无效点”比较项退化为真空命题，须如实报告为
> “双方均无无效点，官方 mask 表达规则仍未获得证据”。

当前 provenance 判定：

| 目录 | 已确认的官方导出 | 判定 |
|---|---|---|
| `csv文件/120正式` | 0 / 120 | 全部 `cag_decoder_derived` |
| `csv文件/20补充pass` | 10 / 10 | 用户手工导出，WP1 需逐文件核实后登记为 `keyence_official_export` |
| `csv文件/60pass组` | 0 / 30 | 全部 `cag_decoder_derived` |

**不得因为文件数量齐全就让 CAG–CSV 等价门禁 PASS。**
`cag_decoder_derived` 永远不能作为独立验证真值（见 §8.2）。

### 6.4 产物

```text
config/height_source_manifest.csv
outputs/rectangle_registration/phase0/height_file_inventory.csv
outputs/rectangle_registration/phase0/height_inventory_validation.json
```

### 硬验收

- 120：120/120 measurement 均可解析。
- 20：10/10 measurement 均可解析。
- 60：30/30 measurement 均可解析。
- 每个 measurement 与映射表唯一连接。
- 缺号、重复或 provenance 未登记时 STOP。

---

## 7. WP2：重构解析器，先保护 raw mask

### 7.1 数据契约

在 `src/data_contracts.py` 定义：

```python
@dataclass(frozen=True)
class HeightMap:
    z: np.ndarray
    valid_mask: np.ndarray
    dx_um: float
    dy_um: float
    x_um: np.ndarray
    y_um: np.ndarray
    metadata: dict
```

构造时强制检查：

- `z.shape == valid_mask.shape`；
- `valid_mask.dtype == bool`；
- `z[~valid_mask]` 全为 `NaN`；
- `z[valid_mask]` 全为有限数；
- `dx_um > 0`、`dy_um > 0`；
- 坐标数组长度与矩阵一致。

### 7.2 CAG 解析器

把可复用逻辑从 `export_height_csv.py` 移入 `src/io_cag.py`。

必须修改：

1. 删除正式读取路径中的默认填补行为。
2. `raw >= 0xFF000000` 形成 `valid_mask=False`。
3. 无效位置转换成 `NaN`，原始哨兵值不能参与 min/max、leveling 或配准。
4. LUT 偏移 776 必须通过结构检查和同系列 fixture 验证，不能只依赖常量注释。
5. metadata 保存原始 group、尺寸、pitch、Z 量化、无效点数、时间戳、内部条目名和源文件 SHA-256。
6. 解析函数不得生成“修复后”的 raw 数据。

`export_height_csv.py` 改为调用 `src/io_cag.py`。如保留兼容导出，必须提供明确选项：

#### 7.2.1 调色板偏移 776 的判定方法（2026-08-30 实测结论）

原文只说“必须通过结构检查”，未规定方法。实现时确立了**两级判据**：

**（一）结构推导（决定性，硬门禁）**

VK4 头部的 18 项段偏移表给出了每个段的字节起点，而各段紧密相邻。于是调色板
长度可以直接量出来，与样本形貌无关：

```text
palette = 下一段起点 - 高度段起点 - 20 - 宽×高×4
```

`src/io_cag.py::derive_lut_bytes()` 实现该式。三个容器、全部 160 个 measurement
的推导结果**全部为 776 字节**。这是 O(1) 操作，每次读取都执行且失败即抛错。

**（二）形貌接缝（佐证，不单独成门禁）**

错位读取会把图像水平滚动 `Δoffset/4` 个采样，每行出现一道竖向接缝。其台阶高度
**恰好等于卷绕跳变** `|z[r][0] − z[r−1][−1]|`。实测各 session 的分辨力：

| session | 接缝比（正确偏移） | 卷绕跳变 / 中位列梯度 = 分辨力上限 | 结论 |
|---|---:|---:|---|
| 120 正式 | 10.9 ~ 16.5 | 41 ~ 55 | 判据有效 |
| 20 补充 | 3.0 ~ 5.8 | 15 ~ 18 | 判据有效 |
| 60pass | 2.7 ~ 5.3 | **4.9 ~ 7.5** | **判据失效** |

失效原因：60pass 的加工矩形位于视场中部，左右两侧都是未加工原始面且几乎同高
（左缘 44.457 μm vs 右缘 45.857 μm），卷绕跳变只有 1.417 μm，与真实加工边台阶
（1.039 μm，1918–1923 列连续 6 列）同量级。**这是样本几何决定的，不是解码错误。**

因此门禁规则定为：

- 结构推导失败 → STOP；
- 结构通过、接缝判据有分辨力但与结构矛盾 → STOP（真冲突）；
- 结构通过、接缝判据无分辨力 → PASS，并在报告中标记
  `seam_has_power_on_this_sample=false`，写明结论仅由结构推导支撑。

**没有分辨力的检查不能拥有否决权**，否则一个已知失效的启发式会否决一个按构造
精确的测量。`src/io_cag.py` 同时输出 `seam_headroom` 供人工复核。

真正的独立旁证是 WP3 的字节级复现：20 补充 pass 第 2 组的派生导出与官方
`3 4_高度.csv` **SHA-256 完全一致**，一次比对同时钉死了偏移、舍入、行主序和单位换算。

```text
--invalid-policy preserve_nan   # 默认且用于科学流水线
--invalid-policy keyence_compat # 仅用于格式复现验证，产物标记 derived
```

禁止默认使用 `keyence_compat`。

### 7.3 KEYENCE CSV 解析器

在 `src/io_vk_csv.py` 实现 `parse_vk_csv(path) -> HeightMap`：

- 自动识别 GBK/UTF-8-sig，但把最终编码写入 metadata；
- 解析非标准头部，不用固定 DataFrame 行号代替字段校验；
- 校验宽、高、XY calibration、单位和矩阵行列数；
- 单位统一转换到 μm；
- 明确定义 CSV 无效值规则；不得把数值 0 当成无效；
- 若 CSV 格式没有有效 mask 信息，metadata 写 `mask_source=unavailable`，不得伪造全有效 mask 后宣称 mask 等价。

### 7.4 原生中间格式

科学流水线不要用只含数值的 CSV 作为唯一中间格式。每张 measurement 保存：

```text
height float64        # R5：原文写 float32，见 §0
valid_mask bool
dx_um / dy_um
metadata JSON
```

> **R5 依据**：WP3 的 CAG–CSV 等价门禁要求有效像素高度"完全一致"，实操容差
> 是 0.0000005 μm（§8.3）。float32 的相对分辨率为 6e-8，在 100 μm 量程上即
> 0.000006 μm 误差，比容差大 12 倍。一旦中间缓存用 float32，重跑时出现的任何
> 微小差异都无法区分是解码分歧还是舍入。缓存因此保持位精确；代价是 160 张
> 1536×2048 的图约 4 GB（磁盘实测可用 129 GiB）。

可使用 `.npz`，但必须提供读写 round-trip 测试。若使用其他格式，不能新增未固定依赖。

### 7.5 单元测试

至少覆盖：

- raw sentinel 到 `NaN + False`；
- 有效零高度仍保持有效；
- 半向上 0.001 μm 舍入；
- CSV GBK 头解析；
- 尺寸、单位、行列错误会失败；
- `.npz` round trip 不改变高度和 mask；
- 旧 `_fill_invalid` 不会出现在正式路径。

### 硬验收

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

全部通过；并在三个 session 各读取至少一个真实 measurement，输出尺寸、pitch、有限比例和无效点数。任何 session 解析失败即 STOP。

#### 7.5.1 WP2 验收结果（2026-08-30）

单元测试：**53 项全部通过**（含 5 项打在真实 `20补充pass.cag` 上的端到端检查）。

`scripts/probe_session_decode.py --groups 0` 全量结果：

| session | measurement | 尺寸 | pitch (μm) | 有效比例 | 无效点 | 推导调色板 | 结构校验 |
|---|---:|---|---|---:|---:|---:|---|
| 120 正式 | 120/120 | 2048×1536 | 0.344174 | 1.0000 | 0 | 776 | 120/120 通过 |
| 60pass | 30/30 | 2048×1536 | 0.344174 | 1.0000 | 0 | 776 | 30/30 通过 |
| 20 补充 | 10/10 | 2048×1536 | 0.344174 | 1.0000 | 0 | 776 | 10/10 通过 |

`decision=PASS`，零 blocker。视场统一为 704.87 × 529.13 μm。

slot 映射在容器内部复核：60pass 的 **12 个回折数据名确认存在**，位于 group
7–12（`14 13`、`16 15`、`18 17`、`20 19`、`22 21`、`24 23`）与 group
19–24（`38 37`、`40 39`、`42 41`、`44 43`、`46 45`、`48 47`）。
三个 session 的 slot 多重集与设计表样本号集合逐一相等（120=120、60=60、20=20）。

产物：

```text
outputs/rectangle_registration/phase0/session_decode_probe.json
outputs/rectangle_registration/phase0/session_decode_probe.csv
outputs/rectangle_registration/phase0/derived_csv_parity_60pass.json
```

---

## 8. WP3：建立独立 CAG–CSV 等价证据

### 8.1 fixture 选择

每个 session 至少选 3 个 measurement，覆盖浅、中、深形貌。选择规则在比较前冻结：

1. 首先按设计参数覆盖 pass、频率、速度、hatch；
2. 再用不依赖单阈值的 contrast diagnostics 选浅、中、深；
3. 保存选择表，不得在看到误差后更换样本。

paired measurement 的一张 CSV 同时覆盖左、右两个矩形，但等价验证比较的是完整 measurement 高度矩阵，不先拆 ROI。

### 8.2 独立性要求

- fixture 必须是 `keyence_official_export`。
- `cag_decoder_derived` 和 `unknown` 不得计入 3 个独立 fixture。
- 120 个脚本导出文件即使数量完整，也不能自证解码器正确；60pass 新补齐的 29 个同理。
- 若某 session 不足 3 个官方 fixture，报告明确缺少哪些 measurement，并 STOP；不要伪造 PASS。

#### 8.2.1 当前 fixture 供给预测（2026-08-30）

| session | 候选官方导出 | 是否满足每 session ≥3 | 预测 |
|---|---:|---|---|
| `zro2_20_supplement` | 10（需 WP1 逐文件核实） | 是 | 有望 PASS |
| `zro2_120_formal` | 0 | 否 | **预计 STOP** |
| `zro2_60_pass` | 0 | 否 | **预计 STOP** |

若预测成立，正确的执行结果是在 WP3 停止，输出缺口清单，由用户从 KEYENCE 软件
为 120 正式和 60pass 各补导出至少 3 个 measurement 的官方 CSV。
**不得**用以下任何方式绕过：用 120 正式的结果替代另两个系列、用脚本导出文件自证、
用 20 补充的结论外推到其他 session、或把 `cag_decoder_derived` 改标为官方来源。

### 8.3 比较项目与预注册容差

对每个 fixture 比较：

| 项目 | 通过条件 |
|---|---|
| width / height | 完全一致 |
| dx / dy | 与 CSV 三位 nm 精度一致，绝对差不超过 0.0000005 μm |
| 高度单位 | 都能确定转换为 μm |
| 有效像素高度 | CAG 值按 0.001 μm 半向上舍入后与官方 CSV 完全一致；同时报告 max/median/RMSE |
| 有效 mask | 若官方文件含 mask，逐像素完全一致 |
| 无效点 | 数量、位置和官方表达规则有书面结论 |
| 方向 | 四角和至少 5 个固定像素坐标对应，禁止转置、翻转和循环移位 |

如果官方 CSV 对无效点做了软件填补：

1. 高度等价只在 CAG 原始有效像素上判断；
2. 另行验证官方填补值仅用于兼容复现；
3. 正式流水线仍以 CAG raw sentinel 生成 mask；
4. 报告必须写明“CSV 无法独立证明 mask”，不能声称完整 mask 等价；
5. 若没有另一种官方 mask 证据，Phase 0 对“高度解码”可单独 PASS，但“mask 语义”保持 STOP，后续不得运行。

### 8.4 实现和产物

实现 `scripts/00b_validate_cag_csv_equivalence.py`，读取来源清单，不硬编码 fixture。

输出：

```text
outputs/rectangle_registration/phase0/equivalence/
├── fixture_selection.csv
├── cag_csv_equivalence_metrics.csv
├── cag_csv_equivalence.json
├── qa/<session>_<measurement>_difference.png
└── qa/<session>_<measurement>_mask.png
```

`cag_csv_equivalence.json` 至少包含代码提交、配置哈希、CAG/CSV 哈希、fixture 独立来源、每项容差、逐 session 结论和总决策。

### 硬验收

三个 session 必须分别为 `PASS`。任一 session 的 fixture 数不足、来源不独立、高度不一致、方向不一致或 mask 语义未闭合，Phase 0 为 STOP。

---

## 9. WP4：升级 Phase 0 验证器

修改 `scripts/00_validate_inputs.py`，使它真正消费 WP1–WP3 的证据。

### 必须实现

1. 通过 `height_source_manifest.csv` 给每个 CSV 指定 session 和 measurement，不再产生 `session_id=unassigned`。
2. 对三个 session 分别检查期望 measurement 集合、文件存在性、命名、尺寸、pitch、单位和来源。
3. 检查映射唯一性：120 产生 120 个 sample 行；60 产生 60 个；20 产生 20 个；总计 200 个 sample 行；`sample_id` 全集恰好覆盖设计表的 `加工顺序`。
4. paired 的 slot→sample 必须来自 CAG 数据名，**禁止校验 `left=2m-1`、`right=2m`**（R3）。校验内容改为：
   - 每个 measurement 的数据名解析出且仅解析出 2 个 `sample_id`；
   - 两个 `sample_id` 全局唯一，不与其他 measurement 重复；
   - 同一 measurement 的两个 `sample_id` 在设计表中对应的 `中心X / 中心Y` 应彼此邻近；
     该项仅作软提示写入报告，不作硬门禁（因为 paired 分组依据来自数据名，不由坐标决定）。
5. 扫描输入目录时排除 `_待清理_*`，并对剩余文件检查缺号、重复和无法解析名称（R4）。
6. 读取 `cag_csv_equivalence.json`，校验其输入 SHA-256、配置 SHA-256 和当前解析代码版本未过期。
7. 只有三个 session 的等价结论都通过才移除 `cag_csv_equivalence_not_established`。
8. 正式验收不得带 `--skip-hashes`。
9. 自动重写 `PHASE0_RESULT.md`，不允许 Markdown 结论与 JSON 不一致。
10. 旧产物 `PHASE0_RESULT.md`、`phase0_validation.json`、`run_manifest.json` 指向提交 `85c0fea`，
    全部作废并由本次运行重写；不得沿用旧 JSON 的任何结论。

### 运行

```powershell
.\.venv\Scripts\python.exe scripts\00_validate_inputs.py
```

### 产物和验收

```text
outputs/rectangle_registration/phase0/input_inventory.csv
outputs/rectangle_registration/phase0/height_file_inventory.csv
outputs/rectangle_registration/phase0/sample_design_mapping.csv
outputs/rectangle_registration/phase0/session_manifest_resolved.csv
outputs/rectangle_registration/phase0/phase0_validation.json
outputs/rectangle_registration/phase0/PHASE0_RESULT.md
outputs/rectangle_registration/run_manifest.json
```

必须满足：

- `decision == PASS`；
- blockers 为空；
- session 数为 3；
- measurement 总数为 160；
- sample 总数为 200；
- manifest 的 Git 提交、dirty 状态和输入哈希与本次运行一致。

未达到以上条件就停止，不得创建 Phase A 正式产物。

---

## 10. WP5：全量 metadata、mask 与 contrast 盘点

实现 `scripts/01_inventory_height_maps.py`，只解析和诊断，不配准。

### paired measurement 拆分策略

这一阶段不对原图做破坏性裁剪。为每个 sample 建立逻辑视图：

```text
session_id
measurement_id
sample_id
slot = single | slot_1 | slot_2
slot_sample_id_source = cag_data_name
shared_height_source_id
initial_search_region_um
```

paired measurement 的两个初始搜索域由图像中心和物理 FOV 确定，两个搜索域不得重叠到可匹配同一矩形。正式中心仍由 WP8 优化。

在 D4 标定完成前，不写 `left` / `right`，只写 `slot_1` / `slot_2`（对应数据名的两个 token）。
D4 确定后由 WP8 追加 `slot_axis` 字段（例如 `slot_1_is_image_left=true`），
但 `sample_id` 与 slot 的绑定关系不再改变。

参考量级：图像物理 FOV 约 705 μm × 529 μm（2048×1536 @ 0.344174 μm）；
同一 measurement 的两个样本载台 `中心Y` 相差 400–500 μm，
因此两矩形在图像中会接近 FOV 边缘。**这一量级必须在 WP5 用真实数据复核**，
若实测间距与载台坐标推算值差异超过 10%，先查 FOV 与 pitch，不得继续。

### 输出字段

每个 measurement 和 sample 至少记录：

- shape、dx/dy、物理 FOV；
- valid fraction、无效连通域统计；
- Q01/Q05/Q50/Q95/Q99；
- negative-tail amplitude；
- edge energy；
- candidate modified-area fraction；
- 初始 reference-frame coverage；
- 解析 warning。

### 产物

```text
outputs/rectangle_registration/inventory/measurement_metrics.csv
outputs/rectangle_registration/inventory/sample_view_manifest.csv
outputs/rectangle_registration/inventory/contrast_diagnostics.csv
outputs/rectangle_registration/inventory/inventory_summary.json
```

### 硬验收

- 160 个 measurement 全部可读，200 个 sample view 唯一。
- 三组尺寸和 pitch 分布明确。
- 无效点未被填补，`NaN` 数与 mask false 数一致。
- paired 左右搜索域有序且不重复。

---

## 11. WP6：coarse leveling 与 calibration sample 冻结

### coarse leveling

实现 `src/leveling.py`：

- 优先使用中央排除区外的 reference frame；
- 使用 IRLS/Huber 或固定非对称 sigma clipping；
- 不对全图直接普通最小二乘；
- 检查参考点面积、x/y 跨度、象限覆盖和残差；
- 输出 plane a/b/c、RMSE、reference fraction 和 warning。

### calibration 选择

实现 `src/calibration_selection.py`：

- 先按脉宽、频率、速度、pass、hatch 分层；
- 层内按固定 contrast 规则选高 SNR；
- 选择比例和最小层样本数写入配置；
- 不得只选最深的 top 25%；
- paired 样本按独立加工矩形计入设计覆盖，但保留 measurement 聚类关系。

### 产物

```text
outputs/rectangle_registration/calibration/calibration_sample_ids.csv
outputs/rectangle_registration/calibration/calibration_coverage_by_process.csv
outputs/rectangle_registration/metrics/coarse_leveling_metrics.csv
```

### 验收

- 三个 session 都有 calibration 样本。
- 主要工艺水平覆盖表完整。
- reference frame 不足的样本有 hard flag，未被静默删除。
- 选择规则和样本名单在姿态拟合前冻结并写入 manifest。

---

## 12. WP7：session 连续角度与 D4 标定

实现 `src/session_geometry.py` 和 `scripts/02_calibrate_sessions.py`。

### 连续角度

- 只在 calibration samples 上允许自由拟合中心、角度和矩形边界。
- 每个 session 将角度折算到 `[-45°, 45°)`。
- 使用预先定义的质量权重和稳健统计得到 `theta_session_deg`。
- `MAD > 0.3°` warning，`MAD > 0.8°` hard review。
- 多峰时停止，不强行平均。

### D4

- 每个 session 只允许一个固定 D4 变换。
- 依据 pooled morphology、原始显微镜方向或人工确认决定。
- 不允许单样本按“更深的一边”自动翻转。
- 把人工选择写入 `config/manual_orientation.yaml`，包含确认人、日期、依据和 QA 图路径。

### paired 特殊验收

- 应用 session D4 后仍能唯一恢复两个 slot 的样本身份。
- 若 D4 会使 `slot_1` 在图像中跑到 `slot_2` 之后，必须在映射层显式记录该几何事实
  （写入 `slot_axis` / `slot_order_in_image`），
  **但 `sample_id` 与 slot 的绑定不变**，不能悄悄按图像位置重排 sample_id。

### 产物

```text
outputs/rectangle_registration/calibration/session_geometry.json
outputs/rectangle_registration/calibration/theta_sample_distribution.csv
outputs/rectangle_registration/qa/session_orientation/
```

### 人工门禁 A0

如果任一 session 的 D4 未确认或角度多峰，停止。只有 `manual_orientation.yaml` 中三个 session 都是 `confirmed` 才能执行 WP8。

---

## 13. WP8：只优化中心的单样本正式配准

实现 `src/registration.py` 和 `scripts/03_register_all.py`。

### 固定项

- `theta_session_deg` 固定；
- `d4_transform_session` 固定；
- 名义矩形固定为 200 μm × 200 μm；
- 高度符号按 session 固定；
- 单样本只优化 `(center_x_um, center_y_um)`。

### paired measurement

- 同一 measurement 的两个中心应联合或带互斥约束求解。
- `slot_1` 中心必须落在 slot_1 搜索域，`slot_2` 中心落在 slot_2 搜索域；
  两个搜索域沿分离轴划分，不重叠。
- 两中心最小间距、次序和边界写入配置。
- 若两个解落到同一候选矩形，两个样本都标记失败，不择优保留一个。
- 若解出的中心次序与数据名 token 次序相反，标记 `slot_assignment_conflict=True`
  并对该 measurement 的两个样本同时置 `registration_status=failed`，
  **不得按位置静默交换 sample_id**（见 §2.2.2）。

### 目标函数

- region score 和 edge score 分别无量纲化；
- 主权重固定 `[0.5, 0.5]`；
- 同时运行 `[0.25,0.75]`、`[0.5,0.5]`、`[0.75,0.25]`；
- 权重导致中心跨度超过 3 μm 时 `registration_unstable=True`；
- center prior 只限制搜索域，不进入目标函数。

### 测试

- 合成平移恢复；
- 浅槽、强槽、缺边和 mask 缺失；
- paired 两矩形不串位；
- 搜索命中边界会失败；
- 权重敏感性标志正确。

### 产物

```text
outputs/rectangle_registration/metrics/registration_metrics_pregrid.csv
outputs/rectangle_registration/transforms/sample_transforms.jsonl
```

---

## 14. WP9：公共视场、mask-aware 重采样与 final leveling

实现 `src/canonical_grid.py`，完成 `H_reg` 和 `H_200`。

### 网格

1. 先变换真实有效视场边界，计算每个 session 的公共交集。
2. `L_reg = min(300 μm, L_common)`，且必须 `L_reg >= 260 μm`。
3. canonical pixel 不细于全体输入中最粗原始步长。
4. 主插值为 linear。
5. 用 normalized interpolation 同时变换 `z*M` 和 `M`。
6. 只有 mask 权重超过配置阈值且距真实有效点不超过上限时才有效。
7. 禁止外推和跨大缺失区插值。

### final leveling

- 只用 `H_reg` 中名义加工区外的 reference frame；
- 默认排除半宽 120 μm，但必须检查剩余参考宽度；
- 检查面积、跨度、象限和 robust residual；
- final leveling 失败时不得导出可供下游使用的 `H_200`。

### 保存格式

每个 sample 的 `H_reg` 与 `H_200` 必须一起保存 height、mask、x/y、transform 和 leveling metadata。文件名必须含 session_id 与 sample_id，避免三个系列 ID 冲突。

### 产物

```text
outputs/rectangle_registration/registered/H_reg/
outputs/rectangle_registration/registered/H_200/
outputs/rectangle_registration/registered/masks/
outputs/rectangle_registration/metrics/registration_metrics.csv
```

### 硬验收

- 三个 session 都有 `L_reg >= 260 μm`，否则对应 session STOP。
- 没有坐标外推。
- 每个 height 与 mask 同形同变换。
- `H_200` 精确对应 canonical `[-100,100] × [-100,100] μm`。
- final reference frame 失败样本明确为 failed。

---

## 15. WP10：Phase A QA 与人工批准

实现 `scripts/04_generate_registration_qa.py`。

### 每样本 QA 必须显示

1. raw height；
2. coarse-leveled height；
3. raw/registered valid mask；
4. 理论 ±100 μm 边界；
5. `H_reg` 和 `H_200` 边界；
6. 中心、session angle、D4；
7. region/edge score；
8. 权重敏感性位移；
9. leveling 残差；
10. warning/failed 状态。

同时生成统一绝对色标和 local-contrast 两套 montage。local 图必须标注：

```text
LOCAL CONTRAST — NOT COMPARABLE IN ABSOLUTE DEPTH
```

### 自动验收摘要

报告以下比例和分层结果：

- registration PASS/review/failed；
- 搜索边界命中率；
- 权重不稳定率；
- shallow 样本失败率；
- final leveling 失败率；
- registered valid fraction；
- paired 左右串位检查；
- 按 session 和主要工艺参数分组的失败分布。

### 人工门禁 A

生成 `outputs/rectangle_registration/PHASE_A_APPROVAL.md`，默认：

```text
status: PENDING
```

WorkBuddy 必须在此停止。只有用户人工检查 montage 后把状态明确改为 `PASS`，才允许执行 WP11。不得由脚本替用户批准。

---

## 16. WP11：四边污染距离估计

仅在 Phase A 人工 PASS 后实现和运行 `scripts/05_estimate_boundary_effects.py`。

### 输入约束

- 只读取已批准的 `H_reg`；
- 不从 `H_200` 估计外侧参考或边缘污染；
- 使用统一深度定义 `D = z_reference - z`；
- 四边独立估计向内 margin。

### 主判据

- profile 半宽、平滑和中心参考区由配置固定；
- 梯度阈值使用中心背景 median + k×MAD；
- 异常必须持续至少 `minimum_persistence_um`；
- 从理论边界向中心搜索“污染消失并持续稳定”的位置；
- 不允许单个尖峰决定 margin。

同时运行一个预注册 alternative criterion，保存两者差异 `margin_sensitivity_um`。

### 产物

```text
outputs/rectangle_registration/boundary/boundary_influence_metrics.csv
outputs/rectangle_registration/boundary/boundary_profiles_absolute.csv
outputs/rectangle_registration/boundary/boundary_profiles_normalized.csv
outputs/rectangle_registration/boundary/boundary_coverage_by_process.csv
```

---

## 17. WP12：工艺分层覆盖与统一 stable ROI

实现 `scripts/06_select_stable_roi.py`。

### 分层

检查 margin 与脉宽、频率、速度、pass、hatch 的分布关系。只做覆盖诊断，不作因果解释。

若 strata 样本量达到配置下限：

```text
每边安全 margin = 所有主要 strata 中该边 Q0.90 的最大值
```

否则使用总体 Q0.90，并明确报告分层证据不足。

### ROI 规则

```text
m_x = max(m_left, m_right)
m_y = max(m_top, m_bottom)
ROI_x = [-100 + m_x, 100 - m_x]
ROI_y = [-100 + m_y, 100 - m_y]
```

- 宽度向下保守取整到配置的 10 μm 整倍数；
- 所有正式样本使用相同、关于原点对称的物理 ROI；
- 同时冻结 Q0.95 敏感性 ROI；
- 不允许看结果后改用更有利的分位数。

### invalid/missing 协议

固定 ROI 后逐样本复查边界污染和 mask coverage。失败样本：

- `stable_roi_valid=False`；
- stable 特征保持 missing；
- 不填补、不单独缩 ROI、不静默删除；
- 保留在 registration 与 boundary 存档；
- 检查 missingness 是否集中在特定工艺条件。

### 产物

```text
outputs/rectangle_registration/stable_roi/H_stable/
outputs/rectangle_registration/stable_roi/masks/
outputs/rectangle_registration/stable_roi/stable_roi_definition.json
outputs/rectangle_registration/stable_roi/stable_roi_validity.csv
```

---

## 18. WP13：Phase B QA、最终 manifest 与停止点

实现 `scripts/07_generate_stable_roi_qa.py`。

### QA 产物

```text
outputs/rectangle_registration/qa/boundary_margin_distributions.png
outputs/rectangle_registration/qa/boundary_margin_by_process.png
outputs/rectangle_registration/qa/stable_roi_selection.png
outputs/rectangle_registration/qa/stable_roi_montage_absolute.png
outputs/rectangle_registration/qa/stable_roi_montage_local_contrast.png
```

`stable_roi_selection.png` 必须显示四边分布、总体/分层 Q0.90、Q0.95、取整前后尺寸、各工艺层 coverage 和 invalid 数。

更新最终 `run_manifest.json`，至少记录：

- Git 与环境；
- 所有输入/配置哈希；
- 来源与 mapping；
- resolved auto 参数；
- session angle 与 D4；
- canonical grid；
- 90%/95% ROI；
- warning 和失败；
- Phase A/B 人工审批状态。

生成 `PHASE_B_APPROVAL.md`，默认 `PENDING`，然后停止。不得由脚本自行标记 PASS。

---

## 19. 必跑测试矩阵

### 每个工作包后

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
git status --short
```

### Phase 0 正式验收

```powershell
.\.venv\Scripts\python.exe scripts\00b_validate_cag_csv_equivalence.py
.\.venv\Scripts\python.exe scripts\00_validate_inputs.py
```

### Phase A 正式运行

```powershell
.\.venv\Scripts\python.exe scripts\01_inventory_height_maps.py
.\.venv\Scripts\python.exe scripts\02_calibrate_sessions.py
# 人工确认三个 session 的 D4
.\.venv\Scripts\python.exe scripts\03_register_all.py
.\.venv\Scripts\python.exe scripts\04_generate_registration_qa.py
# STOP，等待 PHASE_A_APPROVAL.md = PASS
```

### Phase B 正式运行

```powershell
.\.venv\Scripts\python.exe scripts\05_estimate_boundary_effects.py
.\.venv\Scripts\python.exe scripts\06_select_stable_roi.py
.\.venv\Scripts\python.exe scripts\07_generate_stable_roi_qa.py
# STOP，等待 PHASE_B_APPROVAL.md = PASS
```

所有正式脚本必须支持 `--config`、`--output-dir` 和 `--dry-run`；失败返回非零退出码。

---

## 20. WorkBuddy 每阶段汇报格式

每完成一个 WP，只汇报以下内容：

```text
WP 编号：
状态：PASS / STOP
修改文件：
运行命令：
测试结果：
关键数值：
新产物：
阻断项：
下一步：
```

禁止只说“已经完成”而不提供命令、产物和关键数值。禁止在 STOP 后继续执行下一个 WP。

---

## 21. 建议提交边界

如由 WorkBuddy 负责提交，建议每个门禁一个提交，避免把方法修正和批量结果混在一起：

1. `fix: preserve CAG invalid mask and add height data contracts`
2. `test: establish same-series CAG CSV equivalence evidence`
3. `feat: close phase0 inventory mapping and provenance gates`
4. `feat: add session calibration and constrained registration`
5. `feat: export mask-aware H_reg H_200 and phase A QA`
6. `feat: estimate boundary margins and freeze stable ROI`

提交前必须确认没有把 CAG、全量高度 CSV、`.npz` 高度阵列或大批 PNG 强制加入 Git。

---

## 22. 最终完成定义

只有同时满足以下条件，工程实现才算完成：

- 三个 session 的输入、来源、mapping 和独立等价证据均闭合；
- Phase 0 JSON 为 PASS 且无 blocker；
- 160 个 measurement、200 个 sample 均有可审计记录；
- raw mask 从解析到 H_stable 全程保留，无静默填补；
- session angle 与 D4 已固定并有人工证据；
- 正式配准只优化单样本中心；
- paired 左右身份没有交换或重复匹配；
- 所有合格 session 的 `H_reg >= 260 μm`；
- final leveling 只使用 H_reg 外侧参考面；
- Phase A QA 完整且人工 PASS；
- Q0.90 主 ROI、Q0.95 敏感性 ROI 与 invalid/missing 均冻结并报告；
- Phase B QA 完整且人工 PASS；
- 全部标准库单元测试通过；
- run manifest 能把结果追溯到代码、配置和原始输入哈希。

如果独立 KEYENCE fixture 或官方 mask 证据不足，正确结果是停在 WP3 并明确列出缺口，而不是绕过门禁继续配准。
