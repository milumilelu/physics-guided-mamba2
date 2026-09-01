# Physics-Guided Mamba-2：超快激光烧蚀形貌预测

> **当前实际主路线（2026-09-01）**：200 个矩形槽的人工四边已经冻结，快速
> 预处理已完成“measurement 级背景调平 → 固定中心稳定区 → 矩形面锥坑保守
> 修复 → raw/repaired 数据集打包”。唯一执行说明见
> `任务说明书/WORKBUDDY_下一阶段_人工框内稳定ROI_v1.md`，仓库入口与历史边界
> 见 `CURRENT_PIPELINE.md`。
>
> 本 README 其余部分记录早期单线形貌与专利相关实验，保留作审计，不代表当前
> 矩形稳定 ROI 的执行入口。`专利/` 及专利材料保持原状。

面向氧化锆（ZrO₂）超快激光多脉冲烧蚀的**部分可观测条件下跨尺度物理引导隐状态动力学建模**课题。

核心科学问题：在只能观测到「单线/槽压缩观测量」的条件下，用隐状态动力学模型刻画**历史材料状态 $Z_n$** 的选择性演化，从而由脉冲历史预测烧蚀形貌。

$$
\text{真实脉冲历史} \rightarrow (H_n, Z_n) \rightarrow \text{材料去除状态门控} \rightarrow H_{n+1} \rightarrow \text{单线/槽压缩观测}
$$

---

## 1. 仓库结构

```
physics-guided Mamba-2/
├── export_height_csv.py               # ★ CAG → 高度矩阵 CSV 批量导出（与官方导出逐字节一致）
├── extract_zro2_single_line.py        # ★ 主流水线：CAG 解码 → 几何提取 → 特征表
├── experiments/mechanism_virtual_augmentation/     # 独立、未完成的虚拟增强实验
├── compare_raw_vs_repaired.py         # 圆锥伪影修复对观测算子 Y=[W_line, D_line] 的影响
├── inventory_cone_repair.py           # 15 个 pilot 组的圆锥修复盘点与质检
├── preview_cag_group.py               # 单组 CAG 预览（高度图 + 斜视点云，自包含）
├── render_cag_group1_raw.py           # 原始未修正高度场渲染，用于对照
├── validate_conical_preprocess_group1.py  # 第 1 组修复前/后诊断
├── probe_cag.py                       # CAG/ZIP 结构探测小工具
│
├── 氧化锆/                             # 原始测量数据（3.2 GB，已被 .gitignore 排除）
│   ├── 60Pass组.cag       2.3 GB
│   ├── 20组.cag           756 MB
│   ├── 120组直线.cag      147 MB      # pilot 实验使用
│   ├── 72组单脉冲直线.cag  86 MB
│   └── *_design.csv                   # 实验设计表（DOE）
│
├── outputs/                            # 实验产物（PNG 已忽略，只跟踪 CSV/JSON/MD）
│   ├── zro2_single_line_pilot/                    # ★ 主结果：原始口径
│   ├── zro2_single_line_pilot_cone_repaired/      # 圆锥修复后
│   ├── zro2_single_line_pilot_cut_only_stable_region/  # 切廊稳定区
│   ├── group001_cut_only_stable_region/
│   ├── cone_repair_impact/             # 修复 vs 未修复的量化对比
│   ├── cone_repair_inventory/          # 290 个圆锥伪影的逐个体检表
│   ├── cag_preview/ 与 cag_raw_verification/      # 解码正确性验证
│
├── archive/rectangle_registration_history/  # 自动配准 v2-v7 与旧计划，只读历史
├── 任务说明书/WORKBUDDY_下一阶段_人工框内稳定ROI_v1.md  # 当前执行规范
├── pasted.txt                          # 论文主线收敛笔记（部分可观测 + 隐状态动力学）
└── 超快激光多脉冲烧蚀形貌预测_专利技术交底书_V3_微观宏观耦合版 (1).docx
```

## 2. 技术链路

KEYENCE 显微镜的 `.cag` 本质是一个 ZIP 容器，内部按 `Path/<组号>/...` 组织，高度数据以 VK4 二进制段存储。

```
.cag (ZIP)
  └─ 解码 VK4 头部偏移表 → 定位 height 段 → uint32 高度阵列 → reshape (1024×64)
       ↓
  非对称稳健参考平面拟合（plane_negative_clip_sigma=2.5 / positive=4.0，迭代 12 次）
       ↓
  圆锥缺失伪影修复 repair_conical_dropouts（共聚焦在高陡壁处的典型失效）
       ↓
  阈值分割（threshold_k=4.0）→ 连通域 → 连续线 / 离散坑判定
       ↓
  中心 70% 区域取剖面 → 输出 Y_line = [W_line, D_line] 及 40+ 个 QC 字段
```

关键中间量：`fluence`、`threshold`、`margin = log(fluence/threshold)`、`inc`、`z`、`total_defocus`、pass 级去除增量、累计深度、`core5` 物理代理特征。

## 3. CAG 高度矩阵批量导出

`export_height_csv.py` 把 `.cag` 中每一组测量导出为 KEYENCE `ImageDataCsv` 格式的 CSV，
**输出与官方软件导出逐字节一致**（已在 `20补充pass` 的 10 个手工导出文件上验证：10/10 完全一致，
含"测量数据名"字段）。

```bash
.venv/Scripts/python.exe export_height_csv.py \
    --cag       "氧化锆/pass实验数据/120正式.cag" \
    --output-dir "氧化锆/pass实验数据/csv文件/120正式"

# 整棵目录树，按源结构镜像输出
.venv/Scripts/python.exe export_height_csv.py \
    --batch-root  "E:/博士课题资料/光机所实验原始数据" \
    --output-root "E:/博士课题资料/csv高度" --skip-existing --dry-run
```

### 文件名来自容器本身，不要手工推导

`--naming data`（默认）读取 CAG 内 `MeasurementDataMap → FileItemAccessor` 槽位中存储的字符串，
也就是 KEYENCE 软件数据列表里显示、并写进 CSV `"测量数据名"` 字段的那个名字。

```
<root>/<组号>/<uuid>/<FileItemAccessor uuid>/e57e75b1-707b-4a6f-a095-1485b8b95efb
```

该槽位是**无长度前缀的 UTF-8/GBK 文本**，因此按 UTF-16LE 结构扫描容器是找不到的。

⚠️ **不要按"组号 → 加工顺序"推导命名。** 蛇形扫描会让编号回折，例如 `60Pass组.cag` 中：

| 组号 | 1 | 6 | 7 | 8 | … | 12 | 13 |
|---|---|---|---|---|---|---|---|
| 真实数据名 | `1 2` | `11 12` | **`14 13`** | **`16 15`** | … | **`24 23`** | `25 26` |

若按 `(2n-1, 2n)` 推导，第 7 组会错成 `13 14`。一切以容器内记录为准。

其它命名模式：`--naming original`（采集文件名 `MeasureData20260528105621`）、
`--naming index`（`001`）。

### 解码要点

* 高度段起点是 `height_offset + 20 + 776`；`776` 是伪彩色 LUT，漏掉会造成**列循环移位 194**。
* 换算必须**四舍五入远离零**：`(raw * z_step_pm + 500) // 1000`
  （`np.round` 是银行家舍入，会产生 0.001 μm 偏差）。
* 无效像素哨兵 `raw >= 0xFF000000`；VK4 blob 大小阈值取 100 KB（旧直线扫描只有 568 KB）。

## 4. 运行方式

主流水线（默认跑 15 组 pilot：13/19/33/34/43/44/48/51/60/68/94/95/101/104/116）：

```bash
python extract_zro2_single_line.py \
    --cag       "氧化锆/120组直线.cag" \
    --design    "氧化锆/氧化锆_line_design.csv" \
    --output-dir outputs/zro2_single_line_pilot \
    --groups pilot --pilot-count 15
```

固定环境：CPython 3.12.13，完整版本见 `requirements.txt`。仓库内环境初始化：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_environment.ps1 `
    -Python "C:\path\to\Python312\python.exe"
```

国内网络可显式指定镜像；包版本仍由 `requirements.txt` 固定：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_environment.ps1 `
    -Python "C:\path\to\Python312\python.exe" `
    -IndexUrl "https://pypi.tuna.tsinghua.edu.cn/simple"
```

需要脚本中的非默认 XGBoost/CatBoost 模型时增加 `-WithExtraModels`，版本见
`requirements-extra-models.txt`。需要训练可学习物理参数或后续 Mamba 模型时增加
`-WithTorch`。该扩展固定为
`torch==2.11.0+cu128`，详见 `requirements-torch-cu128.txt`。安装后使用：

```powershell
.\.venv\Scripts\python.exe scripts/verify_environment.py
```

> **注意**：`render_cag_group1_raw.py`、`validate_conical_preprocess_group1.py`、`probe_cag.py` 中硬编码了历史绝对路径
> （如 `C:\Users\RZF\Desktop\专利\...`、`F:\20260528-xgd\...`），在当前目录下无法直接运行。
> 建议后续改为 `argparse` 参数或读取统一的 `config.toml`。

## 5. 版本控制约定

| 内容 | 策略 | 原因 |
|---|---|---|
| `.cag` / `.vk4` 原始数据 | **不跟踪**（`.gitignore`） | 3.2 GB 不可变采集数据，留在本地/移动硬盘 |
| `outputs/**/*.png` | **不跟踪** | 54 张、6.7 MB，脚本可确定性重建 |
| `outputs/**/*.csv`、`.json`、`.md` | **跟踪** | 可复核的关键数值结果与质检记录 |
| `*.docx` 专利交底书 | 跟踪 | 版本需留痕 |
| `.workbuddy-ai/` | 不跟踪 | 工具目录，仅本地使用 |

若确需保留某张诊断图，用 `git add -f <路径>` 强制加入。

## 6. 当前进展

- [x] CAG 容器解码与 VK4 高度标定还原（已通过 `cag_raw_verification` 交叉验证）
- [x] 圆锥缺失伪影检测与修复（15 组共修复 290 个锥、48,256 像素，残余强缺陷 0 处）
- [x] 单线几何提取 pilot，输出 `W_line / D_line` 及完整 QC 字段
- [x] 量化修复对观测算子的偏差影响（`outputs/cone_repair_impact`）
- [ ] 机理事件（E1/E2/E5）虚拟数据增强的严格对照实验（独立存放于 `experiments/mechanism_virtual_augmentation/`，不属于当前 ROI 路线）
- [ ] Physics-guided Mamba-2 主体模型
