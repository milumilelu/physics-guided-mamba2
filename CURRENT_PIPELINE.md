# 当前实际主路线

更新时间：2026-09-01  
当前状态：快速预处理已完成，200/200 稳定 ROI 已导出并打包。

## 唯一执行入口

当前已经执行：

```text
200 条冻结人工四边
→ 160 个 measurement 级背景平面
→ 矩形面锥坑候选审计与双轨修复
→ 框内四边 profile / margin
→ 全体共用 Q90 与 Q95 稳定 ROI
→ raw/repaired/mask/QA/manifest
```

权威实施规范：

```text
任务说明书/WORKBUDDY_下一阶段_人工框内稳定ROI_v1.md
```

当前人工输入：

```text
outputs/rectangle_registration/registration/manual_v1/
  manual_four_edge_validation_frozen.csv
```

旧 260 µm `manual_v1` Phase A 保持 `BLOCKED`，只作为人工标注和方法演进的
上游审计证据。新的 stable ROI 路线不依赖旧 `H_reg`，特别是 20 补充 pass
直接由原始 measurement 和人工框进入分析。

## 目录职责

| 目录 | 当前职责 |
|---|---|
| `任务说明书/` | 只放当前待执行规范 |
| `config/` | 冻结输入、session 方向、方法配置 |
| `src/` | 可测试、无硬编码路径的科学计算模块 |
| `scripts/` | 已完成 00–23；`32` 是环境、测试、提取、打包的一键入口 |
| `tests/` | 与主路线同步的自动测试 |
| `outputs/rectangle_registration/manual_v1/` | 已完成但 BLOCKED 的旧 Phase A 审计链 |
| `outputs/rectangle_registration/manual_internal_roi_v1/` | 200 份 80×80 µm 稳定 ROI、QA 和数据集 |

直接供下游读取的数据集：

```text
outputs/rectangle_registration/manual_internal_roi_v1/dataset/
  stable_roi_80um_dataset.npz
  stable_roi_80um_index.csv
```

主数组为 `height_raw[200,160,160]`；`height_repaired` 是带明确
`repair_mask` 的 Level-3 可选派生数据。
| `archive/rectangle_registration_history/` | v2–v7 自动配准与旧执行计划，只读历史 |
| `experiments/mechanism_virtual_augmentation/` | 下游独立实验，不进入当前流程 |
| `专利/` | 专利材料，保持原状 |
| `氧化锆/` | 原始和派生测量数据，不整理、不覆盖 |

## 明确不做

- 不再以 v2–v7 自动中心替代人工中心；
- 不修复旧 Phase A 的 260 µm 公共视场门禁；
- 不删除旧输出或重写旧审批状态；
- 不把单线锥坑算法直接套用到矩形面；
- 不在本阶段做形貌特征、PCA、预测、Mamba 或虚拟数据增强；
- 不修改专利材料。

## 环境

固定 Python `3.12.13`，核心依赖见 `requirements.txt`。执行任何正式阶段前运行：

```powershell
.\.venv\Scripts\python.exe scripts\verify_environment.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

任何新脚本必须使用仓库相对路径或 CLI 参数，禁止新增个人电脑绝对路径。
