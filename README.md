# 氧化锆矩形加工区数据预处理

这个仓库只保留当前数据预处理流程及三项可扩展能力：人工标注、锥坑修复和
虚拟数据增强。

## 当前流程

```text
原始 CAG
→ 人工矩形框与固定 session 方向
→ measurement 背景平面校正
→ 中心 80×80 µm 稳定 ROI（0.5 µm/pixel）
→ 保守二维锥坑修复 + repair mask
→ 200×160×160 数据集
```

运行：

```powershell
.\.venv\Scripts\python.exe scripts\32_run_manual_internal_roi_v1.py
```

结果：

```text
outputs/rectangle_registration/manual_internal_roi_v1/dataset/
  stable_roi_80um_dataset.npz
  stable_roi_80um_index.csv
```

数据包数组：

- `height_raw[200,160,160]`：主数据；
- `height_repaired[200,160,160]`：锥坑修复版本；
- `valid_mask[200,160,160]`；
- `repair_mask[200,160,160]`；
- `session_id`、`measurement_id`、`sample_id`、`x_um`、`y_um`。

## 保留的扩展功能

- `annotations/` 与 `scripts/15_manual_four_edge_annotator.py`：继续检查或扩展人工框选；
- `src/conical_dropout.py`：可调整并复用的二维锥坑修复；
- `experiments/mechanism_virtual_augmentation/`：独立的机理事件虚拟数据增强实验；
- `专利/`：专利材料，未参与本次清理。

## 目录

```text
annotations/   人工标注及标注器所需固定几何
config/        当前数据映射、平面与 ROI 参数
scripts/       5 个执行/环境脚本
src/           当前流程所需模块
tests/         当前流程测试
experiments/   虚拟数据增强
outputs/       当前 80 µm 预处理结果
氧化锆/        原始数据
专利/          专利材料
```

Python 固定为 3.12.13，依赖见 `requirements.txt`。

