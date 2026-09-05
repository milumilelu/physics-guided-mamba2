# 脚本

- `15_manual_four_edge_annotator.py`：人工框选矩形加工区。
- `22_extract_stable_roi_fast.py`：从原始 CAG 提取 200 份 80×80 µm ROI，并执行锥坑修复。
- `23_build_stable_roi_dataset.py`：打包为一个可直接读取的 NPZ 数据集。
- `32_run_manual_internal_roi_v1.py`：环境检查、测试、提取、打包一键执行。
- `33_build_single_line_annotation_table.py`：为 120 条单线生成视图清单（平面拟合 + 线方向角）与空标注表。
- `34_manual_single_line_annotator.py`：单线加工范围盲标（狭长矩形交互标注器）。
- `setup_environment.ps1` / `verify_environment.py`：固定与检查 Python 环境。
