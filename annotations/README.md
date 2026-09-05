# 人工标注

- `manual_four_edge_validation.csv`：可继续编辑的 200 条人工框；
- `manual_four_edge_validation_frozen.csv`：当前数据集使用的冻结副本；
- `session_geometry.csv`：三个 session 的固定旋转角；
- `sample_view_manifest.csv`：双槽 measurement 的左右视图映射；
- `single_line_view_manifest.csv`：120 条单线每组冻结的平面/方向角/裁剪视图清单（脚本 33 生成）；
- `single_line_range_annotation.csv`：可继续编辑的单线加工范围人工标注表（狭长矩形，脚本 34）。

双击仓库根目录的 `启动四边盲标_A.bat` 可打开标注器。保存仍写入可编辑 CSV，
不会自动覆盖冻结副本。

标注器只有在存在未保存框时才临时生成 `manual_annotation_draft_a.json`；成功保存
后会自动清除。

单线标注：双击仓库根目录的 `启动单线盲标_A.bat`，操作见 `单线盲标操作说明.md`；
未保存框的草稿文件为 `single_line_annotation_draft_a.json`。

