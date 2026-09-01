# scripts 路线索引

- `00*–01*`：数据、CAG/CSV 等价性和 inventory；仍是当前输入门禁。
- `02–07`、`04b–04m`：历史自动配准 v2–v7；只用于复现/审计，不再决定中心。
- `15`：人工标注工具；标注已经完成，保留用于复查。
- `16–21`：已完成的 `manual_v1` 冻结、QA 与旧 Phase A 全链复现。
- `22`：全量提取统一 80×80 µm raw/repaired 稳定 ROI。
- `23`：把 200 份 ROI 打包成直接可读的数据集。
- `32`：环境、测试、提取和打包的一键入口。

不要删除或改号历史脚本：manifest 已记录这些文件名。新工作只能从 22 开始，
并把输出写入 `outputs/rectangle_registration/manual_internal_roi_v1/`。
