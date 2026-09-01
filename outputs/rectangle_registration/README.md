# 矩形加工区输出状态

- `manual_v1/`、`registration/manual_v1/`、`qa/manual_v1/`：已完成的人工四边
  冻结与旧 260 µm Phase A 审计链；审批状态保持 `BLOCKED`。
- `manual_internal_roi_v1/`：已完成的快速稳定 ROI 路线；200/200 样本均已导出，
  统一为 80×80 µm、0.5 µm/pixel，并已生成直接可读的数据集包。
- `registration/`、`registered/`、`geometry/`、`resampling/` 等其他目录：自动
  配准 v2–v7 和旧路线证据，不删除，但不再作为主中心估计。

不要把不同方法的文件复制到同一个输出标签下，也不要用新运行覆盖旧 manifest。
