# mamba_depth_quality_v1 pilot 结果

- OOF 覆盖：1440 条记录，模型数 8，外层折 5。
- 本次 Mamba 模型使用 reference_pure_torch_ssd_mamba2_verified_adapter_v1；full profile 使用 300 epoch 上限并执行 inner early stopping（外层重拟合至少 30 epoch）。
- seed=17，5 折；结果为开发性 pilot，不是独立确认。

## 指标

|模型|目标|MAE (µm)|RMSE (µm)|Q²|
|---|---|---:|---:|---:|
|CONST_DQ|D|18.2171|20.2707|0.0000|
|CONST_DQ|Sq|1.1709|1.7959|0.0000|
|TREE_U_DQ|D|4.9912|6.5126|0.8968|
|TREE_U_DQ|Sq|1.0370|1.7867|0.0102|
|STATIC_PHY_K4_AUX|D|14.1688|20.3740|-0.0102|
|STATIC_PHY_K4_AUX|Sq|1.1841|1.8688|-0.0829|
|GRU_PHY_K4_AUX|D|14.2670|20.6171|-0.0345|
|GRU_PHY_K4_AUX|Sq|1.1071|1.8036|-0.0086|
|MAMBA_RAW_K4_AUX|D|15.3793|21.9300|-0.1704|
|MAMBA_RAW_K4_AUX|Sq|1.0990|1.7247|0.0777|
|MAMBA_PHY_K4_NOAUX|D|14.9949|20.2642|0.0006|
|MAMBA_PHY_K4_NOAUX|Sq|1.1473|2.1164|-0.3888|
|MAMBA_PHY_K4_AUX|D|14.9895|20.2830|-0.0012|
|MAMBA_PHY_K4_AUX|Sq|1.1511|2.1559|-0.4411|
|MAMBA_PHY_K32_AUX|D|10.2933|12.9740|0.5904|
|MAMBA_PHY_K32_AUX|Sq|1.2921|2.1810|-0.4749|


## 协议完成状态

本 run 已执行完整 pilot：5 折、seed=17、8 个模型、每模型 180 条 OOF（共 1440 条）；inner reference cross-fit=3，full profile 上限 300 epoch，神经模型外层重拟合至少 30 epoch；G3 fresh import/reload 与 full/stepwise 数值检查通过，最大差异 1.86e-8。

按分量平衡 MAE，`TREE_U_DQ` 为 D 5.1224 µm、Sq 1.1451 µm；主模型 `MAMBA_PHY_K4_AUX` 为 D 16.2759 µm、Sq 1.2820 µm；`MAMBA_PHY_K32_AUX` 为 D 10.6122 µm、Sq 1.4137 µm。当前单 seed 下，树模型仍明显优于 Mamba；k=32 改善深度但损害 Sq，不能宣称压缩无损。

经验参考相对 raw Mamba 使 D 分量平衡 MAE 从 16.8170 降至 16.2759 µm，属于本次固定训练设置下的有限改善；AUX 相对 NOAUX 的 Sq 为 1.2820 vs 1.2754 µm，未显示稳定帮助。未观察到主模型负深度预测；GRU 和部分 Mamba 变体有少量负预测，已在指标表记录。

结论状态：`IMPLEMENTATION_PASS`；预测结论为 `NO_STABLE_GAIN`（Mamba 未胜过树基线），不是物理因果或真实时间记忆证明。下一步只建议在不改变模型/输入合同的前提下做固定多 seed 复核。

