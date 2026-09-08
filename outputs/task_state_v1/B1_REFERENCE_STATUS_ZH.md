# B1 独立参考与共同接口进展（2026-09-07）

参考门 PASS，学习效果 NOT_TESTED。没有启动网络训练，没有产生 memory necessity 或 Mamba 优势结论。

## 已运行

- 数据 seed=20260906，train/validation/test 独立 SeedSequence 子流，家族数 512/128/256。每家族八个等物理时长控制块，K∈{32,64,128}、dt∈[0.01,0.08]，初态全零。
- 独立 float64 RK4，控制跳变处分段，dt≤0.001；每 split 前 8 家族分别核验步长 0.0005 与独立 DOP853。coupled/null 两轨共 96 项检查，最大相对误差 5.09645e-9，小于冻结 1e-6 门槛。
- 家族无跨 split 重复。终态文件仅有 control、physical_dt、mask、terminal_x、family_id；真隐藏态与块边界轨迹放 evaluator_only，加载器拒绝额外隐藏数组。文件路径隔离属于程序数据流契约，不是操作系统权限隔离。
- H0、GRU、CT-SSM 使用同一预测 x/u/log(dt) 输入、同一逐分量 2*tanh 有界闭合与解析已解析骨架。零闭合恢复独立 null 参考；相邻独立样本默认清零，显式 continuation 可续算；padding 不推进状态。
- 参数量 H0=7746、GRU=7810、CT-SSM=7778；float32 每样本持续状态分别 8/136/136 bytes（包含 x，未将参数计入 cache）。尚未实现 Mamba-2，不把 CT-SSM 称为 Mamba。
- 55 项回归测试通过。新 finite_scan 测试因编译依赖仍在安装而未运行，本数字明确排除它，不代表整个 E03 或整个 G1 通过。

## 封存证据

- `20260906T164858Z_E04_B1_REFERENCE_18349780_533009c_29e663/gate_G1_B1_REF.json`
- `20260906T165326Z_E04_B1_BACKEND_18349780_533009c_bb14ef/backend_contract_gate.json`

## 待完成

有界闭合激活时的 AD/FD 和 memory-block 收敛、完整历史静态对照、Mamba-2 后端、冻结训练选择规则、三 seed 主训练与 null、continuation/OOD、G2 判定。当前不能声称 E04 完成。
