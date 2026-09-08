# Task-State 首批执行报告

日期：2026-09-06。协议执行修订：v1.2.1。**已执行 E00、完整 E01/G1（canonical observer、解析物理、可微 observer 后端、AD/FD 梯度门与扫描收敛）、E02 MAIN180 静态诊断；尚未完成 E03 校准、B1 或 Mamba 主实验。**

## 产物索引

| 项目 | 实际状态 | 产物 |
|---|---|---|
| E00 | G0_DATA scoped PASS；MAIN180 126 分量，每 outer fold 36 ROI，inner 分组已冻结 | [20260906T111435Z_E00_a1a1c9a8_533009c_9e0686](20260906T111435Z_E00_a1a1c9a8_533009c_9e0686/gate_G0.json) |
| E01 canonical | 2000 个目标值全部匹配；最大误差 7.1054e-15 | [parity](20260906T111445Z_E01_OBS_e17ab365_533009c_9fd9ef/observer_parity.csv) |
| E01 物理解析 | 16 例通过；单脉冲 finest grid 相对误差 7.0267e-6（网格加密按描述符 ≤1% 门槛判定，解析标量例 ≤1.15e-15） | [检查](20260906T112222Z_E01_PHYS_UNIT_d0670347_533009c_8ccbfd/gate_physics_unit.json) |
| E01 可微后端 | 4050 个比较全过；float64 最大误差 4.44e-15、float32 输入 4.44e-15；validity/reason 先行一致 | [gate](20260906T132630Z_E01_BACKEND_OBS_337bc8f7_533009c_440a38/gate_G1_BACKEND_OBS.json) |
| E01 梯度门 | 32 预注册点 × 4 方向 × 5 步长 = 128 比较，通过率 100%，最大相对误差 5.08e-4，0 个未解释；网格加密 ≤0.25%、脉冲包加密 ≤0.24% | [gate](20260906T132733Z_E01_GRAD_5fd0d231_533009c_f3cce6/gate_G1_GRAD.json) |
| **G1 收口** | **四子 Gate 全 PASS，gate_G1.json 正式 PASS；gradient_based_inverse_design = ALLOWED** | [gate_G1](20260906T132805Z_E01_G1_CLOSE_e17ab365_533009c_c06c1d/gate_G1.json) |
| E02 MAIN180 | 18 个模型/输入组合，90 次 outer refit，22680 条 OOF | [OOF](20260906T111859Z_E02_b901bea5_533009c_68a562/diagnostic_oof.csv) |
| 验证 | 37 个测试通过（23 物理/分组/observer + 6 可微 observer + 8 扫描 fixture）；90 个模型重载预测一致 | [验证记录](execution_verification.json) |

231 个审计输入 hash 复核未变；4 份上游 sealed release 通过验证。80 个 depth crossfit 子分区逐一核查无 component 交叠；保存模型重放与 OOF 的最大偏差 8.8818e-16。

SUPP20 的独立 family-purged 轨道需从 MAIN180 移除 40 行，余 140 行训练；已生成跨 cohort 对应表，尚未训练该轨道。

## E02 当前可以说什么

以下是 A/P/T 三块归一化的 component-balanced risk，越低越好；它不是 E06 的 D/A/P/T 四块主风险，也不是物理误差百分比。

| 输入 | Ridge risk | GAM risk |
|---|---:|---:|
| 完整工艺 U（加工前） | 0.791367 | 0.849964 |
| U + 交叉拟合 Dhat（加工前） | 0.791851 | 0.818868 |
| 仅总剂量 Qa（加工前） | 0.986868 | 0.980240 |
| 实测 D（加工后诊断） | 0.932201 | 0.935409 |
| 实测 D + hatch（加工后诊断） | 0.812207 | 0.756685 |

完整 U 相比仅 Qa 的风险差：Ridge −0.1955（描述性 95% 区间 [−0.2758,−0.1159]）；GAM −0.1303（[−0.2191,−0.0477]）。U+Dhat 相比 U 的区间均跨 0，当前不支持稳定额外增益。D+h 优于仅 D 的现象也不能直接用于加工前设计，因为 D 是实测终态。

这些区间条件于固定分割/已拟合模型，不是本研究 C1/C2/C3 主检验；还未完成 support-matched 检查和全 E02 科学判读。不得由这些结果推出 memory necessity、材料机制或 Mamba 优势。模型/特征/阈值没有在查看 outer 结果后改动。

完整表：[风险](20260906T111859Z_E02_b901bea5_533009c_68a562/risk_summary.csv)、[逐目标误差](20260906T111859Z_E02_b901bea5_533009c_68a562/native_metrics.csv)、[ILR联合指标](E02_P_native_summary.csv)、[配对区间](20260906T111859Z_E02_b901bea5_533009c_68a562/paired_diagnostic_intervals.csv)、[fold/session](20260906T111859Z_E02_b901bea5_533009c_68a562/fold_session_risk.csv)。

## 必须保留的失败与协议修订

首次 E01 使用原手册的 18 箱时，400 个 entropy 比较全部未过，其他 1600 值通过。检查冻结生产配置后确认主 target 为 36 箱；r1 的 18 箱属于 raw 重算支路。已修订执行 YAML/runbook §29 为 36 箱，保留 18 箱失败 run `20260906T111320Z_E01_OBS_9aa00965_533009c_8477ff`，未改历史数据和验收门槛。

原 E00 run `20260906T111139Z_E00_a1a1c9a8_533009c_501f84` 保留；36 箱执行修订后重新冻结的 E00 为本报告入口。两次正式 split SHA 完全相同：`cd90d1a1147c256df3f0bfdffb1b5e5fca5976323f9bae2aabfa8eb7e3bf7d2a`。

G1 期间保留的 FAIL run（详见 [决策日志](DECISION_LOG_G1_ZH.md)）：梯度门平台规则 v1（`20260906T131109Z_E01_GRAD_c5bacea6_533009c_79dd6a`，4/128 落在带截断误差的粗步长上，规则 v2 只依赖 FD 表内部一致性）与脉冲包语义错误版（`20260906T131541Z_E01_GRAD_c5bacea6_533009c_373887`，把 1 脉冲拆成 E/4 子脉冲改变了物理实验，孵化效应下深度单调增长是预期行为；修正为固定物理脉冲串的冻结子步状态近似，并将情景改为每格点 8 脉冲以避免平凡通过）。协议数值门槛一律未放宽。

## 尚未完成

- E03 物理参数校准（P00/P10/P01/P11 独立侧校准 + 可识别性审计）与 E05/E08。
- E04/B1（reference、loader、continuation、common memory API、memory-block refinement）与 Mamba 接入。
- E06 主实验、E07 状态预算、E09 inverse design、E10 robustness、E11 计算成本。
- E02 的 HIST200 重跑与 support-matched 科学诊断。

硬件预检：RTX 5060 Ti、8151 MiB；WSL 未安装。CPU 版 PyTorch 2.14.0 已装入历史 `.venv`（纯增量，历史 run 的 environment_lock 不受影响；G1 梯度门为 CPU float64，不依赖 GPU）。CUDA/WSL 环境决策推迟到 E04 前硬件预检。没有任务在后台继续训练。

执行命令与接续步骤见仓库 `experiments/task_state_v1/README.md`。此报告是第一批实测结果，不是完整研究任务完成证明。
