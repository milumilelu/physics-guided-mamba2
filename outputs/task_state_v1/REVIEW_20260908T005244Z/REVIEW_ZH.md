# 当前仓库独立审查（2026-09-08，北京时间约 08:52–08:58）

结论：存在会影响数值正确性、统计有效性和运行能否完成的实质问题。当前 E03 不应作为可直接放行的正式校准；E06_STATIC 和 E10_SUPP20 的 PASS 不能解释为相应科学阶段完整通过。本次只读审查计算实现与运行产物，仅在本审查目录新增报告、复现证据和源码快照；没有停止进程、修改训练代码或覆盖历史 run。

## 一、实际任务状态

| 分支 | 当前证据 | 可以确认的状态 |
|---|---|---|
| E03 最新 run | `20260907T115004Z_E03_CALIBRATION_e1f7205c_533009c_c3d084`，只有环境/源码快照，无 CSV、gate 或 release manifest | 未封账；完成作业数未知 |
| E03 运行进程 | 两次本机 Win32_Process 盘点均无校准主进程、worker 或 WSL 进程 | 不能确认“60 作业池仍在运行”；本机未发现活动 E03 |
| B1 训练 | PID 43664→47136，命令 `04_b1_terminal.py --tracks coupled null`，最新目录 `...154333Z...a69ab7` 无 gate | 训练进程存在；不能据此断言完成度 |
| E04 后端 | `...021335Z...67c988/gate_E04_BACKEND.json` PASS | 指定后端测试通过，不等于 B1 学习实验完成 |
| E06 hybrid 验证 | `...060512Z...cf3329/gate_E06_VERIFY.json` FAIL | D 最大相对差 5.659%，A 为 676.090%；阈值漂移 0.69218、Δd/zR 1.53017 |
| E06 static | `...084747Z...c0a874` PASS，11340 行 OOF | 有静态预测产物，但缺少 D、内层选择与有效哈希等完整合同 |
| E10 SUPP20 | `...095134Z...68315e` PASS | 只有 shift tracking，没有 SUPP20 预测或评分 |
| E07/E09/E11 | 有部分函数/脚本骨架 | 未见本次审查可支撑完整状态预算、真实设计评价和最终 release 的产物链 |

根目录 `execution_status.json` 仍停在上一轮 B1 参考实现阶段，不能当作当前运行状态。最新 E03 runner、cell solver、配置，以及活动 B1 runner/training.py 与各自 run 源码快照逐字节一致，因此以下问题适用于这些运行版本，而不只是后来编辑的工作区。

## 二、阻断级问题

### P0-1：E03 绕过尚未通过的数值和 ROI 注册要求

位置：`experiments/task_state_v1/03_physics_calibration.py:53`。

`simulate_row()` 直接调用 `simulate_sample()`，虽然导入了 `simulate_sample_checked`，却完全没有使用。使用 `CellConfig()` 默认 3w 截断，输出仍明确标为 `infinite_raster_interior_cell`、`finite_region_simulated=False`、注册待完成。没有读取或要求相关 G0/G1/求解器 gate 通过，没有执行配置中的 checked_fraction=0.1。已有 0.874 μm/P11 强反馈连续近似失败证据尚未被该 runner 消除；有限 80 μm 观测也没有接入。

影响：优化器可能吸收求解器/观测近似偏差，即使拟合误差低也不能作为注册物理参数基线。不能通过等待 60 个任务结束来补足这个问题。

建议：先确定同一有限 ROI 观测下的已验证求解器及参数域/运行时误差核验，再把门作为校准入口的硬约束。

### P0-2：参数边界没有限制优化器，越界不一定被发现

位置：同文件 121、141、162–171 行。

两处 Nelder–Mead 均未传 bounds，也没有目标函数边界拒绝或变换。BOUNDS 只用于初值/事后靠边检测；靠边检测用绝对距离，不覆盖远越界点。已实测：配置 gamma_F/gamma_delta 上界为 1，传 2 仍被 physics_params 接受。配置 n_starts=2，但 starts 实际返回 3 个初值；参数 bounds 也使用硬编码全局常量而非解析配置。

影响：可能得到未注册的越界有效参数并报告零 boundary hits，且冻结配置不反映实际执行预算。

建议：优化全程 enforce bounds；同时报告 lower/upper hit、out-of-bounds、优化状态和全部初值成本。

### P0-3：B1 在长时间训练后会碰到已复现的评价错误，且缺少可复用 checkpoint

位置：`experiments/task_state_v1/04_b1_terminal.py:68–81`。

1. `** 2 .sum()` 实际访问整数 2 的 sum；使用两条合成候选调用 sealed_design 已复现 `AttributeError: 'int' object has no attribute 'sum'`。
2. 候选写作 A_i、B_j，chosen_id 却总写 A_chosen。选中 B_j 时得到 A_j；独立复现 pairwise_regret 报“Chosen id not among candidates”。
3. 训练脚本没有持久化模型权重、epoch 日志或每作业 checkpoint。inner 表也在整个 tuning 循环完成后才写；最终指标更晚落盘。设计评价又重新拟合第一 seed，而非复用已完成的 refit。

影响：当前 B1 即使完成大量训练也会在后段失败，并且无法从已有模型断点恢复。建议让执行该训练的 agent 优先处理这一阻断点；本次未代替用户停止进程。

### P0-4：B1 continuation evaluator 的真值和模型恢复均不可靠

位置：`experiments/task_state_v1/05_g2_gate.py:61,83–85,98–104`。

- 读取 hidden['hidden']，但封存 NPZ 的实际键是 full_terminal_state 和 block_boundary_states 等，没有 hidden。
- 找不到 `state_{model}_{track}_{seed}.pt` 时静默继续使用新初始化模型；当前训练器也没有写这些文件。
- 将零控制下未来真值写成 x1*exp(-0.4T)、x2*exp(-0.7T)，忽略 coupled 系统仍在衰减的 q 对 x 的作用。即使修好文件键，这也不是独立 B1 coupled 参考真值。
- DP 区间按 pair_id 重采样，而协议统计单元是 family；共享候选的 pair 不是独立样本。

建议：模型/数据缺失立即 fail closed；使用完整真状态调用独立 reference 做 continuation；按家族依赖结构重采样。禁止由此生成 memory necessity 或 decision preservation 结论。

## 三、E03 实验设计和可审计性问题

### P1-1：所谓 full-train refinement 实际只有 16–24 行，且没有 inner 校准

位置：E03 runner 140、229–252 行。

所有 fold 复用从 fold 0 outer-train 选出的 subset/refine_ids，再与当前 outer-train 取交集。按冻结 split 实算：

| outer fold | search 实际行数 | refine 实际行数 |
|---|---:|---:|
| 0 | 8 | 24 |
| 1 | 4 | 16 |
| 2 | 8 | 20 |
| 3 | 4 | 16 |
| 4 | 8 | 20 |

每折 outer-train 实际有 144 行。代码注释说 64→full-train，配置为 8→24，summary 却命名 objective_full_train，三者不一致。outer-train 取交集避免了这里直接把 outer-test 标签拿去拟合，但造成预算/代表性差异，不能称 full-train 校准。

inner CSV 仅检查索引属于 outer-train，完全没有 inner-fold 参数拟合。60=5×4×3 只覆盖 outer 情景；按三 inner 子集加 outer 重校准需要至少 240 个校准单元（不含多初值与 profiles）。这对未来 hybrid 调参的信息隔离是必要条件。

### P1-2：refine 前后在不同目标函数之间直接比较分数

位置：E03 runner 140–144、175 行。

search 与 refine 不仅样本不同，Objective 还分别用各自数据的 D/A 标准差归一化。直接比较 result.fun 与 best.fun 决定最终 vector、并取 min 作为 full_train objective，没有可比性。

此外目标按 ROI 均匀平均，未做 component-balanced 权重。参数可辨识性代码只是固定其他参数的 coordinate slice，并非 nuisance 参数重拟合的 profile；它还重新校准另一个起点，未绑定最终保存参数。flat_profile_rtol 实际用绝对 spread≤0.02。

建议：在同一完整训练侧目标上重评候选；权重/尺度固定于当前训练子集；真正 profile 必须固定单参数后重拟合其他参数，否则只能改名为条件切片、不能用其判可辨识。

### P1-3：失败、随机种子和作业进度的记录不足

- seed=hash((fold,switch,waist)) 依赖 Python 进程 hash 随机化，未冻结 PYTHONHASHSEED 或保存实际 seed，Windows spawn 下尤其不能保证相同运行复现。
- best_info 只记录胜出初值成本，refine 成本/失败不进入完整统计。
- 求解失败变成常数 1e6 或预测 NaN；末尾只要没抛到顶层就设 PASS/VALID，未强制 finite coverage、优化成功、界内、各 fold/switch/waist 完整。
- sim_info 被丢弃，缺少 per-recipe 数值失败原因/漂移记录。
- pool.map 的结果全部汇总完才写文件；无任务 ID、开始/结束状态、PID/heartbeat、checkpoint/resume。当前无 gate 不能推导“仍在跑”，也无法知道完成了多少作业。

建议：每任务完成即原子落盘、保留失败与全部配置/身份 hash，再由单独聚合器审计完整性并决定 gate。

## 四、后续阶段的假通过和实现缺口

### P1-4：E06_STATIC 的 PASS 缺少整个 D 任务块

位置：`06_real_main.py:27,64,85,125–154`。

复用 E02 diagnostics.Y（仅 A/P/T，7 个目标），封存 oof.csv 11340 行实际没有 D。model_sha/data_sha/split_sha 全空。Ridge/spline 固定 alpha=1；MLP 没有 inner validation，拿训练损失选模型，且 `epoch-(best is not None)` 把布尔值当最佳 epoch。MLP loss 不按 component-balanced 四块风险计算。

MLP_UPHYS 在未提供校准 run 时静默使用硬编码物理参数。未来提供 run 后，physics_descriptors 没有按当前 outer fold 筛选，而是取第一个 P11/waist 向量；若修复 CSV 向量解析后照此复用，会把别折训练信息带入当前 fold。当前不是已证实发生该泄漏，而是应在接入校准前修复的代码路径。

### P1-5：E06 hybrid 不仅数值门 FAIL，还有接口与训练目标错误

- 当前直接导入 laser_hybrid 已复现 ImportError：laser_rollout 没有 apply_packet。
- 现有最近验证 FAIL 的 D/A 误差及漂移远超过门槛；dense 仍整道冻结反馈，不能借独立 finite_scan 的正确性证明该近似正确。
- hybrid_batch 使用硬编码 F1/delta/kappa/rho，未加载当前训练子集的 E03 参数。
- `_DP` 仅剥掉名字，全部调用同一 hybrid_loss，未加入 decision-gap loss。
- closure 正则被 float(bs.pow(2).mean()) 脱离梯度。
- hybrid 使用 8 维目标但继承 7 维 scale/OOF schema，有维度合同冲突。
- LaserHybrid 给 memory.step 传全零 dt，会使显式物理时间 CTSSM 对照失去应有的状态推进。
- packet 分组取决于总 pass/总道数，并非冻结 4 μm 沿轨块；不得据此宣称协议规定的 memory-block 收敛已通过。

建议：先通过同一有限扫描参考、异构批次与闭环梯度反例测试，再训练。当前 E06 主 runner 没有把这些失败门作为硬阻断。

### P1-6：B1 当前优化口径与配置及主张不一致

training.py 实际 BATCH=16、ACCUM=1，配置仍 4×4；无 dropout/相同平均损失时这种变更可等价，但当前 terminal loss 用 sum 而不是 mean，且没有训练尺度归一化，不能直接认定等价。DP 是每 epoch 额外一次 AdamW 步而非共同 objective；该步未裁剪梯度。closure 正则依赖 hasattr(model,'closure_sq')，而当前 B1ClosureModel 没有此方法，实际不生效。tuning best_epoch 是零基索引，refit 使用它作为 epoch 数，少一轮。应明确冻结实际优化方案并用公平对照验证，而非仅沿用旧 YAML。

### P1-7：SUPP20 与最终 claim ledger 只到骨架，不能按 PASS 宣称完成

`10_robustness_release.py:59–85` 的 run_supp20 只统计已有 OOF run、模型名和 family overlap；没有 MAIN180 全训练重拟合、SUPP20 预测、误差或 family-purged 轨道，末尾仍设 VALID/PASS。

null 生成使用空间坐标半径给未 fftshift 的频谱分壳，未做 Hermitian 对称且丢弃 IFFT 虚部，不能证明与参考数据径向功率匹配的角向各向同性 null。

claim_ledger 把 B1 memory claim 预写为 S 等级，并不依实际 G2 verdict 选择支持/不支持；release 不能以“文件能生成”取代证据逐项检查。

## 五、建议的执行顺序

1. 先由执行 agent 核实并纠正运行状态：当前没有本机 E03 worker；B1 存在已复现的后段必失败路径。不盲目再启动新的同配置大运行。
2. 先修校准入口门、有限 ROI/参考求解器、优化边界、完整训练/inner 子集和可比较目标；然后做一个最小校准单元的端到端产物验收。小运行用于验收工程链，不能替代正式矩阵。
3. 修 B1 sealed choice/continuation 与 checkpoint，先跑合成候选和极短训练的完整 save→reload→evaluate 反例，再继续全部注册训练预算。
4. 修 E06 四块目标、nested selection、真实校准参数接入、DP/正则梯度和 hash；SUPP20 实际运行后再判门。
5. 所有旧 run 保留。已存在的 PASS 若 scope 不足，用新的独立审查 ledger 标明“仅部分实现/不足以证明阶段完成”；不要覆盖原 gate 伪造历史。

## 六、本次验证范围与证据

- reviewed_sources.json 与 reviewed_source/：保存本次审查源码 hash 和快照。
- process_snapshot.json：本机实际 Python/WSL 进程快照。
- reproduction_results.json：边界未约束、初值数不符、B1 设计报错、候选 ID 报错、hybrid 导入失败、OOF 缺 D/空 hash、evaluator 键不匹配的最小复现。
- seal_checks.json：抽查的后端、hybrid FAIL、static PASS、SUPP20 PASS 四个 run 的 seal 全部完整。这仅说明产物未改写，不证明科学内容正确。

本次没有运行全套训练或 GPU 测试；结论来自源码、封存快照、实际产物、进程盘点及不涉及训练的最小反例。没有发现足以证明 E03/E04/E06 主科研目标已经完成的证据。
