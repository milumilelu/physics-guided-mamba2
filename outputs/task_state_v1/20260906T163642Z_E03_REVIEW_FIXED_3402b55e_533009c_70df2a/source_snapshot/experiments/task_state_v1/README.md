# Task-State v1.2.1：已实施入口

当前完成第一批实现与运行，结果入口为 `outputs/task_state_v1/EXECUTION_STATUS_ZH.md`。所有代码位于新命名空间，历史 canonical 输入/模块保持不变。Git 工作区原有修改未提交、未覆盖。

## 可执行命令

在仓库根目录 PowerShell 使用历史环境：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B experiments/task_state_v1/00_audit.py
.\.venv\Scripts\python.exe -X utf8 -B experiments/task_state_v1/01_observer_parity.py --audit-run <E00输出目录>
.\.venv\Scripts\python.exe -X utf8 -B experiments/task_state_v1/02_depth_compression.py --audit-run <E00输出目录> --observer-run <E01_OBS输出目录>
.\.venv\Scripts\python.exe -X utf8 -B experiments/task_state_v1/01_physics_numeric.py
.\.venv\Scripts\python.exe -X utf8 -B experiments/task_state_v1/01_differentiable_backend.py --audit-run <E00输出目录>
.\.venv\Scripts\python.exe -X utf8 -B experiments/task_state_v1/01_gradient_gate.py
.\.venv\Scripts\python.exe -X utf8 -B experiments/task_state_v1/01_g1_close.py --observer-run <E01_OBS输出目录> --physics-run <E01_PHYS_UNIT输出目录> --backend-run <E01_BACKEND_OBS输出目录> --gradient-run <E01_GRAD输出目录>
.\.venv\Scripts\python.exe -X utf8 -B -m unittest discover -s tests/task_state_v1 -v
```

每次执行排他创建新的 run 目录，不覆盖已有成绩。下游必须传入明确的上游 run，验收 release manifest 和 gate，不能按“最新目录”猜测版本。G1 收口必须显式引用四个子 Gate 的 run 目录。

## 范围

- E00：本地身份/raw/target 导出、126 个 MAIN180 分量、5 outer×3 inner 分割、跨 cohort purge 表、环境与代码快照。
- E01_OBS：canonical adapter 与冻结标签的 2000 值 parity、合成场/无定义谱测试。
- E01_PHYS_UNIT：SI NumPy 参考递推、P00/P10/P01/P11、阈下零尾、保持单脉冲能量的精确 multiplicity、单脉冲解析/grid 检查。
- E01_BACKEND_OBS：DifferentiableObserver（torch float64）对 CanonicalObserver 在全部 200 真实 ROI + 5 合成场上的 parity（validity/reason 先行），float64 ≤1e-8、float32 输入 ≤1e-6，并链式复检 canonical 对冻结 target 未漂移。
- E01_GRAD：冻结扫描 fixture 上的 AD/FD 梯度门（32 预注册点 × log(tau)/log(f)/log(v)/log(h) × 5 步长，相邻步长稳定平台选值，≥95% 点 ≤1e-3），加扫描网格加密与脉冲包（冻结子步状态近似、固定物理脉冲串）收敛 ≤1%。平台规则 v2 与 packet 语义修正见 `outputs/task_state_v1/DECISION_LOG_G1_ZH.md`。
- E01_G1_CLOSE：四子 Gate seal 验证 + 聚合 gate_G1.json。
- E02：MAIN180 的 9 种输入×Ridge/GAM，严格内层选择与 Dhat 交叉拟合，90 个外层模型及完整 OOF。C5 与 M_U 等同，不重复拟合。HIST200 重跑、support-matched 解释尚未完成。

E02 的配置只授权静态诊断拟合，不能据此启动 Mamba。尚未实现的 E03–E11 入口不可当作现有命令使用。memory-block refinement 按 runbook 归 E04/B1（tokenization 冻结后做 block 对照），不在 E01 数值门内。

## 已确认的执行修订

主方向 entropy 使用 **36 个角箱**，与 Phase 2.5 冻结产物及 r1 主 loader 一致。原协议的 18 箱来自 r1 raw-recompute 支路，不匹配主 CSV。保留首次 18 箱 FAIL run；未改历史 target、未放宽 tolerance。详见 runbook §29。

## 接续工作

1. 完成训练侧 E03 物理参数校准（P00/P10/P01/P11 独立侧校准 + 可识别性审计）。
2. 完成 B1 reference/loader/continuation 合同与 common memory API（H0/GRU/SSM 先行，Mamba 接入同一 API）。
3. E05/E06 前完成实际反馈 BPTT/backend smoke test 与 compute_budget.csv。
4. HIST200 重跑与 E02 support-matched 科学解释可在 E03 并行补齐。

本机 GPU 为 RTX 5060 Ti（约 8 GB）；CPU 版 PyTorch 2.14.0 已装入历史 `.venv`（纯增量，历史 run 的 environment_lock 不受影响）。CUDA/WSL 环境决策推迟到 E04 前硬件预检；8 GB 不是经过实测的全实验显存可行性保证。
