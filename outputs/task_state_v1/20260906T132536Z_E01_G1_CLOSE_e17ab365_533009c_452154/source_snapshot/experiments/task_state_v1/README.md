# Task-State v1.2.1：已实施入口

当前完成第一批实现与运行，结果入口为 `outputs/task_state_v1/EXECUTION_STATUS_ZH.md`。所有代码位于新命名空间，历史 canonical 输入/模块保持不变。Git 工作区原有修改未提交、未覆盖。

## 可执行命令

在仓库根目录 PowerShell 使用历史环境：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B experiments/task_state_v1/00_audit.py
.\.venv\Scripts\python.exe -X utf8 -B experiments/task_state_v1/01_observer_parity.py --audit-run <E00输出目录>
.\.venv\Scripts\python.exe -X utf8 -B experiments/task_state_v1/02_depth_compression.py --audit-run <E00输出目录> --observer-run <E01_OBS输出目录>
.\.venv\Scripts\python.exe -X utf8 -B experiments/task_state_v1/01_physics_numeric.py
.\.venv\Scripts\python.exe -X utf8 -B -m unittest discover -s tests/task_state_v1 -v
```

每次执行排他创建新的 run 目录，不覆盖已有成绩。下游必须传入明确的上游 run，验收 release manifest 和 gate，不能按“最新目录”猜测版本。

## 范围

- E00：本地身份/raw/target 导出、126 个 MAIN180 分量、5 outer×3 inner 分割、跨 cohort purge 表、环境与代码快照。
- E01_OBS：canonical adapter 与冻结标签的 2000 值 parity、合成场/无定义谱测试；尚未实现可微 observer。
- E02：MAIN180 的 9 种输入×Ridge/GAM，严格内层选择与 Dhat 交叉拟合，90 个外层模型及完整 OOF。C5 与 M_U 等同，不重复拟合。HIST200 重跑、support-matched 解释尚未完成。
- E01_PHYS_UNIT：SI NumPy 参考递推、P00/P10/P01/P11、阈下零尾、保持单脉冲能量的精确 multiplicity、单脉冲解析/grid 检查。不是完整扫描求解器、真实参数校准或梯度 Gate。

E02 的配置只授权静态诊断拟合，不能据此启动 Mamba。尚未实现的 E03–E11 入口不可当作现有命令使用。

## 已确认的执行修订

主方向 entropy 使用 **36 个角箱**，与 Phase 2.5 冻结产物及 r1 主 loader 一致。原协议的 18 箱来自 r1 raw-recompute 支路，不匹配主 CSV。保留首次 18 箱 FAIL run；未改历史 target、未放宽 tolerance。详见 runbook §29。

## 接续工作

1. 在独立算法环境实现可微 observer，与 36 箱 canonical 对齐。
2. 冻结扫描情景/guard region，完成 spatial/event convergence 和 AD/FD。
3. 完成训练侧 E03 校准、B1 reference/loader/continuation 合同与 common memory API。
4. 通过实际反馈 BPTT/backend 与资源 smoke test 后，才开展深度主实验。

本机 GPU 为 RTX 5060 Ti（约 8 GB），当前历史环境无 PyTorch，WSL 未安装；未改动系统或安装深度依赖。8 GB 不是经过实测的全实验显存可行性保证。
