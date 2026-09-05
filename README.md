# physics-guided-mamba2

> 超快激光加工氧化锆的多尺度形貌—工艺关系研究。

**注意**：仓库名保留为 `physics-guided-mamba2` 是历史原因——项目最初计划直接做 Mamba 预测模型，但数据驱动的探索表明应**先建立可解释的统计结构，再考虑深度模型**。当前核心栈是 NumPy/SciPy/scikit-learn；PyTorch/Mamba 属于 Phase 3+ 的候选方向，尚无实现。

## 研究概览

```text
Phase 1    形貌不稳定性排序（LOCO + 多描述符）
Phase 1.5  形貌描述符 → DCT 谱分解 → 五段带能量
Phase 2    工艺→形貌的 grouped-CV 可预测性（Route P/T 判别）
Phase 2.5  谱组成 + 方向 PSD + 机制桥 + 误差图集
Phase 2.6  单线扫描尺度溯源（W50 / m 分解 / direct bridge）
Phase 2.7  单轨谱包络 × hatch 阵列尺度选择（消融 / m 分解 / forward model）
Phase 2.8  信息分解 + 单轨 kernel → 多轨桥（规划中）
Phase 3    预测建模（未开始——先完成结构收敛 + confirmation 实验设计）
```

## 当前数据

| 数据集 | 样本 | 说明 |
|---|---:|---|
| 矩形 ROI（主） | 200 | 80×80 µm @ 0.5 µm/px，五因素 DOE |
| 单线扫描 | 120 | 285×17.8 µm @ 0.278657 µm/px，四因素 DOE |
| 72 组单脉冲 | 72 | 无设计表，排除 |

矩形主数据共 **200 个 ROI / 实验记录**（120 formal + 60 pass_main + 20 pass_supplement），对应 **160 个唯一 height-source**（`shared_height_source_id`，即部分 measurement 含多个 ROI）与 **134 个 `cv_process_group`**。统计独立性由分组变量单独定义，不以"independent measurement"表述。

## 核心结论（截至 Phase 2.7r1）

- **Route T（方向纹理）**：hatch spacing 单变量几乎承载全部预测能力（ΔR²_h = 0.651/0.645，proc 同向）
- **Route P（谱组成）**：多因素共同调节（ΔR²_h = 0.181/0.350，去 h 后仍有独立贡献）
- **单轨宽度不在 8–16 µm 带内**（pooled W50 = 5.78 µm）
- **λ_peak/h 聚集于 {1,2,3}**（A_obs 0.904，TV 置换 p=0.0001）；m=2 份额逐 h 描述递减，但 block permutation 后 h-dependence 不显著（p=0.4103，descriptive only）
- **简单线性叠加模型不足以重现峰选择**（G27-3 MODEL_INADEQUATE）

## 已知风险与限制

- 功率（已登记项，非风险）：P_obj = 5.3333 W 为**物镜后独立实测平均功率**（post-objective average power，测量物理可信），已升级 canonical 物理输入并登记于 `src/provenance.py`（`POWER_REGISTRY`）；仪器型号/日期元数据 unavailable。Phase 2–2.7 的 `pulse_energy_proxy_uJ` / `areal_dose_proxy_J_per_mm2` 旧列保留供复现，Phase 2.8 起用 canonical 列（`pulse_energy_uJ` / `areal_dose_J_per_mm2`）。注意 P 恒定 ⇒ f 与 E_p 完全耦合（frequency / pulse-energy coupled effect）
- 方向 provenance：无逐样本 scan/hatch 方向 → G-SL4 / G27-4 均为 NOT_APPLICABLE
- 单线 QA 由 AI 辅助标注（GPT），如需论文核心机制证据应补双人独立盲标
- Discovery/Confirmation 未分离：当前 200 样本同时用于探索与报告

## 运行环境

Python 3.12（`.venv`），依赖见 `requirements.txt`。**复现必须使用 `.venv`**——不同 sklearn 版本会导致 Ridge 内层 α 翻转（详见 `outputs/phase2_6/summary/RUNTIME_ENVIRONMENT.md`）。

## 目录

```text
annotations/   人工标注（四边 + 单线盲标）
config/        数据映射、平面与 ROI 参数（含 frozen/）
experiments/   phase1 → phase2.7 研究管线（各自 _lib + scripts + 细则）
outputs/       各 phase 冻结产物（manifest / gate_eval / 科学数据）
src/           共享库（CAG 解码、锥坑修复、重采样、数据契约）
scripts/       顶层执行脚本（15/22/23/32/33/34）
tests/         全 phase 单测（unittest discover）
氧化锆/        原始数据（CAG / 设计表）
专利/          专利材料
```
