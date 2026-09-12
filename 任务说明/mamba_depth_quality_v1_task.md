# mamba_depth_quality_v1 执行细则

版本：1.0（2026-09-12）  
依据：`_task_unpack/mamba_depth_quality_task_v1/TASK_SPEC_ZH.md`、`protocol.template.yaml`、`CODEX_EXECUTION_PROMPT_EN.md`。  
执行范围：仅执行 Route A pilot；Route B、机理选择、因果压缩、expanded 多种子均不在本轮。

## 一、工作区与边界

1. 首先记录当前 HEAD、`git status --porcelain=v1`、Python/依赖环境和任务包哈希；保留已有未提交改动。
2. 只新增以下白名单路径：
   - `config/mamba_depth_quality_v1/`
   - `src/mamba_depth_quality/`
   - `experiments/mamba_depth_quality_v1/`
   - `tests/mamba_depth_quality_v1/`
   - `outputs/mamba_depth_quality_v1/`
   - `任务说明/mamba_depth_quality_v1_task.md`
3. 不 reset、clean、stash、push，不覆盖旧协议、旧输出或其他工作流文件。若共享模块必须适配，单独记录文件、原因和差异。

## 二、执行顺序与门禁

### G0：数据与合同冻结

- 严格加载并校验协议 YAML；未知字段、缺字段、未知模型 ID 直接失败。
- 对齐 MAIN180 高度包与冻结身份键，验证 180 行、126 个关联分量、5 外层折、3 内层折及无组间重叠；SUPP20 排除。
- 在固定 160×160、0.5 µm/pixel raw ROI 上计算 `D=-median_valid(H)`、`Sq=rms_about_mean(H)`，并保存目标字典、身份哈希、输入哈希、分组清单。
- 对 D 做 canonical parity；用独立 NumPy float64 实现核对 Sq；验证整体平移只改变 D、不改变 Sq/谱标签。
- 检查目标列、形貌列、身份列不会进入推理输入。

### G1：经验参考与 token 合同

- 实现 `RLOG_EQEXP_v1`：SI 内部计算 `Ep=P/f`、`F0=2Ep/(πw0²)`、`Aeq=πw0²/2`、`νpass=fAeq/(vh)`、`ν=Nνpass`、`D0=δν[max(log(F0/F1),0)]`。
- 在每个训练分区内用 64 个 F1 候选和 component-balanced 非负最小二乘标定 δ；不查看验证/测试目标；保存完整 profile、边界标记和参数来源。
- 内层训练序列使用训练子集内部 3 折 cross-fit 参考；验证/测试只用对应训练分区全量拟合参考。
- 每个 pass 取 8 个等效曝光进度 token，严格使用协议 16 列及 raw-ablation 列合同；长度为 `8*N`，mask/最后有效位置规则固定；归一化只用训练侧分量/样本平衡统计。
- 验证剂量恒等式、token 单调性、参考终态与最后 token 一致性。

### G2：模型与数值验收

- 复用并核验纯 PyTorch Mamba-2 参考实现，必要时在新命名空间写 adapter；记录 backend、来源提交和修改。
- 验证 full/step parity、float64/float32 容差、mask 不改变有效 token、inactive cache 不更新、单样本/混合 batch/换序一致、梯度有限且无目标输入。
- 实现共享终态头：4/32 维 bottleneck；深度参考残差有界且末层零初始化；Sq 为 softplus；AUX 为 `ilr_z1/z2` 标准化头。
- 实现 `STATIC_PHY`、`GRU_PHY`、`MAMBA_RAW`、`MAMBA_PHY` 及 `CONST_DQ`、`TREE_U_DQ`，固定协议中的容量、种子、损失和训练预算。

### G3：pilot 运行与发布

- 执行 5 折、seed=17 的全部 8 个模型；内层只选择 epoch，外层测试不早停；输出完整 OOF 覆盖。
- 计算分量平衡 D/Sq MAE、RMSE、Q²、负深度比例、残差界饱和，并做以 component/outer-fold 分层的配对 bootstrap（5000 次；资源不足时记录降级）。
- 保存协议解析、输入/目标/token 字典、split、参考 profile/参数、训练历史、模型预算、OOF、指标、配对差异、失败注册、成本、checkpoint、预处理、源码快照、环境、git 状态、测试日志、release manifest、中文结果报告。
- 在干净临时检出/独立目录中做导入、reload、process-only 推理和封缄文件检查；失败运行保留且标 INVALID，不覆盖。

## 三、固定实验矩阵

`CONST_DQ`、`TREE_U_DQ`、`STATIC_PHY_K4_AUX`、`GRU_PHY_K4_AUX`、`MAMBA_RAW_K4_AUX`、`MAMBA_PHY_K4_NOAUX`、`MAMBA_PHY_K4_AUX`、`MAMBA_PHY_K32_AUX`。所有模型同时预测 D 与 Sq；RLOG 单目标 D 仅作为附加报告，Q 标为 `NOT_MODELED`。

## 四、完成判定

- `IMPLEMENTATION_PASS`：G0–G3、导入/reload、数据/数值/泄漏/覆盖检查均通过。
- 预测结论分为 `PREDICTIVE_GAIN`、`NO_STABLE_GAIN`、`INCONCLUSIVE`；模型不赢不影响工程完成。
- 报告必须明确：Q1–Q4、实际执行与仅实现内容、D/Q 是否冲突、参考失配/可辨识性、种子覆盖、压缩代价、下一步仅推荐一个实验。
- 任何未执行内容不得写成已完成；不把终态监督结果解释为真实时间记忆、因果识别或物理状态维数。

## 五、执行命令合同

```powershell
python experiments/mamba_depth_quality_v1/00_audit_contract.py --config config/mamba_depth_quality_v1/protocol.yaml
python experiments/mamba_depth_quality_v1/01_reference_and_backend_tests.py --contract-run <G0目录>
python experiments/mamba_depth_quality_v1/02_run_pilot.py --contract-run <G0目录> --backend-run <G1_G2目录>
python experiments/mamba_depth_quality_v1/04_report.py --run <pilot目录>
```

本文件是实施前的执行细则；运行后在同一任务目录追加实际命令、run ID、失败项、资源成本和结果，不修改本细则中的冻结设计。

## 六、实际执行记录（2026-09-12）

- G0：`outputs/mamba_depth_quality_v1/20260912T065752Z_G0`。MAIN180=180、关联分量=126、5 外层折；高度包哈希与协议一致；D raw parity 最大绝对差 `1.91e-6 µm`（冻结表为有限小数），Sq 独立重算，且与 A_med 最大差 `1.5794 µm`。
- G1/G2：`outputs/mamba_depth_quality_v1/20260912T070149Z_G1_G2`。RLOG 64 点 profile、16 列 token、full/step 最大差 0、padding 不变最大差 0、梯度有限。
- Pilot：`outputs/mamba_depth_quality_v1/20260912T070210Z_pilot`。8 个模型、5 折、seed=17、每模型 180 条 OOF，共 1440 条；真实纯 PyTorch Mamba adapter 运行 12 个 fast-profile epoch。生成指标、配对 bootstrap 5000、源码快照、环境、git 状态、测试日志和发布清单。
- 状态：`PARTIAL_IMPLEMENTATION_PASS`。内层 3 折参考 cross-fit、内层早停/300 epoch 固定预算及干净检出 reload 尚未完成，结果不能当作完整协议验收或多 seed 独立确认。
