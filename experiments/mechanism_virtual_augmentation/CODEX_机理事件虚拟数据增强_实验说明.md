# Codex 实验任务说明：检验 E1/E2/E5 是否是真实有效的机理信息，而不是“为了创新而创新”

## 0. 任务定位

不要把本任务当成“把 E1/E2/E5 做得更复杂”的开发任务。

本实验唯一目的，是检验下面这个研究假设是否成立：

> 与普通 LHS 虚拟样本、以及现有连续机理特征空间 coverage 相比，基于 E1/E2/E5 机理事件选择虚拟工艺样本，是否能在相同虚拟样本预算下，对真实测试样本产生稳定、可复现的额外预测收益。

如果没有收益，应明确输出“当前数据不支持 event-signature augmentation”，不要为了得到正结果调整阈值、筛选样本或更换评价协议。

本实验还要单独判断 E2 是否提供额外价值。E2 当前属于探索性候选事件，不要预设其一定有效。

---

## 1. 代码基础

优先基于现有脚本：

- `depth_mechanism_transition_virtual_data_v2.py`（如果当前工作区存在）
- 否则基于 `depth_mechanism_sequence_surrogate.py`

不要覆盖原始脚本。

新建：

- `experiment_event_augmentation_ablation.py`

尽量复用原脚本已有的数据读取、物理参数、机理递推、feature construction、CV 和 sklearn model 函数，不要大规模重写整个工程。

原有递推已经计算了：

- `fluence`
- `threshold`
- `margin = log(fluence / threshold)`
- `inc`
- `z`
- `total_defocus`
- pass-level removal increment
- cumulative depth
- core5/physics proxy features

本实验只在此基础上增加“事件提取”和严格对照实验。

---

## 2. 实验问题

要回答 4 个问题：

### Q1
E1/E5 是否在真实样本中具有足够的分布差异，而不是几乎所有样本都得到同一个事件状态？

### Q2
E2 是否提供 E1/E5 之外的新增区分信息？

### Q3
基于事件覆盖选择虚拟样本，是否优于普通 LHS？

### Q4
更关键地，基于事件覆盖选择虚拟样本，是否优于现有连续 physics/core5 feature-space coverage？

如果 Q4 不成立，就不能认为事件签名对当前研究问题提供了额外技术价值。

---

## 3. 事件定义

### 3.1 E1：烧蚀停止事件

每一个 active pass 内，对每个递推 step 保存：

```text
margin_j = log(fluence_j / threshold_j)
```

定义：

```text
E1_stop:
margin_(j-1) > 0  and  margin_j <= 0
```

含义：

```text
有效烧蚀 -> 无有效烧蚀
```

记录：

- `E1_stop_exists`
- `E1_stop_pass`
- `E1_stop_step`
- `E1_stop_t_norm`

其中全局归一化发生位置建议：

```text
t_norm = ((pass_index - 1) + (step_index + 1) / steps) / pass_count
```

范围约为 `[0, 1]`。

本轮实验只使用 `E1_stop` 作为主事件。

可以额外记录：

```text
E1_start:
margin_(j-1) <= 0 and margin_j > 0
```

但不要把 E1_start 放进第一轮主评分；只作为诊断输出。

---

### 3.2 E2：孵化—离焦竞争事件（探索性）

E2 必须限制在同一个 pass 内检测，不允许跨 pass 直接比较，避免 `pulse_index` 在下一 pass 重新开始造成伪事件。

对每一个 step：

```text
g2_j =
    (1 - S) * log(pulse_index_j)
    - log(1 + (total_defocus_j / zR)^2)
```

分别记录两个方向：

```text
E2_inc_to_def:
g2_(j-1) > 0 and g2_j <= 0

E2_def_to_inc:
g2_(j-1) < 0 and g2_j >= 0
```

不要只保留一个方向。

记录：

- `E2_exists`
- `E2_direction`
- `E2_pass`
- `E2_step`
- `E2_t_norm`

如果一条样本有多次 E2 crossing：
- 保存第一次 crossing 作为主事件；
- 同时输出 `E2_crossing_count`；
- 不要人为挑选“最好看的” crossing。

E2 当前是探索性变量，不得在实验后根据结果修改公式再重跑主结果。

---

### 3.3 E5：边际去除由增强转为衰减

优先使用 step-level `inc_j`，不要只用最终 pass depth。

对每个 active pass 内：

```text
dinc_j = inc_j - inc_(j-1)
```

定义：

```text
E5:
dinc_(j-1) > 0 and dinc_j <= 0
```

也就是 `inc_j` 首次从持续增加转为不再增加。

记录：

- `E5_exists`
- `E5_pass`
- `E5_step`
- `E5_t_norm`
- `E5_inc_peak`

如果由于浮点数噪声出现高频符号抖动：
- 仅允许使用一个固定的数值 epsilon；
- epsilon 必须在实验开始前固定并写入 config；
- 不允许针对不同 fold 或不同样本调 epsilon。

建议默认：

```text
eps = max(1e-12, 1e-8 * median_positive_inc_in_training_fold)
```

所有 event strategy 共用同一个定义。

---

## 4. 事件表示

构造两个版本，不要只测试一个版本。

### Event-15

只使用：

```text
E1 + E5
```

用于判断两个相对自然的事件是否已经有价值。

每个样本的 signature 至少包含：

```text
E1_stop_exists
E1_stop_t_bin
E5_exists
E5_t_bin
event_order_E15
```

`t_bin` 使用固定 3 区间：

```text
early: 0 <= t < 1/3
middle: 1/3 <= t < 2/3
late: 2/3 <= t <= 1
```

### Event-125

使用：

```text
E1 + E2 + E5
```

在 Event-15 基础上增加：

```text
E2_exists
E2_direction
E2_t_bin
event_order_E125
```

目的不是一定让 Event-125 赢，而是直接检验：

> E2 是否真的提供额外价值。

---

## 5. 对照策略

必须使用相同的 candidate pool、相同的 virtual labeler、相同的下游模型、相同虚拟样本预算，只改变“虚拟样本怎么选”。

至少比较 5 组：

### S0 — Real-only baseline
不增加虚拟样本。

### S1 — LHS
在当前 training fold 的可执行参数范围内生成候选池后，按固定随机种子从候选池中选择 K 个样本。

### S2 — Continuous physics/core5 coverage
利用现有 core5 或 compact physics features：

1. 只用当前 training fold 计算标准化均值和标准差；
2. 将真实训练样本和候选样本映射到同一连续物理特征空间；
3. 使用 greedy farthest-point sampling 选择 K 个候选，使其尽量补充现有训练集的连续 feature-space coverage。

这是最重要的对照组。

### S3 — Event-15 coverage
优先选择能够增加 E1/E5 signature coverage 的候选。

候选优先级：

1. 新 event combination
2. 新 event order
3. 已有 event/order 下的新 timing bin
4. 若所有候选都不再增加 event coverage，则使用 core5 farthest-point 作为 fallback 补足到 K 个

必须记录：

```text
event_positive_gain_count
event_fallback_count
event_fallback_fraction
```

如果大部分样本都依赖 fallback，本身就是“事件空间信息量不足”的证据。

### S4 — Event-125 coverage
与 S3 相同，但使用 E1/E2/E5。

---

## 6. Candidate pool

每个 CV training fold 只根据该训练 fold 构造候选范围。

禁止读取 test fold 的最小值、最大值、类别集合或统计量。

保守做法：

- `pulse_width_fs`：从训练 fold 已出现的离散档位采样
- `repetition_rate_khz`：若训练数据是明显离散档位，则只从训练 fold 已出现档位采样
- `scan_speed_mm_s`：训练 fold `[min, max]`
- `hatch_spacing_um`：训练 fold `[min, max]`
- `pass_count`：训练 fold `[min, max]` 内整数

候选池大小第一轮固定：

```text
M = 5000
```

同一个 fold、同一个 repeat 下，S1/S2/S3/S4 必须共享完全相同的 5000 个 candidate。

建议用 `scipy.stats.qmc.LatinHypercube`；如果项目当前不依赖 scipy，则使用已有稳定实现，不能因此大改依赖。

---

## 7. 虚拟样本预算

第一轮固定测试：

```text
K = 10, 20, 30
```

所有策略严格使用同样数量的虚拟样本。

如果 event strategy 无法找到足够 positive-gain 样本，必须使用前述 fallback 补到 K，同时报告 fallback fraction。

不要改变每种方法的虚拟样本数量来追求更好结果。

---

## 8. 虚拟标签

所有策略必须使用同一个 fold-internal labeler。

推荐第一轮使用：

```text
residual_ridge
```

流程：

1. 当前 training fold 用机理模型得到：
   `y_phys_train`
2. 计算：
   `residual = y_measured - y_phys_train`
3. 仅在 training fold 上训练一个 Ridge residual mapper：
   `r_hat = Ridge(physics/core5 features -> residual)`
4. 候选虚拟标签：
   `y_virtual = y_phys_virtual + r_hat(x_virtual)`
5. clip 到合理非负范围。

严禁：
- 用全数据训练 labeler；
- 用 test fold 测量值参与 physics fitting；
- 用 test fold 参与 scaler；
- 用 test fold 参与 candidate range；
- 用 test fold 参与虚拟样本选择。

如果原脚本已有严格 fold-internal physics fitting，应直接复用。

---

## 9. 下游评价模型

第一轮主模型固定为：

```text
GBDT + core5
```

目的：只比较 augmentation strategy，不同时比较模型选择。

虚拟样本训练权重固定：

```text
real sample weight = 1.0
virtual sample weight = 0.5
```

如果当前 sklearn pipeline 对 sample_weight 支持不方便，可在 GBDT fit 层显式传入，不要悄悄复制虚拟样本模拟权重。

可选二级敏感性分析：

```text
GPR + core5
```

但 GPR 结果只能作为 secondary result，不能用“哪个模型结果好就报告哪个”。

---

## 10. 交叉验证协议

必须采用：

```text
5-fold × 5-repeat repeated CV
```

使用固定 seed。

每一个 split 内完整执行：

```text
train/test split
    ↓
train-only physics parameter fitting
    ↓
train-only feature scaling / event calibration
    ↓
train-only candidate domain
    ↓
shared candidate pool generation
    ↓
candidate physics traces + events + virtual labels
    ↓
S1/S2/S3/S4 各自选 K
    ↓
real + selected virtual training
    ↓
predict untouched real test fold
```

test fold 在最终 predict 之前不能参与任何环节。

---

## 11. 第一层诊断：先看事件本身有没有信息

在 augmentation 结果之前，必须先输出真实样本的事件诊断。

对每个 measured row 输出：

```text
run_id
E1_stop_exists
E1_stop_t_norm
E1_start_exists
E5_exists
E5_t_norm
E5_inc_peak
E2_exists
E2_direction
E2_t_norm
E2_crossing_count
signature_E15
signature_E125
```

同时输出：

1. 每个事件出现率
2. E15 signature 的样本数分布
3. E125 signature 的样本数分布
4. event timing 分布
5. 是否存在某个 signature 占全部样本 > 80%

如果某个 event 对 >95% 样本都完全相同，要在 summary 中明确写：

```text
low discriminative value
```

不要隐藏。

---

## 12. 第二层诊断：事件是否对应原模型的困难区域

使用 real-only baseline 的 OOF residual：

```text
residual_i = prediction_i - measured_i
abs_error_i = abs(residual_i)
```

比较：

- E1 exists vs not exists
- E5 exists vs not exists
- E2 exists vs not exists
- E1/E5 early/middle/late
- E15 signatures
- E125 signatures

输出每组：

```text
n
mean_abs_error
median_abs_error
rmse
```

只做描述性分析。

由于数据量小，不要把普通 p-value 当成核心结论。

如果 event group 与 baseline error 完全没有关系，应在最终报告中指出：

> 当前事件可能有物理解释，但没有显示出与当前预测困难区域的对应关系。

---

## 13. 主要评价指标

主指标：

```text
RMSE on real held-out samples
```

副指标：

```text
MAE
R2
repeat-wise RMSE mean ± std
```

对每个：

```text
strategy × K
```

输出：

```text
RMSE
delta_RMSE_vs_real_only
delta_RMSE_vs_LHS
delta_RMSE_vs_core5
percentage_improvement_vs_core5
win_count_vs_core5_across_repeats
```

其中最重要的是：

```text
Event-15 vs Core5 coverage
Event-125 vs Core5 coverage
Event-125 vs Event-15
```

---

## 14. 不要只看平均 RMSE

额外计算：

### A. 每个 repeat 的配对差值

```text
ΔRMSE_repeat =
RMSE_event_repeat - RMSE_core5_repeat
```

报告 5 个 repeat 的具体值。

### B. OOF per-sample absolute error difference

```text
ΔAE_i =
abs_error_event_i - abs_error_core5_i
```

对样本做 bootstrap，给出平均 `ΔAE` 的 95% bootstrap CI。

不要因为 repeated CV 的 fold 相关性，直接声称普通独立样本 t-test 有严格显著性。

---

## 15. 研究判定规则

这些是研究筛选规则，不是统计学定理。

### 支持 Event-15 继续研究

至少同时满足：

1. Event-15 相对 Core5 coverage 在至少一个 K 下：
   ```text
   mean RMSE improvement >= 5%
   ```
2. 5 个 repeat 中至少 3 个 repeat 的 RMSE 更低；
3. MAE 没有明显恶化；
4. event fallback fraction 不应长期接近 1；
5. 真实样本中 E1/E5 不是几乎常量。

### 支持 E2 继续研究

Event-125 必须相对 Event-15 显示额外收益，例如：

```text
RMSE 再下降 >= 2%
```

或者在多个 K / repeats 上稳定改善。

如果 Event-125 ≈ Event-15 或更差：

> 标记 E2 为“当前数据不支持”，后续专利/论文基线优先去掉 E2。

### 否定整个 event-signature 路线

如果：

```text
Event-15 <= LHS
```

或者：

```text
Event-15 ≈ Core5 coverage
```

且 Event-125 也无稳定增益，

最终 README 必须明确写：

> 当前实验没有证据表明离散机理事件表示比连续机理特征空间更适合指导虚拟样本生成。建议停止围绕 event signature 增加复杂度。

不要继续调 event threshold 找正结果。

---

## 16. 输出文件

输出目录例如：

```text
outputs/event_augmentation_ablation/
```

至少生成：

```text
experiment_config.json
event_diagnostics_real_samples.csv
event_signature_distribution.csv
baseline_residual_by_event.csv

cv_split_metrics.csv
cv_summary_by_strategy_budget.csv
oof_predictions.csv

virtual_candidates_by_fold.csv
virtual_selected_by_fold.csv
event_coverage_gain_by_fold.csv

bootstrap_event_vs_core5.csv

README_EXPERIMENT_RESULT.md
```

图：

```text
rmse_vs_virtual_budget.png
delta_rmse_vs_core5.png
repeatwise_rmse_delta_event15_vs_core5.png
repeatwise_rmse_delta_event125_vs_event15.png
event_frequency.png
event_timing_histograms.png
```

---

## 17. README_EXPERIMENT_RESULT.md 必须回答

不要只贴指标表。

README 最终必须明确回答：

1. E1 在多少真实样本中出现？
2. E5 在多少真实样本中出现？
3. E2 在多少真实样本中出现？
4. E15/E125 是否真的把数据划分成多个有样本量的 regime？
5. 哪些 event/regime 对 real-only baseline 来说更难预测？
6. Event-15 是否稳定优于 LHS？
7. Event-15 是否稳定优于 continuous core5 coverage？
8. Event-125 是否优于 Event-15？
9. event strategy 中有多少虚拟样本实际上是 fallback 选出来的？
10. 最终结论必须在下面三项中选一项：

```text
A. 支持继续研究 event-guided virtual data generation
B. 只支持 E1/E5，当前不支持 E2
C. 当前数据不支持 event-signature 路线，应回退到连续 physics-feature augmentation
```

---

## 18. 编程要求

- 不覆盖原始脚本；
- 所有随机数统一 seed；
- 所有关键超参数写入 `experiment_config.json`；
- 关键函数加 docstring；
- 添加断言检查 train/test leakage；
- candidate pool 在同一 fold 内必须被所有策略共享；
- 所有 strategy 必须使用同一 virtual labeler；
- 所有 strategy 必须使用同一 downstream model；
- 所有 strategy 在同一 K 下虚拟样本数量一致；
- 输出中保存每个虚拟样本被哪种策略选中以及选择原因；
- 对 NaN、无事件、pass_count=1 等边界情况显式处理；
- 先执行 `python -m py_compile`；
- 再做一个小规模 smoke test；
- 最后再跑正式 5×5 repeated CV；
- 如果完整实验耗时过长，先报告预计耗时和瓶颈，再在不改变实验逻辑的前提下优化缓存；不要擅自降低 CV 次数作为最终结果。

---

## 19. 最重要的实验纪律

本实验是在判断一个假设是否值得继续，不是在证明它一定正确。

禁止：

- 为了让 Event-125 赢而反复修改 E2；
- 根据 test fold 结果修改 event threshold；
- 给不同策略使用不同 candidate pool；
- 给不同策略使用不同 virtual labeler；
- 给事件策略更多虚拟样本；
- 只报告最好的 K；
- 只报告最好的 repeat；
- 删除对事件方案不利的样本；
- 发现负结果后继续增加新事件直到得到正结果。

如果结果为负，保留负结果并明确说明。

负结果本身就是本实验的重要结论。
