# WorkBuddy 下一阶段执行计划：人工四边配准与 Phase A 重建

版本：v1（执行前冻结）  
制定日期：2026-08-31  
适用仓库：`physics-guided Mamba-2`  
证据等级：Level 3（单人、盲于自动结果的形貌四边标注）

## 0. 本阶段目标

使用已完成的 200 条人工四边标注建立统一的 `manual_v1` 配准结果，重建
Phase A 的公共画布、`H_reg`、`H_200`、mask 和 QA 证据。v2–v7 只作为
算法一致性 QA，不再作为主中心来源。

本阶段结束于新的 `PHASE_A_APPROVAL_MANUAL_V1.md`，状态必须保持
`PENDING`，等待研究者人工审阅。不得自动批准 Phase A，不得开始 Phase B。

## 1. 当前冻结基线

正式人工标注表：

```text
outputs/rectangle_registration/registration/manual_four_edge_validation.csv
```

已核验基线：

- 共 200 行；`zro2_120_formal=120`、`zro2_60_pass=60`、
  `zro2_20_supplement=20`；
- `annotator_a_state=complete` 为 200/200；
- `annotator_b_*` 全空，符合单人标注决定；
- 无重复 `(session_id, sample_id)`，无缺失四边、非法边界顺序或非正宽高；
- 正式 CSV 的 SHA-256 为
  `536BF658AD75F1BC7DCC6E9461D821E046F58A7636D26A7EAABF51A80D56AF9D`。

现有自动方法状态：v3 为 126 PASS / 38 REVIEW / 36 STOP；v4 为
161/11/28；v5 为 149/24/27；v6 为 190/2/8；v7 为 183/0/17。
v6 是冻结规则下的最佳自动候选，但仍未通过硬门禁；v7 已被拒绝。

## 2. 不可变的科学口径

1. 单人标注不是 Level 2 真值。报告中使用“人工—算法差异”、
   “一致性”或“偏移”，禁止使用“绝对误差”“精度真值”“ground truth”。
2. `annotator_a` 是唯一人工标注者；不得补造、复制或推断 `annotator_b`。
3. 所有 200 个样本统一采用人工中心，不得只对 v6 STOP/REVIEW 使用人工
   fallback，也不得按样本在 manual/v3/v4/v5/v6/v7 之间择优。
4. 人工框宽高保留为 QA 观测；主坐标系和下游名义加工区仍采用固定
   200 × 200 µm。禁止把每条人工框的宽高用于逐样本改变 `H_200` 尺寸。
5. v2 region/edge 分数与 v3–v7 中心全部只作 QA，不参与 `manual_v1`
   主中心计算。
6. 不覆盖或删除任何旧版结果。新结果必须使用独立版本目录或文件名。
7. 脚本不得把任何人工审批文件写成 `PASS`。

## 3. 执行前冻结协议

先新增 `config/manual_registration_v1.yaml`，至少固定以下内容，再运行任何
人工—算法比较：

```yaml
method: manual_four_edge_a_v1
evidence_level: 3
annotator: a
center_source: midpoint_of_four_manual_edges
nominal_box_um: [200.0, 200.0]
manual_geometry_gate:
  observed_width_um: [180.0, 220.0]
  observed_height_um: [180.0, 220.0]
  require_all_complete: true
  allow_unusable: false
  require_paired_order: true
  minimum_paired_center_separation_um: 300.0
algorithm_manual_qa_bands_um:
  close: 2.0
  moderate: 5.0
  large: 5.0
primary_automatic_comparator: v6
secondary_comparators: [v3, v4, v5, v7]
forbid_samplewise_fallback: true
forbid_absolute_error_language: true
```

这里的 2/5 µm 仅用于标记一致性等级，不是绝对误差阈值，也不用于在不同
自动版本间事后选优。将配置文件 SHA-256 写入后续 summary 和 run manifest。

## 4. 工作包与顺序

### WP1：冻结、备份与数据合同检查

新增脚本：`scripts/16_freeze_manual_registration_v1.py`。

职责：

- 只读正式标注表并复核第 1 节全部基线；
- 校验当前 SHA-256；若不一致则 STOP，除非研究者明确确认这是有意修订；
- 校验 200 条的四边、中心、宽高内部恒等关系；
- 使用 `session_geometry.csv` 的 `theta_session_deg` 重新从 `(u,v)` 反算
  `(x,y)`，与保存值比较，容差建议 `1e-6 µm`；
- 校验 paired 测量中槽 1 在左、槽 2 在右且中心间距至少 300 µm；
- 校验人工中心在对应实测视场和冻结物理可行域内；
- 将正式表复制为只读证据快照：
  `registration/manual_v1/manual_four_edge_validation_frozen.csv`；
- 输出 `registration/manual_v1/freeze_manifest.json`，记录源/快照/配置哈希、
  行数、session 数量、状态计数和时间。

失败条件：任一重复、缺失、`unusable`、几何恒等式错误、paired 顺序错误、
越出物理视场或哈希不符。失败时不得自动修表。

### WP2：人工—自动一致性评估

新增：

```text
src/manual_registration_evaluation.py
scripts/17_evaluate_manual_vs_automatic.py
tests/test_manual_registration_evaluation.py
```

对 v3–v7 分别按 `(session_id, sample_id)` 一对一合并。以 session 固定角度
将自动中心转到 canonical 坐标，计算：

```text
delta_u_um = auto_center_u - manual_center_u
delta_v_um = auto_center_v - manual_center_v
center_disagreement_um = hypot(delta_u_um, delta_v_um)
auto_left_u_um   = auto_center_u - 100
auto_right_u_um  = auto_center_u + 100
auto_top_v_um    = auto_center_v - 100
auto_bottom_v_um = auto_center_v + 100
```

再计算四条 `auto edge - manual edge` 差异。输出：

```text
registration/manual_v1/manual_vs_automatic_per_sample.csv
registration/manual_v1/manual_vs_automatic_summary.json
registration/manual_v1/manual_vs_automatic_by_session.csv
registration/manual_v1/manual_vs_automatic_by_status.csv
registration/manual_v1/manual_vs_automatic_by_depth_quartile.csv
qa/manual_v1/manual_vs_automatic_scatter.png
qa/manual_v1/manual_vs_automatic_ecdf.png
qa/manual_v1/manual_width_height_distribution.png
qa/manual_v1/manual_vs_automatic_outliers.png
```

每版至少报告 `delta_u/delta_v` 的 median、MAD、Q05/Q95，径向差异的
median/Q90/Q95/max，以及 `<=2 µm`、`2–5 µm`、`>5 µm` 比例。必须按
session、自动状态和槽深四分位分层，专门列出 v6 的 8 STOP、2 REVIEW，
但不得据此修改其冻结状态。

`outliers.png` 只用于发现明显标注录入错误。若研究者要求纠正，必须保存
原始行、修改原因、修改前后值、操作者和时间到
`manual_annotation_corrections.csv`，更新哈希后从 WP1 全量重跑。禁止因
人工—算法差异大而静默改框。

### WP3：生成全量统一的 manual_v1 主配准表

新增 `scripts/18_build_manual_registration_v1.py`，输入只能是 WP1 的冻结
快照，不直接读取可编辑正式表。

输出：

```text
registration/manual_v1/translation_metrics_manual_v1.csv
registration/manual_v1/translation_summary_manual_v1.json
```

每行至少保留：样本键、measurement/slot、固定 theta/D4、四条人工边、
人工中心 `(u,v)/(x,y)`、人工宽高、`registration_method`、
`evidence_level=3`、源文件哈希、几何 gate、paired gate、状态和 warning。
同时附加 v6 状态及 manual–v6 差异作为 QA 字段。

状态规则必须机械执行：所有人工几何 gate 通过为 `PASS`；否则 `STOP`。
不得因为自动版本一致或不一致改变 manual_v1 状态。

### WP4：让公共画布与重采样脚本显式支持版本化输入输出

修改 `scripts/05_resolve_common_canvas.py` 和
`scripts/06_resample_and_final_level.py`：

- 新增明确参数，例如 `--registration-metrics`、`--output-tag manual_v1`；
- 默认行为保持兼容，不得覆盖现有 v2 档案；
- manual_v1 输出写入：

```text
outputs/rectangle_registration/manual_v1/registered/H_reg/
outputs/rectangle_registration/manual_v1/registered/H_200/
outputs/rectangle_registration/manual_v1/registered/masks/
outputs/rectangle_registration/manual_v1/resampling/
outputs/rectangle_registration/manual_v1/metrics/
```

- `H_reg` 仍需至少 260 µm；`H_200` 固定为中心对称的 200 × 200 µm；
- final leveling 继续只用理论加工区外的参考面并沿用冻结门禁；
- 任一 final-leveling 失败时不得为该样本生成可下游使用的 `H_200`；
- 每个 NPZ 元数据写入方法名、人工标注哈希、配置哈希和生成时间。

### WP5：重建 manual_v1 Phase A QA

将 `scripts/07_generate_phase_a_qa.py` 参数化，或新增
`scripts/19_generate_manual_v1_phase_a_qa.py`。不得覆盖旧 montage。

每张 individual QA 至少显示：

- raw 与 coarse-levelled 高度；
- 人工实际四边框；
- 以人工中心为中心的名义 200 µm 框；
- `H_reg`、`H_200`、raw/registered mask；
- fixed theta、D4、人工宽高；
- v6 中心和状态作为不同颜色的 QA 叠加；
- final plane RMSE、valid fraction、warning；
- local-contrast 图明确水印
  `LOCAL CONTRAST — NOT COMPARABLE IN ABSOLUTE DEPTH`。

输出：

```text
qa/manual_v1/registration_individual/
qa/manual_v1/registration_montage_absolute.png
qa/manual_v1/registration_montage_local.png
qa/manual_v1/phase_a_qa_summary_manual_v1.json
PHASE_A_APPROVAL_MANUAL_V1.md
```

审批文件只能写 `Status: PENDING` 或 `Status: BLOCKED`。自动检查至少包括：

- 200/200 manual geometry gate 通过；
- D4/角度仍为已冻结确认值；
- paired 槽顺序与间距通过；
- 所有样本 `L_reg >= 260 µm`；
- final leveling 全部通过且参考面充分；
- mask/公共画布完整；
- 不存在按样本方法混用；
- provenance 哈希链闭合。

### WP6：测试、重跑与审计

至少增加以下测试：

1. canonical `(u,v)` 与原始 `(x,y)` 双向变换；
2. 人工边中点与保存中心一致；
3. 自动理论四边的符号和 y 向下约定；
4. 一对一 merge、缺样本和重复样本硬失败；
5. paired 槽顺序/间距 gate；
6. 单样本选择不同中心来源时硬失败；
7. 旧输出不会被 manual_v1 运行覆盖；
8. 审批脚本不能写 `PASS`；
9. 合成平面上的 manual center 重采样与 final leveling；
10. 一个真实 CAG 样本的端到端 smoke test。

正式运行前先执行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\verify_environment.py
```

然后按 WP1 → WP6 顺序运行。每一步非零退出立即停止，不得跳过 gate。将
完整命令、退出码、配置/输入/输出哈希、Python 与依赖版本追加到
`outputs/rectangle_registration/manual_v1/run_manifest.json`。

## 5. 最终验收与停止点

WorkBuddy 交付时必须满足：

- 所有新增/修改测试通过；
- 正式人工源表未被改写；
- `manual_v1` 200 行唯一、完整、无 sample-wise fallback；
- 旧 v2–v7 CSV、图像和 registered 数据未被覆盖；
- manual_v1 的 resampling/final-leveling 自动门禁全部有明确结果；
- individual QA 数量为 200，两套 montage 可打开且标注清楚；
- summary、manifest、CSV、NPZ 之间的哈希和样本数一致；
- `PHASE_A_APPROVAL_MANUAL_V1.md` 仍为 `PENDING`。

到此必须停止并请研究者人工查看 montage/individual QA。只有研究者本人明确
将 manual_v1 审批状态改为 `PASS` 后，才能另立 Phase B 任务；不得在本任务
中继续计算边缘污染距离或 stable ROI。

## 6. 明确禁止事项

- 不得新增 v8 自动算法或继续调 v6/v7 阈值；
- 不得删除 v6 STOP 或浅槽样本；
- 不得把 manual_v1 与 v6 混成逐样本最优结果；
- 不得将人工—算法差异称为绝对误差；
- 不得因看见比较结果而修改第 3 节阈值；
- 不得使用 `qa/manual_ui_click_test.csv` 或草稿 JSON 作为正式证据；
- 不得开始 Phase B，不得自动批准 Phase A。
