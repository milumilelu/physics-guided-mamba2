# E03 Solver-Registration FAIL Triage（commit `f80bca0`）

- **Verdict：`FAIL_FORMAL_RETAIN`**
- 旧 run 保留为正式科研证据，**不删除、不覆盖、不重新解释**。
- 未发现 identity 错误、数值实现错误、单位错误、registration 契约错误或 supervision 泄漏。
- 三项 remaining 全部是**在预注册阈值下未达标**，不是代码缺陷。
- 本 triage **未修改任何阈值**；`tail_relative_tolerance=0.01`、`max_window_attempts_allowed=4`、
  `implausible_depth_factor=10.0` 全部保持原值。

---

## 0. 证据来源（全部实际读取，非依据 commit message）

| 项目 | 路径 / 值 |
|---|---|
| Commit | `f80bca0`（父 `8c80743`；run 时 repo sha `77c2c3cbbaf408c3feac77aa1f627147c6e6ae84`，dirty） |
| Run 目录 | `outputs/task_state_v1/20260908T044922Z_E03_SOLVER_REGISTRATION_3d172ad2_77c2c3c_d7e592` |
| 逐作业 JSON | `jobs/` = **3884 / 3884**（`job_plan.json` 计划 48+3744+40+48+4） |
| Gate | `solver_registration_gate.json` → `status=FAIL`, `allowed_to_calibrate=false` |
| Resolved config | `config_source.yaml`，与 `config/task_state_v1/solver_registration.yaml` **逐字节相同**（`diff` 无输出） |
| Calibration config 快照 | `calibration_config_source.yaml` |
| Release manifest | `release_manifest.json`（3981 文件 + sha256；gate / config / job_plan / summary CSV 校验全部 match） |
| Source snapshot | `source_snapshot.json`（79 文件含 `03_register_solver.py`） |
| Aggregate summaries | `recipe_coverage_summary.csv`、`parameter_domain_summary.csv`、`*_checks.csv`、`domain_enforcement.csv`、`determinism.csv`、`registration_inputs.json` |
| 环境 | Python 3.10.12 / numpy 1.26.4 / scipy 1.15.3 / pandas 2.3.2 / scikit-learn 1.7.1 / PyYAML 5.4.1，Linux 5.15 |
| 契约绑定 | `calibration_semantic_sha256=9c3a6cae…`、`calibration_config_sha256=037c0605…`、`input_sha256=0396071b…` |

运行完整性：`all_jobs_accounted=true`；五个阶段的 `worker_errors` 全为空；无 traceback。
→ **run 本身执行完整、可审计、可复现**。

---

## 1. 实际失败的 Gate 是什么？

`solver_registration_gate.json → status = FAIL`，`remaining` 三项：

```text
full_recipe_coverage
parameter_domain_finite_and_converged
grid_refinement_converged
```

已通过的项（不要误读为失败）：

| 条件 | 结果 |
|---|---|
| `all_jobs_accounted` | **true**（3884/3884） |
| `window_self_consistent` | **true**（观测 max 2 次 ≤ 4） |
| `recipe_coverage_finite_and_converged` | **true**（528/528 门检查通过，0 nonfinite，0 solver failure） |
| `out_of_bounds_rejected` | **true**（40/40 拒绝，0 误收） |
| `determinism_bitwise` | **true**（4/4 逐位一致） |

---

## 2–4. 逐项：预注册阈值 / 实测值 / FAIL 性质

### F1 — `full_recipe_coverage`

| 项 | 值 |
|---|---|
| 预注册阈值 | `full_recipe_coverage := (limit == 0)`，即全部 MAIN180 配方 → **2160 案例**（180 × 4 开关 × 3 光腰） |
| 实测 | 本次 `--limit-recipes 4` → **48 案例**（4 配方 × 4 开关 × 3 光腰），覆盖 **2.22 %** |
| 性质 | **范围（scope）未完成**，不是物理/数值负结果，也不是代码缺陷 |
| 分类 | `IMPLEMENTATION_REMAINING` |

### F2 — `parameter_domain_finite_and_converged`

| 项 | 值 |
|---|---|
| 预注册阈值 | 3744 个参数向量 **全部** `accepted=True 且 all_finite=True`；`max_window_attempts ≤ max_window_attempts_allowed = 4`；41154 项 descriptor 检查全部 `error ≤ tail_relative_tolerance = 0.01` |
| 实测 | **3 / 3744 = 0.080 %** 向量被拒，**3 / 41154** 项检查失败 |
| 失败原因 | 3 例均为 `ValueError: Window attempt registration budget exceeded` |
| 失败作业 | `domain_P10_45_w0.874_r3`、`domain_P11_77_w0.874_r3`、`domain_P11_124_w0.874_r1` |
| 共性 | **全部在最小光腰 w0 = 0.874 µm**，**全部是 defocus 开关（P10 / P11）**；向量落在盒内（非顶点） |
| 性质 | **求解器注册负结果，非代码缺陷**：求解器在其内部 `max_window_retries=16` 预算内**确实收敛**（否则触发的会是 `Finite scan failed window/update budget`，而不是本消息），只是所需窗口扩张次数超过预注册的 4 次验收预算。这正是 `E03_REGISTRATION_AUDIT_20260908_ZH.md` 第 5 点预设的行为：「超过 4 次应报告失败并审查重放成本，不应静默接受或为通过而放宽」 |
| 分类 | `BLOCKING_SOLVER_REGISTRATION` |

### F3 — `grid_refinement_converged`

| 项 | 值 |
|---|---|
| 预注册阈值 | 640 → 1280 点网格细化（双线性映射回固定 160 像素中心）+ 1280 映射 vs 直接 ROI，共 960 项 descriptor 检查，全部 `error ≤ 0.01` |
| 实测 | **2 / 960 = 0.21 %** 失败，均为 `check=grid_finest`、`descriptor=ilr_z2` |
| 失败作业 | `grid_0_P00_w2.0` → 0.0108043；`grid_0_P10_w2.0` → 0.0108875（超阈 **8.9 %**） |
| 其他 descriptor 最大值 | `A_med` 0.008252、`D` 0.005934、`ilr_z1` 0.005832、其余 < 0.002；`grid_to_direct_points` 全部通过（ilr_z2 max 0.006080） |
| 性质 | **数值离散化收敛负结果**，不是代码缺陷。单一描述子 `ilr_z2` 在最密曝光（dataset_index 0）、w=2.0 µm 下，640→1280 尚未收敛到 1 % |
| 分类 | `BLOCKING_NUMERIC` |

### N1（非门，仅记录）— `implausible_depth_cases`

| 项 | 值 |
|---|---|
| 阈值 | `implausible_depth_um = 650.8416 µm` = `implausible_depth_factor 10.0` × MAIN180 最大观测深度 65.08 µm |
| 实测 | **304 / 3744 = 8.12 %** 越界；`D_um_max = 134594.73 µm` |
| 性质 | **信息性标记，按 config 明文"never silently dropped, never a gate"不是门**。域角落处物理模型外推出不合理深度 |
| 分类 | `NONBLOCKING_MODEL_INADEQUACY` |

---

## 5. 是否影响 B1 / E04？**否。**

- B1 参考数据集 `20260906T164858Z_E04_B1_REFERENCE_18349780_533009c_29e663` 由 `04_b1_reference.py`
  从**合成 ODE 基准** `b1_rhs` 生成，与 E03 物理校准参数**完全无关**。
- `gate_G1_B1_REF.json`：`status=PASS`，96 项参考检查，`max_relative_error = 5.10e-09`，
  `dataset_sha256 = f9950746…` 与 `b1_training.yaml → dataset_sha256_expected` 一致。
- E04 是独立的 hidden-state computational benchmark，E03 FAIL **不构成停止 B1 的理由**。

## 6. 是否影响真实 E06？**是（physics / hybrid），静态块不受 E03 影响。**

- `06_real_main.py --stage physics` 硬要求 `--calibration-run`（第 379 行 `require`）；
  `hybrid` 走 `run_physics`，同样需要冻结校准。
- `03_physics_calibration.py` 以 `solver_registration_gate.json`（PASS 且 `allowed_to_calibrate`）为硬入口；
  本 run `allowed_to_calibrate=false` → **E03 校准仍被冻结**。
- `--stage static` 不消费校准结果 → **E03 FAIL 在科学上不阻断 E06 static**
  （但 E06_STATIC 另有 2026-09-08 审计 hold，见 `gate_transition_decision.json`）。

## 7. 是否允许 P11 作为 numerically-valid-but-inadequate inductive skeleton？**有条件允许。**

- **允许** P11（defocus + incubation 双开）作为**归纳骨架 / 结构先验**：其 recipe coverage 12/12 案例全通过，
  参数域 3741/3744 可接受，`grid_to_direct_points` 全通过，确定性逐位一致。
- **不允许** 把 P11 当作**全域数值有效的 ground-truth 生成器**：
  - 3 例域失败中有 **2 例是 P11**，且都在 w0 = 0.874 µm；
  - `ilr_z2` 在 w = 2.0 µm 的 grid_finest 未收敛到 1 %；
  - 域角落 D 可达 1.35e5 µm（> 信息阈值 208 倍）。
- 因此：作为**归纳骨架可用**；作为**监督目标或校准正向模型**必须先用
  （a）完成全配方覆盖、（b）明确 w0=0.874 µm + defocus 子域的处置（缩域或改进求解器初始 bound）、
  （c）`ilr_z2` 收敛判定，三者齐备后才可用。

## 8. 是否需要修代码？**不需要修"错误"，但有 2 项诊断性缺口建议补。**

| # | 建议 | 是否改阈值 | 说明 |
|---|---|---|---|
| C1 | 被拒向量同时记录 `max_window_attempts_observed` | 否 | 当前只记录 `failure` 字符串，无法区分"需 5 次"与"需 16 次"，阻碍域/求解器决策 |
| C2 | 细化网格增加第三档（如 2560）作为**诊断**，判定 `ilr_z2` 是否收敛而非超差 | 否 | 用于判定是"未收敛"还是"描述子本身对网格敏感" |
| C3（可选） | 求解器初始 `bound` 由解析深度估计给定，减少扩张次数 | 否 | 属求解器改进，会改变数值结果，**必须重新注册**，不得用于让本次 run 变 PASS |

**未发现** identity / 单位 / registration / supervision-contract 错误。

## 9. 是否需要重跑？**需要。** 但理由必须是范围与协议完成，而非让结果变正。

1. 全配方覆盖：`--limit-recipes 0` → 2160 配方案例（补 F1）。
2. 处置 w0 = 0.874 µm + defocus 子域：在 C1 诊断数据出来后，二选一并写入研究设计修订——
   （a）收缩注册域并明文记录；（b）改进求解器初始 bound 后**重新注册**。
3. `ilr_z2` 网格收敛判定（C2 诊断 + 决策：加细网格 / 标记该描述子不参与 1 % 门）。
4. 重跑前**不得**改动 `tail_relative_tolerance`、`max_window_attempts_allowed`、
   `implausible_depth_factor` 或任何物理参数边界。

## 10. 重跑理由声明

> 重跑（如需）的合法理由仅为：**范围未完成（F1）**、**域/求解器口径决策（F2）**、
> **离散化收敛判定（F3）**、以及**诊断性代码缺口（C1/C2）**。
> **明确禁止**以"使 gate 变 PASS"为目的调整阈值、边界或描述子集合。
> 若三项 remaining 在补齐范围与诊断后仍为 FAIL，则结论为**科学负结果并保留**。

---

## 结论

```text
status = FAIL_FORMAL_RETAIN
```

旧 run 有效、完整、可复现；其 FAIL 是在**未改动的预注册阈值**下得到的**科学/工程负结果**。
它**冻结 E03 校准 → E06 physics/hybrid → gradient design**，但**不冻结 E04 / B1**。
