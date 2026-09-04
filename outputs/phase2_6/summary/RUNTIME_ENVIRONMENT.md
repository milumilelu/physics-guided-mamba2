# Phase 2.6 复现环境与运行说明（审查者必读）

> 目的：Phase 2.6 formal 期间发生过一次**静默的数值环境分歧**——同一代码在两个 Python 下给出不同的 M0_RECON 结果（α 选择翻转）。本文件登记运行环境约定与全部复现命令，审查复现时必须遵守。

## 1. 运行环境（强制）

- **Python 解释器：仓库自带 `.venv`**（`.venv/Scripts/python.exe`），依赖以 `requirements.txt` 为准：numpy 2.3.5 / pandas 3.0.1 / scikit-learn **1.7.2** / scipy 1.18.1 / matplotlib 3.10.7。
- **禁止**用系统/conda 解释器复现：实测 `D:\anaconda`（Python 3.11.7，sklearn 版本不同）下 Ridge 内层验证 R² 有微小数值差异，导致 `p25_select_alpha` 的 argmax 在 α∈{0.01, 1, 10} 之间翻转，M0_RECON 的 Q² 中位偏移 0.031（> 0.005 容差）——即 gate 评估文档 §0 记录的"对账失败根因"。
- 参考值锚点（M0_RECON，ridge / input A / src_gkf，全 200）：
  - `ilr_z1_z4` Q2_Aitchison 中位 = **0.312690**（逐折 [0.3744, 0.3127, 0.3047, 0.3743, 0.2876]，α = [1,1,1,0.01,0.01]）
  - `A2_8_16` R² 中位 = **0.509533**
  - `angular_entropy_8_16` R² 中位 = **0.607148**
  复现偏差 ≤ 0.005 视为环境正确；更大偏差先查解释器。

## 2. formal 运行链（提交链对应）

```powershell
.\.venv\Scripts\python.exe experiments\phase2_6\17_line_width_process_model.py      # G-SL1（需 Task 16 输出 + QA 标注完成）
.\.venv\Scripts\python.exe experiments\phase2_6\18_scale_bridge_model_compare.py    # M0_RECON + direct bridge + M0–M_GEO（G-SL3）
.\.venv\Scripts\python.exe experiments\phase2_6\19_lambda_ratio_test.py             # λ 比值检验 + shuffled-h null（G-SL2）
.\.venv\Scripts\python.exe experiments\phase2_6\20_orientation_provenance_check.py  # G-SL4（NA 路径）
.\.venv\Scripts\python.exe -m unittest tests.test_phase2_6_lib                      # 27 项单测
```

Task 15/16（manifest 与几何提取）见 `Phase2.6_落地执行细则.md` §12；`--quick` 只改输出根（`outputs/phase2_6_quick/`），输入一律读 formal 冻结产物。

## 3. 关键复现锚点（审查抽查用）

| 断言 | 冻结数值 | 文件 |
|---|---|---|
| 单线 DOE 盒覆盖 | exact_match 20 样本 / 19 条件；in_box_pred 81；out_of_box 99 | `scale_bridge/morphology_scale_match.csv` |
| direct bridge 可用性 | estimable 13 / W_unavailable 5 / rejected_by_qa 1（缺失率 26%） | `scale_bridge/direct_bridge_exact_match.csv` |
| G-SL1 | pooled W50 median 5.78 µm；线级带内 1.2%（1/81）；W_eq 5.86 µm → NOT_SUPPORTED 0/3 | `summary/gsl1_evaluation.json` |
| G-SL2 | A_obs 0.904（valid 104/200）vs null 中位 0.471，p = 0.0001 → SUPPORTED | `summary/gsl2_evaluation.json` |
| G-SL3 | composition retention 0.592（proc 0.490）→ NOT_SUPPORTED | `summary/gsl3_evaluation.json` |
| QA 标注 | 120/120：usable 18 / uncertain 78 / reject 24（annotator=GPT，盲态保持） | `single_line/geometry_qa_labels*.csv/json` |

QA montage 原图（120 张，盲评输入）在 `single_line/qa_montages/`（本地，不入库）与已提交的 `单线QA标注包_20260904*.zip` 内；blind 审查（T22：montage 无任何 8/16 µm 信息）可据 zip 内原图复核。
