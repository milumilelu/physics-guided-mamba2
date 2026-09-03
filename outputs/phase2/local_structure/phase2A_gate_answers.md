# Phase 2A gate answers

> 自动填数 by 03;结论行必须由 reviewer 人工填写后 gate 才算关闭(细则 §16)。

## Q1 高 leverage 是否主要由 artifact 驱动?
- 自动证据: repair>0 样本 52/200; plane_rmse 中位 0.258 um; repair 最大连通域中位 0px。
- 待两轮盲评(02 的 manual_review.csv)。
- **人工结论:【待填写】**

## Q2 高 leverage 是否集中在某类真实形貌结构?
- 自动证据: 盲评 morphology_pattern 分布(待人工)。
- **人工结论:【待填写】**

## Q3 高 leverage 是否只是连续幅度尾部?
- Spearman(loco_total_pc1, Sq) = 0.627; Spearman(loco_total_pc1, peak_to_valley) = 0.621。
- **人工结论:【待填写】**

## Q4 process-near / morphology-far(Type II)是否真实存在?

total 残差口径(primary continuous statistic T_lambda 与 Type II 计数):
- [raw] T_lambda=1.040 um (within-null p=1.000, global-null p=1.000); TypeII=115 (within-null p=0.992, global-null p=0.991); formal-only TypeII=81 (p=0.945)
- [phys] T_lambda=1.408 um (within-null p=0.074, global-null p=0.218); TypeII=146 (within-null p=0.718, global-null p=0.825); formal-only TypeII=58 (p=0.750)

- 全部 12 个 (space x metric) 组合见 neighborhood_summary.csv; Type II 行明细见 process_near_morph_far_pairs.csv。
- 判读规则: 以 within-session null 为主,global null 与 formal-only 口径方向须并列呈现;Route U 还需通过 P15/P85 扰动(09/细则 §17)。
- **人工结论:【待填写】**
