# Phase 2A gate answers

> **状态:CLOSED (AI-assisted visual audit, user-accepted) — 2026-09-03**
> Provenance(细则 §16 修订条款):盲评/揭盲两轮视觉审计由 AI reviewer **GPT-5.6 Sol** 执行(AUDIT-01..28,盲评页 `outputs/phase2/instability/round1/`,揭盲页 `round2/`);原始审计记录归档于 `outputs/phase2/instability/盲评/`(instability_manual_review_completed.csv、phase2A_gate_answers_final.md)。用户于 2026-09-03 接受该审计结论并指示进入 Phase 2B。如需严格人工口径,可对 round1 盲评页重新审计,结论以重审为准。
> 本文件为 canonical gate 记录;03 重跑不会覆盖(存在即保留)。自动证据由 03 填数;四条结论转录自归档文件。

## Q1 高 leverage 是否主要由 artifact 驱动?

- 自动证据: repair>0 样本 52/200; plane_rmse 中位 0.258 um; repair 最大连通域中位 0px。
- 审计计数: unblind artifact flag yes=3 / uncertain=9 / no=16;repair 区与极端结构空间重合的明确 artifact-sensitive 高杠杆样本: #37(AUDIT-08)、#82(AUDIT-17)、#149(AUDIT-13,plane RMSE 0.78 um 且主结构贴边)。
- 非 repair 高杠杆反例: #65(AUDIT-04,global LOCO rank 1,repair≈0.19%)、#152(AUDIT-24,rank≈5,repair=0)、#23(AUDIT-16,rank≈7)、#167(AUDIT-14,rank≈9)。
- **结论(转录):否,不是"主要由 artifact 驱动";但存在不能忽略的 artifact-sensitive 子集。Gate = PASS_WITH_FLAGS:Phase 2B 不暂停,但 09 必须新增 exclude-artifact-yes 臂;uncertain 样本不删除。**

## Q2 高 leverage 是否集中在某类真实形貌结构?

- 自动证据: 盲评 morphology_pattern 计数(多标签): low-frequency waviness 18、anisotropic texture 11、edge contamination 9、localized collapse 8、multi-lobe 7、periodic stripe 6、large pit 5、repair-driven 3、large-area dropout 2。
- 两类主要家族:(1) 长波/大尺度强形貌(global LOCO 前列);(2) 规则方向性周期条纹(DCT 8–16 scale-specific leverage,global LOCO 可很低)——global leverage 与 scale-specific leverage 不是同一问题。
- **结论(转录):是,存在清楚的形貌家族倾向,但不足以称离散 processing regime。Gate = YES_AS_MORPHOLOGY_FAMILIES:08 local probe 触发;命名保持数据驱动(R1/R2…),不赋物理机制。**

## Q3 高 leverage 是否只是连续幅度尾部?

- 自动证据: Spearman(loco_total_pc1, Sq) = 0.627;Spearman(loco_total_pc1, peak_to_valley) = 0.621。
- 审计关键反例: #152 global LOCO≈rank5 而 Sq≈0.75 um(repair=0);#110/#199 大坑显眼但 global LOCO 不高;周期条纹样本可 8–16 高杠杆而 total LOCO 低。
- **结论(转录):否。幅度尾部是重要因素但非充分条件;空间组织与尺度组成同样重要。Phase 2B 必须按 spatial scale 分解 target。**

## Q4 process-near / morphology-far(Type II)是否真实存在?

- 自动证据(total 口径):
  - [raw] T_lambda=1.040 um (within-null p=1.000, global-null p=1.000); TypeII=115 (within p=0.992, global p=0.991); formal-only TypeII=81 (p=0.945)
  - [phys] T_lambda=1.408 um (within p=0.074, global p=0.218); TypeII=146 (within p=0.718, global p=0.825); formal-only TypeII=58 (p=0.750)
  - 全部 12 个 (space x metric) 组合的 within-null p > 0.17(最小 phys+desc 0.173);formal-only 全不显著(最小 p=0.394)。
  - 唯一过常规线的单点: phys × DCT 16–32 的 T_lambda p=0.010,但 global-null p=0.130、formal-only p=0.166,不跨 null 复现 → 只登记为探索性局部信号,不写 branching。R 集合为有损压缩((N,h)→N/(vh)),可能人为制造 process-near,该信号需更加谨慎。
  - 阈值扰动(P5/P95、P10/P90、P15/P85,within-null):Type II p 仍 ≥0.21(见同目录 threshold_perturbation.csv)。
- **结论(转录):Type II 个案存在,但总体上没有超过置换 null 的统计过量 → Gate 4 = NOT SUPPORTED AS A POPULATION-LEVEL EXCESS;不触发 hidden-variable / stochastic-branching 优先路线;也不等价于证明 deterministic。**
