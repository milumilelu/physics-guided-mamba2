# Phase 1.5 执行细则（残差不稳定性来源分解）

> ## 1.5R.2 修订（当前生效版本，Phase 1.5 收尾）
>
> 1. **turning cosine 推断修正**：`pass_step_stats.csv` 保留
>    **bootstrap-median 分布**的五分位数，并新增逐 pass 独立置换的
>    shared-middle null（Q2.5/Q50/Q97.5 与 lower-tail rank）。相邻差分共享中间
>    表面，负 cosine 不能以 0 为科学空假设；panel B 用列名取 Q50，并同时画
>    shared-middle null 的 95% 区间。
> 2. **mixed-session 基线 cache key**：加入 per-session occupancy
>    composition（`comp_key`），总 occupancy 相同但 session 构成不同的子集
>    不再共享基线；`baseline_matched.csv` 记录逐 session 签名。
> 3. **stable_call 规则**：matched random subset 每次分别保存 bootstrap angle
>    的 stat_Q50 与 stat_Q90，形成两个独立 null；`robust_stable` 要求两者的
>    empirical rank 均 ≥0.95 且 LOCO max<45°。该标签仅为 exploratory screen；其余同前
>    （fragile_stable / not_called）。

> ## 1.5R.1 修订（当前生效版本）
>
> 1. **pass 图索引修复**：`pass_scale_evolution.png` panel A 曾误用 Q25 当
>    Q50 线、step 列当 Q25；panel B 曾把 q95 当 cos12 且漏画 cos23——已修正，
>    cos 两段折线分开显示。
> 2. **描述符命名/单位**：`kurtosis_excess_fisher`（无单位，Fisher 定义）、
>    `lap_rms_um_per_px2`（scipy laplace 为逐像素二阶差分）、
>    `acf_e_fold_lag_um`（零延迟截线 ACF 的 1/e 折点，x/y 均值）、
>    `pit_density_per_Mpx`（每百万像素负离群点数，阈值 med−3.5×1.4826·MAD）、
>    `Ssk_skewness`。
> 3. **49/50 分类撤销**：variability/repeatability 汇总只报数值与分位，
>    不再给 high/medium/low 标签。
> 4. **conditional 判据**：新增 `stable_call` 列——`prank1>=0.95 且
>    theta1_Q50<rand_Q25` 之外，还须 `theta1_Q90<rand_Q75 且
>    loco_max<45deg` 才记 `robust_stable`，否则 `fragile_stable`；
>    基线分布同时报 Q25/Q50/Q75/Q90。
> 5. **LOCO 明细**：`loco_top5_influencers.csv`（每 subset x scale 的 top-5
>    cluster_id、角度、成员 sample/加工条件/深度）与
>    `loco_top5_montage.png`（total 残差 PC1，每子集 5 格）。
> 6. **shuffled-depth null**：10→200 次（`conditional.shuffle_null_perms`），
>    逐次结果存 `depth_window_null.csv`。
> 7. **turning cosine CI**：`pass_step_stats.csv` 增 cos(step1,2) 与
>    cos(step2,3) 的 trajectory-bootstrap Q25/50/75/90/95。
> 8. **depth-Q 基线**：跨 session 子集改 `session_composition_matched`
>    （按子集内各 session 的 ROI 数与 within-session occupancy 分别抽取）。

> ## 1.5R 修订（当前生效版本）
>
> 1. **尺度命名**：废除 `<2 / 2–8 / >8 µm` 三带标签；改用滤波器命名 `G2/G4/G8/G16`
>   （σ 单位 px），并输出 Gaussian 传递函数的 **-3dB 波长**
>    （解析式 λ=2πσ/√ln2 ≈ 7.546σ px，附离散核 DFT 数值核对，写入
>    `scale_energy_table.csv`）。另加一组按**物理波长**定义的 DCT 带敏感性
>    （λ∈[8,16)/[16,32)/[32,64)/[64,∞) µm，DCT-II 掩码，报告覆盖率）。
> 2. **bootstrap**：B=1000（quick 50）；预生成**同一份 cluster resample bank**
>    跨所有尺度场复用；CSV 保存 **Q25/Q50/Q75/Q90/Q95**（分布不对称，不再用
>    median±IQR/2 画带，改画 Q25–Q75 分位带）。
> 3. **conditional**：基线改为**同 session（跨 session 子集用全局池）+ 同 ROI 数
>    + 同 within-subset cluster-size pattern**；新增 **leave-one-cluster-out
>    influence** 与 **eigengap（λ1/λ2）**；对 N1/N2、depth Q1 这类
>    "Q50≈1°、分布极宽"的子集做专门报告。
> 4. **depth-window**：PC1–3 相邻窗夹角取**最大主角**（sv3[-1]，修掉原来的
>    sv3[0]）；新增 **non-overlap 窗**（step=窗宽）与 **shuffled-depth
>    overlap-null**（打乱深度序、保持同窗结构，量化重叠样本造成的 cos 虚高）。
> 5. **deterministic–stochastic map 撤销**：S(q) 分数删除；保留跨样本 SD/IQR
>    与 49/50 delta+分位，改名 **variability/repeatability summary**。
> 6. **pass 表**：两段相邻步 turning cosine（step1vs2 / step2vs3）分开报告；
>    对 15 个 base conditions 做 **trajectory-level bootstrap**（B=1000）；
>    全部图表标注 **pseudo-trajectory / cross-sectional**。
> 7. **测试**：新增 `tests/test_phase1_5_lib.py`（16 项合成数据单测）与
>    `.github/workflows/tests.yml`（push/PR 即跑 unittest，给出 CI status）。
>
> 本轮只修以上各项，不重新设计项目；下文为 1.5 初版方法记录，与 1.5R 冲突处以本节为准。

对应说明：`任务说明/Phase1.5说明`（用户提供的 Phase 1.5 说明）。
Phase 1.5 只回答一个问题：**为什么去掉深度后的形貌结构在全局 PCA 中不稳定？**
仍然不做预测、不做神经网络/autoencoder/Mamba/SINDy，全部是低模型假设的诊断。

---

## 0. 仓库状态核查（执行前只读核查结果）

| 事实 | 状态 |
|---|---|
| Phase 1 冻结产物 `outputs/phase1_minimal/exploration_manifest.csv` | 存在，200 行 / 23 列，含 `session_role, design_group, median_depth_um, residual_Sq_um, pulse_duration_fs` 等 |
| `stable_roi_80um_dataset.npz` | 200×160×160，全部像素有效（Phase 1 已验证 valid_fraction≡1.0） |
| Phase 1 关键数字（本阶段对照基准） | absolute PC1 EVR=98.77%；residual PC1 EVR=25.12%；θ_boot(k=1)≈31°，θ_boot(k=6)≈71°；49/50 depth 差 0.0272 µm |
| residual 定义 | R = H − per-sample valid-median（与 Phase 1 完全一致，本阶段从 NPZ 重算，不读取 Phase 1 中间数组） |
| session 构成 | formal 120（frequency/pulse/pass 各 5 水平 × 24）、60-pass 60（15 组 × N{1..4}）、supplement 20（10 组 × N{5,6}） |
| bootstrap cluster | 160 个 `shared_height_source_id`（120 单 + 40 双），沿用 |

---

## 1. 四个假设 → 实验映射

| 假设 | 检验实验 |
|---|---|
| H1 regime mixing（工艺区间混杂） | 03 conditional PCA + 样本量匹配基线 |
| H2 多空间尺度混合 | 01 尺度分解 + 02 分尺度 PCA/bootstrap |
| H3 nonlinear manifold（模态随深度旋转） | 03 深度滑窗 PCA 相邻窗主角 |
| H4 真实 stochasticity / hidden variables | 04 工艺距离 vs 形貌距离 + 分尺度 sentinel、05 步进方向余弦 + 描述符映射表 |

---

## 2. 新增文件（其余一律只读）

```text
experiments/phase1_5/
  Phase1.5_本细则.md            # 本文件（决策规则在分析前写定）
  phase1_5_config.yaml
  _lib.py                       # 共享：冻结输入加载、尺度分解、快速 cluster-bootstrap PCA
  01_scale_decomposition.py
  02_scale_pca_stability.py
  03_conditional_pca.py
  04_pairwise_repeatability.py
  05_pass_scale_evolution.py
outputs/phase1_5/               # 全部输出
```

随机种子 20260902（Phase 1 为 20260901，登记区分）。

---

## 3. 尺度分解定义

对每个样本的残差 R_i(x,y)（µm），高斯滤波 `scipy.ndimage.gaussian_filter(mode="reflect")`（**σ_low > σ_high**，保证 R_mid 是真带通、三个分量方差近似分割）：

```text
R_low  = G_{σ_low} * R          （低通：σ_low = 16 px = 8 µm，槽底弯曲/波纹）
R_high = R − G_{σ_high} * R     （高通：σ_high = 4 px = 2 µm，细粗糙度）
R_mid  = R − R_low − R_high = G_{σ_high}*R − G_{σ_low}*R   （2–8 µm 带通，扫描纹理）
```

- 三个分量精确相加还原 R（运行时断言 < 1e-8）；方差占比之和 ≈ 1（断言 0.85–1.15，高斯带非严格正交）。
- 依据：hatch spacing 2–10 µm → 扫描纹理主要落入 mid；槽底弯曲/波纹 > 8 µm → low；像素级粗糙度/崩坑边缘 < 2 µm → high。**不预设哪个尺度对应哪个机制**，另做 σ ∈ {2,4,8,16} px 的 sweep（对每个 σ 的低通场 G_σ*R 重复 PCA + bootstrap）。
- 数据全有效；若未来出现 NaN，先按 per-sample median 填充再滤波（防御分支，记录日志）。

---

## 4. 各脚本规格与输出

### 01_scale_decomposition.py
- 方差分割表：各 band / 各 σ 低通场的方差占比（跨样本均值/中位/IQR）→ `scale_energy_table.csv`。
- 代表样本图 `scale_decomposition_examples.png`：按深度取最浅/中位/最深 3 个样本 × 5 列（H, R, R_low, R_mid, R_high）。

### 02_scale_pca_stability.py（核心）
- 8 个场（total/low/mid/high + 4 个 σ 低通场）各自：gram PCA k=1..10 EVR；cluster bootstrap B=200（各场用同一种子 → 同一组重采样抽签，跨场可比）→ k=1..6 最大主角 median/IQR。
- total 场必须复现 Phase 1：θ(k=1)≈31°、θ(k=6)≈71°（数量级校验，打印）。
- 输出：`scale_evr_curves.png`（图2）、`scale_bootstrap_angles.png`（图3，**最重要**）、`scale_pca_bootstrap.csv`。

### 03_conditional_pca.py
- 条件子集（残差与各 band 分别做）：
  - formal 内 frequency_kHz ∈ {2,10,50,100,200}、pulse_duration_fs ∈ {500,1000,2000,4000,6000}、pass_count ∈ {1..5}（各 n=24）；
  - 全体 200 按深度四分位 Q1–Q4（n=50，注明 session 混杂警告）；
  - 60-pass 内 N ∈ {1..4}（n=15，样本量警告）。
- 每个子集 × band：EVR PC1、cum3；**子集内部** cluster bootstrap B=1000 → θ(k=1)、θ(k=1..3) 分位数。
- **样本量与 session composition 匹配基线**（§5）：同一 bootstrap 协议分别形成 stat_Q50 与 stat_Q90 的对照分布；分别报告 empirical rank。
- 深度滑窗（H3）：按深度排序，窗 50 步 10 → 16 窗，相邻窗 PC1 与 PC1–3 子空间夹角 vs 窗中心深度。
- 输出：`conditional_pca_table.csv`、`baseline_matched.csv`、`conditional_stability_heatmap.png`（图4）、`depth_window_table.csv`、`depth_window_mode_rotation.png`。

### 04_pairwise_repeatability.py
- 每 band 全对形貌距离 D_band（Gram 法，µm RMSE）；工艺距离 D_process：5 参数 z-score 欧氏（pulse/frequency/hatch/velocity/pass）。
- Spearman ρ(D_process, D_band)（仅 ordinary 对：剔除 40 个共享来源对与 sentinel 对）；D_process 前 10% 的"工艺邻近对" D_band 分布 vs 全体。
- session 可分性（§12）：D_within vs D_between 中位数与比值，按 band。
- 49/50 分尺度 sentinel：D_band(49,50) 及其在 ordinary 对分布中的分位（§8）。
- 输出：`pairwise_distance_summary.csv`、`session_separability.csv`、`sentinel_multiscale_table.csv`、`morph_vs_process_distance.png`（图5）、`sentinel_multiscale_49_50.png`（图6）。

### 05_pass_scale_evolution.py
- 描述符（§10，逐样本）：Sq、Sa、Ssk、Fisher Sku、梯度 RMS、Laplacian RMS、自相关 1/e 相关长度、x-y 各向异性（grad_x/grad_y RMS 比）、三 band 能量分数、坑密度（R < med − 3.5×1.4826·MAD 像素数/Mpx）、最深负残差。→ `morphology_descriptors.csv`。
- pass 演化（§9）：15 条 cross-sectional pseudo-trajectories × band：步长 RMS ‖ΔR_N‖、相邻步方向 cos θ_N、Δd_N。turning cosine 另与逐 pass 独立置换、但保留共享中间项的 null 比较 → `pass_step_stats.csv`、`pass_scale_evolution.png`（图7）。
- deterministic–stochastic map（§11）：对每个量 q：
  - V(q) = 跨样本 SD（附 IQR）；
  - S(q) = 1 − SE_boot(median)/SD(q)（cluster bootstrap B=200，截断 [0,1]）；
  - R(q) = 49/50 |Δq| 在 ordinary 对 |Δq| 分布中的分位（越小越可复现）。
  → `deterministic_stochastic_map.csv` + 表格图 `deterministic_stochastic_map.png`（第 8 张，超出 7 张核心图，已登记）。

---

## 5. 预注册决策规则（分析前写定，不事后修改）

### 5.1 描述性基准（非通过/失败阈值）

- **稳定**：某场/子集 θ_boot(k=1) median < 20°；**不稳定**：> 40°；20°–40° 记为中间。
- 条件子集稳定性筛查：stat_Q50 与 stat_Q90 相对各自 matched null 的 empirical rank 均 ≥0.95，且 LOCO max <45°；仅作探索性标签。
- pass turning 只有在 shared-middle permutation lower-tail rank ≤0.05 时，才记为超出相邻差分代数耦合的额外负方向证据；仅仅小于 0 不作 reversal 证据。
- sentinel"高可复现"：分位 ≤ 5%；"低"：≥ 20%。
- 滑窗"平滑旋转"：相邻窗 PC1 |cos| 中位数 ≥ 0.8 且随深度无跳变；"无组织"：中位数 < 0.5。

### 5.2 Phase 2 路线映射（说明 §16 原表）

| Phase 1.5 结果 | Phase 2 倾向 |
|---|---|
| low 尺度稳定、high 尺度不稳定（θ_low ≪ θ_high） | deterministic/stochastic decomposition |
| 某条件子集明显比 matched 基线稳定 | regime-specific modeling |
| 模态随深度平滑旋转（滑窗 |cos| 高） | nonlinear manifold / local linear |
| 工艺邻近对形貌也近（ρ 高、near 对分布窄） | process→morphology modeling |
| 工艺邻近但形貌散 | hidden-state / stochastic study |
| pass 低尺度步进方向一致（cos≈1） | reduced-order pass dynamics |
| session 间 D_between ≫ D_within | 先做 domain/session 校正 |

---

## 6. 运行

```powershell
.\.venv\Scripts\python.exe experiments\phase1_5\01_scale_decomposition.py --quick   # 冒烟
.\.venv\Scripts\python.exe experiments\phase1_5\01_scale_decomposition.py           # 正式
# 依次 02 03 04 05
```

`--quick` 仅降 B（10）与滑窗密度，用于冒烟；正式数字一律以无 flag 运行为准。

## 7. 边界重申

不做预测/NN/autoencoder/Mamba/SINDy；不做事后阈值筛选；PCA 维数不解释为物理维数；
49/50 只有一对，永远只叫 repeatability sentinel；深度四分位与 session 混杂处必须带警告陈述；
pass 数据永远只叫 cross-sectional pseudo-trajectories，不作同槽 dynamics 或 oscillation 解释。

---

## 8. Phase 1.5 冻结结论

当前结果支持三项描述性结论：

1. absolute height 的主变化与加工深度相关，去除深度后的 residual morphology 不是同一个一维问题；
2. residual variance 主要分布在 ≥8 µm 的多个空间波段，但空间平滑不等于样本维度低；
3. 少数表面对全局/条件 PCA 有很高杠杆，因此总体分布存在异质性。

当前结果不单独支持：离散加工 regime、同一工艺下空间相位不唯一、稳定的 pass reversal，
或同一位置的逐 pass dynamics。这些均保留为后续审计或补重复实验的候选假设。
