#!/usr/bin/env python3
"""Quantify how much confocal conical artifacts bias the observation operator.

For each of the 15 pilot groups we run the exact extract_one pipeline twice:
  * repaired : cone_repair_enabled = True  (current / fixed observation)
  * raw     : cone_repair_enabled = False  (cones left in the point cloud)
Everything else (reference plane, threshold, profile extraction, width/depth
estimators) is identical, so the delta isolates the effect of cone removal on
the patent observation operator Y_line = [W_line, D_line].

Output: outputs/cone_repair_impact/
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from extract_zro2_single_line import CagReader, Config, extract_one

CAG = Path(r"C:\Users\RZF\Desktop\专利\氧化锆\120组直线.cag")
DESIGN = Path(r"C:\Users\RZF\Desktop\专利\氧化锆\氧化锆_line_design.csv")
OUT = Path(r"C:\Users\RZF\Desktop\专利\outputs\cone_repair_impact")
OUT.mkdir(parents=True, exist_ok=True)

GROUPS = [13, 19, 33, 34, 43, 44, 48, 51, 60, 68, 94, 95, 101, 104, 116]
design = pd.read_csv(DESIGN, encoding="gbk")
design_map = {int(r["加工顺序"]): r for _, r in design.iterrows()}

cfg_rep = Config()                 # cone repair ON
cfg_raw = Config(cone_repair_enabled=False)  # cone repair OFF


def grab(res):
    r = res[0]
    return r


rows = []
with CagReader(CAG) as reader:
    for g in GROUPS:
        raw = reader.read_group(g)
        drow = design_map.get(g)
        if drow is None:
            print("WARN no design for", g); continue
        rr = grab(extract_one(raw, drow, cfg_rep))
        rn = grab(extract_one(raw, drow, cfg_raw))
        w_rep = rr.get("W_line_um"); d_rep = rr.get("D_line_um")
        w_raw = rn.get("W_line_um"); d_raw = rn.get("D_line_um")
        sw_rep = rr.get("sigma_W_um"); sd_rep = rr.get("sigma_D_um")
        sw_raw = rn.get("sigma_W_um"); sd_raw = rn.get("sigma_D_um")
        dm_rep = rr.get("D_max_um"); dm_raw = rn.get("D_max_um")
        both_ok = all(np.isfinite(v) for v in [w_rep, d_rep, w_raw, d_raw])
        dd = (d_raw - d_rep) if both_ok else float("nan")
        dw = (w_raw - w_rep) if both_ok else float("nan")
        rel_d = (dd / d_rep) if (both_ok and d_rep not in (0, None) and np.isfinite(d_rep)) else float("nan")
        rel_w = (dw / w_rep) if (both_ok and w_rep not in (0, None) and np.isfinite(w_rep)) else float("nan")
        rows.append({
            "group": g,
            "status_rep": rr.get("status"), "status_raw": rn.get("status"),
            "mode_rep": rr.get("processing_mode"), "mode_raw": rn.get("processing_mode"),
            "W_line_repaired_um": w_rep, "W_line_raw_um": w_raw,
            "D_line_repaired_um": d_rep, "D_line_raw_um": d_raw,
            "sigma_W_rep": sw_rep, "sigma_W_raw": sw_raw,
            "sigma_D_rep": sd_rep, "sigma_D_raw": sd_raw,
            "D_max_rep_um": dm_rep, "D_max_raw_um": dm_raw,
            "delta_D_um": round(dd, 4) if both_ok else float("nan"),
            "delta_W_um": round(dw, 4) if both_ok else float("nan"),
            "rel_D_pct": round(100 * rel_d, 2) if np.isfinite(rel_d) else float("nan"),
            "rel_W_pct": round(100 * rel_w, 2) if np.isfinite(rel_w) else float("nan"),
            "N_cones_repaired": rr.get("N_conical_artifacts_repaired"),
            "max_cone_correction_um": rr.get("max_conical_correction_um"),
        })
        print(f"group {g}: D_rep={d_rep} D_raw={d_raw} dD={dd:+.3f}um "
              f"({100*rel_d:+.1f}%) | W_rep={w_rep} W_raw={w_raw} dW={dw:+.3f}um")

df = pd.DataFrame(rows).sort_values("group")
df.to_csv(OUT / "raw_vs_repaired_comparison.csv", index=False, encoding="utf-8-sig")

# ---------------- aggregates ----------------
valid = df[df["rel_D_pct"].notna() & df["rel_W_pct"].notna()]
n = len(valid)
mean_abs_rel_d = valid["rel_D_pct"].abs().mean()
mean_rel_d = valid["rel_D_pct"].mean()
max_abs_rel_d = valid["rel_D_pct"].abs().max()
mean_abs_rel_w = valid["rel_W_pct"].abs().mean()
max_abs_rel_w = valid["rel_W_pct"].abs().max()
n_d_gt5 = int((valid["rel_D_pct"].abs() > 5).sum())
n_d_gt10 = int((valid["rel_D_pct"].abs() > 10).sum())
n_w_gt5 = int((valid["rel_W_pct"].abs() > 5).sum())

# ---------------- figure ----------------
fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
g = df["group"].values
# D scatter
ax[0, 0].scatter(df["D_line_raw_um"], df["D_line_repaired_um"], c="#d95f0e", s=45, zorder=3)
lim = [np.nanmin(df[["D_line_raw_um", "D_line_repaired_um"]].values),
       np.nanmax(df[["D_line_raw_um", "D_line_repaired_um"]].values)]
ax[0, 0].plot(lim, lim, "k--", lw=1, label="y=x")
ax[0, 0].set_xlabel("D_line raw (um)"); ax[0, 0].set_ylabel("D_line repaired (um)")
ax[0, 0].set_title("Line depth: raw vs repaired\n(points below diagonal = cones deepened depth)")
ax[0, 0].legend()
# W scatter
ax[0, 1].scatter(df["W_line_raw_um"], df["W_line_repaired_um"], c="#2c7fb8", s=45, zorder=3)
limw = [np.nanmin(df[["W_line_raw_um", "W_line_repaired_um"]].values),
        np.nanmax(df[["W_line_raw_um", "W_line_repaired_um"]].values)]
ax[0, 1].plot(limw, limw, "k--", lw=1, label="y=x")
ax[0, 1].set_xlabel("W_line raw (um)"); ax[0, 1].set_ylabel("W_line repaired (um)")
ax[0, 1].set_title("Line width: raw vs repaired")
ax[0, 1].legend()
# delta D bar
ax[0, 2].bar(g, df["delta_D_um"].fillna(0).values,
              color=np.where(df["delta_D_um"].fillna(0) >= 0, "#d95f0e", "#2c7fb8"))
ax[0, 2].axhline(0, color="k", lw=0.8)
ax[0, 2].set_title("Depth bias from cones (raw - repaired, um)")
ax[0, 2].set_xlabel("group")
# delta W bar
ax[1, 0].bar(g, df["delta_W_um"].fillna(0).values,
              color=np.where(df["delta_W_um"].fillna(0) >= 0, "#756bb1", "#41b6c4"))
ax[1, 0].axhline(0, color="k", lw=0.8)
ax[1, 0].set_title("Width bias from cones (raw - repaired, um)")
ax[1, 0].set_xlabel("group")
# rel D hist
ax[1, 1].hist(valid["rel_D_pct"].values, bins=12, color="#d95f0e")
ax[1, 1].axvline(0, color="k", lw=0.8)
ax[1, 1].set_title("Relative depth bias distribution (%)")
ax[1, 1].set_xlabel("(D_raw - D_repaired)/D_repaired * 100")
# text box
ax[1, 2].axis("off")
txt = (f"Groups compared: {n}/{len(df)}\n"
       f"Mean |rel depth bias|: {mean_abs_rel_d:.2f}%\n"
       f"Mean signed depth bias: {mean_rel_d:+.2f}%  (positive = cones deepen)\n"
       f"Max |rel depth bias|: {max_abs_rel_d:.2f}%\n"
       f"Groups |rel D|>5%: {n_d_gt5}/{n}\n"
       f"Groups |rel D|>10%: {n_d_gt10}/{n}\n"
       f"Mean |rel width bias|: {mean_abs_rel_w:.2f}%\n"
       f"Max |rel width bias|: {max_abs_rel_w:.2f}%\n"
       f"Groups |rel W|>5%: {n_w_gt5}/{n}")
ax[1, 2].text(0.02, 0.98, txt, va="top", ha="left", family="monospace", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "cone_repair_impact.png", dpi=130)
print("figure saved")

md = f"""# 圆锥伪影对观测算子的偏差量化（raw vs repaired）

## 方法
同一组跑两遍 `extract_one`：开启圆锥修复（repaired）与关闭修复（raw，圆锥留在点云里）。
其余管线（参考平面、阈值、剖面提取、线宽/线深估计）完全一致，故差值即圆锥去除对观测算子
Y_line=[W_line, D_line] 的净影响。

## 聚合（{n} 组可比）
- 线深相对偏差：均值 |rel| = {mean_abs_rel_d:.2f}%，有符号均值 = {mean_rel_d:+.2f}%（正=圆锥使线深变深）
- 最大 |rel 线深| = {max_abs_rel_d:.2f}%；|rel|>5% 的组 {n_d_gt5}/{n}，>10% 的组 {n_d_gt10}/{n}
- 线宽相对偏差：均值 |rel| = {mean_abs_rel_w:.2f}%，最大 |rel| = {max_abs_rel_w:.2f}%；|rel|>5% 的组 {n_w_gt5}/{n}

## 解读
- 圆锥主要表现为**向下尖刺**，会把局部深度人为加深，因此 raw 的 D_line 普遍 >= repaired（点散在对角线下方）。
- 线宽受影响通常更小（圆锥窄、不改变横向跨越阈值宽度），但个别组仍有变化，需结合图核对。
- 偏差量级决定"去除圆锥"对专利观测算子的必要性；若多数组 |rel D| 较小则圆锥属次要噪声，
  若多个组 >5–10% 则是观测算子必须的前置清洗步骤。

## 文件
- raw_vs_repaired_comparison.csv：每组 raw / repaired 的 W_line、D_line、sigma、D_max 及差值与相对偏差
- cone_repair_impact.png：散点 + 偏差柱状 + 分布
"""
(OUT / "README_cone_impact.md").write_text(md, encoding="utf-8")
print("done")
