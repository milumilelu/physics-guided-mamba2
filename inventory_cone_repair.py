#!/usr/bin/env python3
"""Inventory the conical-dropout repair across the 15 pilot CAG groups.

Reuses the exact repair segment used by extract_one (locate_cut_corridor ->
repair_conical_dropouts with the cut corridor as allowed_mask), then adds a
first-pass quality check:
  * per-group: how many cones, how many pixels, correction magnitudes, adaptive
    noise scale / seed / grow thresholds;
  * per-artifact: size, span, correction, spatial centroid (to flag wall-proximal
    repairs that may be real steep geometry rather than measurement cones);
  * residual strong deficits inside the cut corridor that the repair left untouched
    (candidate missed cones / correct rejections of real steep walls).

Outputs go to outputs/cone_repair_inventory/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from extract_zro2_single_line import (
    CagReader, Config, locate_cut_corridor,
    repair_conical_dropouts, mad_scale,
    _max_filter_rows, _min_filter_rows, connected_components,
)

CAG = Path(r"C:\Users\RZF\Desktop\专利\氧化锆\120组直线.cag")
DESIGN = Path(r"C:\Users\RZF\Desktop\专利\氧化锆\氧化锆_line_design.csv")
OUT = Path(r"C:\Users\RZF\Desktop\专利\outputs\cone_repair_inventory")
OUT.mkdir(parents=True, exist_ok=True)

GROUPS = [13, 19, 33, 34, 43, 44, 48, 51, 60, 68, 94, 95, 101, 104, 116]
cfg = Config()

design = pd.read_csv(DESIGN, encoding="gbk")
design_map = {int(r["加工顺序"]): r for _, r in design.iterrows()}


def residual_deficits(raw_z, valid, cut_corridor, radius, seed_thr, cols):
    closed = _max_filter_rows(_min_filter_rows(raw_z, radius), radius)
    deficit = np.maximum(closed - raw_z, 0.0)
    deficit[~valid] = 0.0
    strong = (deficit >= seed_thr) & cut_corridor
    comps = connected_components(strong)
    n_miss = 0
    pix_miss = 0
    wall_touch = 0
    for comp in comps:
        rr, cc = comp[:, 0], comp[:, 1]
        n_miss += 1
        pix_miss += len(comp)
        if int(cc.min()) == 0 or int(cc.max()) == cols - 1:
            wall_touch += 1
    return n_miss, pix_miss, wall_touch


group_rows = []
artifact_rows = []
residual_rows = []

with CagReader(CAG) as reader:
    for g in GROUPS:
        raw = reader.read_group(g)
        drow = design_map.get(g)
        raw_z = np.asarray(raw["z_um"], dtype=float)
        valid = np.asarray(raw["valid"], dtype=bool)
        dx, dy = float(raw["dx_um"]), float(raw["dy_um"])
        rows, cols = raw_z.shape
        try:
            cut_corridor, _ = locate_cut_corridor(raw_z, valid, dx, dy, cfg)
        except Exception as e:
            print(f"group {g}: locate_cut_corridor failed: {e}")
            group_rows.append({"group": g, "status": "failed_corridor",
                               "N_cones": 0, "n_pixels_repaired": 0,
                               "max_correction_um": 0.0, "mean_correction_um": 0.0,
                               "noise_2nd_diff_um": np.nan, "seed_threshold_um": np.nan,
                               "grow_threshold_um": np.nan, "frac_valid_repaired": 0.0})
            continue
        corrected, cone_mask, cone_table, cone_metrics = repair_conical_dropouts(
            raw_z, valid, cfg, allowed_mask=cut_corridor)

        n_pix = int(cone_mask.sum())
        if n_pix:
            corr = corrected[cone_mask] - raw_z[cone_mask]
            max_corr = float(corr.max())
            mean_corr = float(corr.mean())
        else:
            max_corr = 0.0
            mean_corr = 0.0
        frac = n_pix / max(1, int(valid.sum()))

        group_rows.append({
            "group": g, "status": "ok",
            "N_cones": int(len(cone_table)),
            "n_pixels_repaired": n_pix,
            "frac_valid_repaired": round(frac, 5),
            "max_correction_um": round(max_corr, 4),
            "mean_correction_um": round(mean_corr, 4),
            "noise_2nd_diff_um": round(float(cone_metrics["noise_second_difference_um"]), 5),
            "seed_threshold_um": round(float(cone_metrics["seed_threshold_um"]), 4),
            "grow_threshold_um": round(float(cone_metrics["grow_threshold_um"]), 4),
        })

        for _, a in cone_table.iterrows():
            pc = int(a["pixel_count"])
            col_span = int(a["col_max"] - a["col_min"]) + 1
            row_span = int(a["row_max"] - a["row_min"]) + 1
            row_frac = float(a["centroid_row"]) / max(1, rows - 1)
            col_frac = float(a["centroid_col"]) / max(1, cols - 1)
            edge_prox = bool(a["centroid_row"] < 2 or a["centroid_row"] > rows - 3)
            artifact_rows.append({
                "group": g, "artifact": int(a["artifact"]),
                "pixel_count": pc, "col_span_px": col_span, "row_span_px": row_span,
                "max_correction_um": round(float(a["max_correction_um"]), 4),
                "mean_correction_um": round(float(a["mean_correction_um"]), 4),
                "centroid_row_frac": round(row_frac, 3),
                "centroid_col_frac": round(col_frac, 3),
                "edge_proximal": edge_prox,
            })

        n_miss, pix_miss, wall_touch = residual_deficits(
            raw_z, valid, cut_corridor, cfg.cone_half_window_px,
            cone_metrics["seed_threshold_um"], cols)
        residual_rows.append({
            "group": g, "residual_strong_deficits": n_miss,
            "residual_pixels": pix_miss, "residual_touch_wall": wall_touch,
        })
        print(f"group {g}: cones={len(cone_table)} pixels={n_pix} "
              f"max_corr={max_corr:.3f}um residual_deficits={n_miss}")

group_df = pd.DataFrame(group_rows).sort_values("group")
art_df = pd.DataFrame(artifact_rows)
res_df = pd.DataFrame(residual_rows).sort_values("group")

# merge residual into group summary
group_df = group_df.merge(res_df, on="group", how="left")

group_df.to_csv(OUT / "cone_repair_group_summary.csv", index=False, encoding="utf-8-sig")
art_df.to_csv(OUT / "cone_repair_artifact_table.csv", index=False, encoding="utf-8-sig")

# ---------------- figure ----------------
fig, ax = plt.subplots(2, 3, figsize=(15, 9))
g = group_df["group"].values
ax[0, 0].bar(g, group_df["N_cones"].values, color="#2c7fb8")
ax[0, 0].set_title("Cones repaired per group"); ax[0, 0].set_xlabel("group"); ax[0, 0].set_ylabel("count")

if not art_df.empty:
    ax[0, 1].hist(art_df["pixel_count"].values, bins=20, color="#41b6c4")
    ax[0, 1].set_title("Cone size distribution (pixels)"); ax[0, 1].set_xlabel("pixel_count")
    ax[1, 0].hist(art_df["max_correction_um"].values, bins=20, color="#fe9929")
    ax[1, 0].set_title("Max correction per cone (um)"); ax[1, 0].set_xlabel("max_correction_um")
    edge = art_df["edge_proximal"].values.astype(bool)
    sc = ax[1, 1].scatter(art_df["centroid_col_frac"].values, art_df["centroid_row_frac"].values,
                          s=np.clip(art_df["pixel_count"].values * 1.5, 8, 120),
                          c=art_df["max_correction_um"].values, cmap="viridis", alpha=0.8)
    ax[1, 1].set_title("Cone centroid (col vs row fraction)\nsize=pixels, color=max correction")
    ax[1, 1].set_xlabel("along scan (col)"); ax[1, 1].set_ylabel("across width (row)")
    ax[1, 1].axhspan(-0.02, 0.08, color="red", alpha=0.06)
    ax[1, 1].axhspan(0.92, 1.02, color="red", alpha=0.06)
    fig.colorbar(sc, ax=ax[1, 1])
else:
    for (r, c) in [(0, 1), (1, 0), (1, 1)]:
        ax[r, c].text(0.5, 0.5, "no cones", ha="center", va="center")

ax[0, 2].bar(g, group_df["max_correction_um"].values, color="#d95f0e")
ax[0, 2].set_title("Max correction per group (um)"); ax[0, 2].set_xlabel("group")
ax[1, 2].bar(g, group_df["residual_strong_deficits"].fillna(0).values, color="#7a0177")
ax[1, 2].set_title("Residual strong deficits (unrepaired) per group")
ax[1, 2].set_xlabel("group"); ax[1, 2].set_ylabel("count (candidate misses)")

fig.tight_layout()
fig.savefig(OUT / "cone_repair_inventory.png", dpi=130)
print("figure saved")

# ---------------- summary readme ----------------
total_cones = int(art_df["pixel_count"].size) if not art_df.empty else 0
total_pixels = int(art_df["pixel_count"].sum()) if not art_df.empty else 0
total_residual = int(res_df["residual_strong_deficits"].sum()) if not res_df.empty else 0
edge_cones = int(art_df["edge_proximal"].sum()) if not art_df.empty else 0
big_corr = int((art_df["max_correction_um"] > 2.0).sum()) if not art_df.empty else 0
md = f"""# 圆锥伪影修复盘点（15 个 pilot 组）

## 方法口径
与 `extract_one` 内部一致：先 `locate_cut_corridor` 得到切割走廊，再把它作为
`allowed_mask` 传给 `repair_conical_dropouts`。Config 取默认（cone_repair_enabled=True,
half_window_px={cfg.cone_half_window_px}, seed_sigma={cfg.cone_seed_sigma},
grow_sigma={cfg.cone_grow_sigma}, min_seed_depth_um={cfg.cone_min_seed_depth_um},
max_component_span_px={cfg.cone_max_component_span_px}）。

## 总体
- 修复组数：{len(group_df)}（status=ok: {int((group_df['status']=='ok').sum())}）
- 修复锥总数：{total_cones}，修复像素总数：{total_pixels}
- 每组建模：mean N_cones = {group_df['N_cones'].mean():.1f}，max = {group_df['N_cones'].max()}，min = {group_df['N_cones'].min()}
- 最大单点修正（全局）：{group_df['max_correction_um'].max():.3f} um
- 残余强缺陷（走廊内未被修复的向下尖刺，疑似漏检或合理拒绝真实陡壁）：{total_residual} 处

## 第一遍质检（需人工目视确认）
- 贴壁锥（centroid 落在 Y 边界 2px 内，可能是真实陡壁而非测量锥）：{edge_cones} 个
- 大修正锥（max_correction > 2.0 um，需重点核对是否误修真实几何）：{big_corr} 个
- 注意：修复只向上修正（np.maximum），不会把真实沟槽往下填；列入"需核对"仅表示可疑，不等于错误。

## 文件
- cone_repair_group_summary.csv：每组统计
- cone_repair_artifact_table.csv：每个锥的尺寸/位置/修正
- cone_repair_inventory.png：汇总图
"""
(OUT / "README_cone_inventory.md").write_text(md, encoding="utf-8")
with open(OUT / "inventory_config.json", "w", encoding="utf-8") as f:
    json.dump({"groups": GROUPS, "config": {
        "cone_repair_enabled": cfg.cone_repair_enabled,
        "cone_half_window_px": cfg.cone_half_window_px,
        "cone_seed_sigma": cfg.cone_seed_sigma,
        "cone_grow_sigma": cfg.cone_grow_sigma,
        "cone_min_seed_depth_um": cfg.cone_min_seed_depth_um,
        "cone_max_component_span_px": cfg.cone_max_component_span_px,
    }}, f, ensure_ascii=False, indent=2)
print("done")
