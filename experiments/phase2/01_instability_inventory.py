#!/usr/bin/env python3
"""Phase 2 experiment 01: instability inventory (Phase 2A-1).

Full-200 LOCO recomputation (PC1 and PC1-3 subspaces over 5 fields) with a
consistency sentinel against the Phase 1.5 global/total top-5 table; amplitude,
spectral, leverage, isolation and artifact feature blocks; a conservative
consensus rank whose spectral term is the MOST anomalous of the four DCT band
ranks (not the fixed >=64 um band); and the blinded selection pool for the
manual montage audit (anon codes assigned here, identity revealed only in the
round-2 unblind montage).

Deterministic (LOCO and kNN are exact); --quick runs the identical computation.

Seed offsets: none.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import _lib as p2

EXPECTED = ["instability_inventory.csv", "instability_selected.csv",
            "loco_full.csv", "README.md"]

FIELDS = ["total", "DCT_8_16", "DCT_16_32", "DCT_32_64", "DCT_64_inf"]

README = """# instability inventory (Phase 2A-1)

- `instability_inventory.csv`: per-sample audit features (identity, amplitude,
  spectral, audit-only band PC scores, leverage, isolation, artifact,
  ranks + A_consensus). `band_PC*_score_audit_*` columns are descriptive only
  and must never enter Phase 2B CV targets (细则 §0.7).
- `instability_selected.csv`: the blinded manual-audit pool (20-30 rows,
  `anon_code` round-1 identity, `selection_reason` multi-label).
- `loco_full.csv`: full LOCO table (field x {pc1, pc123} x 160 clusters);
  per-sample leverage columns inherit the cluster angle (double slots share
  the measurement, so both rows carry the same influence value by design).
- `distance_to_ROI_boundary` is NOT constructed: all 200 ROIs are the same
  central 80x80 um window (no variation; see 细则 §3.2 step 7).
"""


def main() -> int:
    cfg, quick = p2.load_config(__doc__)
    t0 = time.time()
    out = p2.output_dir(cfg, "instability")
    p2.log("== Phase 2 / 01: instability inventory ==")
    frozen = p2.l15.load_frozen(cfg)
    man = p2.build_manifest(cfg)
    desc = pd.read_csv(p2.l15.REPO / "outputs/phase1_5/morphology_descriptors.csv")
    p2.require(list(desc["dataset_index"]) == list(range(200)),
               "descriptor row order != dataset_index")
    R = frozen["R"]
    V = frozen["V"]
    Rn = np.where(V, R, np.nan)

    inv = man[["dataset_index", "session_id", "sample_id", "session_role",
               "processing_order", "design_group",
               "shared_height_source_id"]].copy()

    # ---- full LOCO recomputation (5 fields x {pc1, pc123}) -----------------
    fields = p2.l15.multiscale_fields(R, cfg)
    clusters = p2.l15.cluster_lists(man)
    cid_of_row = np.empty(200, dtype=int)
    for ci, c in enumerate(clusters):
        cid_of_row[c] = ci
    cluster_name = man["shared_height_source_id"].to_numpy()
    members_str = {ci: "|".join(str(int(x)) for x in clusters[ci])
                   for ci in range(len(clusters))}
    loco_rows = []
    per_row_angle = {}
    for fname in FIELDS:
        X = fields[fname]
        for k, tag in ((1, "pc1"), (3, "pc123")):
            ang = p2.l15.loco_angles(X, clusters, k=k)
            order = np.argsort(-ang)
            for rank1, ci in enumerate(order, start=1):
                loco_rows.append((fname, k, int(rank1),
                                  str(cluster_name[clusters[ci][0]]),
                                  members_str[ci], float(ang[ci])))
            per_row_angle[(fname, tag)] = ang[cid_of_row]
            p2.log(f"  loco [{fname} {tag}] max={ang.max():.2f} deg at "
                   f"{cluster_name[clusters[int(np.argmax(ang))][0]]}")
    loco_df = pd.DataFrame(loco_rows, columns=["field", "k", "rank",
                                               "cluster_id", "members",
                                               "loco_angle_deg"])
    loco_df.to_csv(out / "loco_full.csv", index=False)
    p2.log(f"  wrote loco_full.csv ({len(loco_df)} rows)")

    old = pd.read_csv(p2.l15.REPO / "outputs/phase1_5/loco_top5_influencers.csv")
    old5 = old[(old["subset"] == "global") & (old["scale"] == "total")] \
        .sort_values("rank")["cluster_id"].tolist()
    new5 = loco_df[(loco_df["field"] == "total") & (loco_df["k"] == 1)] \
        .sort_values("rank")["cluster_id"].head(5).tolist()
    p2.require(old5 == new5,
               f"LOCO sentinel mismatch vs 1.5: {old5} vs {new5}")
    p2.log(f"  sentinel OK: 1.5 global/total top-5 reproduced {new5}")

    for fname in FIELDS:
        for tag in ("pc1", "pc123"):
            inv[f"loco_{fname}_{tag}_deg"] = per_row_angle[(fname, tag)]

    # ---- amplitude block ----------------------------------------------------
    inv["peak_to_valley_p98p2_um"] = (np.nanpercentile(Rn, 98, axis=(1, 2))
                                      - np.nanpercentile(Rn, 2, axis=(1, 2)))
    inv["R_max_minus_min_um"] = (np.nanmax(Rn, axis=(1, 2))
                                 - np.nanmin(Rn, axis=(1, 2)))
    inv["deepest_negative_residual_um"] = -np.nanmin(Rn, axis=(1, 2))

    # ---- spectral block (descriptors + audit-only band PC scores) ----------
    for c in desc.columns:
        if c != "dataset_index":
            inv[c] = desc[c].to_numpy()
    for band in FIELDS[1:]:
        X = fields[band]
        comps, _ = p2.l15.gram_pca(X, 2)
        sc = (X - X.mean(axis=0, keepdims=True)) @ comps.T
        inv[f"band_PC1_score_audit_{band}"] = sc[:, 0]
        inv[f"band_PC2_score_audit_{band}"] = sc[:, 1]

    # ---- isolation block ----------------------------------------------------
    dcols = cfg["instability"]["descriptor_cols"]
    Zd = p2.robust_z(inv[dcols].to_numpy(float))
    for k in cfg["instability"]["knn_k"]:
        inv[f"D_morph_k{k}"] = p2.knn_median_distance(Zd, int(k))
    Zr = p2.zscore(man[p2.PROC_RAW_COLS].to_numpy(float))
    Zp = p2.zscore(man[p2.PROC_PHYS_COLS].to_numpy(float))
    inv["D_proc_raw_k5"] = p2.knn_median_distance(Zr, 5)
    inv["D_proc_phys_k5"] = p2.knn_median_distance(Zp, 5)

    # ---- artifact block -----------------------------------------------------
    # NOTE: outputs/cone_repair_inventory covers only the 15 single-line pilot
    # groups, NOT the 200-ROI dataset (join-key check proved the mismatch);
    # per-sample repair diagnostics come from the frozen repair_mask instead.
    from scipy.ndimage import label as ndlabel
    # l15.load_frozen does not expose repair_mask (Phase 1.5 contract); load
    # it directly from the same frozen NPZ.
    repair_mask = np.load(p2.l15.REPO / cfg["paths"]["dataset_npz"])["repair_mask"]
    inv["repair_pixel_count"] = repair_mask.reshape(200, -1).sum(1).astype(int)
    inv["repair_fraction"] = man["repair_fraction"].to_numpy()
    inv["valid_fraction"] = man["valid_fraction"].to_numpy()
    largest = np.empty(200, dtype=int)
    for i in range(200):
        lab, n = ndlabel(repair_mask[i])
        largest[i] = int(np.bincount(lab.ravel())[1:].max()) if n > 0 else 0
    inv["repair_largest_component_px"] = largest
    inv["plane_rmse_um"] = man["plane_rmse_um"].to_numpy()
    inv["plane_status"] = man["plane_status"].to_numpy()

    # ---- ranks + conservative consensus (spectral term = min band rank) ----
    def _desc_rank(s: pd.Series) -> np.ndarray:
        return s.rank(ascending=False, method="average").to_numpy()

    inv["rank_Sq"] = _desc_rank(inv["Sq_um"])
    inv["rank_loco_total_pc1"] = _desc_rank(inv["loco_total_pc1_deg"])
    inv["rank_D_morph_k10"] = _desc_rank(inv["D_morph_k10"])
    inv["rank_pit_density"] = _desc_rank(inv["pit_density_per_Mpx"])
    band_ranks = []
    for band in FIELDS[1:]:
        col = f"rank_E_{band}"
        inv[col] = _desc_rank(inv[f"E_{band}_frac"])
        band_ranks.append(inv[col].to_numpy())
    inv["spectral_rank_min"] = np.min(np.column_stack(band_ranks), axis=1)
    inv["A_consensus"] = np.median(np.column_stack([
        inv["rank_Sq"], inv["rank_loco_total_pc1"], inv["rank_D_morph_k10"],
        inv["rank_pit_density"], inv["spectral_rank_min"]]), axis=1)

    # ---- blinded selection pool (cap 30; sentinel + LOCO top5 permanent) ---
    reasons: dict[int, set] = {}

    def _add(idx, reason: str) -> None:
        reasons.setdefault(int(idx), set()).add(reason)

    def _top(col: str, n: int, reason: str) -> None:
        for i in np.argsort(-inv[col].to_numpy())[:n]:
            _add(i, reason)

    _top("loco_total_pc1_deg", 10, "loco_total_pc1_top10")
    for band in FIELDS[1:]:
        _top(f"loco_{band}_pc1_deg", 5, f"loco_{band}_top5")
    _top("Sq_um", 10, "sq_top10")
    _top("D_morph_k10", 10, "morph_isolation_top10")
    ok = inv["D_proc_phys_k5"].to_numpy() > 1e-12
    ratio = np.where(ok, inv["D_morph_k10"].to_numpy()
                     / np.maximum(inv["D_proc_phys_k5"].to_numpy(), 1e-12),
                     -np.inf)
    for i in np.argsort(-ratio)[:10]:
        if ok[i]:
            _add(int(i), "proc_near_morph_far_top10")
    ia, ib = p2.l15.sentinel_rows(man, cfg)
    _add(ia, "sentinel")
    _add(ib, "sentinel")

    candidates = sorted(reasons, key=lambda i: float(inv["A_consensus"].iloc[i]))
    permanent = {i for i, r in reasons.items() if "sentinel" in r}
    permanent |= set(int(i) for i in
                     np.argsort(-inv["loco_total_pc1_deg"].to_numpy())[:5])
    top_cap = int(cfg["instability"]["top_cap"])
    # strict drop tiers (细则 §0.13): ratio-only members first, then
    # morph-isolation-only, then sq-only, then the remaining LOCO-pool-only
    # members; within a tier the least extreme consensus goes first.
    tiers = [["proc_near_morph_far_top10"], ["morph_isolation_top10"],
             ["sq_top10"],
             ["loco_total_pc1_top10"] + [f"loco_{b}_top5" for b in FIELDS[1:]]]
    for tier in tiers:
        if len(candidates) <= top_cap:
            break
        tier_set = set(tier)
        droppable = [i for i in candidates if i not in permanent
                     and reasons[i].issubset(tier_set)]
        droppable.sort(key=lambda j: -float(inv["A_consensus"].iloc[j]))
        for i in droppable:
            if len(candidates) <= top_cap:
                break
            candidates.remove(i)
    p2.require(len(candidates) <= top_cap,
               f"cannot reach top_cap={top_cap}; raise cap or widen pools")

    sel_rows = sorted(candidates, key=lambda i: float(inv["A_consensus"].iloc[i]))
    sel = inv.loc[sel_rows, ["dataset_index", "session_id", "sample_id",
                             "session_role", "processing_order", "A_consensus",
                             "spectral_rank_min", "rank_Sq",
                             "rank_loco_total_pc1", "rank_D_morph_k10",
                             "rank_pit_density", "Sq_um", "D_morph_k10",
                             "D_proc_phys_k5", "loco_total_pc1_deg",
                             "repair_fraction"]].copy()
    sel.insert(0, "anon_code", [f"AUDIT-{j + 1:02d}" for j in range(len(sel))])
    sel["selection_reason"] = ["|".join(sorted(reasons[i])) for i in sel_rows]
    # repair>0 is a LABEL on selected rows only — never an entry pool (细则 §3.2)
    sel["repair_present"] = [bool(inv.loc[i, "repair_fraction"] > 0)
                             for i in sel_rows]
    sel.to_csv(out / "instability_selected.csv", index=False)
    inv.to_csv(out / "instability_inventory.csv", index=False)
    (out / "README.md").write_text(README, encoding="utf-8")

    # ---- manifest backfill (LOCO total/pc1 rank+angle per row) -------------
    rank_by_cluster = np.empty(len(clusters), dtype=int)
    sub = loco_df[(loco_df["field"] == "total") & (loco_df["k"] == 1)]
    name_to_ci = {str(cluster_name[c[0]]): ci
                  for ci, c in enumerate(clusters)}
    for _, r in sub.iterrows():
        rank_by_cluster[name_to_ci[r["cluster_id"]]] = int(r["rank"])
    man_out = man.copy()
    man_out["phase1_global_loco_rank"] = rank_by_cluster[cid_of_row]
    man_out["phase1_global_loco_angle_deg"] = per_row_angle[("total", "pc1")]
    p2.write_manifest(cfg, man_out)

    num_cols = inv.select_dtypes(include=[float]).columns
    p2.require(not inv[list(num_cols)].isna().any().any(),
               "NaN in inventory numeric columns")

    p2.log(f"  selected pool ({len(sel)}): "
           + ", ".join(f"{r.anon_code}=#{r.dataset_index}" for r in sel.itertuples()))
    missing = [f for f in EXPECTED if not (out / f).exists()]
    p2.require(not missing, f"missing outputs: {missing}")
    p2.log(f"01 done in {time.time() - t0:.1f}s; all {len(EXPECTED)} outputs present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
