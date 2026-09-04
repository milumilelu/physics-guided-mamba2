#!/usr/bin/env python3
"""Task 21 (G27-2): P(m|h) decomposition + block-structured shuffled-h null.

Frozen definitions live in 任务说明 v2.1 (FROZEN) — intervals, two-layer
distributions, coverage guard, mutual-exclusive DOMINANT/MIXED labels,
four-class weighted TV with the pooled-center permutation p, and the
H_DEPENDENT sample-level logistic flag.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import _lib as p27  # noqa: E402

EXPECTED = [
    "outputs/phase2_7/peak_selection/peak_selection_m.csv",
    "outputs/phase2_7/peak_selection/family_coverage.csv",
    "outputs/phase2_7/peak_selection/shuffled_h_null_tv.csv",
    "outputs/phase2_7/summary/gsl27_2_evaluation.json",
]


def main() -> int:
    cfg, quick = p27.load_config(__doc__)
    g2 = cfg["g27_2"]
    seed = int(cfg["meta"]["random_seed"])
    out = p27.output_dir(cfg, "peak_selection")
    summary_dir = p27.output_dir(cfg, "summary")
    hatch = pd.read_csv(p27.REPO / cfg["paths"]["lambda_over_hatch"],
                        encoding="utf-8-sig")
    p27.require(len(hatch) == 200, "lambda_over_hatch must hold 200 rows")
    p27.log(f"Task 21 start | quick={quick}")

    r_peak = hatch["lambda_peak_4_32_um"].to_numpy(dtype=float)
    valid = hatch["lambda_peak_valid"].astype(bool).to_numpy()
    h_arr = hatch["hatch_spacing_um"].to_numpy(dtype=float)
    classes = p27.assign_class(r_peak / h_arr, valid)
    family = valid & (classes != p27.CODE_OUT)
    hatch["class_code"] = classes
    hatch["class_name"] = [p27.CLASS_NAMES[c] for c in classes]
    hatch.to_csv(out / "peak_selection_m.csv", index=False,
                 encoding="utf-8-sig")

    # two-layer distributions: q_h four-class over peak-valid per h + overall
    h_levels = sorted(np.unique(h_arr).tolist())
    q_h, n_h, c_family_h = {}, {}, {}
    for h in h_levels:
        sel = valid & (h_arr == h)
        n_h[h] = int(sel.sum())
        q_h[h] = p27.q_distribution(classes[sel])
        fam = int((classes[sel] != p27.CODE_OUT).sum())
        c_family_h[h] = fam / n_h[h] if n_h[h] else np.nan
    valid_n = int(valid.sum())
    q_overall = p27.q_distribution(classes[valid])
    c_family_all = float((classes[valid] != p27.CODE_OUT).mean())
    conditional = {int(m): float((classes[family] == code).mean())
                   for code, m in ((p27.CODE_M1, 1), (p27.CODE_M2, 2),
                                   (p27.CODE_M3, 3))}
    coverage = pd.DataFrame([
        {"hatch_spacing_um": h, "n_peak_valid": n_h[h],
         "C_family_h": c_family_h[h],
         "P_OUT_h": q_h[h][p27.CODE_OUT],
         "P_m1_h": q_h[h][p27.CODE_M1], "P_m2_h": q_h[h][p27.CODE_M2],
         "P_m3_h": q_h[h][p27.CODE_M3], "LOW_N": n_h[h] < g2["low_n_family"]}
        for h in h_levels])
    coverage.to_csv(out / "family_coverage.csv", index=False,
                    encoding="utf-8-sig")

    # ---- block-structured shuffled-h null (four-class TV, pooled center) -- #
    n_perm = int(g2["n_perm_tv"])
    manifest = pd.read_csv(p27.REPO / __import__("yaml").safe_load(
        (Path(__file__).resolve().parent / "phase2_7_config.yaml")
        .read_text(encoding="utf-8"))["paths"]["phase2_manifest"])
    frame = (hatch[["dataset_index", "hatch_spacing_um", "class_code"]]
             .merge(manifest[["dataset_index", "session_id",
                              "base_condition_group"]],
                    on="dataset_index"))
    weights = {h: n_h[h] / valid_n for h in h_levels}
    q_obs_h = {h: q_h[h] for h in h_levels}
    q_null_h = {h: [] for h in h_levels}
    for b in range(n_perm):
        shuffled = p27.shuffle_h_by_block(frame, unit_columns=("session_id",
                                        "base_condition_group"),
                                        seed=seed + int(cfg["seeds"]["tv_perm"]) + b)
        h_shuf = shuffled.to_numpy(dtype=float)
        cls_shuf = p27.assign_class(r_peak / h_shuf, valid)
        for h in h_levels:
            sel = valid & (h_shuf == h)
            q_null_h[h].append(p27.q_distribution(cls_shuf[sel]))
    q_null_h = {h: q_null_h[h] for h in h_levels}
    perm = p27.tv_perm_p(q_obs_h, q_null_h, weights, n_perm=n_perm)
    tv_w = perm["t_obs"]
    null_frame = pd.DataFrame({
        "perm": range(n_perm),
        "T_b": [sum(weights[h] * p27.tv(q_null_h[h][b], perm["q_bar"][h])
                    for h in h_levels) for b in range(n_perm)]})
    null_frame.to_csv(out / "shuffled_h_null_tv.csv", index=False,
                      encoding="utf-8-sig")

    # ---- verdict (v2.1: coverage 优先 → TV/p 前提 → MIXED 先于 DOMINANT) -- #
    ps = sorted(conditional.values(), reverse=True)
    if c_family_all < g2["c_family_min"]:
        verdict = "INSUFFICIENT_FAMILY_COVERAGE"
    elif tv_w < g2["tv_min"] or perm["p_value"] > g2["p_max"]:
        verdict = "NO_DOMINANT"
    elif (ps[0] - ps[1] < g2["mixed_gap"]) and (ps[1] >= g2["mixed_p2_min"]):
        verdict = "MIXED"
    elif ps[0] >= g2["dominant_min"] and ps[0] - ps[1] >= g2["mixed_gap"]:
        verdict = f"DOMINANT_m={int(max(conditional, key=conditional.get))}"
    else:
        verdict = "NO_DOMINANT"

    # ---- H_DEPENDENT: sample-level logistic I(m=2) ~ h -------------------- #
    family_rows = hatch[family]
    h_f = family_rows["hatch_spacing_um"].to_numpy(dtype=float)
    is_m2 = (family_rows["class_code"] == p27.CODE_M2).to_numpy(dtype=int)
    slope_obs = p27.logistic_slope(h_f, is_m2)
    n_perm_log = int(g2["n_perm_logistic"])
    rng = np.random.default_rng(seed + int(cfg["seeds"]["logistic_perm"]))
    slope_perm = np.empty(n_perm_log)
    for b in range(n_perm_log):
        shuf = p27.shuffle_h_by_block(frame, unit_columns=("session_id",
                                      "base_condition_group"),
                                      seed=seed + int(cfg["seeds"]["logistic_perm"]) + b)
        # shuffle returns a Series aligned to `frame`'s index; family is a
        # subset, so reindex by family dataset_index before fitting
        h_perm = shuf.reindex(family_rows["dataset_index"]).to_numpy(
            dtype=float)
        slope_perm[b] = p27.logistic_slope(h_perm, is_m2)
    p_logistic = float((1 + int((slope_perm <= slope_obs).sum()))
                       / (1 + n_perm_log))
    h_dependent = "YES" if (slope_obs < 0 and p_logistic <= 0.05) else "NO"

    evaluation = {
        "n_peak_valid": valid_n,
        "C_family_all": c_family_all,
        "C_family_min": g2["c_family_min"],
        "q_overall_four_class": {"OUT": q_overall[p27.CODE_OUT],
                                 "m1": q_overall[p27.CODE_M1],
                                 "m2": q_overall[p27.CODE_M2],
                                 "m3": q_overall[p27.CODE_M3]},
        "conditional_P_m_family": conditional,
        "P_m2_by_h": {str(h): float((classes[valid & (h_arr == h)]
                                     == p27.CODE_M2).mean())
                      for h in h_levels},
        "tv_w": tv_w, "tv_min": g2["tv_min"],
        "p_perm": perm["p_value"], "p_max": g2["p_max"],
        "t_obs": perm["t_obs"], "t_null_median": perm["t_null_median"],
        "H_DEPENDENT": {"slope": slope_obs, "p_value": p_logistic,
                        "flag": h_dependent, "n_perm": n_perm_log},
        "G_SL27_2": verdict,
    }
    (summary_dir / "gsl27_2_evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2),
        encoding="utf-8")
    p27.log(f"G27-2 = {verdict} | C_family={c_family_all:.3f} | TV_w="
            f"{tv_w:.3f} (p={perm['p_value']:.4f}) | conditional "
            f"P(m)={ {k: round(v_, 3) for k, v_ in conditional.items()} } | "
            f"H_DEPENDENT={h_dependent} (slope={slope_obs:.3f}, "
            f"p={p_logistic:.4f})")
    p27.log("Task 21 done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
