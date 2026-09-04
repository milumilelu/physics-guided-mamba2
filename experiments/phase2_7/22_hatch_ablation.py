#!/usr/bin/env python3
"""Task 22 (G27-1): hatch unique-contribution ablation.

M_{-h}: Y~[tau,f,N,v]  vs  M_h: Y~h  vs  M_full: Y~[tau,f,h,N,v].
Fold-paired ΔR²_h = R²(M_full) − R²(M_{-h}) on full-200 (src_gkf/proc_gkf)
with the in-box 101 as sensitivity.  Route T = {A2_8_16, angular_entropy_8_16};
Route P = {p_8_16, ilr_z1_z4} (Q² in ILR space).  G27-1 verdict with the
v2.1 proc-GKF consistency cap.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import _lib as p27  # noqa: E402

EXPECTED = [
    "outputs/phase2_7/hatch_ablation/hatch_ablation_cv.csv",
    "outputs/phase2_7/summary/gsl27_1_evaluation.json",
]

U_COLUMNS = ["pulse_duration_fs", "frequency_kHz", "hatch_spacing_um",
             "pass_count", "velocity_mm_s"]
MINUS_H = ["pulse_duration_fs", "frequency_kHz", "pass_count",
           "velocity_mm_s"]


def main() -> int:
    cfg, quick = p27.load_config(__doc__)
    g1 = cfg["g27_1"]
    seed = int(cfg["meta"]["random_seed"])
    out = p27.output_dir(cfg, "hatch_ablation")
    summary_dir = p27.output_dir(cfg, "summary")
    p27.log(f"Task 22 start | quick={quick}")

    manifest = pd.read_csv(REPO / cfg["paths"]["phase2_manifest"])
    spectral = pd.read_csv(REPO / cfg["paths"]["p25_spectral_csv"])
    ilr = pd.read_csv(REPO / cfg["paths"]["p25_ilr_csv"])
    directional = pd.read_csv(REPO / cfg["paths"]["p25_directional_csv"])
    dir_816 = directional[directional["band"] == "8_16"].set_index(
        "dataset_index")
    targets = {
        "p_8_16": spectral.set_index("dataset_index")["p_8_16"],
        "ilr_z1_z4": ilr.set_index("dataset_index")[
            ["ilr_z1", "ilr_z2", "ilr_z3", "ilr_z4"]],
        "A2_8_16": dir_816["A2"],
        "angular_entropy_8_16": dir_816["angular_entropy"],
    }
    splits_variants = {
        "src_gkf": p27.p2.gkf_splits(manifest["shared_height_source_id"], 5),
        "proc_gkf": p27.p2.gkf_splits(manifest["cv_process_group"], 5),
    }
    for name, splits in splits_variants.items():
        groups = (manifest["shared_height_source_id"] if name == "src_gkf"
                  else manifest["cv_process_group"])
        p27.p2.check_gkf_contract(groups, splits)

    X_full = manifest[U_COLUMNS].to_numpy(dtype=float)
    X_minus_h = manifest[MINUS_H].to_numpy(dtype=float)
    X_h = manifest[["hatch_spacing_um"]].to_numpy(dtype=float)
    matrices = {"M_full": X_full, "M_minus_h": X_minus_h, "M_h": X_h}
    rows = []
    for variant, splits in splits_variants.items():
        for model_name, X in matrices.items():
            for target, y in targets.items():
                y_all = y.reindex(manifest["dataset_index"]).to_numpy(
                    dtype=float)
                valid = (np.isfinite(y_all).all(axis=1) if y_all.ndim > 1
                         else np.isfinite(y_all))
                for fold, (tr, te) in enumerate(splits):
                    tr = np.array([i for i in tr if valid[i]])
                    te = np.array([i for i in te if valid[i]])
                    if target == "ilr_z1_z4":
                        alpha = p27.ridge_alpha_inner_gkf(
                            X[tr], y_all[tr][:, 0],
                            manifest["cv_process_group"].iloc[tr])
                        model = p27.make_ridge(alpha).fit(X[tr], y_all[tr])
                        pred = model.predict(X[te])
                        score = p27.q2_aitchison_ilr(y_all[te], pred, y_all[tr])
                        metric = "Q2_Aitchison"
                    else:
                        alpha = p27.ridge_alpha_inner_gkf(
                            X[tr], y_all[tr],
                            manifest["cv_process_group"].iloc[tr])
                        model = p27.make_ridge(alpha).fit(X[tr], y_all[tr])
                        pred = model.predict(X[te])
                        score = float(r2_score(y_all[te], pred))
                        metric = "R2"
                    rows.append({"variant": variant, "fold": fold,
                                 "model": model_name, "target": target,
                                 "metric": metric, "score": float(score),
                                 "alpha": alpha,
                                 "n_train": int(len(tr)),
                                 "n_test": int(len(te))})
    folds = pd.DataFrame(rows)

    # in-box 101 sensitivity (regenerated splits, same protocol)
    box = yaml.safe_load((Path(__file__).resolve().parent.parent
        / "phase2_6" / "phase2_6_config.yaml").read_text(encoding="utf-8")
    )["bridge"]["box"]
    in_box = p27.in_box_mask(manifest, box)
    sub = manifest[in_box]
    sub_splits = {"src_gkf_inbox": p27.p2.gkf_splits(
        sub["shared_height_source_id"], 5)}
    rows_inbox = []
    for name, splits in sub_splits.items():
        p27.p2.check_gkf_contract(sub["shared_height_source_id"], splits)
        for model_name, X in matrices.items():
            Xs = X[in_box.to_numpy()]
            for target, y in targets.items():
                y_all = y.reindex(manifest["dataset_index"]).to_numpy(
                    dtype=float)[in_box.to_numpy()]
                valid = (np.isfinite(y_all).all(axis=1) if y_all.ndim > 1
                         else np.isfinite(y_all))
                for fold, (tr, te) in enumerate(splits):
                    tr = np.array([i for i in tr if valid[i]])
                    te = np.array([i for i in te if valid[i]])
                    if target == "ilr_z1_z4":
                        alpha = p27.ridge_alpha_inner_gkf(
                            Xs[tr], y_all[tr][:, 0],
                            sub["cv_process_group"].iloc[tr])
                        model = p27.make_ridge(alpha).fit(Xs[tr], y_all[tr])
                        pred = model.predict(Xs[te])
                        score = p27.q2_aitchison_ilr(y_all[te], pred, y_all[tr])
                        metric = "Q2_Aitchison"
                    else:
                        alpha = p27.ridge_alpha_inner_gkf(
                            Xs[tr], y_all[tr],
                            sub["cv_process_group"].iloc[tr])
                        model = p27.make_ridge(alpha).fit(Xs[tr], y_all[tr])
                        pred = model.predict(Xs[te])
                        score = float(r2_score(y_all[te], pred))
                        metric = "R2"
                    rows_inbox.append({"variant": name, "fold": fold,
                                 "model": model_name, "target": target,
                                 "metric": metric, "score": float(score),
                                 "alpha": alpha, "n_train": int(len(tr)),
                                 "n_test": int(len(te))})
    # 2.7r1 fix: keep sensitivity rows in a SEPARATE list -- appending them to
    # the primary `rows` duplicated every full-200 fold row and inflated
    # n_folds_positive to 10
    folds = pd.concat([folds, pd.DataFrame(rows_inbox)], ignore_index=True)
    p27.require(not folds.duplicated(
        ["variant", "model", "target", "metric", "fold"]).any(),
        "duplicate fold rows detected (formal-contract violation)")
    folds.to_csv(out / "hatch_ablation_cv.csv", index=False,
                 encoding="utf-8-sig")

    # fold-paired ΔR²_h + G27-1 verdict (v2.1: src 主门槛 + proc cap)
    delta_rows = []
    for variant in ("src_gkf", "proc_gkf", "src_gkf_inbox"):
        for (target, metric), block in folds[folds["variant"] == variant].groupby(
                ["target", "metric"]):
            full = block[block["model"] == "M_full"].set_index("fold")["score"]
            minus = block[block["model"] == "M_minus_h"].set_index("fold")["score"]
            # 2.7r1 fix: FROZEN statistic is the median of FOLD-PAIRED deltas,
            # not the difference of medians (report both; gate uses paired)
            delta = (full - minus).dropna()
            delta_rows.append({
                "variant": variant, "target": target, "metric": metric,
                "R2_full_median": float(full.median()),
                "R2_minus_h_median": float(minus.median()),
                "median_delta_R2_h": float(delta.median()),
                "n_folds_positive": int((delta > 0).sum()),
                "retention_h": (float(minus.median() / full.median())
                                if full.median() > 0 else np.nan)})
    deltas = pd.DataFrame(delta_rows)
    route_t = deltas[(deltas["variant"] == "src_gkf")
                     & deltas["target"].isin(g1["route_t"])
                     & (deltas["metric"] == "R2")]
    src_pass = bool((route_t["median_delta_R2_h"] >= g1["delta_min"]).all()
                    and (route_t["n_folds_positive"]
                         >= g1["min_folds_positive"]).all())
    proc_t = deltas[(deltas["variant"] == "proc_gkf")
                    & deltas["target"].isin(g1["route_t"])
                    & (deltas["metric"] == "R2")]
    proc_ok = bool((proc_t["median_delta_R2_h"]
                    > g1["proc_cap"]["median_delta_gt"]).all()
                   and (proc_t["n_folds_positive"]
                        >= g1["proc_cap"]["min_folds_positive"]).all())
    if not src_pass:
        verdict = "NOT_SUPPORTED"
    elif proc_ok:
        verdict = "SUPPORTED"
    else:
        verdict = "PARTIAL(proc consistency cap)"
    gsl27_1 = {
        "type": "hatch_unique_contribution",
        "population": "full_200_primary + in_box_101_sensitivity",
        "delta_table": deltas.to_dict(orient="records"),
        "thresholds": {"delta_min": g1["delta_min"],
                       "min_folds_positive": g1["min_folds_positive"],
                       "proc_cap": g1["proc_cap"]},
        "src_pass": src_pass, "proc_consistency_ok": proc_ok,
        "G_SL27_1": verdict,
        "reading": ("Route T hatch 主导（unique contribution 成立）"
                    if verdict == "SUPPORTED" else
                    "unique contribution 未达冻结门槛；Route P 仍为多因素问题"
                    "（contrast 为描述性报告）"),
    }
    (summary_dir / "gsl27_1_evaluation.json").write_text(
        json.dumps(gsl27_1, ensure_ascii=False, indent=2),
        encoding="utf-8")
    p27.log(f"G27-1 = {verdict} | Route T src median ΔR²_h = "
            f"{route_t['median_delta_R2_h'].round(3).tolist()} "
            f"(proc {proc_t['median_delta_R2_h'].round(3).tolist()})")
    p27.log("Task 22 done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
