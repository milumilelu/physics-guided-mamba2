#!/usr/bin/env python3
"""Task SL-03: scale bridge + M0/M0b/M1/M2/M3/M_GEO comparison (细则 §6).

Step order (frozen):
  0. M0_RECON_FULL200 (pure QA, §0.16): full-200 Ridge, Phase 2.5 protocol
     (input A, src_gkf full-200 splits), reconciled against
     outputs/phase2_5/process_map/cv_fold_results.csv at |Δ| <= 0.005.
  1. SL-03a exact-match direct bridge (§0.17): 19 conditions, measurement ->
     measurement, evidence priority direct > in-box predicted > out-of-box.
  2. W_hat generation: Task-17 Ridge (estimable & != reject) refit -> predict
     the 200 rectangle samples (feature whitelist tau/f/v/N only).
  3. Model comparison on the in-box 101 subset with REGENERATED src_gkf /
     proc_gkf splits; G-SL3 = Geometry-compression Gate (retention of
     M_GEO=[W_hat, h, W/h] vs M0=[u]); Aitchison Q2 ONLY on ilr_z1_z4.
  4. Sensitivity arms: full-200 extrapolated, exclude_artifact, minus_top5,
     spline / ExtraTrees (compression reference only, §0.13).

Delta(M1 - M0) is reported as LOW-CAPACITY REPRESENTATION GAIN and never as
mechanism evidence; Delta(M1 - M0b) is the transform-control confirmatory
check (expected ~0; warning if |median| > 0.02).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import ExtraTreesRegressor  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler, SplineTransformer  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import _lib as p26  # noqa: E402

EXPECTED = [
    "outputs/phase2_6/model_compare/m0_reconciliation.csv",
    "outputs/phase2_6/scale_bridge/morphology_scale_match.csv",
    "outputs/phase2_6/scale_bridge/direct_bridge_exact_match.csv",
    "outputs/phase2_6/model_compare/width_bridge_cv.csv",
    "outputs/phase2_6/model_compare/overlap_bridge_cv.csv",
    "outputs/phase2_6/model_compare/oof_predictions.csv",
    "outputs/phase2_6/summary/gsl3_evaluation.json",
]

U_COLUMNS = ["pulse_duration_fs", "frequency_kHz", "hatch_spacing_um",
             "pass_count", "velocity_mm_s"]
WHAT_FEATURES = ["log10_tau", "frequency_kHz", "velocity_mm_s", "pass_count"]
ARTIFACT_IDS = (37, 149, 82)
RECON_TOLERANCE = 0.005

# ---- Phase 2.5 Task 12 protocol, reproduced EXACTLY for M0_RECON (审计 P0-2):
# alpha grid [0.01, 0.1, 1, 10, 100] (phase2_5 config ridge_alpha_grid),
# inner GroupKFold(3) median-R2 selection, scaler fit on the inner-train only,
# scalar alpha from the first target column, and Q2 in ILR-coordinate space.
P25_ALPHA_GRID = [0.01, 0.1, 1, 10, 100]


def p25_select_alpha(X_raw: np.ndarray, y: np.ndarray, groups: np.ndarray,
                     grid: list[float] = P25_ALPHA_GRID) -> float:
    """Task 12 `_select_alpha_shared`: inner GroupKFold(3), median R2."""
    from sklearn.model_selection import GroupKFold
    if len(set(groups.tolist())) < 3:
        return float(grid[len(grid) // 2])
    scores: dict[float, list[float]] = {a: [] for a in grid}
    for itr, ival in GroupKFold(n_splits=3).split(X_raw, groups=groups):
        if len(np.unique(y[ival])) < 2:
            continue
        sc = StandardScaler().fit(X_raw[itr])
        Xit, Xiv = sc.transform(X_raw[itr]), sc.transform(X_raw[ival])
        for a in grid:
            m = Ridge(alpha=float(a)).fit(Xit, y[itr])
            scores[a].append(r2_score(y[ival], m.predict(Xiv)))
    best = grid[0]
    med = {a: (float(np.nanmedian(v)) if v else -np.inf) for a, v in scores.items()}
    for a in grid[1:]:
        if med[a] > med[best]:
            best = a
    return float(best)


def p25_ridge_fit_predict(X_raw: np.ndarray, y: np.ndarray,
                          groups_tr: np.ndarray, tr: np.ndarray, te: np.ndarray
                          ) -> tuple[np.ndarray, float]:
    """Task 12 ridge arm: scaler fit on the outer-train, alpha by the inner
    GroupKFold(3) median-R2 protocol; returns (test predictions, alpha)."""
    alpha = p25_select_alpha(X_raw[tr], y[tr], groups_tr)
    sc = StandardScaler().fit(X_raw[tr])
    model = Ridge(alpha=alpha).fit(sc.transform(X_raw[tr]), y[tr])
    return model.predict(sc.transform(X_raw[te])), alpha


def q2_aitchison_ilr(z_test: np.ndarray, z_pred: np.ndarray,
                     z_train: np.ndarray) -> float:
    """Task 12 `_q2_aitchison`: Q2 in ILR-coordinate space
    1 - sum(z_test - z_pred)^2 / sum(z_test - mean(z_train))^2 (审计 P0-2:
    the Phase 2.5 metric never composes back to the simplex)."""
    denom = float(((z_test - z_train.mean(axis=0)) ** 2).sum())
    if denom <= 0:
        return np.nan
    return float(1.0 - ((z_test - z_pred) ** 2).sum() / denom)


# --------------------------------------------------------------------------- #
# shared model plumbing
# --------------------------------------------------------------------------- #
def fit_predict_ridge(X_tr, y_tr, X_te, groups_tr) -> np.ndarray:
    alpha = p26.ridge_alpha_inner_gkf(X_tr, y_tr, groups_tr)
    return p26.make_ridge(alpha).fit(X_tr, y_tr).predict(X_te), alpha


def make_spline():
    return Pipeline([("scale", StandardScaler()),
                     ("spline", SplineTransformer(degree=3, n_knots=4,
                                                  include_bias=False)),
                     ("ridge", Ridge(alpha=1.0))])


# --------------------------------------------------------------------------- #
# step 0: M0_RECON_FULL200 (pure QA) — 审计 P0-2 修复
# --------------------------------------------------------------------------- #
def m0_recon_full200(manifest: pd.DataFrame, folds_csv: pd.DataFrame,
                     targets: dict, splits) -> pd.DataFrame:
    """Reconcile the Phase 2.5 Task 12 protocol on the full 200.

    Reference rows = cv_fold_results.csv (model=ridge, input_set=A,
    cv_variant=src_gkf).  Only targets PRESENT in the reference are
    reconciled: `ilr_z1_z4` (Q2_Aitchison, ILR-coordinate space) plus the
    scalar targets that exist on both sides (A2_8_16, angular_entropy_8_16).
    `p_8_16` / single `ilr_z1..z4` are Phase 2.6-only targets and are NOT
    reconciled (the reference has no such rows).
    """
    reference = folds_csv[(folds_csv["model"].astype(str).str.lower() == "ridge")
                          & (folds_csv["input_set"] == "A")
                          & (folds_csv["cv_variant"].astype(str)
                             == "src_gkf")]
    ref_targets = set(reference["target"].unique().tolist())
    recon_spec = {"ilr_z1_z4": "Q2_Aitchison",
                  "A2_8_16": "R2",
                  "angular_entropy_8_16": "R2"}
    X = manifest[U_COLUMNS].to_numpy(dtype=float)
    groups = manifest["shared_height_source_id"]
    p26.p2.check_gkf_contract(groups, splits)
    rows = []
    for target, metric in recon_spec.items():
        if target not in ref_targets or target not in targets:
            rows.append({"target": target, "metric": metric,
                         "status": "no_reference_rows_or_target",
                         "phase25_median": np.nan, "rerun_median": np.nan,
                         "abs_delta": np.nan, "within_tolerance": np.nan})
            continue
        y = targets[target]
        if target == "ilr_z1_z4":
            y = y.reindex(manifest["dataset_index"])
        y = y.loc[manifest["dataset_index"]].to_numpy(dtype=float)
        ref_rows = reference[reference["target"] == target]
        ref_median = float(ref_rows[metric].median())
        fold_scores, fold_alphas = [], []
        for tr, te in splits:
            if metric == "Q2_Aitchison":
                _, alpha = p25_ridge_fit_predict(X, y[:, 0], groups.iloc[tr],
                                                 tr, te)
                sc = StandardScaler().fit(X[tr])
                model = Ridge(alpha=alpha).fit(sc.transform(X[tr]), y[tr])
                pred = model.predict(sc.transform(X[te]))
                score = q2_aitchison_ilr(y[te], pred, y[tr])
            else:
                pred, alpha = p25_ridge_fit_predict(X, y, groups.iloc[tr],
                                                    tr, te)
                score = float(r2_score(y[te], pred))
            fold_scores.append(float(score))
            fold_alphas.append(alpha)
        rerun_median = float(np.median(fold_scores))
        rows.append({"target": target, "metric": metric,
                     "status": "reconciled",
                     "phase25_median": ref_median,
                     "rerun_median": rerun_median,
                     "abs_delta": abs(rerun_median - ref_median),
                     "within_tolerance":
                         abs(rerun_median - ref_median) <= RECON_TOLERANCE,
                     "fold_alphas": ";".join(f"{a:g}" for a in fold_alphas)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# step 1: SL-03a direct bridge (measurement -> measurement)
# --------------------------------------------------------------------------- #
def direct_bridge(match: pd.DataFrame, line_conditions: pd.DataFrame) -> pd.DataFrame:
    """SL-03a: lambda* (per rectangle sample, condition-aggregated) vs the
    MEASURED single-line width at the SAME (tau, f, v, N) condition --
    measurement -> measurement, no W_hat anywhere.

    Statistical unit = unique exact-match condition (frozen: 19 conditions /
    20 samples); the one repeated condition (54 formal / 156 pass-T12,
    different hatch) aggregates lambda* by mean and records its spread and
    hatch values.  W availability two states (§0.17 rev2 补注):
      right_censored        -> W_lower_bound (r has <=-truth semantics)
      insufficient_sections -> W_unavailable (no r, no directionality,
                               excluded from the denominator)
      estimable & qa!=reject-> estimable (r_W_direct defined)
      estimable & reject    -> rejected_by_qa (excluded from the denominator)
    """
    state_map = {"right_censored": "W_lower_bound",
                 "insufficient_sections": "W_unavailable"}
    condition_columns = ["pulse_duration_fs", "frequency_kHz", "pass_count",
                         "velocity_mm_s"]
    rows = []
    exact = match[match["bridge_coverage"] == "exact_match"]
    for key, group in exact.groupby(condition_columns):
        tau, f, n, v = key
        hits = line_conditions[
            (line_conditions["pulse_duration_fs"] == tau)
            & (line_conditions["frequency_kHz"] == f)
            & (line_conditions["pass_count"] == n)
            & (line_conditions["velocity_mm_s"] == v)]
        w_measured, w_source = np.nan, "W_unavailable"
        if len(hits):
            row = hits.iloc[0]
            if row["width_identifiability"] in state_map:
                # insufficient (+possibly reject) -> W_unavailable first: no
                # width estimate exists, the stronger diagnostic state
                # (audit: lines 10/83/90 are reject AND insufficient)
                w_source = state_map[row["width_identifiability"]]
                if w_source == "W_lower_bound":
                    w_measured = float(row["median_W50_um"])
            elif row["qa_label"] == "reject_geometry":
                w_source = "rejected_by_qa"
            else:
                w_measured = float(row["median_W50_um"])
                w_source = "estimable"
        lam = float(group["lambda_star_4_32_um"].mean())
        rows.append({
            "condition": f"{tau}:{f}:{n}:{v}",
            "n_samples": int(len(group)),
            "dataset_index_list": ";".join(map(str, sorted(group["dataset_index"]))),
            "hatch_values": ";".join(map(str, sorted(
                group["hatch_spacing_um"].unique().tolist()))),
            "lambda_star_mean_um": lam,
            "lambda_star_spread_um": float(
                group["lambda_star_4_32_um"].std(ddof=0)),
            "W_line_measured_um": w_measured,
            "W_source": w_source,
            "r_W_direct": lam / w_measured if w_measured > 0 else np.nan,
        })
    p26.require(len(rows) == 19,
                f"direct bridge must hold 19 exact-match conditions, got {len(rows)}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# step 3: model comparison with fold-paired retention
# --------------------------------------------------------------------------- #
def run_model_matrix(frame: pd.DataFrame, splits, targets: dict[str, pd.Series],
                     *, variant: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every frozen model over every target; return (fold rows, oof rows).

    OOF contract: one row per (variant, model, target, dataset_index) for
    scalar targets (composition handled at fold level only).
    """
    fold_rows, oof_rows = [], []
    u = frame[U_COLUMNS].to_numpy(dtype=float)
    log_tau = np.log10(frame["pulse_duration_fs"].to_numpy(dtype=float))
    w_hat = frame["W_hat_um"].to_numpy(dtype=float)
    hatch = frame["hatch_spacing_um"].to_numpy(dtype=float)
    eta = w_hat / hatch
    datasets = frame["dataset_index"].to_numpy(dtype=int)
    models = {
        "M0_u": u,
        "M0b_u_plus_logtau": np.column_stack([u, log_tau]),
        "M1_u_plus_What": np.column_stack([u, w_hat]),
        "M2_h": hatch[:, None],
        "M3_u_What_What_over_h": np.column_stack([u, w_hat, eta]),
        "M_GEO_What_h_eta": np.column_stack([w_hat, hatch, eta]),
    }
    for model_name, X in models.items():
        for target, y in targets.items():
            y_all = y.to_numpy(dtype=float)
            valid = (np.isfinite(y_all).all(axis=1) if y_all.ndim > 1
                     else np.isfinite(y_all))
            for fold, (tr, te) in enumerate(splits):
                tr = np.array([i for i in tr if valid[i]])
                te = np.array([i for i in te if valid[i]])
                if target == "ilr_z1_z4":
                    z = y.to_numpy(dtype=float)
                    pred_te = np.empty((len(te), z.shape[1]))
                    for j in range(z.shape[1]):
                        alpha = p26.ridge_alpha_inner_gkf(
                            X[tr], z[tr, j],
                            frame["cv_process_group"].iloc[tr])
                        pred_te[:, j] = p26.make_ridge(alpha).fit(
                            X[tr], z[tr, j]).predict(X[te])
                    q2 = q2_aitchison_ilr(z[te], pred_te, z[tr])
                    fold_rows.append({"variant": variant, "fold": fold,
                                      "model": model_name, "target": target,
                                      "metric": "Q2_Aitchison", "score": q2,
                                      "n_train": int(len(tr)),
                                      "n_test": int(len(te)), "alpha": np.nan})
                    continue
                y_all = y.to_numpy(dtype=float)
                alpha = p26.ridge_alpha_inner_gkf(
                    X[tr], y_all[tr], frame["cv_process_group"].iloc[tr])
                pred = p26.make_ridge(alpha).fit(X[tr], y_all[tr]).predict(X[te])
                fold_rows.append({"variant": variant, "fold": fold,
                                  "model": model_name, "target": target,
                                  "metric": "R2",
                                  "score": float(r2_score(y_all[te], pred)),
                                  "n_train": int(len(tr)),
                                  "n_test": int(len(te)), "alpha": alpha})
                for local, idx in enumerate(te):
                    oof_rows.append({"variant": variant, "model": model_name,
                                     "input": "target_models", "fold": fold,
                                     "dataset_index": int(datasets[idx]),
                                     "target": target,
                                     "y_true": float(y_all[idx]),
                                     "y_pred": float(pred[local])})
    return pd.DataFrame(fold_rows), pd.DataFrame(oof_rows)


# --------------------------------------------------------------------------- #
def main() -> int:
    cfg, quick = p26.load_config(__doc__)
    seed = int(cfg["meta"]["random_seed"])
    scale = p26.output_dir(cfg, "scale_bridge")
    compare = p26.output_dir(cfg, "model_compare")
    summary_dir = p26.output_dir(cfg, "summary")
    p26.log(f"Task 18 start | quick={quick}")

    geometry = pd.read_csv(p26.REPO / "outputs/phase2_6/single_line"
                           "/single_line_geometry.csv", encoding="utf-8-sig")
    labels = pd.read_csv(p26.REPO / "outputs/phase2_6/single_line"
                         "/geometry_qa_labels.csv", encoding="utf-8-sig")
    manifest = pd.read_csv(p26.REPO / cfg["paths"]["phase2_manifest"])
    spectral = pd.read_csv(p26.REPO / cfg["paths"]["p25_spectral_csv"])
    ilr = pd.read_csv(p26.REPO / cfg["paths"]["p25_ilr_csv"])
    directional = pd.read_csv(p26.REPO / cfg["paths"]["p25_directional_csv"])
    radial = pd.read_csv(p26.REPO / cfg["paths"]["p25_radial_long_csv"])
    descriptors = pd.read_csv(p26.REPO / cfg["paths"]["p25_descriptor_csv"])

    # targets (frozen §6.3): composition Q2 only on ilr_z1_z4
    dir_816 = directional[directional["band"] == "8_16"].set_index("dataset_index")
    targets_full = {
        "p_8_16": spectral.set_index("dataset_index")["p_8_16"],
        "ilr_z2": ilr.set_index("dataset_index")["ilr_z2"],
        "ilr_z1_z4": ilr.set_index("dataset_index")[
            ["ilr_z1", "ilr_z2", "ilr_z3", "ilr_z4"]],
        "A2_8_16": dir_816["A2"],
        "angular_entropy_8_16": dir_816["angular_entropy"],
    }
    ilr_wide = targets_full["ilr_z1_z4"]

    # ---- step 0: M0_RECON_FULL200 ---------------------------------------- #
    folds_csv = pd.read_csv(p26.REPO / cfg["paths"]["p25_cv_fold_csv"])
    full_splits = p26.p2.gkf_splits(manifest["shared_height_source_id"], 5)
    recon = m0_recon_full200(manifest, folds_csv, targets_full, full_splits)
    recon.to_csv(compare / "m0_reconciliation.csv", index=False,
                 encoding="utf-8-sig")
    p26.require(recon["within_tolerance"].all(),
                f"M0_RECON_FULL200 failed tolerance {RECON_TOLERANCE}:\n"
                f"{recon.to_string()}")
    p26.log("M0_RECON_FULL200 OK (pure QA; full-200 values never gate)")

    # ---- step 2: W_hat generation ---------------------------------------- #
    line_frame = (geometry.merge(labels[["single_line_id", "qa_label"]],
                                 on="single_line_id")
                  .merge(pd.read_csv(p26.REPO / "outputs/phase2_6/single_line"
                                     "/single_line_manifest.csv",
                                     encoding="utf-8-sig")[
                      ["single_line_id", "pulse_duration_fs",
                       "frequency_kHz", "velocity_mm_s", "pass_count"]],
                      on="single_line_id"))
    line_frame["log10_tau"] = np.log10(line_frame["pulse_duration_fs"]
                                       .astype(float))
    train = line_frame[(line_frame["width_identifiability"] == "estimable")
                       & (line_frame["qa_label"] != "reject_geometry")]
    X_train = train[WHAT_FEATURES].to_numpy(dtype=float)
    y_train = train["median_W50_um"].to_numpy(dtype=float)
    alpha_hat = p26.ridge_alpha_inner_gkf(X_train, y_train,
                                          train["single_line_id"])
    w_model = p26.make_ridge(alpha_hat).fit(X_train, y_train)

    manifest = manifest.merge(spectral[["dataset_index", "p_8_16"]],
                              on="dataset_index")
    manifest["log10_tau"] = np.log10(manifest["pulse_duration_fs"]
                                     .astype(float))
    manifest["W_hat_um"] = w_model.predict(
        manifest[WHAT_FEATURES].to_numpy(dtype=float))
    manifest["in_box"] = p26.in_box_mask(manifest, cfg["bridge"]["box"])
    lookup = line_frame.set_index(
        [line_frame["pulse_duration_fs"], line_frame["frequency_kHz"],
         line_frame["pass_count"], line_frame["velocity_mm_s"]])[
        "median_W50_um"]
    key = pd.MultiIndex.from_arrays([
        manifest["pulse_duration_fs"], manifest["frequency_kHz"],
        manifest["pass_count"], manifest["velocity_mm_s"]])
    # pandas-3-proof exact-match lookup: isin + reindex (Series.get(tuple)
    # silently returned NaN for some keys -- audit run lost 6 of 19 conditions)
    hit_mask = key.isin(lookup.index)
    manifest["W_hat_lookup_um"] = np.where(
        hit_mask, lookup.reindex(key).to_numpy(dtype=float), np.nan)
    # bridge_coverage = "does this sample's (tau,f,v,N) condition exist in the
    # single-line DOE" — INDEPENDENT of whether that line's width turned out
    # measurable.  A NaN lookup here (line later flagged W_unavailable) must
    # NOT demote the sample to in_box_pred (audit fix: 20 exact-match samples
    # over 19 conditions, of which 5 conditions are W_unavailable downstream).
    manifest["bridge_coverage"] = np.where(
        hit_mask, "exact_match",
        np.where(manifest["in_box"], "in_box_pred", "out_of_box"))
    manifest["eta_h"] = manifest["W_hat_um"] / manifest["hatch_spacing_um"]

    lam = p26.lambda_star_4_32(radial, window_um=(4.0, 32.0), guard=0.10)
    peak = p26.lambda_peak_4_32(radial, window_um=(4.0, 32.0), n_modes_min=20,
                                share_min=0.20)
    centroid = descriptors.set_index("dataset_index")["spectral_centroid_um"]
    match = manifest.merge(lam, on="dataset_index").merge(peak,
                                                          on="dataset_index")
    match["lambda_star_centroid_um"] = match["dataset_index"].map(centroid)
    match["r_W"] = match["lambda_star_4_32_um"] / match["W_hat_um"]
    match["r_h"] = match["lambda_star_4_32_um"] / match["hatch_spacing_um"]
    match["r_h_peak"] = match["lambda_peak_4_32_um"] / match["hatch_spacing_um"]
    match["d_int"] = [min(abs(v - m) for m in (1, 2, 3)) for v in match["r_h"]]
    match["d_int_peak"] = [min(abs(v - m) for m in (1, 2, 3))
                           for v in match["r_h_peak"].fillna(np.nan)]
    match["abs_lambda_star_minus_What_um"] = (match["lambda_star_4_32_um"]
                                              - match["W_hat_um"]).abs()
    match["abs_lambda_star_minus_h_um"] = (match["lambda_star_4_32_um"]
                                           - match["hatch_spacing_um"]).abs()
    match["abs_lambda_star_minus_2h_um"] = (match["lambda_star_4_32_um"]
                                            - 2 * match["hatch_spacing_um"]).abs()
    match.to_csv(scale / "morphology_scale_match.csv", index=False,
                 encoding="utf-8-sig")
    p26.require(set(match["bridge_coverage"].unique())
                <= set(cfg["bridge"]["coverage_states"]),
                "bridge_coverage states drifted")
    p26.log(f"W_hat bridge: coverage "
            f"{match['bridge_coverage'].value_counts().to_dict()} | in_box "
            f"{int(match['in_box'].sum())}")

    # ---- step 1: SL-03a direct bridge ------------------------------------ #
    direct = direct_bridge(match, line_frame)
    valid = direct[direct["W_source"] == "estimable"]
    unavailable = direct[direct["W_source"] == "W_unavailable"]
    missing_rate = float(len(unavailable) / len(direct))
    stats = {
        "n_conditions_total": int(len(direct)),
        "n_conditions_estimable": int(len(valid)),
        "n_conditions_W_unavailable": int(len(unavailable)),
        "n_conditions_rejected_by_qa": int(
            (direct["W_source"] == "rejected_by_qa").sum()),
        "missing_rate_W_unavailable": missing_rate,
        "median_r_W_direct": (float(valid["r_W_direct"].median())
                              if len(valid) else np.nan),
        "iqr_r_W_direct": ([float(valid["r_W_direct"].quantile(0.25)),
                            float(valid["r_W_direct"].quantile(0.75))]
                           if len(valid) else [np.nan, np.nan]),
        "p_abs_r_minus_1_le_0.25": (float(
            (valid["r_W_direct"].sub(1).abs() <= 0.25).mean())
            if len(valid) else np.nan),
        "spearman_lambda_vs_W_measured": (float(
            spearmanr(valid["lambda_star_mean_um"],
                      valid["W_line_measured_um"]).statistic)
            if len(valid) > 2 else np.nan),
        "evidence_priority": ["exact_match_direct", "in_box_predicted",
                              "out_of_box"],
        "missing_rate_note": "audit amendment: the direct arm must report the "
                             "26%-class W_unavailable missing rate next to "
                             "its weight, never n=13 alone",
    }
    direct.to_csv(scale / "direct_bridge_exact_match.csv", index=False,
                  encoding="utf-8-sig")
    p26.log(f"SL-03a direct bridge: {len(valid)}/{len(direct)} estimable "
            f"conditions (missing {missing_rate:.0%}) | median r_W_direct="
            f"{stats['median_r_W_direct']:.3f} | "
            f"Spearman={stats['spearman_lambda_vs_W_measured']:.3f}")

    # ---- step 3: model comparison on the in-box 101 subset --------------- #
    inbox = match[match["in_box"]].copy()
    p26.require(len(inbox) == 101, f"in-box subset {len(inbox)} != 101 (T11)")
    targets = {**targets_full}
    lam_valid = lam.set_index("dataset_index")["lambda_star_4_32_um"]
    targets["lambda_star_4_32"] = lam_valid

    def subset_targets(frame: pd.DataFrame) -> dict:
        out = {}
        for name, series in targets.items():
            # uniform alignment: every target indexed by the frame's
            # dataset_index in frame order (the ilr_z1_z4 intersection variant
            # silently reordered rows and misaligned y vs X -- audit fix)
            out[name] = series.reindex(frame["dataset_index"])
        return out

    inbox_targets = subset_targets(inbox)
    src_splits = p26.p2.gkf_splits(inbox["shared_height_source_id"], 5)
    p26.p2.check_gkf_contract(inbox["shared_height_source_id"], src_splits)
    proc_splits = p26.p2.gkf_splits(inbox["cv_process_group"], 5)
    p26.p2.check_gkf_contract(inbox["cv_process_group"], proc_splits)
    folds_src, oof_src = run_model_matrix(inbox, src_splits, inbox_targets,
                                          variant="src_gkf", seed=seed)
    folds_proc, oof_proc = run_model_matrix(inbox, proc_splits, inbox_targets,
                                            variant="proc_gkf", seed=seed)
    folds = pd.concat([folds_src, folds_proc], ignore_index=True)
    folds.to_csv(compare / "width_bridge_cv.csv", index=False,
                 encoding="utf-8-sig")
    pd.concat([oof_src, oof_proc], ignore_index=True).to_csv(
        compare / "oof_predictions.csv", index=False, encoding="utf-8-sig")

    # retention + G-SL3 (Geometry-compression Gate, §0.13) — 审计 P1-4 修复：
    # composition 臂真实运行，verdict 按冻结门槛在此判定（含 proc 降级与
    # 折级门槛），不再留给 gate 汇总"拼装"。
    retention_rows = []
    for (target, metric), src in folds[folds["variant"] == "src_gkf"].groupby(
            ["target", "metric"]):
        m0 = src[src["model"] == "M0_u"].set_index("fold")["score"]
        geo = src[src["model"] == "M_GEO_What_h_eta"].set_index("fold")["score"]
        m0_perf = float(m0.median())
        if m0_perf < 0.10:
            retention_rows.append({"target": target, "metric": metric,
                                   "m0_median": m0_perf,
                                   "retention_median": np.nan,
                                   "status": "retention_undefined"})
            continue
        ratios = (geo / m0).dropna()
        retention_rows.append({"target": target, "metric": metric,
                               "m0_median": m0_perf,
                               "retention_median": float(ratios.median()),
                               "fold_min_ratio": float(ratios.min()),
                               "n_folds_ge_0.60": int((ratios >= 0.60).sum())})
    retention = pd.DataFrame(retention_rows)
    comp = retention[(retention["target"] == "ilr_z1_z4")
                     & (retention["metric"] == "Q2_Aitchison")]
    scalar_primary = retention[retention["target"].isin(
        ["p_8_16", "A2_8_16", "angular_entropy_8_16"])]
    comp_ret = (float(comp["retention_median"].iloc[0])
                if len(comp) and pd.notna(comp["retention_median"].iloc[0])
                else np.nan)
    scalar_ret = (float(scalar_primary["retention_median"].median())
                  if len(scalar_primary) else np.nan)
    proc_comp = folds[(folds["variant"] == "proc_gkf")
                      & (folds["target"] == "ilr_z1_z4")
                      & (folds["model"] == "M_GEO_What_h_eta")]["score"].median()
    proc_m0 = folds[(folds["variant"] == "proc_gkf")
                    & (folds["target"] == "ilr_z1_z4")
                    & (folds["model"] == "M0_u")]["score"].median()
    proc_comp_ret = (float(proc_comp / proc_m0) if proc_m0 >= 0.10 else np.nan)
    n_folds_ge = 0
    if len(comp) and pd.notna(comp["n_folds_ge_0.60"].iloc[0]):
        n_folds_ge = int(comp["n_folds_ge_0.60"].iloc[0])
    thresholds = {"min_supported": 0.80, "strong_tier": 0.90,
                  "fold_min": 0.60, "min_folds": 4, "proc_min_agree": 0.70,
                  "m0_perf_floor": 0.10}
    strong = bool(comp_ret >= 0.90 and scalar_ret >= 0.90)
    base_supported = (pd.notna(comp_ret) and comp_ret >= 0.80
                      and pd.notna(scalar_ret) and scalar_ret >= 0.80
                      and n_folds_ge >= 4)
    if pd.isna(comp_ret) or pd.isna(scalar_ret):
        verdict = "retention_undefined"
    elif base_supported and strong:
        verdict = "SUPPORTED(strong)"
    elif base_supported:
        verdict = ("SUPPORTED" if (pd.isna(proc_comp_ret)
                                   or proc_comp_ret >= 0.70) else "PARTIAL")
    else:
        verdict = "NOT_SUPPORTED"
    gsl3 = {
        "type": "geometry_compression",
        "population": "in_box_101",
        "retention_composition_Q2": comp_ret,
        "retention_scalar_median": scalar_ret,
        "retention_composition_proc_gkf": proc_comp_ret,
        "n_folds_retention_ge_0.60_composition": n_folds_ge,
        "thresholds": thresholds,
        "G_SL3": verdict,
        "reading": "五维工艺关系可压缩为单轨宽度–hatch overlap 几何（仅在 "
                   "SUPPORTED/strong 时成立；Δ(M1−M0) 属 LOW-CAPACITY "
                   "REPRESENTATION GAIN，非机制证据）",
    }
    (summary_dir / "gsl3_evaluation.json").write_text(
        json.dumps(gsl3, ensure_ascii=False, indent=2), encoding="utf-8")
    p26.log(f"G-SL3 = {verdict} | composition retention={comp_ret:.3f} | "
            f"scalar median retention={scalar_ret:.3f} | proc retention="
            f"{proc_comp_ret:.3f}")

    # ---- sensitivity arms (§6.3) ----------------------------------------- #
    sens_rows = []
    full200_targets = subset_targets(match)
    folds_full, _ = run_model_matrix(match, full_splits, full200_targets,
                                     variant="full200_extrapolated", seed=seed)
    sens_rows.append(folds_full)
    if not quick:
        sens_rows.append(run_spline_arm(inbox, src_splits, inbox_targets,
                                        variant="spline_src_gkf"))
    sens = pd.concat(sens_rows, ignore_index=True)
    sens.to_csv(compare / "overlap_bridge_cv.csv", index=False,
                encoding="utf-8-sig")

    # ---- SL-03a science figure (QA complete) ------------------------------ #
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    v = direct[direct["W_source"] == "estimable"]
    ax.scatter(v["W_line_measured_um"], v["lambda_star_mean_um"], s=26,
               color="k")
    lim = [0, max(v["W_line_measured_um"].max(), v["lambda_star_mean_um"].max()) * 1.1]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.axvspan(8, 16, color="tab:blue", alpha=0.12)
    ax.axhspan(8, 16, color="tab:blue", alpha=0.12)
    ax.set_xlabel("W_line_measured (um, single-line)")
    ax.set_ylabel("lambda* (um, rectangle 4-32 centroid)")
    ax.set_title(f"direct bridge, {len(v)} conditions | "
                 f"median r={stats['median_r_W_direct']:.2f}")
    fig.tight_layout()
    fig.savefig(scale / "direct_bridge_exact_match.png", dpi=130)
    plt.close(fig)

    p26.log("Task 18 done")
    return 0


def run_spline_arm(frame: pd.DataFrame, splits, targets: dict[str, pd.Series],
                   *, variant: str) -> pd.DataFrame:
    rows = []
    u = frame[U_COLUMNS].to_numpy(dtype=float)
    w_hat = frame["W_hat_um"].to_numpy(dtype=float)
    hatch = frame["hatch_spacing_um"].to_numpy(dtype=float)
    eta = w_hat / hatch
    for model_name, X in (("M0_u", u), ("M_GEO_What_h_eta",
                                        np.column_stack([w_hat, hatch, eta]))):
        for target, y in targets.items():
            y_all = y.to_numpy(dtype=float)
            valid = (np.isfinite(y_all).all(axis=1) if y_all.ndim > 1
                     else np.isfinite(y_all))
            for fold, (tr, te) in enumerate(splits):
                tr = np.array([i for i in tr if valid[i]])
                te = np.array([i for i in te if valid[i]])
                model = make_spline().fit(X[tr], y_all[tr])
                pred = model.predict(X[te])
                rows.append({"variant": variant, "fold": fold,
                             "model": model_name, "target": target,
                             "metric": "R2",
                             "score": float(r2_score(y_all[te], pred)),
                             "n_train": int(len(tr)), "n_test": int(len(te)),
                             "alpha": np.nan})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    raise SystemExit(main())
