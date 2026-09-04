#!/usr/bin/env python3
"""Task SL-02: single-line width process model + G-SL1 evaluation (细则 §5).

Sample sets (frozen §5):
  - G-SL1 gate population AND Ridge training primary = estimable lines whose
    human label != reject_geometry (the 3 estimable-but-rejected lines are
    geometry-untrustworthy and are excluded from both; registered reading).
  - sensitivity arm = primary ∪ uncertain.
  - right_censored / insufficient_sections lines enter no training numbers.

Target: per-line `median_W50_um` (raw primary; `median_W50_um_rep` only for
the raw-repaired divergence check).  Features: u_line = [log10 tau, f, v, N]
(standardized within the training fold).  C-extra (pulse_pitch_um, E_line_J_mm,
E_line_J) are descriptive columns only -- never features (T10 whitelist).

CV: GKF(5) grouped by single_line_id + `check_gkf_contract`; GSS(seed+100)
sensitivity.  Ridge alpha = fold-internal grid logspace(-3, 3, 13) selected by
inner GKF(5) on the train fold.  W_line_distribution_vs_band is generated
because human QA is complete (geometry_qa_labels_provenance.json, 120/120).
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

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import _lib as p26  # noqa: E402

EXPECTED = [
    "outputs/phase2_6/scale_bridge/line_width_process_model.csv",
    "outputs/phase2_6/scale_bridge/width_identifiability_summary.csv",
    "outputs/phase2_6/model_compare/W_line_response_curves.csv",
    "outputs/phase2_6/model_compare/W_line_distribution_vs_band.csv",
    "outputs/phase2_6/model_compare/W_line_distribution_vs_band.png",
    "outputs/phase2_6/summary/gsl1_evaluation.json",
]

FEATURES = ["log10_tau", "frequency_kHz", "velocity_mm_s", "pass_count"]
C_EXTRA = ["pulse_pitch_um", "E_line_J_mm", "E_line_J"]
BAND_LO, BAND_HI = 8.0, 16.0  # frozen half-open [8, 16)


def in_band(values: pd.Series) -> pd.Series:
    v = pd.Series(values, dtype=float)
    return (v >= BAND_LO) & (v < BAND_HI)


def pick_alpha(X_train: np.ndarray, y_train: np.ndarray, groups_train: pd.Series,
               seed: int) -> float:
    """Fold-internal alpha: inner GKF(5) on the train fold, mean MSE (shared
    frozen helper, `_lib.ridge_alpha_inner_gkf`)."""
    return p26.ridge_alpha_inner_gkf(X_train, y_train, groups_train)


def evaluate_folds(model_name: str, frame: pd.DataFrame, splits, *,
                   target: str, seed: int, variant: str) -> list[dict]:
    # explicit variant label (封账修正：frame.attrs 在 df.copy() 链上不可靠，
    # 曾把 primary 数据的 GSS 折误报为另一 variant)
    rows = []
    X = frame[FEATURES].to_numpy(dtype=float)
    y = frame[target].to_numpy(dtype=float)
    groups = frame["single_line_id"]
    for fold, (tr, te) in enumerate(splits):
        alpha = np.nan
        model: object
        if model_name == "ridge":
            alpha = pick_alpha(X[tr], y[tr], groups.iloc[tr], seed)
            model = p26.make_ridge(alpha)
        elif model_name == "spline":
            model = Pipeline([
                ("scale", StandardScaler()),
                ("spline", SplineTransformer(degree=3, n_knots=4,
                                             include_bias=False)),
                ("ridge", Ridge(alpha=1.0))])
        elif model_name == "extratrees":
            model = ExtraTreesRegressor(n_estimators=500, random_state=seed + 700 + fold)
        else:
            raise ValueError(model_name)
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        rows.append({
            "variant": variant,
            "fold": fold, "model": model_name,
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "R2": float(r2_score(y[te], pred)),
            "MAE_um": float(mean_absolute_error(y[te], pred)),
            "alpha": alpha,
        })
    return rows


def main() -> int:
    cfg, quick = p26.load_config(__doc__)
    seed = int(cfg["meta"]["random_seed"])
    single = p26.output_dir(cfg, "single_line")
    scale = p26.output_dir(cfg, "scale_bridge")
    compare = p26.output_dir(cfg, "model_compare")
    summary_dir = p26.output_dir(cfg, "summary")
    p26.log(f"Task 17 start | quick={quick}")

    # labels complete check (T22 precondition)
    formal_single = p26.REPO / "outputs/phase2_6/single_line"  # inputs read from the formal root in every mode
    provenance = json.loads((formal_single / "geometry_qa_labels_provenance.json")
                            .read_text(encoding="utf-8"))
    p26.require(provenance["n_labels"] == 120,
                "human QA must be 120/120 before any vs-band figure (T22)")
    labels = pd.read_csv(formal_single / "geometry_qa_labels.csv", encoding="utf-8-sig")
    p26.require(labels["qa_label"].isin(["usable", "uncertain",
                                         "reject_geometry"]).all(),
                "qa_label values drifted")
    geometry = pd.read_csv(formal_single / "single_line_geometry.csv", encoding="utf-8-sig")
    manifest = pd.read_csv(formal_single / "single_line_manifest.csv", encoding="utf-8-sig")
    p26.require(len(geometry) == 120 and len(manifest) == 120,
                "geometry/manifest must hold 120 rows")

    frame = (geometry.merge(labels[["single_line_id", "qa_label"]],
                            on="single_line_id")
             .merge(manifest[["single_line_id", "pulse_duration_fs",
                              "frequency_kHz", "velocity_mm_s", "pass_count"]
                             + C_EXTRA],
                    on="single_line_id"))
    frame["log10_tau"] = np.log10(frame["pulse_duration_fs"].astype(float))

    # three-state counts + censored-by-process table (§5 outputs)
    state_counts = frame["width_identifiability"].value_counts().to_dict()
    summary = pd.DataFrame({
        "width_identifiability": list(state_counts),
        "n_lines": [int(state_counts[k]) for k in state_counts],
    })
    summary.to_csv(scale / "width_identifiability_summary.csv",
                   index=False, encoding="utf-8-sig")

    gate = frame[(frame["width_identifiability"] == "estimable")
                 & (frame["qa_label"] != "reject_geometry")].copy()
    gate.attrs["variant"] = "primary"
    primary = gate.copy()
    # sensitivity arm (§5): primary ∪ uncertain (frozen).  Frozen rule:
    # `insufficient_sections` rows enter NO training numbers -- their targets
    # are NaN and are dropped before fitting (review P1-5).  With
    # right_censored = 0 the trainable arm coincides with primary; the
    # coincidence is REPORTED rather than hidden.  The arm still gets its OWN
    # regenerated splits (primary indices would misalign rows if they differed).
    sensitivity = frame[((frame["width_identifiability"] == "estimable")
                         & (frame["qa_label"] != "reject_geometry"))
                        | (frame["qa_label"] == "uncertain")].copy()
    n_sensitivity_defined = len(sensitivity)
    sensitivity = sensitivity[np.isfinite(
        sensitivity["median_W50_um"].to_numpy(dtype=float))].copy()
    sensitivity.attrs["variant"] = "sensitivity"
    coincides = (set(sensitivity["single_line_id"])
                 == set(primary["single_line_id"]))
    p26.log(f"populations: gate/primary={len(primary)} | "
            f"sensitivity defined={n_sensitivity_defined} "
            f"trainable={len(sensitivity)} "
            f"(coincides_with_primary={coincides}) | states={state_counts}")

    # T10 whitelist: feature matrix carries ONLY the registered factors
    p26.require(set(FEATURES) <= set(primary.columns),
                "feature columns missing")
    p26.require(not any(column in FEATURES for column in
                        ("median_W50_um", "median_W_eq_um", "CV_W50",
                         "median_max_depth_um")),
                "morphology column leaked into feature whitelist")

    splits = p26.p2.gkf_splits(primary["single_line_id"], 5)
    p26.p2.check_gkf_contract(primary["single_line_id"], splits)
    gss = p26.p2.gss_splits(primary["single_line_id"], 5, test_size=0.25,
                            seed=seed + 100)
    p26.log("CV splits OK: GKF(5) contract + GSS sensitivity")

    rows: list[dict] = []
    rows += evaluate_folds("ridge", primary, splits, target="median_W50_um",
                           seed=seed, variant="primary_gkf")
    rows += evaluate_folds("spline", primary, splits, target="median_W50_um",
                           seed=seed, variant="primary_gkf")
    rows += evaluate_folds("extratrees", primary, splits,
                           target="median_W50_um", seed=seed,
                           variant="primary_gkf")
    sens_splits = p26.p2.gkf_splits(sensitivity["single_line_id"], 5)
    p26.p2.check_gkf_contract(sensitivity["single_line_id"], sens_splits)
    rows += evaluate_folds("ridge", sensitivity, sens_splits,
                           target="median_W50_um", seed=seed,
                           variant="sensitivity_gkf")
    rows += evaluate_folds("ridge", primary, gss, target="median_W50_um",
                           seed=seed, variant="primary_gss")
    # usable-only sensitivity (Phase 2.7 封账补充)：18 条 usable 线的
    # W50/W_eq 一致性对照——uncertain 占多数时的保守下界
    usable_only = primary[primary["qa_label"] == "usable"].copy()
    p26.require(len(usable_only) >= 10,
                f"usable-only arm too small: {len(usable_only)}")
    usable_splits = p26.p2.gkf_splits(usable_only["single_line_id"], 5)
    p26.p2.check_gkf_contract(usable_only["single_line_id"], usable_splits)
    rows += evaluate_folds("ridge", usable_only, usable_splits,
                           target="median_W50_um", seed=seed,
                           variant="usable_only_gkf")
    folds = pd.DataFrame(rows)
    for column in ("R2", "MAE_um"):
        folds[column] = folds[column].astype(float)
    summary_rows = (folds.groupby(["variant", "model"])
                    .agg(n_folds=("fold", "count"),
                         median_R2=("R2", "median"),
                         median_MAE_um=("MAE_um", "median"))
                    .reset_index())
    model_table = pd.concat([folds, summary_rows.assign(fold=np.nan)],
                            ignore_index=True)
    model_table.to_csv(scale / "line_width_process_model.csv",
                       index=False, encoding="utf-8-sig")
    p26.log(f"line_width_process_model.csv written: {len(folds)} fold rows | "
            f"primary Ridge median R2="
            f"{summary_rows[(summary_rows['variant']=='primary_gkf') & (summary_rows['model']=='ridge')]['median_R2'].iloc[0]:.3f}")

    # refit Ridge on the full primary set for response curves (alpha inner-GKF)
    X = primary[FEATURES].to_numpy(dtype=float)
    y = primary["median_W50_um"].to_numpy(dtype=float)
    alpha_full = pick_alpha(X, y, primary["single_line_id"], seed)
    ridge_full = p26.make_ridge(alpha_full).fit(X, y)
    curve_rows = []
    for factor, column in (("log10_tau", "pulse_duration_fs"),
                           ("frequency_kHz", "frequency_kHz"),
                           ("velocity_mm_s", "velocity_mm_s"),
                           ("pass_count", "pass_count")):
        grid = np.linspace(primary[factor].min(), primary[factor].max(), 40)
        base = primary[FEATURES].median()
        curve_X = np.tile(base.to_numpy(dtype=float), (grid.size, 1))
        curve_X[:, FEATURES.index(factor)] = grid
        curve_rows.append(pd.DataFrame({
            "factor": factor,
            "factor_value": grid,
            "factor_native": (10.0 ** grid) if factor == "log10_tau" else grid,
            "W50_hat_um": ridge_full.predict(curve_X),
            "note": ("f (Ep-coupled): P fixed 5.333 W -> f and Ep=P/f not separable"
                     if factor == "frequency_kHz" else "")}))
    curves = pd.concat(curve_rows, ignore_index=True)
    curves.to_csv(compare / "W_line_response_curves.csv",
                  index=False, encoding="utf-8-sig")

    # science figure + data (allowed: QA complete)
    sections = pd.read_csv(formal_single / "cross_section_widths.csv",
                           encoding="utf-8-sig")
    gate_ids = set(primary["single_line_id"])
    pooled = sections[(sections["single_line_id"].isin(gate_ids))
                      & (sections["arm"] == "raw")
                      & (~sections["censored_W50"].astype(bool))
                      & sections["n_above_threshold"] > 0]
    pooled_median = float(pooled["W50_um"].median())
    line_median = float(primary["median_W50_um"].median())
    # c3 = pooled section W_eq median over the gate population (frozen §5),
    # not a median of per-line medians
    pooled_weq_median = float(pooled["W_eq_um"].median())
    rep_median = float(primary["median_W50_um_rep"].median())
    raw_fraction = float(in_band(primary["median_W50_um"]).mean())
    rep_fraction = float(in_band(primary["median_W50_um_rep"]).mean())
    criteria = {
        "c1_pooled_section_median_in_band": bool(in_band(pd.Series([pooled_median])).iloc[0]),
        "c1_pooled_section_median_um": pooled_median,
        "c2_line_median_in_band_fraction": raw_fraction,
        "c2_min_line_fraction": 0.50,
        "c3_pooled_section_weq_median_in_band": bool(in_band(pd.Series([pooled_weq_median])).iloc[0]),
        "c3_pooled_section_weq_median_um": pooled_weq_median,
    }
    met = int(criteria["c1_pooled_section_median_in_band"])
    met += int(criteria["c2_line_median_in_band_fraction"] >= 0.50)
    met += int(criteria["c3_pooled_section_weq_median_in_band"])
    if met == 3:
        verdict = "SUPPORTED" if len(primary) >= 60 else "PARTIAL(n_estimable<60)"
    elif met == 2:
        verdict = "PARTIAL"
    else:
        verdict = "NOT_SUPPORTED"
    divergence = abs(raw_fraction - rep_fraction) > 0.10
    state_counts_all = {state: int((frame["width_identifiability"] == state).sum())
                        for state in ("estimable", "right_censored",
                                      "insufficient_sections")}
    usable_only_block = {
        "n": int(len(usable_only)),
        "pooled_section_median_W50_um": float(
            pooled[pooled["single_line_id"].isin(usable_only["single_line_id"])]
            ["W50_um"].median()),
        "line_median_W50_in_band_fraction": float(
            in_band(usable_only["median_W50_um"]).mean()),
        "pooled_section_weq_median_um": float(
            pooled[pooled["single_line_id"].isin(usable_only["single_line_id"])]
            ["W_eq_um"].median()),
        "line_median_weq_in_band_fraction": float(
            in_band(usable_only["median_W_eq_um"]).mean()),
        "note": "usable-only sensitivity (Phase 2.7 封账)：18 条人工完全可用线",
    }
    evaluation = {
        "population": "estimable & qa_label != reject_geometry",
        "n_gate": int(len(primary)),
        "n_estimable_all": state_counts_all,
        "n_right_censored_all": state_counts_all["right_censored"],
        "n_insufficient_all": state_counts_all["insufficient_sections"],
        "usable_only_sensitivity": usable_only_block,
        "raw_arm": {"pooled_section_median_W50_um": pooled_median,
                    "line_median_W50_um": line_median,
                    "pooled_section_weq_median_um": pooled_weq_median,
                    "line_median_W50_rep_um": rep_median,
                    "band_um": [BAND_LO, BAND_HI],
                    "band_semantics": "[8, 16) half-open",
                    "line_in_band_fraction_raw": raw_fraction,
                    "line_in_band_fraction_repaired": rep_fraction,
                    "raw_repaired_divergent_note": bool(divergence)},
        "criteria": criteria,
        "criteria_met": met,
        "G_SL1": verdict,
        "ridge_full_alpha": alpha_full,
        "censored_by_process_note": "censored sections enter no gate statistic",
    }
    (summary_dir / "gsl1_evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    axes[0].hist(pooled["W50_um"], bins=40, color="grey", edgecolor="k")
    axes[0].axvspan(BAND_LO, BAND_HI, color="tab:blue", alpha=0.15)
    axes[0].set_title(f"pooled section W50 (n={len(pooled)} uncensored)")
    axes[1].scatter(primary["median_W50_um"],
                    primary["median_max_depth_um"], s=18, color="k")
    axes[1].axvspan(BAND_LO, BAND_HI, color="tab:blue", alpha=0.15)
    axes[1].set_xlabel("per-line median W50 (um)")
    axes[1].set_ylabel("per-line median D_max (um)")
    axes[1].set_title(f"per-line width vs depth (n={len(primary)})")
    fig.suptitle(f"W_line distribution vs band [{BAND_LO}, {BAND_HI}) | "
                 f"G-SL1 = {verdict}")
    fig.tight_layout()
    fig.savefig(compare / "W_line_distribution_vs_band.png", dpi=130)
    plt.close(fig)
    band_frame = pooled[["single_line_id", "s_um", "W50_um"]].copy()
    band_frame["in_band"] = in_band(band_frame["W50_um"]).astype(int)
    band_frame["pooled_median_um"] = pooled_median
    band_frame.to_csv(compare / "W_line_distribution_vs_band.csv",
                      index=False, encoding="utf-8-sig")

    p26.log(f"G-SL1 = {verdict} (criteria met {met}/3) | pooled section median "
            f"W50 = {pooled_median:.2f} um | line in-band fraction "
            f"{raw_fraction:.3f} (repaired {rep_fraction:.3f}, divergent="
            f"{divergence}) | pooled section W_eq median "
            f"{pooled_weq_median:.2f} um")
    p26.log("Task 17 done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
