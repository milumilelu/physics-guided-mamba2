"""D1.1: block-conditional and residual depth analysis on frozen MAIN180.

This is an additive follow-up to the frozen D0/D1 release.  It keeps the E00
outer and inner component partitions and reports, for each morphology block,
whether adding that block to log-process inputs improves depth prediction.
It also fits a cross-fitted U residual model (D - D_hat_U), a small explicit
U-by-morphology interaction model, and an ExtraTrees sensitivity baseline.

All standardisation and GAM spline bases are fitted inside each training fold.
The residual target is built from inner-fold out-of-fold U predictions, so a
row's own depth is never used to produce its residual target.  SUPP20 remains
outside every fit.  These are predictive, descriptive analyses only.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits

from src.task_state_learning.artifacts import ROOT, sha256
from src.task_state_learning.diagnostics import Regressor, group_weights
from src.task_state_learning.grouping import require
from src.depth_target import features as dt_features
from src.depth_target import models as dt_models
from src.depth_target import runs as dt_runs
from src.depth_target.runs import write_json
from src.depth_target import splits as dt_splits


BLOCKS = ["A", "P", "PE", "G", "T"]
ALPHA_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
SEED = 20260911


def _u(train):
    u = train[dt_features.U_COLUMNS].to_numpy(float)
    require((u > 0).all(), "Nonpositive process input")
    return np.log(u)


def _x(frame, feats, spec):
    """Return deterministic design columns; learned transforms are in Regressor."""
    u = _u(frame)
    if spec == "U":
        return u
    if spec.startswith("U+"):
        block = spec.split("+", 1)[1]
        return np.column_stack([u, feats[dt_features.BLOCKS[block]].to_numpy(float)])
    if spec == "INTERACTION":
        m = feats[dt_features.ALL_MORPHOLOGY].to_numpy(float)
        return np.column_stack([u, m, (u[:, :, None] * m[:, None, :]).reshape(len(frame), -1)])
    if spec == "UM":
        return np.column_stack([u, feats[dt_features.ALL_MORPHOLOGY].to_numpy(float)])
    if spec == "M":
        return feats[dt_features.ALL_MORPHOLOGY].to_numpy(float)
    raise ValueError(spec)


def _select_alpha(train, inner, feats, spec, family, grid=ALPHA_GRID, target=None):
    """Select alpha using only the frozen inner partitions of one outer fold."""
    y = train.D.to_numpy(float) if target is None else np.asarray(target, float)
    choices = []
    for alpha in grid:
        losses = []
        for k in sorted(inner.inner_fold.unique()):
            val_ids = inner.loc[inner.inner_fold == k, "dataset_index"]
            is_val = train.dataset_index.isin(val_ids).to_numpy()
            a, b = train.loc[~is_val], train.loc[is_val]
            ya, yb = y[~is_val], y[is_val]
            est = Regressor(family, float(alpha)).fit(
                _x(a, feats.loc[~is_val], spec), ya, group_weights(a))
            pred = est.predict(_x(b, feats.loc[is_val], spec))
            err = pd.Series((yb - pred) ** 2)
            losses.append(float(err.groupby(b.component_id.to_numpy()).mean().mean()))
        choices.append((float(np.mean(losses)), float(alpha)))
    return min(choices)


def _metric_rows(oof):
    rows = []
    for (model_id, family), g in oof.groupby(["model_id", "family"], sort=True):
        w = group_weights(g)
        err = (g.predicted_D - g.observed_D).to_numpy(float)
        nullerr = (g.train_null_D - g.observed_D).to_numpy(float)
        denom = float(w @ (nullerr ** 2))
        rows.append({
            "model_id": model_id, "family": family, "MAE_um": float(w @ np.abs(err)),
            "RMSE_um": float(np.sqrt(w @ (err ** 2))),
            "component_balanced_Q2": 1.0 - float(w @ (err ** 2)) / denom,
            "risk_MSE_um2": float(w @ (err ** 2)), "coverage": int(len(g)),
            "n_components": int(g.component_id.nunique()),
        })
    return pd.DataFrame(rows)


def _paired_deltas(metrics):
    """Paired summary against BU within model family (risk units are um^2)."""
    ref = metrics[metrics.model_id == "BU"].set_index("family")
    rows = []
    for row in metrics.itertuples(index=False):
        if row.model_id == "BU" or row.family not in ref.index:
            continue
        base = ref.loc[row.family]
        delta_risk = float(row.risk_MSE_um2 - base.risk_MSE_um2)
        rows.append({"model_id": row.model_id, "family": row.family,
                     "delta_MAE_um_vs_BU": float(row.MAE_um - base.MAE_um),
                     "delta_RMSE_um_vs_BU": float(row.RMSE_um - base.RMSE_um),
                     "delta_risk_MSE_um2_vs_BU": delta_risk,
                     "interpretation": ("negative favors this model" if delta_risk < 0
                                         else "positive favors BU" if delta_risk > 0
                                         else "tie")})
    return pd.DataFrame(rows)


def _role_metrics(oof):
    rows = []
    for (model_id, family, role), g in oof.groupby(["model_id", "family", "session_role"], sort=True):
        err = (g.predicted_D - g.observed_D).to_numpy(float)
        rows.append({"model_id": model_id, "family": family, "session_role": role,
                     "MAE_um": float(np.mean(np.abs(err))), "RMSE_um": float(np.sqrt(np.mean(err ** 2))),
                     "n_rows": int(len(g)), "n_components": int(g.component_id.nunique())})
    return pd.DataFrame(rows)


def _bootstrap_deltas(oof, n_boot=5000):
    """Fixed-OOF component bootstrap for paired risk differences (descriptive)."""
    rows = []
    rng = np.random.default_rng(SEED)
    candidates = [(m, f, "BU", f) for m, f in
                  oof[["model_id", "family"]].drop_duplicates().itertuples(index=False, name=None)
                  if m != "BU" and f in {"Ridge", "GAM"}]
    # Tree has no same-family U baseline; include both standard BU references
    # but label these cross-family comparisons exploratory.
    candidates += [("UTREE", "ExtraTrees", "BU", "GAM"), ("UTREE", "ExtraTrees", "BU", "Ridge")]
    for model, family, ref_model, ref_family in candidates:
        a = oof[(oof.model_id == model) & (oof.family == family)]
        b = oof[(oof.model_id == ref_model) & (oof.family == ref_family)]
        if a.empty or b.empty:
            continue
        aa = a.set_index(["component_id", "dataset_index"])
        bb = b.set_index(["component_id", "dataset_index"])
        common = aa.index.intersection(bb.index)
        aa, bb = aa.loc[common], bb.loc[common]
        da = (aa.predicted_D.to_numpy(float) - aa.observed_D.to_numpy(float)) ** 2
        db = (bb.predicted_D.to_numpy(float) - bb.observed_D.to_numpy(float)) ** 2
        comp = aa.reset_index().component_id.to_numpy()
        vals = pd.DataFrame({"component_id": comp, "delta": da - db}).groupby("component_id").delta.mean()
        draws = vals.to_numpy()[rng.integers(len(vals), size=(n_boot, len(vals)))].mean(axis=1)
        rows.append({"model_id": model, "family": family, "reference_model": ref_model,
                     "reference_family": ref_family, "mean_delta_risk_MSE_um2": float(vals.mean()),
                     "ci95_low": float(np.quantile(draws, 0.025)),
                     "ci95_high": float(np.quantile(draws, 0.975)), "n_components": int(len(vals)),
                     "scope": "fixed OOF component bootstrap; descriptive"})
    return pd.DataFrame(rows)


def _fit_residual(train, test, train_feats, test_feats, inner, family, alpha_u, alpha_r):
    """Fit residual D|U using inner-fold OOF U predictions on outer training rows."""
    u_oof = np.full(len(train), np.nan)
    for k in sorted(inner.inner_fold.unique()):
        is_val = train.dataset_index.isin(inner.loc[inner.inner_fold == k, "dataset_index"]).to_numpy()
        a, b = train.loc[~is_val], train.loc[is_val]
        est = Regressor(family, float(alpha_u)).fit(_x(a, train_feats.loc[~is_val], "U"), a.D, group_weights(a))
        u_oof[is_val] = est.predict(_x(b, train_feats.loc[is_val], "U"))
    require(np.isfinite(u_oof).all(), "Incomplete inner OOF U predictions")
    residual = train.D.to_numpy(float) - u_oof
    # Select residual alpha with a strict second layer: for each residual
    # validation fold k, every residual target in its fit set is generated by
    # a BU model that also excludes k.  Thus validation depths cannot enter
    # either the residual target or the residual fit through the BU stage.
    _, alpha_r_selected = _select_residual_alpha(train, train_feats, inner, family, alpha_u)
    alpha_r = alpha_r_selected if alpha_r is None else alpha_r
    est_u = Regressor(family, float(alpha_u)).fit(_x(train, train_feats, "U"), train.D, group_weights(train))
    est_r = Regressor(family, float(alpha_r)).fit(_x(train, train_feats, "M"), residual, group_weights(train))
    pred_u = est_u.predict(_x(test, test_feats, "U"))
    pred = pred_u + est_r.predict(_x(test, test_feats, "M"))
    return pred, pred_u, float(alpha_r), u_oof


def _select_residual_alpha(train, feats, inner, family, alpha_u, grid=ALPHA_GRID):
    choices = []
    folds = sorted(inner.inner_fold.unique())
    y = train.D.to_numpy(float)
    labels = np.full(len(train), -1, dtype=int)
    for k in folds:
        ids = inner.loc[inner.inner_fold == k, "dataset_index"]
        labels[train.dataset_index.isin(ids).to_numpy()] = int(k)
    require((labels >= 0).all(), "Incomplete inner residual labels")
    for alpha in grid:
        losses = []
        for k in folds:
            val = labels == k
            fit_res = np.full(len(train), np.nan)
            # Build leakage-safe residual targets for folds j != k.  Every BU
            # fit excludes both j (the row being predicted) and k (held-out
            # residual validation rows).
            for j in folds:
                if j == k:
                    continue
                jmask = labels == j
                umask = (~val) & (~jmask)
                est_u = Regressor(family, float(alpha_u)).fit(
                    _x(train.loc[umask], feats.loc[umask], "U"), y[umask], group_weights(train.loc[umask]))
                fit_res[jmask] = y[jmask] - est_u.predict(_x(train.loc[jmask], feats.loc[jmask], "U"))
            require(np.isfinite(fit_res[~val]).all(), "Incomplete nested residual fit targets")
            est = Regressor(family, float(alpha)).fit(
                _x(train.loc[~val], feats.loc[~val], "M"), fit_res[~val], group_weights(train.loc[~val]))
            # Validation residual uses a BU fit on the k-complement only.
            est_u_val = Regressor(family, float(alpha_u)).fit(
                _x(train.loc[~val], feats.loc[~val], "U"), y[~val], group_weights(train.loc[~val]))
            val_res = y[val] - est_u_val.predict(_x(train.loc[val], feats.loc[val], "U"))
            pred = est.predict(_x(train.loc[val], feats.loc[val], "M"))
            e = pd.Series((val_res - pred) ** 2)
            losses.append(float(e.groupby(train.loc[val, "component_id"].to_numpy()).mean().mean()))
        choices.append((float(np.mean(losses)), float(alpha)))
    return min(choices)


def run(config, frame, inner, features):
    records, tuning = [], []
    specs = [f"U+{b}" for b in BLOCKS] + ["INTERACTION"]
    # Keep Ridge and GAM for the block matrix; explicit interaction and tree
    # models are intentionally compact sensitivity baselines.
    for fold in sorted(frame.outer_fold.unique()):
        is_test = (frame.outer_fold == fold).to_numpy()
        train, test = frame.loc[~is_test].reset_index(drop=True), frame.loc[is_test].reset_index(drop=True)
        ftr, fte = features.loc[~is_test].reset_index(drop=True), features.loc[is_test].reset_index(drop=True)
        inner_f = inner.loc[inner.outer_fold == fold].reset_index(drop=True)
        null = float(group_weights(train) @ train.D.to_numpy(float))
        # U baseline is retained as the paired reference for every block.
        u_alpha = {}
        for family in ["Ridge", "GAM"]:
            _, au = _select_alpha(train, inner_f, ftr, "U", family)
            u_alpha[family] = au
            est = Regressor(family, au).fit(_x(train, ftr, "U"), train.D, group_weights(train))
            pred = est.predict(_x(test, fte, "U"))
            for pos, row in enumerate(test.itertuples(index=False)):
                records.append(_record(fold, row, pred[pos], null, "BU", family, au))
        for spec in specs:
            family_list = ["Ridge", "GAM"] if spec != "INTERACTION" else ["Ridge"]
            for family in family_list:
                _, alpha = _select_alpha(train, inner_f, ftr, spec, family)
                tuning.append({"outer_fold": fold, "model_id": spec, "family": family,
                               "selected_alpha": alpha, "selection_scope": "frozen inner"})
                est = Regressor(family, alpha).fit(_x(train, ftr, spec), train.D, group_weights(train))
                pred = est.predict(_x(test, fte, spec))
                for pos, row in enumerate(test.itertuples(index=False)):
                    records.append(_record(fold, row, pred[pos], null, spec, family, alpha))
        for family in ["Ridge", "GAM"]:
            pred, pred_u, alpha_r, u_oof = _fit_residual(
                train, test, ftr, fte, inner_f, family, u_alpha[family], None)
            tuning.append({"outer_fold": fold, "model_id": "RESIDUAL_M|U", "family": family,
                           "selected_alpha": alpha_r, "alpha_u": u_alpha[family],
                           "selection_scope": "cross-fitted U residual + frozen inner"})
            for pos, row in enumerate(test.itertuples(index=False)):
                records.append(_record(fold, row, pred[pos], null, "RESIDUAL_M|U", family, alpha_r))
        # Tree baseline: fold-local fit, component weights, fixed small budget.
        xt = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, max_features=1.0,
                                 random_state=SEED + int(fold), n_jobs=1)
        xt.fit(_x(train, ftr, "UM"), train.D.to_numpy(float),
               sample_weight=group_weights(train) * len(train))
        pred = xt.predict(_x(test, fte, "UM"))
        for pos, row in enumerate(test.itertuples(index=False)):
            records.append(_record(fold, row, pred[pos], null, "UTREE", "ExtraTrees", np.nan))
        print(f"D1.1 outer fold {fold + 1}/5 completed", flush=True)
    oof = pd.DataFrame(records)
    require(not oof.duplicated(["model_id", "family", "dataset_index"]).any(), "OOF duplicates")
    require(len(oof.dataset_index.unique()) == len(frame), "Dataset OOF coverage mismatch")
    return oof, _metric_rows(oof), pd.DataFrame(tuning)


def _record(fold, row, pred, null, model_id, family, alpha):
    return {"outer_fold": fold, "dataset_index": int(row.dataset_index), "session_id": row.session_id,
            "sample_id": int(row.sample_id), "session_role": row.session_role,
            "component_id": row.component_id, "model_id": model_id, "family": family,
            "selected_alpha": alpha, "observed_D": float(row.D), "predicted_D": float(pred),
            "train_null_D": float(null)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract-run", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=ROOT / "config/depth_target_v1/protocol.yaml")
    args = ap.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    contract, seal_report = dt_runs.verify_upstream_seal_tolerant(args.contract_run, dt_runs.RUNS_DIR)
    gate0 = json.loads((contract / "contract_gate.json").read_text(encoding="utf-8"))
    require(gate0["status"] == "PASS", "D0 contract not PASS")
    binding_report = dt_runs.verify_contract_binding(contract, config)
    out = dt_runs.new_run("D1_1_block_residual", args.config)
    write_json(out / "contract_seal_report.json", seal_report)
    write_json(out / "input_binding_report.json", binding_report)
    t0 = time.perf_counter()
    gate = {"experiment_id": "D1_1_block_residual", "execution_status": "FAIL",
            "training_executed": False, "deep_training_executed": False}
    try:
        suite = subprocess.run([sys.executable, "-X", "utf8", "-B", "-m", "unittest",
                                "discover", "-s", "tests/depth_target_v1", "-v"],
                               cwd=ROOT, capture_output=True, encoding="utf-8")
        (out / "tests.log").write_text(suite.stdout + suite.stderr, encoding="utf-8", newline="\n")
        require(suite.returncode == 0, "tests/depth_target_v1 failed")
        audit = ROOT / config["inputs"]["audit_run"]
        frame, inner = dt_splits.load_frozen_partition(audit)
        frame = frame.sort_values("dataset_index").reset_index(drop=True)
        targets = pd.read_csv(audit / "frozen_targets.csv")
        pack = dict(np.load(ROOT / config["inputs"]["height_package"], allow_pickle=False))
        rows = dt_features.align_height_rows(targets, pack)
        main_idx = rows[frame.dataset_index.to_numpy()]
        features = dt_features.build_feature_frame(frame, pack["height_raw"][main_idx],
                                                    pack["valid_mask"][main_idx], config["inputs"]["pixel_um"])
        with threadpool_limits(limits=1):
            gate["training_executed"] = True
            oof, metrics, tuning = run(config, frame, inner, features)
        for table in (oof,):
            table["split_sha"] = sha256(audit / "split_manifest.csv")
            table["data_sha"] = sha256(audit / "frozen_targets.csv")
        oof.to_csv(out / "oof_depth_predictions.csv", index=False, lineterminator="\n")
        metrics.to_csv(out / "d1_1_metrics.csv", index=False, lineterminator="\n")
        _paired_deltas(metrics).to_csv(out / "paired_deltas_vs_BU.csv", index=False, lineterminator="\n")
        _bootstrap_deltas(oof).to_csv(out / "paired_bootstrap_intervals.csv", index=False, lineterminator="\n")
        _role_metrics(oof).to_csv(out / "per_role_metrics.csv", index=False, lineterminator="\n")
        (out / "d1_1_scope.md").write_text(
            "# D1.1 scope\n\n"
            "Frozen MAIN180 only: E00 outer and inner component partitions are reused after D0 binding. "
            "SUPP20 is excluded from every fit and is not an OOF claim.\n\n"
            "Models are BU, U+{A,P,PE,G,T}, explicit U-by-morphology interaction Ridge, "
            "cross-fitted residual M|U (Ridge/GAM), and a fixed-budget ExtraTrees sensitivity baseline "
            "on raw log-U plus all morphology columns. "
            "U inputs are log transformed deterministically; Regressor scaling and GAM splines are fit "
            "inside each training fold. Residual targets use strict nested inner cross-fitting.\n\n"
            "Metrics are component-balanced OOF MAE/RMSE/Q2 and fixed-OOF component bootstrap risk deltas. "
            "All results are predictive/descriptive; they do not identify mediation or causality. "
            "ExtraTrees uses a fixed exploratory hyperparameter budget and is not a confirmatory model comparison.\n",
            encoding="utf-8", newline="\n")
        tuning.to_csv(out / "inner_tuning.csv", index=False, lineterminator="\n")
        best = metrics.loc[metrics.MAE_um.idxmin()]
        gate.update(execution_status="PASS", oof_rows=int(len(oof)), n_models=int(metrics.shape[0]),
                    best_model=f"{best.family}/{best.model_id}", best_MAE_um=float(best.MAE_um),
                    scope="MAIN180 frozen E00 grouped OOF; blocks, residual, interaction and tree sensitivity; SUPP20 excluded; descriptive only")
    except Exception as exc:
        gate["error"] = f"{type(exc).__name__}: {exc}"
    gate["status"] = gate["execution_status"]
    gate["walltime_seconds"] = time.perf_counter() - t0
    write_json(out / "d1_1_gate.json", gate)
    dt_runs.seal(out)
    print(json.dumps({"output": str(out), **gate}, ensure_ascii=False, indent=2))
    return 0 if gate["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
