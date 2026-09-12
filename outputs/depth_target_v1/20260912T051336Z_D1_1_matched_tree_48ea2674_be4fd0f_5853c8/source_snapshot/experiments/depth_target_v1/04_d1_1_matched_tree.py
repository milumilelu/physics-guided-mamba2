"""D1.1 matched ExtraTrees controls on frozen MAIN180.

Fits the same fold-local ExtraTrees estimator and hyperparameter budget to five
feature sets: process coordinates U, morphology block P, all morphology M,
U+P, and U+M.  All fits reuse the sealed E00 outer/inner partitions and
component-balanced sample weights.  This is a predictive/descriptive control;
it does not identify a causal effect of morphology.
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
from src.task_state_learning.grouping import require
from src.task_state_learning.diagnostics import group_weights
from src.depth_target import features as dt_features
from src.depth_target import runs as dt_runs
from src.depth_target import splits as dt_splits
from src.depth_target.runs import write_json

SEED = 20260911
TREE_PARAMS = {"n_estimators": 300, "min_samples_leaf": 3,
               "max_features": 1.0, "n_jobs": 1}


def _x(frame, feats, spec):
    u = np.log(frame[dt_features.U_COLUMNS].to_numpy(float))
    require(np.isfinite(u).all(), "Nonfinite process input")
    if spec == "U":
        return u
    if spec == "P":
        return feats[dt_features.BLOCKS["P"]].to_numpy(float)
    if spec == "M":
        return feats[dt_features.ALL_MORPHOLOGY].to_numpy(float)
    if spec == "U+P":
        return np.column_stack([u, feats[dt_features.BLOCKS["P"]].to_numpy(float)])
    if spec == "U+M":
        return np.column_stack([u, feats[dt_features.ALL_MORPHOLOGY].to_numpy(float)])
    raise ValueError(spec)


def _record(fold, row, pred, null, model_id):
    return {"outer_fold": int(fold), "dataset_index": int(row.dataset_index),
            "session_id": row.session_id, "sample_id": int(row.sample_id),
            "session_role": row.session_role, "component_id": row.component_id,
            "model_id": model_id, "family": "ExtraTrees", "selected_alpha": np.nan,
            "observed_D": float(row.D), "predicted_D": float(pred), "train_null_D": float(null)}


def _metrics(oof):
    rows = []
    for model, g in oof.groupby("model_id", sort=True):
        w = g.groupby("component_id").size().rdiv(1.0).reindex(g.component_id).to_numpy()
        # group_weights is equivalent but avoid depending on row ordering here
        counts = g.component_id.value_counts()
        w = g.component_id.map(lambda c: 1.0 / counts[c]).to_numpy(dtype=float, copy=True)
        w /= w.sum()
        err = (g.predicted_D - g.observed_D).to_numpy(float)
        nullerr = (g.train_null_D - g.observed_D).to_numpy(float)
        denom = float(w @ (nullerr ** 2))
        rows.append({"model_id": model, "family": "ExtraTrees",
                     "MAE_um": float(w @ np.abs(err)),
                     "RMSE_um": float(np.sqrt(w @ (err ** 2))),
                     "component_balanced_Q2": float(1.0 - (w @ (err ** 2)) / denom),
                     "risk_MSE_um2": float(w @ (err ** 2)), "coverage": int(len(g)),
                     "n_components": int(g.component_id.nunique())})
    return pd.DataFrame(rows)


def _bootstrap(oof, n_boot=5000):
    models = sorted(oof.model_id.unique())
    ref = "Tree(U+M)"
    rng = np.random.default_rng(SEED)
    rows = []
    for model in models:
        if model == ref:
            continue
        a = oof[oof.model_id == model].set_index(["component_id", "dataset_index"])
        b = oof[oof.model_id == ref].set_index(["component_id", "dataset_index"])
        common = a.index.intersection(b.index)
        da = (a.loc[common].predicted_D.to_numpy(float) - a.loc[common].observed_D.to_numpy(float)) ** 2
        db = (b.loc[common].predicted_D.to_numpy(float) - b.loc[common].observed_D.to_numpy(float)) ** 2
        comp = common.get_level_values("component_id").to_numpy()
        vals = pd.DataFrame({"component_id": comp, "delta": da - db}).groupby("component_id").delta.mean()
        draws = vals.to_numpy()[rng.integers(len(vals), size=(n_boot, len(vals)))].mean(axis=1)
        rows.append({"model_id": model, "reference_model": ref,
                     "mean_delta_risk_MSE_um2": float(vals.mean()),
                     "ci95_low": float(np.quantile(draws, .025)),
                     "ci95_high": float(np.quantile(draws, .975)),
                     "n_components": int(len(vals)),
                     "interpretation": "negative favors model; positive favors U+M",
                     "scope": "fixed OOF component bootstrap; descriptive"})
    return pd.DataFrame(rows)


def run(frame, inner, features):
    del inner  # frozen inner manifest is sealed upstream; trees use fixed budget
    records = []
    specs = ["U", "P", "M", "U+P", "U+M"]
    for fold in sorted(frame.outer_fold.unique()):
        is_test = (frame.outer_fold == fold).to_numpy()
        train = frame.loc[~is_test].reset_index(drop=True)
        test = frame.loc[is_test].reset_index(drop=True)
        ftr = features.loc[~is_test].reset_index(drop=True)
        fte = features.loc[is_test].reset_index(drop=True)
        null = float(group_weights(train) @ train.D.to_numpy(float))
        for spec in specs:
            est = ExtraTreesRegressor(random_state=SEED + int(fold), **TREE_PARAMS)
            est.fit(_x(train, ftr, spec), train.D.to_numpy(float),
                    sample_weight=group_weights(train) * len(train))
            pred = est.predict(_x(test, fte, spec))
            for pos, row in enumerate(test.itertuples(index=False)):
                records.append(_record(fold, row, pred[pos], null, f"Tree({spec})"))
        print(f"D1.1 matched-tree outer fold {fold + 1}/5 completed", flush=True)
    oof = pd.DataFrame(records)
    require(not oof.duplicated(["model_id", "dataset_index"]).any(), "OOF duplicates")
    require(len(oof.dataset_index.unique()) == len(frame), "Dataset OOF coverage mismatch")
    return oof, _metrics(oof)


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
    out = dt_runs.new_run("D1_1_matched_tree", args.config)
    write_json(out / "contract_seal_report.json", seal_report)
    write_json(out / "input_binding_report.json", binding_report)
    t0 = time.perf_counter()
    gate = {"experiment_id": "D1_1_matched_tree", "execution_status": "FAIL",
            "training_executed": False, "deep_training_executed": False}
    try:
        suite = subprocess.run([sys.executable, "-X", "utf8", "-B", "-m", "unittest",
                                "discover", "-s", "tests/depth_target_v1", "-v"], cwd=ROOT,
                               capture_output=True, encoding="utf-8")
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
            oof, metrics = run(frame, inner, features)
        for table in (oof,):
            table["split_sha"] = sha256(audit / "split_manifest.csv")
            table["data_sha"] = sha256(audit / "frozen_targets.csv")
        oof.to_csv(out / "oof_matched_tree_predictions.csv", index=False, lineterminator="\n")
        metrics.to_csv(out / "matched_tree_metrics.csv", index=False, lineterminator="\n")
        _bootstrap(oof).to_csv(out / "matched_tree_bootstrap_vs_U+M.csv", index=False, lineterminator="\n")
        scope = ("# D1.1 matched Tree scope\n\n"
                 "Frozen MAIN180 only; E00 outer component partitions are reused and SUPP20 is excluded. "
                 "Tree(U), Tree(P), Tree(M), Tree(U+P), and Tree(U+M) use the identical ExtraTrees budget: "
                 "n_estimators=300, min_samples_leaf=3, max_features=1.0, n_jobs=1; only feature set changes. "
                 "Process coordinates are log transformed; morphology uses median-residual features. "
                 "Metrics are component-balanced OOF errors and fixed-OOF component bootstrap deltas vs Tree(U+M). "
                 "Results are descriptive predictive controls and do not identify causal mediation.\n")
        (out / "matched_tree_scope.md").write_text(scope, encoding="utf-8", newline="\n")
        best = metrics.loc[metrics.MAE_um.idxmin()]
        gate.update(execution_status="PASS", oof_rows=int(len(oof)), n_models=int(metrics.shape[0]),
                    best_model=str(best.model_id), best_MAE_um=float(best.MAE_um),
                    scope="MAIN180 frozen E00 grouped OOF; matched ExtraTrees feature-set controls; SUPP20 excluded; descriptive only")
    except Exception as exc:
        gate["error"] = f"{type(exc).__name__}: {exc}"
    gate["status"] = gate["execution_status"]
    gate["walltime_seconds"] = time.perf_counter() - t0
    write_json(out / "matched_tree_gate.json", gate)
    dt_runs.seal(out)
    print(json.dumps({"output": str(out), **gate}, ensure_ascii=False, indent=2))
    return 0 if gate["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
