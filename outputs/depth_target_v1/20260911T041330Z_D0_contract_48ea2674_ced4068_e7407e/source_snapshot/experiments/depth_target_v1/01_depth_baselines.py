"""D1: depth-primary minimal experiment matrix (B0/BQ/BU/BA/BP/BPE/BG/BT/BM/BUM).

Answers the first two protocol questions with grouped OOF risk:
  Delta_{M|U} = R(BU) - R(BUM)   (does morphology add depth information beyond process?)
  Delta_{U|M} = R(BM) - R(BUM)   (does process add depth information beyond morphology?)
Intervals: fixed-OOF component bootstrap (descriptive, conditional on fitted models).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json
import subprocess
import time

import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from src.task_state_learning.artifacts import ROOT, sha256, write_json
from src.task_state_learning.diagnostics import group_weights
from src.task_state_learning.grouping import require
from src.depth_target import features as dt_features
from src.depth_target import models as dt_models
from src.depth_target import runs as dt_runs
from src.depth_target import splits as dt_splits

CONTRASTS = [("BU", "BUM", "Delta_{M|U}=R(BU)-R(BUM)"),
             ("BM", "BUM", "Delta_{U|M}=R(BM)-R(BUM)"),
             ("BU", "BM", "R(BU)-R(BM)"),
             ("B0", "BU", "R(B0)-R(BU)"),
             ("B0", "BM", "R(B0)-R(BM)")]


def run_experiment(config, frame, inner, features):
    records, tuning, rowrisk = [], [], []
    for fold in sorted(frame.outer_fold.unique()):
        is_test = (frame.outer_fold == fold).to_numpy()
        train, test = frame[~is_test], frame[is_test]
        ftr, fte = features[~is_test].reset_index(drop=True), features[is_test].reset_index(drop=True)
        inner_table = inner[inner.outer_fold == fold]
        for family in config["models"]["families"]:
            for model_id in config["models"]["depth_models"]:
                inner_risk, alpha = dt_models.select_alpha(
                    train.reset_index(drop=True), inner_table, ftr, model_id, family,
                    config["models"]["alpha_grid"])
                tuning.append({"outer_fold": fold, "model_id": model_id, "family": family,
                               "selected_alpha": alpha, "inner_risk": inner_risk})
                pred, null = dt_models.fit_depth_model(
                    train.reset_index(drop=True), test.reset_index(drop=True),
                    ftr, fte, model_id, family, alpha)
                comp = test.component_id.to_numpy()
                mse_by_row = (test.D.to_numpy(float) - pred) ** 2
                for pos, row in enumerate(test.itertuples(index=False)):
                    records.append({"outer_fold": fold, "dataset_index": int(row.dataset_index),
                                    "session_id": row.session_id, "sample_id": int(row.sample_id),
                                    "session_role": row.session_role, "component_id": row.component_id,
                                    "model_id": model_id, "family": family, "selected_alpha": alpha,
                                    "observed_D": float(row.D), "predicted_D": float(pred[pos]),
                                    "train_null_D": null})
                rowrisk.extend({"outer_fold": fold, "model_id": model_id, "family": family,
                                "component_id": c, "risk": r}
                               for c, r in zip(comp, mse_by_row))
        print(f"D1 outer fold {fold + 1}/5 completed", flush=True)
    oof = pd.DataFrame(records)
    risk = pd.DataFrame(rowrisk)
    require(not oof.duplicated(["model_id", "family", "dataset_index"]).any(), "OOF duplicates")
    expected = len(frame) * len(config["models"]["families"]) * len(config["models"]["depth_models"])
    require(len(oof) == expected, "OOF coverage mismatch")

    metric_rows = []
    for (family, model_id), g in oof.groupby(["family", "model_id"]):
        w = group_weights(g)
        err = (g.predicted_D - g.observed_D).to_numpy()
        nullerr = (g.train_null_D - g.observed_D).to_numpy()
        denom = float(w @ (nullerr ** 2))
        metric_rows.append({"model_id": model_id, "family": family,
                            "MAE_um": float(w @ np.abs(err)),
                            "RMSE_um": float(np.sqrt(w @ (err ** 2))),
                            "component_balanced_Q2": 1 - float(w @ (err ** 2)) / denom,
                            "risk_MSE_um2": float(w @ (err ** 2)),
                            "coverage": len(g)})
    metrics = pd.DataFrame(metric_rows)

    role_rows = []
    for (family, model_id, role), g in oof.groupby(["family", "model_id", "session_role"]):
        err = (g.predicted_D - g.observed_D).to_numpy()
        role_rows.append({"model_id": model_id, "family": family, "session_role": role,
                          "MAE_um": float(np.mean(np.abs(err))), "n_rows": len(g),
                          "n_components": int(g.component_id.nunique())})
    per_role = pd.DataFrame(role_rows)

    comp_risk = risk.groupby(["family", "model_id", "outer_fold", "component_id"]).risk.mean()
    rng = np.random.default_rng(config["uncertainty"]["bootstrap_seed"])
    n_boot = config["uncertainty"]["bootstrap_replicates"]
    interval_rows = []
    for family in config["models"]["families"]:
        for first, second, label in CONTRASTS:
            delta = (comp_risk.loc[(family, first)] - comp_risk.loc[(family, second)]).sort_index()
            sums = np.zeros(n_boot)
            count = 0
            for fold in sorted(frame.outer_fold.unique()):
                values = delta.loc[fold].to_numpy()
                count += len(values)
                sums += values[rng.integers(len(values), size=(n_boot, len(values)))].sum(axis=1)
            draws = sums / count
            interval_rows.append({"family": family, "contrast": label,
                                  "mean_difference_um2": float(delta.mean()),
                                  "ci95_low": float(np.quantile(draws, 0.025)),
                                  "ci95_high": float(np.quantile(draws, 0.975)),
                                  "scope": config["uncertainty"]["scope"]})
    intervals = pd.DataFrame(interval_rows)
    return oof, metrics, per_role, pd.DataFrame(tuning), intervals


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract-run", required=True, type=Path)
    ap.add_argument("--config", type=Path,
                    default=ROOT / "config/depth_target_v1/protocol.yaml")
    args = ap.parse_args()
    contract = dt_runs.verify_seal(args.contract_run)
    gate0 = json.loads((contract / "contract_gate.json").read_text(encoding="utf-8"))
    require(gate0["status"] == "PASS", "D0 contract not PASS")
    snapshot = yaml.safe_load((contract / "protocol_snapshot.yaml").read_text(encoding="utf-8"))
    require(snapshot["schema"] == "depth_target_v1", "Contract bound to another schema")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out = dt_runs.new_run("D1_depth_baselines", args.config)
    t0 = time.perf_counter()
    gate = {"experiment_id": "D1_depth_baselines", "execution_status": "FAIL",
            "training_executed": False, "deep_training_executed": False}
    try:
        suite = subprocess.run([sys.executable, "-X", "utf8", "-B", "-m", "unittest",
                                "discover", "-s", "tests/depth_target_v1", "-v"],
                               cwd=ROOT, capture_output=True, encoding="utf-8")
        (out / "tests.log").write_text(suite.stdout + suite.stderr, encoding="utf-8")
        require(suite.returncode == 0, "tests/depth_target_v1 failed")
        audit = ROOT / config["inputs"]["audit_run"]
        frame, inner = dt_splits.load_frozen_partition(audit)
        frame = frame.sort_values("dataset_index").reset_index(drop=True)
        targets = pd.read_csv(audit / "frozen_targets.csv")
        pack = dict(np.load(ROOT / config["inputs"]["height_package"], allow_pickle=False))
        rows = dt_features.align_height_rows(targets, pack)
        main_idx = rows[frame.dataset_index.to_numpy()]
        features = dt_features.build_feature_frame(
            frame, pack["height_raw"][main_idx], pack["valid_mask"][main_idx],
            config["inputs"]["pixel_um"])
        with threadpool_limits(limits=1):
            gate["training_executed"] = True
            oof, metrics, per_role, tuning, intervals = run_experiment(config, frame, inner, features)
        oof["split_sha"] = sha256(audit / "split_manifest.csv")
        oof["data_sha"] = sha256(audit / "frozen_targets.csv")
        oof.to_csv(out / "oof_depth_predictions.csv", index=False)
        metrics.to_csv(out / "depth_metrics.csv", index=False)
        per_role.to_csv(out / "per_role_metrics.csv", index=False)
        tuning.to_csv(out / "inner_tuning.csv", index=False)
        intervals.to_csv(out / "paired_risk_intervals.csv", index=False)
        best = metrics.loc[metrics.MAE_um.idxmin()]
        gate.update(execution_status="PASS", oof_rows=int(len(oof)),
                    best_model=f"{best.family}/{best.model_id}",
                    best_MAE_um=float(best.MAE_um),
                    scope="MAIN180 depth-primary OOF; SUPP20 untouched; causal claims not authorized")
    except Exception as exc:
        gate["error"] = f"{type(exc).__name__}: {exc}"
    gate["status"] = gate["execution_status"]
    gate["walltime_seconds"] = time.perf_counter() - t0
    write_json(out / "depth_baselines_gate.json", gate)
    dt_runs.seal(out)
    print(json.dumps({"output": str(out), **gate}, ensure_ascii=False, indent=2))
    return 0 if gate["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
