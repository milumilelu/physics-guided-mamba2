#!/usr/bin/env python3
"""Phase 2 experiment 08: local regime probe (same held-out population).

Triggered by 2A gate Q2 = YES_AS_MORPHOLOGY_FAMILIES. The v1 design
(R2_local - R2_global) was rejected in review: stratum-restricted variance
makes that difference spuriously positive, and outcome-defined strata must
not score their own variable. The implemented design (细则 §10):

  for every src_gkf outer fold and stratum r:
    - global model: trained on ALL outer-train rows;
    - local model:  trained ONLY on the outer-train rows of stratum r;
    - both predict the IDENTICAL held-out fold rows of stratum r;
    - comparison: MAE on that same batch, and
      Skill = 1 - MAE_model / MAE_dummy with the stratum-mean dummy fitted
      on the same local training rows.
  - quartile edges are computed on the TRAINING fold;
  - depth quartile never scores median_depth_um, Sq quartile never scores
    Sq_um (outcome-defined stratum exclusion).

Targets: primary subset (细则 §11). Models: ridge / extratrees. Input: A raw.
Seed offsets: ExtraTrees 800 + fold.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import _lib as p2

EXPECTED = ["local_vs_global.csv", "local_vs_global_summary.csv",
            "local_regime_probe.png", "README.md"]

STRATA = [("depth_q4", "median_depth_um"), ("sq_q4", "Sq_um"),
          ("consensus_half", "A_consensus")]
EXCLUDE_OWN = {"depth_q4": "median_depth_um", "sq_q4": "Sq_um",
               "consensus_half": None}

README = """# local regime probe (same held-out comparison, 细则 §10)

- `local_vs_global.csv`: fold × stratum × target × model rows with MAE_global,
  MAE_local, MAE_dummy (stratum-mean, local training rows) and
  Skill = 1 - MAE_model/MAE_dummy, all evaluated on the IDENTICAL held-out
  rows of the stratum.
- Strata: depth quartile / Sq quartile (edges from the training fold;
  own-variable targets excluded) and A_consensus median split.
- `delta_skill = Skill_local - Skill_global` is the only regime readout;
  positive values with a small local training set must be checked against the
  dummy column before any "local representation" language (细则 §18).
"""


def main() -> int:
    cfg, quick = p2.load_config(__doc__)
    t0 = time.time()
    out = p2.output_dir(cfg, "local_structure")
    seed = int(cfg["random_seed"])
    p2.log("== Phase 2 / 08: local regime probe (same held-out) ==")

    _spec = importlib.util.spec_from_file_location(
        "phase2_05", Path(__file__).with_name("05_process_explainability_cv.py"))
    e05 = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(e05)

    man = p2.read_manifest(cfg, require_loco=True)
    inv = pd.read_csv(p2.output_dir(cfg, "instability")
                      / "instability_inventory.csv")
    tdir = p2.output_dir(cfg, "multiscale_targets")
    tgt = pd.read_csv(tdir / "multiscale_targets.csv")
    targets = [t for t in cfg["targets"]["primary_subset"]]
    models = ["ridge", "extratrees"]
    X = man[p2.PROC_RAW_COLS].to_numpy(float)
    strat_cols = {"depth_q4": tgt["median_depth_um"].to_numpy(float),
                  "sq_q4": tgt["Sq_um"].to_numpy(float),
                  "consensus_half": inv["A_consensus"].to_numpy(float)}

    groups = man["shared_height_source_id"].to_numpy()
    splits = p2.gkf_splits(groups, int(cfg["cv"]["n_splits"]))
    p2.check_gkf_contract(groups, splits)

    rows = []
    for fi, (tr, te) in enumerate(splits):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        # global predictions for every (model, target), once per fold
        glob_pred = {}
        for mname in models:
            for tid in targets:
                y = tgt[tid].to_numpy(float)
                glob_pred[(mname, tid)] = e05._fit_predict(
                    mname, X[tr], Xtr, y[tr], Xte, groups[tr], cfg,
                    seed + 800 + fi)
        for sname, svar in STRATA:
            v = strat_cols[sname]
            if sname.endswith("_q4"):
                edges = np.quantile(v[tr], [0.25, 0.50, 0.75])
                lab = np.digitize(v, edges, right=True)  # 0..3 on all rows
                labels = [f"Q{i + 1}" for i in range(4)]
                labs_all = [labels[i] for i in lab]
            else:
                med = float(np.median(v[tr]))
                labs_all = ["lower" if x <= med else "upper" for x in v]
                labels = ["lower", "upper"]
            for label in labels:
                tr_s = np.array([i for i, j in zip(tr, labs_all) if j == label])
                te_s = np.array([i for i, j in zip(te, labs_all) if j == label])
                if len(tr_s) < 10 or len(te_s) < 3:
                    p2.log(f"  [{sname}/{label}] fold {fi}: skipped "
                           f"(n_train={len(tr_s)}, n_test={len(te_s)})")
                    continue
                sc_s = StandardScaler().fit(X[tr_s])
                Xtr_s, Xte_s = sc_s.transform(X[tr_s]), sc_s.transform(X[te_s])
                excl = EXCLUDE_OWN[sname]
                for mname in models:
                    for tid in targets:
                        if tid == excl:
                            continue  # outcome-defined stratum exclusion
                        y = tgt[tid].to_numpy(float)
                        mae_g = float(np.mean(np.abs(
                            y[te_s] - glob_pred[(mname, tid)][
                                np.searchsorted(te, te_s)])))
                        yp_local = e05._fit_predict(
                            mname, X[tr_s], Xtr_s, y[tr_s], Xte_s,
                            groups[tr_s], cfg, seed + 800 + fi)
                        mae_l = float(np.mean(np.abs(y[te_s] - yp_local)))
                        mae_d = float(np.mean(np.abs(
                            y[te_s] - np.mean(y[tr_s]))))
                        rows.append({
                            "stratum_spec": sname, "stratum": label,
                            "target_id": tid, "model": mname, "fold": fi,
                            "n_train_local": len(tr_s), "n_test": len(te_s),
                            "MAE_global": mae_g, "MAE_local": mae_l,
                            "MAE_dummy": mae_d,
                            "Skill_global": 1 - mae_g / mae_d,
                            "Skill_local": 1 - mae_l / mae_d,
                            "delta_skill": (1 - mae_l / mae_d)
                            - (1 - mae_g / mae_d)})
        p2.log(f"  fold {fi} done ({len(rows)} rows so far, "
               f"{time.time() - t0:.0f}s)")

    res = pd.DataFrame(rows)
    res.to_csv(out / "local_vs_global.csv", index=False)
    summary = (res.groupby(["stratum_spec", "stratum", "target_id", "model"])
               ["delta_skill"]
               .agg(delta_skill_median="median",
                    delta_q25=lambda s: s.quantile(0.25),
                    delta_q75=lambda s: s.quantile(0.75),
                    n_pos=lambda s: int((s > 0).sum()), n_folds="size")
               .reset_index())
    summary.to_csv(out / "local_vs_global_summary.csv", index=False)

    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 4.4),
                             squeeze=False, sharey=True)
    for j, mname in enumerate(models):
        ax = axes[0, j]
        sub = summary[summary["model"] == mname]
        specs = sorted(sub["stratum_spec"].unique())
        targets_u = sorted(sub["target_id"].unique())
        xpos = np.arange(len(targets_u))
        for k, spec in enumerate(specs):
            vals = sub[sub["stratum_spec"] == spec] \
                .groupby("target_id")["delta_skill_median"].sum() \
                .reindex(targets_u)
            ax.bar(xpos + (k - (len(specs) - 1) / 2) * 0.27, vals, width=0.27,
                   label=spec, color=f"C{k}")
        ax.axhline(0.0, color="0.4", lw=0.8)
        ax.set_xticks(xpos, [t.replace("_", " ") for t in targets_u],
                      rotation=45, fontsize=7)
        ax.set_ylabel("delta skill (local - global), median over folds")
        ax.set_title(f"model = {mname}", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Local vs global probe — same held-out population, "
                 "Skill = 1 - MAE/MAE_dummy (exploratory)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "local_regime_probe.png", dpi=cfg["plot"]["dpi"])
    plt.close(fig)

    (out / "README.md").write_text(README, encoding="utf-8")
    missing = [f for f in EXPECTED if not (out / f).exists()]
    p2.require(not missing, f"missing outputs: {missing}")
    p2.log(f"08 done in {time.time() - t0:.1f}s: {len(res)} rows, "
           f"{len(summary)} summary cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
