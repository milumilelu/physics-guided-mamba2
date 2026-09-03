#!/usr/bin/env python3
"""Phase 2 experiment 09: sensitivity checks (细则 §11 + 2A gate addendum).

Arms, all on the primary target subset x input {A, R} x {ridge, extratrees} x
src_gkf (CV-A), compared against the 05 main-run fold medians:

  repaired              targets recomputed from the height_repaired residual;
                        median_depth_um stays on the raw field (raw authority,
                        v2 §7) and the change is noted in the summary
  formal_only           the 120 formal DOE rows only
  minus_top1/minus_top5 drop rows with phase1_global_loco_rank <= k (sensitivity
                        ONLY — never interpreted as "these samples should be
                        deleted", 规划 §23.3)
  dog                   the four DCT band-RMS targets replaced by
                        Difference-of-Gaussians band stds (true band-definition
                        cross-check; baseline = corresponding DCT band)
  exclude_artifact_yes  drop the artifact=yes samples flagged by the 2A blind
                        audit (gate PASS_WITH_FLAGS addendum)

|dR2| buckets: <0.05 stable, 0.05-0.15 moderate, >0.15 strong — descriptive
only (细则 §11). Seed offsets: ExtraTrees 900 + fold.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

import _lib as p2

EXPECTED = ["sensitivity_summary.csv", "sensitivity_fold_results.csv",
            "README.md"]

BANDS = ["8_16", "16_32", "32_64", "64_inf"]
DOG_TO_DCT = {f"std_DoG_{b}_um": f"rms_DCT_{b}_um" for b in BANDS}

README = """# sensitivity checks (Phase 2B-6)

- `sensitivity_fold_results.csv`: per-arm fold-level R2.
- `sensitivity_summary.csv`: arm medians vs the 05 main run (src_gkf), with
  descriptive |dR2| buckets (stable/moderate/strong). No arm result may be
  read as "these samples should be deleted" (规划 §23.3).
- `repaired` arm: median_depth_um intentionally unchanged (raw height is the
  authority, v2 §7); Sq and band RMS recomputed from the repaired residual.
- `dog` arm: DoG stds are octave-like band amplitudes (G2-G4, G4-G8, G8-G16,
  G16 low-pass); baseline pairs each DoG band with its DCT counterpart.
- `exclude_artifact_yes` arm implements the 2A gate PASS_WITH_FLAGS addendum.
"""


def main() -> int:
    cfg, quick = p2.load_config(__doc__)
    t0 = time.time()
    out = p2.output_dir(cfg, "sensitivity")
    seed = int(cfg["random_seed"])
    p2.log("== Phase 2 / 09: sensitivity checks ==")

    _spec = importlib.util.spec_from_file_location(
        "phase2_05", Path(__file__).with_name("05_process_explainability_cv.py"))
    e05 = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(e05)

    frozen = p2.l15.load_frozen(cfg)
    man = p2.read_manifest(cfg, require_loco=True)
    tgt = pd.read_csv(p2.output_dir(cfg, "multiscale_targets")
                      / "multiscale_targets.csv")
    models = ["ridge"] if quick else ["ridge", "extratrees"]
    inputs = {"A": man[p2.PROC_RAW_COLS].to_numpy(float),
              "R": man[p2.PROC_PHYS_COLS].to_numpy(float)}
    targets = list(cfg["targets"]["primary_subset"])
    y_full = {t: tgt[t].to_numpy(float) for t in targets}
    R3 = frozen["R"]

    # ---- baseline: 05 main-run fold medians ---------------------------------
    res05 = pd.read_csv(p2.output_dir(cfg, "process_explainability")
                        / "cv_fold_results.csv")
    base = (res05[(res05["cv_variant"] == "src_gkf")
                  & (res05["model"].isin(models))]
            .groupby(["target_id", "input_set", "model"])["R2"]
            .median().reset_index(name="R2_med_base"))

    def _run_arm(arm: str, man_arm: pd.DataFrame, y_by_target: dict,
                 X_by_input: dict, rows: list) -> None:
        groups = man_arm["shared_height_source_id"].to_numpy()
        splits = p2.gkf_splits(groups, int(cfg["cv"]["n_splits"]))
        p2.check_gkf_contract(groups, splits)
        for fi, (tr, te) in enumerate(splits):
            for iname, X in X_by_input.items():
                sc = StandardScaler().fit(X[tr])
                Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
                for tid, y in y_by_target.items():
                    for mname in models:
                        yp = e05._fit_predict(
                            mname, X[tr], Xtr, y[tr], Xte, groups[tr], cfg,
                            seed + 900 + fi)
                        r2 = (float(r2_score(y[te], yp))
                              if len(np.unique(y[te])) > 1 else np.nan)
                        rows.append({"arm": arm, "target_id": tid,
                                     "input_set": iname, "model": mname,
                                     "fold": fi, "R2": r2})
        p2.log(f"  arm [{arm}] done ({time.time() - t0:.0f}s)")

    rows = []

    # ---- arm: repaired -------------------------------------------------------
    npz = np.load(p2.l15.REPO / cfg["paths"]["dataset_npz"])
    Hrep = npz["height_repaired"].astype(float)
    V = frozen["V"]
    Hrepn = np.where(V, Hrep, np.nan)
    R_rep = Hrepn - np.nanmedian(Hrepn, axis=(1, 2))[:, None, None]
    bands_rep, _ = p2.l15.dct_band_fields(R_rep, float(cfg["scales"]["pixel_um"]),
                                          cfg["scales"]["dct_bands_um"])
    y_rep = {"median_depth_um": man["median_depth_um"].to_numpy(float),
             "Sq_um": np.sqrt(np.nanmean(R_rep ** 2, axis=(1, 2)))}
    for b in BANDS:
        y_rep[f"rms_DCT_{b}_um"] = bands_rep[f"DCT_{b}"] \
            .reshape(200, -1).std(axis=1)
    _run_arm("repaired", man, y_rep, inputs, rows)

    # ---- arm: formal_only ----------------------------------------------------
    idx = np.flatnonzero((man["session_role"] == "formal").to_numpy())
    man_f = man.iloc[idx].reset_index(drop=True)
    _run_arm("formal_only", man_f,
             {t: y_full[t][idx] for t in targets},
             {k: v[idx] for k, v in inputs.items()}, rows)

    # ---- arms: minus_top1 / minus_top5 ---------------------------------------
    rank = man["phase1_global_loco_rank"].to_numpy()
    for k in (1, 5):
        idx = np.flatnonzero(rank > k)
        _run_arm(f"minus_top{k}", man.iloc[idx].reset_index(drop=True),
                 {t: y_full[t][idx] for t in targets},
                 {k2: v[idx] for k2, v in inputs.items()}, rows)

    # ---- arm: dog (band-definition cross-check) ------------------------------
    dog = p2.dog_band_stds(R3, cfg["scales"]["sigmas_px"])
    y_dog = {"median_depth_um": y_full["median_depth_um"],
             "Sq_um": y_full["Sq_um"]}
    y_dog.update({k: v for k, v in dog.items()})
    _run_arm("dog", man, y_dog, inputs, rows)

    # ---- arm: exclude_artifact_yes (2A gate addendum) -------------------------
    rev_path = (p2.l15.REPO / "outputs/phase2/instability/盲评"
                / "instability_manual_review_completed.csv")
    rev = pd.read_csv(rev_path)
    col = [c for c in rev.columns if "unblind" in c.lower()
           and "artifact" in c.lower()][0]
    yes_idx = rev[rev[col].astype(str).str.strip().str.lower() == "yes"] \
        ["dataset_index"].astype(int).tolist()
    p2.log(f"  artifact=yes samples from 2A audit: {yes_idx}")
    keep = np.flatnonzero(~man["dataset_index"].isin(yes_idx).to_numpy())
    _run_arm("exclude_artifact_yes", man.iloc[keep].reset_index(drop=True),
             {t: y_full[t][keep] for t in targets},
             {k2: v[keep] for k2, v in inputs.items()}, rows)

    # ---- summary --------------------------------------------------------------
    arm_res = pd.DataFrame(rows)
    arm_res.to_csv(out / "sensitivity_fold_results.csv", index=False)
    med = (arm_res.groupby(["arm", "target_id", "input_set", "model"])["R2"]
           .median().reset_index(name="R2_med_arm"))
    med["baseline_target_id"] = med["target_id"].map(
        lambda t: DOG_TO_DCT.get(t, t))
    med = med.merge(base.rename(columns={"target_id": "baseline_target_id"}),
                    on=["baseline_target_id", "input_set", "model"],
                    how="left", validate="many_to_one")
    med["dR2"] = med["R2_med_arm"] - med["R2_med_base"]
    med["sensitivity"] = np.where(med["dR2"].abs() < 0.05, "stable",
                                  np.where(med["dR2"].abs() <= 0.15,
                                           "moderate", "strong"))
    med.to_csv(out / "sensitivity_summary.csv", index=False)
    (out / "README.md").write_text(README, encoding="utf-8")

    missing = [f for f in EXPECTED if not (out / f).exists()]
    p2.require(not missing, f"missing outputs: {missing}")
    p2.log(f"09 done in {time.time() - t0:.1f}s: {len(arm_res)} fold rows, "
           f"{len(med)} summary cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
