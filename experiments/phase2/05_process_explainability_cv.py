#!/usr/bin/env python3
"""Phase 2 experiment 05: process explainability CV (Phase 2B core).

Scale-resolved out-of-sample explainability R2(target | process) under two
grouping keys and four CV variants (细则 §7.3):
  src_gkf / src_gss : groups = shared_height_source_id   (CV-A, unseen source)
  proc_gkf / proc_gss: groups = cv_process_group          (CV-B, unseen process
                      condition; 49/50 exact-repeat pair always co-grouped)
Split contracts are type-specific: gkf test groups pairwise disjoint + union =
all; gss only per-split train/test disjointness, with per-group test counts
reported (细则 §0.7).

Input sets: A raw 5 controls; R physics-motivated reparameterized (proxy)
coordinates; C hybrid. Models: Dummy, Ridge (fold-internal alpha selection via
GroupKFold(3) on the training rows), ExtraTrees.

Targets: 21 stored targets (families A/B/C from 04) + 12 family-D fold-internal
band PC scores. Family-D PCA + cluster-bootstrap stability are computed ONCE
per (cv_variant, fold, band) and cached (细则 §7.6); cross-fold mode identity
is reported in fold_pc_alignment.csv. Family-D is a SECONDARY diagnostic: the
primary conclusions must rest on band RMS/energy, descriptors and depth
(细则 §7.4).

Seed offsets: gss 100/200, fold-PCA bank 600 + 100*fold + band, ExtraTrees 700
+ fold. Quick mode: Dummy/Ridge only, src_gkf only, B=20.
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import _lib as p2

EXPECTED = ["cv_fold_results.csv", "cv_summary.csv", "gss_test_counts.csv",
            "fold_pc_alignment.csv", "README.md"]

BANDS = ["8_16", "16_32", "32_64", "64_inf"]
FAMILY_OF = ({"median_depth_um": "A"}
             | {c: "B" for c in ["Sq_um", "Sa_um", "Ssk_skewness",
                                 "kurtosis_excess_fisher", "grad_rms_um_per_um",
                                 "lap_rms_um_per_px2", "acf_e_fold_lag_um",
                                 "aniso_gradx_over_y", "pit_density_per_Mpx",
                                 "pit_depth_um", "peak_to_valley_p98p2_um",
                                 "deepest_negative_residual_um"]}
             | {f"rms_DCT_{b}_um": "C" for b in BANDS}
             | {f"E_DCT_{b}_frac": "C" for b in BANDS})

README = """# process explainability CV (Phase 2B core)

- `cv_fold_results.csv`: fold-level long table; family-D rows carry band /
  pc_index / theta_boot_q50_deg / target_flag.
- `cv_summary.csv`: fold-quantile aggregation (median/Q10/Q25/Q75/Q90 of R2).
- `gss_test_counts.csv`: per-group test membership counts for the two gss
  variants (their test sets may repeat across splits by design, 细则 §0.7).
- `fold_pc_alignment.csv`: cross-fold PC1 / PC1-3 subspace angles per band;
  family-D PC targets may only be quoted as "clear observables" when BOTH
  within-fold bootstrap (theta < 40 deg) and between-fold alignment
  (median theta < 45 deg) are stable (细则 §7.4).
- All R2 values are exploratory cross-validated explainability estimates on
  n=200 (细则 §18); no model-validation language.
"""


def _metrics(y: np.ndarray, yp: np.ndarray) -> tuple[float, float, float, float]:
    if len(np.unique(y)) < 2:
        return np.nan, mean_absolute_error(y, yp), \
            float(np.sqrt(np.mean((y - yp) ** 2))), np.nan
    rho = (float(spearmanr(y, yp).statistic)
           if len(np.unique(yp)) > 1 else np.nan)
    return (float(r2_score(y, yp)), float(mean_absolute_error(y, yp)),
            float(np.sqrt(np.mean((y - yp) ** 2))), rho)


def _select_alpha(X_raw: np.ndarray, y: np.ndarray, groups: np.ndarray,
                  grid: list) -> float:
    """Fold-internal Ridge alpha: GroupKFold(3) on the training rows, pick the
    alpha with the best median inner-fold R2, then refit on the full training
    fold (细则 §7.2). Standardization happens INSIDE each inner split so the
    inner validation rows never contribute to the scaler (review 2026-09-03)."""
    n_groups = len(set(groups.tolist()))
    if n_groups < 3:
        return float(grid[len(grid) // 2])
    scores = {a: [] for a in grid}
    for itr, ival in GroupKFold(n_splits=3).split(X_raw, groups=groups):
        if len(np.unique(y[ival])) < 2:
            continue
        sc = StandardScaler().fit(X_raw[itr])
        Xit, Xiv = sc.transform(X_raw[itr]), sc.transform(X_raw[ival])
        for a in grid:
            m = Ridge(alpha=float(a)).fit(Xit, y[itr])
            scores[a].append(r2_score(y[ival], m.predict(Xiv)))
    med = {a: (np.nanmedian(v) if v else -np.inf) for a, v in scores.items()}
    best = grid[0]
    for a in grid[1:]:
        if med[a] > med[best]:
            best = a
    return float(best)


def _fit_predict(model_name: str, X_raw_tr: np.ndarray, Xtr: np.ndarray,
                 ytr: np.ndarray, Xte: np.ndarray, groups_tr: np.ndarray,
                 cfg: dict, rs: int) -> np.ndarray:
    if model_name == "dummy":
        return DummyRegressor(strategy="mean").fit(Xtr, ytr).predict(Xte)
    if model_name == "ridge":
        best = _select_alpha(X_raw_tr, ytr, groups_tr,
                             cfg["models"]["ridge_alpha_grid"])
        return Ridge(alpha=best).fit(Xtr, ytr).predict(Xte)
    if model_name == "extratrees":
        ecfg = cfg["models"]["extratrees"]
        return ExtraTreesRegressor(
            n_estimators=int(ecfg["n_estimators"]),
            min_samples_leaf=int(ecfg["min_samples_leaf"]),
            random_state=rs, n_jobs=-1).fit(Xtr, ytr).predict(Xte)
    raise KeyError(model_name)


def main() -> int:
    cfg, quick = p2.load_config(__doc__)
    t0 = time.time()
    out = p2.output_dir(cfg, "process_explainability")
    seed = int(cfg["random_seed"])
    p2.log("== Phase 2 / 05: process explainability CV ==")

    man = p2.read_manifest(cfg, require_loco=True)
    tdir = p2.output_dir(cfg, "multiscale_targets")
    tgt = pd.read_csv(tdir / "multiscale_targets.csv")
    p2.require(list(tgt["dataset_index"]) == list(range(200)),
               "targets row order != manifest")
    band_npz = np.load(tdir / "band_fields.npz")
    main_targets = [c for c in tgt.columns if c != "dataset_index"]
    p2.require(set(FAMILY_OF) == set(main_targets),
               f"target family map mismatch: "
               f"{set(FAMILY_OF) ^ set(main_targets)}")
    B = p2.l15.n_boot(cfg, quick)
    models = ["dummy", "ridge"] if quick else ["dummy", "ridge", "extratrees"]

    X_sets = {"A": man[p2.PROC_RAW_COLS].to_numpy(float),
              "R": man[p2.PROC_PHYS_COLS].to_numpy(float),
              "C": man[p2.PROC_RAW_COLS + p2.PROC_PHYS_COLS].to_numpy(float)}

    src_groups = man["shared_height_source_id"].to_numpy()
    proc_groups = man["cv_process_group"].to_numpy()
    variants = {"src_gkf": ("gkf", src_groups, None),
                "src_gss": ("gss", src_groups, seed + 100),
                "proc_gkf": ("gkf", proc_groups, None),
                "proc_gss": ("gss", proc_groups, seed + 200)}
    if quick:
        variants = {"src_gkf": variants["src_gkf"]}
    n_splits = int(cfg["cv"]["n_splits"])
    variant_splits = {}
    gss_count_rows = []
    for vname, (kind, groups, sseed) in variants.items():
        if kind == "gkf":
            splits = p2.gkf_splits(groups, n_splits)
            p2.check_gkf_contract(groups, splits)
        else:
            splits = p2.gss_splits(groups, n_splits, 0.2, sseed)
            counts = p2.check_gss_contract(groups, splits)
            gss_count_rows.append(pd.DataFrame(
                {"cv_variant": vname, "group": counts.index,
                 "n_test": counts.to_numpy()}))
        # every shared height source must stay inside one fold (double slots
        # share one measurement and never straddle train/test in any variant)
        src = man["shared_height_source_id"].to_numpy()
        for tr, te in splits:
            p2.require(not (set(src[tr]) & set(src[te])),
                       f"{vname} fold splits a shared height source")
        variant_splits[vname] = splits
    if gss_count_rows:
        pd.concat(gss_count_rows).to_csv(out / "gss_test_counts.csv",
                                         index=False)
    else:
        pd.DataFrame(columns=["cv_variant", "group", "n_test"]).to_csv(
            out / "gss_test_counts.csv", index=False)

    # ---- family-D fold cache: PCA + stability once per (variant, fold, band)
    clusters = p2.l15.cluster_lists(man)
    fold_cache = {}
    align_rows = []
    for vname, splits in variant_splits.items():
        for fi, (tr, te) in enumerate(splits):
            for bi, band in enumerate(BANDS):
                Xb = band_npz[f"R_band_{band}"].reshape(200, -1).astype(float)
                model = p2.fit_fold_pca(Xb[tr], 3)
                ytr = p2.project_fold_pca(model, Xb[tr])
                yte = p2.project_fold_pca(model, Xb[te])
                # cluster_lists uses the dataset_index column, which stays
                # global inside a fold; rebuild clusters at fold positions.
                src_tr = man["shared_height_source_id"].to_numpy()[tr]
                clusters_sub = [np.flatnonzero(src_tr == s)
                                for s in pd.unique(src_tr)]
                bank = p2.l15.build_resample_bank(
                    clusters_sub, B, seed + 600 + 100 * fi + bi)
                Gb = Xb[tr] @ Xb[tr].T
                angles, _ = p2.l15.boot_angles_bank(Gb, Xb[tr], bank,
                                                    model["comps"], 3)
                theta_q50 = np.percentile(angles, 50, axis=0)
                fold_cache[(vname, fi, band)] = {
                    "comps": model["comps"], "ytr": ytr, "yte": yte,
                    "theta_q50": theta_q50}
        for band in BANDS:
            for a, b in itertools.combinations(range(len(splits)), 2):
                th1, th3 = p2.pc_alignment_deg(
                    fold_cache[(vname, a, band)]["comps"],
                    fold_cache[(vname, b, band)]["comps"])
                align_rows.append((vname, band, a, b, th1, th3))
        p2.log(f"  family-D cache done for [{vname}] "
               f"({len(splits)} folds x {len(BANDS)} bands, B={B})")
    pd.DataFrame(align_rows,
                 columns=["cv_variant", "band", "fold_a", "fold_b",
                          "theta_pc1_deg", "theta_pc123_deg"]
                 ).to_csv(out / "fold_pc_alignment.csv", index=False)

    # ---- main CV loop --------------------------------------------------------
    rows = []
    n_main = len(main_targets)
    for vname, splits in variant_splits.items():
        groups = variants[vname][1]
        for fi, (tr, te) in enumerate(splits):
            groups_tr = groups[tr]
            for iname, X in X_sets.items():
                sc = StandardScaler().fit(X[tr])
                Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
                tgt_list = [(tid, FAMILY_OF[tid], tgt[tid].to_numpy(float)[tr],
                             tgt[tid].to_numpy(float)[te], np.nan, "", "", 0)
                            for tid in main_targets]
                for band in BANDS:
                    c = fold_cache[(vname, fi, band)]
                    for pc in (1, 2, 3):
                        tgt_list.append((
                            f"DCT_{band}_PC{pc}", "D",
                            c["ytr"][:, pc - 1], c["yte"][:, pc - 1],
                            float(c["theta_q50"][pc - 1]),
                            "stable" if c["theta_q50"][pc - 1] < 40.0
                            else "unstable_pc", band, pc))
                for tid, fam, ytr, yte, theta, flag, band_lbl, pc_lbl in tgt_list:
                    for mname in models:
                        yp = _fit_predict(mname, X[tr], Xtr, ytr, Xte,
                                          groups_tr, cfg, seed + 700 + fi)
                        r2, mae, rmse, rho = _metrics(yte, yp)
                        row = {"target_id": tid, "family": fam,
                               "input_set": iname, "model": mname,
                               "cv_variant": vname, "fold": fi,
                               "n_train": len(tr), "n_test": len(te),
                               "R2": r2, "MAE": mae, "RMSE": rmse,
                               "spearman_rho": rho}
                        if fam == "D":
                            row.update(band=band_lbl, pc_index=pc_lbl,
                                       theta_boot_q50_deg=theta,
                                       target_flag=flag)
                        rows.append(row)
            p2.log(f"  [{vname}] fold {fi} done "
                   f"({len(rows)} rows so far, {time.time() - t0:.0f}s)")
    res = pd.DataFrame(rows)
    res.to_csv(out / "cv_fold_results.csv", index=False)

    summary = (res.groupby(["target_id", "family", "input_set", "model",
                            "cv_variant"])
               .agg(R2_median=("R2", "median"),
                    R2_q10=("R2", lambda s: s.quantile(0.10)),
                    R2_q25=("R2", lambda s: s.quantile(0.25)),
                    R2_q75=("R2", lambda s: s.quantile(0.75)),
                    R2_q90=("R2", lambda s: s.quantile(0.90)),
                    MAE_median=("MAE", "median"),
                    RMSE_median=("RMSE", "median"),
                    n_folds=("R2", "size"))
               .reset_index())
    summary.to_csv(out / "cv_summary.csv", index=False)
    (out / "README.md").write_text(README, encoding="utf-8")

    missing = [f for f in EXPECTED if not (out / f).exists()]
    p2.require(not missing, f"missing outputs: {missing}")
    p2.log(f"05 done in {time.time() - t0:.1f}s: {len(res)} fold rows, "
           f"{len(summary)} (target, input, model, variant) cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
