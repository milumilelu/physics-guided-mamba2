#!/usr/bin/env python3
"""Phase 2.5 Task 12: spectral process map (main experiment).

u -> spectral composition & directional texture, under grouped CV.

Targets:
  P1 multivariate: ILR balances z1..z4 (five-part composition, Task 10);
     metrics: Aitchison Q2 (fold-train-mean dummy), Aitchison distance d_A,
     inverse-ILR fraction MAE, per-balance R2.
  P2 directional: A2_8_16, angular_entropy_8_16 (Task 11 metrics).
  secondary: centroid / entropy / N_eff / A2_16_32 / A2_32_64.
  reference (never gate-relevant): median_depth_um, Sq_um, rms_DCT_8_16_um.

Inputs: A (raw 5 controls) and C (A + 4 derived proxy features); R only as an
appendix arm. Models: dummy / ridge (fold-internal alpha, inner scaler) /
spline+ridge / extratrees. Primary CVs: src_gkf + proc_gkf; sensitivity:
src_gss/proc_gss, formal_only, exclude_artifact_yes, minus_top5 (selection
logic identical to Phase 2-09). Task 12 config (thresholds) is committed
BEFORE the run (细则 §0.8).

OOF contract: per-sample predictions stored for every (variant, model, input,
fold) of src_gkf; uniqueness asserted (single test #15). 14B primary model =
ridge (细则 §0.7); ET = sensitivity.

Seed offsets: gss 100/200, ET 700+fold, permutation importance 800.
"""

from __future__ import annotations

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.pipeline import Pipeline

import _lib as p25

EXPECTED = ["cv_fold_results.csv", "cv_summary.csv",
            "composition_oof_predictions.csv", "directional_oof_predictions.csv",
            "input_comparison.csv", "nonlinear_comparison.csv",
            "permutation_importance.csv", "additive_response_curves.csv",
            "spectrum_predictability_map.png", "ilr_balance_predictability.png",
            "predicted_vs_true_composition.png",
            "process_feature_importance.png", "README.md"]

README = """# spectral process map (Task 12, main experiment)

- `cv_fold_results.csv`: fold-level long table. Composition rows report
  Q2_Aitchison / dA_median / MAE_p_* / R2_z1..z4; scalar rows report
  R2/MAE/RMSE/Spearman.
- `composition_oof_predictions.csv`: per-sample OOF z (and inverse-ILR p) for
  every (variant, model, input) — src_gkf rows are unique per sample
  (contract test #15). 14B primary model = ridge (细则 §0.7).
- `input_comparison.csv`: fold-paired dQ2(C-A) for ALL variants (rev2-06
  lesson); positive = derived combinations give simple models a better
  inductive bias — never "new experimental information".
- `nonlinear_comparison.csv`: dQ2(ET-spline) and dR2(ET-Ridge) per variant.
- `permutation_importance.csv`: ET permutation importance on the composition
  (Aitchison Q2 drop) — only meaningful where Q2 clearly beats dummy
  (median > 0.10, 细则 §14).
- Gates G1/G2b/G3a are evaluated from these files (细则 §12).
"""


def _q2_aitchison(z_test: np.ndarray, z_pred: np.ndarray,
                  z_train: np.ndarray) -> float:
    denom = float(((z_test - z_train.mean(axis=0)) ** 2).sum())
    if denom <= 0:
        return np.nan
    return float(1.0 - ((z_test - z_pred) ** 2).sum() / denom)


def _spline_ridge_fit_predict(Xtr_raw, Xte_raw, ytr, cfg, alpha):
    """Spline basis fitted on the (already standardized) training fold only."""
    spline = SplineTransformer(degree=int(cfg["models"]["spline"]["degree"]),
                               n_knots=int(cfg["models"]["spline"]["n_knots"]),
                               include_bias=False)
    Xtr_s = spline.fit_transform(Xtr_raw)
    Xte_s = spline.transform(Xte_raw)
    model = Ridge(alpha=alpha).fit(Xtr_s, ytr)
    return model.predict(Xte_s)


def main() -> int:
    cfg, quick = p25.load_config(__doc__)
    t0 = time.time()
    out = p25.output_dir(cfg, "process_map")
    seed = int(cfg["random_seed"])
    p25.log("== Phase 2.5 / 12: spectral process map ==")

    man = p25.read_phase2_manifest(cfg)
    comp = pd.read_csv(p25.output_dir(cfg, "spectral_composition")
                       / "spectral_composition.csv")
    ilr = pd.read_csv(p25.output_dir(cfg, "spectral_composition")
                      / "ilr_coordinates.csv")
    dmetrics = pd.read_csv(p25.output_dir(cfg, "directional_spectrum")
                           / "directional_metrics.csv")
    tgt05 = pd.read_csv(p25.l15.REPO / "outputs/phase1_5/morphology_descriptors.csv")

    Z = ilr[[f"ilr_z{j}" for j in range(1, 5)]].to_numpy(float)
    P = comp[[f"p_{b}" for b in p25.ILR_BANDS]].to_numpy(float)
    dmm = dmetrics.set_index(["dataset_index", "band"])
    scal = {}
    scal["A2_8_16"] = dmm.xs("8_16", level="band")["A2"].to_numpy(float)
    scal["angular_entropy_8_16"] = dmm.xs("8_16", level="band")[
        "angular_entropy"].to_numpy(float)
    for name, col in (("A2_16_32", "A2"), ("A2_32_64", "A2")):
        scal[name] = dmm.xs(name.replace("A2_", ""), level="band")[
            col].to_numpy(float)
    spec_desc = pd.read_csv(p25.output_dir(cfg, "spectral_composition")
                            / "spectrum_descriptor_summary.csv")
    scal["spectral_centroid_log_um"] = spec_desc["spectral_centroid_log_um"].to_numpy(float)
    scal["spectral_entropy"] = spec_desc["spectral_entropy"].to_numpy(float)
    scal["effective_band_number"] = spec_desc["effective_band_number"].to_numpy(float)
    for t in cfg["targets"]["reference"]:
        scal[t] = (man["median_depth_um"].to_numpy(float)
                   if t == "median_depth_um"
                   else tgt05[f"{t}"].to_numpy(float))

    X_by_input = {
        "A": man[cfg["input_sets"]["A"]].to_numpy(float),
        "C": man[cfg["input_sets"]["A"] + cfg["input_sets"]["C_extra"]].to_numpy(float),
        "R": man[p25.p2.PROC_PHYS_COLS].to_numpy(float),
    }
    models = ["dummy", "ridge"] if quick else \
        ["dummy", "ridge", "spline", "extratrees"]

    src_groups = man["shared_height_source_id"].to_numpy()
    proc_groups = man["cv_process_group"].to_numpy()
    variants = {"src_gkf": ("gkf", src_groups, None),
                "proc_gkf": ("gkf", proc_groups, None),
                "src_gss": ("gss", src_groups, seed + 100),
                "proc_gss": ("gss", proc_groups, seed + 200)}
    if quick:
        variants = {"src_gkf": variants["src_gkf"]}
    variant_splits = {}
    for vname, (kind, groups, sseed) in variants.items():
        if kind == "gkf":
            splits = p25.gkf_splits(groups, int(cfg["cv"]["n_splits"]))
            p25.check_gkf_contract(groups, splits)
        else:
            splits = p25.gss_splits(groups, int(cfg["cv"]["gss_repeats"]),
                                    0.2, sseed)
            p25.check_gss_contract(groups, splits)
        src = man["shared_height_source_id"].to_numpy()
        for tr, te in splits:
            p25.require(not (set(src[tr]) & set(src[te])),
                        f"{vname} fold splits a shared height source")
        variant_splits[vname] = splits

    fit_rows, oof_comp, oof_dir = [], [], []

    def _fit_predict(model_name, Xtr_raw, Xtr, ytr, Xte, groups_tr, fold_i):
        if model_name == "dummy":
            return DummyRegressor(strategy="mean").fit(Xtr, ytr).predict(Xte)
        if model_name == "ridge":
            best = _alpha_inner(Xtr_raw, ytr, groups_tr)
            return Ridge(alpha=best).fit(Xtr, ytr).predict(Xte)
        if model_name == "spline":
            best = _alpha_inner(Xtr_raw, ytr, groups_tr)
            return _spline_ridge_fit_predict(Xtr_raw, Xte, ytr, cfg, best)
        if model_name == "extratrees":
            ec = cfg["models"]["extratrees"]
            return ExtraTreesRegressor(
                n_estimators=int(ec["n_estimators"]),
                min_samples_leaf=int(ec["min_samples_leaf"]),
                random_state=seed + 700 + fold_i, n_jobs=-1
            ).fit(Xtr, ytr).predict(Xte)
        raise KeyError(model_name)

    def _alpha_inner(X_raw, y, groups):
        # scalar y for alpha selection: use the first target column (multiout)
        y1 = y[:, 0] if y.ndim == 2 else y
        return _select_alpha_shared(X_raw, y1, groups,
                                    cfg["models"]["ridge_alpha_grid"])

    from sklearn.model_selection import GroupKFold

    def _select_alpha_shared(X_raw, y, groups, grid):
        if len(set(groups.tolist())) < 3:
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
        best = grid[0]
        med = {a: (np.nanmedian(v) if v else -np.inf) for a, v in scores.items()}
        for a in grid[1:]:
            if med[a] > med[best]:
                best = a
        return float(best)

    ARM_MODELS = models
    for vname, splits in variant_splits.items():
        groups = variants[vname][1]
        for fi, (tr, te) in enumerate(splits):
            groups_tr = groups[tr]
            for iname, X in X_by_input.items():
                if iname == "R" and vname != "src_gkf":
                    continue                       # R appendix arm only
                sc = StandardScaler().fit(X[tr])
                Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
                # ---- composition (multivariate) ----
                for mname in ARM_MODELS:
                    zp = _fit_predict(mname, X[tr], Xtr, Z[tr], Xte,
                                      groups_tr, fi)
                    q2 = _q2_aitchison(Z[te], zp, Z[tr])
                    # d_A(p_hat, p) == ||z_hat - z|| exactly (ILR isometry):
                    # never round-trip predictions through inverse-ILR here
                    dA = np.linalg.norm(zp - Z[te], axis=1)
                    ph = p25.ilr_inverse(zp)
                    row = {"target": "ilr_z1_z4", "input_set": iname,
                           "model": mname, "cv_variant": vname, "fold": fi,
                           "n_train": len(tr), "n_test": len(te),
                           "Q2_Aitchison": q2,
                           "dA_median": float(np.median(dA)),
                           "dA_q25": float(np.quantile(dA, 0.25)),
                           "dA_q75": float(np.quantile(dA, 0.75))}
                    for j in range(4):
                        row[f"R2_z{j + 1}"] = (
                            float(r2_score(Z[te][:, j], zp[:, j]))
                            if len(np.unique(Z[te][:, j])) > 1 else np.nan)
                    for j, b in enumerate(p25.ILR_BANDS):
                        row[f"MAE_p_{b}"] = float(
                            np.mean(np.abs(ph[:, j] - P[te, j])))
                    fit_rows.append(row)
                    if vname == "src_gkf" and not quick:
                        for s in range(len(te)):
                            oof_comp.append({
                                "dataset_index": int(te[s]), "fold": fi,
                                "model": mname, "input_set": iname,
                                "cv_variant": vname,
                                **{f"z{j + 1}_pred": zp[s, j] for j in range(4)},
                                **{f"p_{b}_pred": ph[s, j]
                                   for j, b in enumerate(p25.ILR_BANDS)}})
                # ---- scalar targets ----
                for tid, y in scal.items():
                    for mname in ARM_MODELS:
                        yp = _fit_predict(mname, X[tr], Xtr, y[tr], Xte,
                                          groups_tr, fi)
                        rho = (float(spearmanr(y[te], yp).statistic)
                               if len(np.unique(yp)) > 1
                               and len(np.unique(y[te])) > 1 else np.nan)
                        fit_rows.append({
                            "target": tid, "input_set": iname,
                            "model": mname, "cv_variant": vname, "fold": fi,
                            "n_train": len(tr), "n_test": len(te),
                            "R2": float(r2_score(y[te], yp))
                            if len(np.unique(y[te])) > 1 else np.nan,
                            "MAE": float(mean_absolute_error(y[te], yp)),
                            "spearman_rho": rho})
                        if vname == "src_gkf" and not quick:
                            for s in range(len(te)):
                                oof_dir.append({
                                    "dataset_index": int(te[s]), "fold": fi,
                                    "model": mname, "input_set": iname,
                                    "cv_variant": vname, "target": tid,
                                    "y_true": float(y[te][s]),
                                    "y_pred": float(yp[s])})
            p25.log(f"  [{vname}] fold {fi} done ({len(fit_rows)} rows, "
                    f"{time.time() - t0:.0f}s)")

    # ---- sensitivity arms (composition only; 细则 §0.10/§11 S1–S3) ----------
    if not quick:
        rev = pd.read_csv(p25.l15.REPO / "outputs/phase2/instability/盲评"
                          / "instability_manual_review_completed.csv")
        col = [c for c in rev.columns if "unblind" in c.lower()
               and "artifact" in c.lower()][0]
        yes_idx = rev[rev[col].astype(str).str.strip().str.lower() == "yes"][
            "dataset_index"].astype(int).tolist()
        rank = man["phase1_global_loco_rank"].to_numpy()
        arms = {"formal_only": (man["session_role"] == "formal").to_numpy(),
                "exclude_artifact_yes":
                    ~man["dataset_index"].isin(yes_idx).to_numpy(),
                "minus_top5": rank > 5}
        for arm, mask in arms.items():
            idx = np.flatnonzero(mask)
            man_arm = man.iloc[idx].reset_index(drop=True)
            groups_arm = man_arm["shared_height_source_id"].to_numpy()
            splits_a = p25.gkf_splits(groups_arm, int(cfg["cv"]["n_splits"]))
            p25.check_gkf_contract(groups_arm, splits_a)
            for fi, (tr, te) in enumerate(splits_a):
                for iname in ("A", "C"):
                    X = X_by_input[iname][idx]
                    sc = StandardScaler().fit(X[tr])
                    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
                    groups_tr = groups_arm[tr]
                    for mname in ("dummy", "ridge", "extratrees"):
                        zp = _fit_predict(mname, X[tr], Xtr, Z[tr], Xte,
                                          groups_tr, fi)
                        fit_rows.append({
                            "target": "ilr_z1_z4", "input_set": iname,
                            "model": mname,
                            "cv_variant": f"{arm}_src_gkf", "fold": fi,
                            "n_train": len(tr), "n_test": len(te),
                            "Q2_Aitchison": _q2_aitchison(Z[idx][te], zp,
                                                          Z[idx][tr])})
            p25.log(f"  [sensitivity {arm}] done")

    res = pd.DataFrame(fit_rows)
    res.to_csv(out / "cv_fold_results.csv", index=False)
    pd.DataFrame(oof_comp).to_csv(out / "composition_oof_predictions.csv",
                                  index=False)
    pd.DataFrame(oof_dir).to_csv(out / "directional_oof_predictions.csv",
                                 index=False)

    qcols = ["Q2_Aitchison", "dA_median", "R2_z1", "R2_z2", "R2_z3", "R2_z4"]
    comp_rows = res[res.target == "ilr_z1_z4"]
    summary = comp_rows.groupby(["target", "input_set", "model",
                                 "cv_variant"]).agg(
        Q2_median=("Q2_Aitchison", "median"),
        Q2_q25=("Q2_Aitchison", lambda s: s.quantile(0.25)),
        Q2_q75=("Q2_Aitchison", lambda s: s.quantile(0.75)),
        n_pos=("Q2_Aitchison", lambda s: int((s > 0).sum())),
        n_folds=("Q2_Aitchison", "size"),
        **{f"{c}_median": (c, "median") for c in qcols[1:]}).reset_index()
    scalar_rows = res[res.target != "ilr_z1_z4"]
    summary = pd.concat([
        summary,
        scalar_rows.groupby(["target", "input_set", "model", "cv_variant"])
        .agg(R2_median=("R2", "median"),
             R2_q25=("R2", lambda s: s.quantile(0.25)),
             R2_q75=("R2", lambda s: s.quantile(0.75)),
             n_pos=("R2", lambda s: int((s > 0).sum())),
             n_folds=("R2", "size")).reset_index()])
    summary.to_csv(out / "cv_summary.csv", index=False)

    # ---- A vs C paired dQ2 (all variants) ------------------------------------
    keep = ["target", "cv_variant", "fold"]
    comp_q = comp_rows[comp_rows.model != "dummy"]
    parts = []
    for mname in comp_q.model.unique():
        a = comp_q[(comp_q.input_set == "A") & (comp_q.model == mname)][
            keep + ["Q2_Aitchison"]].rename(columns={"Q2_Aitchison": "Q2_A"})
        c = comp_q[(comp_q.input_set == "C") & (comp_q.model == mname)][
            keep + ["Q2_Aitchison"]].rename(columns={"Q2_Aitchison": "Q2_C"})
        m = a.merge(c, on=keep, validate="one_to_one")
        m["dQ2_C-A"] = m["Q2_C"] - m["Q2_A"]
        m["model"] = mname
        parts.append(m)
    input_cmp = pd.concat(parts)
    agg = input_cmp.groupby(["target", "model", "cv_variant"])["dQ2_C-A"] \
        .agg(dQ2_median="median",
             dQ2_q25=lambda s: s.quantile(0.25),
             dQ2_q75=lambda s: s.quantile(0.75),
             n_folds="size").reset_index()
    agg["n_folds_pos"] = input_cmp.assign(
        pos=input_cmp["dQ2_C-A"] > 0).groupby(
        ["target", "model", "cv_variant"])["pos"].sum().to_numpy()
    agg.to_csv(out / "input_comparison.csv", index=False)

    # ---- nonlinear comparison (fold-paired, all variants) --------------------
    nl_parts = []
    for iname in ("A", "C"):
        sub = comp_q[comp_q.input_set == iname]
        piv = sub.pivot_table(index=["cv_variant", "fold"], columns="model",
                              values="Q2_Aitchison")
        for a, b, col in (("extratrees", "spline", "dQ2_ET-spline"),
                          ("extratrees", "ridge", "dQ2_ET-ridge")):
            if a in piv.columns and b in piv.columns:
                d = (piv[a] - piv[b]).rename(col).reset_index()
                d["input_set"] = iname
                d["target"] = "ilr_z1_z4"
                nl_parts.append(d)
    for iname in ("A", "C"):
        sr = scalar_rows[scalar_rows.input_set == iname]
        piv = sr.pivot_table(index=["target", "cv_variant", "fold"],
                             columns="model", values="R2")
        for a, b, col in (("extratrees", "spline", "dR2_ET-spline"),
                          ("extratrees", "ridge", "dR2_ET-ridge")):
            if a in piv.columns and b in piv.columns:
                d = (piv[a] - piv[b]).rename(col).reset_index()
                d["input_set"] = iname
                nl_parts.append(d)
    if nl_parts:
        nl = pd.concat(nl_parts, ignore_index=True)
    else:
        nl = pd.DataFrame(columns=["target", "input_set", "cv_variant"])
    nl_agg = []
    for keys, sub in nl.groupby(["target", "input_set", "cv_variant"]):
        row = {"target": keys[0], "input_set": keys[1], "cv_variant": keys[2]}
        for c in ("dQ2_ET-spline", "dQ2_ET-ridge", "dR2_ET-spline",
                  "dR2_ET-ridge"):
            if c in sub.columns:
                row[f"{c}_median"] = float(sub[c].median())
                row[f"{c}_n_pos"] = int((sub[c] > 0).sum())
                row["n_folds"] = len(sub)
        nl_agg.append(row)
    pd.DataFrame(nl_agg).to_csv(out / "nonlinear_comparison.csv", index=False)

    if not quick:
        _permutation_importance(cfg, X_by_input["A"], Z, variant_splits,
                                seed, out)
        _figures(cfg, summary, P, oof_comp, out)

    (out / "README.md").write_text(README, encoding="utf-8")
    expected = [f for f in EXPECTED
                if not (quick and f in ("permutation_importance.csv",
                                        "additive_response_curves.csv",
                                        "spectrum_predictability_map.png",
                                        "ilr_balance_predictability.png",
                                        "predicted_vs_true_composition.png",
                                        "process_feature_importance.png"))]
    missing = [f for f in expected if not (out / f).exists()]
    p25.require(not missing, f"missing outputs: {missing}")
    p25.log(f"12 done in {time.time() - t0:.1f}s: {len(res)} fold rows")
    return 0


def splits_first(variant_splits):
    return variant_splits


def _permutation_importance(cfg, X, Z, variant_splits, seed, out):
    """ET permutation importance on the composition (Aitchison Q2 drop),
    src_gkf folds pooled; only registered where Q2 beats dummy clearly."""
    splits = variant_splits["src_gkf"]
    cols = cfg["input_sets"]["A"]
    rng = np.random.default_rng(seed + 800)
    rows = []
    base_q2 = []
    for fi, (tr, te) in enumerate(splits):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        ec = cfg["models"]["extratrees"]
        model = ExtraTreesRegressor(
            n_estimators=int(ec["n_estimators"]),
            min_samples_leaf=int(ec["min_samples_leaf"]),
            random_state=seed + 700 + fi, n_jobs=-1).fit(Xtr, Z[tr])
        q2 = _q2_aitchison(Z[te], model.predict(Xte), Z[tr])
        base_q2.append(q2)
        drops = {c: [] for c in cols}
        for _ in range(10):
            for j, c in enumerate(cols):
                Xp = Xte.copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                drops[c].append(_q2_aitchison(Z[te], model.predict(Xp), Z[tr]))
        for c in cols:
            rows.append({"fold": fi, "feature": c, "q2_base": q2,
                         "q2_perm_mean": float(np.mean(drops[c]))})
    df = pd.DataFrame(rows)
    agg = df.groupby("feature").agg(
        q2_base_mean=("q2_base", "mean"),
        q2_perm_mean=("q2_perm_mean", "mean")).reset_index()
    agg["importance_q2_drop"] = agg["q2_base_mean"] - agg["q2_perm_mean"]
    agg.sort_values("importance_q2_drop", ascending=False) \
        .to_csv(out / "permutation_importance.csv", index=False)
    p25.log(f"  permutation importance done (composition Q2 median "
            f"{float(np.median(base_q2)):.3f})")

    # additive response curves from a spline+ridge fit on all data (descriptive)
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    spline = SplineTransformer(degree=int(cfg["models"]["spline"]["degree"]),
                               n_knots=int(cfg["models"]["spline"]["n_knots"]),
                               include_bias=False)
    Xb = spline.fit_transform(Xs)
    rid = Ridge(alpha=1.0).fit(Xb, Z)
    curves = []
    for j, c in enumerate(cols):
        grid = np.quantile(Xs[:, j], np.linspace(0.05, 0.95, 25))
        Xg = np.tile(Xs.mean(axis=0), (len(grid), 1))
        Xg[:, j] = grid
        zg = rid.predict(spline.transform(Xg))
        for r, g in enumerate(grid):
            for k in range(4):
                curves.append({"feature": c, "value_std": float(g),
                               "balance": f"z{k + 1}",
                               "response": float(zg[r, k])})
    pd.DataFrame(curves).to_csv(out / "additive_response_curves.csv",
                                index=False)


def _figures(cfg, summary, P, oof_comp, out):
    vkey = "src_gkf"
    sub = summary[(summary.cv_variant == vkey) & (summary.input_set == "A")]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    models = [m for m in ("ridge", "spline", "extratrees")
              if m in set(sub.model)]
    xpos = np.arange(4)
    for k, mname in enumerate(models):
        vals = [float(sub[sub.model == mname][f"R2_z{j}_median"].iloc[0])
                if len(sub[sub.model == mname]) else np.nan
                for j in range(1, 5)]
        ax.bar(xpos + (k - 1) * 0.26, vals, width=0.26, label=mname,
               color=f"C{k}")
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xticks(xpos, ["z1 fine/coarse", "z2 <8/8-16",
                         "z3 16-32/coarse", "z4 32-64/>=64"], fontsize=8)
    ax.set_ylabel("fold-median R2 per balance")
    ax.set_title("ILR balance predictability (src_gkf, input A)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out / "ilr_balance_predictability.png", dpi=cfg["plot"]["dpi"])
    plt.close(fig)

    oof = pd.DataFrame(oof_comp)
    oof = oof[(oof.model == "ridge") & (oof.input_set == "A")]
    true_p = pd.DataFrame(P, columns=[f"p_{b}_true" for b in p25.ILR_BANDS])
    true_p["dataset_index"] = np.arange(len(P))
    m = oof.merge(true_p, on="dataset_index", validate="one_to_one")
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.4))
    for ax, b in zip(axes, p25.ILR_BANDS):
        tcol, pcol = f"p_{b}_true", f"p_{b}_pred"
        lo = float(min(m[tcol].min(), m[pcol].min()))
        hi = float(max(m[tcol].max(), m[pcol].max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.scatter(m[tcol], m[pcol], s=8, alpha=0.6)
        ax.set_title(f"p_{b}", fontsize=9)
        ax.set_xlabel("true")
        if ax is axes[0]:
            ax.set_ylabel("OOF pred (ridge)")
    fig.suptitle("Predicted vs true composition (ridge, src_gkf OOF)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "predicted_vs_true_composition.png",
                dpi=cfg["plot"]["dpi"])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    s = summary[summary.cv_variant == vkey]
    labels, vals, errs = [], [], []
    for mname in ("ridge", "spline", "extratrees"):
        row = s[(s.target == "ilr_z1_z4") & (s.input_set == "A")
                & (s.model == mname)]
        if len(row):
            labels.append(mname)
            vals.append(float(row["Q2_median"].iloc[0]))
            errs.append(float(row["Q2_q75"].iloc[0]
                              - row["Q2_q25"].iloc[0]) / 2)
    ax.bar(labels, vals, yerr=errs, capsize=4,
           color=["C0", "C1", "C2"][:len(labels)])
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.axhline(cfg["gates"]["G1_multivariate_q2"], color="tab:red", ls="--",
               lw=1.0, label="G1 threshold")
    ax.set_ylabel("Aitchison Q2 (fold median, src_gkf, input A)")
    ax.set_title("Composition predictability vs G1 threshold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out / "spectrum_predictability_map.png", dpi=cfg["plot"]["dpi"])
    plt.close(fig)

    imp = pd.read_csv(out / "permutation_importance.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.barh(imp["feature"], imp["importance_q2_drop"], color="C0")
    ax.invert_yaxis()
    ax.set_xlabel("Aitchison Q2 drop under permutation (ET, src_gkf)")
    ax.set_title("Process feature importance for spectral composition")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(out / "process_feature_importance.png", dpi=cfg["plot"]["dpi"])
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
