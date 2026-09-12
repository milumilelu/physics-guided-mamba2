#!/usr/bin/env python3
"""Depth-conditioned spectral/amplitude contribution analysis (Task 3).

This is a descriptive, frozen-input analysis of the MAIN180 support
(120 formal + 60 pass_main; pass_supplement is excluded).  It asks two
paired questions using the same rows and the same predictors:

1. process -> spectrum/amplitude: how much does D, N or log10(Q) add after
   conditioning on the other two process coordinates?
2. spectrum/amplitude -> depth: after conditioning on N and log10(Q), does a
   spectral or amplitude descriptor explain residual depth variation?

The contribution statistic is the nested OLS partial R2.  Standardized beta
and partial Spearman correlation are reported as sign/shape diagnostics; no
causal interpretation is assigned.  Depth-quartile tables expose residual
N/Q associations within common depth support.  All support counts/ranges are
written to disk so the comparison cannot silently change population.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "outputs/phase2/manifest/phase2_manifest.csv"
OUT = REPO / "outputs/phase2_5/depth_conditioned_spectral"


def _ols_sse(X: np.ndarray, y: np.ndarray) -> float:
    if X.shape[1] == 0:
        return float(np.sum((y - y.mean()) ** 2))
    return float(np.sum((y - LinearRegression().fit(X, y).predict(X)) ** 2))


def _partial_row(df: pd.DataFrame, y_name: str, x_name: str,
                 controls: list[str], direction: str) -> dict:
    cols = [y_name, x_name] + controls
    z = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    n = len(z)
    if n < max(20, len(controls) + 8):
        return {"direction": direction, "target": y_name, "predictor": x_name,
                "controls": "+".join(controls), "n": n,
                "partial_r2": np.nan, "std_beta": np.nan,
                "partial_spearman": np.nan}
    y = z[y_name].to_numpy(float)
    x = z[x_name].to_numpy(float)
    c = z[controls].to_numpy(float) if controls else np.empty((n, 0))
    sc = StandardScaler()
    # Standardized coefficients put D, N and Q on one scale.  The intercept
    # remains implicit in LinearRegression and is therefore not a predictor.
    allx = np.column_stack([x, c])
    allx = sc.fit_transform(allx)
    yz = (y - y.mean()) / (y.std() if y.std() > 0 else 1.0)
    full = _ols_sse(allx, yz)
    reduced = _ols_sse(allx[:, 1:], yz)
    pr2 = (reduced - full) / reduced if reduced > 1e-12 else np.nan
    beta = float(LinearRegression().fit(allx, yz).coef_[0])
    # Partial Spearman: rank-transform each column, then residualize ranks.
    ranks = pd.DataFrame(np.column_stack([x, c])).rank(method="average").to_numpy(float)
    rx = ranks[:, 0]
    rc = ranks[:, 1:]
    ry = pd.Series(y).rank(method="average").to_numpy(float)
    if rc.shape[1]:
        ex = rx - LinearRegression().fit(rc, rx).predict(rc)
        ey = ry - LinearRegression().fit(rc, ry).predict(rc)
    else:
        ex, ey = rx - rx.mean(), ry - ry.mean()
    ps = float(np.corrcoef(ex, ey)[0, 1]) if np.std(ex) > 0 and np.std(ey) > 0 else np.nan
    return {"direction": direction, "target": y_name, "predictor": x_name,
            "controls": "+".join(controls), "n": n,
            "partial_r2": float(max(0.0, pr2)) if np.isfinite(pr2) else np.nan,
            "std_beta": beta, "partial_spearman": ps,
            "target_min": float(np.min(y)), "target_max": float(np.max(y)),
            "predictor_min": float(np.min(x)), "predictor_max": float(np.max(x))}


def _load() -> tuple[pd.DataFrame, list[str], dict]:
    man = pd.read_csv(MANIFEST)
    # MAIN180 is the predeclared common support.  Supplement rows are not
    # mixed into estimates because their process family is not balanced with
    # the formal/pass_main population.
    keep = man["session_role"].isin(["formal", "pass_main"])
    man = man.loc[keep].sort_values("dataset_index").reset_index(drop=True)
    comp = pd.read_csv(REPO / "outputs/phase2_5/spectral_composition/spectral_composition.csv")
    desc = pd.read_csv(REPO / "outputs/phase2_5/spectral_composition/spectrum_descriptor_summary.csv")
    morph = pd.read_csv(REPO / "outputs/phase1_5/morphology_descriptors.csv")
    dmet = pd.read_csv(REPO / "outputs/phase2_5/directional_spectrum/directional_metrics.csv")
    # Dataset-index joins prevent accidental positional mixing after support
    # filtering.  Directional metrics are reshaped into one row per sample.
    dwide = dmet.pivot(index="dataset_index", columns="band", values=["A2", "angular_entropy"])
    dwide.columns = [f"{a}_{b}" for a, b in dwide.columns]
    out = man[["dataset_index", "session_id", "session_role", "base_condition_group",
               "cv_process_group", "median_depth_um", "pass_count",
               "areal_dose_proxy_J_per_mm2"]].copy()
    # Merge only new columns; morphology descriptors repeat D, which is
    # already frozen in the Phase 2 manifest and must not create *_x/*_y
    # ambiguity.
    for table in (comp, desc, morph, dwide.reset_index()):
        add = table[[c for c in table.columns if c == "dataset_index" or c not in out.columns]]
        out = out.merge(add, on="dataset_index", how="left", validate="one_to_one")
    out["D"] = out["median_depth_um"].astype(float)
    out["N"] = out["pass_count"].astype(float)
    out["Q_log10"] = np.log10(out["areal_dose_proxy_J_per_mm2"].astype(float).clip(lower=1e-12))
    targets = [
        "p_lt8", "p_8_16", "p_16_32", "p_32_64", "p_64_inf",
        "dc_offset_frac", "dc_to_non_dc_ratio", "spectral_centroid_log_um",
        "spectral_entropy", "effective_band_number", "lambda_peak_um",
        "A2_8_16", "A2_16_32", "A2_32_64", "angular_entropy_8_16",
        "Sq_um", "Sa_um", "rms_DCT_8_16_um", "rms_DCT_16_32_um",
        "rms_DCT_32_64_um", "rms_DCT_64_inf_um", "E_DCT_8_16_frac",
        "E_DCT_16_32_frac", "E_DCT_32_64_frac", "E_DCT_64_inf_frac",
    ]
    targets = [t for t in targets if t in out.columns]
    return out, targets, {"support_label": "MAIN180", "excluded_session_role": "pass_supplement"}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df, targets, scope = _load()
    predictors = ["D", "N", "Q_log10"]
    rows = []
    # process -> each spectral/amplitude metric, with the other process
    # coordinates as controls
    for y in targets:
        for x in predictors:
            rows.append(_partial_row(df, y, x, [c for c in predictors if c != x], "process_to_spectrum"))
    # reverse direction: each metric -> depth after N and Q are controlled
    for x in targets:
        rows.append(_partial_row(df, "D", x, ["N", "Q_log10"], "spectrum_to_depth"))
    contrib = pd.DataFrame(rows)
    contrib.to_csv(OUT / "conditional_contributions_long.csv", index=False)
    # Bidirectional matrix: forward process coordinates and reverse metric->D
    fwd = contrib[contrib.direction == "process_to_spectrum"].pivot(index="target", columns="predictor", values="partial_r2")
    fwd = fwd.rename(columns={"D": "D_to_metric__given_N_Q", "N": "N_to_metric__given_D_Q", "Q_log10": "Q_to_metric__given_D_N"})
    revtab = contrib[contrib.direction == "spectrum_to_depth"].set_index("predictor")
    rev = revtab["partial_r2"].rename("metric_to_D__given_N_Q")
    revb = revtab["std_beta"].rename("metric_to_D_std_beta__given_N_Q")
    revs = revtab["partial_spearman"].rename("metric_to_D_partial_spearman__given_N_Q")
    matrix = fwd.join([rev, revb, revs], how="left").reset_index().rename(columns={"target": "metric"})
    matrix.to_csv(OUT / "bidirectional_condition_contribution_matrix.csv", index=False)
    # Common-support ranges and count are explicit in a machine-readable file.
    support = {"n": int(len(df)), "dataset_index_min": int(df.dataset_index.min()),
               "dataset_index_max": int(df.dataset_index.max()),
               "depth_um": {"min": float(df.D.min()), "max": float(df.D.max())},
               "pass_count": {"min": float(df.N.min()), "max": float(df.N.max())},
               "dose_proxy_J_per_mm2": {"min": float(df.areal_dose_proxy_J_per_mm2.min()), "max": float(df.areal_dose_proxy_J_per_mm2.max())},
               "dose_log10": {"min": float(df.Q_log10.min()), "max": float(df.Q_log10.max())},
               "session_role_counts": df.session_role.value_counts().to_dict()}
    (OUT / "support_scope.json").write_text(json.dumps({**scope, **support}, indent=2), encoding="utf-8")
    # Process-coordinate dependence is part of the identification boundary.
    # Report both Pearson and Spearman correlations, plus linear VIF, rather
    # than silently treating D/N/Q as orthogonal interventions.
    pc = df[predictors].astype(float)
    corr = []
    for method, cmat in (("pearson", pc.corr(method="pearson")),
                         ("spearman", pc.corr(method="spearman"))):
        for a in predictors:
            for b in predictors:
                if a < b:
                    corr.append({"method": method, "predictor_a": a,
                                 "predictor_b": b, "correlation": float(cmat.loc[a, b])})
    vifs = []
    for x in predictors:
        others = [c for c in predictors if c != x]
        r2 = 1.0 - _ols_sse(StandardScaler().fit_transform(pc[others]),
                             StandardScaler().fit_transform(pc[[x]]).ravel()) / (len(pc) - 1)
        vifs.append({"predictor": x, "linear_vif": float(1.0 / max(1e-12, 1.0 - r2))})
    pd.DataFrame(corr).to_csv(OUT / "process_coordinate_dependence.csv", index=False)
    pd.DataFrame(vifs).to_csv(OUT / "process_coordinate_vif.csv", index=False)
    # Within-depth-bin table: N and Q effects that remain after restricting to
    # one depth quartile.  This is descriptive and deliberately reports n.
    df["depth_bin"] = pd.qcut(df.D, q=4, labels=["D_Q1", "D_Q2", "D_Q3", "D_Q4"], duplicates="drop")
    strata = []
    for b, g in df.groupby("depth_bin", observed=True):
        for y in targets:
            for x in ["N", "Q_log10"]:
                z = g[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(z) >= 8 and z[x].nunique() > 1 and z[y].nunique() > 1:
                    rho, p = spearmanr(z[x], z[y])
                else:
                    rho, p = np.nan, np.nan
                strata.append({"depth_bin": str(b), "metric": y, "predictor": x,
                               "n": len(z), "spearman": rho, "spearman_p": p,
                               "depth_min": float(g.D.min()), "depth_max": float(g.D.max())})
    pd.DataFrame(strata).to_csv(OUT / "within_depth_quartile_associations.csv", index=False)
    # Human-readable interpretation with ranked effects, avoiding causal words.
    top = matrix.assign(max_forward=matrix[[c for c in matrix.columns if c.startswith(("D_to", "N_to", "Q_to"))]].max(axis=1))
    lines = ["# Depth-conditioned spectral/amplitude analysis (MAIN180)", "",
             f"Support: n={len(df)} ({scope['support_label']}); excluded `pass_supplement` rows. Predictors are D=median depth (µm), N=pass count, and Q=log10(areal dose proxy).", "",
             "Partial R² is a nested descriptive contribution: each process coordinate is added after the other two are fitted. Reverse columns report metric → D after N and Q are fitted. With the same linear model and controls, forward/reverse partial R² are mathematically symmetric; the reverse standardized beta and partial Spearman columns expose orientation and sign. Process-coordinate dependence is reported in `process_coordinate_dependence.csv` and `process_coordinate_vif.csv`. Signs/shape checks are in `conditional_contributions_long.csv`; depth-quartile associations are in `within_depth_quartile_associations.csv`.", "",
             "## Largest conditional contributions"]
    long = contrib[contrib.direction == "process_to_spectrum"].dropna(subset=["partial_r2"]).sort_values("partial_r2", ascending=False).head(12)
    for _, r in long.iterrows():
        lines.append(f"- `{r.target}`: `{r.predictor}` partial R²={r.partial_r2:.3f}, standardized beta={r.std_beta:.3f}, partial Spearman={r.partial_spearman:.3f} (n={int(r.n)}).")
    lines += ["", "These are support-limited associations, not causal effects or evidence that a descriptor is a sufficient state. Any follow-up intervention must preserve the same depth/process support and measure repeated depth noise.", ""]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT} (n={len(df)}, metrics={len(targets)}, rows={len(contrib)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
