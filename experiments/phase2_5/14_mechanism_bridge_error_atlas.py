#!/usr/bin/env python3
"""Phase 2.5 Task 14: mechanism bridge (14A) + OOF error atlas (14B).

14A — mechanism bridge: the mechanism module's PhysicsParams are FIXED
constants (threshold/incubation/delta_eff lookups; TorchPhysicsModel fitting
is NOT used). Label-free, process-only mechanism summaries are computed for
the 200 real rows; provenance table carries the three states
(APPLICABLE / NOT_APPLICABLE / REDUNDANT_WITH_C). Bridge = M0: z ~ A vs
M1: z ~ [A, m(u)] on identical grouped folds; even a positive dQ2 only means
"mechanistic transformation provides incremental inductive bias".

14B — error atlas: OOF predictions from Task 12 (primary = ridge, ET =
sensitivity, src_gkf). Scalar errors normalized by the TRAINING-fold IQR;
composition error = Aitchison distance. Process-coverage kNN, Moran I
(Monte-Carlo permutation, 10000), diagnostics, hotspots with the cross-model
robustness rule: a hotspot is "model-robust unresolved" only if the elevated
error persists under ET; targets where Task 12 showed ET-Ridge >= 0.1 with
>=4/5 folds same sign get their Ridge hotspots labelled
"linear-baseline hotspot" (细则 §0.15).

Seed offsets: Moran permutation = seed + 900, kNN/coverage deterministic.
"""

from __future__ import annotations

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import _lib as p25

EXPECTED = ["mechanism_feature_provenance.csv", "mechanism_features_real200.csv",
            "mechanism_bridge_cv.csv", "mechanism_bridge_summary.csv",
            "oof_error_atlas.csv", "process_density.csv",
            "error_density_association.csv", "error_moran_test.csv",
            "error_hotspots.csv", "error_process_map.png",
            "error_vs_density.png", "mechanism_increment.png", "README.md"]

PHI_TH1_J_CM2 = 16.3967     # mechanism module fixed constants (frozen)
INCUBATION_S = 0.79652
W0_UM = 0.874               # v2 §11 nominal 1/e^2 waist radius

README = """# mechanism bridge + OOF error atlas (Task 14)

- `mechanism_feature_provenance.csv`: three-state provenance
  (APPLICABLE / NOT_APPLICABLE / REDUNDANT_WITH_C). The mechanism constants
  (threshold 16.3967 J/cm2, incubation S=0.79652, delta_eff lookup) are fixed
  dataclass values in the mechanism module; the fitted TorchPhysicsModel
  recursion is NOT used (label dependency). Caveat: the constants' original
  calibration predates this repo.
- `mechanism_bridge_summary.csv`: dQ2_mech = Q2(z ~ [A, m(u)]) - Q2(z ~ A),
  fold-paired on identical grouped folds. Positive only means the mechanistic
  transformation adds inductive bias — never "E1/E2/E5 verified".
- `oof_error_atlas.csv`: per-sample OOF errors (ridge primary, ET
  sensitivity) normalized by the training-fold IQR; composition error =
  Aitchison distance.
- Hotspot rule: "model-robust unresolved" requires the elevated-error pattern
  to persist under ET (error Spearman + top-10% Jaccard reported); targets
  with ET-Ridge >= 0.1 (>=4/5 folds) in Task 12 are labelled
  "linear-baseline hotspot" instead.
"""


def _mechanism_features(man: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame({"dataset_index": man["dataset_index"].astype(int)})
    tau = man["pulse_duration_fs"].to_numpy(float)
    f_khz = man["frequency_kHz"].to_numpy(float)
    n = man["pass_count"].to_numpy(float)
    v = man["velocity_mm_s"].to_numpy(float)
    h = man["hatch_spacing_um"].to_numpy(float)
    ep_uJ = 1000.0 * 5.3333 / f_khz                    # proxy (v2 §11)
    fluence_j_cm2 = (ep_uJ * 1e-6) / (np.pi * W0_UM ** 2 * 1e-8)  # uJ->J, um^2->cm^2
    keys = np.array([200.0, 500.0, 1000.0, 2000.0, 4000.0])
    dvals = np.array([0.043899, 0.0450921, 0.0284155, 0.0164486, 0.0333915])
    delta_eff = dvals[np.argmin(np.abs(keys[:, None] - tau[None, :]), axis=0)]
    phi_th = PHI_TH1_J_CM2 * INCUBATION_S ** (n - 1.0)
    feat["pulse_fluence_proxy_j_cm2"] = fluence_j_cm2
    feat["incubation_threshold_j_cm2"] = phi_th
    feat["log_fluence_margin_proxy"] = np.log(fluence_j_cm2 / phi_th)
    feat["delta_eff_tau_um"] = delta_eff
    feat["scan_spacing_um"] = v / f_khz
    feat["areal_dose_proxy_j_mm2"] = 1000.0 * 5.3333 * n / (v * h)
    return feat


def main() -> int:
    cfg, quick = p25.load_config(__doc__)
    t0 = time.time()
    out_m = p25.output_dir(cfg, "mechanism_bridge")
    out_e = p25.output_dir(cfg, "error_atlas")
    seed = int(cfg["random_seed"])
    p25.log("== Phase 2.5 / 14: mechanism bridge + error atlas ==")
    man = p25.read_phase2_manifest(cfg)
    ilr = pd.read_csv(p25.output_dir(cfg, "spectral_composition")
                      / "ilr_coordinates.csv")
    Z = ilr[[f"ilr_z{j}" for j in range(1, 5)]].to_numpy(float)
    comp = pd.read_csv(p25.output_dir(cfg, "spectral_composition")
                       / "spectral_composition.csv")
    P = comp[[f"p_{b}" for b in p25.ILR_BANDS]].to_numpy(float)
    tgt05 = pd.read_csv(p25.l15.REPO
                        / "outputs/phase1_5/morphology_descriptors.csv")

    # ---- 14A: provenance + features + bridge --------------------------------
    prov = pd.DataFrame([
        {"feature_name": "pulse_fluence_proxy_j_cm2",
         "physical_meaning": "per-pulse fluence from post-objective power "
                             "proxy and fixed waist w0=0.874um",
         "source_code": "mechanism_virtual_augmentation/"
                        "depth_mechanism_transition_virtual_data_v2.py "
                        "(OpticalConfig) + phase2_5/14",
         "depends_only_on_process_controls": True,
         "depends_on_fixed_constants": True,
         "depends_on_measured_depth": False,
         "depends_on_measured_morphology": False,
         "was_fitted_using_labels": False, "fit_scope": "n/a",
         "allowed_primary": False,
         "redundant_with_C": True,
         "notes": "monotone in pulse_energy_proxy_uJ (already in C)"},
        {"feature_name": "incubation_threshold_j_cm2",
         "physical_meaning": "incubation-law threshold phi_th(N)=phi_th1*S^(N-1)",
         "source_code": "mechanism module PhysicsParams (frozen constants)",
         "depends_only_on_process_controls": True,
         "depends_on_fixed_constants": True,
         "depends_on_measured_depth": False,
         "depends_on_measured_morphology": False,
         "was_fitted_using_labels": False, "fit_scope": "n/a",
         "allowed_primary": False, "redundant_with_C": True,
         "notes": "function of pass_count only; constant original "
                  "calibration predates this repo"},
        {"feature_name": "log_fluence_margin_proxy",
         "physical_meaning": "log(fluence / incubation threshold) — ablation "
                             "margin proxy",
         "source_code": "mechanism module recursion formula (margin), "
                        "label-free variant",
         "depends_only_on_process_controls": True,
         "depends_on_fixed_constants": True,
         "depends_on_measured_depth": False,
         "depends_on_measured_morphology": False,
         "was_fitted_using_labels": False, "fit_scope": "n/a",
         "allowed_primary": True, "redundant_with_C": False,
         "notes": "log(E_p) - (N-1)log(S): nonlinear combination outside C's "
                  "linear span"},
        {"feature_name": "delta_eff_tau_um",
         "physical_meaning": "pulse-width-conditioned effective ablation scale",
         "source_code": "mechanism module PhysicsParams.delta_for_tau",
         "depends_only_on_process_controls": True,
         "depends_on_fixed_constants": True,
         "depends_on_measured_depth": False,
         "depends_on_measured_morphology": False,
         "was_fitted_using_labels": False, "fit_scope": "n/a",
         "allowed_primary": True, "redundant_with_C": False,
         "notes": "fixed lookup by pulse duration"},
        {"feature_name": "scan_spacing_um / areal_dose_proxy_j_mm2",
         "physical_meaning": "already present in input set C",
         "source_code": "phase2 _lib derived coordinates",
         "depends_only_on_process_controls": True,
         "depends_on_fixed_constants": True,
         "depends_on_measured_depth": False,
         "depends_on_measured_morphology": False,
         "was_fitted_using_labels": False, "fit_scope": "n/a",
         "allowed_primary": False, "redundant_with_C": True, "notes": ""},
    ])
    prov.to_csv(out_m / "mechanism_feature_provenance.csv", index=False)

    feat = _mechanism_features(man)
    feat.to_csv(out_m / "mechanism_features_real200.csv", index=False)
    prim_feats = prov[prov.allowed_primary == True]["feature_name"].tolist()  # noqa: E712
    p25.log(f"  14A provenance: APPLICABLE features {prim_feats}")

    X_A = man[p25.p2.PROC_RAW_COLS].to_numpy(float)
    Xm = feat[prim_feats].to_numpy(float)
    X_AC = np.hstack([X_A, feat[[c for c in feat.columns
                                 if c != "dataset_index"]].to_numpy(float)])
    src_groups = man["shared_height_source_id"].to_numpy()
    proc_groups = man["cv_process_group"].to_numpy()
    variants = {"src_gkf": p25.gkf_splits(src_groups, int(cfg["cv"]["n_splits"])),
                "proc_gkf": p25.gkf_splits(proc_groups, int(cfg["cv"]["n_splits"]))}
    for g, sp in ((src_groups, variants["src_gkf"]),
                  (proc_groups, variants["proc_gkf"])):
        p25.check_gkf_contract(g, sp)

    def _ridge_pred(Xtr_raw, Xte_raw, ytr, groups_tr):
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import GroupKFold
        best = 1.0
        if len(set(groups_tr.tolist())) >= 3:
            grid = cfg["models"]["ridge_alpha_grid"]
            scores = {a: [] for a in grid}
            for itr, ival in GroupKFold(n_splits=3).split(Xtr_raw,
                                                          groups=groups_tr):
                if len(np.unique(ytr[ival])) < 2:
                    continue
                scc = StandardScaler().fit(Xtr_raw[itr])
                Xit, Xiv = scc.transform(Xtr_raw[itr]), scc.transform(
                    Xtr_raw[ival])
                for a in grid:
                    mm = Ridge(alpha=float(a)).fit(Xit, ytr[itr])
                    scores[a].append(r2_score(ytr[ival], mm.predict(Xiv)))
            med = {a: (np.nanmedian(v) if v else -np.inf)
                   for a, v in scores.items()}
            best = grid[0]
            for a in grid[1:]:
                if med[a] > med[best]:
                    best = a
        sc = StandardScaler().fit(Xtr_raw)
        return Ridge(alpha=float(best)).fit(sc.transform(Xtr_raw),
                                            ytr).predict(sc.transform(Xte_raw))

    bridge_rows = []
    for vname, splits in variants.items():
        groups = src_groups if vname == "src_gkf" else proc_groups
        for fi, (tr, te) in enumerate(splits):
            q2_m0 = p25._q2_aitchison if hasattr(p25, "_q2_aitchison") else None
            from importlib import import_module as _imp
            e12 = _imp("phase2_5_12") if "phase2_5_12" in __import__("sys").modules \
                else None
            # compute Q2 inline (same definition as Task 12)
            def q2_of(zp):
                denom = float(((Z[te] - Z[tr].mean(axis=0)) ** 2).sum())
                if denom <= 0:
                    return np.nan
                return float(1.0 - ((Z[te] - zp) ** 2).sum() / denom)
            z_m0 = _ridge_pred(X_A[tr], X_A[te], Z[tr], groups[tr])
            q2_0 = q2_of(z_m0)
            z_m1 = _ridge_pred(np.hstack([X_A, Xm])[tr],
                               np.hstack([X_A, Xm])[te], Z[tr], groups[tr])
            q2_1 = q2_of(z_m1)
            bridge_rows.append({"cv_variant": vname, "fold": fi,
                                "Q2_M0_A": q2_0, "Q2_M1_A_plus_mech": q2_1,
                                "dQ2_mech": q2_1 - q2_0})
            p25.log(f"  [bridge {vname}] fold {fi}: Q2_M0={q2_0:.3f} "
                    f"Q2_M1={q2_1:.3f}")
    bridge = pd.DataFrame(bridge_rows)
    bridge.to_csv(out_m / "mechanism_bridge_cv.csv", index=False)
    bsum = bridge.groupby("cv_variant").agg(
        dQ2_mech_median=("dQ2_mech", "median"),
        dQ2_q25=("dQ2_mech", lambda s: s.quantile(0.25)),
        dQ2_q75=("dQ2_mech", lambda s: s.quantile(0.75)),
        n_pos=("dQ2_mech", lambda s: int((s > 0).sum())),
        n_folds=("dQ2_mech", "size")).reset_index()
    bsum["state"] = "APPLICABLE"
    gate = float(cfg["gates"]["G3_delta_q2"])
    bsum["G3b"] = np.where((bsum.dQ2_mech_median >= gate)
                           & (bsum.n_pos >= 4), "SUPPORTED",
                           "NOT SUPPORTED")
    bsum.to_csv(out_m / "mechanism_bridge_summary.csv", index=False)

    # ---- 14B: OOF error atlas ------------------------------------------------
    oof_c = pd.read_csv(p25.output_dir(cfg, "process_map")
                        / "composition_oof_predictions.csv")
    oof_d = pd.read_csv(p25.output_dir(cfg, "process_map")
                        / "directional_oof_predictions.csv")
    splits = variants["src_gkf"]
    fold_of = np.empty(len(man), dtype=int)
    for fi, (tr, te) in enumerate(splits):
        fold_of[te] = fi
    true_p = pd.DataFrame(P, columns=[f"p_{b}_true" for b in p25.ILR_BANDS])
    true_p["dataset_index"] = np.arange(len(man))
    scal_all = {"Sq_um": tgt05["Sq_um"].to_numpy(float),
                "median_depth_um": man["median_depth_um"].to_numpy(float),
                "A2_8_16": pd.read_csv(
                    p25.output_dir(cfg, "directional_spectrum")
                    / "directional_metrics.csv")
                    .query("band == '8_16'")["A2"].to_numpy(float)}

    # nonlin flags from Task 12 (linear-baseline hotspot labelling)
    nl_path = p25.output_dir(cfg, "process_map") / "nonlinear_comparison.csv"
    lin_base_targets = set()
    if nl_path.exists():
        nl = pd.read_csv(nl_path)
        rule = cfg["error_atlas"]["linear_baseline_rule"]
        ok = nl[(nl.target.isin(scal_all))
                & (nl.input_set == "A")
                & (nl.cv_variant == "src_gkf")
                & (nl["dR2_ET-ridge_median"] >= rule["dR2_et_ridge"])
                & (nl["dR2_ET-ridge_n_pos"] >= rule["min_folds_same_sign"])]
        lin_base_targets = set(ok["target"])

    atlas_parts = []
    for mname in ("ridge", "extratrees"):
        oc = oof_c[(oof_c.model == mname) & (oof_c.input_set == "A")
                   & (oof_c.cv_variant == "src_gkf")].sort_values("dataset_index")
        ph = oc[[f"p_{b}_pred" for b in p25.ILR_BANDS]].to_numpy(float)
        idx = oc["dataset_index"].to_numpy(int)
        dA_comp = p25.aitchison_distance(ph, P[idx])
        od = oof_d[(oof_d.model == mname) & (oof_d.input_set == "A")
                   & (oof_d.cv_variant == "src_gkf")]
        err_rows = []
        for tid, y in scal_all.items():
            sub = od[od.target == tid].sort_values("dataset_index")
            y_true = sub["y_true"].to_numpy(float)
            y_pred = sub["y_pred"].to_numpy(float)
            # per-fold train IQR
            norm_err = np.empty(len(sub))
            for fi, (tr, te) in enumerate(splits):
                iqr_f = float(np.subtract(*np.percentile(y[tr], [75, 25])))
                sel = sub["fold"].to_numpy() == fi
                norm_err[sel] = (np.abs(y_true[sel] - y_pred[sel])
                                 / max(iqr_f, 1e-300))
            part = pd.DataFrame({"dataset_index": idx, "model": mname,
                                 "target": tid, "fold": sub["fold"].to_numpy(),
                                 "y_true": y_true, "y_pred": y_pred,
                                 "abs_error": np.abs(y_true - y_pred),
                                 "norm_abs_error": norm_err})
            err_rows.append(part)
        cpart = pd.DataFrame({"dataset_index": idx, "model": mname,
                              "target": "ilr_composition",
                              "fold": fold_of[idx],
                              "abs_error": dA_comp,
                              "norm_abs_error": dA_comp})
        err_rows.append(cpart)
        atlas_parts.append(pd.concat(err_rows))
    atlas = pd.concat(atlas_parts).sort_values(
        ["model", "target", "dataset_index"])
    atlas.to_csv(out_e / "oof_error_atlas.csv", index=False)

    # coverage density (standardized raw A space)
    sc = StandardScaler().fit(X_A)
    Xs = sc.transform(X_A)
    dens_rows = []
    for k in cfg["error_atlas"]["knn_k"]:
        d_k = p25.p2.knn_median_distance(Xs, int(k)) \
            if hasattr(p25.p2, "knn_median_distance") else None
        if d_k is None:
            D = np.sqrt(((Xs[:, None, :] - Xs[None, :, :]) ** 2).sum(-1))
            np.fill_diagonal(D, np.inf)
            part = np.partition(D, int(k) - 1, axis=1)[:, :int(k)]
            d_k = np.median(part, axis=1)
        dens_rows.append(pd.DataFrame({"dataset_index": np.arange(len(man)),
                                       "k": int(k),
                                       "d_proc_knn": d_k}))
    dens = pd.concat(dens_rows)
    dens.to_csv(out_e / "process_density.csv", index=False)

    assoc_rows = []
    for (mname, tid), sub in atlas.groupby(["model", "target"]):
        e = sub.sort_values("dataset_index")["norm_abs_error"].to_numpy()
        for k in cfg["error_atlas"]["knn_k"]:
            dk = dens[dens.k == int(k)].sort_values("dataset_index")[
                "d_proc_knn"].to_numpy()
            assoc_rows.append({"model": mname, "target": tid, "k": int(k),
                               "spearman_error_vs_dproc": float(
                                   spearmanr(e, dk).statistic)})
    pd.DataFrame(assoc_rows).to_csv(out_e / "error_density_association.csv",
                                    index=False)

    # Moran I per (model, target)
    moran_rows = []
    for (mname, tid), sub in atlas.groupby(["model", "target"]):
        e = sub.sort_values("dataset_index")["norm_abs_error"].to_numpy()
        i_obs, p_val = p25.moran_permutation_p(
            e, p25.knn_row_standardized_graph(Xs, 5),
            int(cfg["error_atlas"]["moran_permutations"]), seed + 900)
        moran_rows.append({"model": mname, "target": tid, "moran_I": i_obs,
                           "p_perm": p_val})
    moran = pd.DataFrame(moran_rows)
    moran["G5_candidate"] = moran["p_perm"] <= cfg["gates"]["G5_moran_p"]
    moran.to_csv(out_e / "error_moran_test.csv", index=False)

    # cross-model robustness + hotspots
    hot_rows = []
    q = float(cfg["gates"]["hotspot_quantile"])
    for tid in set(atlas.target):
        r = atlas[(atlas.model == "ridge") & (atlas.target == tid)] \
            .sort_values("dataset_index")
        e = atlas[(atlas.model == "extratrees") & (atlas.target == tid)] \
            .sort_values("dataset_index")
        thr_r = np.quantile(r["norm_abs_error"], q)
        thr_e = np.quantile(e["norm_abs_error"], q)
        hs_r = set(r.loc[r["norm_abs_error"] >= thr_r, "dataset_index"])
        hs_e = set(e.loc[e["norm_abs_error"] >= thr_e, "dataset_index"])
        jacc = len(hs_r & hs_e) / max(len(hs_r | hs_e), 1)
        sp = float(spearmanr(r["norm_abs_error"], e["norm_abs_error"]).statistic)
        tag = "unresolved_candidate"
        if tid in lin_base_targets:
            tag = "linear-baseline hotspot"
        elif jacc < 0.3 or sp < 0.2:
            tag = "model-sensitive (not robust unresolved)"
        for gi in sorted(hs_r):
            mrow = man.iloc[gi]
            hot_rows.append({"target": tid, "dataset_index": int(gi),
                             "norm_abs_error_ridge": float(
                                 r.loc[r.dataset_index == gi,
                                       "norm_abs_error"].iloc[0]),
                             "in_et_hotspot": gi in hs_e,
                             "hotspot_jaccard_ridge_et": jacc,
                             "error_spearman_ridge_et": sp,
                             "label": tag,
                             "session_id": mrow["session_id"],
                             "repair_fraction": mrow["repair_fraction"],
                             "plane_rmse_um": mrow["plane_rmse_um"]})
    pd.DataFrame(hot_rows).to_csv(out_e / "error_hotspots.csv", index=False)

    # ---- figures ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bs = bsum.sort_values("cv_variant")
    ax.bar(bs.cv_variant, bs.dQ2_mech_median,
           yerr=[bs.dQ2_mech_median - bs.dQ2_q25,
                 bs.dQ2_q75 - bs.dQ2_mech_median], capsize=4)
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.axhline(cfg["gates"]["G3_delta_q2"], color="tab:red", ls="--", lw=1.0)
    ax.set_ylabel("dQ2_mech (M1 - M0, fold median)")
    ax.set_title("Mechanism feature incremental value (G3b threshold dashed)")
    fig.tight_layout()
    fig.savefig(out_e / "mechanism_increment.png", dpi=cfg["plot"]["dpi"])
    plt.close(fig)

    at = atlas[(atlas.model == "ridge") & (atlas.target == "Sq_um")]
    d1 = dens[dens.k == 5].sort_values("dataset_index")
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.scatter(d1["d_proc_knn"], at["norm_abs_error"], s=14, alpha=0.7)
    rho = float(spearmanr(d1["d_proc_knn"],
                          at["norm_abs_error"]).statistic)
    ax.set_xlabel("process coverage: kNN distance (k=5, std space)")
    ax.set_ylabel("Sq OOF normalized error (ridge)")
    ax.set_title(f"Error vs process coverage (Spearman {rho:+.2f})")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_e / "error_vs_density.png", dpi=cfg["plot"]["dpi"])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    sc_sorted = at.sort_values("dataset_index")
    ax.scatter(sc_sorted.dataset_index, sc_sorted.norm_abs_error, s=10,
               alpha=0.7, label="Sq_um")
    ax.set_xlabel("dataset index (sorted by depth)")
    ax.set_ylabel("normalized OOF error (ridge)")
    ax.set_title("Sq error atlas (ridge primary)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_e / "error_process_map.png", dpi=cfg["plot"]["dpi"])
    plt.close(fig)

    (out_m / "README.md").write_text(README, encoding="utf-8")
    (out_e / "README.md").write_text(README, encoding="utf-8")
    missing = [f for f in EXPECTED if not (out_m / f).exists()
               and not (out_e / f).exists()]
    p25.require(not missing, f"missing outputs: {missing}")
    p25.log(f"14 done in {time.time() - t0:.1f}s; all outputs present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
