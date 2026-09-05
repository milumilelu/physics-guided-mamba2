#!/usr/bin/env python3
"""Task 24 (Phase 2.8A): hierarchical information channels ->
Predictability Spectrum + realization diagnostic.

Channels (v2.1 section 2.1): D (median_depth_um), A (Sq of the residual,
parity-checked against manifest residual_Sq_um), P_lambda (ILR z1..z4,
Aitchison), O_theta (A2_8_16 / angular_entropy_8_16 as separate scalar rows,
joint standardized 2-D as secondary summary).  phi(k_x,k_y) is NOT
regressed; it is covered by the shift-invariant phase-only realization
diagnostic (section 2.8).

Unified protocol (v2.1 section 2.2): one frozen folds artifact
(src_gkf = GroupKFold(5) on shared_height_source_id, proc_gkf = GroupKFold(5)
on cv_process_group -- historical semantics, F1), three models per target
(M_full / M_h / M_-h) plus a train-mean dummy, Ridge with fold-internal
StandardScaler and target-native inner alpha (src.cv.select_alpha_inner,
inner GKF(5) on cv_process_group -- frozen Task 22 semantics), outer skill
= Q^2 with train-mean null; the O_theta joint summary is computed on
fold-internally standardized coordinates (equal weighting, no leakage).
Sensitivities: in-box 101 (sub-splits, Task 22 semantics) and the 16-32 um
directional band.  raw/repaired sensitivity is registered N/A after
verification: the rectangle dataset has exactly one registered height field
(frozen height_raw == registered H_stable_raw, checked for all 200 ROIs);
cone repair is a single-line observation-operator concern (quantified in
outputs/cone_repair_impact/).  No historical numbers are reused
(G28-A condition 9).

G28-A = VALID/INVALID QA completion gate (nine conditions, section 2.6).

Expected artifacts (formal):
    outputs/phase2_8/folds/fold_assignments.csv
    outputs/phase2_8/predictability_spectrum_folds.csv
    outputs/phase2_8/predictability_spectrum.csv
    outputs/phase2_8/predictability_spectrum.png
    outputs/phase2_8/realization_diagnostic.csv
    outputs/phase2_8/summary/gsl28_a_evaluation.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src import data as sdata  # noqa: E402
from src import geometry as sgeo  # noqa: E402
from src import provenance as prov  # noqa: E402
from src import spectrum as sspec  # noqa: E402
from src.cv import (check_gkf_contract, gkf_splits, make_ridge,  # noqa: E402
                    select_alpha_inner)

EXPECTED = [
    "folds/fold_assignments.csv",
    "predictability_spectrum_folds.csv",
    "predictability_spectrum.csv",
    "predictability_spectrum.png",
    "realization_diagnostic.csv",
    "summary/gsl28_a_evaluation.json",
]

SCALARS = {
    "D": "median_depth_um",
    "A": "residual_Sq_um",
    "Ot_A2": "A2_8_16",
    "Ot_ent": "angular_entropy_8_16",
    "Ot_A2_16_32": "A2_16_32",
    "Ot_ent_16_32": "angular_entropy_16_32",
}
JOINT_COLS = ["A2_8_16", "angular_entropy_8_16"]
PRIMARY = ["D", "A", "Pl", "Ot_A2", "Ot_ent"]


def log(msg: str = "") -> None:
    prov.log(msg)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# targets
# --------------------------------------------------------------------------- #

def load_targets(cfg: dict, frozen: dict) -> tuple[pd.DataFrame, dict]:
    # grouping + u columns + D/A all live on the frozen phase2 manifest;
    # row order must align with the frozen NPZ stack (load_frozen contract)
    p2man = pd.read_csv(REPO / cfg["paths"]["phase2_manifest"]).sort_values(
        "dataset_index").reset_index(drop=True)
    prov.require(len(p2man) == 200, "phase2 manifest rows != 200")
    fman = frozen["man"]
    prov.require((p2man["session_id"].to_numpy()
                  == fman["session_id"].to_numpy()).all()
                 and (p2man["sample_id"].to_numpy()
                      == fman["sample_id"].to_numpy()).all(),
                 "phase2 manifest row order != frozen NPZ order")
    df = p2man[["dataset_index", "session_id", "sample_id",
                "shared_height_source_id", "cv_process_group",
                "median_depth_um", "residual_Sq_um",
                "pulse_duration_fs", "frequency_kHz", "hatch_spacing_um",
                "pass_count", "velocity_mm_s"]].copy()

    # A parity / reconciliation: manifest residual_Sq_um vs Sq of the frozen
    # residual.  Observed (registered finding): every row differs by a
    # UNIFORM ~3e-5 relative drift (corr = 1.0, repair-independent) --
    # the phase-1 manifest stats predate the v1 field re-registration.
    # Channel decision: D and A are recomputed from the frozen fields so all
    # four channels derive from the same stack the spectral targets use.
    R, V = frozen["R"], frozen["V"]
    Hnan = frozen["Hnan"]
    sq_fields = np.sqrt(np.nanmean(np.where(V, R, np.nan) ** 2,
                                   axis=(1, 2)))
    d_fields = -np.nanmedian(np.where(V, Hnan, np.nan), axis=(1, 2))
    sq_manifest = df["residual_Sq_um"].to_numpy(float)
    med_manifest = df["median_depth_um"].to_numpy(float)
    rel = np.abs(sq_fields - sq_manifest) / np.maximum(np.abs(sq_manifest),
                                                       1e-300)
    corr_sq = float(np.corrcoef(sq_fields, sq_manifest)[0, 1])
    prov.require(corr_sq > 1.0 - 1e-6 and float(np.max(rel)) < 1e-3,
                 f"manifest Sq is not a near-copy of the field Sq "
                 f"(corr={corr_sq}, max_rel={np.max(rel)}) -- wrong column?")
    df["median_depth_um"] = d_fields
    df["residual_Sq_um"] = sq_fields
    recon = {
        "status": "manifest_stats_predate_v1_field_registration",
        "decision": "D/A recomputed from frozen fields (same stack as the "
                    "spectral channels); manifest columns kept for audit",
        "max_abs_diff_sq": float(np.max(np.abs(sq_fields - sq_manifest))),
        "max_rel_diff_sq": float(np.max(rel)),
        "corr_sq": corr_sq,
        "max_abs_diff_depth": float(np.max(np.abs(d_fields - med_manifest))),
        "n_rows_differing": int((rel > 1e-12).sum()),
    }

    ilr = pd.read_csv(REPO / cfg["paths"]["ilr_csv"])
    df = df.merge(ilr, on="dataset_index", how="left")
    dm = pd.read_csv(REPO / cfg["paths"]["directional_csv"])
    for b, suffix in (("8_16", "8_16"), (cfg["task24"]["sensitivity_band"],
                                         "16_32")):
        sub = dm[dm["band"] == b][["dataset_index", "A2", "angular_entropy"]]
        sub = sub.rename(columns={"A2": f"A2_{suffix}",
                                  "angular_entropy": f"angular_entropy_{suffix}"})
        df = df.merge(sub, on="dataset_index", how="left")

    coverage = {c: int(df[c].notna().sum()) for c in
                ("median_depth_um", "residual_Sq_um", "ilr_z1", "A2_8_16",
                 "angular_entropy_8_16", "A2_16_32")}
    return df, {"coverage": coverage, "DA_reconciliation": recon}


def target_matrix(df: pd.DataFrame, name: str) -> np.ndarray:
    if name == "Pl":
        return df[["ilr_z1", "ilr_z2", "ilr_z3", "ilr_z4"]].to_numpy(float)
    if name == "Ot_joint":
        return df[JOINT_COLS].to_numpy(float)
    return df[SCALARS[name]].to_numpy(float)


def scorer_for(name: str) -> str:
    if name == "Pl":
        return "aitchison_ilr_q2"
    if name == "Ot_joint":
        return "multi_mse"
    return "mse"


def q2_outer(y_test: np.ndarray, pred: np.ndarray,
             y_train: np.ndarray) -> float:
    """Q^2 with train-mean null (one convention for scalar / multi / ILR)."""
    denom = float(((y_test - y_train.mean(axis=0)) ** 2).sum())
    if denom <= 0:
        return np.nan
    return float(1.0 - ((y_test - pred) ** 2).sum() / denom)


# --------------------------------------------------------------------------- #
# folds artifact (F1: historical double-GroupKFold semantics, role column)
# --------------------------------------------------------------------------- #

def fold_artifact(df: pd.DataFrame, out_dir: Path) -> tuple[dict, str, Path]:
    records, splits = [], {}
    for variant, group_col in (("src_gkf", "shared_height_source_id"),
                               ("proc_gkf", "cv_process_group")):
        groups = df[group_col].to_numpy()
        folds = gkf_splits(groups, 5)
        check_gkf_contract(groups, folds)
        splits[variant] = folds
        for fi, (tr, te) in enumerate(folds):
            for idx in tr:
                records.append({"variant": variant, "fold": fi,
                                "dataset_index": int(idx), "role": "train"})
            for idx in te:
                records.append({"variant": variant, "fold": fi,
                                "dataset_index": int(idx), "role": "test"})
    art = pd.DataFrame(records).sort_values(
        ["variant", "fold", "role", "dataset_index"]).reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fold_assignments.csv"
    art.to_csv(path, index=False)
    return splits, sha256_of(path), path


# --------------------------------------------------------------------------- #
# CV engine
# --------------------------------------------------------------------------- #

def run_cv(df: pd.DataFrame, splits: dict, models: dict, targets: list,
           *, tag: str = "") -> pd.DataFrame:
    inner_groups = df["cv_process_group"].to_numpy()  # frozen Task 22 semantics
    rows = []
    for name in targets:
        y_all = target_matrix(df, name)
        scorer = scorer_for(name)
        for variant, folds in splits.items():
            for model_name, cols in list(models.items()) + [("dummy", None)]:
                X = np.zeros((len(df), 0)) if model_name == "dummy" \
                    else df[cols].to_numpy(float)
                for fi, (tr, te) in enumerate(folds):
                    if model_name == "dummy":
                        mu = y_all[tr].mean(axis=0)
                        pred = np.broadcast_to(mu, y_all[te].shape)
                        skill = q2_outer(y_all[te], pred, y_all[tr])
                        alpha = np.nan
                    else:
                        model, pred, skill, alpha = _fit_eval(
                            X, y_all, scorer, tr, te, inner_groups)
                    rows.append({
                        "target": name, "variant": f"{variant}{tag}",
                        "model": model_name, "fold": fi,
                        "skill_q2": skill, "alpha": alpha,
                        "n_train": int(len(tr)), "n_test": int(len(te)),
                    })
    return pd.DataFrame(rows)


def _fit_eval(X, y_all, scorer, tr, te, inner_groups):
    """Fold-internal preprocessing + target-native alpha + outer Q^2."""
    if scorer == "multi_mse":
        mu = y_all[tr].mean(axis=0)
        sd = y_all[tr].std(axis=0, ddof=0)
        sd = np.where(sd > 0, sd, 1.0)
        y_fit = (y_all - mu) / sd          # fold-internal standardization
        alpha = select_alpha_inner(X[tr], y_fit[tr], inner_groups[tr],
                                   scorer=scorer)
        model = make_ridge(alpha).fit(X[tr], y_fit[tr])
        pred = model.predict(X[te])
        skill = q2_outer(y_fit[te], pred, y_fit[tr])
        return model, pred, skill, alpha
    alpha = select_alpha_inner(X[tr], y_all[tr], inner_groups[tr],
                               scorer=scorer)
    model = make_ridge(alpha).fit(X[tr], y_all[tr])
    pred = model.predict(X[te])
    skill = q2_outer(y_all[te], pred, y_all[tr])
    return model, pred, skill, alpha


def verify_frozen_equals_raw(cfg: dict) -> dict:
    """Rectangle raw/repaired N/A verification: the frozen height field must
    equal the registered raw height for ALL 200 ROIs up to storage-level
    float noise (threshold 1e-4 um = 0.1 nm; cone-repair differences would
    be at um scale)."""
    raw_dir = REPO / cfg["paths"]["raw_height_dir"]
    frozen = np.load(REPO / cfg["paths"]["dataset_npz"])
    H = frozen["height_raw"]
    V = frozen["valid_mask"].astype(bool)
    sid = frozen["session_id"].astype(str)
    smp = frozen["sample_id"].astype(np.int64)
    max_diff = 0.0
    for i in range(H.shape[0]):
        f = raw_dir / f"{sid[i]}__sample_{int(smp[i]):03d}.npz"
        prov.require(f.exists(), f"missing registered raw height: {f.name}")
        data = np.load(f)
        m = data["valid_mask"].astype(bool) & V[i]
        max_diff = max(max_diff, float(np.max(np.abs(
            data["height"][m] - H[i][m]))))
    prov.require(max_diff <= 1e-4,
                 f"frozen height diverges from registered raw beyond "
                 f"storage noise: {max_diff} um")
    return {"status": "N/A_registered",
            "reason": "rectangle dataset has exactly one registered height "
                      "field (cone repair is a single-line observation-"
                      "operator concern; see outputs/cone_repair_impact/); "
                      "raw/repaired target variants do not exist",
            "verification": {"n_checked": int(H.shape[0]),
                             "max_abs_diff_frozen_vs_raw_um": max_diff,
                             "tolerance_um": 1e-4}}


# --------------------------------------------------------------------------- #
# realization diagnostic (phi; NOT in the spectrum, NOT in G28-A)
# --------------------------------------------------------------------------- #

def realization_diagnostic(cfg: dict, frozen: dict, df: pd.DataFrame,
                           out_dir: Path) -> dict:
    t = dict(cfg["task24"]["realization_diagnostic"])
    eps, max_shift, trim = (float(t["phase_only_eps"]),
                            int(t["max_shift_px"]), int(t["trim_px"]))
    R = frozen["R"]
    q = np.stack([sspec.phase_only_field(R[i], eps=eps) for i in range(len(R))])
    sl = slice(trim, R.shape[1] - trim)
    q_trim = q[:, sl, sl]

    ck = (df["pulse_duration_fs"].astype(str) + ":" +
          df["frequency_kHz"].astype(str) + ":" +
          df["pass_count"].astype(str) + ":" +
          df["velocity_mm_s"].astype(str))
    u_all = ck + ":" + df["hatch_spacing_um"].astype(str)

    pairs, seen, seen_u = [], set(), set()
    n = len(df)
    for i in range(n):
        for j in range(i + 1, n):
            if ck.iloc[i] == ck.iloc[j]:
                seen.add((i, j))
                pairs.append((i, j, "same_condition_key"))
            if u_all.iloc[i] == u_all.iloc[j]:
                seen_u.add((i, j))
                pairs.append((i, j, "exact_repeat_u"))
    rng = np.random.default_rng(int(cfg["seeds"]["diagnostic_pairs"]))
    for _ in range(int(t["n_random_pairs"])):
        i, j = rng.integers(0, n, size=2)
        if int(i) != int(j):
            pairs.append((int(i), int(j), "random_ordinary"))

    rows = []
    for i, j, kind in pairs:
        d = sspec.shift_invariant_phase_distance(q_trim[i], q_trim[j],
                                                 max_shift_px=max_shift)
        rows.append({"i": i, "j": j, "pair_type": kind, "d_phi": d})
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "realization_diagnostic.csv", index=False)

    summary = {}
    ord_d = table[table["pair_type"] == "random_ordinary"]["d_phi"].to_numpy()
    for kind in ("same_condition_key", "exact_repeat_u", "random_ordinary"):
        sub = table[table["pair_type"] == kind]["d_phi"]
        summary[kind] = {"n": int(len(sub)), "mean_d_phi": float(sub.mean()),
                         "median_d_phi": float(sub.median())}
    rep = table[table["pair_type"] == "exact_repeat_u"]
    summary["repeat_percentile_vs_ordinary"] = [
        {"pair": [int(r.i), int(r.j)], "d_phi": float(r.d_phi),
         "percentile_among_ordinary": float((ord_d < r.d_phi).mean() * 100.0)}
        for r in rep.itertuples(index=False)]
    summary["config"] = {"eps": eps, "max_shift_px": max_shift,
                         "trim_px": trim, "n_random_pairs": len(ord_d)}
    return summary


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "phase2_8_config.yaml"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if args.quick:
        cfg["task24"]["realization_diagnostic"]["n_random_pairs"] = min(
            int(cfg["task24"]["realization_diagnostic"]["n_random_pairs"]),
            300)
    out = REPO / (args.output_root or (
        cfg["meta"]["quick_output_root"] if args.quick
        else cfg["meta"]["formal_output_root"]))
    out.mkdir(parents=True, exist_ok=True)
    log(f"Task 24 start | quick={args.quick} | root={out}")

    # ---- inputs & targets --------------------------------------------------
    frozen = sdata.load_frozen(cfg)
    df, meta = load_targets(cfg, frozen)
    prov.require(len(df) == 200, f"population {len(df)} != 200")

    n_by_target = {}
    for name in PRIMARY:
        y = target_matrix(df, name)
        y2 = y.reshape(len(y), -1)
        n_by_target[name] = int((~np.isnan(y2).any(axis=1)).sum())
    intersection = min(n_by_target.values())
    prov.require(all(v == intersection for v in n_by_target.values()),
                 f"primary targets lack a common population: {n_by_target}")

    models = cfg["task24"]["models"]
    splits, folds_sha, folds_path = fold_artifact(df, out / "folds")

    # ---- primary spectrum + sensitivities ----------------------------------
    folds_table = run_cv(df, splits, models, PRIMARY)
    band_targets = ["Ot_A2_16_32", "Ot_ent_16_32"]
    if not any(np.isnan(df[SCALARS[t]].to_numpy(float)).any()
               for t in band_targets):
        folds_table = pd.concat(
            [folds_table, run_cv(df, {"src_gkf": splits["src_gkf"]}, models,
                                 band_targets)], ignore_index=True)
    folds_table = pd.concat(
        [folds_table, run_cv(df, {"src_gkf": splits["src_gkf"]}, models,
                             ["Ot_joint"])], ignore_index=True)

    # in-box 101 sensitivity (sub-splits regenerated within, Task 22 semantics)
    with open(REPO / cfg["paths"]["phase2_6_config"], encoding="utf-8") as fh:
        box = yaml.safe_load(fh)["bridge"]["box"]
    sub = df[sgeo.in_box_mask(df, box).to_numpy()].reset_index(drop=True)
    prov.require(len(sub) == 101, f"in-box population {len(sub)} != 101")
    sub_splits = {"src_gkf_inbox": gkf_splits(
        sub["shared_height_source_id"].to_numpy(), 5)}
    check_gkf_contract(sub["shared_height_source_id"].to_numpy(),
                       sub_splits["src_gkf_inbox"])
    folds_table = pd.concat(
        [folds_table, run_cv(sub, sub_splits, models, PRIMARY)],
        ignore_index=True)
    folds_table.to_csv(out / "predictability_spectrum_folds.csv", index=False)

    # ---- raw/repaired sensitivity: verify + register N/A -------------------
    raw_na = verify_frozen_equals_raw(cfg)

    # ---- dummy baseline (G28-A condition 5) --------------------------------
    dummy = folds_table[folds_table["model"] == "dummy"]["skill_q2"].abs().max()
    prov.require(float(dummy) < float(cfg["task24"]["dummy_tolerance"]),
                 f"dummy baseline skill {dummy} not ~0")

    # ---- summary table ------------------------------------------------------
    summary_rows = []
    for (target, variant), grp in folds_table.groupby(["target", "variant"]):
        med = grp.groupby("model")["skill_q2"].median()
        full = grp[grp["model"] == "full"].set_index("fold")["skill_q2"]
        noh = grp[grp["model"] == "minus_h"].set_index("fold")["skill_q2"]
        delta = (full - noh).dropna()
        summary_rows.append({
            "target": target, "variant": variant,
            "skill_full": float(med.get("full", np.nan)),
            "skill_h_only": float(med.get("h_only", np.nan)),
            "skill_minus_h": float(med.get("minus_h", np.nan)),
            "delta_h_median": float(delta.median()) if len(delta) else np.nan,
            "n_folds_positive_delta": int((delta > 0).sum())
            if len(delta) else 0,
        })
    spectrum = pd.DataFrame(summary_rows)
    spectrum.to_csv(out / "predictability_spectrum.csv", index=False)

    # ---- realization diagnostic --------------------------------------------
    diag = realization_diagnostic(cfg, frozen, df, out)

    # ---- G28-A ---------------------------------------------------------------
    three_models_complete = bool(
        folds_table.groupby(["target", "variant"])["model"].nunique().eq(4)
        .all())
    conditions = {
        "1_same_population": {"n_common_intersection": intersection,
                              "n_by_target": n_by_target},
        "2_folds_identical_artifact": {"sha256": folds_sha,
                                       "path": str(folds_path.relative_to(REPO))},
        "3_fold_internal_preprocessing": True,
        "4_target_native_alpha": True,
        "5_dummy_baseline_zero": {"max_abs_skill": float(dummy)},
        "6_three_models_complete": three_models_complete,
        "7_raw_repaired_sensitivity": raw_na,
        "8_coverage": meta["coverage"],
        "9_no_historic_scores": "all numbers recomputed in this run; see "
                                "predictability_spectrum_folds.csv",
    }
    g28a_valid = (intersection == 200 and three_models_complete
                  and float(dummy) < float(cfg["task24"]["dummy_tolerance"])
                  and raw_na["status"] == "N/A_registered")
    gate = {
        "type": "phase_2_8A_predictability_spectrum",
        "protocol": "v2.1 section 2.2 (frozen 2026-09-04)",
        "frozen_inputs": {
            "population": 200,
            "folds": "src_gkf=GroupKFold(shared_height_source_id); "
                     "proc_gkf=GroupKFold(cv_process_group)",
        },
        "G28_A": "VALID" if g28a_valid else "INVALID",
        "G28_A_conditions": conditions,
        "skill_definition": "Q2 = 1 - SSE/SS(train-mean null); "
                            "cross-validated normalized predictive skill, "
                            "NOT information content (section 2.4)",
        "spectrum": spectrum.to_dict(orient="records"),
        "realization_diagnostic": diag,
        "DA_channel_reconciliation": meta["DA_reconciliation"],
    }
    summary_dir = out / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_dir / "gsl28_a_evaluation.json", "w",
              encoding="utf-8") as fh:
        json.dump(gate, fh, indent=1, ensure_ascii=False)

    plot_spectrum(spectrum, out / "predictability_spectrum.png")

    for rel in EXPECTED:
        prov.require((out / rel).exists(), f"missing artifact {rel}")
    log(f"G28-A = {gate['G28_A']} | spectrum rows: {len(spectrum)}")
    log("Task 24 done")
    return 0


def plot_spectrum(spectrum: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prim = spectrum[spectrum["variant"] == "src_gkf"]
    prim = prim[~prim["target"].str.contains("16_32|joint")]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(prim))
    w = 0.27
    for k, (model, label) in enumerate([("full", "M_full"),
                                        ("h_only", "M_h"),
                                        ("minus_h", "M_-h")]):
        ax.bar(x + (k - 1) * w, prim[f"skill_{model}"], width=w, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(prim["target"], rotation=20, ha="right")
    ax.set_ylabel("cross-validated normalized predictive skill (Q$^2$)\n"
                  "NOT information content")
    ax.set_title("Predictability Spectrum (Phase 2.8A, src_gkf primary; "
                 "the axis is a metric, not a hierarchy claim)")
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
