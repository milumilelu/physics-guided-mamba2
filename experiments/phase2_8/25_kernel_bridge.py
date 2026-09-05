#!/usr/bin/env python3
"""Task 25 (Phase 2.8B): measured single-track kernel -> multi-track bridge.

Model family (v2.1 section 3.2, 4 tiers / 5 models):
  L0  kernel-only: the measured g(x) replicated as ONE line -> same-pipeline
      r = lambda_peak / h  (hatch enters only through the observation ratio)
  L1  linear array: z = sum_n g(x - n h - phi), a_n = 1
  L2  pointwise saturation: z = F_beta(s),  F_beta(s) = D_sat (1 - e^{-s/D_sat})
  L3a legacy alternating: a_n = 1 + c (-1)^n   (Phase 2.7 continuity control)
  L3b pairwise interaction: z = sum_n g_n + gamma_per_um * sum_n g_n g_{n+1}

Population (F3/F4): primary = MEASURED g(x) + exact process match
(direct_bridge_exact_match.csv).  n_candidate_exact_match = 19 is the
CANDIDATE count; n_usable_kernel_groups is determined programmatically
(estimable line, QA-pass, suitable profile, valid observation) and reported
-- no pre-written floor.  Holdout unit = kernel_group = (tau, f, N, v,
line identity): every rectangle row sharing one measured kernel is held out
together.  Global parameters (D_sat*, c*, gamma*) are selected by
LOGO_kernel median TV_cond over training kernel groups only; gamma
candidates producing z < -tol removal depth anywhere in the training
simulations are physical-invalid and excluded (F6) -- never clipped.

Metrics (F5): primary = out-of-group TV_cond = mean_i (1 - q_i(y_i));
legacy reference = TV_pooled graded with the Phase 2.7 0.20/0.30 thresholds
(labelled pooled-TV legacy adequacy reference); secondary = 24-bin spectral
TV.  G28-B1: delta TV_cond >= 0.05 and kernel-group paired bootstrap CI
lower > 0 (three comparisons are one model-family exploration; 98.33%
Bonferroni simultaneous CI reported as sensitivity).  G28-B2: pooled-TV
legacy reference only.

Descriptive only (not in the gate): overlap descriptor O(h), spectral
transfer peak r_pred, Spearman associations, L1->L2->L3 monotonicity.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src import geometry as sgeo  # noqa: E402
from src import provenance as prov  # noqa: E402
from src.forward_models import (array_transfer, field_class,  # noqa: E402
                                overlap_descriptor,
                                pairwise_interaction_field, saturate,
                                synth_field)
from src.io_cag import CagHeightReader  # noqa: E402
from src.manual_single_line_annotation import PlaneFit, plane_depth  # noqa: E402

EXPECTED = [
    "kernel_bridge_levels.csv",
    "kernel_bridge_groups.csv",
    "summary/gsl28_b_evaluation.json",
]
CLASS_NAMES = sgeo.CLASS_NAMES


def log(msg: str = "") -> None:
    prov.log(msg)


# --------------------------------------------------------------------------- #
# 81-line kernel library (Task 23 extraction path, via src.geometry)
# --------------------------------------------------------------------------- #

def build_kernel_library(cfg: dict) -> dict[int, dict]:
    t = cfg["task25"]
    g = t["profile"]
    geometry = pd.read_csv(REPO / t["paths"]["single_line_geometry"],
                           encoding="utf-8-sig")
    labels = pd.read_csv(REPO / t["paths"]["geometry_qa_labels"],
                         encoding="utf-8-sig")
    line_manifest = pd.read_csv(REPO / t["paths"]["single_line_manifest"],
                                encoding="utf-8-sig")
    view = pd.read_csv(REPO / t["paths"]["line_view_manifest"],
                       encoding="utf-8-sig").set_index("measurement_id")
    frame = (geometry.merge(labels[["single_line_id", "qa_label"]],
                            on="single_line_id")
             .merge(line_manifest[["single_line_id", "pulse_duration_fs",
                                   "frequency_kHz", "velocity_mm_s",
                                   "pass_count"]], on="single_line_id"))
    population = frame[(frame["width_identifiability"] == "estimable")
                       & (frame["qa_label"] != "reject_geometry")].copy()
    prov.require(len(population) == 81,
                 f"kernel population must be 81, got {len(population)}")

    reader = CagHeightReader(REPO / t["paths"]["line_cag"])
    profiles: dict[int, dict] = {}
    try:
        for row in population.itertuples(index=False):
            line_id = int(row.single_line_id)
            hm = reader.read_height_map(line_id)
            vr = view.loc[line_id]
            fit = PlaneFit(float(vr["plane_a"]), float(vr["plane_b"]),
                           float(vr["plane_c"]), float(vr["plane_rmse_um"]),
                           float(vr["sigma_ref_um"]), -1)
            depth = plane_depth(hm.z, hm.valid_mask, hm.dx_um, hm.dy_um, fit)
            theta = float(vr["theta_line_deg"])
            anchor = (float(vr["orientation_center_x_um"]),
                      float(vr["orientation_center_y_um"]))
            t_hat, _ = sgeo.axis_frame(theta)
            lo, hi = -np.inf, np.inf
            for p_, d_, half in ((anchor[0], t_hat[0], hm.width_um / 2),
                                 (anchor[1], t_hat[1], hm.height_um / 2)):
                s1, s2 = (-half - p_) / d_, (half - p_) / d_
                if s1 > s2:
                    s1, s2 = s2, s1
                lo, hi = max(lo, s1), min(hi, s2)
            s_scan = np.arange(lo + 1, hi, 1.0)
            v_pos = sgeo.lateral_positions(int(g["lateral_samples"]),
                                           float(g["dy_um"]))
            profs_fine, _ = sgeo.sample_profiles(
                depth, hm.valid_mask, hm, theta, anchor, s_scan, v_pos)
            online = sgeo.detect_online_flags(
                profs_fine, float(vr["orientation_threshold_um"]), 8)
            s_start, s_end = sgeo.line_extent(
                s_scan, online, min_run_um=3.0, merge_gap_um=10.0)
            dp, _aw = sgeo.scan_plateau_features(
                profs_fine, float(vr["orientation_threshold_um"]), hm.dy_um)
            stable_flags, stable_lo, stable_hi = sgeo.plateau_stable_run(
                s_scan, online, dp, dp,
                depth_frac=0.5, ref_quantile=0.90, width_band_frac=None,
                gap_merge_um=10.0, min_stable_len_um=60.0, min_stable_frac=0.5)
            sel_s = s_scan[(s_scan >= stable_lo) & (s_scan <= stable_hi)]
            kept, last = [], -np.inf
            for s_val in sel_s:
                if s_val - last >= float(g["section_step_um"]) - 1e-9:
                    kept.append(s_val)
                    last = s_val
            profs_sec, _ = sgeo.sample_profiles(
                depth, hm.valid_mask, hm, theta, anchor,
                np.array(kept, dtype=float), v_pos)
            mean_profile = np.nanmean(profs_sec, axis=0)
            profiles[line_id] = {
                "profile": mean_profile,
                "x": sgeo.lateral_positions(len(mean_profile),
                                            float(g["dy_um"])),
                "suitable": sgeo.profile_suitable(
                    mean_profile, edge_frac_max=float(g["edge_frac_max"])),
                "tau": float(row.pulse_duration_fs),
                "f": float(row.frequency_kHz),
                "v": float(row.velocity_mm_s),
                "N": float(row.pass_count),
            }
    finally:
        reader.close()
    return profiles


# --------------------------------------------------------------------------- #
# exact-match primary population (F3: candidate != usable)
# --------------------------------------------------------------------------- #

def build_primary(cfg: dict, library: dict,
                  lam: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    t = cfg["task25"]
    bridge = pd.read_csv(REPO / t["paths"]["direct_bridge"],
                         encoding="utf-8-sig")
    lam = lam.set_index("dataset_index")
    rows = []
    n_candidate = len(bridge)
    for rec in bridge.itertuples(index=False):
        tau_s, f_s, n_s, v_s = str(rec.condition).split(":")
        tau, f_k, n_p, v_mm = float(tau_s), float(f_s), float(n_s), float(v_s)
        matches = [lid for lid, kw in library.items()
                   if (kw["tau"], kw["f"], kw["N"], kw["v"])
                   == (tau, f_k, n_p, v_mm)]
        idx_list = [int(x) for x in
                    str(rec.dataset_index_list).replace(" ", "").split(";")
                    if x != ""]
        hatches = [float(x) for x in
                   str(rec.hatch_values).replace(" ", "").split(";")
                   if x != ""]
        for pos, di in enumerate(idx_list):
            h = (hatches[pos] if pos < len(hatches)
                 else float(lam.loc[di, "hatch_spacing_um"]))
            observed = np.nan
            if bool(lam.loc[di, "lambda_peak_valid"]):
                r_peak = float(lam.loc[di, "r_h_peak"])
                observed = int(sgeo.assign_class(np.array([r_peak]),
                                                 np.array([True]))[0])
            for lid in matches:
                kw = library[lid]
                peak_valid = bool(lam.loc[di, "lambda_peak_valid"])
                r_obs = (float(lam.loc[di, "r_h_peak"])
                         if peak_valid else np.nan)
                ok_obs = peak_valid and observed == observed \
                    and observed != sgeo.CODE_INVALID
                rows.append({
                    "dataset_index": int(di), "condition": rec.condition,
                    "line_id": lid, "kernel_group": f"{rec.condition}#L{lid}",
                    "h_um": h, "observed": observed, "r_observed": r_obs,
                    "usable": True,
                    "tau": tau, "f": f_k, "N": n_p, "v": v_mm,
                    "reason": "" if (kw["suitable"] and ok_obs)
                    else ("profile_unsuitable" if not kw["suitable"]
                          else "observed_class_invalid"),
                })
    table = pd.DataFrame(rows)
    if len(table):
        table.loc[table["reason"] != "", "usable"] = False
    usable = table[table["usable"]].reset_index(drop=True)
    stats = {
        "n_candidate_exact_match_conditions": int(n_candidate),
        "n_candidate_rows": int(len(table)),
        "n_usable_rows": int(len(usable)),
        "n_usable_kernel_groups": int(usable["kernel_group"].nunique())
        if len(usable) else 0,
        "excluded": table[table["usable"] == False][  # noqa: E712
            ["dataset_index", "reason"]].to_dict(orient="records"),
    }
    return usable, stats


# --------------------------------------------------------------------------- #
# simulation helpers
# --------------------------------------------------------------------------- #

def q_from_codes(codes: list[int]) -> np.ndarray:
    q = np.zeros(5)
    for c in codes:
        q[int(c)] += 1.0
    if len(codes):
        q /= len(codes)
    return q


def simulate_levels(kw: dict, h: float, phase: float, params: dict
                    ) -> dict[str, np.ndarray]:
    """L1 base field once; L2/L3a/L3b derived per candidate parameters."""
    g, x = kw["profile"], kw["x"]
    out = {"L1": synth_field(g, x, h, phase, 0.0)}
    if params.get("D_sat") is not None:
        out["L2"] = saturate(out["L1"], params["D_sat"])
    if params.get("c") is not None:
        out["L3a"] = synth_field(g, x, h, phase, params["c"])
    if params.get("gamma_per_um") is not None:
        out["L3b"] = pairwise_interaction_field(g, x, h, phase,
                                                params["gamma_per_um"])
    return out


def kernel_only_class(kw: dict, h: float, pixel_um: float) -> int:
    one_line = synth_field(kw["profile"], kw["x"], 400.0, 0.0, 0.0)
    return field_class(one_line, h=h, pixel_um=pixel_um)[0]


# --------------------------------------------------------------------------- #
# LOGO_kernel selection + gate evaluation
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "phase2_8_config.yaml"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    t25 = cfg["task25"]
    n_phases = 8 if args.quick else int(t25["phase_grid"])
    b_boot = (int(t25["g28b"]["b1"]["bootstrap"]["quick_B"]) if args.quick
              else int(t25["g28b"]["b1"]["bootstrap"]["B"]))
    root = REPO / (args.output_root or (
        cfg["meta"]["quick_output_root"] if args.quick
        else cfg["meta"]["formal_output_root"]))
    (root / "summary").mkdir(parents=True, exist_ok=True)
    seed = int(cfg["meta"]["random_seed"])
    tol = float(t25["physical_guard"]["tol_um"])
    pixel_um = float(t25["field_pipeline"]["pixel_um"])
    roi_um = float(t25["field_pipeline"]["roi_um"])
    log(f"Task 25 start | quick={args.quick} | phases={n_phases} "
        f"| B={b_boot}")

    # ---- population ---------------------------------------------------------
    library = build_kernel_library(cfg)
    lam = pd.read_csv(REPO / t25["paths"]["lambda_over_hatch"],
                      encoding="utf-8-sig")
    rows, pop_stats = build_primary(cfg, library, lam)
    log(f"exact-match: {pop_stats['n_candidate_exact_match_conditions']} "
        f"candidate conditions -> {pop_stats['n_usable_kernel_groups']} "
        f"usable kernel groups / {pop_stats['n_usable_rows']} rows")
    # minimal feasibility floor only (F3 bans pre-written expectations):
    # the paired kernel-group bootstrap needs >= 3 groups
    prov.require(len(rows) >= 5 and rows["kernel_group"].nunique() >= 3,
                 f"exact-match bridge below feasibility floor: "
                 f"{len(rows)} rows / "
                 f"{rows['kernel_group'].nunique()} groups")
    groups = sorted(rows["kernel_group"].unique())

    grids = {
        "L2": [float(x) for x in np.geomspace(
            float(t25["levels"]["L2"]["grid_um"]["lo"]),
            float(t25["levels"]["L2"]["grid_um"]["hi"]),
            int(t25["levels"]["L2"]["grid_um"]["n"]))],
        "L3a": [float(c) for c in t25["levels"]["L3a"]["c_grid"]],
        "L3b": [float(x) for x in np.linspace(
            float(t25["levels"]["L3b"]["gamma_per_um"]["lo"]),
            float(t25["levels"]["L3b"]["gamma_per_um"]["hi"]),
            int(t25["levels"]["L3b"]["gamma_per_um"]["n"]))],
    }

    def row_phases(h: float) -> list[float]:
        return [j * h / n_phases for j in range(n_phases)]

    # ---- per-row phase-marginal class distributions per level ---------------
    # parameters are NOT fixed yet for L2/L3a/L3b -> compute per candidate
    # inside LOGO selection; L0/L1 are parameter-free.
    def level_q(row, level: str, param) -> np.ndarray:
        kw = library[int(row.line_id)]
        h = float(row.h_um)
        codes = []
        if level == "L0":
            codes.append(kernel_only_class(kw, h, pixel_um))
            return q_from_codes(codes)
        for phi in row_phases(h):
            if level == "L1":
                z = synth_field(kw["profile"], kw["x"], h, phi, 0.0)
            elif level == "L2":
                z = saturate(synth_field(kw["profile"], kw["x"], h, phi, 0.0),
                             param)
            elif level == "L3a":
                z = synth_field(kw["profile"], kw["x"], h, phi, param)
            else:  # L3b
                z = pairwise_interaction_field(kw["profile"], kw["x"], h,
                                               phi, param)
            codes.append(field_class(z, h=h, pixel_um=pixel_um)[0])
        return q_from_codes(codes)

    y = rows["observed"].to_numpy(int)

    def tv_cond(indices, level, param) -> float:
        """mean over rows of TV(q, one-hot) = 1 - q_i(y_i)."""
        vals = [1.0 - float(level_q(rows.iloc[i], level, param)[y[i]])
                for i in indices]
        return float(np.mean(vals)) if vals else np.nan

    # ---- LOGO_kernel parameter selection ------------------------------------
    # Single pass per (group, candidate): accumulate the training-TV mean AND
    # the physical-guard minimum in the same sweep (gamma candidates whose
    # training simulations dip below -tol are excluded, never clipped).
    chosen: dict[str, dict[int, float]] = {"L2": {}, "L3a": {}, "L3b": {}}
    guard_log: dict[str, list] = {"L2": [], "L3a": [], "L3b": []}
    for g_name in groups:
        train_rows = rows[rows["kernel_group"] != g_name].reset_index(drop=True)
        for level, grid in grids.items():
            scores, excluded = [], []
            for cand in grid:
                tv_vals, min_z = [], np.inf
                for _, row in train_rows.iterrows():
                    kw = library[int(row.line_id)]
                    h = float(row.h_um)
                    codes = []
                    for phi in row_phases(h):
                        if level == "L2":
                            base = synth_field(kw["profile"], kw["x"], h,
                                               phi, 0.0)
                            z = saturate(base, cand)
                        elif level == "L3a":
                            z = synth_field(kw["profile"], kw["x"], h,
                                            phi, cand)
                        else:
                            z = pairwise_interaction_field(
                                kw["profile"], kw["x"], h, phi, cand)
                        min_z = min(min_z, float(z.min()))
                        codes.append(field_class(z, h=h, pixel_um=pixel_um)[0])
                    q = q_from_codes(codes)
                    tv_vals.append(1.0 - float(q[int(row.observed)]))
                if min_z < -tol:
                    excluded.append(cand)
                    continue
                scores.append((float(np.mean(tv_vals)), cand))
            if excluded:
                guard_log[level].append({"kernel_group": g_name,
                                         "excluded_candidates": excluded})
            prov.require(scores, f"all candidates physical-invalid for {level}")
            best = min(scores, key=lambda sc: (sc[0], sc[1]))
            chosen[level][g_name] = best[1]

    # ---- final out-of-group predictions -------------------------------------
    levels = ["L0", "L1", "L2", "L3a", "L3b"]
    row_records = []
    q_by_level = {lv: {} for lv in levels}
    for i, row in rows.iterrows():
        g_name = row.kernel_group
        kw = library[int(row.line_id)]
        for level in levels:
            param = chosen.get(level, {}).get(g_name)
            q = level_q(row, level, param)
            q_by_level[level][i] = q
            row_records.append({
                "dataset_index": int(row.dataset_index),
                "kernel_group": g_name, "h_um": row.h_um,
                "observed": int(row.observed), "level": level,
                "param": param,
                "q_pred": "/".join(f"{x:.4f}" for x in q),
                "tv_cond_i": 1.0 - float(q[int(row.observed)]),
            })
    table = pd.DataFrame(row_records)
    table.to_csv(root / "kernel_bridge_levels.csv", index=False)

    # ---- gate metrics --------------------------------------------------------
    group_ids = rows["kernel_group"].to_numpy()

    def per_group_tv(level: str) -> dict[str, float]:
        out = {}
        for g_name in groups:
            idx = [i for i in range(len(rows)) if group_ids[i] == g_name]
            vals = [1.0 - float(q_by_level[level][i][y[i]]) for i in idx]
            out[g_name] = float(np.mean(vals))
        return out

    tv_groups = {lv: per_group_tv(lv) for lv in levels}
    tv_cond_all = {lv: float(np.mean(list(tv_groups[lv].values())))
                   for lv in levels}

    q_obs = q_from_codes(y.tolist())
    tv_pooled = {}
    for lv in levels:
        q_mean = np.mean([q_by_level[lv][i] for i in range(len(rows))],
                         axis=0)
        tv_pooled[lv] = float(0.5 * np.abs(q_mean - q_obs).sum())

    # ---- B1: paired kernel-group bootstrap ----------------------------------
    rng = np.random.default_rng(seed + 900)
    deltas = {}
    for level in ("L2", "L3a", "L3b"):
        deltas[level] = np.array([tv_groups["L1"][g] - tv_groups[level][g]
                                  for g in groups])
    boot = {}
    for level, d in deltas.items():
        draws = np.array([np.mean(rng.choice(d, size=len(d), replace=True))
                          for _ in range(b_boot)])
        ci95 = (float(np.percentile(draws, 2.5)),
                float(np.percentile(draws, 97.5)))
        ci_bon = (float(np.percentile(draws, 0.835)),
                  float(np.percentile(draws, 99.165)))
        boot[level] = {"delta_tv_cond": float(np.mean(d)),
                       "ci95_lower": ci95[0], "ci95_upper": ci95[1],
                       "ci9833_bonferroni_lower": ci_bon[0],
                       "ci9833_bonferroni_upper": ci_bon[1],
                       "n_groups": len(groups)}
    b1_delta_min = float(t25["g28b"]["b1"]["delta_min"])
    b1 = {}
    for level in ("L2", "L3a", "L3b"):
        b = boot[level]
        achieved = bool(b["delta_tv_cond"] >= b1_delta_min
                        and b["ci95_lower"] > 0)
        b1[level] = {**b,
                     "MODEL_CLASS_IMPROVEMENT":
                         "achieved" if achieved else "not_achieved"}
    # ---- B2: pooled-TV legacy reference --------------------------------------
    strong = float(t25["g28b"]["b2"]["strong"])
    partial = float(t25["g28b"]["b2"]["partial"])

    def grade(tv: float) -> str:
        if tv <= strong:
            return "strong_reproduction_legacy_reference"
        if tv <= partial:
            return "partial_legacy_reference"
        return "MODEL_INADEQUATE_legacy_reference"

    b2 = {lv: {"tv_pooled": tv_pooled[lv], "grade": grade(tv_pooled[lv])}
          for lv in levels}

    # ---- descriptive: monotonicity + descriptors ----------------------------
    monotone = {lv: tv_cond_all[lv] for lv in ("L1", "L2", "L3a", "L3b")}
    desc_rows = []
    from src.forward_models import array_transfer, overlap_descriptor
    for i, row in rows.iterrows():
        kw = library[int(row.line_id)]
        g_prof, x_prof = kw["profile"], kw["x"]
        dx = float(np.mean(np.diff(x_prof)))
        try:
            ov = overlap_descriptor(g_prof, dx, float(row.h_um))
            freqs = np.fft.rfftfreq(len(g_prof), d=dx)
            power = np.abs(np.fft.rfft(g_prof - g_prof.mean())) ** 2
            k_lo, k_hi = 1.0 / 32.0, 1.0 / 4.0
            m = (freqs >= k_lo) & (freqs <= k_hi)
            transfer = array_transfer(
                freqs[m], float(row.h_um),
                int(t25["descriptors"]["transfer_n_lines"]))
            k_star = freqs[m][int(np.argmax(power[m] * transfer))]
            r_pred = (1.0 / k_star) / float(row.h_um)
        except Exception:
            ov, r_pred = np.nan, np.nan
        desc_rows.append({"dataset_index": int(row.dataset_index),
                          "kernel_group": row.kernel_group,
                          "h_um": row.h_um, "O_h": ov,
                          "r_pred_transfer": r_pred,
                          "r_observed": float(row.r_observed)
                          if np.isfinite(row.r_observed) else np.nan,
                          "observed": int(row.observed)})
    desc = pd.DataFrame(desc_rows)
    from scipy.stats import spearmanr
    desc_summary = {}
    ok = desc.dropna()
    if len(ok) >= 5:
        desc_summary = {
            "n": int(len(ok)),
            "spearman_O_h_vs_observed_class":
                float(spearmanr(ok["O_h"], ok["observed"]).statistic),
            "spearman_r_pred_vs_r_observed":
                float(spearmanr(ok["r_pred_transfer"],
                                ok["r_observed"]).statistic),
            "note": "descriptive associations on the exact-match subset "
                    "only; no CV claims (v2.1 section 2.5)",
        }
    desc.to_csv(root / "kernel_bridge_descriptors.csv", index=False)

    evaluation = {
        "type": "phase_2_8B_kernel_bridge",
        "protocol": "v2.1 section 3 (frozen 2026-09-04)",
        "population": {
            **pop_stats,
            "holdout_unit": "kernel_group=(tau,f,N,v,line identity)",
            "selection_protocol": "LOGO_kernel (global params; training "
                                  "kernel groups only)",
            "physical_guard": {"tol_um": tol, "rule": "exclude_candidate_no_clip",
                               "log": guard_log},
        },
        "tv_cond_out_of_group": tv_cond_all,
        "per_group_tv_cond": tv_groups,
        "G28_B1": {"delta_min": b1_delta_min, "levels": b1,
                   "multiplicity": t25["g28b"]["b1"]["multiplicity"],
                   "note": "three comparisons are one model-family "
                           "exploration; CI not independent confirmatory "
                           "coverage; Bonferroni 98.33% reported"},
        "G28_B2": {"label": t25["g28b"]["b2"]["label"], "levels": b2},
        "monotonicity_tv_cond": monotone,
        "descriptive": desc_summary,
        "parameters_chosen": {lv: {k: float(v) for k, v in d.items()}
                              for lv, d in chosen.items()},
    }
    with open(root / "summary" / "gsl28_b_evaluation.json", "w",
              encoding="utf-8") as fh:
        json.dump(evaluation, fh, indent=1, ensure_ascii=False)

    groups_table = pd.DataFrame(
        [{"kernel_group": g, **{f"tv_cond_{lv}": tv_groups[lv][g]
                                for lv in levels}} for g in groups])
    groups_table.to_csv(root / "kernel_bridge_groups.csv", index=False)

    for rel in EXPECTED:
        prov.require((root / rel).exists(), f"missing artifact {rel}")
    log(f"G28-B1: " + ", ".join(
        f"{lv}={b1[lv]['MODEL_CLASS_IMPROVEMENT']} "
        f"(d={b1[lv]['delta_tv_cond']:.4f})" for lv in ("L2", "L3a", "L3b")))
    log("G28-B2: " + ", ".join(f"{lv}={b2[lv]['grade']}" for lv in levels))
    log("Task 25 done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
