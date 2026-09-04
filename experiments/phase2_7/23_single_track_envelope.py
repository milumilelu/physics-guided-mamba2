#!/usr/bin/env python3
"""Task 23 (G27-3): single-track envelope + finite-array observation model.

2.7r1 re-write: fixes all six implementation deviations identified in the
formal audit (81-line/32-phase/DOE-unit bootstrap/LOHO/direct-guard/
diagnostic output).  All scientific definitions frozen in 任务说明 v2.1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import _lib as p27  # noqa: E402
from src.io_cag import CagHeightReader  # noqa: E402
from src.manual_single_line_annotation import PlaneFit, plane_depth  # noqa: E402

EXPECTED = [
    "outputs/phase2_7/envelope/single_track_envelope.csv",
    "outputs/phase2_7/envelope/envelope_selection_compare.csv",
    "outputs/phase2_7/envelope/forward_model_diagnostic.csv",
    "outputs/phase2_7/envelope/forward_model_simulation.csv",
    "outputs/phase2_7/envelope/bootstrap_delta_tv.csv",
    "outputs/phase2_7/envelope/envelope_selection_compare.csv",
    "outputs/phase2_7/summary/gsl27_3_evaluation.json",
]


def main() -> int:
    cfg, quick = p27.load_config(__doc__)
    g3 = cfg["g27_3"]
    seed = int(cfg["meta"]["random_seed"])
    out = p27.output_dir(cfg, "envelope")
    summary_dir = p27.output_dir(cfg, "summary")

    geometry = pd.read_csv(REPO / cfg["paths"]["single_line_geometry"],
                           encoding="utf-8-sig")
    labels = pd.read_csv(REPO / cfg["paths"]["geometry_qa_labels"],
                         encoding="utf-8-sig")
    line_manifest = pd.read_csv(REPO / cfg["paths"]["single_line_manifest"],
                                encoding="utf-8-sig")
    view = pd.read_csv(REPO / cfg["paths"]["line_view_manifest"],
                       encoding="utf-8-sig").set_index("measurement_id")
    match = pd.read_csv(REPO / cfg["paths"]["direct_bridge"],
                        encoding="utf-8-sig")
    manifest = pd.read_csv(REPO / cfg["paths"]["phase2_manifest"])
    peak_valid_table = pd.read_csv(REPO / cfg["paths"]["lambda_over_hatch"],
                                   encoding="utf-8-sig").set_index("dataset_index")

    # population: estimable ∧ ≠reject (81)
    frame = (geometry.merge(labels[["single_line_id", "qa_label"]],
                            on="single_line_id")
             .merge(line_manifest[["single_line_id", "pulse_duration_fs",
                                   "frequency_kHz", "velocity_mm_s",
                                   "pass_count"]], on="single_line_id"))
    population = frame[(frame["width_identifiability"] == "estimable")
                       & (frame["qa_label"] != "reject_geometry")].copy()
    p27.require(len(population) == 81,
                f"primary population must be 81, got {len(population)}")
    h_levels = sorted(manifest["hatch_spacing_um"].unique().tolist())
    p27.log(f"Task 23 start | quick={quick} | population={len(population)}")

    # ---- (a) extract all 81 single-track profiles -------------------------- #
    profiles = {}
    reader = CagHeightReader(REPO / cfg["paths"]["line_cag"])
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
            t_hat, _ = p27.axis_frame(theta)
            lo, hi = -np.inf, np.inf
            for p_, d_, half in ((anchor[0], t_hat[0], hm.width_um / 2),
                                 (anchor[1], t_hat[1], hm.height_um / 2)):
                s1, s2 = (-half - p_) / d_, (half - p_) / d_
                if s1 > s2:
                    s1, s2 = s2, s1
                lo, hi = max(lo, s1), min(hi, s2)
            s_scan = np.arange(lo + 1, hi, 1.0)
            v_pos = p27.lateral_positions(64, hm.dy_um)
            profs_fine, _ = p27.sample_profiles(
                depth, hm.valid_mask, hm, theta, anchor, s_scan, v_pos)
            online = p27.detect_online_flags(
                profs_fine, float(vr["orientation_threshold_um"]), 8)
            s_start, s_end = p27.line_extent(
                s_scan, online, min_run_um=3.0, merge_gap_um=10.0)
            dp, _aw = p27.scan_plateau_features(
                profs_fine, float(vr["orientation_threshold_um"]), hm.dy_um)
            stable_flags, stable_lo, stable_hi = p27.plateau_stable_run(
                s_scan, online, dp, dp,
                depth_frac=0.5, ref_quantile=0.90, width_band_frac=None,
                gap_merge_um=10.0, min_stable_len_um=60.0, min_stable_frac=0.5)
            sel_s = s_scan[(s_scan >= stable_lo) & (s_scan <= stable_hi)]
            kept, last = [], -np.inf
            for s_val in sel_s:
                if s_val - last >= 2.0 - 1e-9:
                    kept.append(s_val)
                    last = s_val
            s_secs = np.array(kept, dtype=float)
            profs_sec, _ = p27.sample_profiles(
                depth, hm.valid_mask, hm, theta, anchor, s_secs, v_pos)
            mean_profile = np.nanmean(profs_sec, axis=0)
            suitable = p27.profile_suitable(
                mean_profile, edge_frac_max=g3["edge_frac_max"])
            profiles[line_id] = {"profile": mean_profile,
                                 "suitable": suitable}
    finally:
        reader.close()
    n_suitable = sum(1 for v in profiles.values() if v["suitable"])
    p27.log(f"profiles: {len(profiles)} extracted, {n_suitable} suitable")

    # ---- envelope candidates (h_levels × m) -------------------------------- #
    v_pos = p27.lateral_positions(64, 0.278657)
    candidate_rows = []
    for h in h_levels:
        for m in (1, 2, 3):
            lam = m * h
            k = 1.0 / lam
            readings = [p27.hann_projection(
                profiles[lid]["profile"], v_pos, k)
                for lid in sorted(profiles)
                if profiles[lid]["suitable"]]
            candidate_rows.append({
                "h": h, "m": m, "lambda_um": lam, "k_per_um": k,
                "confidence": p27.cycles_level(lam),
                "mean_S_g": float(np.mean(readings)) if readings else np.nan,
                "n_lines_read": len(readings)})
    envelope = pd.DataFrame(candidate_rows)
    envelope.to_csv(out / "single_track_envelope.csv", index=False,
                    encoding="utf-8-sig")

    # ---- 3A: exact-match d_i guard ----------------------------------------- #
    match_cond = match[match["W_source"] == "estimable"].copy()
    _cond = match_cond["condition"].str.split(":", expand=True).astype(int)
    match_cond[["pulse_duration_fs", "frequency_kHz", "pass_count",
                "velocity_mm_s"]] = _cond
    match_cond["dataset_index_list_parsed"] = match_cond[
        "dataset_index_list"].apply(lambda s: [int(x) for x in str(s).split(";")])
    d_rows = []
    for _, group in match_cond.groupby(["pulse_duration_fs", "frequency_kHz",
                                        "pass_count", "velocity_mm_s"]):
        tau, f, n_, v_ = group.iloc[0][
            ["pulse_duration_fs", "frequency_kHz", "pass_count",
             "velocity_mm_s"]].astype(float)
        line_ids = population[
            (population["pulse_duration_fs"] == tau)
            & (population["frequency_kHz"] == f)
            & (population["pass_count"] == n_)
            & (population["velocity_mm_s"] == v_)]["single_line_id"]
        if not len(line_ids):
            continue
        line_id = int(line_ids.iloc[0])
        info = profiles.get(line_id)
        if info is None or not info["suitable"]:
            continue
        prof = info["profile"]
        h_hatch = float(manifest.loc[manifest["dataset_index"].isin(
            group["dataset_index_list_parsed"].iloc[0]),
            "hatch_spacing_um"].iloc[0])
        k_h = 1.0 / h_hatch
        s_h = p27.hann_projection(prof, v_pos, k_h)
        s_2h = (p27.hann_projection(prof, v_pos, 0.5 * k_h)
                if p27.cycles_level(2 * h_hatch) != "UNMEASURABLE" else np.nan)
        for ds in group["dataset_index_list_parsed"].iloc[0]:
            vp = bool(peak_valid_table.loc[ds, "lambda_peak_valid"])
            c_obs = (int(p27.assign_class(
                np.array([peak_valid_table.loc[ds, "lambda_peak_4_32_um"]
                          / h_hatch]),
                np.array([vp]))[0])) if vp else 0
            d_rows.append({"dataset_index": int(ds), "h": h_hatch,
                           "c_obs": c_obs, "S_g_at_h": s_h,
                           "S_g_at_2h": s_2h,
                           "measurable_2h":
                               p27.cycles_level(2 * h_hatch) != "UNMEASURABLE"})
    compare = pd.DataFrame(d_rows)
    compare.to_csv(out / "envelope_selection_compare.csv", index=False,
                   encoding="utf-8-sig")
    p27.log(f"3A: {len(compare)} samples across "
            f"{match_cond.shape[0]} estimable conditions")

    # ---- 3B: simulation with full frozen contract --------------------------- #
    library_profiles = [profiles[lid]["profile"]
                        for lid in sorted(profiles)
                        if profiles[lid]["suitable"]]
    n_lib = len(library_profiles)
    p27.require(n_lib >= 60, f"3B library too small: {n_lib}")
    c_grid = [float(c) for c in g3["c_grid"]]
    n_phases_final = int(g3["phases_final"])
    sim_diag_rows = []

    def simulate_classes(h: float, c: float, n_phases: int) -> np.ndarray:
        phi_grid = np.arange(n_phases, dtype=float) * (
            (h if c == 0.0 else 2 * h) / n_phases)
        classes = np.empty(len(phi_grid) * n_lib, dtype=int)
        idx = 0
        for phi in phi_grid:
            for prof in library_profiles:
                field = p27.synth_field(prof, v_pos, h, phi, c,
                                        pixel_um=g3["pixel_um"],
                                        roi_um=g3["roi_um"])
                cls, lam = p27.field_class(field, h=h)
                classes[idx] = cls
                sim_diag_rows.append(
                    {"h": h, "c": c, "phi": phi,
                     "lambda_peak": lam, "class_code": cls})
                idx += 1
        return classes

    # c-grid scan (LOHO) — use frozen 32 phases for ALL c values (v2.1 收口)
    q_M = {}
    for h in h_levels:
        for c in c_grid:
            cls_arr = simulate_classes(h, c, n_phases_final)
            q_M[(h, c)] = p27.q_distribution(cls_arr)
    sim_diag = pd.DataFrame(sim_diag_rows)
    sim_diag.to_csv(out / "forward_model_diagnostic.csv", index=False,
                    encoding="utf-8-sig")
    p27.log(f"3B simulation: {len(sim_diag)} runs, "
            f"family rate = "
            f"{(sim_diag['class_code'] >= 2).mean():.3f}")

    # ---- five-class q_obs_h (INCLUDING INVALID) ---------------------------- #
    q_obs_h5 = {}
    n_h_all = {}
    for h in h_levels:
        sel = manifest["hatch_spacing_um"] == h
        n_h_all[h] = int(sel.sum())
        classes_h = np.array([
            (int(p27.assign_class(
                np.array([peak_valid_table.loc[ds, "lambda_peak_4_32_um"]
                          / h]),
                np.array([bool(peak_valid_table.loc[ds, "lambda_peak_valid"])]))[0]))
            for ds in manifest.loc[sel, "dataset_index"]])
        q_obs_h5[h] = p27.q_distribution(classes_h)

    # ---- five-class TV + LOHO ---------------------------------------------- #
    def tv_w_for(c: float, q_obs: dict) -> float:
        return sum((n_h_all[h] / 200) * p27.tv(q_obs[h], q_M[(h, c)])
                   for h in h_levels)

    def loho(q_obs: dict) -> tuple[float, float, dict]:
        held_tvs = []
        c_assign = {}
        for held in h_levels:
            train = [h for h in h_levels if h != held]
            c_star = min(c_grid, key=lambda c: sum(
                p27.tv(q_obs[h], q_M[(h, c)]) for h in train))
            held_tvs.append(p27.tv(q_obs[held], q_M[(held, c_star)]))
            c_assign[held] = c_star
        return float(np.mean(held_tvs)), float(np.mean(held_tvs)), c_assign

    tv_const = tv_w_for(0.0, q_obs_h5)
    tv_alt_mean, _, c_assign_full = loho(q_obs_h5)
    tv_alt = tv_alt_mean
    delta_tv = tv_const - tv_alt
    p27.log(f"3B: TV_w(const)={tv_const:.4f} | TV_w(LOHO alt)={tv_alt:.4f} "
            f"| ΔTV={delta_tv:.4f}")

    # ---- h-level consistency (frozen H_eval = {4,6,8,10}) ------------------- #
    h_eval_list = [h for h in g3["h_consistency"]["h_eval"]
                   if n_h_all[h] >= g3["h_consistency"]["min_n_obs"]]
    wins = sum(1 for h in h_eval_list
               if p27.tv(q_obs_h5[h], q_M[(h, 0.0)])
               > p27.tv(q_obs_h5[h], q_M[(h, float(c_assign_full[h]))]))
    n_evaluable = len(h_eval_list)

    # ---- DOE-unit bootstrap with per-replicate LOHO ------------------------- #
    B = int(g3["bootstrap"]["B"])
    boot_seed = seed + int(cfg["seeds"]["bootstrap"])
    ds_by_h = {h: manifest.loc[manifest["hatch_spacing_um"] == h,
                               "dataset_index"].to_numpy()
               for h in h_levels}
    session_of = manifest.set_index("dataset_index")["session_id"]
    delta_boot = np.empty(B)
    rng = np.random.default_rng(boot_seed)
    for b in range(B):
        q_boot = {}
        for h in h_levels:
            ds_h = ds_by_h[h]
            sessions = session_of.loc[ds_h].to_numpy()
            unit_labels = pd.factorize(sessions)[0]
            unique_units = np.unique(unit_labels)
            chosen = rng.choice(unique_units, size=len(unique_units),
                                replace=True)
            take = np.concatenate(
                [np.where(unit_labels == u)[0] for u in chosen])
            ds_take = ds_h[take]
            classes_h = np.array([
                (int(p27.assign_class(
                    np.array([peak_valid_table.loc[ds, "lambda_peak_4_32_um"]
                              / h]),
                    np.array([bool(peak_valid_table.loc[ds,
                                                    "lambda_peak_valid"])]))[0]))
                for ds in ds_take])
            q_boot[h] = p27.q_distribution(classes_h)
        # per-replicate LOHO
        for held in h_levels:
            train = [hh for hh in h_levels if hh != held]
            c_star = min(c_grid, key=lambda c: sum(
                p27.tv(q_boot[hh], q_M[(hh, c)]) for hh in train))
            q_boot[("_pred", held)] = q_M[(held, c_star)]
        tv_const_b = tv_w_for(0.0, q_boot)
        tv_alt_b = sum((n_h_all[h] / 200)
                       * p27.tv(q_boot[h], q_boot[("_pred", h)])
                       for h in h_levels)
        delta_boot[b] = tv_const_b - tv_alt_b
    ci_low = float(np.quantile(delta_boot, 0.025))
    p_boot = float((1 + int((delta_boot <= 0).sum())) / (1 + B))
    pd.DataFrame({"delta_tv": delta_boot}).to_csv(
        out / "bootstrap_delta_tv.csv", index=False)

    # ---- 3A d_i values (frozen formula) ------------------------------------- #
    d_i_values = []
    for cond, group in compare.groupby(["h"]):
        pass
    # aggregate: for each row in compare (one per sample), d_i = q_P2,i(c_i) − q_C,i(c_i)
    for row in compare.itertuples(index=False):
        h = float(row.h)
        c_obs = int(row.c_obs)
        if c_obs == p27.CODE_INVALID:
            continue
        q_C = q_M[(h, 0.0)]
        c_star_fold = c_assign_full.get(h, c_grid[0])
        q_P2 = q_M[(h, float(c_star_fold))]
        d_i = q_P2[c_obs] - q_C[c_obs]
        d_i_values.append(d_i)
    n_eval = len(d_i_values)
    n_contradict = sum(1 for d in d_i_values if d < 0)

    # ---- verdict (frozen order) --------------------------------------------- #
    tv_thresh = g3["tv"]
    inadequate = (tv_const > tv_thresh["inadequate"]
                  and tv_alt > tv_thresh["inadequate"])
    if inadequate:
        verdict = "MODEL_INADEQUATE"
    elif delta_tv <= 0 and (tv_const <= tv_thresh["inadequate"]
                            or tv_alt <= tv_thresh["inadequate"]):
        verdict = "NOT_SUPPORTED"
    else:
        cond = (delta_tv >= tv_thresh["delta_min"]
                and tv_alt <= tv_thresh["period2_max"]
                and ci_low > 0
                and wins >= g3["h_consistency"]["min_wins"]
                and n_evaluable >= g3["h_consistency"]["min_evaluable"])
        verdict = "SUPPORTED" if cond else "PARTIAL"
    if (n_eval >= g3["d_guard"]["n_eval_min"]
            and n_contradict / n_eval > g3["d_guard"]["contradiction_frac"]
            and verdict == "SUPPORTED"):
        verdict = "PARTIAL"

    evaluation = {
        "population": {"3A": "13 exact-match conditions (own envelope)",
                       "3B": f"{n_lib}-line library × "
                             f"{n_phases_final} phases"},
        "tv_w_constant": tv_const,
        "tv_w_period2_loho": tv_alt,
        "delta_tv": delta_tv,
        "bootstrap_ci_low": ci_low,
        "p_boot": p_boot,
        "h_consistency": {"n_evaluable": n_evaluable, "period2_wins": wins,
                          "h_eval_list": h_eval_list},
        "d_guard": {"n_eval": n_eval, "n_contradictions": n_contradict,
                    "frac": (n_contradict / n_eval if n_eval else np.nan)},
        "thresholds": g3["tv"],
        "G_SL27_3": verdict,
        "note": ("linear array model family insufficient; material "
                 "nonlinearity is one candidate, not established"
                 if verdict == "MODEL_INADEQUATE" else ""),
        "language": "two-line / period-doubled spatial organization; "
                    "harmonic forbidden; MODEL_INADEQUATE does not establish "
                    "material nonlinearity",
    }
    (summary_dir / "gsl27_3_evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2),
        encoding="utf-8")
    p27.log(f"G27-3 = {verdict} | TV_w(const)={tv_const:.4f} "
            f"| TV_w(LOHO)={tv_alt:.4f} | ΔTV={delta_tv:.4f} "
            f"(CI low {ci_low:.4f}, p={p_boot:.4f})")
    p27.log("Task 23 done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
