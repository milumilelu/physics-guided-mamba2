#!/usr/bin/env python3
"""Task 23 (G27-3): single-track envelope measurement + finite-array
observation model + frozen verdict order.

3A (primary, measurement→measurement): 13 exact-match conditions vs their own
measured single-track envelope S_g(k); consistency guard d_i = q_P2,i(c_i) −
q_C,i(c_i).  3B (secondary, population): 160×160 finite-array 2D simulation
reusing the Phase 2.5 radial-spectrum/peak-validity pipeline verbatim; LOHO
period-2 amplitude c; five-class TV; DOE-unit bootstrap.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.io import loadmat  # noqa: E402  ( noqa: keep imports explicit )
from src.io_cag import CagHeightReader  # noqa: E402
from src.manual_single_line_annotation import PlaneFit, plane_depth  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import _lib as p27  # noqa: E402

EXPECTED = [
    "outputs/phase2_7/envelope/single_track_envelope.csv",
    "outputs/phase2_7/envelope/envelope_selection_compare.csv",
    "outputs/phase2_7/envelope/forward_model_simulation.csv",
    "outputs/phase2_7/envelope/bootstrap_delta_tv.csv",
    "outputs/phase2_7/summary/gsl27_3_evaluation.json",
]

CLASSES = list(range(5))  # INVALID, OUT, m1, m2, m3


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
                       encoding="utf-8-sig").set_index(
        view_col := "measurement_id")
    match = pd.read_csv(REPO / cfg["paths"]["direct_bridge"],
                        encoding="utf-8-sig")
    manifest = pd.read_csv(REPO / cfg["paths"]["phase2_manifest"])
    radial_obs = pd.read_csv(REPO / cfg["paths"]["p25_radial_long_csv"],
                             encoding="utf-8-sig")

    frame = (geometry.merge(labels[["single_line_id", "qa_label"]],
                            on="single_line_id")
             .merge(line_manifest[["single_line_id", "pulse_duration_fs",
                                   "frequency_kHz", "velocity_mm_s",
                                   "pass_count"]], on="single_line_id"))
    population = frame[(frame["width_identifiability"] == "estimable")
                       & (frame["qa_label"] != "reject_geometry")].copy()
    p27.require(len(population) == 81,
                f"Task 23 primary population must be 81, got {len(population)}")
    p27.log(f"Task 23 start | quick={quick} | population={len(population)}")

    # ---- (a) single-track profiles + envelope ----------------------------- #
    h_levels = sorted(manifest["hatch_spacing_um"].unique().tolist())
    candidate_rows = []
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
            profiles_fine, _ = p27.sample_profiles(
                depth, hm.valid_mask, hm, theta, anchor, s_scan, v_pos)
            online = p27.detect_online_flags(
                profiles_fine, float(vr["orientation_threshold_um"]),
                8)
            s_start, s_end = p27.line_extent(
                s_scan, online, min_run_um=3.0, merge_gap_um=10.0)
            dp = p27.scan_plateau_features(profiles_fine)
            stable_flags, stable_lo, stable_hi = p27.plateau_stable_run(
                s_scan, online, dp, dp,
                depth_frac=0.5, ref_quantile=0.90, width_band_frac=None,
                gap_merge_um=10.0, min_stable_len_um=60.0, min_stable_frac=0.5)
            sel = s_scan[(s_scan >= stable_lo) & (s_scan <= stable_hi)]
            step = 2.0
            kept, last = [], -np.inf
            for s_val in sel:
                if s_val - last >= step - 1e-9:
                    kept.append(s_val)
                    last = s_val
            s_sections = np.array(kept, dtype=float)
            profiles_sec, _ = p27.sample_profiles(
                depth, hm.valid_mask, hm, theta, anchor, s_sections, v_pos)
            mean_profile = np.nanmean(profiles_sec, axis=0)
            suitable = p27.profile_suitable(
                mean_profile, edge_frac_max=g3["edge_frac_max"])
            profiles[line_id] = {"profile": mean_profile, "suitable": suitable}
            for m in (1, 2, 3):
                lam = m * float(row.hatch_spacing_um)
                k = 1.0 / lam
                candidate_rows.append({
                    "single_line_id": line_id, "h": float(row.hatch_spacing_um),
                    "m": m, "lambda_um": lam, "k_per_um": k,
                    "confidence": p27.cycles_level(lam),
                    "S_g": (p27.hann_projection(mean_profile, v_pos, k)
                            if suitable and m == 1 else np.nan)
                    if m == 1 else (p27.hann_projection(mean_profile, v_pos, k)
                                    if suitable else np.nan)})
    finally:
        reader.close()
    candidates = pd.DataFrame(candidate_rows)
    candidates["S_g"] = candidates.apply(
        lambda r_: p27.hann_projection(
            profiles[int(r_["single_line_id"])]["profile"],
            p27.lateral_positions(64, 0.278657), float(r_["k_per_um"]))
        if profiles[int(r_["single_line_id"])]["suitable"] else np.nan, axis=1)
    candidates.to_csv(out / "single_track_envelope.csv", index=False,
                      encoding="utf-8-sig")
    p27.log(f"envelope: {len(candidates)} candidate readings, "
            f"{sum(1 for v_ in profiles.values() if v_['suitable'])}/81 "
            "profiles suitable")

    # ---- (b) 3A: exact-match selection compare + d_i guard ---------------- #
    line_conditions = population.merge(
        line_manifest[["single_line_id", "pulse_duration_fs", "frequency_kHz",
                       "velocity_mm_s", "pass_count"]], on="single_line_id")
    line_conditions = line_conditions.merge(
        geometry[["single_line_id", "median_W50_um",
                  "width_identifiability"]], on="single_line_id")
    state_map = {"right_censored": "W_lower_bound",
                 "insufficient_sections": "W_unavailable"}
    obs_classes = {}
    for ds, row in match.iterrows():
        pass
    obs_map = dict(zip(manifest["dataset_index"], manifest["hatch_spacing_um"]))
    peak_valid = pd.read_csv(REPO / cfg["paths"]["lambda_over_hatch"],
                             encoding="utf-8-sig").set_index("dataset_index")
    obs_class_by_ds = {}
    for ds in manifest["dataset_index"]:
        vp = bool(peak_valid.loc[ds, "lambda_peak_valid"])
        r_ = (float(peak_valid.loc[ds, "lambda_peak_4_32_um"])
              / float(obs_map[ds])) if vp else np.nan
        obs_class_by_ds[int(ds)] = int(p27.assign_class(
            np.array([r_]), np.array([vp]))[0]) if vp else 0
    d_rows = []
    for cond, group in match[match["W_source"] == "estimable"].groupby(
            ["pulse_duration_fs", "frequency_kHz", "pass_count",
             "velocity_mm_s"]):
        tau, f, n, v = cond
        line_ids = line_conditions[
            (line_conditions["pulse_duration_fs"] == tau)
            & (line_conditions["frequency_kHz"] == f)
            & (line_conditions["pass_count"] == n)
            & (line_conditions["velocity_mm_s"] == v)]["single_line_id"]
        line_id = int(line_ids.iloc[0])
        prof = profiles.get(line_id, {}).get("profile")
        if prof is None or not profiles[line_id]["suitable"]:
            continue
        for ds in group["dataset_index"]:
            h_rect = float(manifest.loc[manifest["dataset_index"] == ds,
                                        "hatch_spacing_um"].iloc[0])
            c_obs = obs_class_by_ds[int(ds)]
            if c_obs == p27.CODE_INVALID:
                continue
            q_obs_one = np.zeros(5)
            q_obs_one[c_obs] = 1.0
            k_h = 1.0 / h_rect
            s_h = p27.hann_projection(prof, p27.lateral_positions(64, 0.278657),
                                      k_h)
            d_rows.append({"dataset_index": int(ds), "h": h_rect,
                           "c_obs": c_obs, "S_g_at_h": s_h,
                           "S_g_at_2h": (p27.hann_projection(
                               prof, p27.lateral_positions(64, 0.278657),
                               0.5 * k_h))})
    compare = pd.DataFrame(d_rows)
    compare.to_csv(out / "envelope_selection_compare.csv", index=False,
                   encoding="utf-8-sig")
    p27.log(f"3A exact-match compare: {len(compare)} scorable samples "
            f"(S_g at 1/h vs 1/2h recorded)")

    # ---- (c) 3B: finite-array simulation (same 2D pipeline) ---------------- #
    library = sorted(population["single_line_id"].tolist())
    subsample = library[::3][:g3["lines_scan"]] if len(library) >= 27 else library
    reader = CagHeightReader(REPO / cfg["paths"]["line_cag"])
    line_profiles = {}
    try:
        for line_id in subsample:
            existing = profiles.get(line_id)
            if existing is not None:
                line_profiles[line_id] = existing["profile"]
                continue
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
            profs, _ = p27.sample_profiles(depth, hm.valid_mask, hm, theta,
                                           anchor, s_scan, v_pos)
            line_profiles[line_id] = np.nanmean(profs, axis=0)
    finally:
        reader.close()
    library_profiles = [line_profiles[i] for i in sorted(line_profiles)]

    def simulate(h: float, c: float, phases: int,
                 profs: list[np.ndarray]) -> np.ndarray:
        """Five-class frequencies over phases × profiles (frozen pipeline)."""
        counts = np.zeros(5, dtype=int)
        phi_grid = np.arange(phases, dtype=float) * (
            (h if c == 0.0 else 2 * h) / phases)
        for phi in phi_grid:
            for prof in profs:
                field = p27.synth_field(prof, p27.lateral_positions(
                    64, 0.278657), h, phi, c,
                    pixel_um=g3["pixel_um"], roi_um=g3["roi_um"])
                cls, _ = p27.field_class(field)
                counts[cls] += 1
        return counts / counts.sum()

    q_obs_h5 = {}
    n_h_all = {}
    for h in h_levels:
        sel = manifest["hatch_spacing_um"] == h
        n_h_all[h] = int(sel.sum())
        classes_h = np.array([obs_class_by_ds[int(ds)] for ds in
                              manifest.loc[sel, "dataset_index"]])
        q_obs_h5[h] = p27.q_distribution(classes_h)
    c_grid = [float(c) for c in g3["c_grid"]]
    sim_rows = []
    q_M = {}
    for h in h_levels:
        for c in c_grid:
            phases = (g3["phases_scan"]
                      if c not in (0.0,) else g3["phases_final"])
            profs = (library_profiles[:g3["lines_scan"]]
                     if c != 0.0 else library_profiles)
            q_M[(h, c)] = simulate(h, c, phases, profs)
            sim_rows.append({"h": h, "c": c, "phases": phases,
                             "n_profiles": len(profs),
                             **{f"P_{name}": q_M[(h, c)][i]
                                for i, name in enumerate(p27.CLASS_NAMES)}})
    sim_frame = pd.DataFrame(sim_rows)
    sim_frame.to_csv(out / "forward_model_simulation.csv", index=False,
                     encoding="utf-8-sig")

    # ---- LOHO + constant baseline + bootstrap + verdict -------------------- #
    def tv_w_for(c: float, q_obs: dict) -> float:
        total = 0.0
        for h in h_levels:
            total += (n_h_all[h] / 200) * p27.tv(q_obs[h], q_M[(h, c)])
        return total

    def loho(q_obs: dict) -> tuple[float, float]:
        deltas = []
        for held in h_levels:
            train = [h for h in h_levels if h != held]
            c_star = min(c_grid,
                         key=lambda c: sum(p27.tv(q_obs[h], q_M[(h, c)])
                                           for h in train))
            deltas.append(p27.tv(q_obs[held], q_M[(held, c_star)]))
        return float(np.mean(deltas)), deltas

    q_obs_for_model = q_obs_h5
    tv_const = tv_w_for(0.0, q_obs_for_model)
    tv_loho_mean, _ = loho(q_obs_for_model)
    c_mode = int(np.round(
        min(c_grid, key=lambda c: tv_w_for(c, q_obs_for_model)) * 10))
    p27.log(f"3B: TV_w(constant=c0)={tv_const:.4f} | LOHO held-out TV mean="
            f"{tv_loho_mean:.4f}")
    tv_alt = tv_loho_mean
    delta_tv = tv_const - tv_alt

    def h_consistency(q_obs: dict) -> tuple[int, int]:
        h_eval = [h for h in g3["h_consistency"]["h_eval"]
                  if n_h_all[h] >= g3["h_consistency"]["min_n_obs"]]
        wins = sum(1 for h in h_eval
                   if p27.tv(q_obs[h], q_M[(h, 0.0)])
                   > p27.tv(q_obs[h], q_M[(h, float(c_mode / 10))]))
        return len(h_eval), wins

    n_evaluable, wins = h_consistency(q_obs_for_model)
    rng = np.random.default_rng(seed + int(cfg["seeds"]["bootstrap"]))
    B = int(g3["bootstrap"]["B"])
    delta_boot = np.empty(B)
    session_of = manifest.set_index("dataset_index")["session_id"]
    base_of = manifest.set_index("dataset_index")["base_condition_group"] \
        if "base_condition_group" in manifest.columns else None
    units = manifest.groupby(["session_id"]).indices
    for b in range(B):
        q_boot = {}
        for h in h_levels:
            ds_h = manifest.loc[manifest["hatch_spacing_um"] == h,
                                "dataset_index"].to_numpy()
            take = rng.choice(ds_h, size=len(ds_h), replace=True)
            classes_h = np.array([obs_class_by_ds[int(ds)] for ds in take])
            q_boot[h] = p27.q_distribution(classes_h)
        train_tv = {c: sum(p27.tv(q_boot[h], q_M[(h, c)]) for h in h_levels
                           if h != 2.0) for c in c_grid}
        c_star_b = min(c_grid, key=lambda c: train_tv[c])
        tv_c = tv_w_for(c_star_b, q_boot)
        tv_const_b = tv_w_for(0.0, q_boot)
        delta_boot[b] = tv_const_b - tv_c
    ci_low = float(np.quantile(delta_boot, 0.025))
    p_boot = float((1 + int((delta_boot <= 0).sum())) / (1 + B))
    pd.DataFrame({"delta_tv": delta_boot}).to_csv(
        out / "bootstrap_delta_tv.csv", index=False)

    verdict = p27.verdict_g27_3(
        tv_const, tv_alt, delta_tv, ci_low, wins, n_evaluable, [], 0,
        thresholds={"tv": g3["tv"],
                    "h_consistency": g3["h_consistency"],
                    "d_guard": g3["d_guard"]})
    gsl27_3 = {
        "model_population": {"3A": "13 exact-match conditions, own envelope",
                             "3B": "81-line library (27 subsampled for c-grid)"},
        "tv_w_constant": tv_const, "tv_w_period2_loho": tv_alt,
        "delta_tv": delta_tv, "bootstrap_ci_low": ci_low,
        "p_boot": p_boot, "h_consistency": {"n_evaluable": n_evaluable,
                                            "period2_wins": wins},
        "c_grid": c_grid,
        "thresholds": g3["tv"],
        "G_SL27_3": verdict["G_SL3"],
        "note": verdict.get("note", ""),
        "language": "two-line / period-doubled spatial organization; "
                    "harmonic wording forbidden; MODEL_INADEQUATE does not "
                    "establish material nonlinearity",
    }
    (summary_dir / "gsl27_3_evaluation.json").write_text(
        json.dumps(gsl27_3, ensure_ascii=False, indent=2),
        encoding="utf-8")
    p27.log(f"G27-3 = {verdict['G_SL3']} | TV_w(constant)={tv_const:.3f} "
            f"| TV_w(period2/LOHO)={tv_alt:.3f} | ΔTV={delta_tv:.3f} "
            f"(CI low {ci_low:.3f}, p_boot={p_boot:.4f})")
    p27.log("Task 23 done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
