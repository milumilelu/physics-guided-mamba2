"""Canonical frozen-input loading for all phases.

``load_frozen`` rebuilds the Phase 1 residual convention exactly: from the
frozen exploration manifest + dataset NPZ, with R = H - per-sample valid
median computed from ``height_raw``.  Migrated verbatim from the frozen
Phase 1.5 ``_lib.load_frozen`` (WP1 canonical migration; parity-tested in
``tests/test_src_data.py`` against the original).

All hard contract checks are preserved: 200 ROIs, (session_id, sample_id)
uniqueness, 160 unique height sources, NPZ row order == manifest order,
no non-finite valid pixels.

Binding spec: Phase 2.8 v2.1 §4.1 (`src/data.py` row).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.provenance import log, require

__all__ = ["load_frozen", "REPO"]

REPO = Path(__file__).resolve().parents[1]


def load_frozen(cfg: dict) -> dict:
    """Load Phase 1 manifest + NPZ and rebuild residuals exactly as Phase 1."""
    man = pd.read_csv(REPO / cfg["paths"]["exploration_manifest"])
    require(len(man) == 200, f"manifest rows {len(man)} != 200")
    require(not man.duplicated(["session_id", "sample_id"]).any(),
            "(session_id, sample_id) not unique")
    require(man["shared_height_source_id"].nunique() == 160,
            "unique shared sources != 160")
    require({"median_depth_um", "residual_Sq_um", "session_role",
             "design_group"} <= set(man.columns),
            "manifest lacks Phase 1 columns; run Phase 1 first")

    data = np.load(REPO / cfg["paths"]["dataset_npz"])
    H = data["height_raw"].astype(np.float64)
    V = data["valid_mask"].astype(bool)
    require(H.shape == (200, 160, 160), "NPZ shape mismatch")
    require((man["session_id"].to_numpy() == data["session_id"].astype(str)).all()
            and (man["sample_id"].to_numpy(np.int64)
                 == data["sample_id"].astype(np.int64)).all(),
            "NPZ row order != manifest order")
    bad = int(np.count_nonzero(~np.isfinite(H[V])))
    require(bad == 0, f"{bad} non-finite valid pixels")
    Hnan = np.where(V, H, np.nan)
    med = np.nanmedian(Hnan, axis=(1, 2))
    R = Hnan - med[:, None, None]
    log(f"  frozen inputs OK: 200 ROIs, 160 clusters, "
        f"valid_fraction min = {V.reshape(200, -1).mean(1).min():.4f}")
    return {"man": man, "H": H, "V": V, "R": R, "Hnan": Hnan}


# --------------------------------------------------------------------------- #
# single-track profile library (Phase 2.7 Task 23 / Phase 2.8 Task 25)
# --------------------------------------------------------------------------- #

def build_line_profile_library(paths: dict, *, lateral_samples: int,
                               dy_um: float, section_step_um: float,
                               edge_frac_max: float) -> dict:
    """Extract the measured single-track kernel library g(x) per line.

    Orchestration over canonical components only (src.geometry / io_cag /
    manual_single_line_annotation), reproducing the frozen Task 23 extraction
    path.  2.7r2 revisions (registered in phase2_7_gate_eval r2):

    * section selection uses the ``plateau_stable_run`` MEMBERSHIP FLAGS
      directly -- bridged shallow positions inside the merged interval no
      longer re-enter the mean profile;
    * lateral positions that are out-of-FOV in every section (no removal
      measurement) are fixed at 0 depth before synthesis -- the same
      no-material convention as ``synth_field``'s left/right=0 extension
      (NaN would propagate through ``np.interp`` into the field).

    Returns {"profiles": {line_id: {"profile", "x", "suitable", "tau", "f",
    "v", "N"}}, "population": population DataFrame}.
    """
    import numpy as np
    import pandas as pd

    from src.geometry import (axis_frame, detect_online_flags,
                              lateral_positions, line_extent,
                              plateau_stable_run, profile_suitable,
                              sample_profiles, scan_plateau_features)
    from src.io_cag import CagHeightReader
    from src.manual_single_line_annotation import PlaneFit, plane_depth
    from src.provenance import log, require

    geometry = pd.read_csv(REPO / paths["single_line_geometry"],
                           encoding="utf-8-sig")
    labels = pd.read_csv(REPO / paths["geometry_qa_labels"],
                         encoding="utf-8-sig")
    line_manifest = pd.read_csv(REPO / paths["single_line_manifest"],
                                encoding="utf-8-sig")
    view = pd.read_csv(REPO / paths["line_view_manifest"],
                       encoding="utf-8-sig").set_index("measurement_id")
    frame = (geometry.merge(labels[["single_line_id", "qa_label"]],
                            on="single_line_id")
             .merge(line_manifest[["single_line_id", "pulse_duration_fs",
                                   "frequency_kHz", "velocity_mm_s",
                                   "pass_count"]], on="single_line_id"))
    population = frame[(frame["width_identifiability"] == "estimable")
                       & (frame["qa_label"] != "reject_geometry")].copy()
    require(len(population) == 81,
            f"kernel population must be 81, got {len(population)}")

    reader = CagHeightReader(REPO / paths["line_cag"])
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
            t_hat, _ = axis_frame(theta)
            lo, hi = -np.inf, np.inf
            for p_, d_, half in ((anchor[0], t_hat[0], hm.width_um / 2),
                                 (anchor[1], t_hat[1], hm.height_um / 2)):
                s1, s2 = (-half - p_) / d_, (half - p_) / d_
                if s1 > s2:
                    s1, s2 = s2, s1
                lo, hi = max(lo, s1), min(hi, s2)
            s_scan = np.arange(lo + 1, hi, 1.0)
            v_pos = lateral_positions(int(lateral_samples), float(dy_um))
            profs_fine, _ = sample_profiles(
                depth, hm.valid_mask, hm, theta, anchor, s_scan, v_pos)
            online = detect_online_flags(
                profs_fine, float(vr["orientation_threshold_um"]), 8)
            line_extent(s_scan, online, min_run_um=3.0, merge_gap_um=10.0)
            dp, _aw = scan_plateau_features(
                profs_fine, float(vr["orientation_threshold_um"]), hm.dy_um)
            stable_flags, stable_lo, stable_hi = plateau_stable_run(
                s_scan, online, dp, dp,
                depth_frac=0.5, ref_quantile=0.90, width_band_frac=None,
                gap_merge_um=10.0, min_stable_len_um=60.0, min_stable_frac=0.5)
            # 2.7r2: membership FLAGS (not the merged [lo, hi] interval) --
            # bridged shallow dips stay inside the interval but never
            # contribute sections
            sel_s = s_scan[stable_flags]
            kept, last = [], -np.inf
            for s_val in sel_s:
                if s_val - last >= float(section_step_um) - 1e-9:
                    kept.append(s_val)
                    last = s_val
            require(len(kept) >= 1,
                    f"line {line_id}: no section inside the stable run")
            profs_sec, _ = sample_profiles(
                depth, hm.valid_mask, hm, theta, anchor,
                np.array(kept, dtype=float), v_pos)
            mean_profile = np.nanmean(profs_sec, axis=0)
            mean_profile = np.where(np.isfinite(mean_profile),
                                    mean_profile, 0.0)
            profiles[line_id] = {
                "profile": mean_profile,
                "x": lateral_positions(len(mean_profile), float(dy_um)),
                "suitable": profile_suitable(mean_profile,
                                             edge_frac_max=float(edge_frac_max)),
                "tau": float(row.pulse_duration_fs),
                "f": float(row.frequency_kHz),
                "v": float(row.velocity_mm_s),
                "N": float(row.pass_count),
            }
    finally:
        reader.close()
    log(f"profile library: {len(profiles)} lines, "
        f"{sum(1 for v in profiles.values() if v['suitable'])} suitable")
    return {"profiles": profiles, "population": population}
