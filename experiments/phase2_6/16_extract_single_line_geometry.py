#!/usr/bin/env python3
"""Task SL-01b: extract single-line geometry and frozen threshold widths.

For every CAG group the frozen plane (view manifest) gives removal depth
D = plane - z; perpendicular profiles are sampled DIRECTLY along the frozen
line axis in original coordinates (细则 §0.20 -- no whole-map rotation), the
on-line extent is auto-detected with the pilot groove threshold, sections are
placed every 2 um inside the central 70% stable region, and every frozen §4.2
width feature is computed.  A cone-repaired sensitivity arm re-extracts the
same sections from repaired heights (extent/sections frozen from raw).
Pilot stable-region reconciliation runs on the 15 pilot groups (abort < 0.90).

BLIND QA: montages carry geometry only -- no 8/16 um markers, no band shading,
no bridge results (细则 §0.7).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import _lib as p26  # noqa: E402
from src.conical_dropout import ConicalDropoutConfig, repair_compact_dropouts  # noqa: E402
from src.data_contracts import HeightMap  # noqa: E402  (type reference only)
from src.io_cag import CagHeightReader  # noqa: E402
from src.manual_single_line_annotation import PlaneFit, plane_depth  # noqa: E402

EXPECTED = [
    "outputs/phase2_6/single_line/single_line_geometry.csv",
    "outputs/phase2_6/single_line/cross_section_widths.csv",
    "outputs/phase2_6/single_line/geometry_qa_labels.csv",
    "outputs/phase2_6/single_line/stable_region_reconciliation.csv",
    "outputs/phase2_6/single_line/qa_montages/*.png",
]

LINE_KEYS = [
    "n_sections_total", "n_sections_used", "median_W20_um", "iqr_W20_um",
    "p10_W20_um", "p90_W20_um", "censored_frac_W20", "n_uncensored_sections_W20",
    "median_W50_um", "iqr_W50_um", "p10_W50_um", "p90_W50_um", "censored_frac_W50",
    "n_uncensored_sections_W50", "median_W80_um", "iqr_W80_um", "p10_W80_um",
    "p90_W80_um", "censored_frac_W80", "n_uncensored_sections_W80", "CV_W50",
    "median_W_eq_um", "median_max_depth_um", "width_identifiability",
]

PILOT_GROUPS = (13, 19, 33, 34, 43, 44, 48, 51, 60, 68, 94, 95, 101, 104, 116)


def fov_s_interval(anchor: tuple[float, float], t_hat: np.ndarray,
                   half_w: float, half_h: float, margin_um: float
                   ) -> tuple[float, float] | None:
    """s-interval where anchor + s*t_hat stays inside the centered FOV."""
    lo, hi = -np.inf, np.inf
    for p, d, half in ((anchor[0], t_hat[0], half_w),
                       (anchor[1], t_hat[1], half_h)):
        if abs(d) < 1e-12:
            if abs(p) > half:
                return None
            continue
        s1, s2 = (-half - p) / d, (half - p) / d
        if s1 > s2:
            s1, s2 = s2, s1
        lo, hi = max(lo, s1), min(hi, s2)
    if hi - lo <= 2 * margin_um:
        return None
    return lo + margin_um, hi - margin_um


def auto_qc_flags(row: dict, extent: tuple[float, float],
                  s_bounds: tuple[float, float], cfg: dict) -> list[str]:
    """Hardened review flags (pilot qc_rules where computable)."""
    flags = []
    length = extent[1] - extent[0]
    if extent[0] <= s_bounds[0] + 0.5 or extent[1] >= s_bounds[1] - 0.5:
        flags.append("edge_clipped")
    if np.isfinite(row.get("p10_W50_um")) and np.isfinite(row.get("p90_W50_um")) \
            and np.isfinite(row.get("median_W50_um")) and row["median_W50_um"] > 0:
        relative = (row["p90_W50_um"] - row["p10_W50_um"]) / row["median_W50_um"]
        if relative > 0.25:
            flags.append("threshold_width_relative_range_above")
    if row["n_sections_used"] < cfg["single_line"]["min_sections"]:
        flags.append("few_sections")
    if np.isfinite(row.get("censored_frac_W50")) and \
            row["censored_frac_W50"] > cfg["single_line"]["censored_frac_W50_uncertain_above"]:
        flags.append("W50_censored")
    if length < 150.0:
        flags.append("short_visible_length")
    return flags


def render_montage(target: Path, group: int, hm, depth: np.ndarray,
                   meta: dict, sections: pd.DataFrame, cfg: dict) -> None:
    """Six blind geometry panels (细则 §4.3; no 8/16 um information)."""
    finite = np.where(hm.valid_mask, depth, np.nan)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4))
    ax = axes[0, 0]
    image = ax.imshow(np.where(hm.valid_mask, hm.z, np.nan), cmap="viridis",
                      aspect="auto")
    ax.set_title(f"group {group} absolute height z (um)")
    fig.colorbar(image, ax=ax, fraction=0.046)
    ax = axes[0, 1]
    lo, hi = np.nanpercentile(finite, (2, 98))
    image = ax.imshow(finite, cmap="magma", vmin=lo, vmax=hi, aspect="auto")
    ax.set_title(f"removal depth D (um) | theta={meta['theta']:+.3f} deg")
    fig.colorbar(image, ax=ax, fraction=0.046)
    ax = axes[0, 2]
    used = sections[sections["n_above_threshold"] > 0]
    widths = used["W50_um"].dropna()
    if len(widths):
        pick_s = float(used.loc[(widths - widths.median()).abs().idxmin(), "s_um"])
        profile, _ = p26.sample_profiles(
            depth, hm.valid_mask, hm, meta["theta"], meta["anchor"],
            np.array([pick_s]), meta["v_positions"],
            order=cfg["single_line"]["rotate_order"],
            mask_weight_min=cfg["single_line"]["mask_weight_min"])
        line = profile[0]
        ax.plot(meta["v_positions"], line, color="k", lw=1.0)
        row = used.loc[(used["s_um"] - pick_s).abs().idxmin()]
        for key, y in (("W20", 0.88), ("W50", 0.80), ("W80", 0.72)):
            value = row[f"{key}_um"]
            label = f"{key}={value:.1f}um" if np.isfinite(value) else f"{key}=NA"
            ax.annotate(label, xy=(0.03, y), xycoords="axes fraction",
                        color={"W20": "tab:blue", "W50": "tab:red",
                               "W80": "tab:green"}[key])
        ax.set_title(f"cross-section @ s={pick_s:.1f}um (median-W50 section)")
    else:
        ax.text(0.5, 0.5, "no usable section", ha="center", transform=ax.transAxes)
        ax.set_title("cross-section: none")
    ax = axes[1, 0]
    ax.plot(sections["s_um"], sections["D_max_um"], color="k", lw=1.0)
    if np.isfinite(meta["s_start"]):
        for s_val in (meta["s_start"], meta["s_end"]):
            ax.axvline(s_val, color="c", ls="--", lw=0.9)
        ax.axvspan(meta["stable"][0], meta["stable"][1], color="k", alpha=0.12)
    ax.set_title("section D_max along s | dashed=extent, grey=stable region")
    ax = axes[1, 1]
    uncensored = sections[~sections["censored_W50"].astype(bool)]["W50_um"].dropna()
    ax.hist(uncensored, bins=24, color="grey", edgecolor="k")
    ax.set_title(f"W50 section distribution (n={len(uncensored)} uncensored)")
    ax = axes[1, 2]
    ax.imshow(hm.valid_mask, cmap="Greys", aspect="auto")
    cen = sections["censored_W50"].astype(bool).to_numpy()
    if cen.any():
        ax.plot(meta["s_sections"][cen], np.full(int(cen.sum()), 4), "rx", ms=4,
                label="W50-censored sections")
        ax.legend(loc="lower right")
    ax.set_title("valid mask + FOV censoring")
    for ax_, label in zip(axes.ravel(), ("lateral index", "lateral index",
                                         "lateral v (um)", "s (um)", "W50 (um)",
                                         "s (um)")):
        ax_.set_xlabel(label)
    identifiability = meta["identifiability"]
    fig.suptitle(
        f"group {group} | L_detected={meta['s_end']-meta['s_start']:.1f}um | "
        f"sections={len(used)} | identifiability={identifiability}", y=0.995)
    fig.tight_layout()
    fig.savefig(target, dpi=cfg["qa_montage"]["dpi"])
    plt.close(fig)


def main() -> int:
    cfg, quick = p26.load_config(__doc__)
    sl = cfg["single_line"]
    out_dir = p26.output_dir(cfg, "single_line")
    montage_dir = out_dir / "qa_montages"
    montage_dir.mkdir(parents=True, exist_ok=True)
    view = pd.read_csv(cfg["paths"]["line_view_manifest"], encoding="utf-8-sig")
    view_field = view.set_index(view["measurement_id"].astype(int))
    groups = sorted(int(g) for g in view_field.index)
    if quick:
        groups = [g for g in PILOT_GROUPS if g in groups]
    if cfg["_limit"]:
        groups = groups[:cfg["_limit"]]
    p26.log(f"Task 16 start | quick={quick} | groups={len(groups)}")

    v_positions = p26.lateral_positions(64, sl["pixel_um"])
    cone_cfg = ConicalDropoutConfig(**{
        key: value for key, value in cfg["cone_repair"].items()
        if key in ConicalDropoutConfig.__dataclass_fields__})

    section_rows, line_rows, scan_rows = [], [], []
    reader = CagHeightReader(cfg["paths"]["line_cag"])
    try:
        for group in groups:
            view_row = view_field.loc[group]
            hm = reader.read_height_map(group)
            fit = PlaneFit(a=float(view_row["plane_a"]), b=float(view_row["plane_b"]),
                           c=float(view_row["plane_c"]),
                           rmse_um=float(view_row["plane_rmse_um"]),
                           sigma_ref_um=float(view_row["sigma_ref_um"]),
                           n_inliers=-1)
            depth = plane_depth(hm.z, hm.valid_mask, hm.dx_um, hm.dy_um, fit)
            theta = float(view_row["theta_line_deg"])
            anchor = (float(view_row["orientation_center_x_um"]),
                      float(view_row["orientation_center_y_um"]))
            t_hat, _ = p26.axis_frame(theta)
            s_bounds = fov_s_interval(anchor, t_hat, hm.width_um / 2.0,
                                      hm.height_um / 2.0, margin_um=1.0)
            p26.require(s_bounds is not None,
                        f"group {group}: anchor outside FOV along axis")
            threshold = float(view_row["orientation_threshold_um"])
            p26.require(abs(threshold - sl["detection_threshold_k"] * fit.sigma_ref_um)
                        < 1e-6, f"group {group}: threshold frame mismatch")
            s_scan = np.arange(s_bounds[0], s_bounds[1] + 1e-9, sl["fine_step_um"])
            profiles_fine, _ = p26.sample_profiles(
                depth, hm.valid_mask, hm, theta, anchor, s_scan, v_positions,
                order=sl["rotate_order"], mask_weight_min=sl["mask_weight_min"])
            online = p26.detect_online_flags(
                profiles_fine, threshold, sl["min_profile_points"])
            s_start, s_end = p26.line_extent(
                s_scan, online, min_run_um=sl["min_extent_run_um"],
                merge_gap_um=sl["extent_merge_gap_um"])
            depth_p95_s, abs_width_s = p26.scan_plateau_features(
                profiles_fine, threshold, hm.dy_um)
            fragmented = False
            try:
                stable_flags, stable_start, stable_end = p26.plateau_stable_run(
                    s_scan, online, depth_p95_s, abs_width_s,
                    depth_frac=sl["stable_region"]["depth_frac"],
                    ref_quantile=sl["stable_region"]["ref_quantile"],
                    width_band_frac=sl["stable_region"]["width_band_frac"],
                    gap_merge_um=sl["stable_region"]["gap_merge_um"],
                    min_stable_len_um=sl["stable_region"]["min_stable_len_um"],
                    min_stable_frac=sl["stable_region"]["min_stable_frac"])
                stable = (stable_start, stable_end)
                s_sections = p26.section_positions(stable, sl["cross_section_step_um"])
            except p26.FragmentedStableRegion:
                fragmented = True
                stable_flags = np.zeros_like(online, dtype=bool)
                stable = (np.nan, np.nan)
                s_sections = np.array([], dtype=float)
            empty_sections = pd.DataFrame({
                "s_um": pd.Series(dtype=float),
                **{key: pd.Series(dtype=float) for key in (
                    "n_valid_samples", "n_above_threshold", "D_max_um",
                    "A_remove_um2", "W_eq_um", "W20_um", "W50_um", "W80_um",
                    "n_runs_W20", "n_runs_W50", "n_runs_W80",
                    "total_width_W20_um", "total_width_W50_um",
                    "total_width_W80_um", "censored_W20", "censored_W50",
                    "censored_W80", "W_affected_um", "left_slope", "right_slope",
                    "edge_asymmetry", "ridge_left_um", "ridge_right_um",
                    "ridge_separation_um", "profile_skewness")}})

            def extract(depth_field: np.ndarray) -> pd.DataFrame:
                profiles, _ = p26.sample_profiles(
                    depth_field, hm.valid_mask, hm, theta, anchor, s_sections,
                    v_positions, order=sl["rotate_order"],
                    mask_weight_min=sl["mask_weight_min"])
                frame = pd.DataFrame({"s_um": s_sections})
                features = [p26.section_features(
                    profiles[i], v_positions,
                    tuple(cfg["widths"]["thresholds_q"]),
                    affected_delta_um=max(cfg["widths"]["affected_delta_um"],
                                          3.0 * fit.rmse_um))
                    for i in range(len(s_sections))]
                return pd.concat([frame, pd.DataFrame(features)], axis=1)

            if not fragmented:
                sections_raw = extract(depth)
                z_rep, repair_mask, rep_components, _ = repair_compact_dropouts(
                    hm.z, hm.valid_mask, dx_um=hm.dx_um, dy_um=hm.dy_um,
                    config=cone_cfg)
                sections_rep = extract(plane_depth(z_rep, hm.valid_mask,
                                                   hm.dx_um, hm.dy_um, fit))
            else:
                sections_raw = empty_sections.copy()
                sections_rep = empty_sections.copy()
                z_rep = hm.z
                repair_mask = np.zeros_like(hm.valid_mask)
                rep_components = []
            for frame, arm in ((sections_raw, "raw"), (sections_rep, "repaired")):
                frame.insert(0, "arm", arm)
                frame.insert(0, "single_line_id", group)
            section_rows.extend([sections_raw, sections_rep])

            aggregate = p26.aggregate_line(
                sections_raw, min_sections=sl["min_sections"],
                censored_frac_limit=sl["censored_frac_W50_uncertain_above"])
            aggregate_rep = p26.aggregate_line(
                sections_rep, min_sections=sl["min_sections"],
                censored_frac_limit=sl["censored_frac_W50_uncertain_above"])
            flags = auto_qc_flags(aggregate, (s_start, s_end), s_bounds, cfg)
            if fragmented:
                flags.append("fragmented_stable_region")
            center = (s_start + s_end) / 2.0
            stable_flags = (s_scan >= stable[0]) & (s_scan <= stable[1])
            scan_rows.append(pd.DataFrame({
                "加工顺序": group,
                "s_center_um": s_scan - center,
                "stable_flag": stable_flags.astype(int),
                "depth_p95": depth_p95_s}))
            line_row = {
                "single_line_id": group,
                "theta_line_deg": theta,
                "L_detected_um": s_end - s_start,
                "s_start_um": s_start, "s_end_um": s_end,
                "stable_start_um": stable[0], "stable_end_um": stable[1],
                "n_cones_repaired": len(rep_components),
                "n_repair_pixels": int(repair_mask.sum()),
                "qc_auto_flags": "|".join(flags) if flags else "ok",
            }
            line_row.update({key: aggregate[key] for key in LINE_KEYS})
            for key in LINE_KEYS:
                line_row[f"{key}_rep"] = aggregate_rep[key]
            line_rows.append(line_row)
            meta = {"theta": theta, "anchor": anchor, "s_start": s_start,
                    "s_end": s_end, "stable": stable, "s_sections": s_sections,
                    "v_positions": v_positions,
                    "identifiability": aggregate["width_identifiability"]}
            if not quick:
                render_montage(montage_dir / f"group_{group:03d}_qa.png",
                               group, hm, depth, meta, sections_raw, cfg)
            print(f"group {group:3d}: L={s_end-s_start:6.1f}um "
                  f"sections={aggregate['n_sections_used']:3d} "
                  f"median_W50={aggregate['median_W50_um']:.2f}um "
                  f"cens={aggregate['censored_frac_W50']:.2f} "
                  f"id={aggregate['width_identifiability']:<21s} "
                  f"qc={'|'.join(flags) if flags else 'ok'}", flush=True)
    finally:
        reader.close()

    sections_frame = pd.concat(section_rows, ignore_index=True)
    geometry = pd.DataFrame(line_rows)
    p26.require((geometry["width_identifiability"]
                 .isin(cfg["widths"]["identifiability_states"])).all(),
                "width_identifiability must hold frozen states")
    sections_frame.to_csv(out_dir / "cross_section_widths.csv",
                          index=False, encoding="utf-8-sig")
    geometry.to_csv(out_dir / "single_line_geometry.csv",
                    index=False, encoding="utf-8-sig")
    p26.log(f"cross_section_widths.csv: {len(sections_frame)} rows; "
            f"single_line_geometry.csv: {len(geometry)} rows")

    if not quick:
        pilot_scan = pd.concat(scan_rows, ignore_index=True)
        pilot_long = pd.read_csv(cfg["paths"]["pilot_longitudinal_csv"])
        pilot_features = pd.read_csv(cfg["paths"]["pilot_features_csv"])
        valid_ids = geometry.loc[geometry["width_identifiability"]
                                 != "insufficient_sections", "single_line_id"]
        available = tuple(g for g in valid_ids.tolist()
                          if g in set(pilot_long["加工顺序"].unique().tolist()))
        reconciliation = p26.reconcile_stable_region(
            pilot_scan, pilot_long, groups=available,
            tol_um=sl["pilot_reconcile_match_tol_um"],
            shallow_frac=sl["stable_region"]["depth_frac"],
            max_shift_um=sl["pilot_reconcile_max_shift_um"])
        p26.require(len(available) > 0 and len(reconciliation) == len(available),
                    "pilot reconciliation matched no groups")
        p26.require((reconciliation["n_shallow_invaded"]
                     <= sl["pilot_reconcile_max_shallow_invaded_per_group"]).all()
                    and int(reconciliation["n_shallow_invaded"].sum())
                    <= sl["pilot_reconcile_max_shallow_invaded_total"],
                    "shallow partial-ablation segments retained inside the "
                    "stable region (pilot reconciliation):\n"
                    f"{reconciliation.to_string()}")
        geometry_index = geometry.set_index("single_line_id")
        reconciliation["my_median_W50_um"] = [
            float(geometry_index.loc[g, "median_W50_um"]) for g in reconciliation["加工顺序"]]
        reconciliation["pilot_W_line_um"] = [
            float(pilot_features.set_index("加工顺序").loc[g, "W_line_um"])
            for g in reconciliation["加工顺序"]]
        reconciliation["W50_rel_diff_vs_pilot"] = (
            reconciliation["my_median_W50_um"] / reconciliation["pilot_W_line_um"] - 1.0)
        reconciliation.to_csv(out_dir / "stable_region_reconciliation.csv",
                              index=False, encoding="utf-8-sig")
        p26.log(f"pilot reconciliation OK: max shallow invaded per group "
                f"{int(reconciliation['n_shallow_invaded'].max())} "
                f"(max frac {reconciliation['shallow_invasion_frac'].max():.3f}); "
                f"precision {reconciliation['precision'].min():.3f}.."
                f"{reconciliation['precision'].max():.3f} and agreement "
                f"{reconciliation['agreement'].min():.3f}.."
                f"{reconciliation['agreement'].max():.3f} are informational; "
                f"W50 vs pilot rel diff median "
                f"{reconciliation['W50_rel_diff_vs_pilot'].median():+.3f}")

    labels_path = out_dir / "geometry_qa_labels.csv"
    labels_template = pd.DataFrame({
        "single_line_id": geometry["single_line_id"],
        "qa_label": "",
        "annotator": "",
        "timestamp_utc": "",
        "comment": "",
    })
    if not labels_path.exists():
        labels_template.to_csv(labels_path, index=False, encoding="utf-8-sig")
        p26.log(f"QA label template -> {labels_path} "
                "(blind: fill qa_label in {usable, uncertain, reject_geometry}; "
                "montages carry geometry only)")
    else:
        existing = pd.read_csv(labels_path, encoding="utf-8-sig",
                               keep_default_na=False)
        p26.require(set(geometry["single_line_id"])
                    <= set(existing["single_line_id"]),
                    "QA label template misses some ids")
        p26.log(f"QA label template preserved ({len(existing)} rows)")
    p26.log("Task 16 done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
