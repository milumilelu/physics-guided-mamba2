#!/usr/bin/env python3
"""Task SL-05 (细则 §8) -- orientation provenance check + G-SL4.

Frozen path (§0.8, v2 §12): provenance_valid = false.  The rectangle DOE
records only "serpentine fill" with **no per-sample scan direction and no fill
axis (x/y)**; `annotations/session_geometry.csv::theta_session_deg`
(-0.70 .. -0.45 deg, d4 = identity) is an *image rotation convention*, not a
scan direction; the frozen configs (`manual_registration_200.csv`,
`measurement_planes_160.csv`) carry no direction field; and single lines are
hatchless with an unrecorded start/end sign.

Consequences (frozen, T16-anchored):
  - scan/hatch-relative Delta-theta MUST NOT be computed;
  - G-SL4 = NOT_APPLICABLE;
  - the only thing this task produces is an IMAGE-FRAME descriptive check of
    theta_stripe(8_16) clustering near 0/90 deg, explicitly NON-EVIDENCE. It
    says nothing about scan or hatch alignment and must never be worded as
    though it did.

If a fill axis is ever registered by hand, 细则 §0.8 must be rewritten first and
only then may `gates.gsl4.provenance_valid` be flipped -- this script refuses
to run the conditional arm without a registered source (hard assertion below).

EXPECTED outputs:
    outputs/phase2_6/orientation/stripe_scan_alignment.csv
    outputs/phase2_6/orientation/orientation_provenance.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import _lib as p26  # noqa: E402

EXPECTED = [
    "outputs/phase2_6/orientation/stripe_scan_alignment.csv",
    "outputs/phase2_6/orientation/orientation_provenance.json",
]

# Any column name matching these would mean a scan/hatch-relative angle leaked
# into the output despite provenance_valid = false (T16 negative assertion).
FORBIDDEN_COLUMN_TOKENS = ("scan_relative", "hatch_relative", "delta_theta",
                           "dtheta", "scan_minus", "hatch_minus")


def axis_distance_deg(theta_deg: np.ndarray) -> np.ndarray:
    """Distance to the nearest multiple of 90 deg (image frame, axial 0-180)."""
    wrapped = np.mod(np.asarray(theta_deg, dtype=float), 90.0)
    return np.minimum(wrapped, 90.0 - wrapped)


def main() -> int:
    cfg, quick = p26.load_config(__doc__)
    seed = int(cfg["meta"]["random_seed"])
    cluster_deg = float(cfg["orientation"]["image_frame_cluster_deg"])
    provenance_valid = bool(cfg["gates"]["gsl4"]["provenance_valid"])
    n_perm = 10000 if not quick else 2000
    null_seed = seed + int(cfg["seeds"]["permutation_offset"])

    orientation_dir = p26.output_dir(cfg, "orientation")
    summary_dir = p26.output_dir(cfg, "summary")

    # ---- provenance gate (§0.8) ------------------------------------------- #
    registered_axis_source = cfg["gates"]["gsl4"].get("registered_axis_source")
    if provenance_valid:
        p26.require(bool(registered_axis_source),
                    "HARD ASSERTION FAILED: gates.gsl4.provenance_valid is true "
                    "but no registered_axis_source is recorded. Per 细则 §0.8 a "
                    "fill axis must be registered by hand and §0.8 rewritten "
                    "BEFORE the conditional arm is enabled.")
        raise NotImplementedError(
            "conditional scan/hatch-relative arm not implemented: 细则 §0.8 "
            "requires a hand-registered fill axis and a spec rewrite first.")

    # ---- image-frame descriptive arm (NON-EVIDENCE) ------------------------ #
    directional = pd.read_csv(REPO / cfg["paths"]["p25_directional_csv"],
                              encoding="utf-8-sig")
    band = directional[directional["band"].astype(str) == "8_16"].copy()
    p26.require(len(band) == 200,
                f"HARD ASSERTION FAILED: band 8_16 rows {len(band)} != 200")

    theta = band["theta_stripe_deg"].to_numpy(dtype=float)
    p26.require(bool(np.isfinite(theta).all()),
                "HARD ASSERTION FAILED: theta_stripe_8_16 must be finite")
    p26.require(bool(((theta >= 0.0) & (theta < 180.0)).all()),
                "HARD ASSERTION FAILED: theta_stripe_deg must be axial in [0,180)")

    distance = axis_distance_deg(theta)
    clustered = distance <= cluster_deg
    a_obs = float(clustered.mean())

    rng = np.random.default_rng(null_seed)
    draws = rng.uniform(0.0, 180.0, size=(n_perm, theta.size))
    null = (axis_distance_deg(draws) <= cluster_deg).mean(axis=1)
    p_value = float((1.0 + int(np.sum(null >= a_obs))) / (1.0 + n_perm))

    alignment = pd.DataFrame({
        "dataset_index": band["dataset_index"].to_numpy(),
        "theta_stripe_8_16_deg": theta,
        "distance_to_nearest_90_multiple_deg": distance,
        "clustered_image_frame": clustered,
        "A2_8_16": band["A2"].to_numpy(dtype=float),
        "theta_k_8_16_deg": band["theta_k_deg"].to_numpy(dtype=float),
        "angular_entropy_8_16": band["angular_entropy"].to_numpy(dtype=float),
    })
    p26.require(
        not any(token in column.lower()
                for column in alignment.columns
                for token in FORBIDDEN_COLUMN_TOKENS),
        "HARD ASSERTION FAILED: scan/hatch-relative angle column emitted while "
        "provenance_valid=false (细则 §8 / test T16)")
    alignment.to_csv(orientation_dir / "stripe_scan_alignment.csv", index=False,
                     encoding="utf-8-sig")

    # ---- provenance record -------------------------------------------------- #
    record = {
        "gate": "G-SL4",
        "verdict": "NOT_APPLICABLE",
        "provenance_valid": provenance_valid,
        "scan_relative_angle_computed": False,
        "reason": ("No per-sample scan direction and no fill axis (x/y) exist "
                   "for the rectangle DOE; nothing to align against."),
        "verified_facts": [
            "现有数据基础说明_v2 §12: rectangle areas use serpentine fill; the "
            "record states the fill style only -- no per-sample scan direction "
            "and no fill axis (x/y) are logged.",
            "annotations/session_geometry.csv::theta_session_deg spans "
            "-0.700 .. -0.450 deg with d4_transform_session = identity: this is "
            "an IMAGE ROTATION convention, not a scan direction.",
            "config/frozen/manual_registration_200.csv carries only "
            "theta_session_deg (same image-rotation convention); "
            "config/frozen/measurement_planes_160.csv carries no direction "
            "field at all.",
            "Single lines: line axis ~0 deg (-0.43 .. -0.77 deg) but the "
            "start/end SIGN is unrecorded, and single-line DOE has no hatch.",
        ],
        "descriptive_image_frame_check": {
            "status": "DESCRIPTIVE_ONLY__NOT_EVIDENCE",
            "statistic": ("fraction of 200 samples with theta_stripe(8_16) "
                          f"within {cluster_deg} deg of a multiple of 90 deg"),
            "A_obs": a_obs,
            "n_clustered": int(clustered.sum()),
            "n_total": int(theta.size),
            "null": {"type": "uniform_angle_permutation_on_[0,180)",
                     "n_perm": n_perm, "seed": null_seed,
                     "A_null_median": float(np.median(null)),
                     "A_null_p95": float(np.percentile(null, 95)),
                     # theta is axial in [0,180): two 90-deg periods, each with
                     # a 2*cluster_deg-wide qualifying band -> 4*c/180.
                     "analytic_expectation": float(4 * cluster_deg / 180.0)},
            "p_value": p_value,
            "wording_constraint": (
                "This clustering is measured in the IMAGE FRAME only. It must "
                "never be described as scan alignment, hatch alignment, or as "
                "evidence for/against H1/H2/H3. Without a registered fill axis "
                "the scan-relative question is genuinely undecidable."),
        },
        "conditional_arm": {
            "enabled": False,
            "unlock_condition": ("Hand-register the fill axis per sample, "
                                 "rewrite 细则 §0.8, and set "
                                 "gates.gsl4.registered_axis_source; the "
                                 "conditional arm must be implemented in the "
                                 "same change, never appended after formal."),
        },
    }
    (orientation_dir / "orientation_provenance.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    (summary_dir / "gsl4_evaluation.json").write_text(
        json.dumps({"gate": "G-SL4", "verdict": "NOT_APPLICABLE",
                    "provenance_valid": provenance_valid},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    p26.log(f"G-SL4 = NOT_APPLICABLE (provenance_valid={provenance_valid}; "
            "scan-relative angles not computed)")
    p26.log("image-frame DESCRIPTIVE check (non-evidence): "
            f"A_obs={a_obs:.4f} ({int(clustered.sum())}/{int(theta.size)}) vs "
            f"uniform null median={float(np.median(null)):.4f}, p={p_value:.4f}")
    p26.log("Task 20 done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
