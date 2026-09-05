"""Phase 2.6 shared library: single-line axis sampling, threshold widths,
lambda* descriptors, box membership, block-structured hatch shuffling.

Loads the frozen Phase 2.5 library by explicit file location (module name
`phase2_5_lib_p26`; it in turn loads Phase 2 and Phase 1.5).  No frozen
implementation is copied: composition/ILR/CV primitives are reached through
`p25.*` / `p2.*`.  Binding spec: Phase2.6_落地执行细则.md (FROZEN_EXECUTED).

Geometry convention (细则 §0.20): profiles are sampled directly along the
frozen line axis in ORIGINAL map coordinates -- no whole-map rotation.
  raw_centered = anchor + s*t_hat + v*n_hat
  t_hat = (cos theta, sin theta); n_hat = (-sin theta, cos theta)
with theta = theta_line_deg (identical convention to
`src/resampling.resample_to_canonical`: canonical u runs along t_hat,
canonical +v along n_hat) and anchor = (orientation_center_x_um,
orientation_center_y_um) from the frozen view manifest.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import ndimage

REPO = Path(__file__).resolve().parents[2]

_spec25 = importlib.util.spec_from_file_location(
    "phase2_5_lib_p26",
    Path(__file__).resolve().parents[1] / "phase2_5" / "_lib.py")
p25 = importlib.util.module_from_spec(_spec25)
_spec25.loader.exec_module(p25)
p2 = p25.p2
l15 = p25.l15

log = p25.log
require = p25.require

WIDTH_Q_KEYS = ("W20", "W50", "W80")
Q_BY_KEY = {"W20": 0.2, "W50": 0.5, "W80": 0.8}

# frozen Ridge conventions shared by Tasks 17/18 (细则 §5/§6):
#   - pipeline = StandardScaler -> Ridge (linear basis);
#   - alpha = fold-internal grid logspace(-3, 3, 13) selected by inner GKF(5)
#     mean MSE on the TRAIN fold (never the test fold).
# WP1 canonical migration (parity-tested, tests/test_src_cv.py): the frozen
# Ridge alpha protocol now lives in src/cv.py under the explicit `_v1` name;
# the frozen legacy names are thin re-exports (Phase 2.8 v2.1 section 4.2
# migration step 5).  New target-native alpha selection for Phase 2.8 is
# src.cv.select_alpha_inner and must NOT be wired into frozen phases.
from src.cv import (  # noqa: E402,F401
    make_ridge,
    make_ridge_alpha_grid,
    ridge_alpha_inner_gkf_v1 as ridge_alpha_inner_gkf,
)


def load_config(description: str) -> tuple[dict, bool]:
    """Read `phase2_6_config.yaml` next to this file; honor `--quick`."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="smoke option: only the first N groups")
    args, _ = parser.parse_known_args()
    cfg = yaml.safe_load((Path(__file__).resolve().parent
                          / "phase2_6_config.yaml").read_text(encoding="utf-8"))
    cfg["_output_root"] = (str(cfg["meta"]["quick_output_root"]) if args.quick
                           else "outputs/phase2_6")
    cfg["_quick"] = bool(args.quick)
    cfg["_limit"] = args.limit
    return cfg, bool(args.quick)


def output_dir(cfg: dict, sub: str = "") -> Path:
    path = REPO / cfg["_output_root"]
    if sub:
        path = path / sub
    path.mkdir(parents=True, exist_ok=True)
    return path


# WP1 canonical migration (parity-tested, tests/test_src_geometry.py): the
# single-line geometry machinery now lives in src/geometry.py; the frozen
# names are thin re-exports (Phase 2.8 v2.1 section 4.2 migration step 5).
from src.geometry import (  # noqa: E402,F401
    CODE_INVALID,
    CODE_M1,
    CODE_M2,
    CODE_M3,
    CODE_OUT,
    CLASS_NAMES,
    FragmentedStableRegion,
    INTERVALS,
    Q_BY_KEY,
    WIDTH_Q_KEYS,
    aggregate_line,
    axis_frame,
    condition_key,
    detect_online_flags,
    in_box_mask,
    lambda_peak_4_32,
    lambda_star_4_32,
    lateral_positions,
    line_extent,
    plateau_stable_run,
    reconcile_stable_region,
    sample_profiles,
    scan_plateau_features,
    section_features,
    section_positions,
    shuffle_h_by_block,
    stable_region,
)
from src.geometry import (  # noqa: E402,F401
    _pixel_indices,
    _ridge,
    _run_boundaries,
    _slope,
)

