"""Phase 2.7 shared library: m/OUT interval assignment, two-layer class
distributions, four/five-class TV, block-shuffle permutation p, Hann Fourier
projection, finite-array 2D synthesis (same Phase 2.5 spectrum pipeline),
LOHO period-2 selection, and the frozen G27-3 verdict order.

Loads the frozen Phase 2.6 library by explicit file location (module name
`phase2_6_lib_p27`; it chains p25/p2/l15).  No frozen implementation is
copied.  Binding spec: Phase2.7_落地执行细则.md (FROZEN) + 任务说明 v2.1 (FROZEN).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]

_spec26 = importlib.util.spec_from_file_location(
    "phase2_6_lib_p27",
    Path(__file__).resolve().parents[1] / "phase2_6" / "_lib.py")
p26 = importlib.util.module_from_spec(_spec26)
_spec26.loader.exec_module(p26)
p25 = p26.p25
p2 = p26.p2
l15 = p26.l15

log = p26.log
require = p26.require
# re-exports from the frozen Phase 2.6 library (no copying)
shuffle_h_by_block = p26.shuffle_h_by_block
in_box_mask = p26.in_box_mask
ridge_alpha_inner_gkf = p26.ridge_alpha_inner_gkf
make_ridge = p26.make_ridge
sample_profiles = p26.sample_profiles
lateral_positions = p26.lateral_positions
axis_frame = p26.axis_frame
detect_online_flags = p26.detect_online_flags
line_extent = p26.line_extent
scan_plateau_features = p26.scan_plateau_features
plateau_stable_run = p26.plateau_stable_run
lambda_star_4_32 = p26.lambda_star_4_32
lambda_peak_4_32 = p26.lambda_peak_4_32


def load_config(description: str) -> tuple[dict, bool]:
    """Read `phase2_7_config.yaml` next to this file; honor `--quick`."""
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--quick", action="store_true")
    args, _ = parser.parse_known_args()
    cfg = yaml.safe_load((Path(__file__).resolve().parent
                          / "phase2_7_config.yaml").read_text(encoding="utf-8"))
    cfg["_output_root"] = ("outputs/phase2_7_quick" if args.quick
                           else "outputs/phase2_7")
    cfg["_quick"] = bool(args.quick)
    return cfg, bool(args.quick)


def output_dir(cfg: dict, sub: str = "") -> Path:
    path = REPO / cfg["_output_root"]
    if sub:
        path = path / sub
    path.mkdir(parents=True, exist_ok=True)
    return path

# five-class assignment constants + functions: canonical implementation in
# src/geometry.py (WP1 migration, parity-tested); frozen names kept as
# thin re-exports (Phase 2.8 v2.1 section 4.2 migration step 5).
from src.geometry import (  # noqa: E402,F401
    CLASS_NAMES,
    CODE_INVALID,
    CODE_M1,
    CODE_M2,
    CODE_M3,
    CODE_OUT,
    INTERVALS,
    assign_class,
    profile_suitable,
    q_distribution,
)

# thin re-exports (Phase 2.8 v2.1 section 4.2 migration step 5).
from src.statistics import logistic_slope, tv, tv_perm_p  # noqa: E402,F401


# WP1 canonical migration (parity-tested, tests/test_src_forward_models.py):
# forward models now live in src/forward_models.py; the ILR-space Q2 lives
# in src/cv.py (v1 name).  Frozen names kept as thin re-exports
# (Phase 2.8 v2.1 section 4.2 migration step 5).  verdict_g27_3 stays
# local: frozen G27-3 gate-order logic with no Phase 2.8 consumer.
from src.cv import q2_aitchison_ilr_v1 as q2_aitchison_ilr  # noqa: E402,F401
from src.forward_models import (  # noqa: E402,F401
    cycles_level,
    field_class,
    hann_projection,
    synth_field,
)


def verdict_g27_3(tv_w_const: float, tv_w_p2: float, delta_tv: float,
                  ci_low: float, n_h_win: int, n_h_evaluable: int,
                  d_i_values: list[float], n_eval: int, *, thresholds: dict
                  ) -> dict:
    """Frozen v2.1 verdict order (mutually exclusive):
    MODEL_INADEQUATE → NOT_SUPPORTED → SUPPORTED/PARTIAL → d_i guard cap."""
    tv = thresholds["tv"]
    inadequate = (tv_w_const > tv["inadequate"]
                  and tv_w_p2 > tv["inadequate"])
    if inadequate:
        return {"G_SL3": "MODEL_INADEQUATE",
                "note": "linear array model family insufficient; material "
                        "nonlinearity is one candidate, not established"}
    if delta_tv <= 0 and (tv_w_const <= tv["inadequate"]
                          or tv_w_p2 <= tv["inadequate"]):
        verdict = "NOT_SUPPORTED"
    else:
        cond = (delta_tv >= tv["delta_min"]
                and tv_w_p2 <= tv["period2_max"]
                and ci_low > 0
                and n_h_win >= thresholds["h_consistency"]["min_wins"]
                and n_h_evaluable >= thresholds["h_consistency"]["min_evaluable"])
        verdict = "SUPPORTED" if cond else "PARTIAL"
    contradictions = int(sum(1 for d in d_i_values if d < 0))
    if (n_eval >= thresholds["d_guard"]["n_eval_min"]
            and contradictions / n_eval
            > thresholds["d_guard"]["contradiction_frac"]
            and verdict == "SUPPORTED"):
        verdict = "PARTIAL"
    return {"G_SL3": verdict,
            "n_hard_contradictions": contradictions, "n_eval": n_eval}
