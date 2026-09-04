#!/usr/bin/env python3
"""Task SL-04 (细则 §7) -- formal implementation pending.

Skeleton frozen at the Phase 2.6 pre-freeze commit (细则 §0.14).  Key frozen
pieces this file must implement: H2 primary statistic on the VALID restricted
peak lambda_peak_4_32 (n_modes >= 20 AND peak bin energy share >= 0.20 of the
window, §0.18 rev2), centroid-based r_h as a parallel sensitivity arm, the
DOE-block-structured shuffled-h null (units = unique [session_id,
base_condition_group]; formal 120 row-units, pass 15 bases, supplement 10
bases; §0.19 rev2), and G-SL2 = A_obs >= 0.40 AND p <= 0.05.

EXPECTED outputs:
#   outputs/phase2_6/scale_bridge/lambda_over_hatch.csv
#   outputs/phase2_6/scale_bridge/lambda_over_width.csv
#   outputs/phase2_6/scale_bridge/overlap_metrics.csv
#   outputs/phase2_6/scale_bridge/shuffled_h_null.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import _lib as p26  # noqa: E402


def main() -> int:
    cfg, quick = p26.load_config(__doc__)
    raise NotImplementedError(
        "formal implementation pending (pre-freeze skeleton); run only after "
        f"Task 18 outputs exist. quick={quick}")


if __name__ == "__main__":
    raise SystemExit(main())
