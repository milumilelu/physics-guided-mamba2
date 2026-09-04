#!/usr/bin/env python3
"""Task SL-03 + M0-M3 (细则 §6) -- formal implementation pending.

Skeleton frozen at the Phase 2.6 pre-freeze commit (细则 §0.14).  Key frozen
pieces this file must implement: SL-03a exact-match direct bridge (§0.17,
evidence priority direct > in-box predicted > out-of-box), the M0b transform
control, M_GEO geometry-compression arm, the Geometry-compression Gate
(G-SL3, §0.13 rev2), Aitchison Q2 only on ilr_z1_z4 (§0.13 rev2), and the
M0_RECON_FULL200 QA step vs phase2_5 cv_fold_results (§0.16).

EXPECTED outputs:
#   outputs/phase2_6/scale_bridge/morphology_scale_match.csv
#   outputs/phase2_6/scale_bridge/direct_bridge_exact_match.csv
#   outputs/phase2_6/model_compare/width_bridge_cv.csv
#   outputs/phase2_6/model_compare/overlap_bridge_cv.csv
#   outputs/phase2_6/model_compare/oof_predictions.csv
#   outputs/phase2_6/model_compare/m0_reconciliation.csv
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
        f"Task 17 outputs exist. quick={quick}")


if __name__ == "__main__":
    raise SystemExit(main())
