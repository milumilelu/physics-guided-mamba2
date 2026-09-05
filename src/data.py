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
