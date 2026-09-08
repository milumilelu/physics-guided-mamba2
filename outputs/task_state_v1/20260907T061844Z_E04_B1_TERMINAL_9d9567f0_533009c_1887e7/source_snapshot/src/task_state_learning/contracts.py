"""Strict local identities and canonical terminal target export; no training."""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from src import data as canonical_data
from .artifacts import ROOT
from .grouping import KEYS, require

TARGETS = ["D", "A_med", "ilr_z1", "ilr_z2", "ilr_z3", "ilr_z4", "A2_8_16", "entropy_8_16"]


def load_contract():
    path = ROOT / "experiments/phase2_8_r1/phase2_8_config.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    frozen = canonical_data.load_frozen(cfg)
    df = pd.read_csv(ROOT / cfg["paths"]["phase2_manifest"]).sort_values("dataset_index").reset_index(drop=True)
    require(len(df) == 200 and np.array_equal(df.dataset_index, np.arange(200)), "Dataset index mismatch")
    require(not df.duplicated(KEYS).any(), "Duplicate specimen identity")
    require(np.array_equal(df[KEYS].astype(str), frozen["man"][KEYS].astype(str)), "Manifest identity mismatch")
    require(df.shared_height_source_id.nunique() == 160, "Source count mismatch")
    require(df.cv_process_group.nunique() == 134, "Historical group count mismatch")
    require(df.session_role.value_counts().to_dict() == {"formal": 120, "pass_main": 60, "pass_supplement": 20}, "Role mismatch")
    H, V, R = frozen["H"], frozen["V"], frozen["R"]
    require(H.shape == V.shape == (200, 160, 160), "Height/mask shape mismatch")
    require(V.reshape(200, -1).any(axis=1).all(), "Empty valid mask")
    require(np.isfinite(H[V]).all(), "Nonfinite valid height")
    require(np.array_equal(df.shared_height_source_id, frozen["man"].shared_height_source_id), "Source identity mismatch")
    raw_rows, paths = [], [path]
    for key in ["exploration_manifest", "dataset_npz", "phase2_manifest", "ilr_csv", "directional_csv"]:
        paths.append(ROOT / cfg["paths"][key])
    for i, row in df.iterrows():
        raw_path = ROOT / cfg["paths"]["raw_height_dir"] / f"{row.session_id}__sample_{int(row.sample_id):03d}.npz"
        paths.append(raw_path)
        with np.load(raw_path, allow_pickle=False) as raw:
            h, v = raw["height"], raw["valid_mask"].astype(bool)
        require(h.shape == H[i].shape and v.shape == V[i].shape, "Raw shape mismatch")
        require(np.array_equal(v, V[i]), "Raw mask mismatch")
        require(np.isfinite(h[v]).all(), "Raw nonfinite height")
        error = float(np.abs(h[v] - H[i][v]).max())
        require(error <= 1e-4, f"Raw height mismatch at {i}")
        raw_rows.append({"dataset_index": i, "max_abs_error_um": error, "mask_equal": True})
    df["D"] = -np.nanmedian(frozen["Hnan"], axis=(1, 2))
    df["A_med"] = np.sqrt(np.nanmean(R**2, axis=(1, 2)))
    ilr = pd.read_csv(ROOT / cfg["paths"]["ilr_csv"])
    require(len(ilr) == 200 and set(ilr.dataset_index) == set(range(200)), "ILR identity mismatch")
    df = df.merge(ilr, on="dataset_index", validate="one_to_one")
    directional = pd.read_csv(ROOT / cfg["paths"]["directional_csv"])
    for band in ["8_16", "16_32"]:
        d = directional[directional.band.eq(band)][["dataset_index", "A2", "angular_entropy"]]
        require(len(d) == 200 and set(d.dataset_index) == set(range(200)), "Direction identity mismatch")
        df = df.merge(d.rename(columns={"A2": f"A2_{band}", "angular_entropy": f"entropy_{band}"}),
                      on="dataset_index", validate="one_to_one")
    require(np.isfinite(df[TARGETS].to_numpy()).all(), "Nonfinite targets")
    paths.extend([ROOT / "outputs/phase2_8_r1/folds/fold_assignments.csv",
                  ROOT / "outputs/phase2_8_r1/predictability_oof.csv",
                  ROOT / "outputs/phase2_8_r1/predictability_spectrum.csv",
                  ROOT / "experiments/phase2_5/phase2_5_config.yaml",
                  ROOT / "experiments/phase2_8_r1/24_information_decomposition.py"])
    paths.extend(sorted((ROOT / "src").glob("*.py")))
    return cfg, frozen, df, pd.DataFrame(raw_rows), paths
