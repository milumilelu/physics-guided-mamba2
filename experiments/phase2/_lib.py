"""Phase 2 shared library: manifest build, derived coordinates, CV contracts.

Loads the frozen Phase 1.5 library by explicit file location (module name
`phase1_5_lib`) so that this file and `experiments/phase1_5/_lib.py` can
coexist under the name `_lib` for their respective scripts. No implementation
is copied from Phase 1.5.

Derived process coordinates (units; anchors unit-tested):
  scan_spacing_um          = v[mm/s] / f[kHz]                      (5.5 for v=11, f=2)
  pulse_energy_proxy_uJ    = 1000 * P[W] / f[kHz]                  (2666.65 for P=5.3333, f=2)
  areal_pulse_density_mm2  = 1e6 * N * f[kHz] / (v[mm/s] * h[um])  (74074.07 for N=2, f=2, v=9, h=6)
  areal_dose_proxy_J_mm2   = 1000 * P[W] * N / (v[mm/s] * h[um])   (197.90  for P=5.3333, N=2, v=9, h=6)

Energy/dose carry the `_proxy` suffix and provisional status until the laser
power measurement record is registered (现有数据基础说明_v2 §11).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location(
    "phase1_5_lib",
    Path(__file__).resolve().parents[1] / "phase1_5" / "_lib.py")
l15 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(l15)

# S01–S10 supplement base conditions are process-quadruple-identical to these
# pass_main groups (verified 10/10 on (pulse_duration_fs, frequency_kHz,
# hatch_spacing_um, velocity_mm_s)); anchored by unit test.
S_TO_T = {"S01": "T01", "S02": "T02", "S03": "T06", "S04": "T07", "S05": "T08",
          "S06": "T10", "S07": "T12", "S08": "T13", "S09": "T14", "S10": "T15"}


def log(message: str = "") -> None:
    print(message, flush=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(f"HARD ASSERTION FAILED: {message}")


def load_config(description: str) -> tuple[dict, bool]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(Path(__file__).with_name(
        "phase2_config.yaml")))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    with open(REPO / args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if args.quick:
        # Quick outputs are fully isolated from the formal root (细则 §0.17):
        # same script tree, separate output root, so a smoke run can never
        # overwrite formal results and mixed quick/formal artifacts are
        # impossible. The quick chain re-runs 01/04 first (identical
        # deterministic copies, cheap).
        import copy
        cfg = copy.deepcopy(cfg)
        cfg["paths"]["output_root"] = cfg["paths"]["output_root"] + "_quick"
    return cfg, args.quick


def output_dir(cfg: dict, sub: str = "") -> Path:
    out = REPO / cfg["paths"]["output_root"]
    if sub:
        out = out / sub
    out.mkdir(parents=True, exist_ok=True)
    return out


# --------------------------------------------------------------------------- #
# derived process coordinates -- canonical implementations now live in
# src/provenance.py (WP1 migration; parity-tested in
# tests/test_src_provenance.py).  The frozen legacy names are kept as thin
# re-exports so build_manifest and its tests are untouched
# (Phase 2.8 v2.1 §4.2 migration protocol step 5).  Power provenance:
# P_obj = 5.3333 W is the canonical post-objective measurement; the `_proxy`
# function names are legacy aliases kept for the Phase 2-2.7 chain.
# --------------------------------------------------------------------------- #

from src.provenance import (  # noqa: E402
    areal_dose_J_per_mm2 as areal_dose_proxy_j_mm2,
    areal_pulse_density,
    pulse_energy_uJ as pulse_energy_proxy_uJ,
    scan_spacing_um,
)


def _g(x) -> str:
    return f"{float(x):g}"


# --------------------------------------------------------------------------- #
# phase2 manifest
# --------------------------------------------------------------------------- #

PROC_RAW_COLS = ["pulse_duration_fs", "frequency_kHz", "hatch_spacing_um",
                 "pass_count", "velocity_mm_s"]
PROC_PHYS_COLS = ["pulse_duration_fs", "pulse_energy_proxy_uJ",
                  "scan_spacing_um", "areal_pulse_density_per_mm2",
                  "areal_dose_proxy_J_per_mm2"]
LOCO_COLS = ["phase1_global_loco_rank", "phase1_global_loco_angle_deg"]


def build_manifest(cfg: dict) -> pd.DataFrame:
    """Pure join of frozen inputs + derived coordinates + grouping keys."""
    man = pd.read_csv(REPO / cfg["paths"]["exploration_manifest"])
    require(len(man) == 200, f"manifest rows {len(man)} != 200")
    require(not man.duplicated(["session_id", "sample_id"]).any(),
            "(session_id, sample_id) not unique")
    require(man["shared_height_source_id"].nunique() == 160,
            "unique shared sources != 160")

    planes = pd.read_csv(REPO / "config/frozen/measurement_planes_160.csv",
                         encoding="utf-8-sig")
    man = man.merge(planes[["session_id", "measurement_id", "rmse_um",
                            "status"]]
                    .rename(columns={"rmse_um": "plane_rmse_um",
                                     "status": "plane_status"}),
                    on=["session_id", "measurement_id"],
                    how="left", validate="many_to_one")
    require(man["plane_rmse_um"].notna().all(), "plane join left NaNs")

    pw = cfg["power"]
    man["measured_power_W"] = float(pw["measured_power_W"])
    man["nominal_software_power_W"] = float(pw["nominal_software_power_W"])
    man["power_measurement_source"] = pw["source"]
    man["power_measurement_version"] = "PENDING_REGISTRATION"
    man["constant_power_assumption"] = True
    man["pulse_energy_proxy_uJ"] = pulse_energy_proxy_uJ(
        pw["measured_power_W"], man["frequency_kHz"])
    man["scan_spacing_um"] = scan_spacing_um(man["velocity_mm_s"],
                                             man["frequency_kHz"])
    man["areal_pulse_density_per_mm2"] = areal_pulse_density(
        man["pass_count"], man["frequency_kHz"], man["velocity_mm_s"],
        man["hatch_spacing_um"])
    man["areal_dose_proxy_J_per_mm2"] = areal_dose_proxy_j_mm2(
        pw["measured_power_W"], man["pass_count"], man["velocity_mm_s"],
        man["hatch_spacing_um"])

    qk = (man["pulse_duration_fs"].map(_g) + ":" + man["frequency_kHz"].map(_g)
          + ":" + man["hatch_spacing_um"].map(_g) + ":"
          + man["velocity_mm_s"].map(_g))
    man["quad_key"] = qk

    dg = man["design_group"]
    is_pass = dg.notna().to_numpy()
    base = np.where(
        is_pass,
        dg.map(lambda g: S_TO_T.get(g, g)).astype(str).to_numpy(),
        "F" + man["processing_order"].astype(str))
    man["base_condition_group"] = base

    is_formal = (man["session_role"] == "formal").to_numpy()
    man["cv_process_group"] = np.where(
        is_formal,
        "FQ:" + qk + "|N" + man["pass_count"].map(_g),
        "BASE:" + man["base_condition_group"].astype(str))

    # QA contract (细则 §2.4)
    counts = man["session_role"].value_counts().to_dict()
    require(counts == {"formal": 120, "pass_main": 60, "pass_supplement": 20},
            f"session_role counts {counts}")
    require(int(dg.notna().sum()) == 80, "design_group must cover exactly 80")
    require(man["base_condition_group"].nunique() == 135,
            f"base_condition_group groups "
            f"{man['base_condition_group'].nunique()} != 135")
    require(man["cv_process_group"].nunique() == 134,
            f"cv_process_group groups {man['cv_process_group'].nunique()} != 134")
    require(float(man["valid_fraction"].min()) == 1.0, "valid_fraction < 1")
    proc_cols = PROC_RAW_COLS + ["pulse_energy_proxy_uJ", "scan_spacing_um",
                                 "areal_pulse_density_per_mm2",
                                 "areal_dose_proxy_J_per_mm2"]
    require(man[proc_cols].notna().all().all(), "NaN in process columns")
    sent = man[(man["session_id"] == "zro2_120_formal")
               & (man["processing_order"].isin(cfg["sentinel"]["processing_orders"]))]
    require(len(sent) == 2, "sentinel rows != 2")
    require(sent["quad_key"].nunique() == 1
            and sent["cv_process_group"].nunique() == 1,
            "49/50 must share quad_key and cv_process_group")
    return man


MANIFEST_README = """# phase2_manifest

Built by `experiments/phase2/_lib.build_manifest` from frozen inputs; see
Phase2_执行细则.md §2 for the column contract. `phase1_global_loco_*` columns
are backfilled by `01_instability_inventory.py`.

Power note: `measured_power_W` = 5.3333 W (post-objective, v2 §11) is
**provisional** — no independent measurement record is registered yet, so
`pulse_energy_proxy_uJ` / `areal_dose_proxy_J_per_mm2` carry the `_proxy`
suffix and must not be used in conclusive language until registration.
"""


def write_manifest(cfg: dict, man: pd.DataFrame) -> Path:
    out = output_dir(cfg, "manifest")
    path = out / "phase2_manifest.csv"
    man.to_csv(path, index=False)
    (out / "README.md").write_text(MANIFEST_README, encoding="utf-8")
    return path


def read_manifest(cfg: dict, require_loco: bool = False) -> pd.DataFrame:
    path = output_dir(cfg, "manifest") / "phase2_manifest.csv"
    require(path.exists(), "phase2_manifest.csv missing; run 01 first")
    man = pd.read_csv(path)
    if require_loco:
        require(set(LOCO_COLS) <= set(man.columns),
                "manifest lacks LOCO backfill; run 01 first")
    return man


# --------------------------------------------------------------------------- #
# distances, kNN, consensus
# --------------------------------------------------------------------------- #

def robust_z(A) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    med = np.median(A, axis=0)
    q25, q75 = np.percentile(A, [25, 75], axis=0)
    iqr = q75 - q25
    require(np.all(iqr > 0), "zero-IQR column in robust standardization")
    return (A - med) / iqr


def zscore(A) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    sd = A.std(axis=0, ddof=0)
    require(np.all(sd > 0), "zero-variance column in z-score")
    return (A - A.mean(axis=0)) / sd


def knn_median_distance(Z: np.ndarray, k: int) -> np.ndarray:
    """Median Euclidean distance to the k nearest other rows (self-excluded)."""
    Z = np.asarray(Z, dtype=float)
    D = np.sqrt(np.maximum(
        ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1), 0.0))
    np.fill_diagonal(D, np.inf)
    part = np.partition(D, k - 1, axis=1)[:, :k]
    return np.median(part, axis=1)


def consensus_rank(rank_sq, rank_loco, rank_dmorph, rank_pit,
                   band_ranks) -> tuple[float, float]:
    """A_consensus = median of five ranks; the spectral term is the MOST
    anomalous of the four DCT band energy-fraction ranks (细则 §0.12)."""
    spectral = float(np.min(np.asarray(band_ranks, dtype=float)))
    cons = float(np.median([float(rank_sq), float(rank_loco),
                            float(rank_dmorph), float(rank_pit), spectral]))
    return cons, spectral


def sentinel_normalize(d_morph, d_sentinel: float) -> np.ndarray:
    require(float(d_sentinel) > 0, "sentinel distance must be positive")
    return np.asarray(d_morph, dtype=float) / float(d_sentinel)


def process_near_morph_level(d_proc, d_morph, q: float = 0.10) -> tuple[float, float]:
    """T_lambda = median[D_morph | D_proc <= P_q] plus the threshold itself."""
    d_proc = np.asarray(d_proc, dtype=float)
    d_morph = np.asarray(d_morph, dtype=float)
    thr = float(np.quantile(d_proc, q))
    sel = d_proc <= thr
    require(sel.any(), "empty process-near set")
    return float(np.median(d_morph[sel])), thr


# --------------------------------------------------------------------------- #
# grouped CV splits and contracts (细则 §7.3: per-split-type rules)
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# grouped CV splits + contracts -- canonical implementations now live in
# src/cv.py (WP1 migration; parity-tested in tests/test_src_cv.py).  Frozen
# split semantics (Phase 2.8 v2.1 F1): src_gkf = GroupKFold on
# shared_height_source_id, proc_gkf = GroupKFold on cv_process_group; both
# deterministic.  Thin re-exports keep every frozen call site untouched.
# --------------------------------------------------------------------------- #

from src.cv import (  # noqa: E402,F401
    _group_sets,
    check_gkf_contract,
    check_gss_contract,
    gkf_splits,
    gss_splits,
)


# --------------------------------------------------------------------------- #
# fold-internal PCA (no leakage) and cross-fold mode alignment
# --------------------------------------------------------------------------- #

def fit_fold_pca(X_train: np.ndarray, k: int) -> dict:
    """PCA fitted on training rows only; test rows are only ever projected."""
    mu = X_train.mean(axis=0, keepdims=True)
    comps, evr = l15.gram_pca(X_train - mu, k)
    return {"mu": mu, "comps": comps, "evr": evr}


def project_fold_pca(model: dict, X: np.ndarray) -> np.ndarray:
    return (X - model["mu"]) @ model["comps"].T


def pc_alignment_deg(comps_a: np.ndarray, comps_b: np.ndarray) -> tuple[float, float]:
    """(theta PC1, theta span PC1:k3) between two sets of components."""
    th1 = float(l15.principal_angles(comps_a[:1].T, comps_b[:1].T)[-1])
    k3 = min(3, comps_a.shape[0], comps_b.shape[0])
    th3 = float(l15.principal_angles(comps_a[:k3].T, comps_b[:k3].T)[-1])
    return th1, th3


# --------------------------------------------------------------------------- #
# Difference-of-Gaussians octave-like bands (band-definition sensitivity)
# --------------------------------------------------------------------------- #

def dog_band_stds(R3: np.ndarray, sigmas_px) -> dict[str, np.ndarray]:
    """std of DoG approximate bands; -3dB correspondence (sigma in px, 0.5 um/px):
    DoG_8_16 = G2-G4 (7.5-15.1 um), DoG_16_32 = G4-G8 (15.1-30.2),
    DoG_32_64 = G8-G16 (30.2-60.3), DoG_64_inf = G16 low-pass std (>=60.3).
    Note std(a-b) = std(b-a), so the differencing order only fixes naming."""
    require(len(sigmas_px) == 4, "DoG needs exactly 4 sigmas")
    G = [l15.gaussian_smooth(R3, float(s)) for s in sigmas_px]
    bands = {"DoG_8_16": G[0] - G[1], "DoG_16_32": G[1] - G[2],
             "DoG_32_64": G[2] - G[3], "DoG_64_inf": G[3]}
    return {name: arr.reshape(arr.shape[0], -1).std(axis=1)
            for name, arr in bands.items()}


def pairwise_gram_rmse(X: np.ndarray) -> np.ndarray:
    """Pairwise RMSE (um) matrix from the uncentred Gram; identical to the
    Phase 1.5-04 morphology distance definition."""
    return l15.pairwise_rmse_from_gram(X @ X.T, X.shape[1])
