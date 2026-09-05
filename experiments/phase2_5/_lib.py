"""Phase 2.5 shared library: five-part spectral composition, ILR geometry,
radial spectrum, directional FFT metrics, exact sign-flip enumeration, Moran I.

Loads the frozen Phase 2 library by explicit file location (module name
`phase2_lib_p25`; it in turn loads Phase 1.5 as `l15`). No implementation is
copied. Binding spec: Phase2.5_落地执行细则.md.

Conventions (细则 §0.1, rev2):
  - five-part composition uses NON-DC DCT coefficient energies:
      p_b = sum_{k in band, k != DC} C_k^2 / sum_{k != DC} C_k^2
  - `dc_offset_frac = mean(R)^2 / mean(R^2)`  (= C_DC^2 / sum_all C^2), kept
    OUT of the composition as a separate DC/mean-offset descriptor;
  - frozen fractions (E_b = mean(R_b^2)/mean(R^2), with DC inside the >=64 um
    band) reconcile as: E_b^frozen = (1-r_DC) p_b  (DC-free bands)
                      and   E_64^frozen = r_DC + (1-r_DC) p_64.
"""

from __future__ import annotations

import argparse
import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.fft import dctn

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_spec2 = importlib.util.spec_from_file_location(
    "phase2_lib_p25",
    Path(__file__).resolve().parents[1] / "phase2" / "_lib.py")
p2 = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(p2)
l15 = p2.l15

ILR_BANDS = ["lt8", "8_16", "16_32", "32_64", "64_inf"]


def log(message: str = "") -> None:
    print(message, flush=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(f"HARD ASSERTION FAILED: {message}")


def load_config(description: str) -> tuple[dict, bool]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(Path(__file__).with_name(
        "phase2_5_config.yaml")))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    with open(REPO / args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if args.quick:
        # quick outputs fully isolated from the formal root (细则 §0.9)
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
# five-part composition (non-DC coefficient energies) + DC offset descriptor
# --------------------------------------------------------------------------- #

def frozen_band_fractions(R: np.ndarray, bands_um: list,
                          pixel_um: float) -> tuple[dict, float]:
    """Replicates the Phase 1.5-05 convention exactly:
    E_b = mean(R_b^2) / mean(R^2)  (var_R there is the SECOND moment)."""
    fields, coverage = l15.dct_band_fields(R, pixel_um, bands_um)
    M2 = np.mean(R ** 2, axis=(1, 2))
    E = {name: np.mean(f ** 2, axis=(1, 2)) / M2 for name, f in fields.items()}
    return E, coverage


# WP1 canonical migration (parity-tested, tests/test_src_spectrum_composition.py):
# composition/ILR + radial/directional spectral primitives now live in
# src/composition.py and src/spectrum.py; the frozen names are thin
# re-exports (Phase 2.8 v2.1 section 4.2 migration step 5).
# frozen_band_fractions intentionally stays local: it replicates the
# Phase 1.5-05 second-moment convention through l15.dct_band_fields and has
# no Phase 2.8 consumer.
from src.composition import (  # noqa: E402,F401
    ILR_A,
    aitchison_distance,
    apply_zero_replacement,
    five_part_composition,
    ilr_inverse,
    ilr_matrix,
    ilr_transform,
)
from src.spectrum import (  # noqa: E402,F401
    directional_band_metrics,
    radial_spectrum,
    spectrum_descriptors,
)


# --------------------------------------------------------------------------- #
# exact sign-flip enumeration (Task 13)
# --------------------------------------------------------------------------- #

def sign_matrix(m: int) -> np.ndarray:
    """All 2^m sign configurations as an (2^m, m) float matrix (+1/-1)."""
    bits = np.arange(m)
    return 1.0 - 2.0 * ((np.arange(2 ** m)[:, None] >> bits) & 1)


def exact_signflip_test(dz: np.ndarray) -> dict:
    """Global mean-norm statistic with EXACT enumeration.

    p_exact = #{T_null >= T_obs} / 2^m — no Monte-Carlo +1 correction; the
    observed all-+1 configuration is part of the enumeration space, so
    p >= 1/2^m automatically (细则 §0.16). Also returns coordinate-wise
    two-sided exact p for each column mean.
    """
    dz = np.asarray(dz, dtype=float)
    m = dz.shape[0]
    require(m <= 20, f"exact enumeration with m={m} is infeasible")
    S = sign_matrix(m).astype(float)
    means = (S @ dz) / m
    T = np.linalg.norm(means, axis=1)
    T_obs = float(np.linalg.norm(dz.mean(axis=0)))
    p_global = float(np.count_nonzero(T >= T_obs)) / 2 ** m
    coord = []
    for j in range(dz.shape[1]):
        obs = abs(float(dz[:, j].mean()))
        p_j = float(np.count_nonzero(np.abs(means[:, j]) >= obs)) / 2 ** m
        coord.append({"coordinate": j, "mean_dz": float(dz[:, j].mean()),
                      "p_exact_two_sided": p_j})
    return {"T_obs": T_obs, "p_exact_global": p_global,
            "n_configurations": 2 ** m, "coordinates": coord}


def require_no_n4_to_5(steps) -> None:
    """N4->5 is session-confounded (v2 §10.2): analysing that STEP is
    forbidden. steps = iterable of (from_pass, to_pass) tuples."""
    for s_from, s_to in steps:
        require((int(s_from), int(s_to)) != (4, 5),
                "N4->5 step is session-confounded and must not be analysed")


# --------------------------------------------------------------------------- #
# Moran I with kNN binary graph (Task 14B)
# --------------------------------------------------------------------------- #

def knn_row_standardized_graph(X: np.ndarray, k: int) -> np.ndarray:
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    idx = np.argpartition(D, k - 1, axis=1)[:, :k]
    W = np.zeros((len(X), len(X)))
    rows = np.repeat(np.arange(len(X)), k)
    W[rows, idx.ravel()] = 1.0
    W = np.maximum(W, W.T)                      # symmetric kNN graph
    return W / np.maximum(W.sum(axis=1, keepdims=True), 1e-300)


def moran_i(z: np.ndarray, W: np.ndarray) -> float:
    zc = np.asarray(z, dtype=float) - np.mean(z)
    return float((zc @ W @ zc) / max(zc @ zc, 1e-300))


def moran_permutation_p(z: np.ndarray, W: np.ndarray, n_perm: int,
                        seed: int) -> tuple[float, float]:
    """Monte-Carlo permutation (NOT exact enumeration): keep the
    (1+b)/(1+n_perm) correction here, unlike Task 13's exact test."""
    rng = np.random.default_rng(seed)
    i_obs = moran_i(z, W)
    b = 0
    for _ in range(n_perm):
        if moran_i(rng.permutation(z), W) >= i_obs:
            b += 1
    return i_obs, (1 + b) / (1 + n_perm)


# --------------------------------------------------------------------------- #
# grouped-CV machinery re-exported from the frozen Phase 2 library
# --------------------------------------------------------------------------- #

gkf_splits = p2.gkf_splits
gss_splits = p2.gss_splits
check_gkf_contract = p2.check_gkf_contract
check_gss_contract = p2.check_gss_contract


def read_phase2_manifest(cfg: dict, require_loco: bool = True) -> pd.DataFrame:
    """Read the FORMAL Phase 2 manifest (paths.phase2_manifest). Quick mode
    never redirects this: the manifest is a deterministic frozen input."""
    path = REPO / cfg["paths"]["phase2_manifest"]
    require(path.exists(), "phase2_manifest.csv missing; run Phase 2 01 first")
    man = pd.read_csv(path)
    if require_loco:
        require(set(p2.LOCO_COLS) <= set(man.columns),
                "manifest lacks LOCO backfill; run Phase 2 01 first")
    return man
