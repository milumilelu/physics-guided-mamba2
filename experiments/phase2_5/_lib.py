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

# WP1 canonical migration (parity-tested, tests/test_src_statistics.py):
# sign-flip / Moran statistics now live in src/statistics.py; frozen names
# kept as thin re-exports (Phase 2.8 v2.1 section 4.2 migration step 5).
from src.statistics import (  # noqa: E402,F401
    exact_signflip_test,
    knn_row_standardized_graph,
    moran_i,
    moran_permutation_p,
    require_no_n4_to_5,
    sign_matrix,
)

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
