"""Canonical run utilities and power provenance for all phases.

`src/provenance.py` is the single source of truth for:

* run-level helpers shared by every phase `_lib` (``log`` / ``require``);
* derived process coordinates (scan spacing, areal pulse density);
* the power registry: P_obj = 5.3333 W is a **post-objective independently
  measured average power**, confirmed physically trusted, and is the
  **canonical physical input** from Phase 2.8 onward.

Provenance principle (external review §1, updated decision): *measurement is
physically trusted* is a different statement from *measurement provenance
metadata is perfectly complete*.  Instrument/date metadata is unavailable and
is registered as such -- incomplete metadata does not downgrade the
measurement to a proxy.

Canonical vs legacy derived columns:

* canonical: ``pulse_energy_uJ = 1000*P_obj/f``, ``areal_dose_J_per_mm2 =
  1000*P_obj*N/(v*h)`` -- used by Phase 2.8+ only;
* legacy: ``pulse_energy_proxy_uJ`` / ``areal_dose_proxy_J_per_mm2`` --
  numerically identical formulas, kept for the frozen Phase 2-2.7
  reproduction chain; the frozen ``phase2_manifest.csv`` is not rewritten.

Identification caveat (kept deliberately): P is constant in the DOE, so f and
E_p are fully coupled -- the design cannot separate a repetition-rate effect
from a pulse-energy effect.  Formal wording: *frequency / pulse-energy
coupled effect*.

Binding spec: 任务说明/Phase2.8_...md v2.1 FROZEN §1.1 +
experiments/phase2_8/Phase2.8_落地执行细则.md (FROZEN) §2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "log", "require",
    "POWER_REGISTRY",
    "scan_spacing_um", "areal_pulse_density",
    "pulse_energy_uJ", "areal_dose_J_per_mm2",
    "pulse_energy_proxy_uJ", "areal_dose_proxy_j_mm2",
    "canonical_power_columns", "assert_canonical_power_parity",
]


# --------------------------------------------------------------------------- #
# run utilities (byte-identical semantics to the per-phase _lib originals)
# --------------------------------------------------------------------------- #

def log(message: str = "") -> None:
    print(message, flush=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(f"HARD ASSERTION FAILED: {message}")


# --------------------------------------------------------------------------- #
# power registry (single source of truth; see module docstring)
# --------------------------------------------------------------------------- #

POWER_REGISTRY: dict = {
    "measured_power_W": 5.3333,
    "measurement_type": "post_objective_average_power",
    "trust": "user_confirmed_physically_trusted",
    "instrument": "instrument metadata unavailable",
    "measurement_date": "unavailable",
    "source_doc": "现有数据基础说明_v2.md §11",
    "registration": "Phase2.8 v2.1 §1.1 (external review §1 updated decision)",
    "constant_power_assumption": True,
    "identifiability_caveat": "P constant => f and E_p fully coupled; "
                              "report as frequency / pulse-energy coupled effect",
}


# --------------------------------------------------------------------------- #
# derived process coordinates (canonical implementations; parity-tested
# against the frozen phase2 _lib versions they replace)
# --------------------------------------------------------------------------- #

def scan_spacing_um(v_mm_s, f_khz) -> np.ndarray:
    return np.asarray(v_mm_s, dtype=float) / np.asarray(f_khz, dtype=float)


def areal_pulse_density(n_pass, f_khz, v_mm_s, h_um) -> np.ndarray:
    return 1e6 * np.asarray(n_pass, dtype=float) * np.asarray(f_khz, dtype=float) \
        / (np.asarray(v_mm_s, dtype=float) * np.asarray(h_um, dtype=float))


def pulse_energy_uJ(power_w: float, f_khz) -> np.ndarray:
    """Canonical E_p = P_obj / f  [µJ]  (1000 * W / kHz)."""
    return 1000.0 * float(power_w) / np.asarray(f_khz, dtype=float)


def areal_dose_J_per_mm2(power_w: float, n_pass, v_mm_s, h_um) -> np.ndarray:
    """Canonical nominal areal energy dose  [J/mm²]."""
    return 1000.0 * float(power_w) * np.asarray(n_pass, dtype=float) \
        / (np.asarray(v_mm_s, dtype=float) * np.asarray(h_um, dtype=float))


# frozen Phase 2-2.7 legacy aliases (numerically identical formulas; names
# preserved so the frozen manifest builders and their tests keep working)
pulse_energy_proxy_uJ = pulse_energy_uJ
areal_dose_proxy_j_mm2 = areal_dose_J_per_mm2


def canonical_power_columns(man: pd.DataFrame) -> pd.DataFrame:
    """Add canonical ``pulse_energy_uJ`` / ``areal_dose_J_per_mm2`` columns
    derived from ``measured_power_W`` (Phase 2.8+ inputs).  Returns a copy."""
    man = man.copy()
    pw = float(POWER_REGISTRY["measured_power_W"])
    if "measured_power_W" in man.columns:
        pw_col = pd.to_numeric(man["measured_power_W"])
        require(bool(((pw_col - pw).abs() < 1e-9).all()),
                "manifest measured_power_W diverges from POWER_REGISTRY")
    man["pulse_energy_uJ"] = pulse_energy_uJ(pw, man["frequency_kHz"])
    man["areal_dose_J_per_mm2"] = areal_dose_J_per_mm2(
        pw, man["pass_count"], man["velocity_mm_s"], man["hatch_spacing_um"])
    return man


def assert_canonical_power_parity(man: pd.DataFrame) -> None:
    """Canonical columns must equal the legacy ``_proxy`` columns elementwise
    (the two derivations share one formula; divergence means a registry or
    manifest bug).  No-op on manifests without the legacy columns."""
    if "pulse_energy_proxy_uJ" in man.columns:
        d = (man["pulse_energy_uJ"]
             - pd.to_numeric(man["pulse_energy_proxy_uJ"])).abs().max()
        require(float(d) < 1e-9, f"pulse_energy canonical/proxy parity {d}")
    if "areal_dose_proxy_J_per_mm2" in man.columns:
        d = (man["areal_dose_J_per_mm2"]
             - pd.to_numeric(man["areal_dose_proxy_J_per_mm2"])).abs().max()
        require(float(d) < 1e-9, f"areal_dose canonical/proxy parity {d}")
