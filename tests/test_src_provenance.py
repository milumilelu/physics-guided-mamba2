"""src.provenance parity tests: canonical module vs the frozen per-phase
_lib implementations it replaces (WP1 migration protocol, step 2)."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "phase1_5"))

import _lib as l15  # noqa: E402
from src import provenance as prov  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "phase2_lib_prov_parity", REPO / "experiments" / "phase2" / "_lib.py")
p2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p2)


class RunUtilParity(unittest.TestCase):
    def test_require_raises_with_frozen_message_format(self):
        with self.assertRaises(AssertionError) as ctx:
            prov.require(False, "boom")
        self.assertEqual(str(ctx.exception), "HARD ASSERTION FAILED: boom")
        with self.assertRaises(AssertionError) as ctx2:
            l15.require(False, "boom")
        self.assertEqual(str(ctx2.exception), str(ctx.exception))

    def test_require_passes(self):
        prov.require(True, "never")
        l15.require(True, "never")


class PowerParity(unittest.TestCase):
    def test_pulse_energy_matches_frozen_proxy_impl(self):
        f = np.array([2.0, 5.0, 20.0, 50.0])
        np.testing.assert_array_equal(
            prov.pulse_energy_uJ(5.3333, f),
            p2.pulse_energy_proxy_uJ(5.3333, f))

    def test_areal_dose_matches_frozen_proxy_impl(self):
        n, v, h = np.array([1.0, 2.0, 4.0]), np.array([9.0, 20.0, 50.0]), \
            np.array([4.0, 6.0, 10.0])
        np.testing.assert_array_equal(
            prov.areal_dose_J_per_mm2(5.3333, n, v, h),
            p2.areal_dose_proxy_j_mm2(5.3333, n, v, h))

    def test_scan_spacing_and_density_match(self):
        v, f = np.array([9.0, 20.0]), np.array([2.0, 5.0])
        np.testing.assert_array_equal(prov.scan_spacing_um(v, f),
                                      p2.scan_spacing_um(v, f))
        n, h = np.array([1.0, 4.0]), np.array([4.0, 10.0])
        np.testing.assert_array_equal(
            prov.areal_pulse_density(n, f, v, h),
            p2.areal_pulse_density(n, f, v, h))

    def test_registry_value_and_flagship_number(self):
        self.assertEqual(prov.POWER_REGISTRY["measured_power_W"], 5.3333)
        self.assertEqual(prov.POWER_REGISTRY["measurement_type"],
                         "post_objective_average_power")
        # f=2 kHz flagship derived value (existing docs quote 2666.65 µJ)
        self.assertAlmostEqual(float(prov.pulse_energy_uJ(5.3333, 2.0)),
                               2666.65, places=9)

    def test_canonical_columns_and_parity_on_manifest(self):
        man = pd.read_csv(REPO / "outputs" / "phase2" / "manifest"
                          / "phase2_manifest.csv")
        out = prov.canonical_power_columns(man)
        prov.assert_canonical_power_parity(out)
        self.assertIn("pulse_energy_uJ", out.columns)
        self.assertIn("areal_dose_J_per_mm2", out.columns)
        # canonical derivation reproduces the frozen manifest's legacy values
        np.testing.assert_allclose(out["pulse_energy_uJ"],
                                   man["pulse_energy_proxy_uJ"], rtol=0,
                                   atol=1e-9)
        np.testing.assert_allclose(out["areal_dose_J_per_mm2"],
                                   man["areal_dose_proxy_J_per_mm2"], rtol=0,
                                   atol=1e-9)

    def test_registry_divergence_fails_loudly(self):
        man = pd.read_csv(REPO / "outputs" / "phase2" / "manifest"
                          / "phase2_manifest.csv").head(4)
        man["measured_power_W"] = 9.9
        with self.assertRaises(AssertionError):
            prov.canonical_power_columns(man)


if __name__ == "__main__":
    unittest.main()
