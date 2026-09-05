"""src.forward_models parity tests vs the frozen phase2_7 _lib originals,
plus unit tests for the new Phase 2.8 model-family functions."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src import forward_models as sfw  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "phase2_7_lib_fw_parity", REPO / "experiments" / "phase2_7" / "_lib.py")
p27 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p27)


def _profile(seed: int = 3):
    """Centered lateral profile (real usage: x_profile spans ±30 um)."""
    x = (np.arange(215) - 107) * 0.278657
    g = 6.0 * np.exp(-0.5 * (x / 3.5) ** 2)
    return g, x


class FrozenParity(unittest.TestCase):
    def test_hann_projection_identical(self):
        g, x = _profile()
        for k in (0.05, 0.125, 0.21):
            self.assertEqual(sfw.hann_projection(g, x, k),
                             p27.hann_projection(g, x, k))

    def test_cycles_level_identical(self):
        for lam in (5.0, 8.9, 10.0, 20.0):
            self.assertEqual(sfw.cycles_level(lam), p27.cycles_level(lam))

    def test_synth_field_identical(self):
        g, x = _profile()
        for h, phi, c in ((6.0, 0.5, 0.0), (4.0, 2.0, 0.3)):
            a = sfw.synth_field(g, x, h, phi, c, roi_um=40.0)
            b = p27.synth_field(g, x, h, phi, c, roi_um=40.0)
            np.testing.assert_array_equal(a, b)

    def test_field_class_identical(self):
        g, x = _profile()
        field = sfw.synth_field(g, x, 6.0, 0.5, 0.0, roi_um=80.0)
        a = sfw.field_class(field, h=6.0)
        b = p27.field_class(field, h=6.0)
        self.assertEqual(a, b)


class SaturationFamily(unittest.TestCase):
    def test_monotone_and_asymptote(self):
        s = np.linspace(0.0, 50.0, 200)
        f = sfw.saturate(s, 8.0)
        self.assertTrue(np.all(np.diff(f) >= 0.0))
        self.assertAlmostEqual(f[-1], 8.0, delta=0.02)
        self.assertEqual(f[0], 0.0)

    def test_rejects_nonpositive_dsat(self):
        with self.assertRaises(AssertionError):
            sfw.saturate(np.array([1.0]), 0.0)


class PairwiseInteraction(unittest.TestCase):
    def test_gamma_zero_matches_synth_field_exactly(self):
        g, x = _profile()
        a = sfw.pairwise_interaction_field(g, x, 6.0, 0.5, 0.0, roi_um=40.0)
        b = sfw.synth_field(g, x, 6.0, 0.5, 0.0, roi_um=40.0)
        np.testing.assert_array_equal(a, b)

    def test_gamma_units_change_field_in_um(self):
        g, x = _profile()
        z1 = sfw.pairwise_interaction_field(g, x, 6.0, 0.5, 0.01, roi_um=40.0)
        z0 = sfw.pairwise_interaction_field(g, x, 6.0, 0.5, 0.0, roi_um=40.0)
        # cross term scales linearly with gamma (um^-1) as registered
        z2 = sfw.pairwise_interaction_field(g, x, 6.0, 0.5, 0.02, roi_um=40.0)
        np.testing.assert_allclose(z2 - z0, 2.0 * (z1 - z0), rtol=1e-12)

    def test_physical_guard_flags_negative_depth(self):
        g, x = _profile()
        bad = sfw.pairwise_interaction_field(g, x, 4.0, 0.5, -5.0, roi_um=40.0)
        self.assertFalse(sfw.physical_validity_field(bad))
        good = sfw.synth_field(g, x, 4.0, 0.5, 0.0, roi_um=40.0)
        self.assertTrue(sfw.physical_validity_field(good))


class ArrayTransferAndOverlap(unittest.TestCase):
    def test_single_line_transfer_is_flat(self):
        k = np.linspace(0.01, 0.2, 50)
        np.testing.assert_allclose(sfw.array_transfer(k, 6.0, 1), 1.0)

    def test_two_line_transfer_matches_analytic(self):
        k = np.linspace(0.01, 0.2, 50)
        expect = 2.0 + 2.0 * np.cos(2.0 * np.pi * k * 6.0)
        np.testing.assert_allclose(sfw.array_transfer(k, 6.0, 2), expect)

    def test_overlap_descriptor_self_and_decay(self):
        g, x = _profile()
        self.assertAlmostEqual(sfw.overlap_descriptor(g, 0.278657, 1e-9),
                               1.0, places=10)
        far = sfw.overlap_descriptor(g, 0.278657, 30.0)
        self.assertLess(far, 0.2)


if __name__ == "__main__":
    unittest.main()
