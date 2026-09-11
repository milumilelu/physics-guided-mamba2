"""Anti-leakage and contract tests for depth_target_v1 (synthetic data, fast)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.depth_target import features as dt_features
from src.depth_target import models as dt_models


def synthetic_heights(n=4, shape=(64, 64), seed=7):
    rng = np.random.default_rng(seed)
    yy, xx = np.indices(shape)
    h = (3.0 * np.sin(xx / 9.0) + 2.0 * np.cos(yy / 7.0))[None] \
        + rng.normal(0.0, 0.5, (n,) + shape) \
        + rng.normal(-8.0, 3.0, (n, 1, 1))  # per-sample absolute offset
    return h, np.ones((n,) + shape, dtype=bool)


class TestTranslationInvariance(unittest.TestCase):
    def test_features_invariant_to_constant_shift(self):
        h, v = synthetic_heights()
        err = dt_features.translation_invariance_error(h, v, shift_um=5.2)
        self.assertLess(float(err.max()), 1e-8, f"features moved: {err.max():.3e}")

    def test_depth_moves_with_shift(self):
        h, v = synthetic_heights()
        med = np.median(h.reshape(len(h), -1), axis=1)
        moved = np.median((h + 5.2).reshape(len(h), -1), axis=1)
        np.testing.assert_allclose(-moved, -med - 5.2, rtol=0, atol=1e-10)


class TestBandEnergyParseval(unittest.TestCase):
    def test_band_energies_sum_to_non_dc_variance(self):
        h, v = synthetic_heights()
        r = dt_features.residual_median_centered(h, v)
        energies = dt_features.band_energies_um2(r)
        n_pix = r.shape[1] * r.shape[2]
        centered = r - r.mean(axis=(1, 2), keepdims=True)
        reference = (centered ** 2).sum(axis=(1, 2)) / n_pix
        np.testing.assert_allclose(energies.sum(axis=1), reference, rtol=1e-10, atol=1e-12)


class TestLeakageGuards(unittest.TestCase):
    def test_morphology_list_excludes_depth_and_absolute(self):
        forbidden = {"D", "median_depth_um", "depth", "height_mean", "height_min"}
        self.assertFalse(set(dt_features.ALL_MORPHOLOGY) & forbidden)

    def test_bum_contains_no_depth_column(self):
        import pandas as pd
        h, v = synthetic_heights()
        new = dt_features.compute_new_features(h, v)
        canonical = pd.DataFrame(0.0, index=range(len(h)),
                                 columns=dt_features.REUSED_CANONICAL)
        feats = pd.concat([canonical, new], axis=1)[
            dt_features.REUSED_CANONICAL + dt_features.NEW_FEATURES]
        frame = pd.DataFrame({
            "D": [1.0, 2.0, 3.0, 4.0],
            "areal_dose_proxy_J_per_mm2": [10.0, 20.0, 30.0, 40.0],
            "pulse_duration_fs": [500.0] * 4, "frequency_kHz": [2.0] * 4,
            "velocity_mm_s": [9.0] * 4, "hatch_spacing_um": [3.0] * 4,
            "pass_count": [4.0] * 4})
        x_bm = dt_models.design_matrix(frame, feats, "BM")
        self.assertEqual(x_bm.shape[1], len(dt_features.ALL_MORPHOLOGY))
        x_bum = dt_models.design_matrix(frame, feats, "BUM")
        self.assertEqual(x_bum.shape[1], 5 + len(dt_features.ALL_MORPHOLOGY))
        for j in range(x_bum.shape[1]):
            self.assertFalse(np.allclose(x_bum[:, j], frame.D.to_numpy()),
                             f"column {j} equals D")

    def test_b0_constant_is_train_weighted_mean(self):
        import pandas as pd
        frame = pd.DataFrame({"D": [1.0, 2.0, 9.0], "component_id": ["a", "a", "b"]})
        feats = pd.DataFrame(np.zeros((3, 1)))
        pred, null = dt_models.fit_depth_model(frame, frame, feats, feats, "B0", "Ridge", 1.0)
        self.assertTrue(np.allclose(pred, null))
        self.assertAlmostEqual(null, (0.5 * 1.5 + 0.5 * 9.0), places=12)


if __name__ == "__main__":
    unittest.main()
