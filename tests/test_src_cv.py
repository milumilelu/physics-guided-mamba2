"""src.cv parity tests: canonical CV contracts + frozen v1 alpha protocol vs
the frozen phase2/phase2_6 _lib originals (WP1 migration protocol)."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src import cv as scv  # noqa: E402


def _load(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p2 = _load("phase2_lib_cv_parity", "experiments/phase2/_lib.py")
p26 = _load("phase2_6_lib_cv_parity", "experiments/phase2_6/_lib.py")


def _synth(n_groups: int = 10, per_group: int = 6, d: int = 3, seed: int = 7):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), per_group)
    X = rng.normal(size=(n_groups * per_group, d))
    beta = np.array([1.0, -2.0, 0.5])
    y = X @ beta + rng.normal(scale=0.1, size=len(groups)) \
        + groups[:, None] * 0.05
    return X, y.ravel(), groups


class SplitsParity(unittest.TestCase):
    def test_gkf_splits_identical_to_frozen(self):
        _, _, g = _synth()
        a = p2.gkf_splits(g, 5)
        b = scv.gkf_splits(g, 5)
        self.assertEqual(len(a), len(b))
        for (tr_a, te_a), (tr_b, te_b) in zip(a, b):
            np.testing.assert_array_equal(tr_a, tr_b)
            np.testing.assert_array_equal(te_a, te_b)

    def test_gss_splits_identical_to_frozen(self):
        _, _, g = _synth()
        a = p2.gss_splits(g, 3, 0.25, 123)
        b = scv.gss_splits(g, 3, 0.25, 123)
        for (tr_a, te_a), (tr_b, te_b) in zip(a, b):
            np.testing.assert_array_equal(tr_a, tr_b)
            np.testing.assert_array_equal(te_a, te_b)

    def test_check_gkf_contract_accepts_and_rejects_same(self):
        _, _, g = _synth()
        splits = scv.gkf_splits(g, 5)
        p2.check_gkf_contract(g, splits)
        scv.check_gkf_contract(g, splits)
        with self.assertRaises(AssertionError):
            p2.check_gkf_contract(g, [(np.arange(58), np.arange(58, 60))] * 2)
        with self.assertRaises(AssertionError):
            scv.check_gkf_contract(g, [(np.arange(58), np.arange(58, 60))] * 2)


class AlphaParity(unittest.TestCase):
    def test_v1_alpha_identical_to_frozen_phase26(self):
        X, y, g = _synth(n_groups=12, per_group=5)
        a_frozen = p26.ridge_alpha_inner_gkf(X, y, g, n_splits=5)
        a_canon = scv.ridge_alpha_inner_gkf_v1(X, y, g, n_splits=5)
        self.assertEqual(a_frozen, a_canon)

    def test_scalar_mse_scorer_matches_v1_protocol(self):
        X, y, g = _synth(n_groups=12, per_group=5)
        a_v1 = scv.ridge_alpha_inner_gkf_v1(X, y, g, n_splits=5)
        a_new, scores = scv.select_alpha_inner(X, y, g, scorer="mse",
                                               n_splits=5, return_scores=True)
        self.assertEqual(a_v1, a_new)
        self.assertEqual(len(scores), 13)

    def test_too_few_groups_fails_same_as_frozen(self):
        X, y, g = _synth(n_groups=2, per_group=3)  # too few groups for GKF(5)
        with self.assertRaises(ValueError):
            p26.ridge_alpha_inner_gkf(X, y, g, n_splits=5)
        with self.assertRaises(ValueError):
            scv.ridge_alpha_inner_gkf_v1(X, y, g, n_splits=5)
        with self.assertRaises(ValueError):
            scv.select_alpha_inner(X, y, g, scorer="mse", n_splits=5)

    def test_unknown_scorer_rejected(self):
        X, y, g = _synth()
        with self.assertRaises(ValueError):
            scv.select_alpha_inner(X, y, g, scorer="r2")

    def test_multi_mse_and_aitchison_return_grid_members(self):
        X, _, g = _synth(n_groups=12, per_group=5)
        rng = np.random.default_rng(3)
        y2 = rng.normal(size=(len(g), 2))
        grid = np.logspace(-3, 3, 13)
        a_m, _ = scv.select_alpha_inner(X, y2, g, scorer="multi_mse",
                                        grid=grid, return_scores=True)
        self.assertIn(a_m, list(grid))
        z = rng.normal(size=(len(g), 4)) * 0.1  # ILR-coordinate-like target
        a_z, _ = scv.select_alpha_inner(X, z, g, scorer="aitchison_ilr_q2",
                                        grid=grid, return_scores=True)
        self.assertIn(a_z, list(grid))

    def test_q2_aitchison_v1_matches_frozen_definition(self):
        p27 = _load("phase2_7_lib_cv_parity",
                    "experiments/phase2_7/_lib.py")
        rng = np.random.default_rng(11)
        z_tr, z_te = rng.normal(size=(20, 4)), rng.normal(size=(8, 4))
        pred = z_te + rng.normal(scale=0.05, size=z_te.shape)
        self.assertAlmostEqual(
            p27.q2_aitchison_ilr(z_te, pred, z_tr),
            scv.q2_aitchison_ilr_v1(z_te, pred, z_tr), places=15)


if __name__ == "__main__":
    unittest.main()
