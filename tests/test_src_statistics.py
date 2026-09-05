"""src.statistics parity tests vs the frozen p25/l15/p27 originals (WP1
migration protocol) on synthetic inputs."""

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
from src import statistics as sstat  # noqa: E402


def _load(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p25 = _load("phase2_5_lib_stat_parity", "experiments/phase2_5/_lib.py")
p27 = _load("phase2_7_lib_stat_parity", "experiments/phase2_7/_lib.py")


class SignflipParity(unittest.TestCase):
    def test_sign_matrix_identical(self):
        np.testing.assert_array_equal(sstat.sign_matrix(3), p25.sign_matrix(3))

    def test_exact_signflip_identical(self):
        rng = np.random.default_rng(1)
        dz = rng.normal(size=(4, 5))
        a = p25.exact_signflip_test(dz)
        b = sstat.exact_signflip_test(dz)
        self.assertEqual(a["T_obs"], b["T_obs"])
        self.assertEqual(a["p_exact_global"], b["p_exact_global"])
        self.assertEqual(a["n_configurations"], b["n_configurations"])
        self.assertEqual(a["coordinates"], b["coordinates"])

    def test_n4_to_5_guard(self):
        with self.assertRaises(AssertionError):
            sstat.require_no_n4_to_5([(1, 2), (4, 5)])
        sstat.require_no_n4_to_5([(3, 4)])


class MoranParity(unittest.TestCase):
    def test_graph_moran_and_permutation_identical(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(30, 2))
        z = rng.normal(size=30)
        Wa = p25.knn_row_standardized_graph(X, 5)
        Wb = sstat.knn_row_standardized_graph(X, 5)
        np.testing.assert_array_equal(Wa, Wb)
        self.assertEqual(p25.moran_i(z, Wb), sstat.moran_i(z, Wb))
        pa = p25.moran_permutation_p(z, Wb, 200, seed=7)
        pb = sstat.moran_permutation_p(z, Wb, 200, seed=7)
        self.assertEqual(pa, pb)


class BootstrapTrioParity(unittest.TestCase):
    def test_cluster_lists_and_bank_identical(self):
        man = pd.DataFrame({
            "shared_height_source_id": [1, 1, 2, 3, 3, 3],
            "dataset_index": [0, 1, 2, 3, 4, 5],
        })
        ca = l15.cluster_lists(man)
        cb = sstat.cluster_lists(man)
        self.assertEqual(len(ca), len(cb))
        for x, y in zip(ca, cb):
            np.testing.assert_array_equal(x, y)
        ba = l15.build_resample_bank(ca, 20, seed=9)
        bb = sstat.build_resample_bank(cb, 20, seed=9)
        for x, y in zip(ba, bb):
            np.testing.assert_array_equal(x, y)


class TvFamilyParity(unittest.TestCase):
    def test_tv_and_permutation_p_identical(self):
        q = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        r = np.array([0.3, 0.1, 0.2, 0.3, 0.1])
        self.assertEqual(sstat.tv(q, r), p27.tv(q, r))
        n_perm = 50
        q_obs = {4.0: q, 6.0: r}
        q_null = {h: [np.array(p27.q_distribution(
            np.random.default_rng(b).integers(0, 5, 20))) for b in range(n_perm)]
            for h in (4.0, 6.0)}
        weights = {4.0: 0.5, 6.0: 0.5}
        a = p27.tv_perm_p(q_obs, q_null, weights, n_perm=n_perm)
        b = sstat.tv_perm_p(q_obs, q_null, weights, n_perm=n_perm)
        self.assertEqual(a["t_obs"], b["t_obs"])
        self.assertEqual(a["p_value"], b["p_value"])

    def test_logistic_slope_identical(self):
        rng = np.random.default_rng(5)
        h = rng.choice([4.0, 6.0, 8.0], size=60)
        is_m2 = rng.random(60) < 0.4
        self.assertAlmostEqual(sstat.logistic_slope(h, is_m2),
                               p27.logistic_slope(h, is_m2), places=15)


if __name__ == "__main__":
    unittest.main()
