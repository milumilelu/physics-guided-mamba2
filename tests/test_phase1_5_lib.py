"""Phase 1.5 library unit tests (synthetic data only; CI-safe)."""

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "phase1_5"))

import _lib  # noqa: E402


def _random_data(n=40, f=600, rank=8, seed=3):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, f)) @ rng.normal(size=(f, rank))


class Lambda3dBTests(unittest.TestCase):
    def test_analytic_matches_numeric(self):
        for sigma in (2.0, 4.0, 7.3, 16.0):
            ana = _lib.lambda_3db_px(sigma)
            num = _lib.numeric_lambda_3db_px(sigma)
            self.assertLess(abs(ana - num) / ana, 0.02)

    def test_value(self):
        # 2 pi sigma / sqrt(ln 2) ~= 7.546 sigma
        self.assertAlmostEqual(_lib.lambda_3db_px(1.0), 2 * np.pi / np.sqrt(np.log(2)),
                               places=10)
        self.assertAlmostEqual(_lib.lambda_3db_px(4.0) / _lib.lambda_3db_px(1.0),
                               4.0, places=12)


class DctBandTests(unittest.TestCase):
    def test_exact_mode_lands_in_band(self):
        # lambda_k = 2 N d / k; N=160, d=0.5 -> lambda = 160/k um.
        # Field varies along y only (kx=0, ky=8) -> lambda = 20 um.
        n = 160
        pixel_um = 0.5
        yy = np.arange(n)
        line = np.cos(np.pi * 8.0 * (2 * yy + 1) / (2 * n))
        field = np.tile(line[None, :], (n, 1))
        R3 = (field - field.mean())[None, :, :]
        bands, cov = _lib.dct_band_fields(R3, pixel_um,
                                          [[16, 32], [32, 64], [64, 1e9]])
        var_R = float(np.var(R3[0]))
        frac = {k: float(np.var(v[0]) / var_R) for k, v in bands.items()}
        self.assertGreater(frac["DCT_16_32"], 0.99)
        self.assertLess(frac["DCT_32_64"], 1e-8)
        self.assertLess(frac["DCT_64_inf"], 1e-8)

    def test_roundtrip_identity(self):
        rng = np.random.default_rng(0)
        R3 = rng.normal(size=(2, 32, 32))
        from scipy.fft import dctn, idctn
        rec = idctn(dctn(R3, norm="ortho"), norm="ortho")
        self.assertTrue(np.allclose(rec, R3, atol=1e-12))

    def test_lambda_grid_dc_infinite(self):
        lam = _lib.dct_lambda_grid((8, 8), 0.5)
        self.assertEqual(lam[0, 0], np.inf)
        self.assertTrue(np.isfinite(lam[1, 1]))


class GramPcaTests(unittest.TestCase):
    def test_matches_evr_and_orthonormal(self):
        from sklearn.decomposition import PCA
        X = _random_data()
        comps, evr = _lib.gram_pca(X, 6)
        self.assertTrue(np.allclose(comps @ comps.T, np.eye(6), atol=1e-8))
        sk = PCA(n_components=6, svd_solver="full").fit(X)
        self.assertTrue(np.allclose(np.sort(sk.explained_variance_ratio_)[::-1],
                                    evr, atol=1e-10))
        ang = _lib.principal_angles(comps.T, sk.components_.T)
        self.assertLess(ang[-1], 1e-4)  # same subspace -> max angle ~ 0

    def test_evr_sums_below_one(self):
        X = _random_data()
        _, evr = _lib.gram_pca(X, 5)
        self.assertLessEqual(float(evr.sum()), 1.0 + 1e-10)


class PrincipalAngleTests(unittest.TestCase):
    def test_identical_and_orthogonal(self):
        rng = np.random.default_rng(1)
        Q, _ = np.linalg.qr(rng.normal(size=(50, 50)))
        A, B = Q[:, :3], Q[:, -3:]
        self.assertLess(_lib.principal_angles(A, A)[-1], 1e-5)
        self.assertGreater(_lib.principal_angles(A, B)[-1], 90.0 - 1e-6)


class BootstrapBankTests(unittest.TestCase):
    def test_bank_reproducible(self):
        clusters = [np.array([2 * i, 2 * i + 1]) for i in range(10)]
        b1 = _lib.build_resample_bank(clusters, 5, seed=42)
        b2 = _lib.build_resample_bank(clusters, 5, seed=42)
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(b1, b2)))

    def test_boot_matches_direct(self):
        X = _random_data(n=30, f=300)
        clusters = [np.array([i]) for i in range(30)]
        ref, _ = _lib.gram_pca(X, 3)
        G = X @ X.T
        bank = _lib.build_resample_bank(clusters, 4, seed=7)
        angles, evr = _lib.boot_angles_bank(G, X, bank, ref, 3)
        self.assertTrue(np.all((angles >= 0) & (angles <= 90)))
        for b, idx in enumerate(bank):
            comps, evr_dir = _lib.gram_pca(X[idx], 3)
            ang = _lib.principal_angles(ref.T, comps.T)[-1]
            self.assertAlmostEqual(ang, angles[b, -1], places=7)
            self.assertTrue(np.allclose(evr_dir[:3], evr[b], atol=1e-10))

    def test_quantiles_monotone(self):
        rng = np.random.default_rng(2)
        angles = np.abs(rng.normal(size=(100, 3))) * 30
        q = _lib.angle_quantiles(angles)
        for a, b in (("q25", "q50"), ("q50", "q75"), ("q75", "q90"),
                     ("q90", "q95")):
            self.assertTrue(np.all(q[a] <= q[b] + 1e-12))


class MatchedSubsetTests(unittest.TestCase):
    def _man(self):
        import pandas as pd
        rows = []
        for session, n_cl, size in (("sA", 20, 1), ("sB", 10, 2)):
            r = 0
            for c in range(n_cl):
                for _ in range(size):
                    rows.append((session, c, r))
                    r += 1
        return pd.DataFrame(rows, columns=["session_id", "cid", "dataset_index"])

    def test_signature_and_draw(self):
        import pandas as pd
        man = self._man()
        man["shared_height_source_id"] = man["session_id"] + ":c" + \
            man["cid"].astype(str)
        sub = man[man.session_id == "sA"]  # 20 singletons from sA
        self.assertEqual(_lib.occupancy_signature(sub), (1,) * 20)
        sub2 = man[man.session_id == "sB"]
        self.assertEqual(_lib.occupancy_signature(sub2), (2,) * 10)
        pool = _lib.session_cluster_pools(man)["sA"]
        rng = np.random.default_rng(0)
        idx = _lib.draw_matched_subset(pool, (1,) * 12, rng)
        self.assertEqual(len(idx), 12)
        self.assertEqual(len(set(idx)), 12)
        drawn_sig = tuple(sorted(pd.DataFrame({
            "dataset_index": idx}).merge(man[man.session_id == "sA"],
                                         on="dataset_index")
            .groupby("shared_height_source_id").size().to_list(),
            reverse=True))
        self.assertEqual(drawn_sig, (1,) * 12)


class LocoTests(unittest.TestCase):
    def test_outlier_cluster_has_largest_influence(self):
        rng = np.random.default_rng(5)
        base = rng.normal(size=(30, 400))
        outlier = rng.normal(size=(1, 400)) + 50.0
        X = np.vstack([base, outlier])
        clusters = [np.array([i]) for i in range(31)]
        loco = _lib.loco_angles(X, clusters, k=1)
        self.assertEqual(int(np.argmax(loco)), 30)
        self.assertGreater(loco[30], 20.0)
        self.assertLess(np.median(loco[:30]), 15.0)


class PairwiseRmseTests(unittest.TestCase):
    def test_matches_direct(self):
        rng = np.random.default_rng(6)
        X = rng.normal(size=(7, 23))
        G = X @ X.T
        D = _lib.pairwise_rmse_from_gram(G, X.shape[1])
        for i in range(7):
            for j in range(i + 1, 7):
                direct = np.sqrt(np.mean((X[i] - X[j]) ** 2))
                self.assertAlmostEqual(D[i, j], direct, places=10)


class OrdinaryPairMaskTests(unittest.TestCase):
    def test_excludes_shared_source_and_sentinel(self):
        import pandas as pd
        man = pd.DataFrame({
            "shared_height_source_id": ["a", "a", "b", "c", "d"],
        })
        pu, pv = _lib.ordinary_pair_mask(man, 3, 4)
        pairs = set(zip(pu.tolist(), pv.tolist()))
        self.assertNotIn((0, 1), pairs)   # same shared source
        self.assertNotIn((3, 4), pairs)   # sentinel pair
        self.assertEqual(len(pairs), 8)   # 10 - 1 shared - 1 sentinel


class MultiscaleFieldsTests(unittest.TestCase):
    def test_keys_and_shapes(self):
        cfg = {"scales": {"pixel_um": 0.5, "sigmas_px": [2, 4],
                          "dct_bands_um": [[8, 16], [16, 1e9]]}}
        rng = np.random.default_rng(7)
        R3 = rng.normal(size=(3, 32, 32))
        fields = _lib.multiscale_fields(R3, cfg)
        for name, X in fields.items():
            self.assertEqual(X.shape, (3, 32 * 32))
        for key in ("total", "G2", "G4", "DCT_8_16", "DCT_16_inf"):
            self.assertIn(key, fields)


if __name__ == "__main__":
    unittest.main()
