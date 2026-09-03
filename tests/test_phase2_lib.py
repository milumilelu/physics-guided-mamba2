"""Phase 2 library unit tests (synthetic data + frozen repo inputs; CI-safe)."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "phase1_5"))

import _lib as l15  # noqa: E402  (Phase 1.5 library, used for cross-checks)

_spec = importlib.util.spec_from_file_location(
    "phase2_lib", REPO / "experiments" / "phase2" / "_lib.py")
p2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p2)


def _cfg():
    import yaml
    with open(REPO / "experiments" / "phase2" / "phase2_config.yaml",
              encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class DerivedCoordTests(unittest.TestCase):
    def test_pulse_energy_units(self):
        self.assertAlmostEqual(float(p2.pulse_energy_proxy_uJ(5.3333, 2)),
                               2666.65, places=6)
        self.assertAlmostEqual(float(p2.pulse_energy_proxy_uJ(5.3333, 200)),
                               26.6665, places=6)

    def test_scan_spacing_units(self):
        self.assertAlmostEqual(float(p2.scan_spacing_um(11, 2)), 5.5, places=12)
        self.assertAlmostEqual(float(p2.scan_spacing_um(20, 40)), 0.5, places=12)

    def test_areal_density_and_dose(self):
        n_a = float(p2.areal_pulse_density(2, 2, 9, 6))
        self.assertAlmostEqual(n_a, 1e6 * 2 * 2 / 54, places=3)
        d_e = float(p2.areal_dose_proxy_j_mm2(5.3333, 2, 9, 6))
        self.assertAlmostEqual(d_e, 1000 * 5.3333 * 2 / 54, places=3)


class ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.man = p2.build_manifest(_cfg())

    def test_power_provenance_required(self):
        m = self.man
        self.assertTrue((m["measured_power_W"] == 5.3333).all())
        self.assertTrue(m["power_measurement_source"].str.len().gt(0).all())
        self.assertEqual(m["power_measurement_version"].iloc[0],
                         "PENDING_REGISTRATION")
        # proxy naming must stay until power registration (细则 §0.1)
        for col in ("pulse_energy_proxy_uJ", "areal_dose_proxy_J_per_mm2"):
            self.assertIn(col, m.columns)

    def test_base_condition_group_merge(self):
        m = self.man
        self.assertEqual(m["base_condition_group"].nunique(), 135)
        self.assertEqual(m["cv_process_group"].nunique(), 134)
        self.assertEqual(int(m["design_group"].notna().sum()), 80)
        sup = m[m.session_role == "pass_supplement"]
        self.assertTrue(sup["base_condition_group"].str.startswith("T").all())
        # 10 supplement groups extend pass_main trajectories to N=5..6
        self.assertEqual(sup["base_condition_group"].nunique(), 10)

    def test_cv_process_groups_sentinel(self):
        m = self.man
        sent = m[(m.session_id == "zro2_120_formal")
                 & (m.processing_order.isin([49, 50]))]
        self.assertEqual(len(sent), 2)
        # exact-repeat pair must share one CV-B group: no train/test split
        self.assertEqual(sent["cv_process_group"].nunique(), 1)
        self.assertEqual(sent["quad_key"].nunique(), 1)


class SplitContractTests(unittest.TestCase):
    GROUPS = np.array(["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"])

    def test_gkf_contract(self):
        sp = p2.gkf_splits(self.GROUPS, 5)
        p2.check_gkf_contract(self.GROUPS, sp)
        for _, te in sp:
            self.assertEqual(len(te), 2)  # one group per fold

    def test_gss_contract_allows_test_repeats(self):
        sp = p2.gss_splits(self.GROUPS, 5, test_size=0.4, seed=7)
        counts = p2.check_gss_contract(self.GROUPS, sp)
        n_test_groups = sum(len(set(self.GROUPS[te])) for _, te in sp)
        self.assertEqual(int(counts.sum()), n_test_groups)
        # GroupShuffleSplit test sets are allowed to repeat across splits and
        # need not cover all groups (the v1 assertion was wrong, 细则 §0.7)
        union = set().union(*(set(self.GROUPS[te]) for _, te in sp))
        self.assertTrue(union.issubset(set(self.GROUPS)))

    def test_gkf_contract_rejects_overlapping_test_sets(self):
        idx = np.arange(10)
        same_test = [(idx[2:], idx[:2]), (idx[2:], idx[:2])]
        with self.assertRaises(AssertionError):
            p2.check_gkf_contract(self.GROUPS, same_test)
        # the identical splits are a valid GSS draw (train/test disjoint only)
        p2.check_gss_contract(self.GROUPS, same_test)


class FoldPcaTests(unittest.TestCase):
    def test_fold_internal_pca_no_leak(self):
        rng = np.random.default_rng(0)
        A = rng.normal(size=(40, 6))
        Xtr = rng.normal(size=(60, 40)) @ A
        Xte = rng.normal(size=(20, 40)) @ A
        model = p2.fit_fold_pca(Xtr, 3)
        comps_dir, _ = l15.gram_pca(Xtr, 3)
        self.assertTrue(np.allclose(model["comps"], comps_dir, atol=1e-10))
        y_te = p2.project_fold_pca(model, Xte)
        y_dir = (Xte - Xtr.mean(0, keepdims=True)) @ comps_dir.T
        self.assertTrue(np.allclose(y_te, y_dir, atol=1e-10))
        self.assertEqual(model["comps"].shape, (3, 6))

    def test_pc_alignment_identical_and_orthogonal(self):
        rng = np.random.default_rng(1)
        Q, _ = np.linalg.qr(rng.normal(size=(50, 50)))
        A, B = Q[:, :3], Q[:, -3:]
        t1, t3 = p2.pc_alignment_deg(A.T, A.T)
        self.assertLess(t1, 1e-4)
        self.assertLess(t3, 1e-4)
        _, t3b = p2.pc_alignment_deg(A.T, B.T)
        self.assertGreater(t3b, 90.0 - 1e-6)


class BandTests(unittest.TestCase):
    def test_band_sum_is_masked_reconstruction(self):
        from scipy.fft import dctn, idctn
        rng = np.random.default_rng(2)
        R3 = rng.normal(size=(2, 32, 32))
        bands, cov = l15.dct_band_fields(R3, 0.5,
                                         [[8, 16], [16, 32], [32, 64],
                                          [64, 1e9]])
        rec = sum(bands.values())
        lam = l15.dct_lambda_grid((32, 32), 0.5)
        masked = idctn(dctn(R3[0], norm="ortho") * (lam >= 8), norm="ortho")
        self.assertTrue(np.allclose(rec[0], masked, atol=1e-10))
        self.assertLess(cov, 1.0)

    @staticmethod
    def _single_wavelength_stds(k_y):
        n = 160
        yy = np.arange(n)
        line = np.cos(np.pi * k_y * (2 * yy + 1) / (2 * n))
        R3 = np.tile(line[None, :], (n, 1))[None, :, :]
        return p2.dog_band_stds(R3, [2, 4, 8, 16])

    def test_dog_band_localization(self):
        # lambda = 160/k um: k=16 -> 10 um (fine), k=1 -> 160 um (coarse)
        fine = self._single_wavelength_stds(16)
        self.assertGreater(fine["DoG_8_16"] + fine["DoG_16_32"],
                           5.0 * (fine["DoG_32_64"] + fine["DoG_64_inf"]))
        coarse = self._single_wavelength_stds(1)
        self.assertGreater(coarse["DoG_64_inf"],
                           5.0 * (coarse["DoG_8_16"] + coarse["DoG_16_32"]
                                  + coarse["DoG_32_64"]))


class KnnTests(unittest.TestCase):
    def test_knn_self_exclusion(self):
        Z = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 0.0], [9.0, 0.0]])
        d = p2.knn_median_distance(Z, 2)
        self.assertAlmostEqual(float(d[0]), float(np.median([0.1, 5.0])),
                               places=12)
        self.assertAlmostEqual(float(d[3]), float(np.median([4.0, 8.9])),
                               places=12)


class LocoTests2(unittest.TestCase):
    def test_outlier_cluster_has_largest_influence(self):
        rng = np.random.default_rng(5)
        base = rng.normal(size=(30, 400))
        outlier = rng.normal(size=(1, 400)) + 50.0
        X = np.vstack([base, outlier])
        clusters = [np.array([i]) for i in range(31)]
        loco = p2.l15.loco_angles(X, clusters, k=1)
        self.assertEqual(int(np.argmax(loco)), 30)
        self.assertGreater(loco[30], 20.0)
        self.assertLess(np.median(loco[:30]), 15.0)


class StatHelperTests(unittest.TestCase):
    def test_sentinel_normalization_value(self):
        out = p2.sentinel_normalize(np.array([3.0, 6.0]), 1.5)
        self.assertTrue(np.allclose(out, [2.0, 4.0]))

    def test_consensus_uses_min_band_rank(self):
        cons, spectral = p2.consensus_rank(50, 3, 7, 20, [10, 200, 150, 190])
        self.assertEqual(spectral, 10.0)
        self.assertEqual(cons, float(np.median([50, 3, 7, 20, 10])))

    def test_process_near_morph_level(self):
        d_proc = np.arange(0.1, 1.01, 0.1)
        d_morph = np.arange(10.0)
        t, thr = p2.process_near_morph_level(d_proc, d_morph, 0.10)
        self.assertAlmostEqual(thr, 0.19, places=12)
        self.assertAlmostEqual(t, 0.0, places=12)


class TargetsAlignTests(unittest.TestCase):
    def test_targets_align_manifest(self):
        import pandas as pd
        path = REPO / "outputs/phase2/multiscale_targets/multiscale_targets.csv"
        if not path.exists():
            self.skipTest("04 output not present (run 04 first)")
        tgt = pd.read_csv(path)
        man = p2.build_manifest(_cfg())
        self.assertEqual(len(tgt), 200)
        self.assertTrue((tgt["dataset_index"].to_numpy()
                         == np.arange(200)).all())
        self.assertTrue((tgt["median_depth_um"].to_numpy()
                         == man["median_depth_um"].to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
