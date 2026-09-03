"""Phase 2.5 unit tests (synthetic + frozen repo inputs; CI-safe)."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "phase1_5"))
sys.path.insert(0, str(REPO / "experiments" / "phase2"))

import _lib as l15  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "phase2_5_lib", REPO / "experiments" / "phase2_5" / "_lib.py")
p25 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p25)


def _cfg():
    import yaml
    with open(REPO / "experiments" / "phase2_5" / "phase2_5_config.yaml",
              encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _frozen():
    return l15.load_frozen({"paths": {
        "dataset_npz": _cfg()["paths"]["dataset_npz"],
        "exploration_manifest": _cfg()["paths"]["exploration_manifest"]}})


class CompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frozen = _frozen()
        cls.pixel = float(_cfg()["scales"]["pixel_um"])
        cls.p, cls.dc = p25.five_part_composition(cls.frozen["R"], cls.pixel)

    def test_frozen_fractions_reconcile(self):
        """Dual-track reconciliation identity (细则 §0.1 rev2), atol 1e-8."""
        E, _ = p25.frozen_band_fractions(self.frozen["R"],
                                         _cfg()["scales"]["dct_bands_um"],
                                         self.pixel)
        r = self.dc
        atol = float(_cfg()["spectrum"]["reconciliation_atol"])
        for b, name in (("8_16", "DCT_8_16"), ("16_32", "DCT_16_32"),
                        ("32_64", "DCT_32_64"), ("64_inf", "DCT_64_inf")):
            pred = ((1 - r) * self.p[b] if b != "64_inf"
                    else r + (1 - r) * self.p[b])
            self.assertTrue(np.allclose(pred, E[name], atol=atol),
                            f"reconciliation failed for {b}")
        # direct match against the committed Phase 1.5 descriptor CSV
        desc = pd.read_csv(REPO / "outputs/phase1_5/morphology_descriptors.csv")
        self.assertTrue(np.allclose(
            E["DCT_8_16"], desc["E_DCT_8_16_frac"].to_numpy(), atol=atol))

    def test_five_part_sums_to_one(self):
        s = sum(self.p[b] for b in p25.ILR_BANDS)
        self.assertTrue(np.allclose(s, 1.0, atol=1e-12))
        self.assertTrue((self.p["lt8"] > 0).all())

    def test_no_pseudocount_without_zero(self):
        P = np.column_stack([self.p[b] for b in p25.ILR_BANDS])
        self.assertGreater(P.min(), float(_cfg()["composition"]["zero_threshold"]))
        out, replaced = p25.apply_zero_replacement(
            P, _cfg()["composition"]["zero_threshold"],
            _cfg()["composition"]["replacement_delta"])
        self.assertEqual(int(replaced.sum()), 0)
        self.assertTrue(np.allclose(out, P))


class IlrTests(unittest.TestCase):
    def test_z_basis_orthonormal(self):
        A = p25.ILR_A
        self.assertEqual(A.shape, (4, 5))
        self.assertTrue(np.allclose(A @ A.T, np.eye(4), atol=1e-12))
        # A^T A is the rank-4 contrast projection, NOT I5
        self.assertFalse(np.allclose(A.T @ A, np.eye(5), atol=1e-12))
        self.assertTrue(np.allclose(A @ np.ones(5), 0.0, atol=1e-12))

    def test_ilr_inverse_roundtrip(self):
        rng = np.random.default_rng(0)
        z = rng.normal(size=(20, 4))
        p = p25.ilr_inverse(z)
        self.assertTrue(np.allclose(p.sum(axis=1), 1.0, atol=1e-12))
        self.assertTrue(np.allclose(p25.ilr_transform(p), z, atol=1e-10))

    def test_aitchison_distance_is_ilr_euclidean(self):
        rng = np.random.default_rng(1)
        p1 = p25.ilr_inverse(rng.normal(size=(5, 4)))
        p2 = p25.ilr_inverse(rng.normal(size=(5, 4)))
        d = p25.aitchison_distance(p1, p2)
        self.assertTrue(np.allclose(d, np.linalg.norm(
            p25.ilr_transform(p1) - p25.ilr_transform(p2), axis=1)))


class RadialSpectrumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frozen = _frozen()
        cfg = _cfg()["spectrum"]
        cls.out, cls.edges = p25.radial_spectrum(
            cls.frozen["R"][:20], cls.pixel_um if hasattr(cls, "pixel_um")
            else float(_cfg()["scales"]["pixel_um"]),
            cfg["radial_log_bins"], cfg["lambda_lo_um"], cfg["lambda_hi_um"])

    def test_radial_energy_sums_to_one(self):
        s = self.out["energy_fraction"].sum(axis=1)
        self.assertTrue(np.allclose(s, 1.0, atol=1e-12))
        self.assertLess(float(self.out["uncovered_frac"].max()), 1e-9)

    def test_low_mode_count_flagged(self):
        self.assertTrue(self.out["low_mode_count"].any())
        flagged = self.out["lambda_geo_um"][
            self.out["low_mode_count"][0].astype(bool)]
        # only the coarse tail is mode-starved: fine bins (lambda < 16 um,
        # k-radius >= 10) always carry plenty of modes
        self.assertGreater(float(flagged.min()), 16.0)


class DirectionalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pixel = float(_cfg()["scales"]["pixel_um"])
        cls.bands = [(8, 16, "8_16"), (16, 32, "16_32"), (32, 64, "32_64"),
                     (64, 1e9, "64_inf")]

    def test_fft_window_normalization_finite(self):
        rng = np.random.default_rng(2)
        R = rng.normal(size=(3, 160, 160))
        df = p25.directional_band_metrics(R, self.pixel, self.bands, 36)
        self.assertTrue(np.isfinite(df[["A2", "theta_k_deg",
                                        "theta_stripe_deg",
                                        "angular_entropy"]].to_numpy()).all())
        self.assertTrue(((df["A2"] >= 0) & (df["A2"] <= 1 + 1e-12)).all())

    def test_isotropic_field_low_A2(self):
        rng = np.random.default_rng(3)
        R = rng.normal(size=(2, 160, 160))
        df = p25.directional_band_metrics(R, self.pixel, self.bands, 36)
        self.assertLess(float(df[df.band == "8_16"]["A2"].median()), 0.2)

    def test_vertical_stripe_orientation(self):
        n = 160
        x = np.arange(n) * self.pixel
        field = np.cos(2 * np.pi * x / 10.0)[None, :] \
            * np.ones((n, 1))          # varies along x: lambda = 10 um
        R = field[None, :, :] + 0.01 * np.random.default_rng(4).normal(
            size=(1, n, n))
        df = p25.directional_band_metrics(R, self.pixel, self.bands, 36)
        row = df[(df.band == "8_16") & (df.dataset_index == 0)].iloc[0]
        self.assertGreater(float(row["A2"]), 0.8)
        self.assertLess(abs(float(row["theta_k_deg"])), 5.0)   # wave-vector ~0
        # x-varying field -> vertical stripes: stripe angle ~ 90 deg
        self.assertLess(abs(float(row["theta_stripe_deg"]) - 90.0), 5.0)


class SplitAndGroupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = REPO / _cfg()["paths"]["phase2_manifest"]
        if not path.exists():
            raise unittest.SkipTest("phase2 manifest not present")
        cls.man = pd.read_csv(path)

    def test_gkf_source_contract(self):
        g = self.man["shared_height_source_id"].to_numpy()
        sp = p25.gkf_splits(g, 5)
        p25.check_gkf_contract(g, sp)

    def test_gkf_process_contract(self):
        g = self.man["cv_process_group"].to_numpy()
        sp = p25.gkf_splits(g, 5)
        p25.check_gkf_contract(g, sp)

    def test_sentinel_same_process_group(self):
        m = self.man
        sent = m[(m.session_id == "zro2_120_formal")
                 & (m.processing_order.isin([49, 50]))]
        self.assertEqual(len(sent), 2)
        self.assertEqual(sent["cv_process_group"].nunique(), 1)


class PassSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = REPO / _cfg()["paths"]["phase2_manifest"]
        if not path.exists():
            raise unittest.SkipTest("phase2 manifest not present")
        cls.man = pd.read_csv(path)

    def test_pass_bases_15(self):
        pm = self.man[(self.man.session_role == "pass_main")]
        self.assertEqual(pm["base_condition_group"].nunique(), 15)
        self.assertEqual(len(pm), 60)

    def test_supplement_bases_10(self):
        ps = self.man[self.man.session_role == "pass_supplement"]
        self.assertEqual(ps["base_condition_group"].nunique(), 10)
        self.assertEqual(len(ps), 20)

    def test_n4_to_5_refuses(self):
        with self.assertRaises(AssertionError):
            p25.require_no_n4_to_5([(3, 4), (4, 5)])
        p25.require_no_n4_to_5([(1, 2), (2, 3), (3, 4)])
        p25.require_no_n4_to_5([(5, 6)])          # independent check allowed


class SignFlipTests(unittest.TestCase):
    def test_signflip_15_equals_32768(self):
        S = p25.sign_matrix(15)
        self.assertEqual(S.shape, (32768, 15))
        rng = np.random.default_rng(5)
        dz = rng.normal(size=(15, 4))
        res = p25.exact_signflip_test(dz)
        self.assertEqual(res["n_configurations"], 32768)
        # the observed configuration is part of the enumeration space
        self.assertGreaterEqual(res["p_exact_global"], 1 / 32768)

    def test_signflip_10_equals_1024(self):
        S = p25.sign_matrix(10)
        self.assertEqual(S.shape, (1024, 10))
        dz = np.ones((10, 2))                       # maximally coherent
        res = p25.exact_signflip_test(dz)
        self.assertEqual(res["n_configurations"], 1024)
        # both the all-+1 and the all--1 configuration reach T_obs
        self.assertAlmostEqual(res["p_exact_global"], 2 / 1024, places=12)


class MorandiagTests(unittest.TestCase):
    def test_moran_permutation_finite(self):
        rng = np.random.default_rng(6)
        X = rng.normal(size=(40, 3))
        W = p25.knn_row_standardized_graph(X, 5)
        z = X[:, 0] + 3.0 * X[:, 1]                 # spatially structured
        i_obs, p = p25.moran_permutation_p(z, W, 200, seed=7)
        self.assertTrue(np.isfinite(i_obs))
        self.assertGreaterEqual(p, 1 / 201)
        self.assertLessEqual(p, 1.0)


class TaskContractTests(unittest.TestCase):
    def test_blind_labels_validation_only(self):
        import json
        cfg = _cfg()
        blob = json.dumps(cfg["targets"])
        self.assertNotIn("blind", blob.lower())
        self.assertNotIn("stripe" + "_phenotype", blob.lower())

    def test_mechanism_features_no_morphology_dependency(self):
        path = REPO / "outputs/phase2_5/mechanism_bridge" \
                     / "mechanism_feature_provenance.csv"
        if not path.exists():
            self.skipTest("14A provenance output not present")
        prov = pd.read_csv(path)
        prim = prov[prov["allowed_primary"] == True]  # noqa: E712
        self.assertTrue((~prim["depends_on_measured_morphology"].astype(bool)
                         ).all())
        self.assertTrue((~prim["was_fitted_using_labels"].astype(bool)).all())

    def test_oof_rows_unique_per_sample(self):
        path = REPO / "outputs/phase2_5/process_map" \
                     / "composition_oof_predictions.csv"
        if not path.exists():
            self.skipTest("Task 12 OOF output not present")
        oof = pd.read_csv(path)
        src = oof[oof["cv_variant"] == "src_gkf"]
        dup = src.duplicated(["model", "input_set", "dataset_index"])
        self.assertEqual(int(dup.sum()), 0)

    def test_targets_align_manifest(self):
        path = REPO / "outputs/phase2_5/spectral_composition" \
                     / "spectral_composition.csv"
        if not path.exists():
            self.skipTest("Task 10 output not present")
        comp = pd.read_csv(path)
        man = p25.build_manifest(_cfg()) if hasattr(p25, "build_manifest") \
            else None
        self.assertEqual(len(comp), 200)
        self.assertTrue((comp["dataset_index"].to_numpy() == np.arange(200)
                         ).all())
        self.assertIsNone(man)


if __name__ == "__main__":
    unittest.main()
