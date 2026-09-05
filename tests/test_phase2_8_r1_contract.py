"""Regression tests for review counterexamples and corrected scientific contracts."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from src.forward_models import (array_transfer_v2, physical_validity_relative_v2,
    phase_grid_v2, synth_field, pairwise_interaction_field)

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    sp = importlib.util.spec_from_file_location(name, ROOT / "experiments/phase2_8_r1" / (name + ".py"))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


a = load("24_information_decomposition")
b = load("25_kernel_bridge")


class PhysicalContract(unittest.TestCase):
    def test_positive_baseline_negative_removal_rejected(self):
        x = np.linspace(-10, 10, 201)
        g = 10 * np.exp(-0.5 * (x / 4) ** 2)
        base = synth_field(g, x, 4, 0, 0)
        z = pairwise_interaction_field(g, x, 4, 0, -0.5)
        self.assertTrue((base > 0).all())
        self.assertLess(z.min(), -40)
        self.assertFalse(physical_validity_relative_v2(z, base)["valid"])

    def test_noise_preserved_but_not_amplified(self):
        base = np.array([-0.15, 0.0, 2.0])
        self.assertTrue(physical_validity_relative_v2(base, base)["valid"])
        self.assertFalse(physical_validity_relative_v2(np.array([-0.17, 0., 2.]), base)["valid"])
        self.assertFalse(physical_validity_relative_v2(np.array([-0.15, 0., -0.02]), base)["valid"])

    def test_nonfinite_rejected_and_no_mutation(self):
        z = np.array([np.nan, 2.])
        self.assertFalse(physical_validity_relative_v2(z, np.ones(2))["valid"])
        self.assertTrue(np.isnan(z[0]))

    def test_failed_prediction_kept_as_invalid_class(self):
        q = np.array([0., 0., 1., 0., 0.])
        scored = b.admissible_q(q, False)
        self.assertEqual(scored[b.CODE_INVALID], 1)
        self.assertEqual(1 - scored[2], 1)
        np.testing.assert_array_equal(q, [0, 0, 1, 0, 0])


class PhaseAndTransfer(unittest.TestCase):
    def test_alternating_covers_both_parities(self):
        phase = phase_grid_v2(6, 32, "L3a", 0.2)
        self.assertEqual(sum(phase >= 6), 16)
        self.assertEqual(phase[-1], 12 * 31 / 32)
        for lv, p in [("L1", None), ("L3a", 0), ("L3b", -0.5)]:
            self.assertLess(phase_grid_v2(6, 32, lv, p).max(), 6)

    def test_dc_and_reciprocal_peaks(self):
        np.testing.assert_allclose(array_transfer_v2(np.array([0., 0.25, -0.25, 0.5]), 4, 20), 400)
        np.testing.assert_allclose(array_transfer_v2(np.array([0., 1e-14, .3]), 4, 1), 1)

    def test_two_line_analytic_near_resonance(self):
        k = np.array([0., .25 - 1e-12, .25, .25 + 1e-12, .1])
        np.testing.assert_allclose(array_transfer_v2(k, 4, 2), 2 + 2*np.cos(2*np.pi*k*4), atol=1e-12)


class SelectionContract(unittest.TestCase):
    def test_median_not_mean(self):
        records = [{"group": str(i), "param": p, "loss": loss, "valid": True}
                   for p, losses in [(0., [0,0,0,1,1]), (1., [.3]*5)]
                   for i, loss in enumerate(losses)]
        param, scores = b.select_candidate(pd.DataFrame(records), "held")
        self.assertEqual(param, 0.)

    def test_holdout_response_and_validity_never_select(self):
        records = [{"group": g, "param": p, "loss": p, "valid": True}
                   for g in ["a", "b", "held"] for p in [0., 1.]]
        table = pd.DataFrame(records)
        before = b.select_candidate(table, "held")
        table.loc[table.group == "held", "loss"] = -10000
        table.loc[table.group == "held", "valid"] = False
        after = b.select_candidate(table, "held")
        self.assertEqual(before, after)

    def test_training_invalid_excluded_and_ties_recorded(self):
        table = pd.DataFrame([{"group": g, "param": p, "loss": 0., "valid": p != -1.}
                              for g in ["a", "b"] for p in [-1., 0., 1.]])
        selected, scores = b.select_candidate(table, "held")
        self.assertEqual(selected, 0.)
        self.assertEqual([s["param"] for s in scores if s.get("tied_best")], [0., 1.])
        table["valid"] = False
        self.assertIsNone(b.select_candidate(table, "held")[0])


class OofContract(unittest.TestCase):
    def test_native_predictions_rebuild_scalar_ilr_and_joint_scores(self):
        rng = np.random.default_rng(35)
        n = 30
        df = pd.DataFrame({"dataset_index": np.arange(n) + 100,
                           "shared_height_source_id": np.arange(n).astype(str),
                           "cv_process_group": np.arange(n).astype(str),
                           "u": rng.normal(size=n)})
        for c in ["median_depth_um", "ilr_z1", "ilr_z2", "ilr_z3", "ilr_z4", *a.JOINT_COLS]:
            df[c] = 3 * df.u + rng.normal(size=n)
        df[a.JOINT_COLS[1]] *= 100  # catch native/standardized coordinate mixups
        splits = {"toy": a.gkf_splits(df.cv_process_group, 5)}
        records = []
        with patch.object(a, "select_alpha_inner", return_value=1.):
            folds = a.run_cv(df, splits, {"full": ["u"]}, ["D", "Pl", "Ot_joint"], oof_records=records)
        oof = pd.DataFrame(records)
        qa = a.validate_oof(oof, folds)
        self.assertLess(qa["max_q2_reconstruction_error"], 1e-10)
        self.assertEqual(set(oof.dataset_index), set(df.dataset_index))
        self.assertTrue(folds.loc[folds.target == "D", "r2_scalar"].notna().all())
        with self.assertRaises(AssertionError):
            a.validate_oof(pd.concat([oof, oof.iloc[:1]]), folds)


if __name__ == "__main__":
    unittest.main()
