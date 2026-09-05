"""Phase 2.7r2 formal-contract tests (external-review item 6).

These tests exist to catch the contract-level bugs that survived the r1
library unit tests: weighted-vs-macro TV aggregation, DOE-unit bootstrap
definition, own-envelope d_i semantics, and a forward-model positive
control (narrow kernel -> constant array must recover m=1)."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.forward_models import field_class, synth_field  # noqa: E402


def _load_task23():
    spec = importlib.util.spec_from_file_location(
        "phase2_7_task23_contract", REPO / "experiments" / "phase2_7"
        / "23_single_track_envelope.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t23 = _load_task23()


class WeightedLohoContract(unittest.TestCase):
    def test_evaluation_is_weighted_not_macro(self):
        # 5 h levels with very unequal counts: weighted and macro disagree
        h_levels = [2.0, 4.0, 6.0, 8.0, 10.0]
        n_h_all = {2.0: 40, 4.0: 40, 6.0: 60, 8.0: 40, 10.0: 20}
        q_obs = {h: np.array([0.0, 0.0, 1.0, 0.0, 0.0]) for h in h_levels}
        # q_M: perfect at h=2 (heavy weight), terrible at h=10 (light weight)
        q_m = {}
        for h in h_levels:
            q = np.zeros(5)
            q[2] = 1.0 if h != 10.0 else 0.0
            q[3 if h == 10.0 else 4] = 0.0 if h != 10.0 else 0.0
            q[4] = 1.0 if h == 10.0 else 0.0
            q_m[(h, 0.0)] = q
        c_grid = [0.0]
        tv_w, c_assign = t23.weighted_loho_tv(q_obs, q_m, c_grid, n_h_all,
                                              h_levels)
        # weighted: h=10 contributes 20/200 * 1 = 0.1; others 0
        self.assertAlmostEqual(tv_w, 0.1, places=12)
        # macro mean would be (4*0 + 1)/5 = 0.2 -> must NOT be returned
        self.assertNotAlmostEqual(tv_w, 0.2, places=12)

    def test_loho_selection_uses_train_h_only(self):
        h_levels = [4.0, 8.0]
        n_h_all = {4.0: 100, 8.0: 100}
        q_obs = {4.0: np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
                 8.0: np.array([0.0, 0.0, 0.0, 1.0, 0.0])}
        q_m = {(4.0, 0.0): np.array([0.0, 0.0, 0.0, 1.0, 0.0]),
               (4.0, 0.5): np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
               (8.0, 0.0): np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
               (8.0, 0.5): np.array([0.0, 0.0, 1.0, 0.0, 0.0])}
        tv_w, c_assign = t23.weighted_loho_tv(q_obs, q_m, [0.0, 0.5],
                                              n_h_all, h_levels)
        # held=4.0: c* chosen on train={8.0} where c=0.0 wins (tv 0 < 1)
        self.assertEqual(c_assign[4.0], 0.0)
        # held=8.0: c* chosen on train={4.0}: c=0.0 tv=1, c=0.5 tv=0 -> 0.5
        self.assertEqual(c_assign[8.0], 0.5)
        # held=4: c*=0 (train tie), TV=m1-vs-m2=1 -> 0.5*1
        # held=8: c*=0.5 (train: c=0.5 exact), TV=m2-vs-m1=1 -> 0.5*1
        self.assertAlmostEqual(tv_w, 1.0, places=12)


class DoeUnitBootstrapContract(unittest.TestCase):
    def _toy(self):
        return pd.DataFrame({
            "session_id": ["s1"] * 6 + ["s2"] * 6,
            "base_condition_group": ["B1", "B1", "B2", "B2", "B3", "B3"] * 2,
            "hatch_spacing_um": [4.0] * 6 + [8.0] * 6,
            "dataset_index": range(12),
        })

    def test_unit_is_session_times_base_condition_group(self):
        man = self._toy()
        labels = t23.doe_unit_labels(man)
        # (s1,B1),(s1,B2),(s1,B3),(s2,B1),(s2,B2),(s2,B3) -> 6 units
        self.assertEqual(len(np.unique(labels)), 6)
        # same (session, base group) across sessions must NOT share a unit
        self.assertNotEqual(int(labels[0]), int(labels[6]))

    def test_strata_are_h_times_session(self):
        man = self._toy()
        man["_doe_unit"] = t23.doe_unit_labels(man)
        strata = t23.doe_stratum_units(man, 4.0)
        self.assertEqual(sorted(strata.keys()), ["s1"])
        self.assertEqual(len(strata["s1"]), 3)  # 3 units in the stratum


class OwnEnvelopeDiContract(unittest.TestCase):
    def test_own_envelope_differs_from_population_borrowing(self):
        # narrow kernel -> constant array puts mass at m1; wide kernel's
        # own-envelope q differs -> borrowing the population q would give a
        # different d_i.  Toy scale: 8 phases, one h.
        h = 6.0
        x = (np.arange(215) - 107) * 0.278657
        narrow = 6.0 * np.exp(-0.5 * (x / 0.8) ** 2)
        wide = 6.0 * np.exp(-0.5 * (x / 6.0) ** 2)

        def own_q(prof, c):
            codes = [field_class(synth_field(prof, x, h, phi, c), h=h)[0]
                     for phi in np.arange(8) * (h if c == 0 else 2 * h) / 8]
            q = np.zeros(5)
            for code in codes:
                q[code] += 1
            return q / len(codes)

        # own-envelope semantics: each condition's q is a function of its
        # OWN profile -- two structurally different kernels must be able to
        # carry different q_C (population borrowing would erase this)
        q_narrow = own_q(narrow, 0.0)
        q_wide = own_q(wide, 0.0)
        self.assertFalse(np.allclose(q_narrow, q_wide),
                         "own-envelope q identical for distinct kernels")
        obs = 2  # m1
        d_narrow = float(own_q(narrow, 0.5)[obs] - q_narrow[obs])
        d_wide = float(own_q(wide, 0.5)[obs] - q_wide[obs])
        self.assertTrue(np.isfinite(d_narrow) and np.isfinite(d_wide))


class ForwardModelPositiveControl(unittest.TestCase):
    def test_narrow_kernel_constant_array_recovers_m1(self):
        # physical sanity: a kernel much narrower than h, constant array,
        # must select lambda ~ h (m=1) -- guards against observation-operator
        # regressions of the field_class family.
        h = 8.0
        x = (np.arange(215) - 107) * 0.278657
        prof = 6.0 * np.exp(-0.5 * (x / 0.8) ** 2)
        codes = [field_class(synth_field(prof, x, h, phi, 0.0), h=h)[0]
                 for phi in np.arange(8) * h / 8]
        m1_frac = float(np.mean([c == 2 for c in codes]))
        self.assertGreaterEqual(m1_frac, 0.5,
                                f"positive control failed: m1_frac={m1_frac}")


class DelegationContract(unittest.TestCase):
    def test_task25_uses_shared_builder(self):
        src = (REPO / "experiments" / "phase2_8" / "25_kernel_bridge.py"
               ).read_text(encoding="utf-8")
        self.assertIn("sdata.build_line_profile_library", src)
        self.assertNotIn("CagHeightReader", src)


if __name__ == "__main__":
    unittest.main()
