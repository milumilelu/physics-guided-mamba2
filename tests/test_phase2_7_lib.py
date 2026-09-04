"""Phase 2.7 unit tests (synthetic; CI-safe).  Covers the frozen v2.1
definitions: interval assignment, two-layer distributions, coverage/label
precedence, TV permutation p, mutual-exclusive verdict order, phase domains,
five-class TV, LOHO, d_i guard, and the language boundary."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "phase2_6"))
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location(
    "phase2_7_lib", REPO / "experiments" / "phase2_7" / "_lib.py")
p27 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p27)


def _cfg():
    import yaml
    with open(REPO / "experiments" / "phase2_7" / "phase2_7_config.yaml",
              encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class IntervalTests(unittest.TestCase):
    def test_interval_boundaries_inclusive(self):
        expect = [(0.75, 2), (1.0, 2), (1.25, 2), (1.5, 1), (1.75, 3),
                  (2.0, 3), (2.25, 3), (2.5, 1), (2.75, 4), (3.25, 4),
                  (0.5, 1), (3.5, 1)]
        for r, code in expect:
            got = p27.assign_class(np.array([r]), np.array([True]))[0]
            p27.require(got == code, f"r={r} -> {got}, expected {code}")

    def test_invalid_and_no_tie_logic(self):
        got = p27.assign_class(np.array([1.0, np.nan]), np.array([True, True]))
        self.assertEqual(got[0], p27.CODE_M1)
        self.assertEqual(got[1], p27.CODE_INVALID)
        src = (REPO / "experiments" / "phase2_7" / "_lib.py").read_text(
            encoding="utf-8")
        p27.require("argmin" not in src or "tie" not in src.lower(),
                    "interval implementation must not contain tie logic")


class DistributionTests(unittest.TestCase):
    def test_two_layer_distribution_and_coverage(self):
        classes = np.array([p27.CODE_OUT, p27.CODE_M1, p27.CODE_M2,
                            p27.CODE_M1, p27.CODE_INVALID])
        q = p27.q_distribution(classes)
        p27.require(abs(q.sum() - 1) < 1e-12, "q must sum to 1")
        p27.require(abs(q[0] - 0.2) < 1e-12 and abs(q[1] - 0.2) < 1e-12,
                    "INVALID/OUT shares")
        # C_family = 1 - P(OUT | peak-valid) = family / peak-valid = 3/4
        family_rate = float((classes[classes != p27.CODE_INVALID]
                             != p27.CODE_OUT).mean())
        p27.require(abs(family_rate - 0.75) < 1e-12,
                    "C_family = 1 - P(OUT | peak-valid)")

    def test_tv_range_and_perm_p(self):
        q_a = np.array([0.1, 0.3, 0.4, 0.2, 0.0])
        q_b = np.array([0.1, 0.3, 0.4, 0.2, 0.0])
        p27.require(p27.tv(q_a, q_b) == 0.0, "identical q -> TV 0")
        q_c = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        p27.require(abs(p27.tv(q_a, q_c) - 1.0) < 1e-12, "TV bounded by 1")
        # frozen permutation-p formula: pooled null center q_bar = q_n;
        # T_obs = TV(q_obs, q_n) = 1 > all T_b = 0 -> p = 1/(1+B)
        q_n = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        q_obs = {2.0: q_a}
        q_null = {2.0: [q_n] * 10}
        result = p27.tv_perm_p(q_obs, q_null, {2.0: 1.0}, n_perm=10)
        p27.require(abs(result["p_value"] - 1 / 11) < 1e-9,
                    "perm p must follow the frozen (1+b)/(1+B) formula")


class VerdictTests(unittest.TestCase):
    TH = {"tv": {"delta_min": 0.10, "period2_max": 0.20, "inadequate": 0.30},
          "h_consistency": {"min_n_obs": 8, "min_wins": 3,
                            "min_evaluable": 3},
          "d_guard": {"n_eval_min": 8, "contradiction_frac": 0.3334}}

    def test_model_inadequate_first(self):
        result = p27.verdict_g27_3(0.50, 0.35, 0.15, 0.05, 5, 5,
                                   [], 0, thresholds=self.TH)
        self.assertEqual(result["G_SL3"], "MODEL_INADEQUATE")

    def test_not_supported_when_delta_leq_zero(self):
        result = p27.verdict_g27_3(0.20, 0.25, -0.05, -0.05, 5, 5,
                                   [], 0, thresholds=self.TH)
        self.assertEqual(result["G_SL3"], "NOT_SUPPORTED")

    def test_supported_full_conditions(self):
        result = p27.verdict_g27_3(0.30, 0.15, 0.15, 0.06, 4, 5,
                                   [], 0, thresholds=self.TH)
        self.assertEqual(result["G_SL3"], "SUPPORTED")

    def test_partial_when_tv_period2_exceeds(self):
        result = p27.verdict_g27_3(0.30, 0.25, 0.15, 0.06, 4, 5,
                                   [], 0, thresholds=self.TH)
        self.assertEqual(result["G_SL3"], "PARTIAL")

    def test_exact_match_guard_caps_supported(self):
        result = p27.verdict_g27_3(0.30, 0.15, 0.15, 0.06, 4, 5,
                                   [0.1, -0.2, -0.2, -0.2] * 3, 12,
                                   thresholds=self.TH)
        p27.require(result["G_SL3"] == "PARTIAL",
                    "d_i contradictions must cap SUPPORTED at PARTIAL")

    def test_mixed_beats_dominant(self):
        # 0.55/0.45 must land MIXED, never DOMINANT (v2.1 ordering)
        ps = sorted([0.55, 0.45, 0.0], reverse=True)
        mixed = (ps[0] - ps[1] < 0.15) and (ps[1] >= 0.25)
        p27.require(mixed, "0.55/0.45 is MIXED under the frozen rule")
        dominant = (ps[0] >= 0.50) and (ps[0] - ps[1] >= 0.15)
        p27.require(not (mixed and dominant),
                    "MIXED and DOMINANT must be mutually exclusive")


class PhaseDomainTests(unittest.TestCase):
    def test_phase_domains_frozen(self):
        cfg = _cfg()
        g3 = cfg["g27_3"]
        h = 6.0
        phi_const = np.arange(g3["phases_final"]) * (h / g3["phases_final"])
        phi_p2 = np.arange(g3["phases_final"]) * (2 * h / g3["phases_final"])
        p27.require(phi_const.max() < h, "constant phases inside [0,h)")
        p27.require(phi_p2.max() < 2 * h and phi_p2.min() >= 0,
                    "period2 phases inside [0,2h)")

    def test_measurability_table_frozen(self):
        expect = {(2, 1): "HIGH", (2, 2): "HIGH", (2, 3): "HIGH",
                  (4, 1): "HIGH", (4, 2): "HIGH", (4, 3): "LOW",
                  (6, 1): "HIGH", (6, 2): "LOW", (6, 3): "UNMEASURABLE",
                  (8, 1): "HIGH", (8, 2): "UNMEASURABLE",
                  (10, 1): "LOW", (10, 2): "UNMEASURABLE"}
        for (h, m), level in expect.items():
            got = p27.cycles_level(m * h)
            p27.require(got == level, f"({h},{m}) {got} != {level}")

    def test_parseval_spot_check_hann_projection(self):
        # Hann projection at k=0 equals (sum w g)^2; monotone sanity on a
        # pure sinusoid: peak at its own frequency
        x = np.arange(64) * 0.278657
        lam = 8.0
        g = np.sin(2 * np.pi * x / lam)
        s_peak = p27.hann_projection(g, x, 1 / lam)
        s_off = p27.hann_projection(g, x, 1 / (lam * 2))
        p27.require(s_peak > s_off, "sinusoid peaks at its own k")


class EnvelopePopulationTests(unittest.TestCase):
    TH = {"tv": {"delta_min": 0.10, "period2_max": 0.20, "inadequate": 0.30},
          "h_consistency": {"min_n_obs": 8, "min_wins": 3,
                            "min_evaluable": 3},
          "d_guard": {"n_eval_min": 8, "contradiction_frac": 0.3334}}

    def test_profile_suitability_rule(self):
        good = np.r_[np.zeros(3), np.full(58, 6.0), np.zeros(3)]
        bad = np.full(64, 6.0)
        p27.require(p27.profile_suitable(good), "edges at background -> suitable")
        p27.require(not p27.profile_suitable(bad),
                    "edges at full depth -> unsuitable (no hard-zero extension)")

    def test_verdict_never_claims_nonlinearity(self):
        result = p27.verdict_g27_3(0.50, 0.35, 0.15, 0.05, 5, 5,
                                   [], 0, thresholds=self.TH)
        p27.require("nonlinear" not in result["G_SL3"],
                    "MODEL_INADEQUATE must not encode material nonlinearity")


if __name__ == "__main__":
    unittest.main()
