"""src.geometry parity tests vs the frozen phase2_6/phase2_7 _lib originals
(WP1 migration protocol; synthetic inputs, CI-safe)."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src import geometry as sgeo  # noqa: E402


def _load(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p26 = _load("phase2_6_lib_geo_parity", "experiments/phase2_6/_lib.py")
p27 = _load("phase2_7_lib_geo_parity", "experiments/phase2_7/_lib.py")


def _profile(seed: int = 1, n: int = 64):
    rng = np.random.default_rng(seed)
    v = np.arange(n, dtype=float) - n / 2
    prof = 6.0 * np.exp(-0.5 * (v / 8.0) ** 2) + rng.normal(scale=0.05, size=n)
    prof[:3] = 0.05
    prof[-3:] = 0.05
    return prof, v


class ClassAssignmentParity(unittest.TestCase):
    def test_constants_identical(self):
        self.assertEqual(sgeo.CLASS_NAMES, p27.CLASS_NAMES)
        self.assertEqual(sgeo.INTERVALS, p27.INTERVALS)
        self.assertEqual(sgeo.Q_BY_KEY, p26.Q_BY_KEY)
        self.assertEqual(sgeo.WIDTH_Q_KEYS, p26.WIDTH_Q_KEYS)

    def test_assign_class_and_q_distribution_identical(self):
        r = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.25, 3.0, np.nan])
        valid = np.isfinite(r)
        np.testing.assert_array_equal(sgeo.assign_class(r, valid),
                                      p27.assign_class(r, valid))
        cls = sgeo.assign_class(r, valid)
        np.testing.assert_array_equal(sgeo.q_distribution(cls),
                                      p27.q_distribution(cls))

    def test_profile_suitable_identical(self):
        prof, _ = _profile()
        self.assertEqual(sgeo.profile_suitable(prof, edge_frac_max=0.15),
                         p27.profile_suitable(prof, edge_frac_max=0.15))
        bad = prof.copy()
        bad[:3] = 2.0
        self.assertEqual(sgeo.profile_suitable(bad, edge_frac_max=0.15),
                         p27.profile_suitable(bad, edge_frac_max=0.15))


class GeometryParity(unittest.TestCase):
    def test_axis_frame_lateral_identical(self):
        for theta in (0.0, 37.8, 127.8):
            a = sgeo.axis_frame(theta)
            b = p26.axis_frame(theta)
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(sgeo.lateral_positions(64, 0.278657),
                                      p26.lateral_positions(64, 0.278657))

    def test_section_features_identical(self):
        prof, v = _profile()
        a = sgeo.section_features(prof, v, ("W20", "W50", "W80"),
                                  affected_delta_um=1.0)
        b = p26.section_features(prof, v, ("W20", "W50", "W80"),
                                 affected_delta_um=1.0)
        self.assertEqual(sorted(a.keys()), sorted(b.keys()))
        for key in a:
            np.testing.assert_equal(a[key], b[key])

    def test_lambda_peak_4_32_identical(self):
        rng = np.random.default_rng(3)
        rows = []
        for di in (0, 1):
            for b in range(24):
                rows.append({"dataset_index": di, "bin": b,
                             "lambda_geo_um": np.geomspace(0.7, 160, 24)[b],
                             "energy": rng.random() * (1 if b != 8 else 20),
                             "n_modes": 30})
        df = pd.DataFrame(rows)
        a = sgeo.lambda_peak_4_32(df, window_um=(4.0, 32.0), n_modes_min=20,
                                  share_min=0.20)
        b = p26.lambda_peak_4_32(df, window_um=(4.0, 32.0), n_modes_min=20,
                                 share_min=0.20)
        self.assertTrue(a.equals(b))

    def test_shuffle_h_by_block_identical(self):
        man = pd.DataFrame({
            "session_id": ["s1"] * 8 + ["s2"] * 8,
            "cv_process_group": (["g%d" % i for i in range(8)] * 2),
            "hatch_spacing_um": [2.0, 4.0, 6.0, 8.0, 10.0, 2.0, 4.0, 6.0] * 2,
        })
        a = sgeo.shuffle_h_by_block(man, unit_columns=("cv_process_group",),
                                    seed=42)
        b = p26.shuffle_h_by_block(man, unit_columns=("cv_process_group",),
                                   seed=42)
        np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())

    def test_in_box_mask_and_condition_key_identical(self):
        man = pd.DataFrame({
            "pulse_duration_fs": [223.0, 500.0],
            "frequency_kHz": [5.0, 20.0],
            "velocity_mm_s": [5.0, 50.0],
            "pass_count": [1, 4],
        })
        box = {"tau_fs": [200, 300], "f_khz": [2, 10], "v_mm_s": [2, 10],
               "pass": [1, 2]}
        self.assertTrue(sgeo.in_box_mask(man, box).equals(p26.in_box_mask(man, box)))
        self.assertTrue(sgeo.condition_key(man).equals(p26.condition_key(man)))


if __name__ == "__main__":
    unittest.main()
