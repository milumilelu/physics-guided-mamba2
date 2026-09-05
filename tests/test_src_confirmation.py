"""Unit tests for the confirmation interface skeleton (Phase 2.8 v2.1 §5.1):
lock protocol discipline, version/config binding, and the Phase 2.8
discovery-only boundary (no estimator wired yet -> predict must refuse)."""

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src import confirmation as conf  # noqa: E402


class ConfirmationSkeleton(unittest.TestCase):
    def test_fit_binds_identity_and_features(self):
        man = pd.DataFrame({"a": range(10)})
        model = conf.fit(man, model="ridge_route_T_v1",
                         config={"alpha": 1.0}, feature_columns=["h_um"])
        self.assertEqual(model["model"], "ridge_route_T_v1")
        self.assertEqual(model["feature_columns"], ["h_um"])
        self.assertEqual(model["n_discovery"], 10)
        self.assertEqual(len(model["config_hash"]), 16)

    def test_config_hash_changes_with_config(self):
        man = pd.DataFrame({"a": range(3)})
        h1 = conf.fit(man, model="m", config={"alpha": 1.0},
                      feature_columns=["x"])["config_hash"]
        h2 = conf.fit(man, model="m", config={"alpha": 2.0},
                      feature_columns=["x"])["config_hash"]
        self.assertNotEqual(h1, h2)

    def test_predict_refuses_without_estimator_discovery_only(self):
        man = pd.DataFrame({"a": range(3)})
        model = conf.fit(man, model="m", config={}, feature_columns=["x"])
        with self.assertRaises(conf.LockError):
            conf.predict(model, man)

    def test_evaluate_refuses_unmade_predictions(self):
        with self.assertRaises(conf.LockError):
            conf.evaluate_locked_predictions({"predictions": None}, [1, 2])

    def test_lock_roundtrip_persists_pre_registration(self):
        lock = {"model": "m", "config_hash": "abc", "predictions": [1, 2]}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "lock.json"
            conf.write_lock(lock, path)
            self.assertEqual(conf.read_lock(path), lock)


if __name__ == "__main__":
    unittest.main()
