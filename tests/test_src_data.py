"""src.data parity tests: canonical load_frozen vs the frozen Phase 1.5
_lib.load_frozen (WP1 migration protocol; loads the real frozen dataset)."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "phase1_5"))

import _lib as l15  # noqa: E402
from src import data as sdata  # noqa: E402


def _cfg():
    import yaml
    with open(REPO / "experiments" / "phase1_5" / "phase1_5_config.yaml",
              encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class LoadFrozenParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = _cfg()
        cls.a = l15.load_frozen(cls.cfg)
        cls.b = sdata.load_frozen(cls.cfg)

    def test_keys_identical(self):
        self.assertEqual(sorted(self.a.keys()), sorted(self.b.keys()))

    def test_manifest_identical(self):
        self.assertTrue(self.a["man"].equals(self.b["man"]))

    def test_arrays_bitwise_identical(self):
        for key in ("H", "V", "R", "Hnan"):
            np.testing.assert_array_equal(self.a[key], self.b[key])

    def test_contract_values(self):
        for d in (self.a, self.b):
            self.assertEqual(d["H"].shape, (200, 160, 160))
            self.assertEqual(d["man"]["shared_height_source_id"].nunique(), 160)


if __name__ == "__main__":
    unittest.main()
