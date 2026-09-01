"""The native intermediate format must round-trip without losing the mask."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data_contracts import HeightMap
from src.io_npz import load_height_npz, save_height_npz


def sample(nan_holes: bool = True) -> HeightMap:
    rng = np.random.default_rng(42)
    z = np.round(rng.uniform(-5, 40, size=(32, 48)), 3)
    mask = np.ones_like(z, dtype=bool)
    if nan_holes:
        mask[0, 0] = False
        mask[5, 7:9] = False
        mask[20, :] = False
        z[~mask] = np.nan
    dx = dy = 0.344174
    return HeightMap(
        z=z,
        valid_mask=mask,
        dx_um=dx,
        dy_um=dy,
        x_um=(np.arange(48) + 0.5) * dx,
        y_um=(np.arange(32) + 0.5) * dy,
        metadata={"group": 7, "data_name": "13 14",
                  "mask_source": "cag_raw_sentinel",
                  "mask_is_fabricated": False,
                  "note": "中文 metadata 也要能原样回来"},
    )


class TestRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "nested" / "h.npz"

    def tearDown(self):
        self.tmp.cleanup()

    def test_heights_survive(self):
        original = sample()
        save_height_npz(self.path, original)
        restored = load_height_npz(self.path)
        np.testing.assert_array_equal(restored.z, original.z)

    def test_valid_matches_exactly(self):
        original = sample()
        save_height_npz(self.path, original)
        restored = load_height_npz(self.path)
        np.testing.assert_array_equal(restored.valid_mask, original.valid_mask)
        self.assertEqual(restored.n_invalid, original.n_invalid)

    def test_nan_holes_stay_nan(self):
        original = sample()
        save_height_npz(self.path, original)
        hole = (5, 7)
        self.assertTrue(np.isnan(load_height_npz(self.path).z[hole]))

    def test_pitch_and_coordinates_survive(self):
        original = sample()
        save_height_npz(self.path, original)
        restored = load_height_npz(self.path)
        self.assertEqual(restored.dx_um, original.dx_um)
        self.assertEqual(restored.dy_um, original.dy_um)
        np.testing.assert_allclose(restored.x_um, original.x_um)
        np.testing.assert_allclose(restored.y_um, original.y_um)

    def test_metadata_survives(self):
        original = sample()
        save_height_npz(self.path, original)
        restored = load_height_npz(self.path)
        self.assertEqual(restored.metadata["data_name"], "13 14")
        self.assertEqual(restored.metadata["group"], 7)
        self.assertEqual(restored.metadata["note"],
                         "中文 metadata 也要能原样回来")
        self.assertFalse(restored.metadata["mask_is_fabricated"])

    def test_fully_valid_map_round_trips(self):
        original = sample(nan_holes=False)
        save_height_npz(self.path, original)
        restored = load_height_npz(self.path)
        self.assertEqual(restored.n_invalid, 0)
        np.testing.assert_array_equal(restored.z, original.z)

    def test_restored_object_passes_the_contract(self):
        """Round-tripping must not smuggle a filled pixel past the guard."""
        original = sample()
        save_height_npz(self.path, original)
        restored = load_height_npz(self.path)
        self.assertTrue(np.all(np.isnan(restored.z[~restored.valid_mask])))


if __name__ == "__main__":
    unittest.main()
