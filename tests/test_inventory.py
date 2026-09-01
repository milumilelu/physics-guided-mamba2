import unittest

import numpy as np

from src.data_contracts import HeightMap
from src.inventory import (
    build_sample_search_regions,
    compute_height_diagnostics,
    compute_invalid_components,
)


def height_map(z, mask=None, pitch=1.0):
    z = np.asarray(z, dtype=float)
    if mask is None:
        mask = np.ones_like(z, dtype=bool)
    z = np.where(mask, z, np.nan)
    return HeightMap(z, np.asarray(mask, dtype=bool), pitch, pitch,
                     (np.arange(z.shape[1]) + 0.5) * pitch,
                     (np.arange(z.shape[0]) + 0.5) * pitch, {})


class TestSearchRegions(unittest.TestCase):
    def test_paired_slots_preserve_supplied_order(self):
        regions = build_sample_search_regions(
            width_um=704.0, height_um=528.0, sample_ids=[14, 13])
        self.assertEqual([r["sample_id"] for r in regions], [14, 13])
        self.assertLess(regions[0]["center_search_x_max_um"], 0)
        self.assertGreater(regions[1]["center_search_x_min_um"], 0)

    def test_paired_domains_cannot_overlap(self):
        left, right = build_sample_search_regions(
            width_um=704.0, height_um=528.0, sample_ids=[1, 2])
        self.assertLess(left["center_search_x_max_um"],
                        right["center_search_x_min_um"])

    def test_too_narrow_pair_fails(self):
        with self.assertRaises(ValueError):
            build_sample_search_regions(
                width_um=400.0, height_um=528.0, sample_ids=[1, 2])


class TestDiagnostics(unittest.TestCase):
    def test_invalid_component_is_reported_without_filling(self):
        z = np.arange(100, dtype=float).reshape(10, 10)
        mask = np.ones_like(z, dtype=bool)
        mask[2:4, 3:6] = False
        hm = height_map(z, mask)
        components = compute_invalid_components(hm)
        self.assertEqual(components["invalid_component_count"], 1)
        self.assertEqual(components["invalid_component_max_area_um2"], 6.0)
        self.assertTrue(np.isnan(hm.z[2:4, 3:6]).all())

    def test_quantiles_ignore_invalid_pixels(self):
        z = np.arange(100, dtype=float).reshape(10, 10)
        mask = np.ones_like(z, dtype=bool)
        mask[0, 0] = False
        result = compute_height_diagnostics(height_map(z, mask))
        self.assertEqual(result["selected_valid_pixels"], 99)
        self.assertGreater(result["q50"], 49.0)


if __name__ == "__main__":
    unittest.main()
