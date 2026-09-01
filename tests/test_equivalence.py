import unittest

import numpy as np

from src.data_contracts import HeightMap
from src.equivalence import compare_height_maps


def make_map(z, mask=None, *, fabricated=False, dx=0.344174):
    z = np.asarray(z, dtype=float)
    if mask is None:
        mask = np.ones_like(z, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    z = np.where(mask, z, np.nan)
    return HeightMap(
        z=z,
        valid_mask=mask,
        dx_um=dx,
        dy_um=dx,
        x_um=(np.arange(z.shape[1]) + 0.5) * dx,
        y_um=(np.arange(z.shape[0]) + 0.5) * dx,
        metadata={
            "mask_source": "unavailable" if fabricated else "test_mask",
            "mask_is_fabricated": fabricated,
        },
    )


class TestEquivalence(unittest.TestCase):
    def setUp(self):
        self.z = np.arange(12 * 10, dtype=float).reshape(12, 10) / 1000

    def test_exact_height_can_pass_while_mask_gate_stops(self):
        cag = make_map(self.z)
        official_csv = make_map(self.z, fabricated=True)
        result = compare_height_maps(cag, official_csv)
        self.assertEqual(result["height_decision"], "PASS")
        self.assertEqual(result["mask_decision"], "UNAVAILABLE")
        self.assertEqual(result["overall_decision"], "STOP")

    def test_real_matching_masks_close_the_gate(self):
        mask = np.ones_like(self.z, dtype=bool)
        mask[3, 4] = False
        cag = make_map(self.z, mask)
        independent = make_map(self.z, mask)
        result = compare_height_maps(cag, independent)
        self.assertEqual(result["height_decision"], "PASS")
        self.assertEqual(result["mask_decision"], "PASS")
        self.assertEqual(result["overall_decision"], "PASS")

    def test_all_valid_case_can_be_explicitly_accepted(self):
        cag = make_map(self.z)
        official_csv = make_map(self.z, fabricated=True)
        result = compare_height_maps(
            cag, official_csv, allow_all_valid_mask_case=True)
        self.assertEqual(result["mask_decision"], "PASS_ALL_VALID_CASE")
        self.assertEqual(result["overall_decision"], "PASS")

    def test_all_valid_exception_never_hides_a_real_sentinel(self):
        mask = np.ones_like(self.z, dtype=bool)
        mask[2, 3] = False
        cag = make_map(self.z, mask)
        official_csv = make_map(self.z, fabricated=True)
        result = compare_height_maps(
            cag, official_csv, allow_all_valid_mask_case=True)
        self.assertEqual(result["mask_decision"], "UNAVAILABLE")
        self.assertEqual(result["overall_decision"], "STOP")

    def test_flipped_matrix_fails_orientation_and_height(self):
        cag = make_map(self.z)
        flipped = make_map(self.z[:, ::-1])
        result = compare_height_maps(cag, flipped, require_mask_evidence=False)
        self.assertEqual(result["height_decision"], "FAIL")
        self.assertEqual(result["orientation_best_transform"], "flip_x")
        self.assertEqual(result["overall_decision"], "STOP")

    def test_pitch_outside_tolerance_fails_height_gate(self):
        cag = make_map(self.z)
        other = make_map(self.z, dx=0.344176)
        result = compare_height_maps(cag, other, require_mask_evidence=False)
        self.assertFalse(result["pitch_pass"])
        self.assertEqual(result["height_decision"], "FAIL")


if __name__ == "__main__":
    unittest.main()
