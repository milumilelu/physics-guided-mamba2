"""KEYENCE ImageDataCsv parsing, including everything that must fail loudly."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.io_vk_csv import VkCsvError, parse_vk_csv
from src.data_contracts import HeightMap

DATA = Path(__file__).resolve().parent / "data"


class TestEncodingAndHeader(unittest.TestCase):
    def test_gbk_header_is_parsed(self):
        hm = parse_vk_csv(DATA / "vk_gbk_8x6.csv")
        self.assertEqual(hm.metadata["encoding"], "gbk")
        self.assertEqual(hm.metadata["data_name"], "1 2")
        self.assertEqual(hm.metadata["instrument"], "VK-X3000 Series")
        self.assertEqual(hm.metadata["measurement_datetime"],
                         "2026-05-28 10:56:18")

    def test_utf8_sig_header_is_parsed(self):
        hm = parse_vk_csv(DATA / "vk_utf8_8x6.csv")
        self.assertEqual(hm.metadata["encoding"], "utf-8-sig")
        self.assertEqual(hm.metadata["data_name"], "3 4")

    def test_declared_size_is_checked(self):
        hm = parse_vk_csv(DATA / "vk_gbk_8x6.csv")
        self.assertEqual(hm.shape, (6, 8))


class TestUnits(unittest.TestCase):
    def test_pitch_converted_to_micrometres(self):
        hm = parse_vk_csv(DATA / "vk_gbk_8x6.csv")
        self.assertAlmostEqual(hm.dx_um, 0.344174, places=9)
        self.assertAlmostEqual(hm.dy_um, 0.344174, places=9)

    def test_pitch_given_in_um(self):
        """Some exports state the pitch in micrometres rather than nanometres."""
        hm = parse_vk_csv(DATA / "vk_pitch_um_8x6.csv")
        self.assertAlmostEqual(hm.dx_um, 0.344174, places=9)
        self.assertAlmostEqual(hm.dy_um, 0.344174, places=9)
        self.assertEqual(hm.metadata["xy_calibration_raw"], ["0.344174", "um"])

    def test_nm_and_um_exports_describe_the_same_surface(self):
        """vk_units_nm is one measurement restated in nanometres.

        Both fixtures hold the identical surface, so after unit normalisation
        they must yield the same HeightMap -- not merely a larger one.  A test
        that only checks "nm values are bigger" would pass on a fixture that
        silently misreads the unit.
        """
        hm_nm = parse_vk_csv(DATA / "vk_units_nm_8x6.csv")
        hm_um = parse_vk_csv(DATA / "vk_gbk_8x6.csv")
        self.assertEqual(hm_nm.metadata["z_scale_to_um"], 1e-3)
        self.assertEqual(hm_um.metadata["z_scale_to_um"], 1.0)
        self.assertEqual(hm_nm.metadata["data_name"], hm_um.metadata["data_name"])
        np.testing.assert_allclose(hm_nm.z, hm_um.z, rtol=0, atol=1e-12)
        self.assertEqual(hm_nm.dx_um, hm_um.dx_um)

    def test_unknown_height_unit_fails(self):
        with self.assertRaises(VkCsvError):
            parse_vk_csv(DATA / "vk_bad_unit.csv")


class TestValidation(unittest.TestCase):
    def test_missing_xy_calibration_fails(self):
        with self.assertRaises(VkCsvError):
            parse_vk_csv(DATA / "vk_missing_xy.csv")

    def test_matrix_size_mismatch_fails(self):
        with self.assertRaises(VkCsvError) as ctx:
            parse_vk_csv(DATA / "vk_size_mismatch.csv")
        self.assertIn("matrix is", str(ctx.exception))

    def test_missing_file_fails(self):
        with self.assertRaises(VkCsvError):
            parse_vk_csv(DATA / "does_not_exist.csv")


class TestMaskHonesty(unittest.TestCase):
    def test_mask_is_marked_unavailable(self):
        hm = parse_vk_csv(DATA / "vk_gbk_8x6.csv")
        self.assertEqual(hm.metadata["mask_source"], "unavailable")
        self.assertTrue(hm.metadata["mask_is_fabricated"])
        self.assertTrue(hm.mask_is_fabricated)

    def test_zero_is_not_treated_as_invalid(self):
        """A height of exactly 0.000 um is real data in this format."""
        hm = parse_vk_csv(DATA / "vk_gbk_8x6.csv")
        zeros = np.count_nonzero(hm.z == 0.0)
        self.assertTrue(hm.valid_mask.all())
        self.assertEqual(int(hm.valid_mask[hm.z == 0.0].sum()), zeros)

    def test_result_is_a_valid_height_map(self):
        hm = parse_vk_csv(DATA / "vk_gbk_8x6.csv")
        self.assertIsInstance(hm, HeightMap)
        self.assertEqual(hm.n_invalid, 0)


if __name__ == "__main__":
    unittest.main()
