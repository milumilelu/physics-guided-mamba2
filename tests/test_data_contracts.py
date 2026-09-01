"""The HeightMap contract is the guardrail for the whole pipeline."""

from __future__ import annotations

import unittest

import numpy as np

from src.data_contracts import ContractViolation, HeightMap


def make(z, mask, **kwargs) -> HeightMap:
    h, w = z.shape
    dx = kwargs.pop("dx_um", 1.0)
    dy = kwargs.pop("dy_um", 1.0)
    return HeightMap(
        z=z,
        valid_mask=mask,
        dx_um=dx,
        dy_um=dy,
        x_um=(np.arange(w) + 0.5) * dx,
        y_um=(np.arange(h) + 0.5) * dy,
        **kwargs,
    )


class TestContract(unittest.TestCase):
    def test_accepts_a_clean_map(self):
        z = np.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.ones_like(z, dtype=bool)
        hm = make(z, mask)
        self.assertEqual(hm.n_valid, 4)
        self.assertEqual(hm.n_invalid, 0)
        self.assertAlmostEqual(hm.valid_fraction, 1.0)

    def test_invalid_must_be_nan(self):
        """A filled value inside a masked-out pixel is the bug we are guarding."""
        z = np.array([[1.0, 2.0], [3.0, 4.0]])
        z[0, 0] = 0.0                      # classic "filled with zero"
        mask = np.ones_like(z, dtype=bool)
        mask[0, 0] = False
        with self.assertRaises(ContractViolation) as ctx:
            make(z, mask)
        self.assertIn("not NaN", str(ctx.exception))

    def test_median_filled_pixel_is_rejected(self):
        z = np.array([[1.0, 2.0], [3.0, 4.0]])
        z[1, 1] = 3.5                      # neighbour-median fill
        mask = np.ones_like(z, dtype=bool)
        mask[1, 1] = False
        with self.assertRaises(ContractViolation):
            make(z, mask)

    def test_valid_zero_height_stays_valid(self):
        """0.000 um is a legitimate measurement, not a missing-data marker."""
        z = np.zeros((2, 2))
        mask = np.ones_like(z, dtype=bool)
        hm = make(z, mask)
        self.assertEqual(hm.n_valid, 4)
        self.assertEqual(hm.n_invalid, 0)
        self.assertEqual(hm.summary()["z_min"], 0.0)

    def test_shape_mismatch_rejected(self):
        z = np.ones((3, 4))
        mask = np.ones((3, 3), dtype=bool)
        with self.assertRaises(ContractViolation):
            HeightMap(z=z, valid_mask=mask, dx_um=1.0, dy_um=1.0,
                      x_um=np.arange(4), y_um=np.arange(3), metadata={})

    def test_non_bool_mask_rejected(self):
        z = np.ones((2, 2))
        with self.assertRaises(ContractViolation):
            HeightMap(z=z, valid_mask=np.ones((2, 2), dtype=np.uint8),
                      dx_um=1.0, dy_um=1.0, x_um=np.arange(2),
                      y_um=np.arange(2), metadata={})

    def test_non_finite_valid_pixel_rejected(self):
        z = np.array([[1.0, np.inf], [3.0, 4.0]])
        mask = np.ones_like(z, dtype=bool)
        with self.assertRaises(ContractViolation):
            make(z, mask)

    def test_positive_pitch_required(self):
        z = np.ones((2, 2))
        mask = np.ones_like(z, dtype=bool)
        for bad in (0.0, -1.0, float("nan")):
            with self.assertRaises(ContractViolation):
                HeightMap(z=z, valid_mask=mask, dx_um=bad, dy_um=1.0,
                          x_um=np.arange(2), y_um=np.arange(2), metadata={})

    def test_coordinate_length_must_match(self):
        z = np.ones((2, 3))
        mask = np.ones_like(z, dtype=bool)
        with self.assertRaises(ContractViolation):
            HeightMap(z=z, valid_mask=mask, dx_um=1.0, dy_um=1.0,
                      x_um=np.arange(2), y_um=np.arange(2), metadata={})

    def test_coordinates_must_increase(self):
        z = np.ones((2, 2))
        mask = np.ones_like(z, dtype=bool)
        with self.assertRaises(ContractViolation):
            HeightMap(z=z, valid_mask=mask, dx_um=1.0, dy_um=1.0,
                      x_um=np.array([0.0, -1.0]), y_um=np.arange(2),
                      metadata={})

    def test_fabricated_mask_is_flagged(self):
        z = np.ones((2, 2))
        mask = np.ones_like(z, dtype=bool)
        hm = make(z, mask, metadata={"mask_is_fabricated": True})
        self.assertTrue(hm.mask_is_fabricated)

    def test_summary_reports_the_mask_honestly(self):
        z = np.array([[1.0, np.nan], [3.0, 4.0]])
        mask = np.isfinite(z)
        hm = make(z, mask, dx_um=0.344174, dy_um=0.344174)
        summary = hm.summary()
        self.assertEqual(summary["n_invalid"], 1)
        self.assertAlmostEqual(summary["valid_fraction"], 0.75)
        self.assertFalse(summary["mask_is_fabricated"])


if __name__ == "__main__":
    unittest.main()
