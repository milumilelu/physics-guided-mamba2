import unittest

import numpy as np

from src.data_contracts import HeightMap
from src.resampling import resample_center_crop, resample_to_canonical


class TestMaskAwareResampling(unittest.TestCase):
    def test_identity_plane_and_translation(self):
        x = np.arange(101, dtype=float)
        y = np.arange(81, dtype=float)
        xc, yc = x-50.5, y-40.5
        z = xc[None, :]+2*yc[:, None]
        hm = HeightMap(z=z, valid_mask=np.ones_like(z, bool), dx_um=1,
                       dy_um=1, x_um=x, y_um=y, metadata={})
        out = resample_to_canonical(
            hm, plane=(1, 2, 0), center_x_um=0, center_y_um=0,
            theta_deg=0, length_um=40, pixels=40)
        self.assertTrue(out.valid_mask.all())
        self.assertLess(np.max(np.abs(out.z)), 1e-10)

    def test_mask_is_not_filled_across_a_hole(self):
        x = np.arange(101, dtype=float)
        y = np.arange(101, dtype=float)
        mask = np.ones((101, 101), bool)
        mask[45:56, 45:56] = False
        z = np.zeros(mask.shape, float)
        z[~mask] = np.nan
        hm = HeightMap(z=z, valid_mask=mask, dx_um=1, dy_um=1,
                       x_um=x, y_um=y, metadata={})
        out = resample_to_canonical(
            hm, plane=(0, 0, 0), center_x_um=0, center_y_um=0,
            theta_deg=0, length_um=40, pixels=40)
        self.assertFalse(out.valid_mask[20, 20])
        self.assertTrue(np.isnan(out.z[20, 20]))

    def test_exact_h200_physical_size(self):
        axis = (np.arange(300)+0.5)-150
        z = np.zeros((300, 300))
        hm = HeightMap(z=z, valid_mask=np.ones_like(z, bool), dx_um=1,
                       dy_um=1, x_um=axis, y_um=axis, metadata={})
        out = resample_center_crop(hm, length_um=200, pixels=200)
        self.assertAlmostEqual(out.width_um, 200.0)


if __name__ == "__main__":
    unittest.main()
