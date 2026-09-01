import unittest

import numpy as np
from scipy import ndimage

from src.data_contracts import HeightMap
from src.registration import register_fixed_square


class TestTranslationRegistration(unittest.TestCase):
    def test_recovers_center_with_frozen_angle(self):
        x = np.arange(501, dtype=float)
        y = np.arange(401, dtype=float)
        xc, yc = x-x[-1]/2, y-y[-1]/2
        xx, yy = np.meshgrid(xc, yc)
        true_center = (13.5, -7.0)
        theta_deg = -1.2
        theta = np.deg2rad(theta_deg)
        u = (xx-true_center[0])*np.cos(theta)+(yy-true_center[1])*np.sin(theta)
        v = -(xx-true_center[0])*np.sin(theta)+(yy-true_center[1])*np.cos(theta)
        z = 0.001*xx-0.002*yy-25*((np.abs(u)<=100)&(np.abs(v)<=100))
        z = ndimage.gaussian_filter(z, 1.0)
        hm = HeightMap(z=z, valid_mask=np.ones_like(z, bool), dx_um=1.0,
                       dy_um=1.0, x_um=x, y_um=y, metadata={})
        fit = register_fixed_square(
            hm, plane=(0.001, -0.002, 0), theta_deg=theta_deg,
            center_search=(-50, 50, -50, 50))
        self.assertEqual(fit.status, "PASS")
        self.assertAlmostEqual(fit.center_x_um, true_center[0], delta=1.0)
        self.assertAlmostEqual(fit.center_y_um, true_center[1], delta=1.0)
        self.assertLessEqual(fit.sensitivity_span_um, 3.0)


if __name__ == "__main__":
    unittest.main()
