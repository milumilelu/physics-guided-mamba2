import unittest

import numpy as np
from scipy import ndimage

from src.data_contracts import HeightMap
from src.session_geometry import fit_free_square, pool_session_angle


class TestFreeSquareFit(unittest.TestCase):
    def test_recovers_rotated_depressed_square(self):
        dx = 1.0
        x = np.arange(501, dtype=float)
        y = np.arange(401, dtype=float)
        xc = x - x[-1] / 2
        yc = y - y[-1] / 2
        xx, yy = np.meshgrid(xc, yc)
        theta = np.deg2rad(2.4)
        u = xx * np.cos(theta) + yy * np.sin(theta)
        v = -xx * np.sin(theta) + yy * np.cos(theta)
        z = 0.002 * xx - 0.001 * yy
        z = z - 20.0 * ((np.abs(u) <= 100) & (np.abs(v) <= 100))
        z = ndimage.gaussian_filter(z, 1.0)
        hm = HeightMap(z=z, valid_mask=np.ones_like(z, dtype=bool), dx_um=dx,
                       dy_um=dx, x_um=x, y_um=y, metadata={})
        fit = fit_free_square(
            hm, plane=(0.002, -0.001, 0.0),
            center_search=(-30, 30, -30, 30), angle_grid_step_deg=0.1)
        self.assertEqual(fit.status, "PASS")
        self.assertAlmostEqual(fit.theta_deg, 2.4, delta=0.25)
        self.assertAlmostEqual(fit.center_x_um, 0.0, delta=2.0)
        self.assertAlmostEqual(fit.center_y_um, 0.0, delta=2.0)

    def test_pooling_reports_large_mad(self):
        rows = [{"status": "PASS", "theta_deg": angle,
                 "quality_score": 1.0} for angle in (-1.2, 0.0, 1.2)]
        result = pool_session_angle(rows, warning_mad_deg=0.3,
                                    review_mad_deg=0.8)
        self.assertEqual(result["status"], "STOP")


if __name__ == "__main__":
    unittest.main()
