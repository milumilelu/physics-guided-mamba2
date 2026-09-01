import unittest

import numpy as np

from src.data_contracts import HeightMap
from src.leveling import fit_outer_reference_plane


class TestOuterPlane(unittest.TestCase):
    def test_recovers_plane_while_ignoring_central_depression(self):
        height, width, pitch = 220, 260, 2.0
        x = (np.arange(width) + 0.5) * pitch
        y = (np.arange(height) + 0.5) * pitch
        xc = x - width * pitch / 2
        yc = y - height * pitch / 2
        z = 0.012 * xc[None, :] - 0.008 * yc[:, None] + 30.0
        z[(np.abs(yc[:, None]) < 80) & (np.abs(xc[None, :]) < 80)] -= 12
        mask = np.ones_like(z, dtype=bool)
        hm = HeightMap(z, mask, pitch, pitch, x, y, {})
        fit = fit_outer_reference_plane(hm, frame_width_um=40)
        self.assertEqual(fit.status, "PASS")
        self.assertAlmostEqual(fit.a, 0.012, places=6)
        self.assertAlmostEqual(fit.b, -0.008, places=6)
        self.assertAlmostEqual(fit.c, 30.0, places=6)


if __name__ == "__main__":
    unittest.main()
