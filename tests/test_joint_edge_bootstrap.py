import unittest

import numpy as np
from scipy import ndimage

from src.data_contracts import HeightMap
from src.joint_edge_bootstrap import fit_joint_edge_bootstrap


class TestJointEdgeBootstrap(unittest.TestCase):
    def _map(self):
        rng = np.random.default_rng(7)
        x = np.arange(501, dtype=float)
        y = np.arange(401, dtype=float)
        xc, yc = x-250.5, y-200.5
        xx, yy = np.meshgrid(xc, yc)
        center = (9.5, -6.0)
        theta_deg = -0.8
        theta = np.deg2rad(theta_deg)
        u = (xx-center[0])*np.cos(theta)+(yy-center[1])*np.sin(theta)
        v = -(xx-center[0])*np.sin(theta)+(yy-center[1])*np.cos(theta)
        inside = (np.abs(u)<=100)&(np.abs(v)<=100)
        z = 0.001*xx-0.002*yy-1.2*inside
        z -= 0.7*((u>25)&(v<0)&inside)
        z += rng.normal(0, 0.08, z.shape)
        z = ndimage.gaussian_filter(z, 0.7)
        hm = HeightMap(z=z, valid_mask=np.ones_like(z, bool), dx_um=1,
                       dy_um=1, x_um=x, y_um=y, metadata={})
        return hm, center, theta_deg

    def test_recovers_shallow_asymmetric_square(self):
        hm, center, theta = self._map()
        fit = fit_joint_edge_bootstrap(
            hm, plane=(0.001, -0.002, 0), theta_deg=theta,
            center_search=(-50, 50, -50, 50), local_canvas_um=260,
            bootstrap_replicates_per_scale=16, random_seed=123)
        self.assertAlmostEqual(fit.center_x_um, center[0], delta=1.5)
        self.assertAlmostEqual(fit.center_y_um, center[1], delta=1.5)

    def test_fixed_seed_is_deterministic(self):
        hm, _, theta = self._map()
        kwargs = dict(plane=(0.001, -0.002, 0), theta_deg=theta,
                      center_search=(-50, 50, -50, 50), local_canvas_um=260,
                      bootstrap_replicates_per_scale=8, random_seed=99)
        first = fit_joint_edge_bootstrap(hm, **kwargs)
        second = fit_joint_edge_bootstrap(hm, **kwargs)
        self.assertEqual(first.to_dict(), second.to_dict())


if __name__ == "__main__":
    unittest.main()
