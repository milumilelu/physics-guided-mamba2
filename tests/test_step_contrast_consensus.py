import unittest

import numpy as np
from scipy import ndimage

from src.data_contracts import HeightMap
from src.step_contrast_consensus import (
    _estimate_balanced_axis,
    fit_step_contrast_consensus,
)


class TestStepContrastConsensus(unittest.TestCase):
    def test_balanced_objective_rejects_single_side_dominance(self):
        candidates=np.array([-1.0,0.0,1.0])
        first=np.tile(np.array([1.0,8.0,100.0]),(2,3,1))
        second=np.tile(np.array([1.0,8.0,1.0]),(2,3,1))
        centre,influence=_estimate_balanced_axis(first,second,candidates)
        self.assertEqual(centre,0.0)
        np.testing.assert_array_equal(influence,0.0)

    def test_recovers_offcenter_asymmetric_square(self):
        rng=np.random.default_rng(23)
        x=np.arange(561,dtype=float); y=np.arange(441,dtype=float)
        xx,yy=np.meshgrid(x-280.5,y-220.5)
        center=(-74.0,13.5); theta_deg=-.7; theta=np.deg2rad(theta_deg)
        u=(xx-center[0])*np.cos(theta)+(yy-center[1])*np.sin(theta)
        v=-(xx-center[0])*np.sin(theta)+(yy-center[1])*np.cos(theta)
        inside=(np.abs(u)<=100)&(np.abs(v)<=100)
        z=.001*xx-.0015*yy-.5*inside
        z-=.25*((u>10)&(v<20)&inside)
        z+=.13*np.sin(2*np.pi*u/13)*inside
        z+=rng.normal(0,.07,z.shape)
        z=ndimage.gaussian_filter(z,.8)
        hm=HeightMap(z=z,valid_mask=np.ones_like(z,bool),dx_um=1,dy_um=1,
                     x_um=x,y_um=y,metadata={})
        fit=fit_step_contrast_consensus(
            hm,plane=(.001,-.0015,0),theta_deg=theta_deg,
            center_search=(-160,160,-100,100))
        self.assertAlmostEqual(fit.center_x_um,center[0],delta=1.5)
        self.assertAlmostEqual(fit.center_y_um,center[1],delta=1.5)
        self.assertFalse(fit.local_search_boundary_hit)


if __name__=="__main__":
    unittest.main()
