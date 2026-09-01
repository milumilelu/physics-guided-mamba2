import unittest

import numpy as np
from scipy import ndimage

from src.data_contracts import HeightMap
from src.step_contrast_bootstrap import _pair_curve, fit_step_contrast_bootstrap


class TestStepContrastBootstrap(unittest.TestCase):
    def test_vectorized_curve_matches_inclusive_interval_definition(self):
        profile=np.array([4,4,3,1,1,1,2,5,5],dtype=float)
        axis=np.arange(-4,5,dtype=float)
        candidates=np.array([-1.0,0.0,1.0])
        score,first,second=_pair_curve(
            profile,axis,candidates,half=2,bandwidth=1.5,gap=0.25,scale=2)
        expected_first=[]; expected_second=[]
        for delta in candidates:
            neg=delta-2; pos=delta+2
            median=lambda lo,hi: np.median(profile[(axis>=lo)&(axis<=hi)])
            expected_first.append(max((median(neg-1.5,neg-.25)-median(neg+.25,neg+1.5))/2,0))
            expected_second.append(max((median(pos+.25,pos+1.5)-median(pos-1.5,pos-.25))/2,0))
        np.testing.assert_allclose(first,expected_first)
        np.testing.assert_allclose(second,expected_second)
        np.testing.assert_allclose(score,np.array(expected_first)+expected_second)

    def test_vectorized_curve_matches_on_noninteger_axis(self):
        rng=np.random.default_rng(17)
        axis=np.linspace(-15.2,15.7,94)
        profile=rng.normal(size=axis.size)
        candidates=np.arange(-3,3.01,.25)
        _,first,second=_pair_curve(
            profile,axis,candidates,half=8.2,bandwidth=3.7,gap=.61,scale=1.3)
        scalar_first=[]; scalar_second=[]
        for delta in candidates:
            neg=delta-8.2; pos=delta+8.2
            median=lambda lo,hi: np.median(profile[(axis>=lo)&(axis<=hi)])
            scalar_first.append(max((median(neg-3.7,neg-.61)-median(neg+.61,neg+3.7))/1.3,0))
            scalar_second.append(max((median(pos+.61,pos+3.7)-median(pos-3.7,pos-.61))/1.3,0))
        np.testing.assert_allclose(first,scalar_first)
        np.testing.assert_allclose(second,scalar_second)

    def test_recovers_very_shallow_diffuse_square(self):
        rng=np.random.default_rng(11)
        x=np.arange(501,dtype=float); y=np.arange(401,dtype=float)
        xc,yc=x-250.5,y-200.5; xx,yy=np.meshgrid(xc,yc)
        center=(8.0,-5.5); theta_deg=-0.9; theta=np.deg2rad(theta_deg)
        u=(xx-center[0])*np.cos(theta)+(yy-center[1])*np.sin(theta)
        v=-(xx-center[0])*np.sin(theta)+(yy-center[1])*np.cos(theta)
        inside=(np.abs(u)<=100)&(np.abs(v)<=100)
        z=0.001*xx-0.0015*yy-0.45*inside
        z-=0.25*((u>20)&(v<-15)&inside)
        z+=rng.normal(0,0.07,z.shape)
        z=ndimage.gaussian_filter(z,1.3)
        hm=HeightMap(z=z,valid_mask=np.ones_like(z,bool),dx_um=1,dy_um=1,
                     x_um=x,y_um=y,metadata={})
        fit=fit_step_contrast_bootstrap(
            hm,plane=(0.001,-0.0015,0),theta_deg=theta_deg,
            center_search=(-50,50,-50,50),local_canvas_um=260,
            bootstrap_replicates_per_band=8,random_seed=4)
        self.assertAlmostEqual(fit.center_x_um,center[0],delta=1.5)
        self.assertAlmostEqual(fit.center_y_um,center[1],delta=1.5)


if __name__=="__main__":
    unittest.main()
