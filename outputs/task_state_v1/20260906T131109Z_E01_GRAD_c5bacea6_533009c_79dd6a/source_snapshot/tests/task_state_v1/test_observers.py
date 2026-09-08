import unittest
import numpy as np
from src.task_state_learning.observers import CanonicalObserver


class ObserverTests(unittest.TestCase):
    def test_constant_has_valid_depth_amplitude_but_undefined_spectrum(self):
        h = np.full((1,160,160), -3.)
        y,v = CanonicalObserver()(h, np.ones_like(h, dtype=bool))
        self.assertEqual(y.D.iloc[0], 3)
        self.assertEqual(y.A_med.iloc[0], 0)
        self.assertTrue(y.ilr_z1.isna().all())
        self.assertFalse(v.band_valid.any())
        self.assertTrue(y.A2_8_16.isna().all())
        self.assertTrue(y.entropy_8_16.isna().all())

    def test_even_median_not_lower_order_statistic(self):
        h = np.zeros((1,160,160)); h[:,80:,:] = -4
        y,_ = CanonicalObserver()(h,np.ones_like(h,dtype=bool))
        self.assertEqual(y.D.iloc[0],2.)
        self.assertEqual(y.A_med.iloc[0],2.)

    def test_hole_not_silently_interpolated(self):
        h = np.ones((1,160,160)); v = np.ones_like(h,dtype=bool); v[0,0,0] = False
        y,d = CanonicalObserver()(h,v)
        self.assertEqual(y.D.iloc[0],-1.)
        self.assertTrue(y.ilr_z1.isna().all())
        self.assertIn('INCOMPLETE_GRID',d.undefined_reason.iloc[0])

    def test_axis_rotation_preserves_direction_strength(self):
        x = np.arange(160)*.5
        h = np.broadcast_to(np.cos(2*np.pi*x/10), (160,160)).copy()
        a=np.stack([h,h.T]); y,_=CanonicalObserver()(a,np.ones_like(a,dtype=bool))
        np.testing.assert_allclose(y.A2_8_16, y.A2_8_16.iloc[0], atol=1e-12)
        self.assertGreater(y.A2_8_16.iloc[0],.95)

    def test_nonfinite_valid_pixel_fails_closed(self):
        h=np.ones((1,160,160)); h[0,4,5]=np.nan
        y,v=CanonicalObserver()(h,np.ones_like(h,dtype=bool))
        self.assertTrue(y.D.isna().all())
        self.assertFalse(v.band_valid.any())


if __name__ == '__main__':
    unittest.main()
