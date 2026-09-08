import unittest
from dataclasses import replace
import numpy as np
from src.task_state_learning.physics import PhysicsParameters,step
from src.task_state_learning.finite_scan import FiniteScanConfig,simulate_points,simulate_roi


P=PhysicsParameters(.874e-6,1030e-9,1.2,1e4,1e-8,.2,.1)


class FiniteScanTests(unittest.TestCase):
    def test_all_switches_match_full_finite_pulse_train(self):
        cfg=FiniteScanConfig(scan_region_m=80e-6,phase_x_pitch=.25,phase_y_pitch=-.125,tail_waists=8.)
        x=np.array([-40.,-39.5,-20.25,-.25,.25,20.25,39.5,40.])*1e-6
        y=np.array([-39.5,-38.,-1.,2.,6.,14.,30.,39.5])*1e-6
        dx,h,f=2e-6,8e-6,1e5
        kw=dict(tau_s=1e-12,f_Hz=f,v_m_s=f*dx,h_m=h,passes=2,power_W=5.3333)
        nx,ny=40,10
        ox=-(nx-1)*dx/2+cfg.phase_x_pitch*dx
        oy=-(ny-1)*h/2+cfg.phase_y_pitch*h
        for switch in ('P00','P10','P01','P11'):
            d,q=np.zeros_like(x),np.zeros_like(x)
            for repeat in range(2):
                for j in range(ny):
                    for k in range(nx):
                        d,q=step(d,q,x,y,pulse_energy_J=kw['power_W']/f,duration_s=1e-12,
                                 beam_x_m=ox+k*dx,beam_y_m=oy+j*h,parameters=P,switch=switch)
            actual,aq,info=simulate_points(P,x,y,cfg=cfg,switch=switch,**kw)
            np.testing.assert_allclose(actual,d,rtol=1e-10,atol=1e-18)
            np.testing.assert_allclose(aq,q,rtol=1e-10,atol=1e-15)
            self.assertEqual(info['physical_pulses'],800)

    def test_finite_boundaries_are_not_periodic(self):
        cfg=FiniteScanConfig(scan_region_m=80e-6)
        d,_,_=simulate_points(P,np.array([0.,80e-6]),np.zeros(2),cfg=cfg,
                             tau_s=1e-12,f_Hz=1e5,v_m_s=.01,h_m=2e-6,
                             passes=1,power_W=5.3333,switch='P01')
        self.assertGreater(d[0],0.)
        self.assertEqual(d[1],0.)

    def test_observation_grid_and_sign(self):
        h,q,info=simulate_roi(P,tau_s=1e-12,f_Hz=1e4,v_m_s=.1,h_m=10e-6,
                            passes=1,power_W=5.3333,switch='P00')
        self.assertEqual(h.shape,(160,160))
        self.assertTrue((h<=0).all())
        self.assertTrue((q==0).all())
        self.assertTrue(info['finite_region_simulated'])
        self.assertFalse(info['tail_convergence_validated'])

    def test_budget_failure_does_not_return_zero_field(self):
        with self.assertRaisesRegex(ValueError,'budget'):
            simulate_points(P,[0.],[0.],cfg=FiniteScanConfig(max_updates_per_point=1),
                            tau_s=1e-12,f_Hz=1e5,v_m_s=.01,h_m=2e-6,
                            passes=1,power_W=5.3333,switch='P11')

    def test_noninteger_passes_rejected(self):
        with self.assertRaisesRegex(ValueError,'Invalid recipe'):
            simulate_points(P,[0.],[0.],tau_s=1e-12,f_Hz=1e5,v_m_s=.01,h_m=2e-6,
                            passes=1.5,power_W=5.3333,switch='P11')


if __name__=='__main__':unittest.main()
