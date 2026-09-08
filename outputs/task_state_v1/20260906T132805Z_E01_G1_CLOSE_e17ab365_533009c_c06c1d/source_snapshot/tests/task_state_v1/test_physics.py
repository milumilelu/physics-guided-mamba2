import unittest
import numpy as np
from src.task_state_learning.physics import PhysicsParameters,step,repeat_identical_pulse,positive_part


def parameters():
    return PhysicsParameters(2e-6,1030e-9,1.2,1e4,1e-9,.01,.5)


class PhysicsTests(unittest.TestCase):
    def kwargs(self):
        p=parameters()
        return dict(parameters=p,pulse_energy_J=np.pi*p.waist_m**2*p.F1_reference_J_m2,
                    duration_s=1e-12,beam_x_m=0.,beam_y_m=0.)

    def test_zero_exposure_exact_identity(self):
        kw=self.kwargs();kw['pulse_energy_J']=0
        d,q=step(np.array([1e-6]),np.array([.4]),np.array([0.]),np.array([0.]),**kw)
        np.testing.assert_array_equal(d,[1e-6]);np.testing.assert_array_equal(q,[.4])

    def test_single_pulse_analytic_center(self):
        d,q=step(np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),switch='P00',**self.kwargs())
        self.assertAlmostEqual(d[0]/(1e-9*np.log(2)),1.,places=13)
        self.assertEqual(q[0],0.)

    def test_subthreshold_smoothing_has_no_tail(self):
        np.testing.assert_array_equal(positive_part(np.array([-np.inf,-1.,0.]),.02),np.zeros(3))
        kw=self.kwargs();kw['pulse_energy_J']*=.1
        d,_=repeat_identical_pulse(np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),multiplicity=100,
                                    switch='P00',smoothing_epsilon=.02,**kw)
        self.assertEqual(d[0],0.)

    def test_pulse_energy_aggregation_counterexample(self):
        kw=self.kwargs()
        repeated,_=repeat_identical_pulse(np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),multiplicity=3,switch='P00',**kw)
        kw['pulse_energy_J']*=3
        aggregated,_=step(np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),switch='P00',**kw)
        self.assertGreater(abs(repeated[0]-aggregated[0]),1e-10)

    def test_monotonic_and_packet_partition_invariance(self):
        kw=self.kwargs();d=np.zeros(3);q=np.zeros(3);x=np.array([0.,1e-6,3e-6]);y=np.zeros(3)
        a,b=repeat_identical_pulse(d,q,x,y,multiplicity=30,**kw)
        c,e=repeat_identical_pulse(d,q,x,y,multiplicity=10,**kw)
        f,g=repeat_identical_pulse(c,e,x,y,multiplicity=20,**kw)
        np.testing.assert_array_equal(a,f);np.testing.assert_array_equal(b,g)
        self.assertTrue((a>=d).all() and (b>=q).all())

    def test_invalid_closure_rejected(self):
        with self.assertRaisesRegex(ValueError,'Closure bound'):
            step(np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),closure_log=1.,**self.kwargs())

    def test_switch_no_incubation_ignores_q(self):
        kw=self.kwargs();z=np.zeros(1)
        a,_=step(z,z,z,z,switch='P10',**kw)
        b,_=step(z,np.ones(1)*100,z,z,switch='P10',**kw)
        np.testing.assert_array_equal(a,b)


if __name__=='__main__':unittest.main()
