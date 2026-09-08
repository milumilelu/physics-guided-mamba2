"""Optional in historical environment; mandatory in the E01 autograd runner."""
import importlib.util
import unittest
import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from src.task_state_learning.differentiable_observer import DifferentiableObserver, TARGET_NAMES
    from src.task_state_learning.observers import CanonicalObserver


@unittest.skipUnless(HAS_TORCH,"PyTorch belongs to the separate algorithm environment")
class DifferentiableObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.observer=DifferentiableObserver()

    def test_constant_keeps_spectral_values_undefined(self):
        h=torch.full((1,160,160),-3.,dtype=torch.float64,requires_grad=True)
        result=self.observer(h)
        self.assertEqual(result.values[0,0],3.)
        self.assertEqual(result.values[0,1],0.)
        self.assertFalse(result.valid[0,2:].any())
        self.assertTrue(result.values[0,2:].isnan().all())
        result.values[0,:2].sum().backward()
        self.assertTrue(h.grad.isfinite().all())

    def test_even_median(self):
        h=torch.zeros((1,160,160),dtype=torch.float64);h[:,80:]=-4.
        y=self.observer(h)
        np.testing.assert_array_equal(y.values[0,:2].numpy(),[2.,2.])

    def test_nonfinite_and_missing_grid_fail_closed(self):
        h=torch.ones((2,160,160),dtype=torch.float64)
        h[0,0,0]=float('nan');mask=torch.ones_like(h,dtype=torch.bool);mask[1,1,1]=False
        out=self.observer(h,mask)
        self.assertFalse(out.valid[0].any())
        self.assertTrue(out.valid[1,:2].all())
        self.assertFalse(out.valid[1,2:].any())

    def test_canonical_random_field_and_18_bin_sensitivity(self):
        h=np.random.default_rng(11).normal(size=(2,160,160))
        for bins in [36,18]:
            expected,_=CanonicalObserver(theta_bins=bins)(h,np.ones_like(h,dtype=bool))
            actual=DifferentiableObserver(theta_bins=bins)(torch.tensor(h))
            np.testing.assert_allclose(actual.values.detach().numpy(),expected[list(TARGET_NAMES)].to_numpy(),atol=1e-8,rtol=1e-8)

    def test_first_and_second_derivative_directional_checks(self):
        rng=np.random.default_rng(19)
        base=torch.tensor(rng.normal(size=(1,160,160)),dtype=torch.float64)
        directions=torch.tensor(rng.normal(size=(3,1,160,160)),dtype=torch.float64)
        theta=torch.tensor([.17,-.11,.23],dtype=torch.float64,requires_grad=True)
        def function(t):
            return self.observer(base+(t[:,None,None,None]*directions).sum(0)).values
        self.assertTrue(torch.autograd.gradcheck(function,(theta,),eps=1e-6,atol=1e-5,rtol=1e-3))
        self.assertTrue(torch.autograd.gradgradcheck(function,(theta,),eps=1e-6,atol=1e-5,rtol=1e-3))

    def test_float32_input_retains_gradient_with_float64_accumulation(self):
        h=torch.tensor(np.random.default_rng(8).normal(size=(1,160,160)),dtype=torch.float32,requires_grad=True)
        y=self.observer(h)
        self.assertEqual(y.values.dtype,torch.float64)
        y.values.square().sum().backward()
        self.assertEqual(h.grad.dtype,torch.float32)
        self.assertTrue(h.grad.isfinite().all())


if __name__=='__main__':unittest.main()
