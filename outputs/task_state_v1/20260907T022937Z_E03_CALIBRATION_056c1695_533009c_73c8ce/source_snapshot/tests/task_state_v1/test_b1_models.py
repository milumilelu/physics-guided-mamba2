import unittest
import numpy as np
import torch
from src.task_state_learning.models import B1ClosureModel
from src.task_state_learning.benchmarks import b1_reference,sample_b1_families,terminal_arrays


class B1ModelTests(unittest.TestCase):
    def setUp(self):
        self.f=sample_b1_families(2,np.random.SeedSequence(55),lengths=(32,))
        data=terminal_arrays(self.f,np.zeros((2,4)),['a','b'])
        self.u=torch.tensor(data['control'],dtype=torch.float64)
        self.dt=torch.tensor(data['physical_dt'],dtype=torch.float64)
        self.mask=torch.tensor(data['mask'])

    def test_zero_closure_recovers_independent_null(self):
        expected=b1_reference(self.f['controls'],self.f['block_duration'],coupling=False)[:,:2]
        for name in ('H0','GRU','CTSSM'):
            m=B1ClosureModel(name).double()
            np.testing.assert_allclose(m(self.u,self.dt,self.mask).detach(),expected,rtol=1e-9,atol=1e-11)

    def test_reset_mask_and_continuation(self):
        for name in ('H0','GRU','CTSSM'):
            torch.manual_seed(17)
            m=B1ClosureModel(name).double()
            torch.nn.init.normal_(m.readout[-1].weight,std=.01)
            direct=m(self.u,self.dt,self.mask)
            torch.testing.assert_close(m(self.u,self.dt,self.mask),direct,rtol=0,atol=0)
            state=m(self.u[:,:16],self.dt[:,:16],self.mask[:,:16],return_state=True)
            resumed=m(self.u[:,16:],self.dt[:,16:],self.mask[:,16:],initial_state=state)
            torch.testing.assert_close(resumed,direct,rtol=0,atol=0)
            padded=m(torch.cat((self.u,torch.ones_like(self.u)),1),torch.cat((self.dt,self.dt),1),
                     torch.cat((self.mask,torch.zeros_like(self.mask)),1))
            torch.testing.assert_close(padded,direct,rtol=0,atol=0)
            direct.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_parameter_and_cache_budgets(self):
        budgets=[B1ClosureModel(name).budget() for name in ('H0','GRU','CTSSM')]
        counts=[b['n_trainable_parameters'] for b in budgets]
        self.assertLess(max(counts)/min(counts),1.02)
        self.assertEqual(budgets[0]['n_ssm_or_recurrent_cache_scalars'],0)
        self.assertEqual(budgets[1]['persistent_state_bytes'],136)


if __name__=='__main__':unittest.main()
