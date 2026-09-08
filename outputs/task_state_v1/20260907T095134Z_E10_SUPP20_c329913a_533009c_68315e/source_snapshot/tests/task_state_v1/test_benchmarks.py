import tempfile
from pathlib import Path
import unittest
import numpy as np
from src.task_state_learning.benchmarks import b1_reference,b1_rhs,sample_b1_families,terminal_arrays,load_terminal_only


class B1ReferenceTests(unittest.TestCase):
    def test_zero_control_is_zero_state(self):
        np.testing.assert_array_equal(b1_reference(np.zeros((2,8,2)),np.full((2,8),.03)),0.)

    def test_null_resolved_dynamics_ignore_hidden_state(self):
        a=np.array([[.3,.2,0.,0.]])
        b=np.array([[.3,.2,4.,-7.]])
        np.testing.assert_array_equal(b1_rhs(a,[[.5,-.2]],False)[:,:2],b1_rhs(b,[[.5,-.2]],False)[:,:2])
        self.assertFalse(np.allclose(b1_rhs(a,[[.5,-.2]],True)[:,:2],b1_rhs(b,[[.5,-.2]],True)[:,:2]))

    def test_same_control_function_retokenization_preserves_reference(self):
        u=np.array([[[.2,-.4],[.7,.1]]])
        dt=np.array([[.02,.04]])
        a=b1_reference(u,dt)
        b=b1_reference(np.repeat(u,2,axis=1),np.repeat(dt/2,2,axis=1))
        np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-14)

    def test_terminal_loader_rejects_hidden_state(self):
        f=sample_b1_families(2,np.random.SeedSequence(17))
        values=terminal_arrays(f,np.zeros((2,4)),['a','b'])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'data.npz'
            np.savez(path,**values)
            self.assertEqual(set(load_terminal_only(path)),set(values))
            np.savez(path,**values,true_q=np.zeros((2,2)))
            with self.assertRaisesRegex(ValueError,'forbidden'):
                load_terminal_only(path)


if __name__=='__main__':unittest.main()
