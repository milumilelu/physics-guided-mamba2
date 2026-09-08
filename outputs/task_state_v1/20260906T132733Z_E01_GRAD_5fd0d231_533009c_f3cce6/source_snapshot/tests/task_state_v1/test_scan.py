"""Scan fixture contract tests: layout determinism, budgets, backend parity."""
import importlib.util
import unittest
import numpy as np

from src.task_state_learning.scan import (ScanFixture, descriptors, packet_rollout,
                                          pulse_counts, pulse_positions, rollout_numpy,
                                          split_packets)
from src.task_state_learning.physics import PhysicsParameters

FIXTURE = ScanFixture(region_m=2.0e-5, grid_n=24, power_W=5.3333,
                      n_x_max=8, n_y_max=6, total_pulses_max=48)
PARAMS = PhysicsParameters(waist_m=2.0e-6, wavelength_m=1.03e-6, m_squared=1.2,
                           F1_reference_J_m2=10000.0, delta_reference_m=1.0e-9,
                           kappa=0.01, saturation_ratio=0.5,
                           gamma_F=0.25, gamma_delta=0.10, reference_duration_s=1.0e-12)

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from src.task_state_learning.scan import rollout_torch


class ScanLayoutTests(unittest.TestCase):
    def test_layout_row_major_centered_and_counts(self):
        n_x, n_y = pulse_counts(FIXTURE, 5e-6, 7e-6)
        self.assertEqual((n_x, n_y), (4, 3))
        pos = pulse_positions(n_x, n_y, 5e-6, 7e-6)
        self.assertEqual(pos.shape, (12, 2))
        np.testing.assert_allclose(pos[:4, 0], [-7.5e-6, -2.5e-6, 2.5e-6, 7.5e-6])
        self.assertTrue(np.allclose(pos[:4, 1], pos[0, 1]))
        self.assertTrue(np.allclose(pos[4:8, 1], pos[4, 1]))

    def test_budget_rejected(self):
        with self.assertRaises(ValueError):
            pulse_counts(FIXTURE, 5e-7, 7e-6)
        with self.assertRaises(ValueError):
            pulse_counts(FIXTURE, 5e-6, 1e-6)

    def test_zero_energy_identity(self):
        pos = pulse_positions(3, 2, 6e-6, 8e-6)
        d, q = rollout_numpy(pos, np.zeros(6), 1e-12, PARAMS, FIXTURE, switch='P11')
        np.testing.assert_array_equal(d, 0.0)
        np.testing.assert_array_equal(q, 0.0)

    def test_subpulse_split_preserves_total_energy(self):
        pos, energy = split_packets(np.full((3, 2), 1e-6), 2e-4, 4)
        self.assertEqual(len(pos), 12)
        self.assertAlmostEqual(energy.sum(), 3 * 2e-4, places=20)
        np.testing.assert_array_equal(pos[:4], np.tile(pos[0], (4, 1)))

    def test_packet_multiplicity_one_matches_exact(self):
        pos = pulse_positions(2, 2, 6e-6, 8e-6)
        energy = np.full(4, 5e-4)
        exact, q_exact = rollout_numpy(pos, energy, 1e-12, PARAMS, FIXTURE)
        approx, q_approx = packet_rollout(pos, energy, 1e-12, PARAMS, FIXTURE, packet_size=1)
        np.testing.assert_allclose(approx, exact, rtol=0, atol=0)
        np.testing.assert_allclose(q_approx, q_exact, rtol=0, atol=0)

    def test_packet_run_grouping_fixed_pulse_train(self):
        pos = pulse_positions(3, 2, 6e-6, 8e-6)
        energy = np.full(6, 5e-4)
        d1, q1 = packet_rollout(pos, energy, 1e-12, PARAMS, FIXTURE, packet_size=4)
        d2, q2 = packet_rollout(pos, energy, 1e-12, PARAMS, FIXTURE, packet_size=1)
        self.assertGreaterEqual(float(d1.mean()), float(d2.mean()) * 0.999)
        self.assertTrue(np.isfinite(d1).all() and np.isfinite(q1).all())

    def test_descriptors_finite(self):
        pos = pulse_positions(3, 2, 6e-6, 8e-6)
        d, _ = rollout_numpy(pos, np.full(6, 5e-4), 1e-12, PARAMS, FIXTURE)
        desc = descriptors(d)
        self.assertTrue(np.isfinite(list(desc.values())).all())
        self.assertGreater(desc['mean_depth_m'], 0.0)


@unittest.skipUnless(HAS_TORCH, "PyTorch belongs to the algorithm environment")
class BackendParityTests(unittest.TestCase):
    def test_torch_numpy_rollout_parity(self):
        torch.set_default_dtype(torch.float64)
        ep_J, tau_s, dx_m, hatch_m = 5e-4, 1e-12, 6e-6, 8e-6
        f_Hz = FIXTURE.power_W / ep_J
        pos = pulse_positions(3, 2, dx_m, hatch_m)
        d_ref, q_ref = rollout_numpy(pos, np.full(6, ep_J), tau_s, PARAMS, FIXTURE)
        logs = [torch.tensor(v, requires_grad=True) for v in
                (np.log(tau_s), np.log(f_Hz), np.log(dx_m * f_Hz), np.log(hatch_m))]
        d_t, q_t = rollout_torch(*logs, PARAMS, FIXTURE, 3, 2)
        np.testing.assert_allclose(d_t.detach().numpy(), d_ref, rtol=1e-9, atol=1e-18)
        np.testing.assert_allclose(q_t.detach().numpy(), q_ref, rtol=1e-9, atol=1e-12)
        J = d_t.mean()
        J.backward()
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in
                            (logs[0].grad, logs[1].grad, logs[2].grad, logs[3].grad)))


if __name__ == '__main__':
    unittest.main()
