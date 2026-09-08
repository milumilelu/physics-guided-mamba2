"""Cell-solver contract tests: analytic single pulse, exact lattice parity vs the
2D reference rollout, dense/sparse path agreement, and the drift control bound."""
import math
from dataclasses import replace
import unittest
import numpy as np

from src.task_state_learning.physics import PhysicsParameters, step
from src.task_state_learning.scan import ScanFixture, rollout_numpy
from src.task_state_learning.scan_cell import CellConfig, simulate_sample

PARAMS = PhysicsParameters(waist_m=0.874e-6, wavelength_m=1030e-9, m_squared=1.2,
                           F1_reference_J_m2=1e4, delta_reference_m=1e-9, kappa=1e-3,
                           saturation_ratio=0.3, gamma_F=0.3, gamma_delta=0.2,
                           reference_duration_s=1000e-15)
CFG = CellConfig()
POWER_W = 5.3333


class SinglePulseAnalyticTests(unittest.TestCase):
    def test_single_pulse_field_matches_analytic(self):
        """Isolated hatch/pulse pitch far outside the 3w window: closed form.

        The cell folds both lattices, so each pixel sees its nearest hatch line at
        min(u, h-u) and its nearest pulse at min(phi, dx-phi)."""
        pitch = 20e-6  # h = dx: one hatch line and one pulse per period
        n_u = int(np.clip(round(pitch / CFG.u_target_m), 8, CFG.n_u_max))
        n_phi = CFG.n_phi
        v_m_s = 50e-3
        f_hz = v_m_s / pitch  # dx = v/f = pitch
        D, A, info = simulate_sample(PARAMS, tau_s=1000e-15, f_Hz=f_hz, v_m_s=v_m_s,
                                     h_m=pitch, passes=1, power_W=POWER_W, switch='P11',
                                     cfg=CFG, region_m=pitch)
        u = (np.arange(n_u) + 0.5) * (pitch / n_u)
        phi = (np.arange(n_phi) + 0.5) * (pitch / n_phi)
        U, PHI = np.meshgrid(u, phi, indexing='ij')
        Ep = POWER_W / f_hz
        w = PARAMS.waist_m
        d_u = np.minimum(U, pitch - U)
        d_phi = np.minimum(PHI, pitch - PHI)
        log_fluence = math.log(2 * Ep / (math.pi * w ** 2)) - 2 * (d_u ** 2 + d_phi ** 2) / w ** 2
        field = PARAMS.delta_reference_m * np.clip(
            log_fluence - math.log(PARAMS.F1_reference_J_m2), 0.0, None)
        self.assertAlmostEqual(D, float(np.median(field)), places=20)
        self.assertAlmostEqual(A, float(np.sqrt(np.mean((field - np.median(field)) ** 2))),
                               places=20)


class LatticeParityTests(unittest.TestCase):
    """Sparse path must reproduce the exact 2D pulse rollout on aligned grids."""

    def test_sparse_path_central_period_parity(self):
        """5-track rollout over 6 hatch periods; compare the central period."""
        h, dx, region, n_grid, n_phi = 6e-6, 1.2e-6, 37.2e-6, 186, 6
        fixture = ScanFixture(region_m=region, grid_n=n_grid, power_W=POWER_W,
                              n_x_max=64, n_y_max=64, total_pulses_max=4096)
        tracks = list(range(-2, 3))  # hatch lines at x = j*h: -12,-6,0,6,12 (um)
        pulses_per_track = 31  # odd count: pulse lattice lands on y = k*dx exactly
        pos = np.array([[j * h, k * dx] for j in tracks
                        for k in range(-(pulses_per_track // 2), pulses_per_track // 2 + 1)])
        energy = np.full(len(pos), POWER_W / 100e3)
        cfg = CellConfig(region_m=region, u_target_m=0.2e-6, n_u_max=40, n_phi=n_phi,
                         dense_dx_over_w0=0.0)
        half = n_grid // 2  # x=y=0 pixel; central period is rows/cols 93..98 (x,y in [0,6) um)
        for switch in ('P00', 'P11'):
            d_ref, _ = rollout_numpy(pos, energy, 1000e-15, PARAMS, fixture, switch=switch)
            D, A, info = simulate_sample(PARAMS, tau_s=1000e-15, f_Hz=100e3, v_m_s=120e-3,
                                         h_m=h, passes=1, power_W=POWER_W, switch=switch,
                                         cfg=cfg, region_m=region)
            self.assertFalse(info['dense'])
            # Cell (u_i, phi_j') maps to the central period: rollout column 93+i
            # (x = u_i) and row 93+j' (y = phi_j'), no phase shift on [0,h).
            n_u = info['n_u']
            tiled = d_ref[half:half + n_phi, half:half + n_u].T
            ref_median = float(np.median(tiled))
            ref_rms = float(np.sqrt(np.mean((tiled - ref_median) ** 2)))
            tol_d = 1e-6 * max(abs(ref_median), 1e-12) + 1e-18 if switch == 'P00' \
                else 1e-3 * max(abs(ref_median), 1e-12) + 1e-18
            tol_a = 1e-6 * max(ref_rms, 1e-12) + 1e-18 if switch == 'P00' \
                else 1e-3 * max(ref_rms, 1e-12) + 1e-18
            self.assertAlmostEqual(D, ref_median, delta=tol_d, msg=f'D mismatch {switch}')
            self.assertAlmostEqual(A, ref_rms, delta=tol_a, msg=f'A mismatch {switch}')


class DensePathTests(unittest.TestCase):
    def test_dense_matches_pulse_resolved_reference(self):
        dx = 0.08e-6
        kw = dict(tau_s=1000e-15, f_Hz=1e5, v_m_s=10e-3, h_m=4e-6, passes=2,
                  power_W=POWER_W, switch='P11')
        D_dense, A_dense, _ = simulate_sample(PARAMS, **kw, cfg=CFG)
        cfg_sparse = CellConfig(dense_dx_over_w0=0.01)
        D_sparse, A_sparse, _ = simulate_sample(PARAMS, **kw, cfg=cfg_sparse)
        self.assertGreater(D_dense, 0.0)
        self.assertLess(abs(D_dense - D_sparse) / max(D_sparse, 1e-15), 0.02,
                        f'D dense={D_dense*1e6:.4f}um sparse={D_sparse*1e6:.4f}um')
        self.assertLess(abs(A_dense - A_sparse) / max(A_sparse, 1e-15), 0.02,
                        f'A dense={A_dense*1e6:.4f}um sparse={A_sparse*1e6:.4f}um')

    def test_span_drift_control_reported(self):
        kw = dict(tau_s=2000e-15, f_Hz=200e3, v_m_s=5e-3, h_m=2e-6, passes=1,
                  power_W=POWER_W, switch='P11')
        _, _, info = simulate_sample(PARAMS, **kw, cfg=CFG)
        self.assertLessEqual(info['max_span_log_threshold_drift'], CFG.packet_max_q * 1.5)


class StrongFeedbackRegressionTests(unittest.TestCase):
    def test_packet_one_updates_incubation_inside_track(self):
        """Old implementation missed 13.5% depth even at packet_cap=1."""
        p = PhysicsParameters(.874e-6, 1030e-9, 1.2, 1e4, 1e-8, .2, .1)
        cfg = CellConfig(dense_dx_over_w0=0.)
        h, dx, f = 2e-6, .2e-6, 1e5
        u = (np.arange(10)+.5)*h/10
        phi = (np.arange(cfg.n_phi)+.5)*dx/cfg.n_phi
        x,y = np.meshgrid(u,phi,indexing='ij')
        d,q = np.zeros_like(x),np.zeros_like(x)
        for j in range(-4,5):
            for k in range(-24,25):
                d,q = step(d,q,x,y,pulse_energy_J=POWER_W/f,duration_s=1e-12,
                           beam_x_m=j*h,beam_y_m=k*dx,parameters=p,switch='P01')
        reference = np.array([np.median(d),np.sqrt(np.mean((d-np.median(d))**2))])
        results=[]
        for cap in (1,64):
            D,A,info = simulate_sample(p,tau_s=1e-12,f_Hz=f,v_m_s=f*dx,h_m=h,passes=1,
                                    power_W=POWER_W,switch='P01',cfg=cfg,packet_cap=cap,return_state=True)
            results.append([D,A])
            np.testing.assert_allclose([D,A],reference,rtol=1e-4,atol=1e-14)
            np.testing.assert_allclose(info['depth_field_m'],d,rtol=1e-4,atol=1e-14)
        np.testing.assert_array_equal(results[0],results[1])

    def test_continuum_budget_exhaustion_is_not_success(self):
        cfg=replace(CFG,max_ode_steps=1)
        with self.assertRaisesRegex(ValueError,'budget exhausted'):
            simulate_sample(PARAMS,tau_s=1e-12,f_Hz=1e5,v_m_s=.005,h_m=4e-6,
                            passes=1,power_W=POWER_W,switch='P11',cfg=cfg)

    def test_noninteger_pass_count_rejected(self):
        with self.assertRaisesRegex(ValueError,'Invalid recipe'):
            simulate_sample(PARAMS,tau_s=1e-12,f_Hz=1e5,v_m_s=.005,h_m=4e-6,
                            passes=1.9,power_W=POWER_W,switch='P11',cfg=CFG)

    def test_continuum_accepted_drift_limits_enforced(self):
        cfg=replace(CFG,packet_max_log_threshold=.01,packet_max_dd_over_zr=.005)
        _,_,info=simulate_sample(PARAMS,tau_s=1e-12,f_Hz=1e5,v_m_s=.005,h_m=4e-6,
                                passes=1,power_W=POWER_W,switch='P11',cfg=cfg)
        self.assertLessEqual(info['max_span_log_threshold_drift'],cfg.packet_max_log_threshold*(1+1e-10))
        self.assertLessEqual(info['max_span_dd_over_zr'],cfg.packet_max_dd_over_zr*(1+1e-10))

    def test_cell_does_not_claim_finite_roi(self):
        _,_,info=simulate_sample(PARAMS,tau_s=1e-12,f_Hz=1e5,v_m_s=.1,h_m=4e-6,
                                passes=1,power_W=POWER_W,switch='P00',cfg=CFG)
        self.assertFalse(info['finite_region_simulated'])
        self.assertAlmostEqual(info['nominal_pulses_per_line'],200.)


if __name__ == '__main__':
    unittest.main()
