"""Candidate interior-cell solver; E03 registration requires separate validation.

Sparse: exact current-state pulse updates within scheduling packets.
Dense: adaptive continuous spatial-rate ODE, retaining single-pulse fluence and
current d/q. The dense limit is an approximation, not an identity with the pulse
lattice. Both paths approximate an infinite interior raster, not a finite 80um
ROI. Window/mesh/continuum errors must be assessed before calibration.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import expit
from .grouping import require
from .physics import PhysicsParameters

SWITCHES = ('P00', 'P10', 'P01', 'P11')


@dataclass(frozen=True)
class CellConfig:
    region_m: float = 200e-6
    u_target_m: float = 0.2e-6
    n_u_max: int = 40
    n_phi: int = 8
    dense_dx_over_w0: float = 0.15
    window_over_w: float = 3.0
    t_nodes: int = 256
    max_ode_steps: int = 20000
    max_refinement_attempts: int = 12
    ode_rtol: float = 1e-7
    ode_atol: float = 1e-10
    max_window_retries: int = 8
    max_lattice_lines: int = 401
    max_packet_pulses: int = 64
    packet_max_q: float = 0.05
    packet_max_log_threshold: float = 0.05
    packet_max_dd_over_zr: float = 0.02
    packet_max_beam_over_w: float = 0.25

    def __post_init__(self):
        require(np.isfinite(list(vars(self).values())).all(), 'Nonfinite cell setting')
        require(self.region_m>0 and self.u_target_m>0 and self.window_over_w>0, 'Invalid cell length')
        require(self.n_u_max>=8 and self.n_phi>=1 and self.t_nodes>=3, 'Invalid cell mesh')
        require(self.max_packet_pulses>=1 and self.max_ode_steps>=1 and self.max_refinement_attempts>=1,
                'Invalid solver budget')
        require(self.max_window_retries>=1 and self.max_lattice_lines>=3, 'Invalid window budget')
        require(min(self.packet_max_q,self.packet_max_log_threshold,self.packet_max_dd_over_zr,
                    self.packet_max_beam_over_w,self.ode_rtol,self.ode_atol)>0, 'Invalid solver tolerance')


def physical_params(vector, switch, base: PhysicsParameters) -> PhysicsParameters:
    """Physical parameters from the optimizer vector (log space for positives)."""
    require(switch in SWITCHES, 'Unknown physical switch')
    vector = [float(v) for v in vector]
    if switch in ('P00', 'P10'):
        require(len(vector) == 4, 'P00/P10 calibrate 4 parameters')
        log_F1, log_delta, gamma_F, gamma_delta = vector
        kappa, rho = base.kappa, base.saturation_ratio
    else:
        require(len(vector) == 6, 'P01/P11 calibrate 6 parameters')
        log_F1, log_delta, log_kappa, logit_rho, gamma_F, gamma_delta = vector
        kappa, rho = math.exp(log_kappa), float(expit(logit_rho))
    return PhysicsParameters(
        waist_m=base.waist_m, wavelength_m=base.wavelength_m, m_squared=base.m_squared,
        F1_reference_J_m2=math.exp(log_F1), delta_reference_m=math.exp(log_delta),
        kappa=kappa, saturation_ratio=rho,
        gamma_F=gamma_F, gamma_delta=gamma_delta,
        reference_duration_s=base.reference_duration_s)


def simulate_sample(params: PhysicsParameters, *, tau_s, f_Hz, v_m_s, h_m, passes, power_W,
                    switch, cfg: CellConfig, region_m=None, packet_cap=None, return_state=False):
    """Terminal (D_sim_m, A_med_sim_m, info) for one registered process recipe."""
    require(switch in SWITCHES, 'Unknown physical switch')
    require(np.isfinite([tau_s,f_Hz,v_m_s,h_m,passes,power_W]).all(), 'Nonfinite recipe')
    require(tau_s > 0 and f_Hz > 0 and v_m_s > 0 and h_m > 0 and passes >= 1 and int(passes)==passes, 'Invalid recipe')
    require(power_W > 0, 'Invalid power')
    dx = v_m_s / f_Hz
    pulse_energy = power_W / f_Hz
    region = cfg.region_m if region_m is None else region_m
    require(np.isfinite(region) and region>0, 'Invalid region')
    dense = dx <= cfg.dense_dx_over_w0 * params.waist_m
    n_u = int(np.clip(round(h_m / cfg.u_target_m), 8, cfg.n_u_max))
    n_phi = 1 if dense else cfg.n_phi
    u = (np.arange(n_u) + 0.5) * (h_m / n_u)
    phi = (np.arange(n_phi) + 0.5) * (dx / n_phi) if n_phi > 1 else np.zeros(1)
    U, PHI = np.meshgrid(u, phi, indexing='ij')
    depth = np.zeros((n_u, n_phi))
    incubation = np.zeros((n_u, n_phi))
    diagnostics = {'dense': bool(dense), 'n_u': n_u, 'n_phi': n_phi,
                   'pulse_spacing_m': dx, 'pulse_energy_J': pulse_energy,
                   'max_span_log_threshold_drift': 0.0, 'max_span_dd_over_zr': 0.0,
                   'u_lattice_window': 0,
                   'domain_model': 'infinite_raster_interior_cell',
                   'finite_region_simulated': False,
                   'requested_region_m': float(region),
                   'nominal_pulses_per_line': float(region/dx),
                   'nominal_pulses_per_pass': float(region/dx*region/h_m),
                   'drift_semantics': 'accepted_ODE_span' if dense else 'exact_atomic_pulse',
                   'calibration_registration': 'PENDING_PARAMETER_DOMAIN_AND_OBSERVATION_VALIDATION'}
    for _ in range(int(passes)):
        start_d, start_q = depth.copy(), incubation.copy()
        w_max = float((params.waist_m*np.sqrt(1+(start_d/params.rayleigh_m)**2)).max()) if switch[1]=='1' else params.waist_m
        n_lat = int(math.ceil(cfg.window_over_w*w_max/h_m))+1
        for window_attempt in range(cfg.max_window_retries):
            require(2*n_lat+1 <= cfg.max_lattice_lines, 'Transverse lattice budget exhausted')
            depth, incubation = start_d.copy(), start_q.copy()
            trial_log_drift, trial_dd = 0.0, 0.0
            for n in range(n_lat, -n_lat-1, -1):
                u_shift = u+n*h_m
                if dense:
                    depth, incubation, drift, dd = _track_sweep_continuum(
                        depth,incubation,u_shift,dx,pulse_energy,tau_s,params,switch,cfg)
                else:
                    depth, incubation, drift, dd = _track_sweep_pulses(
                        depth,incubation,U,PHI,u_shift,dx,pulse_energy,tau_s,params,switch,cfg,packet_cap)
                trial_log_drift=max(trial_log_drift,drift)
                trial_dd=max(trial_dd,dd)
            final_w=float((params.waist_m*np.sqrt(1+(depth/params.rayleigh_m)**2)).max()) if switch[1]=='1' else params.waist_m
            required=int(math.ceil(cfg.window_over_w*final_w/h_m))+1
            if required <= n_lat:
                diagnostics['u_lattice_window']=max(diagnostics['u_lattice_window'],2*n_lat+1)
                diagnostics['max_span_log_threshold_drift']=max(diagnostics['max_span_log_threshold_drift'],trial_log_drift)
                diagnostics['max_span_dd_over_zr']=max(diagnostics['max_span_dd_over_zr'],trial_dd)
                break
            n_lat=max(required,int(1.5*n_lat)+1)
        else:
            raise ValueError('Transverse lattice window did not stabilize')
    flat = depth.ravel()
    d_median = float(np.median(flat))
    if return_state:
        diagnostics['depth_field_m'] = depth.copy()
        diagnostics['incubation_field'] = incubation.copy()
    return d_median, float(np.sqrt(np.mean((flat - d_median) ** 2))), diagnostics


def _scales(p, tau_s):
    f1 = p.F1_reference_J_m2 * (tau_s / p.reference_duration_s) ** p.gamma_F
    delta = p.delta_reference_m * (tau_s / p.reference_duration_s) ** p.gamma_delta
    return f1, delta


def _track_sweep_pulses(depth, incubation, U, PHI, u_shift, dx, pulse_energy, tau_s, p,
                        switch, cfg, packet_cap=None, _window_m=None, _retry=0):
    """Sparse regime: exact per-pulse state updates over the pulse lattice.

    u_shift is the cross-track distance from the swept line to each cell pixel
    (already includes the hatch-lattice shift n*h)."""
    defocus = switch[1] == '1'
    incubate = switch[2] == '1'
    z_r = p.rayleigh_m
    f1, delta = _scales(p, tau_s)
    w = p.waist_m * np.sqrt(1.0 + (depth / z_r) ** 2) if defocus else np.full_like(depth, p.waist_m)
    threshold = f1 * (p.saturation_ratio + (1 - p.saturation_ratio) * np.exp(-incubation)) \
        if incubate else np.full_like(depth, f1)
    w_max = float(w.max())
    require(np.isfinite(w).all() and w_max > 0, 'Nonfinite waist')
    m_window = int(math.ceil(cfg.window_over_w * w_max / dx)) + 1
    if _window_m is not None:
        m_window = max(m_window, _window_m)
    a_peak = 2 * pulse_energy / (math.pi * float(w.min()) ** 2)
    frac_max = a_peak / (f1 + a_peak)
    lr_max = max(math.log(a_peak / float(threshold.min())), 1e-9) if a_peak > float(threshold.min()) else 1e-9
    caps = [cfg.max_packet_pulses if packet_cap is None else min(packet_cap, cfg.max_packet_pulses),
            math.floor(cfg.packet_max_beam_over_w * w_max / dx),
            math.floor(cfg.packet_max_q / max(p.kappa * frac_max, 1e-12)),
            math.floor(cfg.packet_max_dd_over_zr * z_r / max(delta * lr_max, 1e-18))]
    m_packet = int(max(1, min(caps)))
    Us = u_shift[:, None]  # cross-track offset from the swept line (signed; squared)
    d, q = depth.copy(), incubation.copy()
    drift_q, drift_dd = 0.0, 0.0
    m_hi = m_window
    while m_hi >= -m_window:
        m_lo = max(m_hi - m_packet + 1, -m_window)
        # A packet is a scheduling block, NOT a frozen full-track state. Visit
        # pulses in arrival order and refresh both geometric and threshold feedback.
        for m in range(m_hi, m_lo - 1, -1):
            w = p.waist_m * np.sqrt(1.0 + (d / z_r) ** 2) if defocus else np.full_like(d, p.waist_m)
            threshold = f1 * (p.saturation_ratio + (1 - p.saturation_ratio) * np.exp(-q)) if incubate else np.full_like(d, f1)
            log_fluence = np.log(2 * pulse_energy / (math.pi * w ** 2)) - 2.0 * (Us ** 2 + (PHI + m * dx) ** 2) / w ** 2
            increment = delta * np.clip(log_fluence - np.log(threshold), 0.0, None)
            d = d + increment
            drift_dd = max(drift_dd, float(increment.max() / z_r))
            if incubate:
                fluence = np.exp(log_fluence)
                next_q = q + p.kappa * fluence / (f1 + fluence)
                next_threshold = f1 * (p.saturation_ratio + (1 - p.saturation_ratio) * np.exp(-next_q))
                drift_q = max(drift_q, float(np.abs(np.log(next_threshold) - np.log(threshold)).max()))
                q = next_q
        m_hi = m_lo - 1
    require(np.isfinite(d).all() and np.isfinite(q).all(), 'Nonfinite cell state')
    final_w = float((p.waist_m*np.sqrt(1+(d/z_r)**2)).max()) if defocus else p.waist_m
    required_m = int(math.ceil(cfg.window_over_w*final_w/dx))+1
    if required_m > m_window:
        require(_retry+1 < cfg.max_window_retries, 'Pulse window did not stabilize')
        return _track_sweep_pulses(depth,incubation,U,PHI,u_shift,dx,pulse_energy,tau_s,p,switch,cfg,
                                  packet_cap,max(required_m,int(1.5*m_window)+1),_retry+1)
    return d, q, drift_q, drift_dd


def _track_sweep_continuum(depth, incubation, u_shift, dx, pulse_energy, tau_s, p, switch, cfg):
    """Continuous spatial-rate ODE; d and q feedback updated at EVERY RHS call.

    This is an approximation of the pulse lattice, not a proof of equivalence.
    Accepted ODE steps must satisfy threshold/depth controls. Exhausting the
    numerical budget raises an error; no unchecked final span is appended.
    """
    f1, delta = _scales(p, tau_s)
    z_r = p.rayleigh_m
    n = len(u_shift)
    initial = np.r_[depth[:, 0] / z_r, incubation[:, 0]]
    def waist(d):
        return p.waist_m * np.sqrt(1 + d**2) if switch[1] == '1' else np.full_like(d, p.waist_m)
    def log_threshold(q):
        return np.log(f1 * (p.saturation_ratio + (1-p.saturation_ratio)*np.exp(-q)))
    def rate(t, state):
        w = waist(state[:n])
        log_f = np.log(2*pulse_energy/(math.pi*w**2)) - 2*(u_shift**2+(t*p.waist_m)**2)/w**2
        th = log_threshold(state[n:]) if switch[2] == '1' else math.log(f1)
        dd = delta / z_r * np.maximum(log_f-th, 0.) * p.waist_m/dx
        dq = p.kappa * expit(log_f-math.log(f1)) * p.waist_m/dx if switch[2] == '1' else np.zeros(n)
        return np.r_[dd,dq]
    radius = cfg.window_over_w * float(waist(initial[:n]).max()) / p.waist_m
    for window_attempt in range(cfg.max_window_retries):
        step_cap = min(cfg.packet_max_beam_over_w, 2*radius/max(cfg.t_nodes-1,1))
        for attempt in range(cfg.max_refinement_attempts):
            # RK45 avoids DOP853's squared-error underflow on Gaussian far tails.
            # The same tolerances and accepted-span checks still apply.
            sol = solve_ivp(rate, (-radius,radius), initial, method='RK45',
                            rtol=cfg.ode_rtol, atol=cfg.ode_atol, max_step=step_cap)
            require(sol.success, 'Continuum ODE failed: '+str(sol.message))
            require(len(sol.t)-1 <= cfg.max_ode_steps, 'Continuum step budget exhausted')
            require(np.isfinite(sol.y).all(), 'Nonfinite continuum state')
            dd = float(np.abs(np.diff(sol.y[:n],axis=1)).max())
            drift = float(np.abs(np.diff(log_threshold(sol.y[n:]),axis=1)).max()) if switch[2]=='1' else 0.
            ratio = max(dd/cfg.packet_max_dd_over_zr, drift/cfg.packet_max_log_threshold)
            if ratio <= 1.0 + 1e-10:
                break
            step_cap *= min(.5,.8/ratio)
        else:
            raise ValueError('Continuum drift control failed after refinement budget')
        required_radius = cfg.window_over_w * float(waist(sol.y[:n,-1]).max()) / p.waist_m
        if required_radius <= radius*(1+1e-3):
            d = sol.y[:n,-1,None]*z_r
            q = sol.y[n:,-1,None]
            require((d >= depth-1e-15).all() and (q >= incubation-1e-12).all(), 'Nonmonotone continuum state')
            return d, q, drift, dd
        radius = max(radius*1.5, required_radius*1.1)
    raise ValueError('Continuum window did not stabilize')
