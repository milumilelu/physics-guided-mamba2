"""Periodic-cell scan solver for E03 calibration.

Registered block model for dense real-scenario pulses (protocol 6.5 escape hatch:
pulse-resolved physics with frozen sub-step state; never one pulse of m*Ep).

Cell state lives on one hatch period (cross-track u in [0,h)) x one pulse-lattice
period (along-track phase phi in [0,dx)). Every pixel maps to the same pulse-distance
set {phi + m*dx}, so a track pass reduces to a windowed sweep with arrival order
given by the ascending pulse index.

Two registered paths:
- sparse (dx > dense_dx_over_w0 * w0): per-pulse frozen-state packet blocks over the
  exact pulse lattice; packet size from the registered controls.
- dense (dx <= dense_dx_over_w0 * w0): continuum sweep quadrature. The along-track
  axis is integrated with arrival-order incubation q_arr(t) (cumulative from the
  approach side); per-pulse fluence and per-node log-threshold are preserved, so no
  m*Ep aggregation occurs. t-spans are split greedily so no pixel's frozen-state
  incubation drift within a span exceeds packet_max_q (dd control verified post hoc).

Contributions beyond window_over_w * w are truncated (documented tail bound).
Switch semantics match src.task_state_learning.physics.step exactly.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
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
    max_spans: int = 64
    max_packet_pulses: int = 64
    packet_max_q: float = 0.05
    packet_max_dd_over_zr: float = 0.02
    packet_max_beam_over_w: float = 0.25


def physical_params(vector, switch, base: PhysicsParameters) -> PhysicsParameters:
    """Physical parameters from the optimizer vector (log space for positives)."""
    require(switch in SWITCHES, 'Unknown physical switch')
    vector = [float(v) for v in vector]
    if switch in ('P00', 'P10'):
        require(len(vector) == 4, 'P00/P10 calibrate 4 parameters')
        log_F1, log_delta, gamma_F, gamma_delta = vector
        log_kappa = math.log(base.kappa)
        logit_rho = math.log(base.saturation_ratio / (1.0 - base.saturation_ratio))
    else:
        require(len(vector) == 6, 'P01/P11 calibrate 6 parameters')
        log_F1, log_delta, log_kappa, logit_rho, gamma_F, gamma_delta = vector
    return PhysicsParameters(
        waist_m=base.waist_m, wavelength_m=base.wavelength_m, m_squared=base.m_squared,
        F1_reference_J_m2=math.exp(log_F1), delta_reference_m=math.exp(log_delta),
        kappa=math.exp(log_kappa), saturation_ratio=1.0 / (1.0 + math.exp(-logit_rho)),
        gamma_F=gamma_F, gamma_delta=gamma_delta,
        reference_duration_s=base.reference_duration_s)


def simulate_sample(params: PhysicsParameters, *, tau_s, f_Hz, v_m_s, h_m, passes, power_W,
                    switch, cfg: CellConfig, region_m=None, packet_cap=None):
    """Terminal (D_sim_m, A_med_sim_m, info) for one registered process recipe."""
    require(switch in SWITCHES, 'Unknown physical switch')
    require(tau_s > 0 and f_Hz > 0 and v_m_s > 0 and h_m > 0 and passes >= 1, 'Invalid recipe')
    require(power_W > 0, 'Invalid power')
    dx = v_m_s / f_Hz
    pulse_energy = power_W / f_Hz
    region = cfg.region_m if region_m is None else region_m
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
                   'u_lattice_window': 0}
    for _ in range(int(passes)):
        # Cross-track lattice sum: the pixel sees every hatch line at |u + n*h|;
        # skipping the mirror tracks (n<0) would under-ablate mid-hatch pixels.
        w_max = float((params.waist_m * np.sqrt(
            1.0 + (depth / params.rayleigh_m) ** 2)).max()) if switch[1] == '1' \
            else params.waist_m
        n_lat = int(math.ceil(cfg.window_over_w * w_max / h_m)) + 1
        diagnostics['u_lattice_window'] = max(diagnostics['u_lattice_window'], 2 * n_lat + 1)
        # Descending n matches the ascending-x raster order seen by the central
        # period (track j = -n); the fold makes this the registered convention.
        for n in range(n_lat, -n_lat - 1, -1):
            u_shift = u + n * h_m
            if dense:
                depth, incubation, dq, ddz = _track_sweep_continuum(
                    depth, incubation, u_shift, dx, pulse_energy, tau_s, params, switch, cfg)
            else:
                depth, incubation, dq, ddz = _track_sweep_pulses(
                    depth, incubation, U, PHI, u_shift, dx, pulse_energy, tau_s, params,
                    switch, cfg, packet_cap)
            diagnostics['max_span_log_threshold_drift'] = max(
                diagnostics['max_span_log_threshold_drift'], dq)
            diagnostics['max_span_dd_over_zr'] = max(diagnostics['max_span_dd_over_zr'], ddz)
    flat = depth.ravel()
    d_median = float(np.median(flat))
    return d_median, float(np.sqrt(np.mean((flat - d_median) ** 2))), diagnostics


def _scales(p, tau_s):
    f1 = p.F1_reference_J_m2 * (tau_s / p.reference_duration_s) ** p.gamma_F
    delta = p.delta_reference_m * (tau_s / p.reference_duration_s) ** p.gamma_delta
    return f1, delta


def _track_sweep_pulses(depth, incubation, U, PHI, u_shift, dx, pulse_energy, tau_s, p,
                        switch, cfg, packet_cap=None):
    """Sparse regime: per-pulse frozen-state packet blocks over the pulse lattice.

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
    a_peak = 2 * pulse_energy / (math.pi * float(w.min()) ** 2)
    frac_max = a_peak / (f1 + a_peak)
    lr_max = max(math.log(a_peak / float(threshold.min())), 1e-9) if a_peak > float(threshold.min()) else 1e-9
    caps = [cfg.max_packet_pulses if packet_cap is None else min(packet_cap, cfg.max_packet_pulses),
            math.floor(cfg.packet_max_beam_over_w * w_max / dx),
            math.floor(cfg.packet_max_q / max(p.kappa * frac_max, 1e-12)),
            math.floor(cfg.packet_max_dd_over_zr * z_r / max(delta * lr_max, 1e-18))]
    m_packet = int(max(1, min(caps)))
    log_peak = np.log(2 * pulse_energy / (math.pi * w ** 2))[:, :, None]
    w2 = w[:, :, None] ** 2
    log_threshold = np.log(threshold)[:, :, None] if incubate \
        else np.full(depth.shape + (1,), math.log(f1))
    Us = u_shift[:, None]  # cross-track offset from the swept line (signed; squared)
    d, q = depth, incubation
    m_hi = m_window
    while m_hi >= -m_window:
        m_lo = max(m_hi - m_packet + 1, -m_window)
        offsets = PHI[:, :, None] + np.arange(m_lo, m_hi + 1, dtype=float)[None, None, :] * dx
        log_fluence = log_peak - 2.0 * (Us[:, :, None] ** 2 + offsets ** 2) / w2
        d = d + delta * np.clip(log_fluence - log_threshold, 0.0, None).sum(axis=-1)
        if incubate:
            fluence = np.exp(log_fluence)
            q = q + p.kappa * (fluence / (f1 + fluence)).sum(axis=-1)
        m_hi = m_lo - 1
    drift_q = float(m_packet * p.kappa * frac_max) if incubate else 0.0
    drift_dd = float(m_packet * delta * lr_max / z_r)
    require(np.isfinite(d).all() and np.isfinite(q).all(), 'Nonfinite cell state')
    return d, q, drift_q, drift_dd


def _track_sweep_continuum(depth, incubation, u_shift, dx, pulse_energy, tau_s, p, switch, cfg):
    """Dense regime: continuum sweep quadrature with arrival-order incubation.

    u_shift: signed cross-track offset from the swept hatch line to the cell pixels
    (includes the hatch-lattice shift n*h)."""
    defocus = switch[1] == '1'
    incubate = switch[2] == '1'
    z_r = p.rayleigh_m
    f1, delta = _scales(p, tau_s)
    w = p.waist_m * np.sqrt(1.0 + (depth[:, 0] / z_r) ** 2) if defocus \
        else np.full(depth.shape[0], p.waist_m)
    require(np.isfinite(w).all() and float(w.max()) > 0, 'Nonfinite waist')
    log_a = np.log(2 * pulse_energy / (math.pi * w ** 2))
    t_half = np.sqrt(np.maximum((cfg.window_over_w * w) ** 2 - u_shift ** 2, 0.0))
    live = t_half > 0  # pixels whose cross-track offset lies inside the sweep window
    if not bool(live.any()):
        return depth, incubation, 0.0, 0.0  # this hatch line is too far to matter
    log_a, t_half, u_live = log_a[live], t_half[live], u_shift[live]
    t_grid = np.linspace(-1.0, 1.0, cfg.t_nodes)
    t_abs = t_half[:, None] * np.abs(t_grid)[None, :]
    log_fluence = log_a[:, None] - 2.0 * (u_live[:, None] ** 2 + t_abs ** 2) / (w[live][:, None] ** 2)
    dt = (2.0 * t_half)[:, None] / (cfg.t_nodes - 1)
    fluence = np.exp(log_fluence)
    rate_q = fluence / (f1 + fluence)
    d = depth[live, 0].copy()
    q = incubation[live, 0].copy() if incubate else np.zeros_like(u_live)
    if incubate:
        cum = np.cumsum(rate_q * dt, axis=1) * p.kappa / dx  # incubation increments
        # Greedy span split on the pixel-max log-threshold drift (the quantity that
        # actually bounds the frozen-state error; reduces to the protocol's dq
        # control when q << 1 and stays meaningful once q saturates).
        edges, start = [0], 0
        q_start = q.copy()
        max_drift = 0.0
        for node in range(1, cfg.t_nodes):
            q_node = q_start + (cum[:, node] - cum[:, start])
            drift = np.abs(np.log(f1 * (p.saturation_ratio
                                        + (1 - p.saturation_ratio) * np.exp(-q_node)))
                           - np.log(f1 * (p.saturation_ratio
                                          + (1 - p.saturation_ratio) * np.exp(-q_start))))
            if float(drift.max()) >= cfg.packet_max_q:
                edges.append(node)
                start = node
                q_start = q_node
                max_drift = max(max_drift, float(drift.max()))
            if len(edges) >= cfg.max_spans:
                break
        if edges[-1] != cfg.t_nodes - 1:
            edges.append(cfg.t_nodes - 1)
        edges = sorted(set(edges))
    else:
        edges = list(np.unique(np.linspace(0, cfg.t_nodes - 1, 9).astype(int)))
    max_dq, max_ddz = 0.0, 0.0
    if incubate:
        max_dq = max_drift  # worst closed-span log-threshold drift (log units)
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        sl = slice(a, b)
        if incubate:
            q_arr = q[:, None] + (cum[:, sl] - cum[:, a][:, None])
            log_threshold = np.log(f1 * (p.saturation_ratio
                                         + (1 - p.saturation_ratio) * np.exp(-q_arr)))
        else:
            log_threshold = np.full(log_fluence[:, sl].shape, math.log(f1))
        d_inc = (delta / dx) * (np.clip(log_fluence[:, sl] - log_threshold, 0.0, None)
                                * dt).sum(axis=1)
        d = d + d_inc
        if incubate:
            q = q + p.kappa * (rate_q[:, sl] * dt).sum(axis=1) / dx
        max_ddz = max(max_ddz, float(d_inc.max()) / z_r)
    require(np.isfinite(d).all() and np.isfinite(q).all(), 'Nonfinite cell state')
    depth_new = depth.copy()
    depth_new[live, 0] = d  # pixels outside the sweep window receive nothing this track
    incubation_new = incubation.copy()
    if incubate:
        incubation_new[live, 0] = q
    return depth_new, incubation_new, max_dq, max_ddz
