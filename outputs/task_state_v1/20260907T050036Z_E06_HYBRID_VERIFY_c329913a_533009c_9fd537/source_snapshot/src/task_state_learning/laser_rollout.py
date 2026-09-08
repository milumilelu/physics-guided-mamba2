"""Differentiable ROI-grid laser rollout for hybrid training (registered approximation).

Registered forward semantics (frozen before any hybrid training):
  - state (d, q) on an 80x80 grid over the central 80 um ROI (1 um/px), float32
    in training / float64 in verification; no lateral coupling;
  - pulse schedule from the full finite 200 um raster with finite_scan
    conventions (centered, unidirectional, no turn exposure, no guard); only
    lines whose Gaussian tail reaches the ROI act on it;
  - per raster line, ONE closed-form sweep update per pixel with the line-start
    state frozen inside the line (operator split): with L(x,y) =
    log(F_peak/F_th_eff)_+ and F_peak = 2*Ep/(pi w^2) exp(-2 dy^2/w^2),
      d += delta * (1/dx) * (2 w /(3 sqrt2)) * L^1.5,
    which is the exact continuum limit of the per-pulse law for a frozen state;
    incubation uses the same frozen state with a fixed 16-node Gauss-Legendre
    sweep of F/(F1+F). Per-line drift (d/zR, log F_th) is monitored and gated;
  - exposure blocks (packets) group consecutive lines: state updates per line,
    ONE token per packet, <= max_packets packets per sample;
  - learned closure enters ONLY as F_th_eff = F_th*exp(b), |b|<=log2, per packet;
    closure=0 reproduces the registered physics exactly;
  - observation: bilinear upsample to the canonical 160x160/0.5 um centers, then
    the differentiable observer.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import torch
import torch.nn.functional as F
from .grouping import require

ROI_M = 80e-6
GRID_N = 80
CANONICAL_N = 160
MAX_PACKETS = 64
GL_NODES = 16
LINE_DRIFT_DD_OVER_ZR = 0.02
LINE_DRIFT_LOG_THRESHOLD = 0.05


@dataclass(frozen=True)
class RolloutConfig:
    tail_waists: float = 6.0
    max_packets: int = MAX_PACKETS
    grid_n: int = GRID_N
    defocus: bool = True
    incubation: bool = True

    def __post_init__(self):
        require(self.tail_waists > 0 and self.max_packets >= 1 and self.grid_n >= 8,
                'Invalid rollout config')


def gl_nodes(n=GL_NODES, device='cpu', dtype=torch.float64):
    """Gauss-Legendre nodes/weights on [-1, 1]."""
    k = torch.arange(1, n + 1, dtype=dtype, device=device)
    beta = .5 / torch.sqrt(1. - (2. * k).pow(-2))
    theta = torch.pi * (k - .25) / (2 * n + .5)
    v0 = 1.
    for _ in range(100):
        v1 = 1.
        v2 = 0.
        for j in range(1, n + 1):
            v3 = v2
            v2 = v1
            v1 = ((2. * j - 1.) * theta.cos() * v2 - (j - 1.) * v3) / j
        dv = n * (theta.sin() * v1 - theta.cos() * v2) / (theta.sin() ** 2 - 1)
        step = v1 / dv
        theta = theta - step
        if float(step.abs().max()) < 1e-14:
            break
    x = theta.cos()
    w = 2. / ((1 - x ** 2) * dv ** 2)
    return x, w


def line_schedule(tau_s, f_Hz, v_m_s, h_m, passes, power_W, waist_m, tail_waists,
                  region_m=200e-6):
    """Raster lines (finite_scan aligned) whose tails reach the central ROI."""
    require(all(x > 0 for x in (tau_s, f_Hz, v_m_s, h_m, power_W, waist_m)), 'Invalid recipe')
    require(int(passes) == passes and passes >= 1, 'Invalid passes')
    dx = v_m_s / f_Hz
    n_lines_total = int(round(region_m / h_m))
    oy = -(n_lines_total - 1) * h_m / 2
    half = ROI_M / 2 + tail_waists * waist_m
    j_lo = max(0, int(math.ceil((-half - oy) / h_m)))
    j_hi = min(n_lines_total - 1, int(math.floor((half - oy) / h_m)))
    return {'line_y': [float(oy + h_m * j) for j in range(j_lo, j_hi + 1)],
            'n_lines_total': n_lines_total, 'dx_m': dx,
            'n_pulses_per_line': int(round(region_m / dx)), 'passes': int(passes),
            'Ep_J': power_W / f_Hz, 'tau_s': tau_s}


def packet_boundaries(schedule, cfg):
    """Equal line groups across all passes; <= min(cfg.max_packets, total_lines)."""
    per_pass = len(schedule['line_y'])
    total = per_pass * schedule['passes']
    require(per_pass >= 1, 'Empty line schedule')
    packets = min(cfg.max_packets, total)
    base = total // packets
    rem = total % packets
    out = []
    start = 0
    for _ in range(packets):
        count = base + (1 if rem > 0 else 0)
        out.append((start, count, start // per_pass))
        start += count
        rem -= 1
    require(start == total, 'Packet cover mismatch')
    return out


def packet_plan(schedule, cfg):
    """Per-sample packet structure: fixed cfg.max_packets groups over total lines.

    Lines are pass-major (pass 0 lines, then pass 1, ...), matching the
    finite_scan pass convention; when a recipe's line count is below
    cfg.max_packets the trailing packets have zero lines (no-ops) so a batch
    shares one packet grid.
    """
    per_pass = len(schedule['line_y'])
    require(per_pass >= 1, 'Empty line schedule')
    lines = [float(y) for _ in range(schedule['passes']) for y in schedule['line_y']]
    total = len(lines)
    packets = min(cfg.max_packets, total) if cfg.max_packets < total else cfg.max_packets
    base, rem = divmod(total, packets) if total >= packets else (0, total)
    counts = [base + (1 if p < rem else 0) for p in range(packets)]
    return lines, counts


def build_batch_plan(schedules, cfg):
    """Pad per-sample line lists into one (B, L) matrix + per-sample packet counts."""
    rows = []
    count_rows = []
    for schedule in schedules:
        lines, counts = packet_plan(schedule, cfg)
        rows.append(lines)
        count_rows.append(counts)
    max_lines = max(len(r) for r in rows)
    packets = max(len(c) for c in count_rows)
    line_matrix = torch.zeros(len(rows), max_lines)
    line_mask = torch.zeros(len(rows), max_lines, dtype=torch.bool)
    for i, r in enumerate(rows):
        line_matrix[i, :len(r)] = torch.tensor(r)
        line_mask[i, :len(r)] = True
    counts = torch.zeros(len(rows), packets, dtype=torch.long)
    for i, c in enumerate(count_rows):
        counts[i, :len(c)] = torch.tensor(c)
    return line_matrix, line_mask, counts


def rollout(d0, q0, schedules, params, commands, closure_log, cfg=RolloutConfig(),
            device='cpu', dtype=torch.float32):
    """Batch rollout with per-sample finite-raster line schedules.

    commands: (B, P, 4) = [log_tau, log_f, log_v, log_h] per packet.
    params: dict of per-sample tensors (w0, zr, F1, delta, kappa) plus scalars
    (rho, Ep, tau_ref, gamma_F, gamma_delta) or per-sample tensors.
    closure_log: (B, P) in [-log2, log2] or None (exact physics).
    Returns terminal d, q and drift diagnostics (detached).
    """
    require(d0.shape == q0.shape and d0.ndim == 3, 'Rollout state shape')
    B, n, _ = d0.shape
    coords = ((torch.arange(n, device=device, dtype=dtype) - (n - 1) / 2) * (ROI_M / n))
    Xg, Yg = torch.meshgrid(coords, coords, indexing='xy')
    Xg = Xg[None]
    Yg = Yg[None]
    d, q = d0.clone(), q0.clone()
    gl_x, gl_w = gl_nodes(device=device, dtype=torch.float64)
    gl_x = gl_x.to(dtype)
    gl_w = gl_w.to(dtype)
    line_matrix, line_mask, counts = build_batch_plan(schedules, cfg)
    line_matrix = line_matrix.to(device=device, dtype=dtype)
    counts = counts.to(device)
    offsets = torch.cat([torch.zeros(len(counts), 1, dtype=torch.long, device=device),
                         counts.cumsum(1)[:, :-1]], dim=1)
    packets = counts.shape[1]
    drift_max_dd = 0.0
    drift_max_lth = 0.0
    for p in range(packets):
        b = closure_log[:, p] if closure_log is not None else None
        d, q, drift = apply_packet(d, q, Xg, Yg, line_matrix, counts, offsets, p, schedules,
                                   params, commands, b, cfg, gl_x, gl_w)
        drift_max_dd = max(drift_max_dd, float(drift[0]))
        drift_max_lth = max(drift_max_lth, float(drift[1]))
    return d, q, {'max_line_dd_over_zr': drift_max_dd,
                  'max_line_log_threshold_drift': drift_max_lth}


def apply_packet(d, q, Xg, Yg, line_matrix, counts, offsets, p, schedules, params,
                 commands, b, cfg, gl_x, gl_w):
    """One packet = consecutive raster lines per sample; returns new state + drift."""
    drift_max_dd = 0.0
    drift_max_lth = 0.0
    for k in range(int(counts[:, p].max())):
        active = counts[:, p] > k
        if not bool(active.any()):
            break
        y_line = line_matrix[active, offsets[active, p] + k]
        d, q, drift = _apply_line_masked(d, q, Xg, Yg, y_line, schedules, params,
                                         commands, p, b, cfg, gl_x, gl_w, active)
        drift_max_dd = max(drift_max_dd, float(drift[0]))
        drift_max_lth = max(drift_max_lth, float(drift[1]))
    return d, q, (drift_max_dd, drift_max_lth)


def _apply_line_masked(d, q, Xg, Yg, y_line, schedules, params, commands, p, b, cfg,
                       gl_x, gl_w, active):
    """Closed-form frozen-state sweep update for the batch's active raster lines.

    y_line: (A,) beam-line positions for the active subset; samples not in the
    subset keep their state (their next line comes later in the packet).
    """
    A = int(active.sum())
    tau = commands[active, p, 0].exp()[:, None, None]
    f_Hz = commands[active, p, 1].exp()[:, None, None]
    v = commands[active, p, 2].exp()[:, None, None]
    F1 = params['F1'][active].view(-1, 1, 1) * (tau / params['tau_ref']) ** params['gamma_F']
    delta = params['delta'][active].view(-1, 1, 1) * \
        (tau / params['tau_ref']) ** params['gamma_delta']
    w0 = params['w0'][active].view(-1, 1, 1)
    zr = params['zr'][active].view(-1, 1, 1)
    kappa = params['kappa'][active].view(-1, 1, 1)
    rho = params['rho'][active].view(-1, 1, 1) if torch.is_tensor(params['rho']) and \
        params['rho'].numel() > 1 else (params['rho'].reshape(1, 1, 1)
                                        if torch.is_tensor(params['rho'])
                                        else torch.tensor([[params['rho']]]))
    Ep = Ep_of(params)
    if not torch.is_tensor(Ep):
        Ep = torch.tensor(float(Ep))
    Ep = Ep.view(-1, 1, 1)[active] if Ep.ndim else Ep
    d_sub, q_sub = d[active], q[active]
    if cfg.defocus:
        w = w0 * torch.sqrt(1. + (d_sub / zr) ** 2)
    else:
        w = w0.expand_as(d_sub)
    dy = Yg - y_line.view(-1, 1, 1)
    log_peak = torch.log(2. * Ep / torch.pi) - 2. * torch.log(w) - 2. * dy ** 2 / w ** 2
    if cfg.incubation:
        log_threshold = torch.log(F1) + torch.log(rho + (1. - rho) * torch.exp(-q_sub))
    else:
        log_threshold = torch.log(F1).expand_as(d_sub)
    if b is not None:
        log_threshold = log_threshold + b[active][:, None, None]
    L = (log_peak - log_threshold).clamp_min(0.)
    dx_m = torch.tensor([s['dx_m'] for s in schedules], dtype=d.dtype,
                        device=d.device)[active].view(-1, 1, 1)
    d_gain = delta * L.sqrt() ** 3 * (4. * w / (3. * math.sqrt(2.))) / dx_m
    d_next = d.clone()
    d_next[active] = d_sub + d_gain
    if cfg.incubation:
        nodes = gl_x[None, None, None] * (6. * w[..., None])
        fluence = torch.exp(log_peak[..., None] - 2. * nodes ** 2 / (w[..., None] ** 2))
        integrand = fluence / (F1[..., None] + fluence)
        q_gain = kappa * f_Hz / v * (12. * w) * \
            (gl_w[None, None, None, :] * integrand).sum(-1)
        q_next = q.clone()
        q_next[active] = q_sub + q_gain
    else:
        q_next = q
    with torch.no_grad():
        drift_dd = float((d_gain / zr).max()) if cfg.defocus else 0.0
        if cfg.incubation:
            new_th = torch.log(F1) + torch.log(rho + (1. - rho) * torch.exp(-q_next[active]))
            old_th = torch.log(F1) + torch.log(rho + (1. - rho) * torch.exp(-q_sub))
            drift_lth = float((new_th - old_th).abs().max())
        else:
            drift_lth = 0.0
    return d_next, q_next, (drift_dd, drift_lth)


def Ep_of(params):
    return params['Ep'] if torch.is_tensor(params['Ep']) else torch.tensor(params['Ep'])


def upsample_depth(d):
    """Bilinear map of the rollout grid onto the canonical 160x160 centers."""
    require(d.shape[1] == d.shape[2] == GRID_N, 'Rollout grid shape')
    up = F.interpolate(d[:, None], size=CANONICAL_N, mode='bilinear', align_corners=True)
    return up[:, 0]


def terminal_height_um(d):
    return -upsample_depth(d) * 1e6


def state_tokens(d, q, w0, zr, h, dx, closure_support=None):
    """Per-sample scalar summary from the predicted state (token features)."""
    flat_d = d.reshape(len(d), -1)
    flat_q = q.reshape(len(q), -1)
    mean_d = flat_d.mean(1, keepdim=True)
    rms_d = flat_d.pow(2).mean(1, keepdim=True).sqrt()
    mean_q = flat_q.mean(1, keepdim=True)
    mean_d_zr = mean_d / zr.reshape(-1, 1)
    h_2w = h.reshape(-1, 1) / (2 * w0.reshape(-1, 1))
    dx_2w = dx.reshape(-1, 1) / (2 * w0.reshape(-1, 1))
    return torch.cat([mean_d, rms_d, mean_q, mean_d_zr, h_2w, dx_2w], dim=1)
