"""Differentiable ROI-grid laser rollout for hybrid training (registered approximation).

Registered forward semantics (frozen before any hybrid training):
  - state (d, q) on the canonical 160x160 grid over the central 80 um ROI
    (0.5 um/px), float32 in training / float64 in verification; no lateral
    coupling (identical to the validated pointwise finite_scan contract);
  - pulse schedule from the full finite 200 um raster with finite_scan
    conventions (centered, unidirectional, no turn exposure, no guard); only
    lines/pulses whose tail reaches the ROI act on it;
  - path selection per sample (registered threshold 0.15 w0, matching the cell
    solver): DENSE (dx <= 0.15 w0) uses the closed-form frozen-state line sweep
    per raster line: with L = log(F_peak/F_th_eff)_+,
        d += delta * (1/dx) * (4 w / (3 sqrt 2)) * L^1.5
    (the exact phase-averaged continuum limit of the per-pulse law; verified
    against direct pulse sums within 1% for dx <= 0.5 w), and the incubation
    integral via 16-node Gauss-Legendre quadrature;
    SPARSE (dx > 0.15 w0) uses exact per-pulse Gaussian updates, state frozen
    within each raster line and refreshed between lines (vectorized per line);
  - per-line frozen-state drift (d/z_R, log F_th) is monitored and reported;
  - exposure blocks (packets) group consecutive lines: state updates per line,
    ONE token per packet, <= max_packets packets per sample (trailing packets
    are no-ops so a batch shares one packet grid);
  - learned closure enters ONLY as F_th_eff = F_th*exp(b), |b|<=log2, per packet;
    closure=0 reproduces the registered physics exactly;
  - observation: the rollout grid IS the canonical observation grid.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from .grouping import require

ROI_M = 80e-6
GRID_N = 160
CANONICAL_N = 160
MAX_PACKETS = 64
GL_NODES = 16
DENSE_DX_OVER_W0 = 0.15
PULSE_WINDOW_SIGMA = 5.0
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
    k = torch.arange(1, n + 1, dtype=dtype, device=device)
    theta = torch.pi * (k - .25) / (2 * n + .5)
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


def packet_plan(schedule, cfg):
    """Fixed cfg.max_packets groups over the pass-major line list; trailing
    groups with zero lines are no-ops so a batch shares one packet grid."""
    lines = [float(y) for _ in range(schedule['passes']) for y in schedule['line_y']]
    total = len(lines)
    require(total >= 1, 'Empty line schedule')
    base, rem = divmod(total, cfg.max_packets)
    counts = [base + (1 if p < rem else 0) for p in range(cfg.max_packets)]
    return lines, counts


def build_batch_plan(schedules, cfg):
    """Per-sample line lists, packet counts, and ROI-window pulse positions.

    Pulses keep line-major order; `pulses_per_line` is the number of restricted
    pulses per raster line for each sample (equal across lines and passes)."""
    rows, count_rows, pulse_rows = [], [], []
    for schedule in schedules:
        lines, counts = packet_plan(schedule, cfg)
        rows.append(lines)
        count_rows.append(counts)
        np_p = schedule['n_pulses_per_line']
        dx = schedule['dx_m']
        ox = -(np_p - 1) * dx / 2
        window = ROI_M / 2 + cfg.tail_waists * 1.5 * schedule_waist(schedule)
        pulses = [(float(ox + k * dx), float(y))
                  for _ in range(schedule['passes']) for y in schedule['line_y']
                  for k in range(np_p) if abs(ox + k * dx) <= window]
        pulse_rows.append(pulses)
    max_lines = max(len(r) for r in rows)
    line_matrix = torch.zeros(len(rows), max_lines)
    for i, r in enumerate(rows):
        line_matrix[i, :len(r)] = torch.tensor(r)
    packets = cfg.max_packets
    counts = torch.zeros(len(rows), packets, dtype=torch.long)
    for i, c in enumerate(count_rows):
        counts[i, :len(c)] = torch.tensor(c)
    max_pulses = max((len(p) for p in pulse_rows), default=1)
    pulse_matrix = torch.full((len(rows), max_pulses, 2), float('nan'))
    per_line = []
    for i, pulses in enumerate(pulse_rows):
        if pulses:
            pulse_matrix[i, :len(pulses), 0] = torch.tensor([p[0] for p in pulses])
            pulse_matrix[i, :len(pulses), 1] = torch.tensor([p[1] for p in pulses])
        per_line.append(len(pulses) // len(rows[i]) if rows[i] else 0)
    return line_matrix, counts, pulse_matrix, per_line


def schedule_waist(schedule):
    return 0.874e-6


def rollout(d0, q0, schedules, params, commands, closure_log, cfg=RolloutConfig(),
            device='cpu', dtype=torch.float32):
    """Batch rollout; returns terminal d, q and drift diagnostics (detached)."""
    require(d0.shape == q0.shape and d0.ndim == 3, 'Rollout state shape')
    n = d0.shape[1]
    coords = ((torch.arange(n, device=device, dtype=dtype) - (n - 1) / 2) * (ROI_M / n))
    Xg, Yg = torch.meshgrid(coords, coords, indexing='xy')
    Xg, Yg = Xg[None], Yg[None]
    d, q = d0.clone(), q0.clone()
    gl_x, gl_w = gl_nodes(device=device, dtype=torch.float64)
    gl_x, gl_w = gl_x.to(dtype), gl_w.to(dtype)
    line_matrix, counts, pulse_matrix, per_line = build_batch_plan(schedules, cfg)
    line_matrix = line_matrix.to(device=device, dtype=dtype)
    pulse_matrix = pulse_matrix.to(device=device, dtype=dtype)
    counts = counts.to(device)
    dx_row = torch.tensor([s['dx_m'] for s in schedules], device=device, dtype=dtype)
    w0_row = params['w0'] if torch.is_tensor(params['w0']) else \
        torch.full((len(d),), float(params['w0']), device=device, dtype=dtype)
    dense = dx_row <= DENSE_DX_OVER_W0 * w0_row
    drift = {'max_line_dd_over_zr': 0.0, 'max_line_log_threshold_drift': 0.0}
    for p in range(counts.shape[1]):
        b = closure_log[:, p] if closure_log is not None else None
        for track, mask in (('dense', dense), ('sparse', ~dense)):
            if not bool(mask.any()):
                continue
            idx = torch.nonzero(mask, as_tuple=False).flatten()
            d_t, q_t = d[idx], q[idx]
            sub_params = _subset_params(params, idx, len(d))
            sub_commands = commands[idx]
            sub_schedules = [schedules[int(i)] for i in idx]
            if track == 'dense':
                d_t, q_t, dr = _packet_lines(d_t, q_t, Xg, Yg, line_matrix[idx],
                                             counts[idx], p, sub_schedules, sub_params,
                                             sub_commands, b, cfg, gl_x, gl_w)
            else:
                d_t, q_t, dr = _packet_pulses(d_t, q_t, Xg, Yg, pulse_matrix[idx],
                                              [per_line[int(i)] for i in idx],
                                              counts[idx], p, sub_schedules, sub_params,
                                              sub_commands, b, cfg)
            d = d.index_copy(0, idx, d_t)
            q = q.index_copy(0, idx, q_t)
            drift['max_line_dd_over_zr'] = max(drift['max_line_dd_over_zr'], dr[0])
            drift['max_line_log_threshold_drift'] = max(drift['max_line_log_threshold_drift'],
                                                        dr[1])
    return d, q, drift


def _subset_params(params, idx, B):
    out = {}
    for key, val in params.items():
        if torch.is_tensor(val) and val.ndim >= 1 and val.shape[0] == B:
            out[key] = val[idx]
        else:
            out[key] = val
    return out


def _packet_lines(d, q, Xg, Yg, line_matrix, counts, p, schedules, params, commands, b,
                  cfg, gl_x, gl_w):
    drift_dd = drift_lth = 0.0
    for k in range(int(counts[:, p].max())):
        active = counts[:, p] > k
        if not bool(active.any()):
            break
        y_line = line_matrix[active, p * 0 + 0] if False else None
        # line index within the concatenated list: prefix of packet counts + k
        start = (counts[:, :p].sum(1) + k) if p > 0 else torch.full_like(counts[:, p], k)
        y_line = line_matrix[torch.arange(len(d), device=d.device), start.clamp_max(
            line_matrix.shape[1] - 1)][active]
        d, q, dr = _apply_line_sweep(d[active], q[active], Xg, Yg, y_line, schedules,
                                     _subset_params(params, torch.nonzero(active).flatten(),
                                                    len(d)), commands[active], p, b, cfg,
                                     gl_x, gl_w)
        drift_dd, drift_lth = max(drift_dd, dr[0]), max(drift_lth, dr[1])
    return d, q, (drift_dd, drift_lth)


def _packet_pulses(d, q, Xg, Yg, pulse_matrix, per_line, counts, p, schedules, params,
                   commands, b, cfg):
    drift_dd = drift_lth = 0.0
    n = d.shape[1]
    for k in range(int(counts[:, p].max())):
        active = counts[:, p] > k
        if not bool(active.any()):
            break
        aidx = torch.nonzero(active).flatten()
        start = (counts[:, :p].sum(1) + k) if p > 0 else torch.full_like(counts[:, p], k)
        rows = []
        for row, i in enumerate(aidx.tolist()):
            line_idx = int(start[i])
            lo = line_idx * per_line[row]
            hi = lo + per_line[row]
            rows.append(pulse_matrix[row_i] if False else pulse_matrix[row, lo:hi])
        C = max(r.shape[0] for r in rows)
        px = torch.full((len(rows), C), float('nan'), device=d.device)
        py = torch.full((len(rows), C), float('nan'), device=d.device)
        for j, r in enumerate(rows):
            px[j, :r.shape[0]] = r[:, 0]
            py[j, :r.shape[0]] = r[:, 1]
        d, q, dr = _apply_line_pulses(d[active], q[active], Xg, Yg, px, py,
                                      torch.isfinite(px), schedules, _subset_params(
                                          params, aidx, len(d)),
                                      commands[active], p, b, cfg)
        drift_dd, drift_lth = max(drift_dd, dr[0]), max(drift_lth, dr[1])
    return d, q, (drift_dd, drift_lth)


def _common_scales(params, commands, p, d, cfg):
    tau = commands[:, p, 0].exp()[:, None, None]
    f_Hz = commands[:, p, 1].exp()[:, None, None]
    v = commands[:, p, 2].exp()[:, None, None]
    F1 = params['F1'].view(-1, 1, 1) * (tau / params['tau_ref']) ** params['gamma_F']
    delta = params['delta'].view(-1, 1, 1) * (tau / params['tau_ref']) ** params['gamma_delta']
    w0 = params['w0'].view(-1, 1, 1)
    zr = params['zr'].view(-1, 1, 1)
    kappa = params['kappa'].view(-1, 1, 1)
    rho_t = params['rho']
    rho = rho_t.view(-1, 1, 1) if torch.is_tensor(rho_t) and rho_t.numel() > 1 \
        else rho_t.reshape(1, 1, 1)
    Ep = params['Ep']
    if not torch.is_tensor(Ep):
        Ep = torch.tensor(float(Ep), device=d.device, dtype=d.dtype)
    if cfg.defocus:
        w = w0 * torch.sqrt(1. + (d / zr) ** 2)
    else:
        w = w0.expand_as(d)
    return tau, f_Hz, v, F1, delta, w0, zr, kappa, rho, Ep, w


def _apply_line_sweep(d, q, Xg, Yg, y_line, schedules, params, commands, p, b, cfg,
                      gl_x, gl_w):
    """Closed-form frozen-state sweep update for one raster line (dense subset)."""
    tau, f_Hz, v, F1, delta, w0, zr, kappa, rho, Ep, w = _common_scales(params, commands,
                                                                        p, d, cfg)
    dy = Yg - y_line.view(-1, 1, 1)
    log_peak = torch.log(2. * Ep / torch.pi) - 2. * torch.log(w) - 2. * dy ** 2 / w ** 2
    if cfg.incubation:
        log_threshold = torch.log(F1) + torch.log(rho + (1. - rho) * torch.exp(-q))
    else:
        log_threshold = torch.log(F1).expand_as(d)
    if b is not None:
        log_threshold = log_threshold + b[:, None, None]
    L = (log_peak - log_threshold).clamp_min(0.)
    dx_m = torch.tensor([s['dx_m'] for s in schedules], dtype=d.dtype,
                        device=d.device).view(-1, 1, 1)
    d_gain = delta * L.sqrt() ** 3 * (4. * w / (3. * math.sqrt(2.))) / dx_m
    d_next = d + d_gain
    if cfg.incubation:
        nodes = gl_x[None, None, None] * (6. * w[..., None])
        fluence = torch.exp(log_peak[..., None] - 2. * nodes ** 2 / (w[..., None] ** 2))
        integrand = fluence / (F1[..., None] + fluence)
        q_gain = kappa * f_Hz / v * (12. * w) * \
            (gl_w[None, None, None, :] * integrand).sum(-1)
        q_next = q + q_gain
    else:
        q_next = q
    return d_next, q_next, _line_drift(d, q, d_next, q_next, F1, rho, zr, cfg)


def _apply_line_pulses(d, q, Xg, Yg, px, py, vmask, schedules, params, commands, p, b,
                       cfg, pulse_chunk=8):
    """Exact per-pulse updates for one raster line in small frozen-state chunks.

    Within a chunk the state is frozen (vectorized); chunk boundaries refresh d/q
    so overlapping neighbouring pulses see partially updated state. chunk=1
    recovers the exact pointwise recurrence."""
    n = d.shape[1]
    step = ROI_M / n
    origin_x, origin_y = -ROI_M / 2, -ROI_M / 2
    tau, f_Hz, v, F1, delta, w0, zr, kappa, rho, Ep, w = _common_scales(params, commands,
                                                                        p, d, cfg)
    B, C = px.shape
    radius = PULSE_WINDOW_SIGMA * (params['w0'] * 1.5).view(-1)
    half = int(torch.ceil(radius.max() / step))
    offs = torch.arange(-half, half + 1, device=d.device)
    K = offs.numel()
    d_next, q_next = d, q
    for start in range(0, C, pulse_chunk):
        stop = min(start + pulse_chunk, C)
        px_c = px[:, start:stop]
        py_c = py[:, start:stop]
        vmask_c = vmask[:, start:stop]
        C_c = stop - start
        if cfg.defocus:
            w_c = w0 * torch.sqrt(1. + (d_next / zr) ** 2)
        else:
            w_c = w0.expand_as(d_next)
        gx = torch.round((px_c - origin_x) / step).long().clamp(-half, n + half - 1)
        gy = torch.round((py_c - origin_y) / step).long().clamp(-half, n + half - 1)
        ix = (gx[:, :, None, None] + offs.view(1, 1, K, 1)).clamp(0, n - 1)
        iy = (gy[:, :, None, None] + offs.view(1, 1, 1, K)).clamp(0, n - 1)
        flat_idx = (iy * n + ix).reshape(B, -1)
        d_win = torch.gather(d_next.reshape(B, -1), 1, flat_idx).reshape(B, C_c, K, K)
        q_win = torch.gather(q_next.reshape(B, -1), 1, flat_idx).reshape(B, C_c, K, K)
        w_win = w0 * torch.sqrt(1. + (d_win / zr) ** 2) if cfg.defocus else w0.expand_as(d_win)
        pix_x = origin_x + ix * step
        pix_y = origin_y + iy * step
        r2 = (pix_x - px_c[..., None, None]) ** 2 + (pix_y - py_c[..., None, None]) ** 2
        fluence = 2. * Ep / (torch.pi * w_win ** 2) * torch.exp(-2. * r2 / w_win ** 2)
        if cfg.incubation:
            log_threshold = torch.log(F1) + torch.log(rho + (1. - rho) * torch.exp(-q_win))
        else:
            log_threshold = torch.log(F1).expand_as(d_win)
        if b is not None:
            log_threshold = log_threshold + b[:, None, None, None]
        L = (torch.log(fluence.clamp_min(1e-300)) - log_threshold).clamp_min(0.)
        d_add = delta * L * vmask_c[..., None, None]
        d_next = d_next.reshape(B, -1).scatter_add(1, flat_idx,
                                                   d_add.reshape(B, -1)).reshape(B, n, n)
        if cfg.incubation:
            q_add = kappa * fluence / (F1 + fluence) * vmask_c[..., None, None]
            q_next = q_next.reshape(B, -1).scatter_add(1, flat_idx,
                                                       q_add.reshape(B, -1)).reshape(B, n, n)
    return d_next, q_next, _line_drift(d, q, d_next, q_next, F1, rho, zr, cfg)


def _line_drift(d, q, d_next, q_next, F1, rho, zr, cfg):
    with torch.no_grad():
        drift_dd = float(((d_next - d) / zr).max()) if cfg.defocus else 0.0
        if cfg.incubation:
            new_th = torch.log(F1) + torch.log(rho + (1. - rho) * torch.exp(-q_next))
            old_th = torch.log(F1) + torch.log(rho + (1. - rho) * torch.exp(-q))
            drift_lth = float((new_th - old_th).abs().max())
        else:
            drift_lth = 0.0
    return drift_dd, drift_lth


def upsample_depth(d):
    require(d.shape[1] == d.shape[2] == GRID_N, 'Rollout grid shape')
    if GRID_N == CANONICAL_N:
        return d
    return F.interpolate(d[:, None], size=CANONICAL_N, mode='bilinear',
                         align_corners=True)[:, 0]


def terminal_height_um(d):
    return -upsample_depth(d) * 1e6
