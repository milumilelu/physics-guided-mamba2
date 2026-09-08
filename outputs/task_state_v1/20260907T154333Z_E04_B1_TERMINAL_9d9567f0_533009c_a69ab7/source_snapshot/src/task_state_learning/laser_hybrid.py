"""Laser hybrid: physics rollout + bounded memory closure (E06/E07 shared module).

Token contract (runbook 10.2, 16 features, current predicted state + command
only; no N, no measured morphology, no specimen identifiers):
  log_tau, log_f, log_v, log_h, pass_index/10, log_packet_dt, log_pulse_count,
  log_packet_energy, predicted_mean_depth_um, predicted_depth_rms_um,
  mean_q, mean_d_over_zR, mean_h_over_2w, mean_dx_over_2w, packet_line_count,
  packet_active.
The closure outputs b = log(2) * tanh(readout) and enters ONLY through
F_th_eff = F_th * exp(b); closure=0 recovers the registered physics exactly.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .grouping import require
from .memory import make_memory
from .laser_rollout import (RolloutConfig, apply_packet, build_batch_plan, gl_nodes,
                            terminal_height_um, ROI_M, GRID_N)

TOKEN_DIM = 16
B_MAX = math.log(2.0)


def packet_tokens(d, q, commands, p, counts, schedules, w0, zr, packet_features):
    """Build the (B, 16) token for packet p from the CURRENT predicted state."""
    flat_d = d.reshape(len(d), -1)
    flat_q = q.reshape(len(q), -1)
    mean_d = flat_d.mean(1, keepdim=True)
    rms_d = flat_d.pow(2).mean(1, keepdim=True).sqrt()
    mean_q = flat_q.mean(1, keepdim=True)
    dt_packet = packet_features['packet_dt'][:, p:p + 1]
    pulse_count = packet_features['pulse_count'][:, p:p + 1]
    dx = packet_features['dx'][:, None]
    h = commands[:, p, 3].exp()[:, None]
    active = (counts[:, p] > 0).float()[:, None]
    line_count = counts[:, p].float()[:, None]
    pass_index = packet_features['pass_index'][:, p:p + 1] / 10.0
    tokens = torch.cat([
        commands[:, p, 0:4],
        pass_index, torch.log(dt_packet.clamp_min(1e-9)), torch.log(pulse_count.clamp_min(1)),
        torch.log((pulse_count * commands[:, p, 1].exp()).clamp_min(1) *
                  torch.tensor(0.0 + 1.0)[:, None].to(commands.dtype)),
        mean_d * 1e6, rms_d * 1e6, mean_q, mean_d / zr[:, None],
        h / (2 * w0[:, None]), dx / (2 * w0[:, None]), line_count / 16.0, active,
    ], dim=1)
    require(tokens.shape[1] == TOKEN_DIM, f'Token width {tokens.shape[1]} != {TOKEN_DIM}')
    return tokens


class LaserHybrid(nn.Module):
    def __init__(self, memory_id='MAMBA2', width=32, d_state=16, seed=0,
                 cfg=RolloutConfig()):
        super().__init__()
        self.memory_id = memory_id
        self.cfg = cfg
        self.encoder = nn.Sequential(nn.Linear(TOKEN_DIM, width), nn.SiLU())
        self.memory = make_memory(memory_id, width, d_state=d_state, seed=seed)
        self.readout = nn.Sequential(nn.Linear(width + TOKEN_DIM, width), nn.SiLU(),
                                     nn.Linear(width, 1))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, d0, q0, schedules, params, commands, packet_features,
                line_matrix, counts, offsets, Xg, Yg, gl_x, gl_w):
        """Interleaved memory/physics rollout; returns b sequence and terminal field.

        Every independent specimen starts with zero memory cache and zero
        physical state (d0/q0 zeros supplied by the caller).
        """
        B = len(d0)
        device = d0.device
        dtype = d0.dtype
        cache = self.memory.initial(B, device, dtype)
        d, q = d0.clone(), q0.clone()
        bs = []
        for p in range(counts.shape[1]):
            token = packet_tokens(d, q, commands, p, counts, schedules,
                                  params['w0'], params['zr'], packet_features)
            local = self.encoder(token)
            output, next_cache = self.memory.step(local, torch.zeros(len(d), device=device,
                                                                     dtype=dtype), cache)
            raw = self.readout(torch.cat((output, token), dim=-1))
            b = B_MAX * torch.tanh(raw[:, 0])
            bs.append(b)
            if isinstance(next_cache, dict):
                cache = {k: v for k, v in next_cache.items()}
            else:
                cache = next_cache
            if bool((counts[:, p] > 0).any()):
                d, q, _ = apply_packet(d, q, Xg, Yg, line_matrix, counts, offsets, p,
                                       schedules, params, commands, b.detach() * 0 +
                                       b, self.cfg, gl_x, gl_w)
        return torch.stack(bs, dim=1), d, q

    def budget(self, counts_cols, dtype=torch.float32):
        cache = self.memory.initial(1, torch.device('cpu'), dtype)
        cache_scalars = sum(int(v.numel()) for v in cache.values()) if cache else 0
        return {'n_physical_state_scalars': 2 * GRID_N * GRID_N,
                'n_ssm_or_recurrent_cache_scalars': cache_scalars,
                'n_conv_cache_scalars': sum(int(v.numel()) for v in cache.values())
                if isinstance(cache, dict) and 'conv' in cache else 0,
                'n_other_persistent_scalars': 0,
                'persistent_state_bytes': (2 * GRID_N * GRID_N + cache_scalars) *
                                          torch.empty((), dtype=dtype).element_size(),
                'n_trainable_parameters': sum(p.numel() for p in self.parameters()
                                              if p.requires_grad)}
