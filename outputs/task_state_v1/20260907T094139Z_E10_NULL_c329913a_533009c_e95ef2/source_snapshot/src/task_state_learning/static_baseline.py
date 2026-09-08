"""Static baseline: pooled full control history + mask/dt statistics (equal info).

Receives the same information the recurrent models see (runbook 28.4), as
pre-frozen summary channels: per-block u1/u2 means/stds, active-token fraction,
mean log-dt and total log-duration. No within-block ordering is used beyond the
registered eight-block segmentation, which the recurrent models also receive.
"""
import torch
from torch import nn
from .grouping import require

BLOCKS = 8


class StaticMLP(nn.Module):
    def __init__(self, width=64, out=2, blocks=BLOCKS):
        super().__init__()
        self.blocks = blocks
        self.net = nn.Sequential(nn.Linear(blocks * 4 + 3, width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU(),
                                 nn.Linear(width, out))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def features(self, control, physical_dt, mask):
        require(control.ndim == 3 and control.shape[:2] == physical_dt.shape == mask.shape,
                'Static feature shape')
        require(mask.dtype == torch.bool, 'Static mask dtype')
        b, t, _ = control.shape
        require(t % self.blocks == 0, 'Token count must divide into control blocks')
        n = t // self.blocks
        cm = mask.view(b, self.blocks, n)
        safe = torch.where(mask[..., None], control, torch.zeros_like(control))
        cb = safe.view(b, self.blocks, n, 2)
        valid = cm.float()[..., None]
        counts = valid.sum(2).clamp_min(1)
        mean = (cb * valid).sum(2) / counts
        std = (((cb - mean[:, :, None, :]) ** 2) * valid).sum(2).div(counts).clamp_min(0).sqrt()
        active_frac = cm.float().mean(dim=(1, 2), keepdim=False).reshape(b, 1)
        log_dt = torch.log(physical_dt.clamp_min(1e-12))
        mean_log_dt = torch.where(mask, log_dt, torch.zeros_like(log_dt)).mean(dim=1).reshape(b, 1)
        total_log_T = torch.where(mask, log_dt, torch.zeros_like(log_dt)).sum(dim=1).reshape(b, 1)
        feats = torch.cat([mean.flatten(1), std.flatten(1), active_frac, mean_log_dt,
                           total_log_T], dim=-1)
        require(feats.shape[1] == self.blocks * 4 + 3, 'Static feature width mismatch')
        return feats

    def forward(self, control, physical_dt, mask):
        return self.net(self.features(control, physical_dt, mask))
