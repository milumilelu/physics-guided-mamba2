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
        rows=[]
        for i in range(len(control)):
            valid=mask[i]
            count=int(valid.sum())
            require(count>0 and count%self.blocks==0,'Active history must divide into eight blocks')
            values=control[i,valid].reshape(self.blocks,count//self.blocks,2)
            durations=physical_dt[i,valid]
            require(bool((durations>0).all()),'Invalid physical duration')
            mean=values.mean(dim=1)
            std=values.var(dim=1,unbiased=False).clamp_min(0).sqrt()
            # Complete registered eight-block history, independent of batch padding.
            extras=torch.stack((durations.new_tensor(count/128.),durations.log().mean(),durations.sum().log()))
            rows.append(torch.cat((mean.flatten(),std.flatten(),extras)))
        feats=torch.stack(rows)
        require(feats.shape[1] == self.blocks * 4 + 3, 'Static feature width mismatch')
        return feats

    def forward(self, control, physical_dt, mask):
        return self.net(self.features(control, physical_dt, mask))
