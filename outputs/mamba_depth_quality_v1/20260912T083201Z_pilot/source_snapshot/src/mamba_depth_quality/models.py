import torch
from torch import nn
from .mamba_adapter import MambaAdapter

class TerminalModel(nn.Module):
    def __init__(self,input_dim=16,encoder="mamba2",bottleneck=4,aux=True):
        super().__init__(); self.encoder_name=encoder; self.proj=nn.Linear(input_dim,32)
        if encoder=="mamba2": self.encoder=MambaAdapter()
        elif encoder=="gru": self.encoder=nn.GRU(32,32,batch_first=True)
        elif encoder=="static": self.encoder=nn.Sequential(nn.Linear(input_dim,64),nn.SiLU(),nn.Linear(64,32))
        else: raise ValueError(encoder)
        self.z=nn.Linear(32,bottleneck); self.depth=nn.Linear(bottleneck,1); self.quality=nn.Linear(bottleneck,1)
        self.aux=nn.Linear(bottleneck,2) if aux else None
        nn.init.zeros_(self.depth.weight); nn.init.zeros_(self.depth.bias)
    def forward(self,x,mask=None,d_base=None,s_d=1.,s_q=1.):
        if self.encoder_name=="static": h=self.encoder(x[:,-1] if x.ndim==3 else x)
        elif self.encoder_name=="gru": h=self.encoder(self.proj(x))[0]; h=h[:,-1]
        else:
            seq=self.encoder(self.proj(x),mask)
            if mask is None: h=seq[:,-1]
            else:
                idx=mask.to(torch.long).sum(dim=1).clamp_min(1)-1
                h=seq[torch.arange(seq.shape[0],device=seq.device),idx]
        z=self.z(h); d=(d_base if d_base is not None else 0.)+3*s_d*torch.tanh(self.depth(z).squeeze(-1)); q=s_q*torch.nn.functional.softplus(self.quality(z).squeeze(-1))
        return {"D":d,"Sq":q,"z":z,"aux":self.aux(z) if self.aux is not None else None}
