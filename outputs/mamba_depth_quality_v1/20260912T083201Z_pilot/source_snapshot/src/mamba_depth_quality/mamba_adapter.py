import torch
from src.task_state_learning.mamba2 import Mamba2Config, Mamba2Mixer

class MambaAdapter(torch.nn.Module):
    def __init__(self,d_model=32,d_state=8,expand=2,head_dim=16,n_groups=1,d_conv=4,chunk_size=32,seed=0):
        super().__init__(); self.input_dim=None
        self.cfg=Mamba2Config(d_model=d_model,expand=expand,head_dim=head_dim,d_state=d_state,n_groups=n_groups,d_conv=d_conv,chunk=chunk_size)
        self.mixer=Mamba2Mixer(self.cfg,seed=seed)
    def forward(self,x,mask=None):
        if mask is None: return self.mixer.full_forward(x)
        # Keep padded tokens from changing the recurrent state; zeroing output alone is insufficient.
        state=self.mixer.initial_cache(x.shape[0],x.device,x.dtype); ys=[]
        for t in range(x.shape[1]):
            y,new=self.mixer.step(x[:,t],state); active=mask[:,t].view(-1,1)
            state={k:torch.where(active.view(-1,*([1]*(v.ndim-1))),new[k],state[k]) for k,v in state.items()}
            ys.append(torch.where(active,y,torch.zeros_like(y)))
        return torch.stack(ys,1)
    def forward_stepwise(self,x,mask=None):
        return self.forward(x,mask)
