"""B1 resolved skeleton with bounded additive closure; no true hidden inputs."""
import torch
from torch import nn
from .memory import make_memory


def b1_resolved_step(x,u,dt,b):
    """Exact resolved linear-in-x dynamics for held u and held closure over dt."""
    e1=torch.exp(-.4*dt);e2=torch.exp(-.7*dt)
    equilibrium=(u[:,0]+b[:,0])/.4
    nx1=equilibrium+(x[:,0]-equilibrium)*e1
    forcing=.5*u[:,1]+b[:,1]+.1*u[:,0]*equilibrium
    nx2=x[:,1]*e2+forcing*(-torch.expm1(-.7*dt))/.7
    nx2=nx2+.1*u[:,0]*(x[:,0]-equilibrium)*(e1-e2)/.3
    return torch.stack((nx1,nx2),dim=-1)


class B1ClosureModel(nn.Module):
    def __init__(self,memory_id,width=32):
        super().__init__();self.memory_id=memory_id;self.width=width
        self.encoder=nn.Sequential(nn.Linear(5,width),nn.SiLU())
        self.memory=make_memory(memory_id,width)
        self.readout=nn.Sequential(nn.Linear(width+5,width),nn.SiLU(),nn.Linear(width,2))
        nn.init.zeros_(self.readout[-1].weight);nn.init.zeros_(self.readout[-1].bias)

    def initial(self,batch,device,dtype):
        return {'x':torch.zeros(batch,2,device=device,dtype=dtype),
                'memory':self.memory.initial(batch,device,dtype)}

    def forward(self,control,physical_dt,mask,*,initial_state=None,return_state=False):
        if control.ndim!=3 or control.shape[-1]!=2 or control.shape[:2]!=physical_dt.shape or mask.shape!=physical_dt.shape:
            raise ValueError('Invalid B1 token arrays')
        if mask.dtype!=torch.bool or not bool(torch.isfinite(control).all() and torch.isfinite(physical_dt).all()):
            raise ValueError('Invalid B1 token values')
        if not bool((physical_dt[mask]>0).all()):raise ValueError('Invalid physical dt')
        state=self.initial(len(control),control.device,control.dtype) if initial_state is None else initial_state
        x,cache=state['x'],state['memory']
        for k in range(control.shape[1]):
            active=mask[:,k,None]
            dt=torch.where(mask[:,k],physical_dt[:,k],torch.zeros_like(physical_dt[:,k]))
            local=torch.cat((x,control[:,k],torch.log(dt.clamp_min(1e-12))[:,None]),dim=-1)
            output,next_cache=self.memory.step(self.encoder(local),dt,cache)
            b=2*torch.tanh(self.readout(torch.cat((output,local),dim=-1)))
            next_x=b1_resolved_step(x,control[:,k],dt,b)
            x=torch.where(active,next_x,x)
            if cache is not None:cache=torch.where(active,next_cache,cache)
        result={'x':x,'memory':cache}
        return result if return_state else x

    def budget(self,dtype=torch.float32):
        cache=0 if self.memory_id=='H0' else self.width
        return {'n_physical_state_scalars':2,'n_ssm_or_recurrent_cache_scalars':cache,
                'n_conv_cache_scalars':0,'n_other_persistent_scalars':0,
                'persistent_state_bytes':(2+cache)*torch.empty((),dtype=dtype).element_size(),
                'n_trainable_parameters':sum(p.numel() for p in self.parameters() if p.requires_grad)}
