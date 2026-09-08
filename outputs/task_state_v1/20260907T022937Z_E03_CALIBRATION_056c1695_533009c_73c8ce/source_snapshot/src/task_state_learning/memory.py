"""Common causal memory API. Cache belongs to one history, never to the module."""
import torch
from torch import nn
import torch.nn.functional as F
from .mamba2 import Mamba2Config, Mamba2Mixer


class Instantaneous(nn.Module):
    def __init__(self,width=32):
        super().__init__();self.width=width
        self.net=nn.Sequential(nn.Linear(width,3*width),nn.SiLU(),nn.Linear(3*width,width))

    def initial(self,batch,device,dtype):return None

    def step(self,token,dt,state):return self.net(token),None


class GRUMemory(nn.Module):
    def __init__(self,width=32):
        super().__init__();self.width=width;self.cell=nn.GRUCell(width,width)

    def initial(self,batch,device,dtype):return torch.zeros(batch,self.width,device=device,dtype=dtype)

    def step(self,token,dt,state):
        updated=self.cell(token,state)
        return updated,updated


class ContinuousTimeSSM(nn.Module):
    """Diagonal relaxation with exact physical-dt decay and nonlinear forcing."""
    def __init__(self,width=32):
        super().__init__();self.width=width
        self.content=nn.Sequential(nn.Linear(width,3*width),nn.SiLU(),nn.Linear(3*width,width))
        self.log_rate=nn.Parameter(torch.zeros(width))

    def initial(self,batch,device,dtype):return torch.zeros(batch,self.width,device=device,dtype=dtype)

    def step(self,token,dt,state):
        decay=torch.exp(-F.softplus(self.log_rate)*dt[:,None])
        updated=decay*state+(1-decay)*self.content(token)
        return updated,updated


class Mamba2Memory(nn.Module):
    """Mamba-2 reference mixer behind the common per-token memory API.

    The physical dt is consumed by the caller's tokenizer; the mixer learns its
    own gating timescale (registered Mamba-2 behavior)."""
    def __init__(self, width=32, d_state=16, seed=0):
        super().__init__()
        self.cfg = Mamba2Config(d_model=width, d_state=d_state)
        self.mixer = Mamba2Mixer(self.cfg, seed=seed)

    def initial(self, batch, device, dtype):
        return self.mixer.initial_cache(batch, device, dtype)

    def step(self, token, dt, state):
        return self.mixer.step(token, state)


def make_memory(name, width=32, d_state=16, seed=0):
    options = {'H0': Instantaneous, 'GRU': GRUMemory, 'CTSSM': ContinuousTimeSSM}
    if name == 'MAMBA2':
        return Mamba2Memory(width, d_state=d_state, seed=seed)
    if name not in options:
        raise ValueError(f'Unimplemented memory model: {name}')
    return options[name](width)
