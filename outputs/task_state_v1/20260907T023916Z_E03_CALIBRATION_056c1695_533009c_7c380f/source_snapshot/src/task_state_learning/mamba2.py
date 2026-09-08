"""Pure-PyTorch Mamba-2 (SSD) reference backend; registered numeric model.

mamba_backend = "reference_pure_torch_ssd_v1" (Dao & Gu 2024 SSD parameterization).

Registered architecture (frozen; deviations must create a new model_id):
  d_inner = expand * d_model; n_heads = d_inner / head_dim; n_groups = 1.
  in_proj: Linear(d_model -> d_inner + 2*n_groups*d_state + n_heads, bias=False)
           producing [z, xBC, dt_raw].
  Causal conv1d (kernel=d_conv, left zero padding, groups=xbc_dim, bias=True)
           on xBC, then SiLU. dt is NOT convolved: dt = softplus(dt_proj(dt_raw)
           + dt_bias); dt_proj init = I*(1+0.001*U[-1,1]), dt_bias =
           inverse_softplus(Uniform[dt_min, dt_max]) (official dt_init).
  A = -exp(A_log), A_log init = log(1..n_heads); D = ones(n_heads).
  Discretization per head: a_t = exp(dt_t * A_h).
  Recurrence: s_t = a_t * s_{t-1} + dt_t * (B_t (x) u_t),  s: (d_state, head_dim);
           y_t = C_t . s_t + D_h * u_t.  B/C shared across heads (n_groups=1).
  Output: out_proj( (y * silu(z)) ), y reshaped to d_inner.
The physical dt enters through the caller's tokenizer; the mixer learns its own
gating timescale (registered behavior, same as official Mamba-2).

Two independent evaluations of the same registered recurrence must agree:
`full_forward` (chunk-parallel scan) and `naive_recurrent` (sequential oracle).
Official fused selective-scan/Triton kernels are unavailable on this platform
(Windows, no CUDA build); the backend identity, torch version and source
snapshot hash are sealed in the backend gate.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from .grouping import require


@dataclass(frozen=True)
class Mamba2Config:
    d_model: int = 32
    expand: int = 2
    head_dim: int = 16
    d_state: int = 16
    n_groups: int = 1
    d_conv: int = 4
    chunk: int = 64
    dt_min: float = 0.001
    dt_max: float = 0.1

    def __post_init__(self):
        require(all(isinstance(v, int) and v > 0 for v in
                    [self.d_model, self.expand, self.head_dim, self.d_state,
                     self.n_groups, self.d_conv, self.chunk]), 'Invalid Mamba2 dimension')
        require((self.d_model * self.expand) % self.head_dim == 0, 'd_inner must divide head_dim')
        require(0 < self.dt_min < self.dt_max, 'Invalid dt range')
        require(self.d_state % self.n_groups == 0, 'd_state must be divisible by n_groups')

    @property
    def d_inner(self):
        return self.d_model * self.expand

    @property
    def n_heads(self):
        return self.d_inner // self.head_dim

    @property
    def xbc_dim(self):
        return self.d_inner + 2 * self.n_groups * self.d_state


class Mamba2Mixer(nn.Module):
    """Single-layer Mamba-2 mixer with identical full-sequence and step semantics."""

    def __init__(self, cfg: Mamba2Config, seed=0):
        super().__init__()
        self.cfg = cfg
        gen = torch.Generator().manual_seed(int(seed))
        self.in_proj = nn.Linear(cfg.d_model, 2 * cfg.d_inner + 2 * cfg.n_groups * cfg.d_state + cfg.n_heads,
                                 bias=False)
        self.conv1d = nn.Conv1d(cfg.xbc_dim, cfg.xbc_dim, cfg.d_conv, groups=cfg.xbc_dim,
                                bias=True)
        self.dt_proj = nn.Linear(cfg.n_heads, cfg.n_heads, bias=False)
        with torch.no_grad():
            perturb = torch.rand(cfg.n_heads, generator=gen) * 2 - 1
            self.dt_proj.weight.copy_(torch.eye(cfg.n_heads) * (1 + 0.001 * perturb[:, None]))
        dt0 = torch.empty(cfg.n_heads).uniform_(cfg.dt_min, cfg.dt_max, generator=gen)
        self.dt_bias = nn.Parameter(torch.log(torch.expm1(dt0)))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, cfg.n_heads + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(cfg.n_heads))
        nn.init.kaiming_uniform_(self.conv1d.weight, a=math.sqrt(5))
        bound = cfg.d_conv ** -0.5
        nn.init.uniform_(self.conv1d.bias, -bound, bound)
        nn.init.kaiming_uniform_(self.in_proj.weight, a=math.sqrt(5))
        self.out_proj = nn.Linear(cfg.d_inner, cfg.d_model, bias=False)
        nn.init.kaiming_uniform_(self.out_proj.weight, a=math.sqrt(5))

    def A(self):
        return -torch.exp(self.A_log)

    def project(self, x):
        parts = self.in_proj(x)
        return torch.split(parts, [self.cfg.d_inner, self.cfg.xbc_dim, self.cfg.n_heads], dim=-1)

    def initial_cache(self, batch, device, dtype):
        c = self.cfg
        return {'conv': torch.zeros(batch, c.d_conv - 1, c.xbc_dim, device=device, dtype=dtype),
                'ssm': torch.zeros(batch, c.n_heads, c.n_groups, c.d_state, c.head_dim,
                                   device=device, dtype=dtype)}

    def step(self, x, cache):
        """Single-token inference with explicit cache; returns output and next cache."""
        c = self.cfg
        z, xbc_raw, dt_raw = self.project(x)
        window = torch.cat((cache['conv'], xbc_raw[:, None]), dim=1)
        xbc = F.silu(F.conv1d(window.transpose(1, 2), self.conv1d.weight, self.conv1d.bias,
                              groups=c.xbc_dim)[..., -1])
        next_cache_conv = window[:, 1:]
        dt = F.softplus(self.dt_proj(dt_raw) + self.dt_bias)
        u, Bv, Cv = self._split(xbc)
        a = torch.exp(dt[:, :, None, None, None] * self.A()[None, :, None, None, None])
        ssm = a * cache['ssm'] + dt[:, :, None, None, None] * torch.einsum('bgp,bhq->bhgpq', Bv, u)
        y = torch.einsum('bgp,bhgpq->bhq', Cv, ssm) + self.D[None, :, None] * u
        out = self.out_proj(y.reshape(-1, c.d_inner) * F.silu(z))
        return out, {'conv': next_cache_conv, 'ssm': ssm}

    def _split(self, xbc):
        c = self.cfg
        u = xbc[..., :c.d_inner].view(-1, c.n_heads, c.head_dim)
        rest = xbc[..., c.d_inner:]
        Bv, Cv = rest.split(c.n_groups * c.d_state, dim=-1)
        return u, Bv[:, None, :], Cv[:, None, :]

    def full_forward(self, x):
        """Parallel chunked SSD over a full sequence (B, T, d_model) -> (B, T, d_model)."""
        c = self.cfg
        B, T, _ = x.shape
        z, xbc_raw, dt_raw = self.project(x)
        conv = F.pad(xbc_raw.transpose(1, 2), (c.d_conv - 1, 0))
        conv = self.conv1d(conv)[..., :T].transpose(1, 2)
        xbc = F.silu(conv)
        dt = F.softplus(self.dt_proj(dt_raw) + self.dt_bias)
        u = xbc[..., :c.d_inner].reshape(B, T, c.n_heads, c.head_dim)
        Bv, Cv = xbc[..., c.d_inner:c.d_inner + c.n_groups * c.d_state], \
            xbc[..., c.d_inner + c.n_groups * c.d_state:]
        Bv, Cv = Bv.reshape(B, T, c.n_groups, c.d_state), Cv.reshape(B, T, c.n_groups, c.d_state)
        ys = []
        state = torch.zeros(B, c.n_heads, c.n_groups, c.d_state, c.head_dim,
                            dtype=x.dtype, device=x.device)
        A, D = self.A().to(x.dtype), self.D.to(x.dtype)
        for start in range(0, T, c.chunk):
            stop = min(start + c.chunk, T)
            ys.append(_ssd_chunk(u[:, start:stop], Bv[:, start:stop], Cv[:, start:stop],
                                 dt[:, start:stop], A, D, state))
            state = _chunk_final_state(u[:, start:stop], Bv[:, start:stop],
                                       dt[:, start:stop], A, state)
        y = torch.cat(ys, dim=1)
        return self.out_proj(y.reshape(B, T, c.d_inner) * F.silu(z))

    def naive_forward(self, x, cache=None):
        """Sequential oracle: per-token recurrence, independent of the chunk scan."""
        B, T, _ = x.shape
        state = self.initial_cache(B, x.device, x.dtype) if cache is None else cache
        outputs = []
        for t in range(T):
            out, state = self.step(x[:, t], state)
            outputs.append(out)
        return torch.stack(outputs, dim=1), state


def naive_recurrent(uh, bh, ch, dt, A, D):
    """Tensor-level recurrence oracle on pre-split (B,T,H,Q) inputs; mirror of step()."""
    b, t, h, q = uh.shape
    g, p = bh.shape[2], bh.shape[3]
    s = torch.zeros(b, h, g, p, q, dtype=uh.dtype, device=uh.device)
    ys = []
    for k in range(t):
        a = torch.exp(dt[:, k, :, None, None, None] * A[None, :, None, None, None])
        s = a * s + dt[:, k, :, None, None, None] * torch.einsum('bgp,bhq->bhgpq', bh[:, k], uh[:, k])
        ys.append(torch.einsum('bgp,bhgpq->bhq', ch[:, k], s) + D[None, :, None] * uh[:, k])
    return torch.stack(ys, dim=1)


def _decay_matrix(dtd, A):
    """L[b,i,j,h] = exp(sum_{k=j+1..i} dt_k A_h); L[i,i]=1, j>i masked to 0."""
    log_a = dtd * A[None, None, :]
    cum = torch.cumsum(log_a, dim=1)
    diff = cum[:, :, None, :] - cum[:, None, :, :]
    length = dtd.shape[1]
    lower = torch.ones(length, length, dtype=torch.bool, device=dtd.device).tril()
    return torch.exp(diff.masked_fill(~lower[None, :, :, None], -torch.inf))


def _ssd_chunk(ud, bd, cdd, dtd, A, D, state_in):
    """One chunk: intra-chunk decayed contribution + initial-state carry."""
    decay = _decay_matrix(dtd, A)
    cu = dtd[..., None] * ud
    bu = torch.einsum('btgp,bthq->bthgpq', bd, cu)
    dec_bu = torch.einsum('bijh,bjhgpq->bihgpq', decay, bu)
    y_intra = torch.einsum('btgp,bthgpq->bthq', cdd, dec_bu)
    cum = torch.cumsum(dtd * A[None, None, :], dim=1)
    carry = torch.einsum('bth,bhgpq->bthgpq', torch.exp(cum), state_in)
    y_state = torch.einsum('btgp,bthgpq->bthq', cdd, carry)
    return y_intra + y_state + D[None, None, :, None] * ud


def _chunk_final_state(ud, bd, dtd, A, state_in):
    cum = torch.cumsum(dtd * A[None, None, :], dim=1)
    total = cum[:, -1]
    tail = torch.exp(total[:, None, :] - cum)
    cu = dtd[..., None] * ud
    bu = torch.einsum('btgp,bthq->bthgpq', bd, cu)
    return torch.exp(total)[:, :, None, None, None] * state_in + \
        torch.einsum('bth,bthgpq->bhgpq', tail, bu)
