"""Mamba-2 reference backend parity; both evaluation paths of one registered math."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch.nn.functional as F
from src.task_state_learning.mamba2 import Mamba2Config, Mamba2Mixer, naive_recurrent


def sample(mixer, batch=3, length=37, dtype=torch.float64, seed=1):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(batch, length, mixer.cfg.d_model, generator=gen, dtype=dtype)


def split_full(mixer, x):
    """Registered projection/conv/split for tensor-level references."""
    c = mixer.cfg
    z, xbc_raw, dt_raw = mixer.project(x)
    conv = F.pad(xbc_raw.transpose(1, 2), (c.d_conv - 1, 0))
    conv = mixer.conv1d(conv)[..., :x.shape[1]].transpose(1, 2)
    xbc = F.silu(conv)
    dt = F.softplus(mixer.dt_proj(dt_raw) + mixer.dt_bias)
    u = xbc[..., :c.d_inner].reshape(x.shape[0], x.shape[1], c.n_heads, c.head_dim)
    rest = xbc[..., c.d_inner:]
    Bv, Cv = rest.split(c.n_groups * c.d_state, dim=-1)
    Bv = Bv.reshape(x.shape[0], x.shape[1], c.n_groups, c.d_state)
    Cv = Cv.reshape(x.shape[0], x.shape[1], c.n_groups, c.d_state)
    return z, u, Bv, Cv, dt


class Mamba2Parity(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_full_vs_step_cache_float64(self):
        mixer = Mamba2Mixer(Mamba2Config()).double()
        x = sample(mixer, seed=2)
        full = mixer.full_forward(x)
        naive, _ = mixer.naive_forward(x)
        error = (full - naive).abs().max().item() / naive.abs().max().item()
        self.assertLessEqual(error, 1e-10)

    def test_full_vs_tensor_recurrence_float64(self):
        mixer = Mamba2Mixer(Mamba2Config()).double()
        x = sample(mixer, length=91, seed=3)
        z, u, Bv, Cv, dt = split_full(mixer, x)
        y = naive_recurrent(u, Bv, Cv, dt, mixer.A().double(), mixer.D.double())
        reference = mixer.out_proj(y.reshape(x.shape[0], x.shape[1], mixer.cfg.d_inner) * F.silu(z))
        full = mixer.full_forward(x)
        error = (full - reference).abs().max().item() / reference.abs().max().item()
        self.assertLessEqual(error, 1e-10)

    def test_cache_continuation_equals_full(self):
        mixer = Mamba2Mixer(Mamba2Config()).double()
        x = sample(mixer, length=50, seed=4)
        full = mixer.full_forward(x)
        first, cache = mixer.naive_forward(x[:, :17])
        second, _ = mixer.naive_forward(x[:, 17:], cache=cache)
        split = torch.cat((first, second), dim=1)
        error = (full - split).abs().max().item() / full.abs().max().item()
        self.assertLessEqual(error, 1e-10)

    def test_samples_are_independent_with_zero_cache(self):
        mixer = Mamba2Mixer(Mamba2Config()).double()
        x = sample(mixer, batch=2, length=30, seed=5)
        single0, _ = mixer.naive_forward(x[:1])
        single1, _ = mixer.naive_forward(x[1:])
        batched = mixer.full_forward(x)
        scale = batched.abs().max().item()
        error = max((batched[:1] - single0).abs().max().item(),
                    (batched[1:] - single1).abs().max().item())
        self.assertLessEqual(error / scale, 1e-10)

    def test_lti_limit_matches_closed_form(self):
        """Constant input => constant u/B/C/dt; closed-form exponential response."""
        cfg = Mamba2Config()
        mixer = Mamba2Mixer(cfg).double()
        with torch.no_grad():
            mixer.conv1d.weight.zero_()
            mixer.conv1d.weight[:, :, -1] = 1.0
            mixer.conv1d.bias.zero_()
        x = torch.full((1, 24, cfg.d_model), 0.3, dtype=torch.float64)
        z, u, Bv, Cv, dt = split_full(mixer, x)
        scale = u.abs().max().item()
        self.assertGreater(scale, 0.0)
        self.assertLessEqual((dt - dt[0, 0]).abs().max().item(), 1e-14)
        self.assertLessEqual((u - u[0, 0]).abs().max().item(), 1e-14)
        self.assertLessEqual((Bv - Bv[0, 0]).abs().max().item(), 1e-14)
        self.assertLessEqual((Cv - Cv[0, 0]).abs().max().item(), 1e-14)
        a = torch.exp(dt[0, 0] * mixer.A())
        hs = []
        h = torch.zeros(cfg.n_heads, cfg.n_groups, cfg.d_state, cfg.head_dim, dtype=torch.float64)
        for i in range(24):
            h = a[:, None, None, None] * h + \
                dt[0, 0][:, None, None, None] * torch.einsum('gp,hq->hgpq', Bv[0, 0], u[0, i])
            y = torch.einsum('gp,hgpq->hq', Cv[0, 0], h) + mixer.D[:, None] * u[0, i]
            hs.append(y.reshape(-1))
        analytic = torch.stack(hs)
        full = mixer.full_forward(x)
        gated = mixer.out_proj(analytic[None] * F.silu(z))
        error = (full - gated).abs().max().item() / gated.abs().max().item()
        self.assertLessEqual(error, 1e-10)

    def test_float32_training_path_finite(self):
        mixer = Mamba2Mixer(Mamba2Config()).float()
        x = sample(mixer, dtype=torch.float32, seed=6)
        full = mixer.full_forward(x)
        step, _ = mixer.naive_forward(x)
        self.assertTrue(torch.isfinite(full).all())
        error = (full - step).abs().max().item() / step.abs().max().item()
        self.assertLessEqual(error, 1e-4)

    def test_gradient_flows_to_early_tokens(self):
        mixer = Mamba2Mixer(Mamba2Config()).double()
        x = sample(mixer, length=33, seed=7).requires_grad_(True)
        out = mixer.full_forward(x)
        out[:, 0].sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().max().item(), 0.0)

    def test_config_budget_reported(self):
        cfg = Mamba2Config()
        mixer = Mamba2Mixer(cfg).double()
        cache = mixer.initial_cache(1, torch.device('cpu'), torch.float32)
        scalars = sum(v.numel() for v in cache.values())
        self.assertEqual(scalars, (cfg.d_conv - 1) * cfg.xbc_dim + cfg.n_heads * cfg.d_state * cfg.head_dim)


if __name__ == '__main__':
    unittest.main()
