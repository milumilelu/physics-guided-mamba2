"""Frozen raster-scan exposure construction and single-pass rollouts (numeric fixture).

The scenario is a registered E01 numeric fixture, not measured scan provenance:
unidirectional row-major raster, centered, single pass, no turn exposure, no guard
region. Pulse counts are frozen per registered point from nominal pitches so the
event topology stays fixed while positions/energy vary smoothly for AD/FD.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from .grouping import require
from . import physics as ref_physics
from . import differentiable_physics as grad_physics


@dataclass(frozen=True)
class ScanFixture:
    """Registered numeric fixture; scenario assumptions are explicit, not measured truth."""
    region_m: float
    grid_n: int
    power_W: float
    n_x_max: int
    n_y_max: int
    total_pulses_max: int
    passes: int = 1

    def __post_init__(self):
        require(self.region_m > 0 and self.grid_n >= 8 and self.power_W > 0, 'Invalid scan fixture')
        require(self.n_x_max >= 2 and self.n_y_max >= 2 and self.total_pulses_max >= 4, 'Invalid pulse budget')
        require(self.passes == 1, 'Only the registered single-pass fixture is implemented')


def pulse_counts(fixture, dx_m, hatch_m):
    require(np.isfinite([dx_m, hatch_m]).all() and dx_m > 0 and hatch_m > 0, 'Nonpositive pitch')
    n_x = max(2, int(round(fixture.region_m / dx_m)))
    n_y = max(2, int(round(fixture.region_m / hatch_m)))
    require(n_x <= fixture.n_x_max, f'Pulse budget exceeded: n_x={n_x}')
    require(n_y <= fixture.n_y_max, f'Pulse budget exceeded: n_y={n_y}')
    require(n_x * n_y <= fixture.total_pulses_max, 'Total pulse budget exceeded')
    return n_x, n_y


def pulse_positions(n_x, n_y, dx_m, hatch_m):
    """Unidirectional row-major raster centered on the grid; no turn exposure."""
    xs = (np.arange(n_x) - (n_x - 1) / 2.0) * dx_m
    ys = (np.arange(n_y) - (n_y - 1) / 2.0) * hatch_m
    x, y = np.meshgrid(xs, ys)
    return np.column_stack([x.ravel(), y.ravel()])


def split_packets(positions, energy_J, m):
    """Temporal sub-pulse refinement: m sub-pulses at one position share its energy."""
    require(isinstance(m, int) and m >= 1, 'Invalid packet split')
    positions = np.asarray(positions, dtype=float)
    if np.ndim(energy_J) == 0:
        energy = np.full(len(positions), float(energy_J))
    else:
        energy = np.asarray(energy_J, dtype=float)
        require(len(energy) == len(positions), 'Pulse count mismatch')
    if m == 1:
        return positions.copy(), energy.copy()
    return np.repeat(positions, m, axis=0), np.repeat(energy / m, m)


def rollout_numpy(positions, energy_J, duration_s, parameters, fixture, switch='P11'):
    positions = np.asarray(positions, dtype=float)
    energy_J = np.asarray(energy_J, dtype=float)
    require(len(positions) == len(energy_J), 'Pulse count mismatch')
    require(np.isfinite(energy_J).all() and (energy_J >= 0).all(), 'Invalid pulse energy')
    coords = (np.arange(fixture.grid_n) - (fixture.grid_n - 1) / 2.0) * (fixture.region_m / fixture.grid_n)
    x, y = np.meshgrid(coords, coords)
    d = np.zeros((fixture.grid_n, fixture.grid_n))
    q = np.zeros((fixture.grid_n, fixture.grid_n))
    for k in range(len(positions)):
        d, q = ref_physics.step(d, q, x, y, pulse_energy_J=float(energy_J[k]), duration_s=float(duration_s),
                                beam_x_m=float(positions[k, 0]), beam_y_m=float(positions[k, 1]),
                                parameters=parameters, switch=switch)
    return d, q


def rollout_torch(log_tau_s, log_f_Hz, log_v_m_s, log_h_m, parameters, fixture, n_x, n_y, switch='P11'):
    """Differentiable single-pass rollout; energy and positions stay inside the autograd graph."""
    energy = fixture.power_W * torch.exp(-log_f_Hz)
    duration = torch.exp(log_tau_s)
    dx = torch.exp(log_v_m_s - log_f_Hz)
    hatch = torch.exp(log_h_m)
    xs = (torch.arange(n_x, dtype=torch.float64) - (n_x - 1) / 2.0) * dx
    ys = (torch.arange(n_y, dtype=torch.float64) - (n_y - 1) / 2.0) * hatch
    beam_x = xs.tile(n_y)
    beam_y = ys.repeat_interleave(n_x)
    coords = (torch.arange(fixture.grid_n, dtype=torch.float64) - (fixture.grid_n - 1) / 2.0) * (fixture.region_m / fixture.grid_n)
    x, y = torch.meshgrid(coords, coords, indexing='xy')
    d = torch.zeros((fixture.grid_n, fixture.grid_n), dtype=torch.float64)
    q = torch.zeros((fixture.grid_n, fixture.grid_n), dtype=torch.float64)
    for k in range(n_x * n_y):
        d, q = grad_physics.step(d, q, x, y, pulse_energy_J=energy, duration_s=duration,
                                 beam_x_m=beam_x[k], beam_y_m=beam_y[k],
                                 parameters=parameters, switch=switch)
    return d, q


def descriptors(depth):
    d = np.asarray(depth, dtype=float)
    return {'mean_depth_m': float(d.mean()),
            'depth_rms_m': float(np.sqrt(np.mean((d - np.median(d)) ** 2)))}
