"""B1 objective blocks: terminal loss, decision-gap pair loss, closure regularizer.

Scales for the DP objective come from the TRAINING split only. Anchors are
deterministic farthest-point medoids of training targets (max 20); pairs are
sampled from distinct families with a fixed per-epoch seed stream. lambda_DP=0.25
and lambda_closure=1e-3 are frozen algorithm constants, not material constants.
"""
import numpy as np
import torch
from .grouping import require

LAMBDA_DP = 0.25
LAMBDA_CLOSURE = 1e-3
N_ANCHORS = 20
PAIRS_PER_EPOCH = 256


def terminal_scale(terminal_x):
    """Per-component scale from the current training split only."""
    return terminal_x.std(dim=0).clamp_min(1e-12)


def farthest_point_anchors(terminal_x, scale, count=N_ANCHORS):
    """Deterministic medoid start + farthest-point sweep; train labels only."""
    require(count >= 2 and len(terminal_x) >= count, 'Insufficient families for anchors')
    z = (terminal_x - terminal_x.mean(dim=0)) / scale
    distance = torch.cdist(z, z)
    medoid = int(distance.sum(dim=1).argmin())
    chosen = [medoid]
    while len(chosen) < count:
        nearest = distance[:, chosen].min(dim=1).values
        next_point = int(nearest.argmax())
        require(next_point not in chosen, 'Anchor sweep revisited a point')
        chosen.append(next_point)
    return terminal_x[chosen].clone()


class PairSampler:
    """Deterministic per-epoch unordered pairs of distinct families (fixed stream)."""

    def __init__(self, n_families, pairs=PAIRS_PER_EPOCH, seed=20260907):
        require(n_families >= 2, 'Need two families for pairs')
        self.n = n_families
        self.pairs = pairs
        self.rng = np.random.default_rng(seed)
        self.epoch = 0

    def sample(self):
        rng = np.random.default_rng(self.rng.integers(2 ** 63 - 1))
        i = rng.integers(0, self.n, self.pairs)
        j = rng.integers(0, self.n - 1, self.pairs)
        j = np.where(j >= i, j + 1, j)
        self.epoch += 1
        return torch.as_tensor(np.stack([i, j], axis=1), dtype=torch.long)


def decision_gap_loss(pred, true, anchors, scale, pairs):
    """Pairwise ranking of predicted selection toward the true ordering per anchor.

    pred/true: (F, 2) terminal predictions/labels; anchors (A, 2); pairs (P, 2).
    Loss per (pair, anchor): softplus( (Jhat_i - Jhat_j) * sign(J_i - J_j) ),
    J_t(y) = ||y - a_t||^2 in train-scale units; pairs with equal truth are skipped.
    """
    require(pairs.ndim == 2 and pairs.shape[1] == 2, 'Pair index shape')
    require(len(anchors) >= 1, 'Anchor set empty')
    a = anchors / scale
    pi, pj = pred[pairs[:, 0]] / scale, pred[pairs[:, 1]] / scale
    ti, tj = true[pairs[:, 0]] / scale, true[pairs[:, 1]] / scale
    ji = (ti[:, None, :] - a[None, :, :]).pow(2).sum(-1)
    jj = (tj[:, None, :] - a[None, :, :]).pow(2).sum(-1)
    d_hat = (pi[:, None, :] - a[None, :, :]).pow(2).sum(-1) - (pj[:, None, :] - a[None, :, :]).pow(2).sum(-1)
    sign = torch.sign(ji - jj)
    keep = sign != 0
    if not bool(keep.any()):
        return pred.new_zeros(())
    return torch.nn.functional.softplus(d_hat[keep] * sign[keep]).mean()


def terminal_loss(pred, true):
    return (pred - true).pow(2).mean()


def total_loss(pred, true, anchors, scale, pairs, closure_sq=None):
    loss = terminal_loss(pred, true)
    parts = {'terminal': float(loss)}
    if pairs is not None:
        dp = decision_gap_loss(pred, true, anchors, scale, pairs)
        loss = loss + LAMBDA_DP * dp
        parts['dp'] = float(dp)
    if closure_sq is not None:
        loss = loss + LAMBDA_CLOSURE * closure_sq
        parts['closure'] = float(closure_sq)
    return loss, parts
