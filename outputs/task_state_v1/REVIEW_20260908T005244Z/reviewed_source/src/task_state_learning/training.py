"""Shared training loop: AdamW, grad accumulation, early stopping on validation.

Protocol constants (runbook 11.4): weight_decay=1e-4, grad clip=1, batch=4,
effective batch=16 (4-step accumulation), max_epochs=300, patience=40.
Terminal loss accumulates over batch=4 micro-batches; the decision-gap pair
loss and the bounded-closure regularizer are epoch-level objectives computed on
one full training forward per epoch (runbook 11.3 allows up to 256 pairs/epoch).
Outer refits use a frozen epoch count, no early stopping, final-epoch state.
"""
import math
import numpy as np
import torch
from .grouping import require
from .losses import decision_gap_loss, LAMBDA_DP, LAMBDA_CLOSURE

WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
BATCH = 16
ACCUM = 1
MAX_EPOCHS = 300
PATIENCE = 40
EVAL_CHUNK = 64


def _batches(n, batch, generator):
    order = torch.randperm(n, generator=generator)
    for start in range(0, n, batch):
        yield order[start:start + batch]


def _forward_chunks(model, control, dt, mask, chunk=EVAL_CHUNK):
    outs = []
    for start in range(0, len(control), chunk):
        outs.append(model(control[start:start + chunk], dt[start:start + chunk],
                          mask[start:start + chunk]))
    return torch.cat(outs)


def fit(model, train, val, *, lr, seed, device='cpu', max_epochs=MAX_EPOCHS,
        patience=PATIENCE, batch=BATCH, accum=ACCUM, dp=None,
        fixed_epochs=None, return_history=False):
    """dp: dict(anchors, scale, sampler) or None. Returns fitted model + bookkeeping."""
    require(lr > 0 and batch >= 1 and accum >= 1, 'Invalid optimization budget')
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    tensors = {k: v.to(device) for k, v in train.items()}
    val_tensors = {k: v.to(device) for k, v in val.items()}
    generator = torch.Generator().manual_seed(seed)
    best = {'epoch': -1, 'val_loss': math.inf, 'state': None}
    history = []
    total_epochs = fixed_epochs if fixed_epochs is not None else max_epochs
    require(total_epochs >= 1, 'Epoch budget must be positive')
    for epoch in range(total_epochs):
        model.train()
        optimizer.zero_grad()
        steps = 0
        for indices in _batches(len(tensors['control']), batch, generator):
            sub = {k: v[indices] for k, v in tensors.items()}
            pred = model(sub['control'], sub['physical_dt'], sub['mask'])
            loss = (pred - sub['terminal_x']).pow(2).sum()
            (loss / accum).backward()
            steps += 1
            if steps % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()
        if steps % accum:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()
        if dp is not None:
            pairs = dp['sampler'].sample().to(device)
            anchors = dp['anchors'].to(device)
            dp_scale = dp['scale'].to(device)
            pred_full = _forward_chunks(model, tensors['control'], tensors['physical_dt'],
                                        tensors['mask'])
            dp_loss = decision_gap_loss(pred_full, tensors['terminal_x'], anchors,
                                        dp_scale, pairs)
            (LAMBDA_DP * dp_loss).backward()
            optimizer.step()
            optimizer.zero_grad()
        if hasattr(model, 'closure_sq'):
            closure = model.closure_sq(tensors['control'], tensors['physical_dt'],
                                       tensors['mask'])
            (LAMBDA_CLOSURE * closure).backward()
            optimizer.step()
            optimizer.zero_grad()
        model.eval()
        with torch.no_grad():
            val_pred = _forward_chunks(model, val_tensors['control'], val_tensors['physical_dt'],
                                       val_tensors['mask'])
            val_loss = float((val_pred - val_tensors['terminal_x']).pow(2).mean())
        history.append({'epoch': epoch, 'val_loss': val_loss})
        if fixed_epochs is None:
            if val_loss < best['val_loss']:
                best = {'epoch': epoch, 'val_loss': val_loss,
                        'state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            if epoch - best['epoch'] >= patience:
                break
    if fixed_epochs is None:
        require(best['state'] is not None, 'Training produced no evaluation point')
        model.load_state_dict(best['state'])
        selected = best['epoch']
    else:
        selected = total_epochs - 1
    model = model.cpu()
    out = {'model': model, 'best_epoch': selected, 'val_loss': best['val_loss'],
           'history': history if return_history else None}
    return out


def predict(model, data, device='cpu', batch=EVAL_CHUNK):
    model = model.to(device).eval()
    outs = []
    with torch.no_grad():
        for start in range(0, len(data['control']), batch):
            sub = {k: v[start:start + batch].to(device) for k, v in data.items()}
            outs.append(model(sub['control'], sub['physical_dt'], sub['mask']).cpu())
    model.cpu()
    return torch.cat(outs)
