#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mechanism-guided recurrent feature surrogate v5.2 for ultrafast laser depth prediction and leakage-free virtual-data benchmarking.

Purpose
-------
This script implements a measured-depth-only pipeline:

1) Rows without measured depth are retained only for prediction-only diagnostics.
2) A compact physical recurrent skeleton generates pass-level mechanism features.
3) Data-driven observation mappers translate these physical sequences/features into
   area-averaged depth.
4) Multiple mappers are evaluated under repeated 5-fold CV on measured rows:
   - low-dimensional GBDT/RF/ExtraTrees/GPR baselines
   - raw, physics-augmented, and compact proxy feature tabular models
   - tree-based observation mappers with compact physics-derived features
   - engineered raw-process baselines with simple process-interaction terms
   - optional tiny neural models: MLP / GRU / Transformer, if PyTorch is available

This v5 script intentionally avoids the older recurrent eta correction, censored loss,
shallow gate, and depth-efficiency decay terms. The physical skeleton is used as a
feature generator, not as an over-parameterized end-to-end predictor. Compared with v1,
it adds compact coverage/dose proxy features, tuned tree-model variants, repeated CV,
leave-one-feature-out ablation, group-level feature ablation, residual diagnostics, and publication-oriented interpretability figures.
It also retains optional fold-wise physics fitting for no-leakage strict checks.

Expected input columns
----------------------
Required columns, with flexible aliases handled where possible:
    run_id
    pulse_width_fs
    repetition_rate_khz
    scan_speed_mm_s
    hatch_spacing_um
    pass_count
    measured_depth_um

Example
-------
python depth_mechanism_sequence_surrogate.py \
  --input ./predictions_depth_only_v2_60rows_for_tuning.csv \
  --output-dir ./outputs/depth_mechanism_sequence_surrogate \
  --physics-fit-scope fixed \
  --models gbdt,extra_trees,random_forest,gpr,ridge \
  --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RepeatedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None


# -----------------------------
# Utility functions
# -----------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out: Dict[str, float] = {}
    if len(y_true) == 0:
        for k in ["rmse", "mae", "r2"]:
            out[prefix + k] = float("nan")
        return out
    out[prefix + "rmse"] = rmse(y_true, y_pred)
    out[prefix + "mae"] = float(mean_absolute_error(y_true, y_pred))
    out[prefix + "r2"] = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan")
    return out


def markdown_table(df: pd.DataFrame, max_rows: Optional[int] = None) -> str:
    if df is None or len(df) == 0:
        return "\n_(empty)_\n"
    sub = df.copy()
    if max_rows is not None and len(sub) > max_rows:
        sub = sub.head(max_rows)
    sub = sub.replace({np.nan: ""})
    cols = list(sub.columns)
    lines = []
    lines.append("| " + " | ".join(str(c) for c in cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in sub.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                if math.isfinite(v):
                    vals.append(f"{v:.6g}")
                else:
                    vals.append("")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    if max_rows is not None and len(df) > max_rows:
        lines.append(f"\n_Only first {max_rows} of {len(df)} rows are shown._")
    return "\n" + "\n".join(lines) + "\n"


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{sec:02d}s"


# -----------------------------
# Data schema
# -----------------------------

COLUMN_ALIASES = {
    "run_id": ["run_id", "id", "sample_id", "编号"],
    "pulse_width_fs": ["pulse_width_fs", "pulse_width", "tau_fs", "脉宽", "脉冲宽度"],
    "repetition_rate_khz": ["repetition_rate_khz", "rep_rate_khz", "frequency_khz", "重复频率", "重复频率Khz", "重复频率_khz"],
    "scan_speed_mm_s": ["scan_speed_mm_s", "scan_speed", "speed_mm_s", "加工速度", "扫描速度", "扫描速度mm/s"],
    "hatch_spacing_um": ["hatch_spacing_um", "hatch_spacing", "line_spacing_um", "填充间距", "扫描层间距", "层间距", "hatch"],
    "pass_count": ["pass_count", "passes", "scan_count", "加工次数", "加工轮次", "pass数", "轮次"],
    "measured_depth_um": ["measured_depth_um", "depth_um", "average_depth_um", "平均深度", "深度", "depth"],
}


def resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename flexible input aliases to canonical column names."""
    out = df.copy()
    lower_to_original = {str(c).strip().lower(): c for c in out.columns}
    rename_map = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in out.columns:
            continue
        for alias in aliases:
            key = alias.strip().lower()
            if key in lower_to_original:
                rename_map[lower_to_original[key]] = canonical
                break
    out = out.rename(columns=rename_map)
    missing = [c for c in COLUMN_ALIASES if c not in out.columns and c != "run_id"]
    if missing:
        raise ValueError(f"Input CSV is missing required columns after alias resolution: {missing}. Available columns: {list(df.columns)}")
    if "run_id" not in out.columns:
        out["run_id"] = np.arange(1, len(out) + 1)
    return out


def read_depth_csv(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")
    df = pd.read_csv(input_path)
    df = resolve_columns(df)
    required_numeric = ["pulse_width_fs", "repetition_rate_khz", "scan_speed_mm_s", "hatch_spacing_um", "pass_count", "measured_depth_um"]
    for c in required_numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["pass_count"] = df["pass_count"].round().astype("Int64")
    df["has_measured_depth"] = df["measured_depth_um"].notna().astype(int)
    df["is_problematic_prediction_only"] = (df["has_measured_depth"] == 0).astype(int)
    df["problematic_note"] = ""
    df.loc[df["is_problematic_prediction_only"] == 1, "problematic_note"] = (
        "missing measured_depth_um; excluded from training/CV/loss/model selection"
    )
    return df


# -----------------------------
# Physics skeleton
# -----------------------------

@dataclass
class OpticalConfig:
    actual_post_objective_average_power_w: float = 5.33333
    wavelength_nm: float = 1030.0
    objective_NA: float = 0.45
    M2: float = 1.2

    @property
    def wavelength_um(self) -> float:
        return self.wavelength_nm / 1000.0

    @property
    def beam_radius_um_1e2(self) -> float:
        # Gaussian waist estimate from filled objective aperture with M^2 correction.
        return self.M2 * self.wavelength_um / (math.pi * self.objective_NA)

    @property
    def spot_diameter_um_1e2(self) -> float:
        return 2.0 * self.beam_radius_um_1e2

    @property
    def rayleigh_length_um(self) -> float:
        # In air; M^2 degrades effective Rayleigh range.
        return math.pi * self.beam_radius_um_1e2 ** 2 / (self.M2 * self.wavelength_um)


@dataclass
class PhysicsParams:
    # Effective fluence threshold and incubation law.
    phi_th1_j_cm2: float = 16.3967
    S: float = 0.79652
    # Optical scale correction. Default can be fixed at 1 by args; this value only applies if learnable/full historical.
    c_w: float = 1.0
    # Inter-pass / shared defocus depth-to-focus coupling.
    alpha_d: float = 0.779889
    # Pulse-width-conditioned effective ablation scale in um.
    delta_eff_200fs_um: float = 0.043899
    delta_eff_500fs_um: float = 0.0450921
    delta_eff_1000fs_um: float = 0.0284155
    delta_eff_2000fs_um: float = 0.0164486
    delta_eff_4000fs_um: float = 0.0333915

    def delta_for_tau(self, tau_fs: float) -> float:
        # Nearest-key mapping. Robust to 223 fs being closest to 200 fs.
        keys = np.array([200.0, 500.0, 1000.0, 2000.0, 4000.0])
        vals = np.array([
            self.delta_eff_200fs_um,
            self.delta_eff_500fs_um,
            self.delta_eff_1000fs_um,
            self.delta_eff_2000fs_um,
            self.delta_eff_4000fs_um,
        ])
        idx = int(np.argmin(np.abs(keys - float(tau_fs))))
        return float(vals[idx])


def bounded_sigmoid(raw: Any, lo: float, hi: float):
    return lo + (hi - lo) * torch.sigmoid(raw)


class TorchPhysicsModel(nn.Module if TORCH_AVAILABLE else object):
    """Tiny differentiable physics-only model for optional full/fold fitting.

    It does not contain eta_net. It only fits effective physical parameters so that
    pure recurrent physics final depth roughly matches measured rows. This is used
    only to generate physical features; downstream mappers still learn observation
    mapping separately.
    """

    def __init__(self, args: argparse.Namespace, optical: OpticalConfig, init: PhysicsParams, unique_taus: Sequence[float]):
        super().__init__()
        self.args = args
        self.optical = optical
        # inverse-logit helper
        def inv_sigmoid_from_value(v: float, lo: float, hi: float) -> float:
            t = (v - lo) / (hi - lo)
            t = min(max(t, 1e-5), 1 - 1e-5)
            return math.log(t / (1 - t))
        self.raw_phi = nn.Parameter(torch.tensor(inv_sigmoid_from_value(init.phi_th1_j_cm2, args.phi_min, args.phi_max), dtype=torch.float32))
        self.raw_S = nn.Parameter(torch.tensor(inv_sigmoid_from_value(init.S, args.S_min, args.S_max), dtype=torch.float32))
        self.raw_alpha = nn.Parameter(torch.tensor(inv_sigmoid_from_value(init.alpha_d, args.alpha_min, args.alpha_max), dtype=torch.float32))
        if args.cw_mode == "learnable":
            self.raw_cw = nn.Parameter(torch.tensor(inv_sigmoid_from_value(init.c_w, args.cw_min, args.cw_max), dtype=torch.float32))
        else:
            self.register_buffer("fixed_cw", torch.tensor(args.fixed_cw, dtype=torch.float32))
            self.raw_cw = None
        self.tau_keys = list(unique_taus)
        init_delta = []
        for tau in self.tau_keys:
            init_delta.append(init.delta_for_tau(tau))
        raw_delta = [inv_sigmoid_from_value(v, args.delta_min, args.delta_max) for v in init_delta]
        self.raw_delta = nn.Parameter(torch.tensor(raw_delta, dtype=torch.float32))

    def params(self) -> Dict[str, Any]:
        phi = bounded_sigmoid(self.raw_phi, self.args.phi_min, self.args.phi_max)
        S = bounded_sigmoid(self.raw_S, self.args.S_min, self.args.S_max)
        alpha = bounded_sigmoid(self.raw_alpha, self.args.alpha_min, self.args.alpha_max)
        if self.raw_cw is None:
            cw = self.fixed_cw
        else:
            cw = bounded_sigmoid(self.raw_cw, self.args.cw_min, self.args.cw_max)
        delta = bounded_sigmoid(self.raw_delta, self.args.delta_min, self.args.delta_max)
        return {"phi": phi, "S": S, "alpha": alpha, "cw": cw, "delta": delta}

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        p = self.params()
        D = torch.zeros_like(x["pulse_width_fs"], dtype=torch.float32)
        max_pass = int(torch.max(x["pass_count"]).item())
        for k in range(1, max_pass + 1):
            active = (x["pass_count"] >= k).float()
            if active.sum() <= 0:
                continue
            z_track, _ = torch_recursive_track_depth(x, D, p, self.tau_keys, self.optical, self.args)
            D = D + active * z_track
        return torch.clamp(D, min=0.0, max=self.args.max_depth_clip_um)


def torch_recursive_track_depth(
    x: Dict[str, Any],
    D_prev: Any,
    params: Dict[str, Any],
    tau_keys: Sequence[float],
    optical: OpticalConfig,
    args: argparse.Namespace,
) -> Tuple[Any, Dict[str, Any]]:
    # All tensors expected torch float32.
    f_hz = x["repetition_rate_khz"] * 1000.0
    v_um_s = x["scan_speed_mm_s"] * 1000.0
    pulse_energy_j = optical.actual_post_objective_average_power_w / torch.clamp(f_hz, min=1e-9)
    w0_um = optical.beam_radius_um_1e2
    zR_um = optical.rayleigh_length_um
    N_eff = 2.0 * w0_um * f_hz / torch.clamp(v_um_s, min=1e-9)
    steps = int(args.max_physics_steps)
    step_weight = torch.clamp(N_eff / max(1, steps), min=0.0, max=args.max_step_weight)
    z = torch.zeros_like(D_prev)
    # tau -> delta via nearest key, vectorized by masks.
    delta = torch.zeros_like(D_prev)
    tau = x["pulse_width_fs"]
    delta_values = params["delta"]
    for i, key in enumerate(tau_keys):
        # Assign by nearest original group with tolerance. Since tau_keys are exact unique from data.
        mask = torch.isclose(tau, torch.tensor(float(key), device=tau.device), rtol=0.0, atol=1e-3)
        delta = torch.where(mask, delta_values[i], delta)
    # Fallback for unmatched: nearest key by absolute distance.
    if len(tau_keys) > 0:
        all_keys = torch.tensor(list(tau_keys), device=tau.device, dtype=torch.float32)
        dists = torch.abs(tau[:, None] - all_keys[None, :])
        nearest_idx = torch.argmin(dists, dim=1)
        delta_nearest = delta_values[nearest_idx]
        delta = torch.where(delta > 0, delta, delta_nearest)

    margin_last = torch.zeros_like(D_prev)
    w_eff_last = torch.ones_like(D_prev) * w0_um
    for j in range(steps):
        pulse_index = 1.0 + (j + 0.5) * step_weight
        if args.defocus_mode == "history_only":
            total_defocus = params["alpha"] * D_prev
        elif args.defocus_mode == "shared":
            total_defocus = params["alpha"] * (D_prev + z)
        else:  # currently separate is approximated with same alpha for feature fitting unless script extended
            total_defocus = params["alpha"] * (D_prev + z)
        w_eff_um = params["cw"] * w0_um * torch.sqrt(1.0 + (total_defocus / max(zR_um, 1e-9)) ** 2)
        w_eff_cm = w_eff_um * 1e-4
        fluence = 2.0 * pulse_energy_j / (math.pi * torch.clamp(w_eff_cm, min=1e-12) ** 2)
        threshold = params["phi"] * torch.clamp(pulse_index, min=1.0) ** (params["S"] - 1.0)
        ratio = torch.clamp(fluence / torch.clamp(threshold, min=1e-9), min=1e-12, max=1e12)
        margin = torch.log(ratio)
        inc = delta * F.relu(margin) * step_weight
        z = torch.clamp(z + inc, min=0.0, max=args.max_depth_clip_um)
        margin_last = margin
        w_eff_last = w_eff_um
    aux = {"N_eff": N_eff, "step_weight": step_weight, "w_eff_last": w_eff_last, "margin_last": margin_last}
    return z, aux


def dataframe_to_torch(df: pd.DataFrame, device: str) -> Dict[str, Any]:
    cols = ["pulse_width_fs", "repetition_rate_khz", "scan_speed_mm_s", "hatch_spacing_um", "pass_count"]
    out = {}
    for c in cols:
        out[c] = torch.tensor(df[c].astype(float).values, dtype=torch.float32, device=device)
    return out


def fit_physics_params(
    measured_df: pd.DataFrame,
    args: argparse.Namespace,
    optical: OpticalConfig,
    init: PhysicsParams,
    device: str,
    label: str = "physics-fit",
) -> PhysicsParams:
    if not TORCH_AVAILABLE:
        print(f"[{label}] PyTorch not available; using fixed physics parameters.")
        return init
    if measured_df.empty:
        return init
    use_device = device
    if use_device == "cuda" and not torch.cuda.is_available():
        use_device = "cpu"
    unique_taus = sorted(float(x) for x in measured_df["pulse_width_fs"].dropna().unique())
    model = TorchPhysicsModel(args, optical, init, unique_taus).to(use_device)
    x = dataframe_to_torch(measured_df, use_device)
    y = torch.tensor(measured_df["measured_depth_um"].astype(float).values, dtype=torch.float32, device=use_device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.physics_lr, weight_decay=args.physics_weight_decay)
    best_loss = float("inf")
    best_state = None
    bad = 0
    start = time.time()
    for epoch in range(1, args.physics_epochs + 1):
        opt.zero_grad(set_to_none=True)
        pred = model(x)
        log_loss = torch.mean((torch.log1p(pred) - torch.log1p(y)) ** 2)
        lin_loss = F.smooth_l1_loss(pred / 50.0, y / 50.0)
        loss = log_loss + args.physics_lambda_linear * lin_loss
        if not torch.isfinite(loss):
            print(f"[{label}] non-finite loss at epoch {epoch}; stopping.")
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        cur = float(loss.detach().cpu().item())
        if cur < best_loss - 1e-6:
            best_loss = cur
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if args.show_progress and epoch % args.physics_print_every == 0:
            with torch.no_grad():
                pred_np = pred.detach().cpu().numpy()
                y_np = y.detach().cpu().numpy()
            print(f"[{label}] epoch {epoch}/{args.physics_epochs} loss={cur:.5g} best={best_loss:.5g} rmse={rmse(y_np, pred_np):.3f} elapsed={format_seconds(time.time()-start)}")
        if bad >= args.physics_patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    p = model.params()
    phi = float(p["phi"].detach().cpu().item())
    S = float(p["S"].detach().cpu().item())
    alpha = float(p["alpha"].detach().cpu().item())
    cw = float(p["cw"].detach().cpu().item())
    delta_vals = p["delta"].detach().cpu().numpy()
    # Map nearest canonical delta keys.
    params = PhysicsParams(phi_th1_j_cm2=phi, S=S, c_w=cw, alpha_d=alpha)
    for tau_key, val in zip(unique_taus, delta_vals):
        canonical = min([200, 500, 1000, 2000, 4000], key=lambda z: abs(z - tau_key))
        setattr(params, f"delta_eff_{canonical}fs_um", float(val))
    return params


def pure_physics_trace_for_row(
    row: pd.Series,
    params: PhysicsParams,
    optical: OpticalConfig,
    args: argparse.Namespace,
    max_pass: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate pass-level physical sequence features and mechanism-transition diagnostics.

    The base recurrent skeleton is unchanged. In addition to the historical pooled
    features, this function extracts three *derived event diagnostics* from the same
    recurrent variables:

    E1 -- ablation-margin cessation:
        g1 = log(Phi/Phi_th).  The event is the first + -> <= 0 crossing.
    E2 -- incubation/defocus dominance transition:
        g2 = (1-S) log(n) - log(1 + (z_defocus/z_R)^2).
        The event is the first + -> <= 0 crossing.
    E5 -- marginal-removal growth/decay transition:
        g5 = inc_j - inc_{j-1} within a pass.
        The event is the first + -> <= 0 crossing.

    Event times are normalized by the total number of compressed recurrent steps
    across active passes. They are diagnostics/virtual-data coordinates; they do not
    change the physical recursion itself.
    """
    tau = safe_float(row["pulse_width_fs"])
    f_khz = safe_float(row["repetition_rate_khz"])
    v_mm_s = safe_float(row["scan_speed_mm_s"])
    hatch = safe_float(row["hatch_spacing_um"])
    pass_count = int(row["pass_count"])
    f_hz = max(f_khz * 1000.0, 1e-12)
    v_um_s = max(v_mm_s * 1000.0, 1e-12)
    pulse_energy_j = optical.actual_post_objective_average_power_w / f_hz
    w0_um = optical.beam_radius_um_1e2
    zR_um = optical.rayleigh_length_um
    N_eff = 2.0 * w0_um * f_hz / v_um_s
    steps = int(args.max_physics_steps)
    step_weight = min(max(N_eff / max(1, steps), 0.0), args.max_step_weight)
    delta_eff = params.delta_for_tau(tau)

    D = 0.0
    tokens: List[List[float]] = []
    z_tracks: List[float] = []
    deltas: List[float] = []
    D_values: List[float] = []
    margin_values: List[float] = []
    w_values: List[float] = []
    defocus_values: List[float] = []

    # Mechanism-transition bookkeeping. The event dictionary stores only the first
    # physically ordered crossing of each type across the complete active process.
    event_pos: Dict[str, Optional[float]] = {"E1": None, "E2": None, "E5": None}
    event_pass: Dict[str, Optional[int]] = {"E1": None, "E2": None, "E5": None}
    event_pulse_index: Dict[str, Optional[float]] = {"E1": None, "E2": None, "E5": None}
    prev_margin_global: Optional[float] = None
    prev_g2_global: Optional[float] = None
    g1_min, g1_max = float("inf"), float("-inf")
    g2_min, g2_max = float("inf"), float("-inf")
    g5_min, g5_max = float("inf"), float("-inf")
    total_active_steps = max(1, pass_count * steps)
    global_active_step = 0

    for k in range(1, max_pass + 1):
        active = 1.0 if k <= pass_count else 0.0
        D_prev = D
        z = 0.0
        margin_mean_acc = 0.0
        active_margin_steps = 0
        w_eff_last = params.c_w * w0_um
        total_defocus_last = 0.0
        margin_last = -50.0

        # E5 is defined from successive pulse increments *within the same pass*.
        # Resetting here avoids an artificial derivative across a pass boundary.
        prev_inc_within_pass: Optional[float] = None
        prev_g5_within_pass: Optional[float] = None

        if active > 0:
            for j in range(steps):
                pulse_index = 1.0 + (j + 0.5) * step_weight
                if args.defocus_mode == "history_only":
                    total_defocus = params.alpha_d * D_prev
                elif args.defocus_mode == "shared":
                    total_defocus = params.alpha_d * (D_prev + z)
                else:
                    # Minimal script: separate mode uses same alpha for intra unless physics params are extended.
                    total_defocus = params.alpha_d * (D_prev + z)

                zeta = total_defocus / max(zR_um, 1e-12)
                w_eff_um = params.c_w * w0_um * math.sqrt(1.0 + zeta ** 2)
                w_eff_cm = max(w_eff_um * 1e-4, 1e-12)
                fluence = 2.0 * pulse_energy_j / (math.pi * w_eff_cm ** 2)
                threshold = params.phi_th1_j_cm2 * max(pulse_index, 1.0) ** (params.S - 1.0)
                ratio_f = min(max(fluence / max(threshold, 1e-12), 1e-12), 1e12)
                margin = math.log(ratio_f)
                inc = delta_eff * max(margin, 0.0) * step_weight

                # ----- Mechanism event functions -----
                # E1: the logarithmic ablation margin is exactly the positive-part
                # activation argument in the base ablation law.
                g1 = margin

                # E2: decompose the *dynamic* log-margin change into incubation gain
                # and defocus penalty. A constant optical scale c_w is deliberately
                # excluded because it is not a history-dependent competition term.
                incubation_gain = max(0.0, 1.0 - params.S) * math.log(max(pulse_index, 1.0))
                defocus_penalty = math.log1p(zeta ** 2)
                g2 = incubation_gain - defocus_penalty

                g1_min = min(g1_min, g1)
                g1_max = max(g1_max, g1)
                g2_min = min(g2_min, g2)
                g2_max = max(g2_max, g2)

                global_pos = (global_active_step + 0.5) / total_active_steps

                # E1: first active -> inactive threshold crossing.
                if (
                    event_pos["E1"] is None
                    and prev_margin_global is not None
                    and prev_margin_global > 0.0
                    and g1 <= 0.0
                ):
                    event_pos["E1"] = float(global_pos)
                    event_pass["E1"] = int(k)
                    event_pulse_index["E1"] = float(pulse_index)

                # E2: first incubation-dominant -> defocus-dominant crossing.
                if (
                    event_pos["E2"] is None
                    and prev_g2_global is not None
                    and prev_g2_global > 0.0
                    and g2 <= 0.0
                ):
                    event_pos["E2"] = float(global_pos)
                    event_pass["E2"] = int(k)
                    event_pulse_index["E2"] = float(pulse_index)

                # E5: first positive derivative of pulse increment followed by
                # non-positive derivative, evaluated only inside the current pass.
                if prev_inc_within_pass is not None:
                    g5 = inc - prev_inc_within_pass
                    g5_min = min(g5_min, g5)
                    g5_max = max(g5_max, g5)
                    if (
                        event_pos["E5"] is None
                        and prev_g5_within_pass is not None
                        and prev_g5_within_pass > 0.0
                        and g5 <= 0.0
                    ):
                        event_pos["E5"] = float(global_pos)
                        event_pass["E5"] = int(k)
                        event_pulse_index["E5"] = float(pulse_index)
                    prev_g5_within_pass = g5
                prev_inc_within_pass = inc
                prev_margin_global = g1
                prev_g2_global = g2

                # ----- Original recurrent update -----
                z = min(max(z + inc, 0.0), args.max_depth_clip_um)
                margin_mean_acc += margin
                active_margin_steps += 1
                margin_last = margin
                w_eff_last = w_eff_um
                total_defocus_last = total_defocus
                global_active_step += 1

            D = min(max(D + z, 0.0), args.max_depth_clip_um)

        margin_mean = margin_mean_acc / max(1, active_margin_steps)
        delta_phys = z if active > 0 else 0.0
        z_track = z if active > 0 else 0.0

        # Token features. They are transformed/normalized but still physically interpretable.
        k_norm = k / max(1, pass_count)
        token = [
            math.log1p(max(z_track, 0.0)),                        # local recursive removal capability
            total_defocus_last / max(zR_um, 1e-12),              # normalized defocus state
            math.log(max(w_eff_last / max(w0_um, 1e-12), 1e-12)),# log beam expansion
            margin_mean,                                         # mean log fluence-to-threshold margin
            margin_last,                                         # final log fluence-to-threshold margin
            math.log1p(max(delta_phys, 0.0)),                    # pass physical increment
            math.log1p(max(D, 0.0)),                             # cumulative physical depth
            k_norm,                                              # relative pass stage
            active,                                              # pass active mask
        ]
        tokens.append(token)
        z_tracks.append(z_track)
        deltas.append(delta_phys)
        D_values.append(D)
        margin_values.append(margin_mean)
        w_values.append(w_eff_last)
        defocus_values.append(total_defocus_last)

    z_arr = np.asarray(z_tracks, dtype=float)
    delta_arr = np.asarray(deltas, dtype=float)
    D_arr = np.asarray(D_values, dtype=float)
    active_arr = np.array([1.0 if k <= pass_count else 0.0 for k in range(1, max_pass + 1)])
    active_z = z_arr[active_arr > 0]
    active_delta = delta_arr[active_arr > 0]
    if len(active_delta) == 0:
        active_delta = np.array([0.0])

    if len(active_delta) >= 2:
        xs = np.arange(1, len(active_delta) + 1, dtype=float)
        slope = float(np.polyfit(xs, active_delta, deg=1)[0])
    else:
        slope = 0.0

    first_delta = float(active_delta[0])
    last_delta = float(active_delta[-1])
    ratio_d = float(last_delta / max(first_delta, 1e-9))
    z_recursive = float(D_arr[min(pass_count, max_pass) - 1]) if pass_count > 0 else 0.0

    event_flags = {e: int(event_pos[e] is not None) for e in ("E1", "E2", "E5")}
    ordered_events = sorted(
        [(e, float(event_pos[e])) for e in ("E1", "E2", "E5") if event_pos[e] is not None],
        key=lambda x: x[1],
    )
    event_order = ">".join(e for e, _ in ordered_events) if ordered_events else "NONE"
    event_combo = "".join(str(event_flags[e]) for e in ("E1", "E2", "E5"))
    mechanism_signature = f"{event_combo}|{event_order}"

    def _finite_or_zero(x: float) -> float:
        return float(x) if np.isfinite(x) else 0.0

    pooled: Dict[str, Any] = {
        "pulse_width_fs": tau,
        "repetition_rate_khz": f_khz,
        "scan_speed_mm_s": v_mm_s,
        "hatch_spacing_um": hatch,
        "inv_hatch_spacing_um": 1.0 / max(hatch, 1e-9),
        "pass_count": float(pass_count),
        "pulse_energy_uJ": pulse_energy_j * 1e6,
        "N_eff": float(N_eff),
        "log1p_N_eff": math.log1p(max(N_eff, 0.0)),
        "z_track_first_pass_um": float(z_arr[0]) if len(z_arr) > 0 else 0.0,
        "z_track_mean_active_um": float(np.mean(active_z)) if len(active_z) > 0 else 0.0,
        "z_track_last_active_um": float(active_z[-1]) if len(active_z) > 0 else 0.0,
        "pass_delta_first_um": first_delta,
        "pass_delta_last_um": last_delta,
        "pass_delta_mean_um": float(np.mean(active_delta)),
        "pass_delta_std_um": float(np.std(active_delta)),
        "pass_delta_ratio_last_first": ratio_d,
        "pass_delta_slope_um_per_pass": slope,
        "z_recursive_um": z_recursive,
        # Compact coverage / dose proxies. These are empirical observation-mapping
        # features derived from the physical recurrent output, not additional physics.
        "coverage_density_pass_per_um": float(pass_count) / max(hatch, 1e-9),
        "log1p_coverage_density": math.log1p(float(pass_count) / max(hatch, 1e-9)),
        "area_proxy_um": z_recursive * float(pass_count) / max(hatch, 1e-9),
        "log1p_area_proxy_um": math.log1p(max(z_recursive * float(pass_count) / max(hatch, 1e-9), 0.0)),
        "sqrt_area_proxy_um": math.sqrt(max(z_recursive * float(pass_count) / max(hatch, 1e-9), 0.0)),
        "z_recursive_per_hatch_um": z_recursive / max(hatch, 1e-9),
        "log1p_z_recursive_um": math.log1p(max(z_recursive, 0.0)),
        "sqrt_z_recursive_um": math.sqrt(max(z_recursive, 0.0)),
        "pulse_line_density_proxy": float(N_eff) / max(hatch, 1e-9),
        "log1p_pulse_line_density_proxy": math.log1p(max(float(N_eff) / max(hatch, 1e-9), 0.0)),
        "mean_margin_active": float(np.mean(np.asarray(margin_values)[active_arr > 0])) if pass_count > 0 else 0.0,
        "last_margin_active": float(np.asarray(margin_values)[active_arr > 0][-1]) if pass_count > 0 else 0.0,
        "max_defocus_over_zR": float(np.max(np.asarray(defocus_values) / max(zR_um, 1e-9))) if len(defocus_values) > 0 else 0.0,
        "last_beam_expansion": float(w_values[min(pass_count, max_pass) - 1] / max(w0_um, 1e-12)) if pass_count > 0 else 1.0,

        # Mechanism-transition diagnostics.
        "event_E1_ablation_off": event_flags["E1"],
        "event_E2_incubation_to_defocus": event_flags["E2"],
        "event_E5_growth_to_decay": event_flags["E5"],
        # 1.25 is an explicit "event absent" sentinel outside the normalized [0,1] interval;
        # the corresponding event flag remains the authoritative presence indicator.
        "event_E1_tnorm": float(event_pos["E1"]) if event_pos["E1"] is not None else 1.25,
        "event_E2_tnorm": float(event_pos["E2"]) if event_pos["E2"] is not None else 1.25,
        "event_E5_tnorm": float(event_pos["E5"]) if event_pos["E5"] is not None else 1.25,
        "event_E1_pass": float(event_pass["E1"]) if event_pass["E1"] is not None else np.nan,
        "event_E2_pass": float(event_pass["E2"]) if event_pass["E2"] is not None else np.nan,
        "event_E5_pass": float(event_pass["E5"]) if event_pass["E5"] is not None else np.nan,
        "event_E1_pulse_index": float(event_pulse_index["E1"]) if event_pulse_index["E1"] is not None else np.nan,
        "event_E2_pulse_index": float(event_pulse_index["E2"]) if event_pulse_index["E2"] is not None else np.nan,
        "event_E5_pulse_index": float(event_pulse_index["E5"]) if event_pulse_index["E5"] is not None else np.nan,
        "event_count": int(sum(event_flags.values())),
        "event_combo_E1E2E5": event_combo,
        "event_order_E1E2E5": event_order,
        "mechanism_signature": mechanism_signature,
        "g1_margin_min": _finite_or_zero(g1_min),
        "g1_margin_max": _finite_or_zero(g1_max),
        "g2_inc_minus_defocus_min": _finite_or_zero(g2_min),
        "g2_inc_minus_defocus_max": _finite_or_zero(g2_max),
        "g5_delta_increment_min_um": _finite_or_zero(g5_min),
        "g5_delta_increment_max_um": _finite_or_zero(g5_max),
    }
    return np.asarray(tokens, dtype=np.float32), pooled


def build_features(
    df: pd.DataFrame,
    params: PhysicsParams,
    optical: OpticalConfig,
    args: argparse.Namespace,
    max_pass: Optional[int] = None,
) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    if max_pass is None:
        max_pass = int(np.nanmax(df["pass_count"].astype(float).values))
    seqs = []
    pooled_rows = []
    masks = []
    for _, row in df.iterrows():
        seq, pooled = pure_physics_trace_for_row(row, params, optical, args, max_pass)
        seqs.append(seq)
        pooled_rows.append(pooled)
        masks.append(seq[:, -1] > 0.5)  # active feature
    seq_arr = np.stack(seqs, axis=0).astype(np.float32)
    pooled_df = pd.DataFrame(pooled_rows)
    pooled_df = add_engineered_raw_features(pooled_df)
    mask_arr = np.stack(masks, axis=0).astype(bool)
    return seq_arr, pooled_df, mask_arr


def add_engineered_raw_features(pooled_df: pd.DataFrame) -> pd.DataFrame:
    """Add raw-process engineered baseline features.

    These columns intentionally use only directly specified process variables
    (tau, repetition rate, scan speed, hatch spacing, and pass count). They do
    not use recurrent physics outputs such as z_recursive_um, N_eff, or
    area_proxy_um, so they can serve as a stronger but still non-physics
    engineering baseline.
    """
    out = pooled_df.copy()

    f_khz = pd.to_numeric(out.get("repetition_rate_khz"), errors="coerce").astype(float)
    v_mm_s = pd.to_numeric(out.get("scan_speed_mm_s"), errors="coerce").astype(float)
    h_um = pd.to_numeric(out.get("hatch_spacing_um"), errors="coerce").astype(float)
    pass_count = pd.to_numeric(out.get("pass_count"), errors="coerce").astype(float)

    f_safe = f_khz.clip(lower=0.0)
    v_safe = v_mm_s.clip(lower=1e-9)
    h_safe = h_um.clip(lower=1e-9)

    # Simple process-interaction terms that a domain engineer would normally try.
    # f_over_v is proportional to along-track pulse line density before optical scaling.
    # Np_over_h is a pass/hatch coverage-density proxy without using track width.
    out["f_over_v"] = f_khz / v_safe
    out["Np_over_h"] = pass_count / h_safe
    out["log1p_f"] = np.log1p(f_safe)
    out["log1p_v"] = np.log1p(v_safe)
    out["log1p_inv_v"] = np.log1p(1.0 / v_safe)

    # Keep NaNs as NaNs when the original raw inputs were missing; downstream imputers handle them.
    for c in ["f_over_v", "Np_over_h", "log1p_f", "log1p_v", "log1p_inv_v"]:
        out.loc[~np.isfinite(out[c].astype(float)), c] = np.nan
    return out


# -----------------------------
# Tabular models
# -----------------------------


SKLEARN_TABULAR_MODELS = {
    "gbdt", "gbdt_tuned", "gbdt_deeper",
    "extra_trees", "extra_trees_tuned", "extra_trees_conservative",
    "random_forest", "rf_tuned", "rf_conservative",
    "ridge", "gpr", "gaussian_process",
    "catboost", "xgboost",
}

EXTERNAL_MODEL_NAMES = {"catboost", "xgboost"}


def canonical_model_name(name: str) -> str:
    """Normalize user-facing aliases while keeping output labels compact."""
    return "gpr" if name == "gaussian_process" else name


def make_sklearn_model(name: str, seed: int, n_features: Optional[int] = None):
    name = canonical_model_name(name)
    if name == "gbdt":
        return GradientBoostingRegressor(
            n_estimators=120,
            learning_rate=0.045,
            max_depth=2,
            min_samples_leaf=3,
            random_state=seed,
            loss="squared_error",
        )
    if name == "gbdt_tuned":
        return GradientBoostingRegressor(
            n_estimators=90,
            learning_rate=0.04,
            max_depth=2,
            min_samples_leaf=4,
            subsample=0.9,
            random_state=seed,
            loss="squared_error",
        )
    if name == "gbdt_deeper":
        return GradientBoostingRegressor(
            n_estimators=80,
            learning_rate=0.035,
            max_depth=3,
            min_samples_leaf=4,
            subsample=0.85,
            random_state=seed,
            loss="squared_error",
        )
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "extra_trees_tuned":
        return ExtraTreesRegressor(
            n_estimators=800,
            max_depth=None,
            min_samples_leaf=2,
            max_features=0.85,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "extra_trees_conservative":
        return ExtraTreesRegressor(
            n_estimators=600,
            max_depth=4,
            min_samples_leaf=3,
            max_features=0.8,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "rf_tuned":
        return RandomForestRegressor(
            n_estimators=900,
            max_depth=None,
            min_samples_leaf=2,
            max_features=0.85,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "rf_conservative":
        return RandomForestRegressor(
            n_estimators=700,
            max_depth=4,
            min_samples_leaf=3,
            max_features=0.75,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "ridge":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ])
    if name == "gpr":
        # Small-sample baseline requested by the literature-alignment note.
        # Kernel optimization is done inside each training fold through this Pipeline,
        # so imputation/scaling/kernel fitting do not leak test-fold information.
        if n_features is None or int(n_features) <= 0:
            length_scale = 1.0
        else:
            length_scale = np.ones(int(n_features), dtype=float)
        kernel = (
            ConstantKernel(1.0, (1e-2, 1e3))
            * RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e3))
            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e2))
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                n_restarts_optimizer=10,
                random_state=seed,
            )),
        ])
    if name == "catboost":
        try:
            from catboost import CatBoostRegressor
            return CatBoostRegressor(
                depth=2,
                iterations=250,
                learning_rate=0.03,
                loss_function="RMSE",
                verbose=False,
                random_seed=seed,
                l2_leaf_reg=6.0,
            )
        except Exception as e:
            raise RuntimeError(f"CatBoost is not available: {e}")
    if name == "xgboost":
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(
                n_estimators=200,
                max_depth=2,
                learning_rate=0.035,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=5.0,
                random_state=seed,
                objective="reg:squarederror",
            )
        except Exception as e:
            raise RuntimeError(f"XGBoost is not available: {e}")
    raise ValueError(f"Unknown sklearn model: {name}")


def cv_tabular_model(
    X: pd.DataFrame,
    y: np.ndarray,
    row_ids: np.ndarray,
    model_name: str,
    args: argparse.Namespace,
    folds: Optional[List[Tuple[Any, ...]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float], Optional[Any]]:
    if folds is None:
        folds = make_cv_splits(X, y, args)
    oof_sum = np.zeros_like(y, dtype=float)
    oof_count = np.zeros_like(y, dtype=float)
    last_fold_ids = np.zeros_like(y, dtype=int)
    split_rows = []
    repeat_pred: Dict[int, np.ndarray] = {}
    repeat_count: Dict[int, np.ndarray] = {}
    skipped_reason = None
    for split_no, item in enumerate(folds, start=1):
        if len(item) == 2:
            repeat, fold, tr, te = 1, split_no, item[0], item[1]
        else:
            repeat, fold, tr, te = item
        repeat = int(repeat)
        fold = int(fold)
        try:
            model = make_sklearn_model(model_name, args.seed + 1000 * repeat + fold, n_features=X.shape[1])
        except Exception as e:
            skipped_reason = str(e)
            break
        if not isinstance(model, Pipeline) and canonical_model_name(model_name) not in EXTERNAL_MODEL_NAMES:
            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", model),
            ])
        target = np.log1p(y) if args.target_transform == "log1p" else y
        model.fit(X.iloc[tr], target[tr])
        pred_t = model.predict(X.iloc[te])
        pred = np.expm1(pred_t) if args.target_transform == "log1p" else pred_t
        pred = np.clip(pred, 0.0, args.max_depth_clip_um)
        oof_sum[te] += pred
        oof_count[te] += 1.0
        last_fold_ids[te] = fold
        if repeat not in repeat_pred:
            repeat_pred[repeat] = np.zeros_like(y, dtype=float)
            repeat_count[repeat] = np.zeros_like(y, dtype=float)
        repeat_pred[repeat][te] = pred
        repeat_count[repeat][te] += 1.0
        if args.save_repeated_cv_predictions:
            split_rows.append(pd.DataFrame({
                "model": model_name, "repeat": repeat, "fold": fold,
                "run_id": row_ids[te], "measured_depth_um": y[te],
                "cv_pred_depth_um": pred, "cv_residual_um": pred - y[te],
                "abs_cv_residual_um": np.abs(pred - y[te]),
            }))
    if skipped_reason:
        pred_df = pd.DataFrame()
        return pred_df, {"model": model_name, "skipped": 1, "skipped_reason": skipped_reason}, None
    oof_count = np.maximum(oof_count, 1.0)
    oof = oof_sum / oof_count
    pred_df = pd.DataFrame({
        "model": model_name,
        "repeat": 0,
        "fold": last_fold_ids,
        "run_id": row_ids,
        "measured_depth_um": y,
        "cv_pred_depth_um": oof,
        "cv_residual_um": oof - y,
        "abs_cv_residual_um": np.abs(oof - y),
        "oof_prediction_count": oof_count,
    })
    # Optionally include repeated split predictions in a separate block after averaged OOF rows.
    if args.save_repeated_cv_predictions and split_rows:
        pred_df = pd.concat([pred_df] + split_rows, axis=0, ignore_index=True)
    met = metrics_dict(y, oof, prefix="cv_")
    # Repeated-CV stability diagnostics: metrics are computed for each repeat before
    # averaging predictions across repeats. These columns are more honest for small datasets.
    repeat_metrics = []
    for rep, pred_rep in sorted(repeat_pred.items()):
        count_rep = repeat_count.get(rep, np.zeros_like(y, dtype=float))
        if np.all(count_rep > 0):
            repeat_metrics.append(metrics_dict(y, pred_rep, prefix=""))
    if repeat_metrics:
        for key in ["rmse", "mae", "r2"]:
            vals = np.asarray([m[key] for m in repeat_metrics if key in m and np.isfinite(m[key])], dtype=float)
            if vals.size:
                met[f"cv_{key}_repeat_mean"] = float(np.mean(vals))
                met[f"cv_{key}_repeat_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    met.update({"model": model_name, "n": len(y), "skipped": 0, "cv_repeats": int(args.cv_repeats)})
    # Fit final model on all measured rows for deployment/prediction-only diagnostics.
    final_model = None
    try:
        final_model = make_sklearn_model(model_name, args.seed, n_features=X.shape[1])
        if not isinstance(final_model, Pipeline) and canonical_model_name(model_name) not in EXTERNAL_MODEL_NAMES:
            final_model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", final_model),
            ])
        target = np.log1p(y) if args.target_transform == "log1p" else y
        final_model.fit(X, target)
    except Exception:
        final_model = None
    return pred_df, met, final_model



def run_leave_one_feature_ablation(
    pooled_m: pd.DataFrame,
    y: np.ndarray,
    row_ids: np.ndarray,
    args: argparse.Namespace,
    metrics_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Leave-one-feature-out ablation for compact feature sets.

    This is intentionally focused on low-dimensional physics proxy sets. It is
    more reliable than tree impurity importance when features are correlated
    (e.g., area_proxy_um = z_recursive_um * pass_count / hatch_spacing_um).
    """
    if not getattr(args, "run_feature_ablation", True):
        return pd.DataFrame()
    feature_sets_all = select_feature_sets(pooled_m)
    requested_fs = [x.strip() for x in str(args.ablation_feature_sets).split(",") if x.strip()]
    requested_models = [x.strip() for x in str(args.ablation_models).split(",") if x.strip()]
    pairs: List[Tuple[str, str]] = []

    # Always include the best available model/feature-set pair if requested.
    if "best" in requested_models and metrics_df is not None and not metrics_df.empty:
        available = metrics_df[metrics_df.get("skipped", 0) == 0]
        if not available.empty:
            best = available.iloc[0]
            base = str(best.get("base_model", "")).strip()
            fs = str(best.get("feature_set", "")).strip()
            if base and fs:
                pairs.append((base, fs))

    explicit_models = [m for m in requested_models if m != "best"]
    for fs in requested_fs:
        for m in explicit_models:
            pairs.append((m, fs))

    # De-duplicate while preserving order.
    seen = set()
    unique_pairs = []
    for m, fs in pairs:
        key = (m, fs)
        if key not in seen:
            seen.add(key)
            unique_pairs.append(key)

    rows = []
    for model_name, fs_name in unique_pairs:
        if fs_name not in feature_sets_all:
            rows.append({
                "base_model": model_name,
                "feature_set": fs_name,
                "removed_feature": "__SKIPPED__",
                "skipped": 1,
                "skipped_reason": f"unknown feature set: {fs_name}",
            })
            continue
        cols_full = filter_cols(pooled_m, feature_sets_all[fs_name])
        if len(cols_full) < 2:
            rows.append({
                "base_model": model_name,
                "feature_set": fs_name,
                "removed_feature": "__SKIPPED__",
                "skipped": 1,
                "skipped_reason": f"feature set has fewer than 2 columns after filtering: {fs_name}",
            })
            continue

        folds = make_cv_splits(pooled_m[cols_full], y, args)

        def eval_cols(cols: List[str]) -> Dict[str, Any]:
            _, met, _ = cv_tabular_model(pooled_m[cols], y, row_ids, model_name, args, folds=folds)
            return met

        try:
            full_met = eval_cols(cols_full)
            if int(full_met.get("skipped", 0)) == 1:
                rows.append({
                    "base_model": model_name,
                    "feature_set": fs_name,
                    "removed_feature": "__FULL__",
                    "n_features": len(cols_full),
                    "skipped": 1,
                    "skipped_reason": full_met.get("skipped_reason", ""),
                })
                continue
            full_rmse = float(full_met.get("cv_rmse", np.nan))
            rows.append({
                "base_model": model_name,
                "feature_set": fs_name,
                "removed_feature": "__FULL__",
                "n_features": len(cols_full),
                "features_used": ", ".join(cols_full),
                "cv_rmse": full_met.get("cv_rmse", np.nan),
                "cv_mae": full_met.get("cv_mae", np.nan),
                "cv_r2": full_met.get("cv_r2", np.nan),
                "cv_rmse_repeat_mean": full_met.get("cv_rmse_repeat_mean", np.nan),
                "cv_rmse_repeat_std": full_met.get("cv_rmse_repeat_std", np.nan),
                "delta_rmse_vs_full": 0.0,
                "skipped": 0,
            })
            for removed in cols_full:
                cols = [c for c in cols_full if c != removed]
                met = eval_cols(cols)
                if int(met.get("skipped", 0)) == 1:
                    rows.append({
                        "base_model": model_name,
                        "feature_set": fs_name,
                        "removed_feature": removed,
                        "n_features": len(cols),
                        "skipped": 1,
                        "skipped_reason": met.get("skipped_reason", ""),
                    })
                else:
                    rmse_val = float(met.get("cv_rmse", np.nan))
                    rows.append({
                        "base_model": model_name,
                        "feature_set": fs_name,
                        "removed_feature": removed,
                        "n_features": len(cols),
                        "features_used": ", ".join(cols),
                        "cv_rmse": met.get("cv_rmse", np.nan),
                        "cv_mae": met.get("cv_mae", np.nan),
                        "cv_r2": met.get("cv_r2", np.nan),
                        "cv_rmse_repeat_mean": met.get("cv_rmse_repeat_mean", np.nan),
                        "cv_rmse_repeat_std": met.get("cv_rmse_repeat_std", np.nan),
                        "delta_rmse_vs_full": rmse_val - full_rmse if np.isfinite(rmse_val) and np.isfinite(full_rmse) else np.nan,
                        "skipped": 0,
                    })
        except Exception as e:
            rows.append({
                "base_model": model_name,
                "feature_set": fs_name,
                "removed_feature": "__ERROR__",
                "skipped": 1,
                "skipped_reason": str(e),
            })

    ablation_df = pd.DataFrame(rows)
    if not ablation_df.empty:
        sort_cols = [c for c in ["base_model", "feature_set", "removed_feature"] if c in ablation_df.columns]
        if sort_cols:
            ablation_df = ablation_df.sort_values(sort_cols).reset_index(drop=True)
        ablation_df.to_csv(output_dir / "feature_ablation_leave_one_out.csv", index=False)
    return ablation_df



# -----------------------------
# Group ablation and residual diagnostics
# -----------------------------


def feature_groups_for_set(feature_set: str) -> Dict[str, List[str]]:
    """Predefined interpretable feature groups for ablation diagnostics."""
    if feature_set == "engineered_baseline":
        return {
            "raw_process_group": ["pulse_width_fs", "repetition_rate_khz", "scan_speed_mm_s", "hatch_spacing_um", "pass_count"],
            "along_track_density_group": ["f_over_v", "log1p_f", "log1p_v", "log1p_inv_v"],
            "hatch_coverage_group": ["Np_over_h", "inv_hatch_spacing_um"],
        }
    if feature_set == "raw_plus_physical_states":
        return {
            "raw_process_group": ["pulse_width_fs", "repetition_rate_khz", "scan_speed_mm_s", "hatch_spacing_um", "pass_count"],
            "recursive_depth_group": ["z_recursive_um", "z_track_first_pass_um", "z_track_mean_active_um", "z_track_last_active_um"],
            "pass_schedule_group": ["pass_count", "pass_delta_first_um", "pass_delta_last_um", "pass_delta_mean_um", "pass_delta_ratio_last_first", "pass_delta_slope_um_per_pass"],
            "pulse_density_group": ["pulse_energy_uJ", "N_eff", "log1p_N_eff"],
            "defocus_margin_group": ["mean_margin_active", "last_margin_active"],
            "hatch_coverage_group": ["hatch_spacing_um"],
        }
    if feature_set == "physics_only_recurrent_states":
        return {
            "recursive_depth_group": ["z_recursive_um", "z_track_first_pass_um", "z_track_mean_active_um", "z_track_last_active_um"],
            "pass_schedule_group": ["pass_delta_first_um", "pass_delta_last_um", "pass_delta_mean_um", "pass_delta_std_um", "pass_delta_ratio_last_first", "pass_delta_slope_um_per_pass"],
            "pulse_density_group": ["pulse_energy_uJ", "N_eff", "log1p_N_eff"],
            "defocus_margin_group": ["mean_margin_active", "last_margin_active", "max_defocus_over_zR", "last_beam_expansion"],
        }
    if feature_set.startswith("lowdim_area_proxy_pulse"):
        return {
            "area_proxy_group": ["area_proxy_um", "log1p_area_proxy_um", "sqrt_area_proxy_um"],
            "pulse_density_group": ["N_eff", "log1p_N_eff", "pulse_line_density_proxy", "log1p_pulse_line_density_proxy"],
            "hatch_coverage_group": ["hatch_spacing_um", "inv_hatch_spacing_um", "coverage_density_pass_per_um", "log1p_coverage_density"],
            "pass_schedule_group": ["pass_count", "coverage_density_pass_per_um", "log1p_coverage_density"],
            "recursive_depth_group": ["z_recursive_um", "log1p_z_recursive_um", "sqrt_z_recursive_um"],
        }
    if feature_set.startswith("mechanism"):
        return {
            "raw_process_group": ["pulse_width_fs", "repetition_rate_khz", "scan_speed_mm_s"],
            "recursive_depth_group": ["z_recursive_um", "log1p_z_recursive_um", "sqrt_z_recursive_um", "z_track_first_pass_um", "z_track_mean_active_um", "z_track_last_active_um"],
            "pass_decay_group": ["pass_delta_first_um", "pass_delta_last_um", "pass_delta_mean_um", "pass_delta_std_um", "pass_delta_ratio_last_first", "pass_delta_slope_um_per_pass"],
            "area_proxy_group": ["area_proxy_um", "log1p_area_proxy_um", "sqrt_area_proxy_um", "z_recursive_per_hatch_um"],
            "hatch_coverage_group": ["hatch_spacing_um", "inv_hatch_spacing_um", "coverage_density_pass_per_um", "log1p_coverage_density"],
            "pulse_density_group": ["N_eff", "log1p_N_eff", "pulse_line_density_proxy", "log1p_pulse_line_density_proxy", "pulse_energy_uJ"],
            "defocus_margin_group": ["mean_margin_active", "last_margin_active", "max_defocus_over_zR", "last_beam_expansion"],
        }
    return {
        "area_proxy_group": ["area_proxy_um", "log1p_area_proxy_um"],
        "pulse_density_group": ["N_eff", "log1p_N_eff", "pulse_line_density_proxy", "log1p_pulse_line_density_proxy"],
        "coverage_group": ["hatch_spacing_um", "inv_hatch_spacing_um", "pass_count", "coverage_density_pass_per_um"],
    }


def run_group_feature_ablation(
    pooled_m: pd.DataFrame,
    y: np.ndarray,
    row_ids: np.ndarray,
    args: argparse.Namespace,
    metrics_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Remove interpretable feature groups from selected model/feature-set pairs."""
    if not getattr(args, "run_group_ablation", True):
        return pd.DataFrame()
    feature_sets_all = select_feature_sets(pooled_m)
    requested_fs = [x.strip() for x in str(args.group_ablation_feature_sets).split(",") if x.strip()]
    requested_models = [x.strip() for x in str(args.group_ablation_models).split(",") if x.strip()]
    pairs: List[Tuple[str, str]] = []
    if "best" in requested_models and metrics_df is not None and not metrics_df.empty:
        available = metrics_df[metrics_df.get("skipped", 0) == 0]
        if not available.empty:
            best = available.iloc[0]
            base = str(best.get("base_model", "")).strip()
            fs = str(best.get("feature_set", "")).strip()
            if base and fs:
                pairs.append((base, fs))
    explicit_models = [m for m in requested_models if m != "best"]
    for fs in requested_fs:
        for m in explicit_models:
            pairs.append((m, fs))
    seen = set()
    unique_pairs = []
    for pair in pairs:
        if pair not in seen:
            seen.add(pair)
            unique_pairs.append(pair)

    rows = []
    for model_name, fs_name in unique_pairs:
        if fs_name not in feature_sets_all:
            rows.append({"base_model": model_name, "feature_set": fs_name, "removed_group": "__SKIPPED__", "skipped": 1, "skipped_reason": f"unknown feature set: {fs_name}"})
            continue
        cols_full = filter_cols(pooled_m, feature_sets_all[fs_name])
        if len(cols_full) < 2:
            rows.append({"base_model": model_name, "feature_set": fs_name, "removed_group": "__SKIPPED__", "skipped": 1, "skipped_reason": "too few columns"})
            continue
        folds = make_cv_splits(pooled_m[cols_full], y, args)
        def eval_cols(cols: List[str]) -> Dict[str, Any]:
            _, met, _ = cv_tabular_model(pooled_m[cols], y, row_ids, model_name, args, folds=folds)
            return met
        try:
            full_met = eval_cols(cols_full)
            if int(full_met.get("skipped", 0)) == 1:
                rows.append({"base_model": model_name, "feature_set": fs_name, "removed_group": "__FULL__", "n_features": len(cols_full), "skipped": 1, "skipped_reason": full_met.get("skipped_reason", "")})
                continue
            full_rmse = float(full_met.get("cv_rmse", np.nan))
            rows.append({
                "base_model": model_name, "feature_set": fs_name, "removed_group": "__FULL__", "removed_features": "", "n_features": len(cols_full), "features_used": ", ".join(cols_full),
                "cv_rmse": full_met.get("cv_rmse", np.nan), "cv_mae": full_met.get("cv_mae", np.nan), "cv_r2": full_met.get("cv_r2", np.nan),
                "cv_rmse_repeat_mean": full_met.get("cv_rmse_repeat_mean", np.nan), "cv_rmse_repeat_std": full_met.get("cv_rmse_repeat_std", np.nan),
                "delta_rmse_vs_full": 0.0, "skipped": 0,
            })
            groups = feature_groups_for_set(fs_name)
            for group_name, group_cols0 in groups.items():
                group_cols = [c for c in group_cols0 if c in cols_full]
                if not group_cols:
                    continue
                cols = [c for c in cols_full if c not in group_cols]
                if len(cols) < 1:
                    continue
                met = eval_cols(cols)
                if int(met.get("skipped", 0)) == 1:
                    rows.append({"base_model": model_name, "feature_set": fs_name, "removed_group": group_name, "removed_features": ", ".join(group_cols), "n_features": len(cols), "skipped": 1, "skipped_reason": met.get("skipped_reason", "")})
                else:
                    rmse_val = float(met.get("cv_rmse", np.nan))
                    rows.append({
                        "base_model": model_name, "feature_set": fs_name, "removed_group": group_name, "removed_features": ", ".join(group_cols), "n_features": len(cols), "features_used": ", ".join(cols),
                        "cv_rmse": met.get("cv_rmse", np.nan), "cv_mae": met.get("cv_mae", np.nan), "cv_r2": met.get("cv_r2", np.nan),
                        "cv_rmse_repeat_mean": met.get("cv_rmse_repeat_mean", np.nan), "cv_rmse_repeat_std": met.get("cv_rmse_repeat_std", np.nan),
                        "delta_rmse_vs_full": rmse_val - full_rmse if np.isfinite(rmse_val) and np.isfinite(full_rmse) else np.nan,
                        "skipped": 0,
                    })
        except Exception as e:
            rows.append({"base_model": model_name, "feature_set": fs_name, "removed_group": "__ERROR__", "skipped": 1, "skipped_reason": str(e)})
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(output_dir / "feature_ablation_groups.csv", index=False)
    return df


def residual_diagnostics_by_group(pred_all: pd.DataFrame, metrics_df: pd.DataFrame, measured_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Group residuals for the best model by process variables and depth bins."""
    if pred_all.empty or metrics_df.empty:
        return pd.DataFrame()
    available = metrics_df[metrics_df.get("skipped", 0) == 0]
    if available.empty:
        return pd.DataFrame()
    best_model = str(available.iloc[0]["model"])
    sub = pred_all[(pred_all["model"] == best_model) & (pred_all.get("repeat", 0) == 0)].copy()
    if sub.empty:
        sub = pred_all[pred_all["model"] == best_model].copy()
    cols_keep = ["run_id", "pulse_width_fs", "repetition_rate_khz", "scan_speed_mm_s", "hatch_spacing_um", "pass_count"]
    merged = sub.merge(measured_df[cols_keep], on="run_id", how="left")
    rows = []
    def add_group(var: str, label: Optional[str] = None) -> None:
        if var not in merged.columns:
            return
        for val, g in merged.groupby(var, dropna=False):
            if len(g) == 0:
                continue
            rows.append({
                "best_model": best_model,
                "group_variable": label or var,
                "group_value": val,
                "n": len(g),
                "measured_mean_um": float(g["measured_depth_um"].mean()),
                "pred_mean_um": float(g["cv_pred_depth_um"].mean()),
                "residual_mean_um": float(g["cv_residual_um"].mean()),
                "residual_mae_um": float(np.mean(np.abs(g["cv_residual_um"]))),
                "residual_rmse_um": float(np.sqrt(np.mean(np.square(g["cv_residual_um"])))),
                "residual_min_um": float(g["cv_residual_um"].min()),
                "residual_max_um": float(g["cv_residual_um"].max()),
            })
    for v in ["pulse_width_fs", "repetition_rate_khz", "scan_speed_mm_s", "hatch_spacing_um", "pass_count"]:
        add_group(v)
    # Depth bins by measured depth.
    try:
        bins = [-np.inf, 10, 30, 60, 90, np.inf]
        labels = ["<=10", "10-30", "30-60", "60-90", ">90"]
        merged["measured_depth_bin_um"] = pd.cut(merged["measured_depth_um"], bins=bins, labels=labels)
        add_group("measured_depth_bin_um")
    except Exception:
        pass
    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(output_dir / "residual_diagnostics_by_group.csv", index=False)
    return out

# -----------------------------
# Tiny neural models
# -----------------------------

if TORCH_AVAILABLE:
    class TinyMLP(nn.Module):
        def __init__(self, in_dim: int, hidden: int = 32, dropout: float = 0.2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    class TinyGRU(nn.Module):
        def __init__(self, seq_dim: int, glob_dim: int, hidden: int = 24, dropout: float = 0.2):
            super().__init__()
            self.gru = nn.GRU(seq_dim, hidden, num_layers=1, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(hidden + glob_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
        def forward(self, seq, glob, mask):
            # mask: True active. Use lengths for packed sequence if possible.
            lengths = mask.sum(dim=1).clamp(min=1).long().cpu()
            packed = nn.utils.rnn.pack_padded_sequence(seq, lengths, batch_first=True, enforce_sorted=False)
            _, h = self.gru(packed)
            h = h[-1]
            return self.head(torch.cat([h, glob], dim=1)).squeeze(-1)

    class TinyTransformer(nn.Module):
        def __init__(self, seq_dim: int, glob_dim: int, d_model: int = 16, nhead: int = 2, layers: int = 1, dropout: float = 0.2, max_pass: int = 10):
            super().__init__()
            self.seq_proj = nn.Linear(seq_dim, d_model)
            self.glob_proj = nn.Linear(glob_dim, d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=max(32, 2 * d_model),
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
            self.pos = nn.Parameter(torch.zeros(1, max_pass + 1, d_model))
            nn.init.normal_(self.pos, std=0.02)
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
        def forward(self, seq, glob, mask):
            # tokens: global token + pass tokens.
            g = self.glob_proj(glob).unsqueeze(1)
            s = self.seq_proj(seq)
            tokens = torch.cat([g, s], dim=1)
            tokens = tokens + self.pos[:, : tokens.shape[1], :]
            # True means ignored for transformer key_padding_mask.
            global_mask = torch.zeros((mask.shape[0], 1), dtype=torch.bool, device=mask.device)
            pad_mask = torch.cat([global_mask, ~mask.bool()], dim=1)
            out = self.encoder(tokens, src_key_padding_mask=pad_mask)
            return self.head(out[:, 0, :]).squeeze(-1)


def standardize_train_test(train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(train, axis=0)
    sd = np.nanstd(train, axis=0)
    sd[sd < 1e-8] = 1.0
    train2 = np.nan_to_num((train - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    test2 = np.nan_to_num((test - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    return train2, test2, mu, sd


def cv_neural_model(
    seq: np.ndarray,
    pooled: pd.DataFrame,
    mask: np.ndarray,
    y: np.ndarray,
    row_ids: np.ndarray,
    model_name: str,
    args: argparse.Namespace,
    folds: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float], Optional[Dict[str, Any]]]:
    if not TORCH_AVAILABLE:
        return pd.DataFrame(), {"model": model_name, "skipped": 1, "skipped_reason": "PyTorch not available"}, None
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if folds is None:
        kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        folds = list(kf.split(pooled, y))
    glob_np = pooled.values.astype(np.float32)
    oof = np.zeros_like(y, dtype=float)
    fold_ids = np.zeros_like(y, dtype=int)
    for split_no, item in enumerate(folds, start=1):
        if len(item) == 2:
            repeat, fold, tr, te = 1, split_no, item[0], item[1]
        else:
            repeat, fold, tr, te = item
        # Train/val split inside training fold for early stopping.
        tr_inner, val_inner = train_test_split(tr, test_size=min(0.25, max(1, int(0.2 * len(tr))) / max(1, len(tr))), random_state=args.seed + fold)
        # Standardize global and seq features using training inner+val? Use full train fold for scaler to avoid val shift.
        glob_tr, glob_te, glob_mu, glob_sd = standardize_train_test(glob_np[tr], glob_np[te])
        glob_all_scaled = np.nan_to_num((glob_np - glob_mu) / glob_sd, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        # For sequence features, standardize over active train tokens.
        seq_train_tokens = seq[tr].reshape(-1, seq.shape[-1])
        mask_train_tokens = mask[tr].reshape(-1)
        active_tokens = seq_train_tokens[mask_train_tokens]
        if len(active_tokens) == 0:
            active_tokens = seq_train_tokens
        seq_mu = np.nanmean(active_tokens, axis=0)
        seq_sd = np.nanstd(active_tokens, axis=0)
        seq_sd[seq_sd < 1e-8] = 1.0
        seq_scaled = np.nan_to_num((seq - seq_mu) / seq_sd, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        y_target = np.log1p(y).astype(np.float32) if args.target_transform == "log1p" else y.astype(np.float32)
        # Build model.
        if model_name == "tiny_mlp":
            model = TinyMLP(glob_np.shape[1], hidden=args.nn_hidden, dropout=args.nn_dropout).to(device)
        elif model_name == "gru":
            model = TinyGRU(seq.shape[-1], glob_np.shape[1], hidden=args.nn_hidden, dropout=args.nn_dropout).to(device)
        elif model_name == "tiny_transformer":
            model = TinyTransformer(seq.shape[-1], glob_np.shape[1], d_model=args.tf_d_model, nhead=args.tf_heads, layers=args.tf_layers, dropout=args.nn_dropout, max_pass=seq.shape[1]).to(device)
        else:
            raise ValueError(model_name)
        opt = torch.optim.AdamW(model.parameters(), lr=args.nn_lr, weight_decay=args.nn_weight_decay)
        best_loss = float("inf")
        best_state = None
        bad = 0

        def batch_tensors(idx: np.ndarray):
            seq_t = torch.tensor(seq_scaled[idx], dtype=torch.float32, device=device)
            glob_t = torch.tensor(glob_all_scaled[idx], dtype=torch.float32, device=device)
            mask_t = torch.tensor(mask[idx], dtype=torch.bool, device=device)
            y_t = torch.tensor(y_target[idx], dtype=torch.float32, device=device)
            return seq_t, glob_t, mask_t, y_t

        start = time.time()
        for epoch in range(1, args.nn_epochs + 1):
            model.train()
            seq_t, glob_t, mask_t, y_t = batch_tensors(tr_inner)
            opt.zero_grad(set_to_none=True)
            if model_name == "tiny_mlp":
                pred = model(glob_t)
            else:
                pred = model(seq_t, glob_t, mask_t)
            loss = F.smooth_l1_loss(pred, y_t)
            if not torch.isfinite(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            # validation
            model.eval()
            with torch.no_grad():
                seq_v, glob_v, mask_v, y_v = batch_tensors(val_inner)
                if model_name == "tiny_mlp":
                    val_pred = model(glob_v)
                else:
                    val_pred = model(seq_v, glob_v, mask_v)
                val_loss = float(F.smooth_l1_loss(val_pred, y_v).detach().cpu().item())
            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            if args.show_progress and epoch % args.nn_print_every == 0:
                print(f"[{model_name} fold {fold}/{len(folds)}] epoch {epoch}/{args.nn_epochs} val={val_loss:.5g} best={best_loss:.5g} elapsed={format_seconds(time.time()-start)}")
            if bad >= args.nn_patience:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            seq_te, glob_te_t, mask_te, _ = batch_tensors(te)
            if model_name == "tiny_mlp":
                pred_t = model(glob_te_t).detach().cpu().numpy()
            else:
                pred_t = model(seq_te, glob_te_t, mask_te).detach().cpu().numpy()
        pred = np.expm1(pred_t) if args.target_transform == "log1p" else pred_t
        pred = np.clip(pred, 0.0, args.max_depth_clip_um)
        oof[te] = pred
        fold_ids[te] = fold
    pred_df = pd.DataFrame({
        "model": model_name,
        "fold": fold_ids,
        "run_id": row_ids,
        "measured_depth_um": y,
        "cv_pred_depth_um": oof,
        "cv_residual_um": oof - y,
        "abs_cv_residual_um": np.abs(oof - y),
    })
    met = metrics_dict(y, oof, prefix="cv_")
    met.update({"model": model_name, "n": len(y), "skipped": 0})
    return pred_df, met, None


# -----------------------------
# Cross-validation orchestration
# -----------------------------


def select_feature_sets(pooled: pd.DataFrame) -> Dict[str, List[str]]:
    # Keep names stable. Missing columns are filtered later.
    return {
        "raw_process": [
            "pulse_width_fs", "repetition_rate_khz", "scan_speed_mm_s", "hatch_spacing_um", "pass_count",
        ],
        "engineered_baseline": [
            "pulse_width_fs", "repetition_rate_khz", "scan_speed_mm_s", "hatch_spacing_um", "pass_count",
            "f_over_v", "Np_over_h", "inv_hatch_spacing_um",
            "log1p_f", "log1p_v", "log1p_inv_v",
        ],
        "physics_only_recurrent_states": [
            "pulse_energy_uJ", "N_eff", "log1p_N_eff",
            "z_track_first_pass_um", "z_track_mean_active_um", "z_track_last_active_um",
            "pass_delta_first_um", "pass_delta_last_um", "pass_delta_mean_um", "pass_delta_std_um",
            "pass_delta_ratio_last_first", "pass_delta_slope_um_per_pass", "z_recursive_um",
            "mean_margin_active", "last_margin_active", "max_defocus_over_zR", "last_beam_expansion",
        ],
        "raw_plus_physical_states": [
            "pulse_width_fs", "repetition_rate_khz", "scan_speed_mm_s", "hatch_spacing_um", "pass_count",
            "pulse_energy_uJ", "N_eff", "log1p_N_eff",
            "z_track_first_pass_um", "z_track_mean_active_um", "z_track_last_active_um",
            "pass_delta_first_um", "pass_delta_last_um", "pass_delta_mean_um",
            "pass_delta_ratio_last_first", "pass_delta_slope_um_per_pass",
            "z_recursive_um", "mean_margin_active", "last_margin_active",
        ],
        "lowdim_minimal_3": ["z_recursive_um", "hatch_spacing_um", "pass_count"],
        "lowdim_area_proxy_4": ["z_recursive_um", "hatch_spacing_um", "pass_count", "area_proxy_um"],
        "lowdim_area_proxy_enhanced": [
            "z_recursive_um", "log1p_z_recursive_um", "sqrt_z_recursive_um",
            "hatch_spacing_um", "inv_hatch_spacing_um", "pass_count",
            "coverage_density_pass_per_um", "log1p_coverage_density",
            "area_proxy_um", "log1p_area_proxy_um", "sqrt_area_proxy_um", "z_recursive_per_hatch_um",
        ],
        "lowdim_area_proxy_pulse": [
            "z_recursive_um", "hatch_spacing_um", "inv_hatch_spacing_um", "pass_count",
            "area_proxy_um", "log1p_area_proxy_um", "coverage_density_pass_per_um",
            "N_eff", "log1p_N_eff", "pulse_line_density_proxy", "log1p_pulse_line_density_proxy",
        ],
        "lowdim_area_proxy_pulse_no_z": [
            "hatch_spacing_um", "inv_hatch_spacing_um", "pass_count",
            "area_proxy_um", "log1p_area_proxy_um", "coverage_density_pass_per_um",
            "N_eff", "log1p_N_eff", "pulse_line_density_proxy", "log1p_pulse_line_density_proxy",
        ],
        "lowdim_area_proxy_pulse_core5": [
            "area_proxy_um", "log1p_area_proxy_um", "pass_count",
            "inv_hatch_spacing_um", "log1p_pulse_line_density_proxy",
        ],
        "lowdim_area_proxy_pulse_core8": [
            "hatch_spacing_um", "inv_hatch_spacing_um", "pass_count",
            "area_proxy_um", "log1p_area_proxy_um", "coverage_density_pass_per_um",
            "log1p_N_eff", "log1p_pulse_line_density_proxy",
        ],
        "lowdim_area_proxy_pulse_no_area_group": [
            "z_recursive_um", "hatch_spacing_um", "inv_hatch_spacing_um", "pass_count",
            "coverage_density_pass_per_um", "N_eff", "log1p_N_eff",
            "pulse_line_density_proxy", "log1p_pulse_line_density_proxy",
        ],
        "lowdim_area_proxy_pulse_no_pulse_group": [
            "z_recursive_um", "hatch_spacing_um", "inv_hatch_spacing_um", "pass_count",
            "area_proxy_um", "log1p_area_proxy_um", "coverage_density_pass_per_um",
        ],
        "lowdim_area_proxy_pulse_no_hatch_group": [
            "z_recursive_um", "pass_count", "area_proxy_um", "log1p_area_proxy_um",
            "N_eff", "log1p_N_eff", "pulse_line_density_proxy", "log1p_pulse_line_density_proxy",
        ],
        "mechanism_events": [
            "event_E1_ablation_off", "event_E2_incubation_to_defocus", "event_E5_growth_to_decay",
            "event_E1_tnorm", "event_E2_tnorm", "event_E5_tnorm",
            "event_count",
            "g1_margin_min", "g1_margin_max",
            "g2_inc_minus_defocus_min", "g2_inc_minus_defocus_max",
            "g5_delta_increment_min_um", "g5_delta_increment_max_um",
        ],
        "mechanism_events_plus_core5": [
            "area_proxy_um", "log1p_area_proxy_um", "pass_count",
            "inv_hatch_spacing_um", "log1p_pulse_line_density_proxy",
            "event_E1_ablation_off", "event_E2_incubation_to_defocus", "event_E5_growth_to_decay",
            "event_E1_tnorm", "event_E2_tnorm", "event_E5_tnorm",
            "event_count",
        ],
        "mechanism_compact_12": [
            "z_recursive_um", "log1p_z_recursive_um", "hatch_spacing_um", "inv_hatch_spacing_um", "pass_count",
            "coverage_density_pass_per_um", "area_proxy_um", "log1p_area_proxy_um",
            "N_eff", "log1p_N_eff", "pass_delta_ratio_last_first", "pass_delta_slope_um_per_pass",
        ],
        "mechanism_pooled": [
            "pulse_width_fs", "repetition_rate_khz", "scan_speed_mm_s", "hatch_spacing_um", "inv_hatch_spacing_um", "pass_count",
            "pulse_energy_uJ", "N_eff", "log1p_N_eff",
            "z_track_first_pass_um", "z_track_mean_active_um", "z_track_last_active_um",
            "pass_delta_first_um", "pass_delta_last_um", "pass_delta_mean_um", "pass_delta_std_um",
            "pass_delta_ratio_last_first", "pass_delta_slope_um_per_pass", "z_recursive_um",
            "coverage_density_pass_per_um", "log1p_coverage_density", "area_proxy_um",
            "log1p_area_proxy_um", "sqrt_area_proxy_um", "z_recursive_per_hatch_um",
            "pulse_line_density_proxy", "log1p_pulse_line_density_proxy",
            "mean_margin_active", "last_margin_active", "max_defocus_over_zR", "last_beam_expansion",
        ],
        "mechanism_without_area_proxy": [
            "pulse_width_fs", "repetition_rate_khz", "scan_speed_mm_s", "hatch_spacing_um", "inv_hatch_spacing_um", "pass_count",
            "pulse_energy_uJ", "N_eff", "log1p_N_eff",
            "z_track_first_pass_um", "z_track_mean_active_um", "z_track_last_active_um",
            "pass_delta_first_um", "pass_delta_last_um", "pass_delta_mean_um", "pass_delta_std_um",
            "pass_delta_ratio_last_first", "pass_delta_slope_um_per_pass", "z_recursive_um",
            "coverage_density_pass_per_um", "log1p_coverage_density",
            "log1p_area_proxy_um", "sqrt_area_proxy_um", "z_recursive_per_hatch_um",
            "pulse_line_density_proxy", "log1p_pulse_line_density_proxy",
            "mean_margin_active", "last_margin_active", "max_defocus_over_zR", "last_beam_expansion",
        ],
    }


def resolve_requested_feature_sets(feature_sets: Dict[str, List[str]], args: argparse.Namespace) -> Dict[str, List[str]]:
    """Apply the main experiment feature-set filter.

    The default follows the manuscript-alignment note: raw process, engineered
    raw baseline, physics-only states, raw + physical states, core5, and core8. Use ``--feature-sets all``
    to recover the broader exploratory matrix.
    """
    raw = str(getattr(args, "feature_sets", "all")).strip()
    if not raw or raw.lower() == "all":
        return feature_sets
    requested = [x.strip() for x in raw.split(",") if x.strip()]
    selected = {name: feature_sets[name] for name in requested if name in feature_sets}
    missing = [name for name in requested if name not in feature_sets]
    if missing:
        print(f"[feature sets] ignored unknown feature sets: {', '.join(missing)}")
    if not selected:
        print("[feature sets] no requested feature sets were found; falling back to all feature sets.")
        return feature_sets
    return selected


def filter_cols(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def make_cv_splits(X: pd.DataFrame, y: np.ndarray, args: argparse.Namespace) -> List[Tuple[int, int, np.ndarray, np.ndarray]]:
    """Return repeated K-fold splits as (repeat, fold, train_idx, test_idx)."""
    repeats = max(1, int(getattr(args, "cv_repeats", 1)))
    splits: List[Tuple[int, int, np.ndarray, np.ndarray]] = []
    if repeats == 1:
        kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        for fold, (tr, te) in enumerate(kf.split(X, y), start=1):
            splits.append((1, fold, tr, te))
    else:
        rkf = RepeatedKFold(n_splits=args.cv_folds, n_repeats=repeats, random_state=args.seed)
        for split_no, (tr, te) in enumerate(rkf.split(X, y), start=1):
            repeat = (split_no - 1) // args.cv_folds + 1
            fold = (split_no - 1) % args.cv_folds + 1
            splits.append((repeat, fold, tr, te))
    return splits


def run_fixed_or_full_cv(
    all_df: pd.DataFrame,
    measured_df: pd.DataFrame,
    params: PhysicsParams,
    optical: OpticalConfig,
    args: argparse.Namespace,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    max_pass = int(np.nanmax(all_df["pass_count"].astype(float).values))
    seq_all, pooled_all, mask_all = build_features(all_df, params, optical, args, max_pass=max_pass)
    all_ids = all_df["run_id"].values
    measured_idx = np.where(all_df["has_measured_depth"].values == 1)[0]
    y = all_df.loc[measured_idx, "measured_depth_um"].astype(float).values
    row_ids = all_df.loc[measured_idx, "run_id"].values
    seq_m = seq_all[measured_idx]
    pooled_m = pooled_all.iloc[measured_idx].reset_index(drop=True)
    mask_m = mask_all[measured_idx]
    # Save generated features.
    np.savez_compressed(output_dir / "physics_sequence_features_all_rows.npz", seq=seq_all, mask=mask_all, run_id=all_ids)
    pooled_export = pd.concat([all_df[["run_id", "has_measured_depth", "is_problematic_prediction_only"]].reset_index(drop=True), pooled_all.reset_index(drop=True)], axis=1)
    pooled_export.to_csv(output_dir / "pooled_physics_features_all_rows.csv", index=False)

    folds = make_cv_splits(pooled_m, y, args)
    # Neural sequence models are optional and expensive. When repeated CV is enabled,
    # keep neural models on the first repeat only unless the user explicitly requests all repeats.
    neural_folds = folds
    if int(args.cv_repeats) > 1 and not args.neural_use_repeated_cv:
        neural_folds = [sp for sp in folds if sp[0] == 1]
    feature_sets = resolve_requested_feature_sets(select_feature_sets(pooled_m), args)
    all_pred_rows = []
    metric_rows = []
    final_models: Dict[str, Tuple[Any, List[str]]] = {}

    sklearn_models = [m for m in args.models_list if canonical_model_name(m) in SKLEARN_TABULAR_MODELS]
    neural_models = [m for m in args.models_list if m in {"tiny_mlp", "gru", "tiny_transformer"}]

    # Low-dimensional and pooled tabular models.
    for fs_name, cols0 in feature_sets.items():
        cols = filter_cols(pooled_m, cols0)
        if len(cols) == 0:
            continue
        X = pooled_m[cols]
        for model_name in sklearn_models:
            label = f"{model_name}__{fs_name}"
            pred_df, met, final_model = cv_tabular_model(X, y, row_ids, model_name, args, folds=folds)
            met["model"] = label
            met["base_model"] = model_name
            met["feature_set"] = fs_name
            met["n_features"] = len(cols)
            met["features"] = ", ".join(cols)
            metric_rows.append(met)
            if not pred_df.empty:
                pred_df["model"] = label
                pred_df["feature_set"] = fs_name
                all_pred_rows.append(pred_df)
            if final_model is not None:
                final_models[label] = (final_model, cols)

    # Neural models: tiny_mlp uses pooled mechanism features; GRU/Transformer use seq + global pooled features.
    neural_pooled_cols = filter_cols(pooled_m, select_feature_sets(pooled_m)["mechanism_pooled"])
    if neural_pooled_cols and neural_models:
        pooled_nn = pooled_m[neural_pooled_cols].copy()
        for model_name in neural_models:
            pred_df, met, _ = cv_neural_model(seq_m, pooled_nn, mask_m, y, row_ids, model_name, args, folds=neural_folds)
            met["model"] = f"{model_name}__mechanism_sequence"
            met["base_model"] = model_name
            met["feature_set"] = "mechanism_sequence" if model_name in {"gru", "tiny_transformer"} else "mechanism_pooled"
            met["n_features"] = len(neural_pooled_cols)
            met["features"] = ", ".join(neural_pooled_cols)
            metric_rows.append(met)
            if not pred_df.empty:
                pred_df["model"] = met["model"]
                pred_df["feature_set"] = met["feature_set"]
                all_pred_rows.append(pred_df)

    metrics_df = pd.DataFrame(metric_rows)
    if not metrics_df.empty and "cv_rmse" in metrics_df.columns:
        metrics_df = metrics_df.sort_values(["skipped", "cv_rmse"], ascending=[True, True]).reset_index(drop=True)
    pred_all = pd.concat(all_pred_rows, axis=0, ignore_index=True) if all_pred_rows else pd.DataFrame()

    # Prediction-only rows with best final tabular model if available.
    best_model_name = None
    final_pred_all = all_df[["run_id", "measured_depth_um", "has_measured_depth", "is_problematic_prediction_only", "problematic_note"]].copy()
    if not metrics_df.empty:
        available = metrics_df[metrics_df.get("skipped", 0) == 0]
        if not available.empty:
            best_model_name = str(available.iloc[0]["model"])
    if best_model_name in final_models:
        final_model, cols = final_models[best_model_name]
        pred_t = final_model.predict(pooled_all[cols])
        pred = np.expm1(pred_t) if args.target_transform == "log1p" else pred_t
        pred = np.clip(pred, 0.0, args.max_depth_clip_um)
        final_pred_all["best_model"] = best_model_name
        final_pred_all["final_pred_depth_um"] = pred
        final_pred_all["full_residual_um"] = final_pred_all["final_pred_depth_um"] - final_pred_all["measured_depth_um"]
        try:
            core_model = final_model.named_steps.get("model", final_model) if isinstance(final_model, Pipeline) else final_model
            imp = getattr(core_model, "feature_importances_", None)
            if imp is not None and len(imp) == len(cols):
                fi = pd.DataFrame({"feature": cols, "importance": np.asarray(imp, dtype=float)})
                fi = fi.sort_values("importance", ascending=False)
                fi.to_csv(output_dir / "best_model_feature_importance.csv", index=False)
        except Exception:
            pass
    context = {
        "seq_all": seq_all,
        "pooled_all": pooled_all,
        "mask_all": mask_all,
        "best_model_name": best_model_name,
        "params": asdict(params),
        "max_pass": max_pass,
    }
    return metrics_df, pred_all, {"final_pred_all": final_pred_all, **context}


def run_fold_physics_cv(
    all_df: pd.DataFrame,
    measured_df: pd.DataFrame,
    optical: OpticalConfig,
    args: argparse.Namespace,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Strict repeated CV with physics fitting inside every train split.

    This implements true ``cv_folds × cv_repeats`` strict fold-physics CV:
    for each repeated split, physics parameters are fitted only on that split's
    training rows, features are rebuilt with the split-specific physics, and the
    observation mapper is then fitted only on the same training rows. This avoids
    information leakage from both the physics-identification and ML-mapping steps.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("--physics-fit-scope fold requires PyTorch for fold-wise physics fitting.")

    measured_df = measured_df.reset_index(drop=True)
    y = measured_df["measured_depth_um"].astype(float).values
    row_ids = measured_df["run_id"].values
    folds = make_cv_splits(measured_df, y, args)
    total_splits = len(folds)

    metric_rows: List[Dict[str, Any]] = []
    all_pred_rows: List[pd.DataFrame] = []
    split_pred_rows: List[pd.DataFrame] = []
    feature_sets = None
    model_names = [m for m in args.models_list if canonical_model_name(m) in SKLEARN_TABULAR_MODELS]
    max_pass = int(np.nanmax(all_df["pass_count"].astype(float).values))
    init = args_to_physics_params(args)

    # Accumulate OOF predictions per (model, feature set), averaged across repeats.
    oof_sum: Dict[str, np.ndarray] = {}
    oof_count: Dict[str, np.ndarray] = {}
    fold_store: Dict[str, np.ndarray] = {}
    repeat_pred: Dict[str, Dict[int, np.ndarray]] = {}
    repeat_count: Dict[str, Dict[int, np.ndarray]] = {}
    skipped: Dict[str, str] = {}

    for split_no, (repeat, fold, tr, te) in enumerate(folds, start=1):
        repeat = int(repeat)
        fold = int(fold)
        print(
            f"[strict fold-physics CV] split {split_no}/{total_splits} "
            f"(repeat {repeat}/{args.cv_repeats}, fold {fold}/{args.cv_folds}): "
            f"fit physics on {len(tr)} measured rows"
        )
        params_fold = fit_physics_params(
            measured_df.iloc[tr], args, optical, init, args.device,
            label=f"fold-physics r{repeat} f{fold}",
        )
        _, pooled_train, _ = build_features(measured_df.iloc[tr], params_fold, optical, args, max_pass=max_pass)
        _, pooled_test, _ = build_features(measured_df.iloc[te], params_fold, optical, args, max_pass=max_pass)
        if feature_sets is None:
            feature_sets = resolve_requested_feature_sets(select_feature_sets(pooled_train), args)

        # Tabular models only for strict fold-physics by default.
        for fs_name, cols0 in feature_sets.items():
            cols = filter_cols(pooled_train, cols0)
            if not cols:
                continue
            for base_raw in model_names:
                base = canonical_model_name(base_raw)
                label = f"{base}__{fs_name}"
                if label in skipped:
                    continue
                if label not in oof_sum:
                    oof_sum[label] = np.zeros_like(y, dtype=float)
                    oof_count[label] = np.zeros_like(y, dtype=float)
                    fold_store[label] = np.zeros_like(y, dtype=int)
                    repeat_pred[label] = {}
                    repeat_count[label] = {}
                try:
                    model = make_sklearn_model(base, args.seed + 1000 * repeat + fold, n_features=len(cols))
                    if not isinstance(model, Pipeline) and canonical_model_name(base) not in EXTERNAL_MODEL_NAMES:
                        model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
                    target = np.log1p(y[tr]) if args.target_transform == "log1p" else y[tr]
                    model.fit(pooled_train[cols], target)
                    pred_t = model.predict(pooled_test[cols])
                    pred = np.expm1(pred_t) if args.target_transform == "log1p" else pred_t
                    pred = np.clip(pred, 0.0, args.max_depth_clip_um)

                    oof_sum[label][te] += pred
                    oof_count[label][te] += 1.0
                    fold_store[label][te] = fold
                    if repeat not in repeat_pred[label]:
                        repeat_pred[label][repeat] = np.zeros_like(y, dtype=float)
                        repeat_count[label][repeat] = np.zeros_like(y, dtype=float)
                    repeat_pred[label][repeat][te] = pred
                    repeat_count[label][repeat][te] += 1.0

                    if args.save_repeated_cv_predictions:
                        split_pred_rows.append(pd.DataFrame({
                            "model": label,
                            "base_model": base,
                            "feature_set": fs_name,
                            "repeat": repeat,
                            "fold": fold,
                            "run_id": row_ids[te],
                            "measured_depth_um": y[te],
                            "cv_pred_depth_um": pred,
                            "cv_residual_um": pred - y[te],
                            "abs_cv_residual_um": np.abs(pred - y[te]),
                        }))
                except Exception as e:
                    skipped[label] = str(e)

        if args.include_neural_in_fold_physics:
            # Strict neural fold-physics would require direct train/test neural training.
            # Keep it disabled here so the strict evidence remains compact and reproducible.
            pass

    for label, sum_pred in oof_sum.items():
        count = oof_count[label]
        base, fs_name = label.split("__", 1)
        if np.any(count <= 0):
            skipped[label] = f"incomplete strict repeated CV predictions: min_count={float(np.min(count)):.0f}"
            continue
        oof = sum_pred / count
        pred_df = pd.DataFrame({
            "model": label,
            "base_model": base,
            "feature_set": fs_name,
            "repeat": 0,
            "fold": fold_store[label],
            "run_id": row_ids,
            "measured_depth_um": y,
            "cv_pred_depth_um": oof,
            "cv_residual_um": oof - y,
            "abs_cv_residual_um": np.abs(oof - y),
            "oof_prediction_count": count,
        })
        all_pred_rows.append(pred_df)

        met = metrics_dict(y, oof, prefix="cv_")
        repeat_metrics = []
        for rep, pred_rep in sorted(repeat_pred.get(label, {}).items()):
            cnt_rep = repeat_count.get(label, {}).get(rep, np.zeros_like(y, dtype=float))
            if np.all(cnt_rep > 0):
                repeat_metrics.append(metrics_dict(y, pred_rep, prefix=""))
        if repeat_metrics:
            for key in ["rmse", "mae", "r2"]:
                vals = np.asarray([m[key] for m in repeat_metrics if key in m and np.isfinite(m[key])], dtype=float)
                if vals.size:
                    met[f"cv_{key}_repeat_mean"] = float(np.mean(vals))
                    met[f"cv_{key}_repeat_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        met.update({
            "model": label,
            "base_model": base,
            "feature_set": fs_name,
            "n": len(y),
            "skipped": 0,
            "cv_folds": int(args.cv_folds),
            "cv_repeats": int(args.cv_repeats),
            "strict_physics_refit_per_split": 1,
        })
        metric_rows.append(met)

    for label, reason in skipped.items():
        base, fs_name = label.split("__", 1)
        metric_rows.append({
            "model": label,
            "base_model": base,
            "feature_set": fs_name,
            "skipped": 1,
            "skipped_reason": reason,
            "cv_folds": int(args.cv_folds),
            "cv_repeats": int(args.cv_repeats),
            "strict_physics_refit_per_split": 1,
        })

    metrics_df = pd.DataFrame(metric_rows)
    if not metrics_df.empty and "cv_rmse" in metrics_df.columns:
        metrics_df = metrics_df.sort_values(["skipped", "cv_rmse"], ascending=[True, True]).reset_index(drop=True)
    pred_blocks = all_pred_rows + (split_pred_rows if args.save_repeated_cv_predictions else [])
    pred_all = pd.concat(pred_blocks, axis=0, ignore_index=True) if pred_blocks else pd.DataFrame()

    # Final all-row features/predictions using all measured physics fit for deployment/prediction-only diagnostics.
    print("[strict fold-physics CV] final physics fit on all measured rows for prediction-only diagnostics")
    final_params = fit_physics_params(measured_df, args, optical, init, args.device, label="final physics fit")
    seq_all, pooled_all, mask_all = build_features(all_df, final_params, optical, args, max_pass=max_pass)
    pooled_export = pd.concat(
        [all_df[["run_id", "has_measured_depth", "is_problematic_prediction_only"]].reset_index(drop=True), pooled_all.reset_index(drop=True)],
        axis=1,
    )
    pooled_export.to_csv(output_dir / "pooled_physics_features_all_rows.csv", index=False)
    np.savez_compressed(output_dir / "physics_sequence_features_all_rows.npz", seq=seq_all, mask=mask_all, run_id=all_df["run_id"].values)

    final_pred_all = all_df[["run_id", "measured_depth_um", "has_measured_depth", "is_problematic_prediction_only", "problematic_note"]].copy()
    best_model_name = None
    if not metrics_df.empty:
        available = metrics_df[metrics_df.get("skipped", 0) == 0]
        if not available.empty:
            best_model_name = str(available.iloc[0]["model"])
            base, fs_name = best_model_name.split("__", 1)
            cols = filter_cols(pooled_all, select_feature_sets(pooled_all)[fs_name])
            try:
                final_model = make_sklearn_model(base, args.seed, n_features=len(cols))
                if not isinstance(final_model, Pipeline) and canonical_model_name(base) not in EXTERNAL_MODEL_NAMES:
                    final_model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", final_model)])
                measured_idx = np.where(all_df["has_measured_depth"].values == 1)[0]
                target_y = all_df.iloc[measured_idx]["measured_depth_um"].astype(float).values
                target = np.log1p(target_y) if args.target_transform == "log1p" else target_y
                final_model.fit(pooled_all.iloc[measured_idx][cols], target)
                pred_t = final_model.predict(pooled_all[cols])
                pred = np.expm1(pred_t) if args.target_transform == "log1p" else pred_t
                final_pred_all["best_model"] = best_model_name
                final_pred_all["final_pred_depth_um"] = np.clip(pred, 0.0, args.max_depth_clip_um)
                final_pred_all["full_residual_um"] = final_pred_all["final_pred_depth_um"] - final_pred_all["measured_depth_um"]
                try:
                    core_model = final_model.named_steps.get("model", final_model) if isinstance(final_model, Pipeline) else final_model
                    imp = getattr(core_model, "feature_importances_", None)
                    if imp is not None and len(imp) == len(cols):
                        fi = pd.DataFrame({"feature": cols, "importance": np.asarray(imp, dtype=float)})
                        fi = fi.sort_values("importance", ascending=False)
                        fi.to_csv(output_dir / "best_model_feature_importance.csv", index=False)
                except Exception:
                    pass
            except Exception as e:
                print(f"[final prediction] failed for {best_model_name}: {e}")

    context = {
        "final_pred_all": final_pred_all,
        "seq_all": seq_all,
        "pooled_all": pooled_all,
        "mask_all": mask_all,
        "best_model_name": best_model_name,
        "params": asdict(final_params),
        "max_pass": max_pass,
    }
    return metrics_df, pred_all, context



# -----------------------------
# Mechanism-transition virtual process generation
# -----------------------------


MECHANISM_EVENT_FLAG_COLS = [
    "event_E1_ablation_off",
    "event_E2_incubation_to_defocus",
    "event_E5_growth_to_decay",
]
MECHANISM_EVENT_TIME_COLS = [
    "event_E1_tnorm",
    "event_E2_tnorm",
    "event_E5_tnorm",
]


def _parse_numeric_list(text_value: str) -> List[float]:
    values: List[float] = []
    for token in str(text_value or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except Exception:
            raise ValueError(f"Invalid numeric list item: {token!r}")
    return values


def _observed_or_requested_values(
    df: pd.DataFrame,
    column: str,
    requested: str,
) -> np.ndarray:
    vals = _parse_numeric_list(requested)
    if vals:
        return np.asarray(sorted(set(vals)), dtype=float)
    observed = pd.to_numeric(df[column], errors="coerce").dropna().astype(float).values
    if observed.size == 0:
        raise ValueError(f"No valid values available for virtual candidate column {column!r}")
    return np.asarray(sorted(set(observed.tolist())), dtype=float)


def _lhs_unit(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Simple Latin-hypercube points in [0, 1]^d without scipy dependency."""
    n = max(1, int(n))
    d = max(1, int(d))
    out = np.empty((n, d), dtype=float)
    for j in range(d):
        perm = rng.permutation(n)
        out[:, j] = (perm + rng.random(n)) / n
    return out


def _bounded_range(
    df: pd.DataFrame,
    column: str,
    lo_arg: Optional[float],
    hi_arg: Optional[float],
) -> Tuple[float, float]:
    vals = pd.to_numeric(df[column], errors="coerce").dropna().astype(float)
    if vals.empty:
        raise ValueError(f"No valid values available for virtual candidate column {column!r}")
    lo = float(vals.min()) if lo_arg is None else float(lo_arg)
    hi = float(vals.max()) if hi_arg is None else float(hi_arg)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi < lo:
        raise ValueError(f"Invalid virtual range for {column}: [{lo}, {hi}]")
    return lo, hi


def generate_executable_virtual_candidate_pool(
    reference_df: pd.DataFrame,
    args: argparse.Namespace,
    candidate_count: Optional[int] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate process candidates inside explicitly executable/allowed ranges.

    Pulse width and repetition rate default to observed discrete settings. Scan
    speed and hatch spacing default to continuous observed ranges. Pass count is
    integer-valued. User-supplied CLI bounds/lists override those defaults.
    """
    rng_seed = int(args.seed) + 1701 if seed is None else int(seed)
    rng = np.random.default_rng(rng_seed)
    n = max(1, int(args.virtual_candidate_count if candidate_count is None else candidate_count))

    tau_values = _observed_or_requested_values(
        reference_df, "pulse_width_fs", args.virtual_pulse_width_values
    )
    f_values = _observed_or_requested_values(
        reference_df, "repetition_rate_khz", args.virtual_repetition_rate_values
    )
    v_lo, v_hi = _bounded_range(
        reference_df, "scan_speed_mm_s",
        args.virtual_scan_speed_min, args.virtual_scan_speed_max,
    )
    h_lo, h_hi = _bounded_range(
        reference_df, "hatch_spacing_um",
        args.virtual_hatch_min, args.virtual_hatch_max,
    )

    observed_pass = pd.to_numeric(reference_df["pass_count"], errors="coerce").dropna().astype(int)
    if observed_pass.empty:
        raise ValueError("No valid pass_count values for virtual generation.")
    p_lo = int(observed_pass.min()) if args.virtual_pass_min is None else int(args.virtual_pass_min)
    p_hi = int(observed_pass.max()) if args.virtual_pass_max is None else int(args.virtual_pass_max)
    if p_hi < p_lo or p_lo < 1:
        raise ValueError(f"Invalid virtual pass range: [{p_lo}, {p_hi}]")

    lhs = _lhs_unit(n, 2, rng)
    scan_speed = v_lo + lhs[:, 0] * (v_hi - v_lo)
    hatch = h_lo + lhs[:, 1] * (h_hi - h_lo)
    tau = rng.choice(tau_values, size=n, replace=True)
    rep = rng.choice(f_values, size=n, replace=True)
    passes = rng.integers(p_lo, p_hi + 1, size=n)

    cand = pd.DataFrame({
        "run_id": [f"VIRTUAL_CAND_{i+1:06d}" for i in range(n)],
        "pulse_width_fs": tau,
        "repetition_rate_khz": rep,
        "scan_speed_mm_s": scan_speed,
        "hatch_spacing_um": hatch,
        "pass_count": passes.astype(int),
        "measured_depth_um": np.nan,
        "has_measured_depth": 0,
        "is_problematic_prediction_only": 0,
        "problematic_note": "mechanism-transition virtual candidate",
    })

    # Remove exact/near-exact duplicate process recipes, including measured recipes.
    key_cols = [
        "pulse_width_fs", "repetition_rate_khz", "scan_speed_mm_s",
        "hatch_spacing_um", "pass_count",
    ]
    cand["_recipe_key"] = (
        cand["pulse_width_fs"].round(6).astype(str) + "|" +
        cand["repetition_rate_khz"].round(6).astype(str) + "|" +
        cand["scan_speed_mm_s"].round(6).astype(str) + "|" +
        cand["hatch_spacing_um"].round(6).astype(str) + "|" +
        cand["pass_count"].astype(int).astype(str)
    )
    ref = reference_df.copy()
    ref_keys = set(
        ref["pulse_width_fs"].round(6).astype(str) + "|" +
        ref["repetition_rate_khz"].round(6).astype(str) + "|" +
        ref["scan_speed_mm_s"].round(6).astype(str) + "|" +
        ref["hatch_spacing_um"].round(6).astype(str) + "|" +
        ref["pass_count"].astype(int).astype(str)
    )
    cand = cand[~cand["_recipe_key"].isin(ref_keys)].drop_duplicates("_recipe_key").drop(columns="_recipe_key")
    return cand.reset_index(drop=True)


def _signature_timing_vector(row: pd.Series) -> np.ndarray:
    # Absent events use 1.25: outside the [0,1] process-time range so absence
    # remains distinct without needing a second distance calculation.
    vals = []
    for flag_col, time_col in zip(MECHANISM_EVENT_FLAG_COLS, MECHANISM_EVENT_TIME_COLS):
        flag = int(safe_float(row.get(flag_col, 0), 0.0) > 0.5)
        t = safe_float(row.get(time_col, np.nan))
        vals.append(float(t) if flag and np.isfinite(t) else 1.25)
    return np.asarray(vals, dtype=float)


def _timing_novelty(row: pd.Series, reference_rows: pd.DataFrame) -> float:
    if reference_rows is None or reference_rows.empty:
        return 1.0
    v = _signature_timing_vector(row)
    refs = np.vstack([_signature_timing_vector(r) for _, r in reference_rows.iterrows()])
    d = np.sqrt(np.mean((refs - v[None, :]) ** 2, axis=1))
    return float(np.min(d)) if d.size else 1.0


def _event_combo(row: pd.Series) -> str:
    return "".join(str(int(safe_float(row.get(c, 0), 0.0) > 0.5)) for c in MECHANISM_EVENT_FLAG_COLS)


def score_and_select_virtual_candidates(
    candidate_df: pd.DataFrame,
    candidate_pooled: pd.DataFrame,
    measured_pooled: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Greedy mechanism-transition coverage selection.

    Selection prioritizes previously unseen E1/E2/E5 signatures, then unseen
    event combinations, then event-timing novelty. This is intentionally a
    transparent first implementation; the score is not claimed to be a
    calibrated information-theoretic acquisition function.
    """
    # Candidate process columns also exist in pooled features; append only genuinely
    # new pooled columns so every DataFrame column name remains unique.
    pooled_extra_cols = [c for c in candidate_pooled.columns if c not in candidate_df.columns]
    scored = pd.concat(
        [
            candidate_df.reset_index(drop=True),
            candidate_pooled[pooled_extra_cols].reset_index(drop=True),
        ],
        axis=1,
    )

    finite_cols = ["z_recursive_um", "N_eff", "g1_margin_min", "g1_margin_max"]
    valid = np.ones(len(scored), dtype=bool)
    for c in finite_cols:
        if c in scored:
            valid &= np.isfinite(pd.to_numeric(scored[c], errors="coerce").values)
    if "z_recursive_um" in scored:
        zvals = pd.to_numeric(scored["z_recursive_um"], errors="coerce").values
        valid &= zvals >= 0.0
        valid &= zvals < float(args.max_depth_clip_um) * 0.999
    scored["virtual_physics_valid"] = valid.astype(int)
    scored = scored[scored["virtual_physics_valid"] == 1].reset_index(drop=True)

    measured_ref = measured_pooled.copy().reset_index(drop=True)
    seen_signatures = set(measured_ref["mechanism_signature"].astype(str)) if "mechanism_signature" in measured_ref else set()
    seen_combos = set(measured_ref.apply(_event_combo, axis=1).astype(str)) if len(measured_ref) else set()

    selected_indices: List[int] = []
    selected_reference = measured_ref.copy()

    target_n = min(max(0, int(args.virtual_select_count)), len(scored))
    remaining = set(range(len(scored)))

    # Greedy recalculation is deliberate: after one novel signature is selected,
    # a second sample with the same signature no longer receives the same bonus.
    for rank in range(1, target_n + 1):
        best_idx = None
        best_score = -float("inf")
        best_parts = None
        for idx in remaining:
            row = scored.iloc[idx]
            sig = str(row.get("mechanism_signature", ""))
            combo = _event_combo(row)
            unseen_sig = 1.0 if sig not in seen_signatures else 0.0
            unseen_combo = 1.0 if combo not in seen_combos else 0.0
            timing_nov = _timing_novelty(row, selected_reference)

            score = (
                float(args.virtual_unseen_signature_bonus) * unseen_sig
                + float(args.virtual_unseen_combo_bonus) * unseen_combo
                + float(args.virtual_timing_novelty_weight) * timing_nov
            )
            if score > best_score:
                best_idx = idx
                best_score = score
                best_parts = (unseen_sig, unseen_combo, timing_nov)

        if best_idx is None:
            break
        selected_indices.append(best_idx)
        remaining.remove(best_idx)
        row = scored.iloc[best_idx]
        sig = str(row.get("mechanism_signature", ""))
        combo = _event_combo(row)
        seen_signatures.add(sig)
        seen_combos.add(combo)

        if best_parts is not None:
            scored.loc[best_idx, "selection_score_at_pick"] = best_score
            scored.loc[best_idx, "selection_unseen_signature_at_pick"] = best_parts[0]
            scored.loc[best_idx, "selection_unseen_combo_at_pick"] = best_parts[1]
            scored.loc[best_idx, "selection_timing_novelty_at_pick"] = best_parts[2]
            scored.loc[best_idx, "selection_rank"] = rank

        selected_reference = pd.concat(
            [selected_reference, pd.DataFrame([row])],
            axis=0, ignore_index=True, sort=False,
        )

    scored["selected_virtual"] = 0
    if selected_indices:
        scored.loc[selected_indices, "selected_virtual"] = 1
    selected = scored.loc[selected_indices].copy().reset_index(drop=True) if selected_indices else pd.DataFrame()
    if not selected.empty:
        selected = selected.sort_values("selection_rank").reset_index(drop=True)
    return scored, selected


def _perturb_physics_params(
    base: PhysicsParams,
    rng: np.random.Generator,
    frac: float,
    args: argparse.Namespace,
) -> PhysicsParams:
    d = asdict(base)
    frac = max(0.0, float(frac))

    def mult(name: str, lo: float, hi: float) -> None:
        v = float(d[name])
        factor = math.exp(rng.normal(0.0, frac))
        d[name] = float(np.clip(v * factor, lo, hi))

    mult("phi_th1_j_cm2", args.phi_min, args.phi_max)
    # S is bounded and should not be multiplicatively perturbed near zero.
    d["S"] = float(np.clip(float(d["S"]) + rng.normal(0.0, frac * 0.25), args.S_min, args.S_max))
    mult("alpha_d", max(args.alpha_min, 1e-9), args.alpha_max)
    d["c_w"] = float(np.clip(float(d["c_w"]) * math.exp(rng.normal(0.0, frac)), args.cw_min, args.cw_max))
    for key in [
        "delta_eff_200fs_um", "delta_eff_500fs_um", "delta_eff_1000fs_um",
        "delta_eff_2000fs_um", "delta_eff_4000fs_um",
    ]:
        v = float(d[key])
        d[key] = float(np.clip(v * math.exp(rng.normal(0.0, frac)), args.delta_min, args.delta_max))
    return PhysicsParams(**d)


def add_signature_stability(
    selected: pd.DataFrame,
    params: PhysicsParams,
    optical: OpticalConfig,
    args: argparse.Namespace,
    max_pass: int,
) -> pd.DataFrame:
    """Add a sensitivity-based signature-stability diagnostic.

    This is *not* a calibrated probability. It is the fraction of small
    parameter perturbations that reproduce the nominal E1/E2/E5 signature.
    """
    if selected is None or selected.empty:
        return selected
    b = max(0, int(args.virtual_stability_samples))
    out = selected.copy()
    if b <= 0:
        out["signature_stability_fraction"] = np.nan
        out["signature_stability_samples"] = 0
        return out

    rng = np.random.default_rng(int(args.seed) + 2701)
    stability = []
    for _, row in out.iterrows():
        nominal = str(row.get("mechanism_signature", ""))
        same = 0
        for _j in range(b):
            p_pert = _perturb_physics_params(
                params, rng, float(args.virtual_param_perturb_frac), args
            )
            _seq, pooled = pure_physics_trace_for_row(row, p_pert, optical, args, max_pass)
            if str(pooled.get("mechanism_signature", "")) == nominal:
                same += 1
        stability.append(same / b)
    out["signature_stability_fraction"] = stability
    out["signature_stability_samples"] = b
    return out


def run_mechanism_transition_virtual_generation(
    all_df: pd.DataFrame,
    measured_df: pd.DataFrame,
    pooled_all: pd.DataFrame,
    metrics_df: pd.DataFrame,
    params: PhysicsParams,
    optical: OpticalConfig,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, Any]:
    """Generate and export mechanism-transition-guided virtual process samples."""
    if measured_df.empty:
        print("[virtual generation] no measured rows; skipped.")
        return {}

    max_pass = max(
        int(np.nanmax(all_df["pass_count"].astype(float).values)),
        int(args.virtual_pass_max) if args.virtual_pass_max is not None else 1,
    )
    candidates = generate_executable_virtual_candidate_pool(measured_df, args)
    if candidates.empty:
        print("[virtual generation] candidate pool is empty after duplicate filtering.")
        return {}

    print(f"[virtual generation] evaluating {len(candidates)} executable candidates")
    _seq_c, pooled_c, _mask_c = build_features(
        candidates, params, optical, args, max_pass=max_pass
    )
    measured_idx = np.where(all_df["has_measured_depth"].values == 1)[0]
    measured_pooled = pooled_all.iloc[measured_idx].reset_index(drop=True)

    scored, selected = score_and_select_virtual_candidates(
        candidates, pooled_c, measured_pooled, args
    )
    selected = add_signature_stability(
        selected, params, optical, args, max_pass=max_pass
    )

    # Physics-only virtual label: explicitly retain its origin. This is an
    # incomplete-physics label and should not be misrepresented as measurement.
    if not selected.empty and "z_recursive_um" in selected:
        selected["physics_virtual_depth_um"] = selected["z_recursive_um"].astype(float)

    # Optional observation-mapper label. The mapper is fitted only on measured
    # rows, but this full-data label is for dataset generation/deployment, not
    # for unbiased augmentation evaluation. Any augmentation benchmark must
    # regenerate labels inside each training fold.
    if (
        not selected.empty
        and args.virtual_label_source in {"best_surrogate", "both"}
        and metrics_df is not None
        and not metrics_df.empty
    ):
        try:
            final_model, best_name, cols = fit_best_final_tabular_model(
                metrics_df, pooled_all, all_df, args
            )
            if final_model is not None and cols:
                selected_idx = selected["run_id"].astype(str).tolist()
                pooled_by_id = pooled_c.copy().reset_index(drop=True)
                pooled_by_id.insert(0, "run_id", candidates["run_id"].astype(str).values)
                pooled_by_id = pooled_by_id.set_index("run_id")
                X_sel = pooled_by_id.loc[selected_idx, cols].reset_index(drop=True)
                pred = _predict_depth_from_final_model(final_model, X_sel, args)
                selected["surrogate_virtual_depth_um"] = pred
                selected["surrogate_virtual_label_model"] = best_name
        except Exception as e:
            print(f"[virtual generation] best-surrogate labelling failed: {e}")

    if args.virtual_label_source == "best_surrogate" and "surrogate_virtual_depth_um" in selected:
        selected["virtual_depth_um"] = selected["surrogate_virtual_depth_um"]
        selected["virtual_depth_source"] = "best_surrogate_fit_on_measured_rows"
    else:
        selected["virtual_depth_um"] = selected.get("physics_virtual_depth_um", np.nan)
        selected["virtual_depth_source"] = "recurrent_physics_z_recursive_um"
        if args.virtual_label_source == "both":
            selected["virtual_depth_source"] = "physics_primary; surrogate_label_also_exported"

    # Real mechanism signatures for direct coverage comparison.
    signature_cols = [
        "run_id", "measured_depth_um", "pulse_width_fs", "repetition_rate_khz",
        "scan_speed_mm_s", "hatch_spacing_um", "pass_count",
    ]
    real_sig = pd.concat(
        [
            all_df.iloc[measured_idx][signature_cols].reset_index(drop=True),
            measured_pooled[
                [c for c in [
                    "mechanism_signature", "event_combo_E1E2E5", "event_order_E1E2E5",
                    *MECHANISM_EVENT_FLAG_COLS, *MECHANISM_EVENT_TIME_COLS,
                    "g1_margin_min", "g1_margin_max",
                    "g2_inc_minus_defocus_min", "g2_inc_minus_defocus_max",
                    "g5_delta_increment_min_um", "g5_delta_increment_max_um",
                ] if c in measured_pooled.columns]
            ].reset_index(drop=True),
        ],
        axis=1,
    )
    real_sig.to_csv(output_dir / "mechanism_signatures_measured_rows.csv", index=False)
    scored.to_csv(output_dir / "virtual_candidate_pool_mechanism_scored.csv", index=False)
    selected.to_csv(output_dir / "virtual_process_samples_selected.csv", index=False)

    real_cov = (
        real_sig.groupby(
            ["mechanism_signature", "event_combo_E1E2E5", "event_order_E1E2E5"],
            dropna=False,
        )
        .size().reset_index(name="n_measured")
        .sort_values("n_measured", ascending=False)
    )
    real_cov.to_csv(output_dir / "mechanism_signature_coverage_measured.csv", index=False)

    summary = {
        "candidate_count_requested": int(args.virtual_candidate_count),
        "candidate_count_after_duplicate_filter": int(len(candidates)),
        "candidate_count_physically_valid": int(len(scored)),
        "selected_count": int(len(selected)),
        "measured_signature_count": int(real_sig["mechanism_signature"].nunique()) if len(real_sig) else 0,
        "selected_signature_count": int(selected["mechanism_signature"].nunique()) if len(selected) else 0,
        "new_selected_signatures_vs_measured": sorted(
            set(selected["mechanism_signature"].astype(str)) -
            set(real_sig["mechanism_signature"].astype(str))
        ) if len(selected) else [],
        "virtual_label_source": args.virtual_label_source,
        "note": (
            "signature_stability_fraction is a local parameter-sensitivity diagnostic, "
            "not a calibrated probability. Full-data surrogate virtual labels must not "
            "be used for unbiased augmentation claims without fold-internal regeneration."
        ),
    }
    (output_dir / "virtual_generation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[virtual generation] selected {len(selected)} samples; "
        f"new signatures={len(summary['new_selected_signatures_vs_measured'])}"
    )
    return {
        "candidates": candidates,
        "scored": scored,
        "selected": selected,
        "real_signatures": real_sig,
        "summary": summary,
    }



# -----------------------------
# Leakage-free augmentation benchmark
# -----------------------------


def _merge_candidate_with_pooled(candidate_df: pd.DataFrame, pooled_df: pd.DataFrame) -> pd.DataFrame:
    """Merge candidate recipes and their physics features without duplicate column names."""
    extra = [c for c in pooled_df.columns if c not in candidate_df.columns]
    return pd.concat(
        [candidate_df.reset_index(drop=True), pooled_df[extra].reset_index(drop=True)],
        axis=1,
    )


def _physics_valid_virtual_rows(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Apply only deterministic physics-sanity filters; no test-fold information is used."""
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()
    out = df.copy()
    valid = np.ones(len(out), dtype=bool)
    for c in ["z_recursive_um", "N_eff", "g1_margin_min", "g1_margin_max"]:
        if c in out.columns:
            valid &= np.isfinite(pd.to_numeric(out[c], errors="coerce").values)
    if "z_recursive_um" in out.columns:
        z = pd.to_numeric(out["z_recursive_um"], errors="coerce").values
        valid &= z >= 0.0
        valid &= z < float(args.max_depth_clip_um) * 0.999
    return out.loc[valid].reset_index(drop=True)


def _standardize_from_reference(
    reference: pd.DataFrame,
    target: pd.DataFrame,
    cols: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Median-impute and standardize using reference rows only."""
    if not cols:
        return np.empty((len(reference), 0)), np.empty((len(target), 0))
    ref = reference[list(cols)].apply(pd.to_numeric, errors="coerce").astype(float)
    tar = target[list(cols)].apply(pd.to_numeric, errors="coerce").astype(float)
    med = ref.median(axis=0).fillna(0.0)
    ref = ref.fillna(med)
    tar = tar.fillna(med)
    mu = ref.mean(axis=0).values
    sd = ref.std(axis=0, ddof=0).values
    sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
    return (ref.values - mu) / sd, (tar.values - mu) / sd


def select_core5_coverage_candidates(
    candidate_df: pd.DataFrame,
    candidate_pooled: pd.DataFrame,
    train_pooled: pd.DataFrame,
    n_select: int,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Greedy farthest-point selection in the *training-fold-defined* core5 space.

    This is the explicit static-feature-space comparator for the mechanism-transition
    strategy. Scaling, imputation, and coverage reference all use training rows only.
    """
    merged = _physics_valid_virtual_rows(
        _merge_candidate_with_pooled(candidate_df, candidate_pooled), args
    )
    if merged.empty or n_select <= 0:
        return pd.DataFrame()
    fs = select_feature_sets(train_pooled).get("lowdim_area_proxy_pulse_core5", [])
    cols = [c for c in fs if c in train_pooled.columns and c in merged.columns]
    if not cols:
        return merged.head(min(n_select, len(merged))).copy().reset_index(drop=True)
    ref_z, cand_z = _standardize_from_reference(train_pooled, merged, cols)
    if cand_z.shape[0] == 0:
        return pd.DataFrame()
    if ref_z.shape[0] > 0:
        d2 = ((cand_z[:, None, :] - ref_z[None, :, :]) ** 2).sum(axis=2)
        min_d2 = np.min(d2, axis=1)
    else:
        min_d2 = np.full(cand_z.shape[0], np.inf)
    selected: List[int] = []
    remaining = np.ones(cand_z.shape[0], dtype=bool)
    for rank in range(1, min(int(n_select), cand_z.shape[0]) + 1):
        masked = np.where(remaining, min_d2, -np.inf)
        idx = int(np.argmax(masked))
        if not np.isfinite(masked[idx]):
            break
        selected.append(idx)
        remaining[idx] = False
        d_new = ((cand_z - cand_z[idx][None, :]) ** 2).sum(axis=1)
        min_d2 = np.minimum(min_d2, d_new)
        merged.loc[idx, "core5_selection_rank"] = rank
        merged.loc[idx, "core5_min_distance_at_pick"] = math.sqrt(max(float(masked[idx]), 0.0))
    return merged.iloc[selected].copy().reset_index(drop=True)


def select_lhs_augmentation_samples(
    train_df: pd.DataFrame,
    params_fold: PhysicsParams,
    optical: OpticalConfig,
    args: argparse.Namespace,
    max_pass: int,
    n_select: int,
    seed: int,
) -> pd.DataFrame:
    """Generate the raw-process-space LHS comparator using training-fold bounds only."""
    if n_select <= 0:
        return pd.DataFrame()
    # A small oversampling factor compensates for recipes removed as duplicates.
    count = max(int(n_select), int(math.ceil(n_select * 1.5)))
    cand = generate_executable_virtual_candidate_pool(
        train_df, args, candidate_count=count, seed=seed
    )
    if cand.empty:
        return pd.DataFrame()
    _seq, pooled, _mask = build_features(cand, params_fold, optical, args, max_pass=max_pass)
    merged = _physics_valid_virtual_rows(_merge_candidate_with_pooled(cand, pooled), args)
    return merged.head(min(int(n_select), len(merged))).copy().reset_index(drop=True)


def _make_aug_model(model_name: str, n_features: int, seed: int):
    model = make_sklearn_model(model_name, seed, n_features=n_features)
    if not isinstance(model, Pipeline) and canonical_model_name(model_name) not in EXTERNAL_MODEL_NAMES:
        model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
    return model


def _fit_with_optional_sample_weight(model: Any, X: pd.DataFrame, y: np.ndarray, weight: Optional[np.ndarray]) -> Any:
    """Fit while using sample weights when the final estimator supports them."""
    if weight is None:
        model.fit(X, y)
        return model
    try:
        if isinstance(model, Pipeline):
            final_name = list(model.named_steps.keys())[-1]
            model.fit(X, y, **{f"{final_name}__sample_weight": weight})
        else:
            model.fit(X, y, sample_weight=weight)
    except (TypeError, ValueError):
        # GPR and some external estimators do not expose sample_weight.
        model.fit(X, y)
    return model


def _predict_aug_model(model: Any, X: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    pred_t = model.predict(X)
    pred = np.expm1(pred_t) if args.target_transform == "log1p" else pred_t
    return np.clip(np.asarray(pred, dtype=float), 0.0, args.max_depth_clip_um)


def generate_fold_internal_virtual_labels(
    train_pooled: pd.DataFrame,
    y_train: np.ndarray,
    virtual_rows: pd.DataFrame,
    args: argparse.Namespace,
    seed: int,
) -> Tuple[np.ndarray, str]:
    """Create virtual labels using *training-fold data only*.

    Choices:
      physics: recurrent z_recursive_um only;
      fold_surrogate: observation mapper fitted only to the current training fold;
      residual_ridge: recurrent physics plus a Ridge residual corrector fitted only
                      to the current training fold.
    """
    if virtual_rows is None or virtual_rows.empty:
        return np.asarray([], dtype=float), "none"
    source = str(args.augmentation_label_source)
    if source == "physics":
        return np.clip(
            pd.to_numeric(virtual_rows["z_recursive_um"], errors="coerce").values.astype(float),
            0.0, args.max_depth_clip_um,
        ), "fold_physics_only"

    feature_sets = select_feature_sets(train_pooled)
    fs_name = str(args.augmentation_label_feature_set)
    if fs_name not in feature_sets:
        raise ValueError(f"Unknown augmentation label feature set: {fs_name}")
    cols = [c for c in feature_sets[fs_name] if c in train_pooled.columns and c in virtual_rows.columns]
    if not cols:
        raise ValueError(f"No usable labeler features for {fs_name}")

    if source == "fold_surrogate":
        model_name = str(args.augmentation_label_model)
        model = _make_aug_model(model_name, len(cols), seed)
        target = np.log1p(y_train) if args.target_transform == "log1p" else y_train
        model.fit(train_pooled[cols], target)
        pred = _predict_aug_model(model, virtual_rows[cols], args)
        return pred, f"fold_surrogate:{model_name}__{fs_name}"

    if source == "residual_ridge":
        if "z_recursive_um" not in train_pooled.columns or "z_recursive_um" not in virtual_rows.columns:
            raise ValueError("residual_ridge requires z_recursive_um")
        phys_train = pd.to_numeric(train_pooled["z_recursive_um"], errors="coerce").fillna(0.0).values.astype(float)
        phys_virtual = pd.to_numeric(virtual_rows["z_recursive_um"], errors="coerce").fillna(0.0).values.astype(float)
        residual = np.asarray(y_train, dtype=float) - phys_train
        residual_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(args.augmentation_residual_ridge_alpha))),
        ])
        residual_model.fit(train_pooled[cols], residual)
        correction = np.asarray(residual_model.predict(virtual_rows[cols]), dtype=float)
        pred = np.clip(phys_virtual + correction, 0.0, args.max_depth_clip_um)
        return pred, f"fold_physics_plus_residual_ridge:{fs_name}"

    raise ValueError(f"Unknown augmentation label source: {source}")


def _evaluate_augmented_fold(
    train_pooled: pd.DataFrame,
    y_train: np.ndarray,
    test_pooled: pd.DataFrame,
    virtual_rows: pd.DataFrame,
    virtual_y: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> np.ndarray:
    fs_name = str(args.augmentation_eval_feature_set)
    feature_sets = select_feature_sets(train_pooled)
    if fs_name not in feature_sets:
        raise ValueError(f"Unknown augmentation evaluation feature set: {fs_name}")
    cols = [c for c in feature_sets[fs_name] if c in train_pooled.columns and c in test_pooled.columns]
    if virtual_rows is not None and not virtual_rows.empty:
        cols = [c for c in cols if c in virtual_rows.columns]
    if not cols:
        raise ValueError(f"No usable evaluation features for {fs_name}")

    X_real = train_pooled[cols].reset_index(drop=True)
    y_real = np.asarray(y_train, dtype=float)
    if virtual_rows is not None and not virtual_rows.empty and len(virtual_y) > 0:
        X_virtual = virtual_rows[cols].reset_index(drop=True)
        X_fit = pd.concat([X_real, X_virtual], axis=0, ignore_index=True)
        y_fit = np.concatenate([y_real, np.asarray(virtual_y, dtype=float)])
        weights = np.concatenate([
            np.ones(len(y_real), dtype=float),
            np.full(len(virtual_y), float(args.augmentation_virtual_weight), dtype=float),
        ])
    else:
        X_fit = X_real
        y_fit = y_real
        weights = np.ones(len(y_real), dtype=float)

    model_name = str(args.augmentation_eval_model)
    model = _make_aug_model(model_name, len(cols), seed)
    target = np.log1p(y_fit) if args.target_transform == "log1p" else y_fit
    model = _fit_with_optional_sample_weight(model, X_fit, target, weights)
    return _predict_aug_model(model, test_pooled[cols], args)


def run_leakage_free_augmentation_benchmark(
    measured_df: pd.DataFrame,
    optical: OpticalConfig,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, Any]:
    """Repeated-CV benchmark with virtual generation, selection, and labelling inside each train fold.

    Test-fold rows are never used to: fit physics parameters, set observed virtual
    candidate ranges, construct mechanism coverage, scale core5 coverage, fit the
    virtual-label model, or train the augmented observation mapper.
    """
    if measured_df.empty:
        return {}
    measured_df = measured_df.reset_index(drop=True)
    y = measured_df["measured_depth_um"].astype(float).values
    row_ids = measured_df["run_id"].values
    folds = make_cv_splits(measured_df, y, args)
    strategies = [x.strip().lower() for x in str(args.augmentation_strategies).split(",") if x.strip()]
    strategies = [x for x in strategies if x in {"lhs", "core5", "mechanism"}]
    if not strategies:
        raise ValueError("--augmentation-strategies must contain at least one of lhs,core5,mechanism")
    all_strategies = ["baseline"] + strategies
    init = args_to_physics_params(args)
    max_pass_global = max(
        int(np.nanmax(measured_df["pass_count"].astype(float).values)),
        int(args.virtual_pass_max) if args.virtual_pass_max is not None else 1,
    )

    pred_rows: List[pd.DataFrame] = []
    split_metric_rows: List[Dict[str, Any]] = []
    virtual_export_rows: List[pd.DataFrame] = []
    oof_sum = {s: np.zeros_like(y, dtype=float) for s in all_strategies}
    oof_count = {s: np.zeros_like(y, dtype=float) for s in all_strategies}
    repeat_pred: Dict[str, Dict[int, np.ndarray]] = {s: {} for s in all_strategies}
    repeat_count: Dict[str, Dict[int, np.ndarray]] = {s: {} for s in all_strategies}

    for split_no, (repeat, fold, tr, te) in enumerate(folds, start=1):
        repeat = int(repeat)
        fold = int(fold)
        train_df = measured_df.iloc[tr].copy().reset_index(drop=True)
        test_df = measured_df.iloc[te].copy().reset_index(drop=True)
        y_train = y[tr]
        print(
            f"[augmentation benchmark] split {split_no}/{len(folds)} "
            f"r{repeat} f{fold}: train={len(tr)} test={len(te)}"
        )

        # Strict rule: any learnable physics parameters are identified from train only.
        if args.physics_fit_scope == "fixed":
            params_fold = init
        else:
            if not TORCH_AVAILABLE:
                raise RuntimeError(
                    "Leakage-free augmentation benchmark with fitted physics requires PyTorch; "
                    "use --physics-fit-scope fixed or install PyTorch."
                )
            params_fold = fit_physics_params(
                train_df, args, optical, init, args.device,
                label=f"augmentation-fold-physics r{repeat} f{fold}",
            )

        _seq_tr, pooled_train, _mask_tr = build_features(
            train_df, params_fold, optical, args, max_pass=max_pass_global
        )
        _seq_te, pooled_test, _mask_te = build_features(
            test_df, params_fold, optical, args, max_pass=max_pass_global
        )

        split_virtual: Dict[str, pd.DataFrame] = {}
        k_virtual = max(0, int(args.augmentation_select_count))
        candidate_seed = int(args.seed) + 50000 + 1000 * repeat + fold

        if "lhs" in strategies:
            split_virtual["lhs"] = select_lhs_augmentation_samples(
                train_df, params_fold, optical, args, max_pass_global,
                k_virtual, candidate_seed + 11,
            )

        need_pool = any(s in strategies for s in ["core5", "mechanism"])
        if need_pool:
            candidates = generate_executable_virtual_candidate_pool(
                train_df, args,
                candidate_count=max(int(args.augmentation_candidate_count), k_virtual),
                seed=candidate_seed + 23,
            )
            if not candidates.empty:
                _seq_c, pooled_c, _mask_c = build_features(
                    candidates, params_fold, optical, args, max_pass=max_pass_global
                )
            else:
                pooled_c = pd.DataFrame()
        else:
            candidates = pd.DataFrame()
            pooled_c = pd.DataFrame()

        if "core5" in strategies:
            split_virtual["core5"] = select_core5_coverage_candidates(
                candidates, pooled_c, pooled_train, k_virtual, args
            ) if not candidates.empty else pd.DataFrame()

        if "mechanism" in strategies:
            if candidates.empty:
                split_virtual["mechanism"] = pd.DataFrame()
            else:
                local_args = argparse.Namespace(**vars(args))
                local_args.virtual_select_count = k_virtual
                _scored, selected = score_and_select_virtual_candidates(
                    candidates, pooled_c, pooled_train, local_args
                )
                split_virtual["mechanism"] = selected

        # Baseline first: identical train/test physics representation, no virtual rows.
        baseline_pred = _evaluate_augmented_fold(
            pooled_train, y_train, pooled_test, pd.DataFrame(), np.asarray([]),
            args, seed=candidate_seed + 101,
        )
        fold_predictions = {"baseline": baseline_pred}
        fold_virtual_counts = {"baseline": 0}
        fold_label_sources = {"baseline": "measured_only"}

        for sname in strategies:
            vr = split_virtual.get(sname, pd.DataFrame())
            if vr is None or vr.empty:
                pred = baseline_pred.copy()
                label_source = "none; fallback_to_baseline"
                vy = np.asarray([], dtype=float)
            else:
                vy, label_source = generate_fold_internal_virtual_labels(
                    pooled_train, y_train, vr, args,
                    seed=candidate_seed + 200 + strategies.index(sname),
                )
                good = np.isfinite(vy)
                if not np.all(good):
                    vr = vr.loc[good].reset_index(drop=True)
                    vy = vy[good]
                pred = _evaluate_augmented_fold(
                    pooled_train, y_train, pooled_test, vr, vy, args,
                    seed=candidate_seed + 300 + strategies.index(sname),
                ) if len(vy) else baseline_pred.copy()

                if not vr.empty:
                    exp = vr.copy()
                    exp["augmentation_strategy"] = sname
                    exp["repeat"] = repeat
                    exp["fold"] = fold
                    exp["fold_virtual_depth_um"] = vy
                    exp["fold_virtual_label_source"] = label_source
                    virtual_export_rows.append(exp)
            fold_predictions[sname] = pred
            fold_virtual_counts[sname] = int(len(vy))
            fold_label_sources[sname] = label_source

        for sname, pred in fold_predictions.items():
            oof_sum[sname][te] += pred
            oof_count[sname][te] += 1.0
            if repeat not in repeat_pred[sname]:
                repeat_pred[sname][repeat] = np.zeros_like(y, dtype=float)
                repeat_count[sname][repeat] = np.zeros_like(y, dtype=float)
            repeat_pred[sname][repeat][te] = pred
            repeat_count[sname][repeat][te] += 1.0
            m = metrics_dict(y[te], pred, prefix="")
            split_metric_rows.append({
                "strategy": sname,
                "repeat": repeat,
                "fold": fold,
                "n_train_real": int(len(tr)),
                "n_test_real": int(len(te)),
                "n_virtual": int(fold_virtual_counts[sname]),
                "virtual_label_source": fold_label_sources[sname],
                "rmse": m["rmse"], "mae": m["mae"], "r2": m["r2"],
            })
            pred_rows.append(pd.DataFrame({
                "strategy": sname,
                "repeat": repeat,
                "fold": fold,
                "run_id": row_ids[te],
                "measured_depth_um": y[te],
                "pred_depth_um": pred,
                "residual_um": pred - y[te],
                "n_virtual": int(fold_virtual_counts[sname]),
            }))

    summary_rows: List[Dict[str, Any]] = []
    baseline_rmse = np.nan
    for sname in all_strategies:
        count = np.maximum(oof_count[sname], 1.0)
        oof = oof_sum[sname] / count
        met = metrics_dict(y, oof, prefix="cv_")
        rep_metrics = []
        for rep, pp in sorted(repeat_pred[sname].items()):
            cnt = repeat_count[sname][rep]
            if np.all(cnt > 0):
                rep_metrics.append(metrics_dict(y, pp, prefix=""))
        row: Dict[str, Any] = {
            "strategy": sname,
            **met,
            "cv_repeats": int(args.cv_repeats),
            "cv_folds": int(args.cv_folds),
            "eval_model": args.augmentation_eval_model,
            "eval_feature_set": args.augmentation_eval_feature_set,
            "virtual_label_source": "measured_only" if sname == "baseline" else args.augmentation_label_source,
            "virtual_weight": 0.0 if sname == "baseline" else float(args.augmentation_virtual_weight),
        }
        if rep_metrics:
            for key in ["rmse", "mae", "r2"]:
                vals = np.asarray([m[key] for m in rep_metrics if np.isfinite(m[key])], dtype=float)
                if vals.size:
                    row[f"cv_{key}_repeat_mean"] = float(np.mean(vals))
                    row[f"cv_{key}_repeat_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        if sname == "baseline":
            baseline_rmse = float(met["cv_rmse"])
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    if np.isfinite(baseline_rmse):
        summary_df["delta_cv_rmse_vs_baseline"] = summary_df["cv_rmse"] - baseline_rmse
        summary_df["rmse_improvement_pct_vs_baseline"] = np.where(
            baseline_rmse > 0,
            100.0 * (baseline_rmse - summary_df["cv_rmse"]) / baseline_rmse,
            np.nan,
        )
    summary_df = summary_df.sort_values("cv_rmse", ascending=True).reset_index(drop=True)
    split_df = pd.DataFrame(split_metric_rows)
    preds_df = pd.concat(pred_rows, axis=0, ignore_index=True) if pred_rows else pd.DataFrame()
    virtual_df = pd.concat(virtual_export_rows, axis=0, ignore_index=True) if virtual_export_rows else pd.DataFrame()

    summary_df.to_csv(output_dir / "augmentation_benchmark_cv.csv", index=False)
    split_df.to_csv(output_dir / "augmentation_benchmark_split_metrics.csv", index=False)
    preds_df.to_csv(output_dir / "augmentation_benchmark_predictions.csv", index=False)
    if not virtual_df.empty:
        virtual_df.to_csv(output_dir / "augmentation_virtual_samples_by_fold.csv", index=False)

    protocol = {
        "strict_no_leakage": True,
        "rule": (
            "For every repeated CV split, physics fitting (when learnable), candidate-range inference, "
            "mechanism/core5 coverage construction, virtual sample selection, virtual labelling, and "
            "augmented-model fitting use training-fold rows only. Test-fold rows are used only for evaluation."
        ),
        "strategies": all_strategies,
        "label_source": args.augmentation_label_source,
        "label_model": args.augmentation_label_model,
        "label_feature_set": args.augmentation_label_feature_set,
        "evaluation_model": args.augmentation_eval_model,
        "evaluation_feature_set": args.augmentation_eval_feature_set,
        "virtual_select_count": int(args.augmentation_select_count),
        "candidate_count": int(args.augmentation_candidate_count),
        "virtual_weight": float(args.augmentation_virtual_weight),
    }
    (output_dir / "augmentation_benchmark_protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("[augmentation benchmark] saved leakage-free augmentation comparison")
    return {
        "summary": summary_df,
        "split_metrics": split_df,
        "predictions": preds_df,
        "virtual_samples": virtual_df,
        "protocol": protocol,
    }


# -----------------------------
# Reporting
# -----------------------------


def _best_model_name(metrics_df: pd.DataFrame) -> Optional[str]:
    if metrics_df is None or metrics_df.empty:
        return None
    available = metrics_df[metrics_df.get("skipped", 0) == 0]
    if available.empty:
        return None
    return str(available.iloc[0]["model"])


def _best_metric_row(metrics_df: pd.DataFrame) -> Optional[pd.Series]:
    if metrics_df is None or metrics_df.empty:
        return None
    available = metrics_df[metrics_df.get("skipped", 0) == 0]
    if available.empty:
        return None
    return available.iloc[0]


def _oof_rows_for_model(pred_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Return one OOF row per measured sample for a selected model.

    The fixed/full CV branch may optionally append individual repeated-split
    predictions. For manuscript plots we use only the averaged OOF block
    (repeat == 0) to avoid duplicating points.
    """
    if pred_df is None or pred_df.empty or not model_name:
        return pd.DataFrame()
    sub = pred_df[pred_df["model"] == model_name].copy()
    if "repeat" in sub.columns:
        rep0 = sub[sub["repeat"] == 0].copy()
        if not rep0.empty:
            sub = rep0
    return sub


def _slug(text: Any) -> str:
    out = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))
    return out.strip("_") or "value"


def plot_predicted_vs_measured(pred_df: pd.DataFrame, metrics_df: pd.DataFrame, output_dir: Path, args: Optional[argparse.Namespace] = None) -> None:
    """Figure 3: predicted-versus-measured plot for the best OOF model.

    In the strict run this is the key no-leakage evidence figure. The function
    writes both the historical root-level filename and the manuscript-specific
    Fig. 3 filename under figures_manuscript/.
    """
    if plt is None or pred_df.empty or metrics_df.empty:
        return
    best_row = _best_metric_row(metrics_df)
    if best_row is None:
        return
    best_model = str(best_row["model"])
    sub = _oof_rows_for_model(pred_df, best_model)
    if sub.empty:
        return
    validation = getattr(args, "physics_fit_scope", "cv") if args is not None else "cv"
    protocol = "strict fold-physics OOF" if validation == "fold" else f"{validation} physics CV"
    rmse_v = float(best_row.get("cv_rmse", np.nan))
    mae_v = float(best_row.get("cv_mae", np.nan))
    r2_v = float(best_row.get("cv_r2", np.nan))

    fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=200)
    ax.scatter(sub["measured_depth_um"], sub["cv_pred_depth_um"], s=30, alpha=0.82)
    lo = 0.0
    hi = float(max(sub["measured_depth_um"].max(), sub["cv_pred_depth_um"].max()) * 1.08)
    if not np.isfinite(hi) or hi <= 0:
        hi = 1.0
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)
    ax.set_xlabel("Measured area-averaged depth (µm)")
    ax.set_ylabel("OOF predicted depth (µm)")
    ax.set_title("Predicted vs measured depth")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    text = f"{protocol}\n{best_model}\nRMSE = {rmse_v:.3f} µm\nMAE = {mae_v:.3f} µm\nR² = {r2_v:.3f}"
    ax.text(0.04, 0.96, text, transform=ax.transAxes, ha="left", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.78, linewidth=0.4))
    ax.grid(True, linewidth=0.3, alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_dir / "predicted_vs_measured_best_cv.png")
    fig.savefig(output_dir / "predicted_vs_measured_best_cv.pdf")
    fig_dir = output_dir / "figures_manuscript"
    ensure_dir(fig_dir)
    if validation == "fold":
        fig.savefig(fig_dir / "fig3_strict_oof_predicted_vs_measured.png")
        fig.savefig(fig_dir / "fig3_strict_oof_predicted_vs_measured.pdf")
    else:
        fig.savefig(fig_dir / "fig3_diagnostic_predicted_vs_measured.png")
        fig.savefig(fig_dir / "fig3_diagnostic_predicted_vs_measured.pdf")
    plt.close(fig)



def _predict_depth_from_final_model(model: Any, X: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    pred_t = model.predict(X)
    pred = np.expm1(pred_t) if args.target_transform == "log1p" else pred_t
    return np.clip(np.asarray(pred, dtype=float), 0.0, args.max_depth_clip_um)


def fit_best_final_tabular_model(
    metrics_df: pd.DataFrame,
    pooled_all: pd.DataFrame,
    all_df: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[Optional[Any], Optional[str], List[str]]:
    if metrics_df.empty:
        return None, None, []
    available = metrics_df[metrics_df.get("skipped", 0) == 0]
    if available.empty:
        return None, None, []
    best_model_name = str(available.iloc[0]["model"])
    base, fs_name = best_model_name.split("__", 1)
    feature_sets = select_feature_sets(pooled_all)
    if fs_name not in feature_sets:
        return None, best_model_name, []
    cols = filter_cols(pooled_all, feature_sets[fs_name])
    measured_idx = np.where(all_df["has_measured_depth"].values == 1)[0]
    y = all_df.iloc[measured_idx]["measured_depth_um"].astype(float).values
    target = np.log1p(y) if args.target_transform == "log1p" else y
    try:
        model = make_sklearn_model(base, args.seed, n_features=len(cols))
        if not isinstance(model, Pipeline) and canonical_model_name(base) not in EXTERNAL_MODEL_NAMES:
            model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
        model.fit(pooled_all.iloc[measured_idx][cols], target)
        return model, best_model_name, cols
    except Exception as e:
        print(f"[interpretability] failed to fit final model for plots: {e}")
        return None, best_model_name, cols


def run_partial_dependence_plots(
    model: Any,
    best_model_name: str,
    X: pd.DataFrame,
    features: List[str],
    output_dir: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if plt is None or model is None or X.empty:
        return pd.DataFrame()
    plot_dir = output_dir / "figures_interpretability"
    ensure_dir(plot_dir)
    for feat in features:
        if feat not in X.columns:
            continue
        vals = X[feat].replace([np.inf, -np.inf], np.nan).dropna().astype(float)
        if vals.nunique() < 2:
            continue
        if vals.nunique() <= 12:
            grid = np.sort(vals.unique())
        else:
            q = np.linspace(0.05, 0.95, int(args.pdp_grid_size))
            grid = np.unique(np.quantile(vals, q))
        pdp = []
        for xval in grid:
            X_mod = X.copy()
            X_mod[feat] = xval
            pred = _predict_depth_from_final_model(model, X_mod, args)
            pdp.append(float(np.mean(pred)))
            rows.append({"kind": "PDP", "model": best_model_name, "feature": feat, "x": float(xval), "pred_depth_um": float(np.mean(pred)), "n": len(pred)})
        fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=180)
        ax.plot(grid, pdp, marker="o", linewidth=1.4, markersize=3)
        ax.set_xlabel(feat)
        ax.set_ylabel("Partial dependence depth (µm)")
        ax.set_title(f"PDP: {feat}")
        ax.grid(True, linewidth=0.3, alpha=0.35)
        fig.tight_layout()
        fig.savefig(plot_dir / f"pdp_{feat}.png")
        fig.savefig(plot_dir / f"pdp_{feat}.pdf")
        plt.close(fig)
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(output_dir / "partial_dependence_curves.csv", index=False)
    return df


def run_ale_plots(
    model: Any,
    best_model_name: str,
    X: pd.DataFrame,
    features: List[str],
    output_dir: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if plt is None or model is None or X.empty:
        return pd.DataFrame()
    plot_dir = output_dir / "figures_interpretability"
    ensure_dir(plot_dir)
    for feat in features:
        if feat not in X.columns:
            continue
        s = X[feat].replace([np.inf, -np.inf], np.nan).astype(float)
        valid = s.dropna()
        if valid.nunique() < 3:
            continue
        q = np.linspace(0, 1, int(args.ale_bins) + 1)
        edges = np.unique(np.quantile(valid, q))
        if len(edges) < 3:
            continue
        effects, centers, counts = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            if hi <= lo:
                continue
            idx = ((s >= lo) & (s <= hi)).fillna(False).values if hi == edges[-1] else ((s >= lo) & (s < hi)).fillna(False).values
            if idx.sum() < 1:
                continue
            X_lo = X.loc[idx].copy(); X_hi = X.loc[idx].copy()
            X_lo[feat] = lo; X_hi[feat] = hi
            diff = _predict_depth_from_final_model(model, X_hi, args) - _predict_depth_from_final_model(model, X_lo, args)
            effects.append(float(np.mean(diff)))
            centers.append(float((lo + hi) / 2.0))
            counts.append(int(idx.sum()))
        if len(effects) < 2:
            continue
        ale = np.cumsum(np.asarray(effects, dtype=float))
        weights = np.asarray(counts, dtype=float)
        ale = ale - float(np.average(ale, weights=weights))
        for xval, aval, cnt in zip(centers, ale, counts):
            rows.append({"kind": "ALE", "model": best_model_name, "feature": feat, "x": xval, "ale_depth_um": float(aval), "bin_count": cnt})
        fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=180)
        ax.plot(centers, ale, marker="o", linewidth=1.4, markersize=3)
        ax.axhline(0, linestyle="--", linewidth=0.8)
        ax.set_xlabel(feat)
        ax.set_ylabel("ALE effect on depth (µm)")
        ax.set_title(f"ALE: {feat}")
        ax.grid(True, linewidth=0.3, alpha=0.35)
        fig.tight_layout()
        fig.savefig(plot_dir / f"ale_{feat}.png")
        fig.savefig(plot_dir / f"ale_{feat}.pdf")
        plt.close(fig)
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(output_dir / "ale_curves.csv", index=False)
    return df


def plot_model_comparison(metrics_df: pd.DataFrame, output_dir: Path, top_n: int = 18) -> None:
    if plt is None or metrics_df.empty or "cv_rmse" not in metrics_df.columns:
        return
    sub = metrics_df[metrics_df.get("skipped", 0) == 0].copy().head(top_n)
    if sub.empty:
        return
    fig_dir = output_dir / "figures_manuscript"
    ensure_dir(fig_dir)
    labels = sub["model"].astype(str).values[::-1]
    vals = sub["cv_rmse"].astype(float).values[::-1]
    fig, ax = plt.subplots(figsize=(7.2, max(3.5, 0.28 * len(labels))), dpi=180)
    ax.barh(np.arange(len(labels)), vals)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("CV RMSE (µm)")
    ax.set_title("Model comparison")
    ax.grid(axis="x", linewidth=0.3, alpha=0.35)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_model_comparison_cv_rmse.png")
    fig.savefig(fig_dir / "fig_model_comparison_cv_rmse.pdf")
    plt.close(fig)


def plot_best_feature_importance(output_dir: Path) -> None:
    if plt is None:
        return
    fi_path = output_dir / "best_model_feature_importance.csv"
    if not fi_path.exists():
        return
    try:
        fi = pd.read_csv(fi_path).head(20)
    except Exception:
        return
    if fi.empty or "feature" not in fi.columns or "importance" not in fi.columns:
        return
    fig_dir = output_dir / "figures_manuscript"
    ensure_dir(fig_dir)
    sub = fi.sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(5.8, max(3.2, 0.28 * len(sub))), dpi=180)
    ax.barh(np.arange(len(sub)), sub["importance"].values)
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels(sub["feature"].astype(str).values, fontsize=8)
    ax.set_xlabel("Feature importance")
    ax.set_title("Best-model feature importance")
    ax.grid(axis="x", linewidth=0.3, alpha=0.35)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_feature_importance_best_model.png")
    fig.savefig(fig_dir / "fig_feature_importance_best_model.pdf")
    plt.close(fig)


def plot_group_ablation(group_ablation_df: pd.DataFrame, output_dir: Path) -> None:
    if plt is None or group_ablation_df is None or group_ablation_df.empty or "delta_rmse_vs_full" not in group_ablation_df.columns:
        return
    fig_dir = output_dir / "figures_manuscript"
    ensure_dir(fig_dir)
    sub = group_ablation_df[group_ablation_df.get("skipped", 0) == 0].copy()
    if sub.empty:
        return
    label_col = "removed_group" if "removed_group" in sub.columns else "feature_set"
    sub = sub.sort_values("delta_rmse_vs_full", ascending=True)
    fig, ax = plt.subplots(figsize=(5.8, max(3.0, 0.35 * len(sub))), dpi=180)
    ax.barh(np.arange(len(sub)), sub["delta_rmse_vs_full"].astype(float).values)
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels(sub[label_col].astype(str).values, fontsize=8)
    ax.set_xlabel("ΔRMSE after removing group (µm)")
    ax.set_title("Group feature ablation")
    ax.axvline(0, linewidth=0.8)
    ax.grid(axis="x", linewidth=0.3, alpha=0.35)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_group_ablation_delta_rmse.png")
    fig.savefig(fig_dir / "fig_group_ablation_delta_rmse.pdf")
    plt.close(fig)


def plot_residual_diagnostics_figures(pred_df: pd.DataFrame, metrics_df: pd.DataFrame, measured_df: pd.DataFrame, output_dir: Path) -> None:
    if plt is None or pred_df.empty or metrics_df.empty:
        return
    available = metrics_df[metrics_df.get("skipped", 0) == 0]
    if available.empty:
        return
    best_model = str(available.iloc[0]["model"])
    sub = pred_df[pred_df["model"] == best_model].copy()
    if sub.empty:
        return
    proc = measured_df[["run_id", "pulse_width_fs", "repetition_rate_khz", "scan_speed_mm_s", "hatch_spacing_um", "pass_count"]]
    sub = sub.merge(proc, on="run_id", how="left")
    fig_dir = output_dir / "figures_manuscript"
    ensure_dir(fig_dir)
    fig, ax = plt.subplots(figsize=(5.4, 3.8), dpi=180)
    ax.scatter(sub["measured_depth_um"], sub["cv_residual_um"], s=26, alpha=0.8)
    ax.axhline(0, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Measured depth (µm)")
    ax.set_ylabel("CV residual (µm)")
    ax.set_title("Residuals vs measured depth")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_residual_vs_measured_depth.png")
    fig.savefig(fig_dir / "fig_residual_vs_measured_depth.pdf")
    plt.close(fig)
    rows = []
    for gv in ["pulse_width_fs", "repetition_rate_khz", "scan_speed_mm_s", "hatch_spacing_um", "pass_count"]:
        if gv not in sub.columns:
            continue
        for val, g in sub.groupby(gv):
            rows.append({"group_variable": gv, "group_value": str(val), "n": len(g), "residual_mae_um": float(np.mean(np.abs(g["cv_residual_um"])))})
    diag = pd.DataFrame(rows)
    if not diag.empty:
        diag.to_csv(output_dir / "residual_mae_for_plot.csv", index=False)
        for gv in diag["group_variable"].unique():
            d = diag[diag["group_variable"] == gv].copy()
            d["sort_val"] = pd.to_numeric(d["group_value"], errors="coerce")
            d = d.sort_values("sort_val")
            fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=180)
            ax.bar(d["group_value"], d["residual_mae_um"])
            ax.set_xlabel(gv)
            ax.set_ylabel("Residual MAE (µm)")
            ax.set_title(f"Residual MAE by {gv}")
            ax.grid(axis="y", linewidth=0.3, alpha=0.35)
            fig.tight_layout()
            fig.savefig(fig_dir / f"fig_residual_mae_by_{gv}.png")
            fig.savefig(fig_dir / f"fig_residual_mae_by_{gv}.pdf")
            plt.close(fig)


def write_manuscript_figure_manifest(output_dir: Path, args: argparse.Namespace) -> None:
    rows = [
        ("Fig. 1", "Technical route schematic", "Use the external vector framework figure; not generated by this script."),
        ("Fig. 2", "Physics skeleton diagnostics", "Use optical configuration and recursive feature definitions; optional schematic outside script."),
        ("Fig. 3", "Predicted vs measured depth", "predicted_vs_measured_best_cv.png"),
        ("Fig. 4", "Model comparison", "figures_manuscript/fig_model_comparison_cv_rmse.png"),
        ("Fig. 5", "Core feature importance", "figures_manuscript/fig_feature_importance_best_model.png"),
        ("Fig. 6", "Group feature ablation", "figures_manuscript/fig_group_ablation_delta_rmse.png, if group ablation is enabled."),
        ("Fig. 7", "Model response trends", "figures_interpretability/pdp_*.png and/or ale_*.png"),
        ("Fig. 8", "Residual diagnostics", "figures_manuscript/fig_residual_vs_measured_depth.png and residual_mae_by_*.png"),
    ]
    lines = ["# Manuscript figure checklist", "", "| figure | purpose | generated file / note |", "| --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} |")
    lines += ["", "Supplementary figures can include all per-feature PDP/ALE curves, full residual-by-process plots, and prediction-only row diagnostics.", "", "Script switches:", f"- run_interpretability = {getattr(args, 'run_interpretability', None)}", f"- run_ale = {getattr(args, 'run_ale', None)}", f"- interpretability_features = {getattr(args, 'interpretability_features', '')}"]
    (output_dir / "manuscript_figure_checklist.md").write_text("\n".join(lines), encoding="utf-8")



# -----------------------------------------------------------------------------
# Manuscript-specific figure assembly for the revised paper figure plan
# -----------------------------------------------------------------------------


def _metric_rows_by_exact_label(metrics_df: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame()
    rows = []
    available = metrics_df[metrics_df.get("skipped", 0) == 0].copy()
    for label in labels:
        hit = available[available["model"].astype(str) == str(label)]
        if not hit.empty:
            rows.append(hit.iloc[0])
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _select_manuscript_comparison_rows(metrics_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Select 6--8 manuscript-relevant rows instead of the script's top-18 scan plot."""
    if metrics_df is None or metrics_df.empty or "cv_rmse" not in metrics_df.columns:
        return pd.DataFrame()
    available = metrics_df[metrics_df.get("skipped", 0) == 0].copy()
    if available.empty:
        return pd.DataFrame()

    exact = [x.strip() for x in str(getattr(args, "manuscript_comparison_rows", "")).split(",") if x.strip()]
    selected = _metric_rows_by_exact_label(available, exact) if exact else pd.DataFrame()

    if selected.empty:
        models = [x.strip() for x in str(getattr(args, "manuscript_comparison_models", "random_forest,rf_tuned,gbdt")).split(",") if x.strip()]
        feature_sets = [x.strip() for x in str(getattr(args, "manuscript_comparison_feature_sets", "lowdim_area_proxy_pulse_core5,lowdim_area_proxy_pulse_core8")).split(",") if x.strip()]
        raw = [x.strip() for x in str(getattr(args, "manuscript_raw_baselines", "gbdt__raw_process,random_forest__raw_process")).split(",") if x.strip()]
        labels = [f"{m}__{fs}" for fs in feature_sets for m in models] + raw
        selected = _metric_rows_by_exact_label(available, labels)

    if selected.empty:
        selected = available.sort_values("cv_rmse", ascending=True).head(int(getattr(args, "manuscript_comparison_max_rows", 8)))
    else:
        selected = selected.drop_duplicates(subset=["model"]).copy()
        selected = selected.sort_values("cv_rmse", ascending=True).head(int(getattr(args, "manuscript_comparison_max_rows", 8)))
    return selected.reset_index(drop=True)


def plot_model_comparison(metrics_df: pd.DataFrame, output_dir: Path, top_n: int = 18, args: Optional[argparse.Namespace] = None) -> None:
    """Figure 4: selected model comparison for the manuscript.

    This replaces the earlier top-18 plot. The selected rows should be drawn from
    one validation protocol only; in the strict run this is the fold-wise physics
    validation table.
    """
    if plt is None or metrics_df.empty or "cv_rmse" not in metrics_df.columns:
        return
    if args is None:
        sub = metrics_df[metrics_df.get("skipped", 0) == 0].copy().head(min(top_n, 8))
    else:
        sub = _select_manuscript_comparison_rows(metrics_df, args)
    if sub.empty:
        return
    fig_dir = output_dir / "figures_manuscript"
    ensure_dir(fig_dir)
    # Keep compact labels readable in the manuscript figure.
    labels = []
    for _, r in sub.iterrows():
        base = str(r.get("base_model", ""))
        fs = str(r.get("feature_set", ""))
        labels.append(f"{base}\n{fs}")
    vals = sub["cv_rmse"].astype(float).values
    order = np.argsort(vals)[::-1]  # worst at top, best at bottom for horizontal bars
    labels = [labels[i] for i in order]
    vals = vals[order]
    fig, ax = plt.subplots(figsize=(7.0, max(3.4, 0.46 * len(labels))), dpi=200)
    ax.barh(np.arange(len(labels)), vals)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("CV RMSE (µm)")
    ax.set_title("Selected model comparison")
    ax.grid(axis="x", linewidth=0.3, alpha=0.35)
    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax.text(v, i, f" {v:.2f}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_model_comparison_cv_rmse.png")
    fig.savefig(fig_dir / "fig_model_comparison_cv_rmse.pdf")
    fig.savefig(fig_dir / "fig4_selected_model_comparison_cv_rmse.png")
    fig.savefig(fig_dir / "fig4_selected_model_comparison_cv_rmse.pdf")
    sub.to_csv(output_dir / "manuscript_selected_model_comparison.csv", index=False)
    plt.close(fig)


def plot_figure5_feature_importance_group_ablation(group_ablation_df: Optional[pd.DataFrame], output_dir: Path) -> None:
    """Figure 5: fixed-physics diagnostic importance + group ablation panel."""
    if plt is None:
        return
    fig_dir = output_dir / "figures_manuscript"
    ensure_dir(fig_dir)
    fi = pd.DataFrame()
    fi_path = output_dir / "best_model_feature_importance.csv"
    if fi_path.exists():
        try:
            fi = pd.read_csv(fi_path).head(12)
        except Exception:
            fi = pd.DataFrame()

    grp = pd.DataFrame()
    if group_ablation_df is not None and not group_ablation_df.empty and "delta_rmse_vs_full" in group_ablation_df.columns:
        grp = group_ablation_df[group_ablation_df.get("skipped", 0) == 0].copy()
        if "removed_group" in grp.columns:
            grp = grp[grp["removed_group"].astype(str) != "__FULL__"].copy()
        grp = grp[np.isfinite(pd.to_numeric(grp["delta_rmse_vs_full"], errors="coerce"))].copy()
        grp["delta_rmse_vs_full"] = grp["delta_rmse_vs_full"].astype(float)
        grp = grp.sort_values("delta_rmse_vs_full", ascending=True)

    if fi.empty and grp.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), dpi=200)
    ax = axes[0]
    if not fi.empty and {"feature", "importance"}.issubset(fi.columns):
        d = fi.sort_values("importance", ascending=True)
        ax.barh(np.arange(len(d)), d["importance"].astype(float).values)
        ax.set_yticks(np.arange(len(d)))
        ax.set_yticklabels(d["feature"].astype(str).values, fontsize=7)
        ax.set_xlabel("Tree split importance")
        ax.set_title("(a) Feature importance")
        ax.grid(axis="x", linewidth=0.3, alpha=0.35)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Feature importance\nnot available", ha="center", va="center")

    ax = axes[1]
    if not grp.empty:
        label_col = "removed_group" if "removed_group" in grp.columns else "feature_set"
        labels = grp[label_col].astype(str).values
        vals = grp["delta_rmse_vs_full"].astype(float).values
        ax.barh(np.arange(len(labels)), vals)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.axvline(0, linewidth=0.8)
        ax.set_xlabel("ΔRMSE after removal (µm)")
        ax.set_title("(b) Group ablation")
        ax.grid(axis="x", linewidth=0.3, alpha=0.35)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Group ablation\nnot available in strict fold-physics run", ha="center", va="center")

    fig.suptitle("Fixed-physics diagnostic feature attribution", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig5_feature_importance_group_ablation.png", bbox_inches="tight")
    fig.savefig(fig_dir / "fig5_feature_importance_group_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_ale_panel(ax: Any, ale_df: pd.DataFrame, feature: str, title: str) -> bool:
    if ale_df is None or ale_df.empty:
        return False
    d = ale_df[(ale_df["kind"].astype(str) == "ALE") & (ale_df["feature"].astype(str) == feature)].copy()
    if d.empty:
        return False
    d = d.sort_values("x")
    ax.plot(d["x"].astype(float).values, d["ale_depth_um"].astype(float).values, marker="o", linewidth=1.3, markersize=3)
    ax.axhline(0, linestyle="--", linewidth=0.8)
    ax.set_xlabel(feature)
    ax.set_ylabel("ALE effect (µm)")
    ax.set_title(title)
    ax.grid(True, linewidth=0.3, alpha=0.35)
    return True


def plot_figure6_ale_residual_panel(pred_df: pd.DataFrame, metrics_df: pd.DataFrame, measured_df: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    """Figure 6: ALE trend diagnostics + strict OOF residual boundary diagnostics."""
    if plt is None or pred_df is None or pred_df.empty or metrics_df is None or metrics_df.empty:
        return
    best_model = _best_model_name(metrics_df)
    if not best_model:
        return
    sub = _oof_rows_for_model(pred_df, best_model)
    if sub.empty:
        return
    proc_cols = [c for c in ["run_id", "pulse_width_fs", "repetition_rate_khz", "scan_speed_mm_s", "hatch_spacing_um", "pass_count"] if c in measured_df.columns]
    sub = sub.merge(measured_df[proc_cols], on="run_id", how="left") if proc_cols else sub
    ale_path = output_dir / "ale_curves.csv"
    ale_df = pd.read_csv(ale_path) if ale_path.exists() else pd.DataFrame()
    features = [x.strip() for x in str(getattr(args, "manuscript_figure6_ale_features", "log1p_area_proxy_um,log1p_pulse_line_density_proxy")).split(",") if x.strip()]
    if len(features) < 2 and not ale_df.empty:
        extras = [f for f in ale_df.get("feature", pd.Series(dtype=str)).astype(str).unique() if f not in features]
        features += extras[: 2 - len(features)]
    while len(features) < 2:
        features.append("")

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.6), dpi=200)
    ok_a = _plot_ale_panel(axes[0, 0], ale_df, features[0], f"(a) ALE: {features[0]}" if features[0] else "(a) ALE")
    if not ok_a:
        axes[0, 0].axis("off")
        axes[0, 0].text(0.5, 0.5, f"ALE not available\n{features[0]}", ha="center", va="center")
    ok_b = _plot_ale_panel(axes[0, 1], ale_df, features[1], f"(b) ALE: {features[1]}" if features[1] else "(b) ALE")
    if not ok_b:
        axes[0, 1].axis("off")
        axes[0, 1].text(0.5, 0.5, f"ALE not available\n{features[1]}", ha="center", va="center")

    ax = axes[1, 0]
    ax.scatter(sub["measured_depth_um"], sub["cv_residual_um"], s=26, alpha=0.82)
    ax.axhline(0, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Measured depth (µm)")
    ax.set_ylabel("OOF residual (µm)")
    ax.set_title("(c) Residual vs measured depth")
    ax.grid(True, linewidth=0.3, alpha=0.35)

    ax = axes[1, 1]
    if "pass_count" in sub.columns:
        d = sub.copy()
        d["pass_count"] = pd.to_numeric(d["pass_count"], errors="coerce")
        by = d.groupby("pass_count", dropna=False)["cv_residual_um"].apply(lambda x: float(np.mean(np.abs(x)))).reset_index(name="residual_mae_um")
        by = by.sort_values("pass_count")
        xlabels = [str(int(v)) if pd.notna(v) and float(v).is_integer() else str(v) for v in by["pass_count"].values]
        ax.bar(xlabels, by["residual_mae_um"].values)
        ax.set_xlabel("Pass count")
        ax.set_ylabel("Residual MAE (µm)")
        ax.set_title("(d) Residual MAE by pass count")
        ax.grid(axis="y", linewidth=0.3, alpha=0.35)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "pass_count not available", ha="center", va="center")

    fig.tight_layout()
    fig_dir = output_dir / "figures_manuscript"
    ensure_dir(fig_dir)
    fig.savefig(fig_dir / "fig6_ale_residual_diagnostics.png")
    fig.savefig(fig_dir / "fig6_ale_residual_diagnostics.pdf")
    plt.close(fig)


def write_manuscript_figure_manifest(output_dir: Path, args: argparse.Namespace) -> None:
    protocol = getattr(args, "physics_fit_scope", "")
    rows = [
        ("Fig. 1", "Technical route schematic", "External vector schematic; not generated by this script."),
        ("Fig. 2", "Physical skeleton and proxy-feature construction", "External vector schematic; not generated by this script."),
        ("Fig. 3", "Strict fold-physics OOF predicted vs measured depth", "figures_manuscript/fig3_strict_oof_predicted_vs_measured.png (generated only when --physics-fit-scope fold; fixed/full runs generate fig3_diagnostic_predicted_vs_measured.png instead)"),
        ("Fig. 4", "Selected model RMSE comparison", "figures_manuscript/fig4_selected_model_comparison_cv_rmse.png"),
        ("Fig. 5", "Fixed-physics diagnostic feature importance and group ablation", "figures_manuscript/fig5_feature_importance_group_ablation.png; group panel is available only in fixed/full diagnostic runs."),
        ("Fig. 6", "ALE response trends and OOF residual-boundary diagnostics", "figures_manuscript/fig6_ale_residual_diagnostics.png"),
        ("Supplementary", "PDP curves", "figures_interpretability/pdp_*.png"),
        ("Supplementary", "Individual ALE curves", "figures_interpretability/ale_*.png"),
        ("Supplementary", "Residual MAE by all process variables", "figures_manuscript/fig_residual_mae_by_*.png"),
    ]
    lines = ["# Manuscript figure checklist", "", f"Run physics_fit_scope: `{protocol}`", "", "| figure | purpose | generated file / note |", "| --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} |")
    lines += [
        "",
        "Notes:",
        "- Use Fig. 3 and Fig. 4 from the `--physics-fit-scope fold` output directory as the main no-leakage performance evidence.",
        "- Use Fig. 5 from the `--physics-fit-scope fixed` or `full` diagnostic run; do not describe it as strict fold-physics evidence.",
        "- ALE/PDP are model-response diagnostics, not causal physical proof.",
        "",
        "Script switches:",
        f"- run_interpretability = {getattr(args, 'run_interpretability', None)}",
        f"- run_ale = {getattr(args, 'run_ale', None)}",
        f"- interpretability_features = {getattr(args, 'interpretability_features', '')}",
        f"- manuscript_comparison_models = {getattr(args, 'manuscript_comparison_models', '')}",
        f"- manuscript_comparison_feature_sets = {getattr(args, 'manuscript_comparison_feature_sets', '')}",
        f"- manuscript_raw_baselines = {getattr(args, 'manuscript_raw_baselines', '')}",
        f"- feature_sets = {getattr(args, 'feature_sets', '')}",
    ]
    (output_dir / "manuscript_figure_checklist.md").write_text("\n".join(lines), encoding="utf-8")


def run_publication_figures(
    metrics_df: pd.DataFrame,
    pred_all: pd.DataFrame,
    measured_df: pd.DataFrame,
    all_df: pd.DataFrame,
    pooled_all: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
    group_ablation_df: Optional[pd.DataFrame] = None,
) -> None:
    if plt is None:
        print("[figures] matplotlib unavailable; skip publication figures.")
        return

    # Figure 4 and standalone diagnostic components.
    plot_model_comparison(metrics_df, output_dir, args=args)
    plot_best_feature_importance(output_dir)
    if group_ablation_df is not None and not group_ablation_df.empty:
        plot_group_ablation(group_ablation_df, output_dir)

    # Final-model response diagnostics. ALE/PDP are generated before assembling Fig. 6.
    if getattr(args, "run_interpretability", True):
        model, best_name, cols = fit_best_final_tabular_model(metrics_df, pooled_all, all_df, args)
        if model is not None and cols:
            measured_idx = np.where(all_df["has_measured_depth"].values == 1)[0]
            X_m = pooled_all.iloc[measured_idx][cols].reset_index(drop=True)
            requested = [x.strip() for x in str(args.interpretability_features).split(",") if x.strip()]
            fig6_feats = [x.strip() for x in str(getattr(args, "manuscript_figure6_ale_features", "")).split(",") if x.strip()]
            features = []
            for f in fig6_feats + requested:
                if f in X_m.columns and f not in features:
                    features.append(f)
            if not features:
                features = cols[: min(5, len(cols))]
            print(f"[figures] running PDP/ALE for features: {features}")
            run_partial_dependence_plots(model, best_name or "best_model", X_m, features, output_dir, args)
            if getattr(args, "run_ale", True):
                run_ale_plots(model, best_name or "best_model", X_m, features, output_dir, args)

    # Standalone residual figures and manuscript-assembled panels.
    plot_residual_diagnostics_figures(pred_all, metrics_df, measured_df, output_dir)
    plot_figure5_feature_importance_group_ablation(group_ablation_df, output_dir)
    plot_figure6_ale_residual_panel(pred_all, metrics_df, measured_df, output_dir, args)
    write_manuscript_figure_manifest(output_dir, args)


def write_summary_md(
    output_dir: Path,
    all_df: pd.DataFrame,
    measured_df: pd.DataFrame,
    problematic_df: pd.DataFrame,
    optical: OpticalConfig,
    args: argparse.Namespace,
    params: Dict[str, Any],
    metrics_df: pd.DataFrame,
    pred_all: pd.DataFrame,
    final_pred_all: pd.DataFrame,
    pooled_all: pd.DataFrame,
    feature_ablation_df: Optional[pd.DataFrame] = None,
    group_ablation_df: Optional[pd.DataFrame] = None,
    residual_diag_df: Optional[pd.DataFrame] = None,
) -> None:
    lines: List[str] = []
    lines.append("# Mechanism-guided recurrent sequence surrogate results")
    lines.append("")
    lines.append("## 1. Positioning")
    lines.append("")
    lines.append(
        "Rows without measured depth are treated as problematic / prediction-only rows. "
        "They are excluded from training, cross-validation, loss computation, and model selection. "
        "The recurrent physical skeleton is used only to generate pass-level mechanism features; "
        "the final area-averaged depth is learned by low-capacity observation mappers."
    )
    lines.append("")
    lines.append("## 2. Data overview")
    overview = pd.DataFrame([
        ["total_rows_retained", len(all_df)],
        ["measured_depth_rows_used_for_training", len(measured_df)],
        ["problematic_prediction_only_rows", len(problematic_df)],
        ["measured_depth_min_um", float(measured_df["measured_depth_um"].min()) if len(measured_df) else np.nan],
        ["measured_depth_max_um", float(measured_df["measured_depth_um"].max()) if len(measured_df) else np.nan],
        ["measured_depth_mean_um", float(measured_df["measured_depth_um"].mean()) if len(measured_df) else np.nan],
    ], columns=["item", "value"])
    lines.append(markdown_table(overview))
    lines.append("")
    lines.append("## 3. Fixed optical inputs")
    optical_df = pd.DataFrame([
        ["actual_post_objective_average_power_w", optical.actual_post_objective_average_power_w],
        ["wavelength_nm", optical.wavelength_nm],
        ["objective_NA", optical.objective_NA],
        ["M2", optical.M2],
        ["beam_radius_um_1e2", optical.beam_radius_um_1e2],
        ["spot_diameter_um_1e2", optical.spot_diameter_um_1e2],
        ["rayleigh_length_um", optical.rayleigh_length_um],
    ], columns=["item", "value"])
    lines.append(markdown_table(optical_df))
    lines.append("")
    lines.append("## 4. Implemented route")
    lines.append("")
    lines.append("```text")
    lines.append("tau, f, calibrated P, scan speed v")
    lines.append("  -> recurrent multi-pulse physics skeleton")
    lines.append("  -> pass-level physical state sequence")
    lines.append("[physical sequence + pooled mechanism features + hatch/pass schedule]")
    lines.append("  -> data-driven observation mapper")
    lines.append("  -> area-averaged depth")
    lines.append("```")
    lines.append("")
    lines.append("## 5. Key configuration")
    cfg_items = [
        ["physics_fit_scope", args.physics_fit_scope],
        ["cw_mode", args.cw_mode],
        ["fixed_cw", args.fixed_cw],
        ["alpha_mode", args.alpha_mode],
        ["defocus_mode", args.defocus_mode],
        ["max_physics_steps", args.max_physics_steps],
        ["cv_folds", args.cv_folds],
        ["cv_repeats", args.cv_repeats],
        ["run_feature_ablation", args.run_feature_ablation],
        ["ablation_feature_sets", args.ablation_feature_sets],
        ["ablation_models", args.ablation_models],
        ["run_group_ablation", args.run_group_ablation],
        ["group_ablation_feature_sets", args.group_ablation_feature_sets],
        ["group_ablation_models", args.group_ablation_models],
        ["target_transform", args.target_transform],
        ["generate_virtual_data", args.generate_virtual_data],
        ["virtual_candidate_count", args.virtual_candidate_count],
        ["virtual_select_count", args.virtual_select_count],
        ["virtual_label_source", args.virtual_label_source],
        ["models", args.models],
        ["nn_epochs", args.nn_epochs],
        ["tf_d_model", args.tf_d_model],
        ["tf_layers", args.tf_layers],
        ["tf_heads", args.tf_heads],
    ]
    lines.append(markdown_table(pd.DataFrame(cfg_items, columns=["parameter", "value"])))
    lines.append("")
    lines.append("## 6. Physics parameters used for final feature generation")
    lines.append(markdown_table(pd.DataFrame([[k, v] for k, v in params.items()], columns=["parameter", "value"])))
    lines.append("")
    lines.append("## 7. Model comparison on measured rows")
    show_cols = [c for c in [
        "model", "base_model", "feature_set", "n_features",
        "cv_rmse", "cv_rmse_repeat_mean", "cv_rmse_repeat_std",
        "cv_mae", "cv_mae_repeat_mean", "cv_mae_repeat_std",
        "cv_r2", "cv_r2_repeat_mean", "cv_r2_repeat_std",
        "skipped", "skipped_reason"
    ] if c in metrics_df.columns]
    lines.append(markdown_table(metrics_df[show_cols] if not metrics_df.empty else metrics_df, max_rows=80))
    lines.append("")
    if not metrics_df.empty and "cv_rmse" in metrics_df.columns:
        available = metrics_df[metrics_df.get("skipped", 0) == 0]
        if not available.empty:
            best = available.iloc[0]
            lines.append("## 8. Best model")
            lines.append("")
            lines.append(f"Best CV model: `{best['model']}`")
            lines.append("")
            lines.append(markdown_table(pd.DataFrame([["cv_rmse", best.get("cv_rmse", np.nan)], ["cv_mae", best.get("cv_mae", np.nan)], ["cv_r2", best.get("cv_r2", np.nan)], ["feature_set", best.get("feature_set", "")], ["features", best.get("features", "")]], columns=["item", "value"])))
            lines.append("")
            fi_path = output_dir / "best_model_feature_importance.csv"
            if fi_path.exists():
                lines.append("### Best-model feature importance")
                try:
                    fi = pd.read_csv(fi_path)
                    lines.append(markdown_table(fi, max_rows=20))
                except Exception:
                    lines.append("\n_(feature importance file could not be read)_\n")
                lines.append("")
    lines.append("## 9. Leave-one-feature-out ablation")
    if feature_ablation_df is not None and not feature_ablation_df.empty:
        show_ab_cols = [c for c in [
            "base_model", "feature_set", "removed_feature", "n_features",
            "cv_rmse", "cv_rmse_repeat_mean", "cv_rmse_repeat_std",
            "delta_rmse_vs_full", "cv_mae", "cv_r2", "skipped", "skipped_reason"
        ] if c in feature_ablation_df.columns]
        lines.append(markdown_table(feature_ablation_df[show_ab_cols], max_rows=120))
    else:
        lines.append("\n_(not run)_\n")
    lines.append("")
    lines.append("## 10. Group-level feature ablation")
    if group_ablation_df is not None and not group_ablation_df.empty:
        show_grp_cols = [c for c in [
            "base_model", "feature_set", "removed_group", "removed_features", "n_features",
            "cv_rmse", "cv_rmse_repeat_mean", "cv_rmse_repeat_std", "delta_rmse_vs_full",
            "cv_mae", "cv_r2", "skipped", "skipped_reason"
        ] if c in group_ablation_df.columns]
        lines.append(markdown_table(group_ablation_df[show_grp_cols], max_rows=80))
    else:
        lines.append("\n_(not run or empty)_\n")
    lines.append("")

    lines.append("## 11. Residual diagnostics by group")
    if residual_diag_df is not None and not residual_diag_df.empty:
        show_res_cols = [c for c in [
            "best_model", "group_variable", "group_value", "n", "measured_mean_um", "pred_mean_um",
            "residual_mean_um", "residual_mae_um", "residual_rmse_um", "residual_min_um", "residual_max_um"
        ] if c in residual_diag_df.columns]
        lines.append(markdown_table(residual_diag_df[show_res_cols], max_rows=120))
    else:
        lines.append("\n_(not run or empty)_\n")
    lines.append("")

    lines.append("## 12. Worst measured-row CV residuals")
    if not pred_all.empty and not metrics_df.empty:
        available = metrics_df[metrics_df.get("skipped", 0) == 0]
        if not available.empty:
            best_model = str(available.iloc[0]["model"])
            worst = pred_all[pred_all["model"] == best_model].sort_values("abs_cv_residual_um", ascending=False)
            # attach process columns
            proc = measured_df[["run_id", "pulse_width_fs", "repetition_rate_khz", "scan_speed_mm_s", "hatch_spacing_um", "pass_count"]]
            worst = worst.merge(proc, on="run_id", how="left")
            lines.append(markdown_table(worst, max_rows=30))
        else:
            lines.append("\n_(no available model)_\n")
    else:
        lines.append("\n_(empty)_\n")
    lines.append("")
    lines.append("## 13. Prediction-only problematic rows")
    if len(problematic_df):
        pred_prob = final_pred_all[final_pred_all["is_problematic_prediction_only"] == 1].merge(
            all_df[["run_id", "pulse_width_fs", "repetition_rate_khz", "scan_speed_mm_s", "hatch_spacing_um", "pass_count"]],
            on="run_id",
            how="left",
        )
        cols = [c for c in ["run_id", "pulse_width_fs", "repetition_rate_khz", "scan_speed_mm_s", "hatch_spacing_um", "pass_count", "measured_depth_um", "has_measured_depth", "problematic_note", "best_model", "final_pred_depth_um"] if c in pred_prob.columns]
        lines.append(markdown_table(pred_prob[cols], max_rows=20))
    else:
        lines.append("\n_(none)_\n")
    lines.append("")
    lines.append("## 14. Output files")
    files = sorted([p.name for p in output_dir.iterdir() if p.is_file()])
    lines.append(markdown_table(pd.DataFrame(files, columns=["file"])))
    (output_dir / "all_key_results_for_analysis.md").write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# Main
# -----------------------------


def args_to_physics_params(args: argparse.Namespace) -> PhysicsParams:
    return PhysicsParams(
        phi_th1_j_cm2=args.phi_th1_j_cm2,
        S=args.S,
        c_w=args.fixed_cw if args.cw_mode == "fixed" else args.c_w,
        alpha_d=args.alpha_d,
        delta_eff_200fs_um=args.delta_eff_200fs_um,
        delta_eff_500fs_um=args.delta_eff_500fs_um,
        delta_eff_1000fs_um=args.delta_eff_1000fs_um,
        delta_eff_2000fs_um=args.delta_eff_2000fs_um,
        delta_eff_4000fs_um=args.delta_eff_4000fs_um,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mechanism-guided recurrent sequence surrogate for measured-only depth prediction.")
    p.add_argument("--input", type=str, default="./predictions_depth_only_v2_60rows_for_tuning.csv")
    p.add_argument("--output-dir", type=str, default="./outputs/depth_mechanism_sequence_surrogate_v3")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--show-progress", action="store_true", default=True)
    p.add_argument("--no-progress", dest="show_progress", action="store_false")
    # Optical inputs
    p.add_argument("--actual-power-w", type=float, default=5.33333)
    p.add_argument("--wavelength-nm", type=float, default=1030.0)
    p.add_argument("--objective-na", type=float, default=0.45)
    p.add_argument("--M2", type=float, default=1.2)
    # Physics mode
    p.add_argument("--physics-fit-scope", type=str, default="fixed", choices=["fixed", "full", "fold"], help="fixed: use provided physics params; full: fit once on all measured rows for diagnostic features; fold: strict fold-wise physics fitting.")
    p.add_argument("--defocus-mode", type=str, default="shared", choices=["history_only", "shared", "separate"])
    p.add_argument("--cw-mode", type=str, default="fixed", choices=["fixed", "learnable"])
    p.add_argument("--fixed-cw", type=float, default=1.0)
    p.add_argument("--alpha-mode", type=str, default="learnable", choices=["fixed", "learnable"], help="Currently used for documentation; fixed alpha can be passed through --alpha-d.")
    # Initial/fixed physics parameters
    p.add_argument("--phi-th1-j-cm2", type=float, default=16.3967)
    p.add_argument("--S", type=float, default=0.79652)
    p.add_argument("--c-w", type=float, default=1.0)
    p.add_argument("--alpha-d", type=float, default=0.779889)
    p.add_argument("--delta-eff-200fs-um", type=float, default=0.043899)
    p.add_argument("--delta-eff-500fs-um", type=float, default=0.0450921)
    p.add_argument("--delta-eff-1000fs-um", type=float, default=0.0284155)
    p.add_argument("--delta-eff-2000fs-um", type=float, default=0.0164486)
    p.add_argument("--delta-eff-4000fs-um", type=float, default=0.0333915)
    # Physics bounds/fitting
    p.add_argument("--phi-min", type=float, default=0.05)
    p.add_argument("--phi-max", type=float, default=300.0)
    p.add_argument("--S-min", type=float, default=0.2)
    p.add_argument("--S-max", type=float, default=1.0)
    p.add_argument("--alpha-min", type=float, default=0.0)
    p.add_argument("--alpha-max", type=float, default=3.0)
    p.add_argument("--cw-min", type=float, default=0.5)
    p.add_argument("--cw-max", type=float, default=3.0)
    p.add_argument("--delta-min", type=float, default=0.001)
    p.add_argument("--delta-max", type=float, default=0.25)
    p.add_argument("--max-physics-steps", type=int, default=64)
    p.add_argument("--max-step-weight", type=float, default=150.0)
    p.add_argument("--max-depth-clip-um", type=float, default=500.0)
    p.add_argument("--physics-epochs", type=int, default=800)
    p.add_argument("--physics-lr", type=float, default=0.002)
    p.add_argument("--physics-weight-decay", type=float, default=1e-4)
    p.add_argument("--physics-lambda-linear", type=float, default=1.0)
    p.add_argument("--physics-patience", type=int, default=250)
    p.add_argument("--physics-print-every", type=int, default=100)
    # ML models
    p.add_argument("--models", type=str, default="random_forest,rf_tuned,rf_conservative,gbdt,gbdt_tuned,gbdt_deeper,extra_trees,extra_trees_tuned,extra_trees_conservative,gpr,ridge")
    p.add_argument("--feature-sets", type=str, default="raw_process,engineered_baseline,physics_only_recurrent_states,raw_plus_physical_states,lowdim_area_proxy_pulse_core5,lowdim_area_proxy_pulse_core8", help="Comma-separated main feature sets to evaluate. Use 'all' for the full exploratory feature-set matrix.")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--cv-repeats", type=int, default=5, help="Repeated K-fold repeats for tabular models; strict fold-physics refits physics inside every repeated split. Neural models use only the first repeat by default for speed.")
    p.add_argument("--no-feature-ablation", dest="run_feature_ablation", action="store_false", help="Disable leave-one-feature-out ablation diagnostics.")
    p.set_defaults(run_feature_ablation=True)
    p.add_argument("--ablation-feature-sets", type=str, default="lowdim_area_proxy_4", help="Comma-separated feature sets for leave-one-feature-out ablation. Default focuses on the current formal 4-feature baseline.")
    p.add_argument("--ablation-models", type=str, default="best", help="Comma-separated base models for feature ablation. Use 'best' to ablate the best CV model/feature set.")
    p.add_argument("--no-group-ablation", dest="run_group_ablation", action="store_false", help="Disable group-level feature ablation diagnostics.")
    p.set_defaults(run_group_ablation=True)
    p.add_argument("--group-ablation-feature-sets", type=str, default="engineered_baseline,lowdim_area_proxy_pulse", help="Comma-separated feature sets for group-level ablation. Use engineered and compact physics proxy sets by default.")
    p.add_argument("--group-ablation-models", type=str, default="best", help="Comma-separated base models for group-level ablation. Use 'best' for best CV model/feature-set pair.")
    p.add_argument("--save-repeated-cv-predictions", action="store_true", help="Also append individual repeated-split predictions to cv_predictions_all_models.csv.")
    p.add_argument("--neural-use-repeated-cv", action="store_true", help="Train optional neural sequence models on all repeated CV splits; slower.")
    p.add_argument("--target-transform", type=str, default="log1p", choices=["none", "log1p"])
    # NN settings
    p.add_argument("--nn-epochs", type=int, default=900)
    p.add_argument("--nn-lr", type=float, default=0.001)
    p.add_argument("--nn-weight-decay", type=float, default=1e-3)
    p.add_argument("--nn-hidden", type=int, default=32)
    p.add_argument("--nn-dropout", type=float, default=0.2)
    p.add_argument("--nn-patience", type=int, default=150)
    p.add_argument("--nn-print-every", type=int, default=100)
    p.add_argument("--tf-d-model", type=int, default=16)
    p.add_argument("--tf-heads", type=int, default=2)
    p.add_argument("--tf-layers", type=int, default=1)
    p.add_argument("--include-neural-in-fold-physics", action="store_true", help="Reserved; strict fold-physics neural training is skipped in v1 to control runtime.")
    p.add_argument("--grad-clip", type=float, default=2.0)
    # Mechanism-transition virtual data generation
    p.add_argument("--generate-virtual-data", action="store_true", help="Generate executable virtual process candidates guided by E1/E2/E5 mechanism-transition coverage.")
    p.add_argument("--virtual-candidate-count", type=int, default=5000, help="Number of executable process candidates generated before mechanism screening.")
    p.add_argument("--virtual-select-count", type=int, default=100, help="Number of mechanism-diverse virtual process samples to retain.")
    p.add_argument("--virtual-pulse-width-values", type=str, default="", help="Comma-separated executable pulse widths (fs). Empty uses observed discrete values.")
    p.add_argument("--virtual-repetition-rate-values", type=str, default="", help="Comma-separated executable repetition rates (kHz). Empty uses observed discrete values.")
    p.add_argument("--virtual-scan-speed-min", type=float, default=None)
    p.add_argument("--virtual-scan-speed-max", type=float, default=None)
    p.add_argument("--virtual-hatch-min", type=float, default=None)
    p.add_argument("--virtual-hatch-max", type=float, default=None)
    p.add_argument("--virtual-pass-min", type=int, default=None)
    p.add_argument("--virtual-pass-max", type=int, default=None)
    p.add_argument("--virtual-unseen-signature-bonus", type=float, default=10.0)
    p.add_argument("--virtual-unseen-combo-bonus", type=float, default=3.0)
    p.add_argument("--virtual-timing-novelty-weight", type=float, default=1.0)
    p.add_argument("--virtual-stability-samples", type=int, default=12, help="Parameter-perturbation replicates for mechanism-signature stability; 0 disables.")
    p.add_argument("--virtual-param-perturb-frac", type=float, default=0.08, help="Log-scale relative perturbation used only for the signature-stability sensitivity diagnostic.")
    p.add_argument("--virtual-label-source", type=str, default="both", choices=["physics", "best_surrogate", "both"], help="Virtual response label export. 'both' retains physics label and an optional best-observation-mapper label.")
    # Strict, fold-internal augmentation benchmark (no test-fold leakage)
    p.add_argument("--run-augmentation-benchmark", action="store_true", help="Compare measured-only vs LHS/core5/mechanism-transition augmentation with all virtual generation and labelling performed inside each CV training fold.")
    p.add_argument("--augmentation-strategies", type=str, default="lhs,core5,mechanism", help="Comma-separated leakage-free augmentation comparators: lhs,core5,mechanism.")
    p.add_argument("--augmentation-candidate-count", type=int, default=2000, help="Per-fold candidate pool size used by core5/mechanism selection.")
    p.add_argument("--augmentation-select-count", type=int, default=30, help="Maximum number of virtual samples added inside each training fold for each strategy.")
    p.add_argument("--augmentation-label-source", type=str, default="residual_ridge", choices=["physics", "fold_surrogate", "residual_ridge"], help="Leakage-free virtual labeler. All choices are fitted/evaluated using training-fold information only.")
    p.add_argument("--augmentation-label-model", type=str, default="gpr", help="Training-fold-only observation mapper used when --augmentation-label-source fold_surrogate.")
    p.add_argument("--augmentation-label-feature-set", type=str, default="lowdim_area_proxy_pulse_core5", help="Feature set used by the fold-internal virtual labeler/residual corrector.")
    p.add_argument("--augmentation-residual-ridge-alpha", type=float, default=3.0, help="Ridge regularization for the fold-internal residual correction labeler.")
    p.add_argument("--augmentation-eval-model", type=str, default="gbdt", help="Fixed downstream model for a fair augmentation comparison; do not choose it from full-data CV results.")
    p.add_argument("--augmentation-eval-feature-set", type=str, default="lowdim_area_proxy_pulse_core5", help="Fixed feature representation used to compare baseline/LHS/core5/mechanism augmentation.")
    p.add_argument("--augmentation-virtual-weight", type=float, default=0.5, help="Training weight for each virtual sample when the downstream estimator supports sample_weight.")
    # Publication-oriented figures and interpretability
    p.add_argument("--no-interpretability", dest="run_interpretability", action="store_false", help="Disable final-model PDP/ALE interpretability plots.")
    p.set_defaults(run_interpretability=True)
    p.add_argument("--interpretability-features", type=str, default="area_proxy_um,log1p_area_proxy_um,log1p_pulse_line_density_proxy,pass_count,inv_hatch_spacing_um", help="Comma-separated features for PDP/ALE trend diagnostics.")
    p.add_argument("--manuscript-comparison-rows", type=str, default="", help="Optional exact comma-separated model labels for manuscript Figure 4, e.g. random_forest__lowdim_area_proxy_pulse_core5,gbdt__raw_process. If empty, rows are assembled from model/feature-set/baseline switches.")
    p.add_argument("--manuscript-comparison-models", type=str, default="random_forest,gbdt,extra_trees,gpr", help="Base models crossed with --manuscript-comparison-feature-sets for manuscript Figure 4.")
    p.add_argument("--manuscript-comparison-feature-sets", type=str, default="raw_process,engineered_baseline,raw_plus_physical_states,physics_only_recurrent_states,lowdim_area_proxy_pulse_core5,lowdim_area_proxy_pulse_core8", help="Feature sets crossed with --manuscript-comparison-models for manuscript Figure 4.")
    p.add_argument("--manuscript-raw-baselines", type=str, default="gpr__raw_process,random_forest__raw_process,gbdt__raw_process,gpr__engineered_baseline,random_forest__engineered_baseline,gbdt__engineered_baseline,gpr__raw_plus_physical_states,random_forest__raw_plus_physical_states,gpr__lowdim_area_proxy_pulse_core5,random_forest__lowdim_area_proxy_pulse_core5", help="Exact baseline labels appended to manuscript Figure 4.")
    p.add_argument("--manuscript-comparison-max-rows", type=int, default=12, help="Maximum number of bars shown in manuscript Figure 4.")
    p.add_argument("--manuscript-figure6-ale-features", type=str, default="log1p_area_proxy_um,log1p_pulse_line_density_proxy", help="Two ALE features used for manuscript Figure 6 panels a-b. Individual ALE/PDP files are still generated for --interpretability-features.")
    p.add_argument("--pdp-grid-size", type=int, default=24, help="Number of quantile grid points for PDP curves.")
    p.add_argument("--ale-bins", type=int, default=8, help="Number of quantile bins for first-order ALE curves.")
    p.add_argument("--no-ale", dest="run_ale", action="store_false", help="Disable ALE curves; PDP curves are still generated when interpretability is enabled.")
    p.set_defaults(run_ale=True)
    args = p.parse_args(argv)
    args.models_list = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.target_transform == "none":
        args.target_transform = "none"
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    seed_everything(args.seed)
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    print("[mechanism-sequence] Starting pipeline")
    print(f"[mechanism-sequence] input={input_path}")
    print(f"[mechanism-sequence] output_dir={output_dir}")
    if args.device == "cuda" and (not TORCH_AVAILABLE or not torch.cuda.is_available()):
        print("[mechanism-sequence] CUDA requested but unavailable; using CPU for neural/physics modules.")
        args.device = "cpu"

    optical = OpticalConfig(
        actual_post_objective_average_power_w=args.actual_power_w,
        wavelength_nm=args.wavelength_nm,
        objective_NA=args.objective_na,
        M2=args.M2,
    )
    all_df = read_depth_csv(input_path)
    measured_df = all_df[all_df["has_measured_depth"] == 1].copy().reset_index(drop=True)
    problematic_df = all_df[all_df["is_problematic_prediction_only"] == 1].copy().reset_index(drop=True)
    all_df.to_csv(output_dir / "clean_depth_dataset_all_rows.csv", index=False)
    measured_df.to_csv(output_dir / "measured_depth_training_rows.csv", index=False)
    problematic_df.to_csv(output_dir / "problematic_rows_raw.csv", index=False)

    init_params = args_to_physics_params(args)
    params_used = init_params
    start = time.time()
    if args.physics_fit_scope == "full":
        print("[mechanism-sequence] Fitting physics parameters once on all measured rows (diagnostic; not strict CV)")
        params_used = fit_physics_params(measured_df, args, optical, init_params, args.device, label="full physics fit")
        metrics_df, pred_all, context = run_fixed_or_full_cv(all_df, measured_df, params_used, optical, args, output_dir)
    elif args.physics_fit_scope == "fold":
        print("[mechanism-sequence] Running strict fold-wise physics feature CV")
        metrics_df, pred_all, context = run_fold_physics_cv(all_df, measured_df, optical, args, output_dir)
        params_used = PhysicsParams(**context["params"])
    else:
        print("[mechanism-sequence] Using fixed physics parameters as feature generator")
        metrics_df, pred_all, context = run_fixed_or_full_cv(all_df, measured_df, params_used, optical, args, output_dir)

    final_pred_all = context["final_pred_all"]
    feature_ablation_df = pd.DataFrame()
    group_ablation_df = pd.DataFrame()
    measured_idx = np.where(all_df["has_measured_depth"].values == 1)[0]
    pooled_m = context["pooled_all"].iloc[measured_idx].reset_index(drop=True) if "pooled_all" in context else pd.DataFrame()
    y_m = all_df.loc[measured_idx, "measured_depth_um"].astype(float).values
    row_ids_m = all_df.loc[measured_idx, "run_id"].values
    if args.run_feature_ablation and args.physics_fit_scope != "fold" and not pooled_m.empty:
        print("[mechanism-sequence] Running leave-one-feature-out ablation diagnostics")
        feature_ablation_df = run_leave_one_feature_ablation(pooled_m, y_m, row_ids_m, args, metrics_df, output_dir)
    elif args.run_feature_ablation and args.physics_fit_scope == "fold":
        print("[mechanism-sequence] Feature ablation skipped for --physics-fit-scope fold to avoid mixing strict fold-physics features with final-fit features.")
    if args.run_group_ablation and args.physics_fit_scope != "fold" and not pooled_m.empty:
        print("[mechanism-sequence] Running group-level feature ablation diagnostics")
        group_ablation_df = run_group_feature_ablation(pooled_m, y_m, row_ids_m, args, metrics_df, output_dir)
    elif args.run_group_ablation and args.physics_fit_scope == "fold":
        print("[mechanism-sequence] Group ablation skipped for --physics-fit-scope fold to avoid mixing strict fold-physics features with final-fit features.")
    virtual_generation = {}
    if args.generate_virtual_data:
        print("[mechanism-sequence] Running mechanism-transition-guided virtual process generation")
        virtual_generation = run_mechanism_transition_virtual_generation(
            all_df=all_df,
            measured_df=measured_df,
            pooled_all=context["pooled_all"],
            metrics_df=metrics_df,
            params=PhysicsParams(**context.get("params", asdict(params_used))),
            optical=optical,
            args=args,
            output_dir=output_dir,
        )
    augmentation_benchmark = {}
    if args.run_augmentation_benchmark:
        print("[mechanism-sequence] Running strict fold-internal augmentation benchmark")
        augmentation_benchmark = run_leakage_free_augmentation_benchmark(
            measured_df=measured_df,
            optical=optical,
            args=args,
            output_dir=output_dir,
        )
    residual_diag_df = residual_diagnostics_by_group(pred_all, metrics_df, measured_df, output_dir)
    metrics_df.to_csv(output_dir / "model_compare_cv.csv", index=False)
    pred_all.to_csv(output_dir / "cv_predictions_all_models.csv", index=False)
    final_pred_all.to_csv(output_dir / "best_model_predictions_all_rows.csv", index=False)
    final_pred_all[final_pred_all["is_problematic_prediction_only"] == 1].to_csv(output_dir / "prediction_only_problematic_rows.csv", index=False)
    Path(output_dir / "physics_params_used.json").write_text(json.dumps(context.get("params", asdict(params_used)), indent=2, ensure_ascii=False), encoding="utf-8")
    Path(output_dir / "optical_config.json").write_text(json.dumps({
        **asdict(optical),
        "beam_radius_um_1e2": optical.beam_radius_um_1e2,
        "spot_diameter_um_1e2": optical.spot_diameter_um_1e2,
        "rayleigh_length_um": optical.rayleigh_length_um,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_predicted_vs_measured(pred_all, metrics_df, output_dir, args)
    run_publication_figures(metrics_df, pred_all, measured_df, all_df, context["pooled_all"], output_dir, args, group_ablation_df)
    write_summary_md(output_dir, all_df, measured_df, problematic_df, optical, args, context.get("params", asdict(params_used)), metrics_df, pred_all, final_pred_all, context["pooled_all"], feature_ablation_df, group_ablation_df, residual_diag_df)
    print(f"[mechanism-sequence] Finished in {format_seconds(time.time() - start)}")
    print(f"[mechanism-sequence] Summary: {output_dir / 'all_key_results_for_analysis.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
