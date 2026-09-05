#!/usr/bin/env python3
"""Phase 2.8r1: versioned correction of Task25; see PROTOCOL.md.

Old scripts and formal artifacts remain immutable. Cache only outcome-free
forward predictions; LOGO selection uses training group outcomes exclusively.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from src.forward_models import (array_transfer_v2, field_class,
    overlap_descriptor, pairwise_interaction_field, phase_grid_v2,
    physical_validity_relative_v2, saturate, synth_field)
from src.geometry import CODE_INVALID

spec = importlib.util.spec_from_file_location("task25_v21_inputs", REPO / "experiments/phase2_8/25_kernel_bridge.py")
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
LEVELS = ["L0", "L1", "L2", "L3a", "L3b"]


def select_candidate(candidate_rows: pd.DataFrame, held_group: str) -> tuple[float | None, list]:
    """Median of TRAINING group mean TV, ties choose smaller parameter.

    candidate_rows has group, param, loss and valid columns. Holding out a
    group excludes its validity as well as its response from selection.
    """
    train = candidate_rows[candidate_rows["group"] != held_group]
    if train.empty:
        raise ValueError("no training groups")
    scores = []
    for param, g in train.groupby("param", sort=True):
        valid = bool(g.valid.all())
        group_means = g.groupby("group").loss.mean()
        score = float(group_means.median())
        scores.append({"param": float(param), "valid": valid,
                       "score": score, "training_groups": sorted(g.group.unique().tolist())})
    eligible = [s for s in scores if s["valid"] and np.isfinite(s["score"])]
    if not eligible:
        return None, scores
    best = min(eligible, key=lambda s: (s["score"], s["param"]))
    for s in scores:
        s["tied_best"] = bool(s["valid"] and s["score"] == best["score"])
    return best["param"], scores


def admissible_q(raw_q: np.ndarray, valid: bool) -> np.ndarray:
    """Keep failed held-out rows in evaluation as the invalid class."""
    if valid:
        return np.asarray(raw_q, float).copy()
    q = np.zeros(5)
    q[CODE_INVALID] = 1.0
    return q


def simulate_row(kw: dict, h: float, level: str, param: float | None,
                 n_phases: int, t: dict) -> dict:
    pixel, roi = float(t["field_pipeline"]["pixel_um"]), float(t["field_pipeline"]["roi_um"])
    tol = float(t["physical_guard"]["tol_um"])
    opts = {"pixel_um": pixel, "roi_um": roi}
    g, x = kw["profile"], kw["x"]
    codes, min_z, min_margin, invalid_phases = [], np.inf, np.inf, 0
    phases = [0.0] if level == "L0" else phase_grid_v2(h, n_phases, level, param)
    for phi in phases:
        base = synth_field(g, x, 400.0 if level == "L0" else h, phi, 0.0, **opts)
        if level in ("L0", "L1"):
            z = base
        elif level == "L2":
            z = saturate(base, param)
        elif level == "L3a":
            z = synth_field(g, x, h, phi, param, **opts)
        else:
            z = pairwise_interaction_field(g, x, h, phi, param, **opts)
        guard = physical_validity_relative_v2(z, base, tol)
        invalid_phases += int(not guard["valid"])
        if guard["finite"]:
            min_z = min(min_z, guard["min_field_um"])
            min_margin = min(min_margin, guard["min_margin_um"])
            codes.append(field_class(z, h=h, pixel_um=pixel)[0])
        else:
            codes.append(CODE_INVALID)
    q = legacy.q_from_codes(codes)
    return {"q": q, "valid": invalid_phases == 0,
            "invalid_phases": invalid_phases,
            "min_field_um": float(min_z) if np.isfinite(min_z) else None,
            "min_margin_um": float(min_margin) if np.isfinite(min_margin) else None}


def evaluate_grid(rows: pd.DataFrame, library: dict, cfg: dict, n_phases: int,
                  b_boot: int, out: Path) -> dict:
    t = cfg["task25"]
    lv = t["levels"]
    grids = {
        "L0": [None], "L1": [None],
        "L2": list(np.geomspace(lv["L2"]["grid_um"]["lo"], lv["L2"]["grid_um"]["hi"], lv["L2"]["grid_um"]["n"])),
        "L3a": lv["L3a"]["c_grid"],
        "L3b": list(np.linspace(lv["L3b"]["gamma_per_um"]["lo"], lv["L3b"]["gamma_per_um"]["hi"], lv["L3b"]["gamma_per_um"]["n"])),
    }
    groups = sorted(rows.kernel_group.unique())
    cache, candidates, simulation_log = {}, [], []
    for i, row in rows.iterrows():
        for level in LEVELS:
            for param in grids[level]:
                param = None if param is None else float(param)
                result = simulate_row(library[int(row.line_id)], float(row.h_um), level, param, n_phases, t)
                cache[(i, level, param)] = result
                # Loss is only consumed after the held-out group is filtered.
                candidates.append({"row": i, "group": row.kernel_group, "level": level,
                    "param": param, "loss": 1 - result["q"][int(row.observed)], "valid": result["valid"]})
                simulation_log.append({"dataset_index": int(row.dataset_index), "kernel_group": row.kernel_group,
                    "level": level, "param": param, "n_phases": n_phases,
                    **{k: v for k, v in result.items() if k != "q"},
                    "q_raw": json.dumps(result["q"].tolist())})
        print(f"{n_phases} phases: simulated {i+1}/{len(rows)} kernels", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(simulation_log).to_csv(out / "candidate_simulations.csv", index=False)
    candidates = pd.DataFrame(candidates)
    chosen, selections, ties = {}, [], []
    for level in LEVELS[2:]:
        for group in groups:
            param, scores = select_candidate(candidates[candidates.level == level], group)
            chosen[(level, group)] = param
            for s in scores:
                selections.append({"level": level, "held_group": group, **s,
                                   "selected": param is not None and s["param"] == param})
            tied = [s["param"] for s in scores if s.get("tied_best", False)]
            ties.append({"level": level, "held_group": group, "chosen": param,
                         "tied_parameters": tied, "n_tied": len(tied),
                         "boundary": param in (min(grids[level]), max(grids[level]))})
    pd.DataFrame(selections).to_csv(out / "candidate_selection.csv", index=False)
    records = []
    for i, row in rows.iterrows():
        for level in LEVELS:
            param = chosen.get((level, row.kernel_group))
            no_candidate = level in LEVELS[2:] and param is None
            r = ({"q": np.eye(5)[CODE_INVALID], "valid": False,
                  "invalid_phases": n_phases, "min_field_um": None, "min_margin_um": None}
                 if no_candidate else cache[(i, level, param)])
            q = admissible_q(r["q"], r["valid"])
            records.append({"dataset_index": int(row.dataset_index), "kernel_group": row.kernel_group,
                "h_um": row.h_um, "observed": int(row.observed), "level": level, "param": param,
                "physical_valid": r["valid"], "no_training_candidate": no_candidate,
                "invalid_phases": r["invalid_phases"], "min_field_um": r["min_field_um"],
                "min_margin_um": r["min_margin_um"], "q_raw": json.dumps(r["q"].tolist()),
                "q_pred": json.dumps(q.tolist()), "tv_cond_i": 1 - q[int(row.observed)],
                "tv_raw_diagnostic": 1 - r["q"][int(row.observed)]})
    table = pd.DataFrame(records)
    table.to_csv(out / "kernel_bridge_levels.csv", index=False)
    group_tv = table.groupby(["level", "kernel_group"]).tv_cond_i.mean().unstack(0)[LEVELS]
    group_tv.to_csv(out / "kernel_bridge_groups.csv")
    invalid = table.groupby("level").physical_valid.apply(lambda s: int((~s).sum())).to_dict()
    rng = np.random.default_rng(int(cfg["meta"]["random_seed"]) + 900)
    b1 = {}
    for level in LEVELS[2:]:
        d = (group_tv.L1 - group_tv[level]).to_numpy()
        draws = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(b_boot)])
        ci = np.percentile(draws, [2.5, 97.5, 0.835, 99.165])
        statistical = bool(d.mean() >= t["g28b"]["b1"]["delta_min"] and ci[0] > 0)
        b1[level] = {"delta_tv_cond": float(d.mean()), "ci95_lower": float(ci[0]), "ci95_upper": float(ci[1]),
            "ci9833_bonferroni_lower": float(ci[2]), "ci9833_bonferroni_upper": float(ci[3]),
            "n_groups": len(d), "n_invalid_heldout_rows": invalid[level],
            "statistical_threshold_met": statistical,
            "MODEL_CLASS_IMPROVEMENT": "physical_invalid" if invalid[level] else "achieved" if statistical else "not_achieved"}
    pooled = {}
    for level in LEVELS:
        q = np.array([json.loads(x) for x in table[table.level == level].q_pred]).mean(axis=0)
        obs = legacy.q_from_codes(rows.observed.astype(int).tolist())
        pooled[level] = float(0.5 * abs(q - obs).sum())
    result = {"revision": "2.8r1", "n_phases": n_phases, "n_rows": len(rows), "n_groups": len(groups),
        "selection": "median of training kernel-group mean TV; smallest numeric parameter on exact ties",
        "guard": "z >= min(base,0) - 0.01 um at all pixels; invalid held-out rows retained as invalid class",
        "tv_cond_out_of_group": {k: float(v) for k, v in group_tv.mean().items()},
        "invalid_heldout_rows": invalid, "G28_B1": {"levels": b1},
        "pooled_tv_legacy_diagnostic": pooled, "selection_ties": ties,
        "interpretation": "Exploratory model-family comparison. Physical-invalid or tied boundary parameters cannot establish a mechanism."}
    (out / "summary").mkdir(exist_ok=True)
    (out / "summary/gsl28_b_evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(HERE / "phase2_8_config.yaml"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    t = cfg["task25"]
    out = REPO / cfg["meta"]["quick_output_root" if args.quick else "formal_output_root"]
    allowed = (REPO / "outputs" / ("phase2_8_r1_quick" if args.quick else "phase2_8_r1")).resolve()
    if not out.resolve().is_relative_to(allowed):
        raise ValueError("corrected output must stay in versioned root")
    out.mkdir(parents=True, exist_ok=True)
    library = legacy.build_kernel_library(cfg)
    rows, pop = legacy.build_primary(cfg, library, pd.read_csv(REPO / t["paths"]["lambda_over_hatch"]))
    if len(rows) < 5 or rows.kernel_group.nunique() < 3:
        raise ValueError("insufficient bridge population")
    (out / "bridge_population.json").write_text(json.dumps(pop, indent=2), encoding="utf-8")
    phases = [8] if args.quick else t["phase_sensitivity"]
    primary = 8 if args.quick else t["phase_grid"]
    b_boot = t["g28b"]["b1"]["bootstrap"]["quick_B" if args.quick else "B"]
    sensitivity = []
    for n in phases:
        dest = out if n == primary else out / "phase_sensitivity" / str(n)
        r = evaluate_grid(rows, library, cfg, n, b_boot, dest)
        for lv in LEVELS:
            sensitivity.append({"n_phases": n, "level": lv,
                "tv_cond": r["tv_cond_out_of_group"][lv], "n_invalid": r["invalid_heldout_rows"][lv],
                "delta_vs_L1": r["tv_cond_out_of_group"]["L1"] - r["tv_cond_out_of_group"][lv]})
    pd.DataFrame(sensitivity).to_csv(out / "phase_grid_sensitivity.csv", index=False)
    descriptors = []
    for row in rows.itertuples():
        kw = library[int(row.line_id)]
        g, x = kw["profile"], kw["x"]
        dx = float(np.diff(x).mean())
        freq = np.fft.rfftfreq(len(g), dx)
        mask = (freq >= 1/32) & (freq <= 1/4)
        power = abs(np.fft.rfft(g - g.mean())) ** 2
        transfer = array_transfer_v2(freq[mask], float(row.h_um), t["descriptors"]["transfer_n_lines"])
        peak = freq[mask][int(np.argmax(power[mask] * transfer))]
        descriptors.append({"dataset_index": int(row.dataset_index), "kernel_group": row.kernel_group,
            "O_h": overlap_descriptor(g, dx, float(row.h_um)),
            "r_pred_transfer": float(1/peak/row.h_um), "r_observed": row.r_observed})
    pd.DataFrame(descriptors).to_csv(out / "kernel_bridge_descriptors.csv", index=False)
    metadata = {"revision": "2.8r1", "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                "protocol_sha256": hashlib.sha256((HERE / "PROTOCOL.md").read_bytes()).hexdigest(),
                "quick": args.quick, "phase_grids": phases}
    (out / "bridge_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Task25 r1 complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
