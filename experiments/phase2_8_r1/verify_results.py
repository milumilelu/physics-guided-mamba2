"""Read-only acceptance checks for corrected results; writes only r1 QA JSON."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "outputs/phase2_8_r1"
sp = importlib.util.spec_from_file_location("task24_r1_qa", HERE / "24_information_decomposition.py")
a = importlib.util.module_from_spec(sp)
sp.loader.exec_module(a)


def main():
    old = pd.read_csv(ROOT / "outputs/phase2_8/predictability_spectrum_folds.csv")
    new = pd.read_csv(OUT / "predictability_spectrum_folds.csv")
    keys = ["target", "variant", "model", "fold"]
    both = old.merge(new, on=keys, suffixes=("_old", "_new"), validate="one_to_one")
    assert len(both) == len(old) == len(new)
    max_d = float(abs(both.skill_q2_old - both.skill_q2_new).max())
    assert max_d < 1e-12, max_d
    oof = pd.read_csv(OUT / "predictability_oof.csv")
    qa = a.validate_oof(oof, new)
    fold_tables = pd.concat([pd.read_csv(OUT / "folds/fold_assignments.csv"),
                            pd.read_csv(OUT / "folds/inbox_fold_assignments.csv")])
    for (v, f), rows in fold_tables.groupby(["variant", "fold"]):
        train = set(rows[rows.role == "train"].dataset_index)
        test = set(rows[rows.role == "test"].dataset_index)
        assert train.isdisjoint(test)
        got = oof[(oof.variant == v) & (oof.fold == f)]
        assert set(got.dataset_index) == test
    scalar_targets = set(new.target) - {"Pl", "Ot_joint"}
    assert new[new.target.isin(scalar_targets)].r2_scalar.notna().all()
    selection = pd.read_csv(OUT / "candidate_selection.csv")
    chosen = selection[selection.selected]
    assert chosen.valid.all()
    for row in chosen.itertuples():
        assert row.held_group not in __import__("ast").literal_eval(row.training_groups)
    levels = pd.read_csv(OUT / "kernel_bridge_levels.csv")
    assert len(levels) == 35 and levels.groupby("level").size().eq(7).all()
    invalid = levels[~levels.physical_valid]
    assert (invalid.tv_cond_i == 1).all()
    assert not levels[levels.physical_valid].min_margin_um.lt(-0.010000000001).any()
    candidates = pd.read_csv(OUT / "candidate_simulations.csv")
    counterexample = candidates[(candidates.dataset_index == 177) &
                               (candidates.level == "L3b") & (candidates.param == -0.5)]
    assert len(counterexample) == 1 and not bool(counterexample.iloc[0].valid)
    assert counterexample.iloc[0].min_field_um < -4.8
    phases = pd.read_csv(OUT / "phase_grid_sensitivity.csv")
    assert set(phases.n_phases) == {16, 32, 64} and len(phases) == 15
    gate = json.loads((OUT / "summary/gsl28_b_evaluation.json").read_text())
    for lv, data in gate["G28_B1"]["levels"].items():
        if len(invalid[invalid.level == lv]):
            assert data["MODEL_CLASS_IMPROVEMENT"] == "physical_invalid"
    a.plot_spectrum(pd.read_csv(OUT / "predictability_spectrum.csv"), OUT / "predictability_spectrum.png")
    result = {"status": "PASS", "n_fold_rows": len(new),
              "max_q2_delta_vs_historical": max_d, "oof": qa,
              "n_bridge_rows": len(levels), "n_invalid_retained": len(invalid),
              "real_counterexample_rejected": True, "phase_sensitivity_complete": True}
    (OUT / "acceptance_checks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
