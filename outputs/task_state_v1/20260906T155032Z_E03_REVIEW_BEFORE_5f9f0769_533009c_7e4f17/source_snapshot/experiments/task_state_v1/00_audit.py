"""E00: create an immutable, scoped data/split contract; never train a model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import subprocess
import numpy as np
import pandas as pd
import yaml
from src.task_state_learning.artifacts import ROOT, new_run, write_json, input_manifest, sha256, seal
from src.task_state_learning.contracts import load_contract, TARGETS
from src.task_state_learning.grouping import components, nested_splits, pair_coverage, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/task_state_v1/data_audit.yaml")
    args = parser.parse_args()
    out = new_run("E00", args.config)
    gate = {"experiment_id": "E00", "execution_status": "FAIL", "hypothesis_status": "NOT_TESTED",
            "training_executed": False, "gate_scope": "DATA_IDENTITIES_TARGET_EXPORT_NESTED_SPLITS", "G1_complete": False}
    try:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        suite = subprocess.run([sys.executable, "-X", "utf8", "-B", "-m", "unittest", "discover",
                                "-s", "tests/task_state_v1", "-p", "test_grouping.py", "-v"],
                               cwd=ROOT, encoding="utf-8", capture_output=True)
        (out / "grouping_tests.log").write_text(suite.stdout + suite.stderr, encoding="utf-8")
        require(suite.returncode == 0, "Grouping tests failed")
        cfg, frozen, df, raw, paths = load_contract()
        protocol = ROOT / config["protocol_path"]
        paths += [args.config.resolve(), protocol]
        hashes, data_hash = input_manifest(paths)
        pd.DataFrame(hashes).to_csv(out / "input_hashes.csv", index=False)
        raw.to_csv(out / "registered_raw_parity.csv", index=False)
        mainframe = components(df[df.session_role.isin(["formal", "pass_main"])])
        sizes = mainframe.groupby("component_id").size()
        require(len(sizes) >= 10 and sizes.max()/180 <= .4, "Fivefold infeasible")
        mainframe, inner = nested_splits(mainframe, 5, 3)
        identity = ["dataset_index", "session_id", "sample_id", "shared_height_source_id", "session_role", "component_id", "base_family_key"]
        mainframe[identity+["outer_fold"]].to_csv(out / "split_manifest.csv", index=False)
        inner.to_csv(out / "inner_split_manifest.csv", index=False)
        mainframe[identity].to_csv(out / "dependency_components.csv", index=False)
        df[["dataset_index", "session_id", "sample_id", "shared_height_source_id", "session_role"]].to_csv(out / "identity_map.csv", index=False)
        df.to_csv(out / "frozen_targets.csv", index=False)
        eligible = df[["dataset_index", "session_role"]].copy()
        eligible["cohort"] = np.where(eligible.session_role.eq("pass_supplement"), "SUPP20", "MAIN180")
        eligible["target_valid"] = np.isfinite(df[TARGETS]).all(axis=1)
        eligible.to_csv(out / "eligibility_manifest.csv", index=False)
        coverage = [pair_coverage(mainframe, "outer_fold").assign(split_level="outer", parent_outer_fold=-1)]
        for fold in range(5):
            frame = mainframe.drop(columns="outer_fold").merge(inner[inner.outer_fold.eq(fold)],
                                                                on=["dataset_index", "component_id"], validate="one_to_one")
            coverage.append(pair_coverage(frame, "inner_fold").assign(split_level="inner", parent_outer_fold=fold))
        pd.concat(coverage).to_csv(out / "pair_coverage_by_fold_session.csv", index=False)
        allframe = components(df)
        supp = allframe[allframe.session_role.eq("pass_supplement")]
        allframe["purged_from_MAIN_for_SUPP20"] = (~allframe.session_role.eq("pass_supplement")) & allframe.component_id.isin(supp.component_id)
        allframe[identity+["purged_from_MAIN_for_SUPP20"]].to_csv(out / "cross_cohort_dependencies.csv", index=False)
        split_hash = sha256(out / "split_manifest.csv")
        (out / "split_hash.txt").write_text(split_hash+"\n", encoding="utf-8")
        audit = {"rows": len(df), "MAIN180_components": len(sizes), "largest_component": int(sizes.max()),
                 "outer_fold_sizes": {str(k): int(v) for k,v in mainframe.outer_fold.value_counts().items()},
                 "SUPP20_purged_training_rows": int(allframe.purged_from_MAIN_for_SUPP20.sum()),
                 "raw_max_error_um": float(raw.max_abs_error_um.max()), "training_executed": False,
                 "remaining": ["canonical_observer_parity", "physics_and_backend_gates", "algorithm_environment_and_scenarios"]}
        write_json(out / "data_audit.json", audit)
        write_json(out / "group_graph_summary.json", {"n_components": len(sizes), "size_histogram": {str(k):int(v) for k,v in sizes.value_counts().items()}})
        config.update(status="FROZEN_E00_SCOPE", data_sha256=data_hash, split_sha256=split_hash,
                      inner_split_sha256=sha256(out / "inner_split_manifest.csv"), target_sha256=sha256(out / "frozen_targets.csv"),
                      environment_sha256=sha256(out / "environment_lock.txt"), allowed_to_train=False)
        (out / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        gate.update(execution_status="PASS", status="PASS", gate_id="G0_DATA", data_sha256=data_hash,
                    split_sha256=split_hash, config_sha256=sha256(out / "config_resolved.yaml"),
                    evidence_boundary="Input/split freeze only; no algorithm results; retrospective internal population")
    except Exception as exc:
        gate.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
    write_json(out / "gate_G0.json", gate)
    seal(out)
    print(json.dumps({"output": str(out), **gate}, ensure_ascii=False, indent=2))
    return 0 if gate["execution_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
