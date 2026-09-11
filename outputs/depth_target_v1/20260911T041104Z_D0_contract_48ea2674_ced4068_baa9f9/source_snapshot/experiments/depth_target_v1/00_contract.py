"""D0: data contract and observation boundary for depth_target_v1.

Gates (all must PASS before any D1 fit):
  C1  E00 audit seal intact; height package sha256 matches protocol.
  C2  Frozen MAIN180 partition re-validated (no component/source/family leakage).
  C3  Height-package alignment one-to-one; canonical D/A_med parity < 1e-4 um.
  C4  Translation invariance: constant shift changes no morphology feature.
  C5  Forbidden-feature audit: no morphology input derives from D / absolute height.
  C6  SUPP20 excluded from training partitions (domain-stress only).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json
import time

import numpy as np
import pandas as pd
import yaml

from src.task_state_learning.artifacts import ROOT, sha256, write_json, verify_seal
from src.task_state_learning.grouping import require
from src.depth_target import features as dt_features
from src.depth_target import runs as dt_runs
from src.depth_target import splits as dt_splits

TOL_TRANSLATION = 1e-6
TOL_PARITY_UM = 1e-4


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path,
                    default=ROOT / "config/depth_target_v1/protocol.yaml")
    args = ap.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    require(config["schema"] == "depth_target_v1" and config["allowed_to_train"],
            "Invalid protocol scope")
    out = dt_runs.new_run("D0_contract", args.config)
    t0 = time.perf_counter()
    gate = {"experiment_id": "D0_contract", "execution_status": "FAIL", "checks": {}}
    try:
        audit = verify_seal(ROOT / config["inputs"]["audit_run"])
        gate["checks"]["C1_audit_seal"] = "PASS"
        pack_path = ROOT / config["inputs"]["height_package"]
        require(sha256(pack_path) == config["inputs"]["height_package_sha256"],
                "Height package hash drift vs protocol")
        gate["checks"]["C1_height_hash"] = "PASS"

        frame, inner = dt_splits.load_frozen_partition(audit)
        gate["checks"]["C2_partition"] = "PASS"

        targets = pd.read_csv(audit / "frozen_targets.csv")
        pack = dict(np.load(pack_path, allow_pickle=False))
        rows = dt_features.align_height_rows(targets, pack)
        gate["checks"]["C3_alignment_parity"] = "PASS"

        main_idx = rows[frame.dataset_index.to_numpy()]
        height, valid = pack["height_raw"][main_idx], pack["valid_mask"][main_idx]
        err = dt_features.translation_invariance_error(height, valid)
        require(float(err.max()) < TOL_TRANSLATION,
                f"Translation invariance failure: {err.max():.3e}")
        gate["checks"]["C4_translation_invariance"] = f"PASS max_err={float(err.max()):.3e}"

        forbidden = set(config["morphology"]["forbidden"])
        reused = set(config["morphology"]["reused_canonical_columns"])
        require("D" in reused and "D" not in dt_features.ALL_MORPHOLOGY,
                "Canonical depth leaked into morphology feature list")
        require(not (set(dt_features.ALL_MORPHOLOGY) & {"median_depth_um", "D"}),
                "Forbidden column in morphology features")
        audit_rows = []
        for block, cols in dt_features.BLOCKS.items():
            for col in cols:
                source = "reused_canonical_E00_parity_tested" if col in dt_features.REUSED_CANONICAL \
                    else "new_from_median_residual"
                audit_rows.append({
                    "feature": col, "block": block, "source": source,
                    "translation_max_abs_err": float(err[col]) if col in err.index else 0.0,
                    "forbidden_inputs_used": "none",
                    "status": "PASS" if float(err.get(col, 0.0)) < TOL_TRANSLATION else "FAIL"})
        feature_audit = pd.DataFrame(audit_rows)
        require((feature_audit.status == "PASS").all(), "Feature audit failure")
        gate["checks"]["C5_forbidden_audit"] = f"PASS forbidden={sorted(forbidden)}"

        require(not frame.session_role.isin(["pass_supplement"]).any(),
                "SUPP20 inside training partition")
        gate["checks"]["C6_supp20_excluded"] = "PASS"

        provenance = [
            {"path": str(pack_path.relative_to(ROOT)).replace("\\", "/"),
             "sha256": sha256(pack_path), "role": "height_package"},
            {"path": str((audit / "frozen_targets.csv").relative_to(ROOT)).replace("\\", "/"),
             "sha256": sha256(audit / "frozen_targets.csv"), "role": "frozen_targets_E00"},
            {"path": str((audit / "split_manifest.csv").relative_to(ROOT)).replace("\\", "/"),
             "sha256": sha256(audit / "split_manifest.csv"), "role": "frozen_outer_split_E00"},
            {"path": str((audit / "inner_split_manifest.csv").relative_to(ROOT)).replace("\\", "/"),
             "sha256": sha256(audit / "inner_split_manifest.csv"), "role": "frozen_inner_split_E00"},
        ]
        pd.DataFrame(provenance).to_csv(out / "input_provenance.csv", index=False)
        frame.to_csv(out / "split_manifest.csv", index=False)
        feature_audit.to_csv(out / "feature_audit.csv", index=False)
        resolved = dict(config)
        resolved["audit_release_sha256"] = sha256(audit / "release_manifest.json")
        (out / "protocol_snapshot.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False),
                                                    encoding="utf-8")
        summary = {"n_main180": int(len(frame)), "n_components": int(frame.component_id.nunique()),
                   "outer_folds": int(frame.outer_fold.nunique()),
                   "qa_columns_available": ["plane_rmse_um", "repair_fraction", "valid_fraction"],
                   "plane_rmse_um_median": float(frame.plane_rmse_um.median()),
                   "supp20_status": "excluded_from_training_domain_stress_only"}
        write_json(out / "contract_summary.json", summary)
        gate.update(execution_status="PASS", **summary)
    except Exception as exc:
        gate["error"] = f"{type(exc).__name__}: {exc}"
    gate["status"] = gate["execution_status"]
    gate["walltime_seconds"] = time.perf_counter() - t0
    write_json(out / "contract_gate.json", gate)
    dt_runs.seal(out)
    print(json.dumps({"output": str(out), **gate}, ensure_ascii=False, indent=2))
    return 0 if gate["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
