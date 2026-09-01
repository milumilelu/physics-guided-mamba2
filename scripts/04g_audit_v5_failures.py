#!/usr/bin/env python3
"""Audit v5 hard failures against pre-registration contrast diagnostics."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "outputs/rectangle_registration"


def main() -> int:
    metrics = pd.read_csv(ROOT / "registration/translation_metrics_v5.csv")
    diagnostics = pd.read_csv(ROOT / "inventory/contrast_diagnostics.csv")
    merged = metrics.merge(
        diagnostics[["session_id", "sample_id", "q50_minus_q05"]],
        on=["session_id", "sample_id"], validate="one_to_one"
    )
    merged["contrast_quartile"] = merged.groupby("session_id")[
        "q50_minus_q05"
    ].transform(
        lambda values: pd.qcut(
            values, 4, labels=["Q1_shallow", "Q2", "Q3", "Q4_deep"],
            duplicates="drop"
        )
    )
    merged["hard_stop"] = merged["status"].eq("STOP")
    rows = []
    for (sid, quartile), group in merged.groupby(
        ["session_id", "contrast_quartile"], observed=True
    ):
        rows.append({
            "session_id": sid,
            "contrast_quartile": str(quartile),
            "samples": len(group),
            "pass": int(group["status"].eq("PASS").sum()),
            "review": int(group["status"].eq("REVIEW").sum()),
            "stop": int(group["hard_stop"].sum()),
            "stop_rate": float(group["hard_stop"].mean()),
        })
    pd.DataFrame(rows).to_csv(
        ROOT / "registration/v5_failure_by_contrast_quartile.csv",
        index=False, encoding="utf-8-sig"
    )
    shallow = merged["contrast_quartile"].eq("Q1_shallow")
    shallow_rate = float(merged.loc[shallow, "hard_stop"].mean())
    other_rate = float(merged.loc[~shallow, "hard_stop"].mean())
    reasons: dict[str, int] = {}
    for warning in merged.loc[merged["hard_stop"], "warning"].fillna(""):
        for reason in str(warning).split("; "):
            if reason:
                reasons[reason] = reasons.get(reason, 0)+1
    decision = (
        "STOP_SYSTEMATIC_SHALLOW_FAILURE"
        if shallow_rate > other_rate else "STOP_NONSHALLOW_FAILURES_REMAIN"
    )
    summary = {
        "stage": "WP8_v5_failure_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "samples": len(merged),
        "hard_stops": int(merged["hard_stop"].sum()),
        "shallow_quartile_stop_rate": shallow_rate,
        "other_quartiles_stop_rate": other_rate,
        "risk_ratio_shallow_vs_other": (
            shallow_rate/other_rate if other_rate > 0 else None
        ),
        "contrast_vs_joint_evidence_correlation": float(np.corrcoef(
            merged["q50_minus_q05"], merged["joint_evidence_total"]
        )[0, 1]),
        "hard_stop_reason_counts": dict(
            sorted(reasons.items(), key=lambda item: (-item[1], item[0]))
        ),
        "evidence_level": 3,
        "note": (
            "All 200 samples used the same frozen v5 estimator and gates. "
            "No failed sample was excluded or replaced by a prior method."
        ),
    }
    (ROOT / "registration/v5_failure_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if int(merged["hard_stop"].sum()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
