"""Confirmation interface skeleton (Phase 2.8 v2.1 section 5.1).

DESIGN-ONLY in Phase 2.8: the three-stage lock pattern is reserved here so
that discovery-side model code can be wired to it later WITHOUT retrofitting.
Phase 2.8 scripts do NOT call this module -- all Phase 2.8 analyses remain
discovery-only, and nothing in this module has access to confirmation
outcomes (none exist yet).

Contract:
  1. ``fit(discovery_manifest)``   -> a frozen model artifact (in-memory dict)
     carrying model identity, config hash, and feature order;
  2. ``predict(confirmation_manifest)`` -> predictions for the confirmation
     conditions, computed BEFORE any confirmation measurement is unblinded;
  3. ``evaluate_locked_predictions(predictions, observed)`` -> one-shot
     comparison against the registered predictions.  Calling this more than
     once per lock is a protocol violation the caller must prevent.

A "locked prediction" is persisted as JSON (``write_lock`` / ``read_lock``)
with model version, config hash, feature order, and timestamp, so the
pre-registration is auditable.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

__all__ = ["fit", "predict", "evaluate_locked_predictions",
           "write_lock", "read_lock", "LockError"]


class LockError(RuntimeError):
    """Raised when the confirmation lock protocol is violated."""


def _config_hash(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def fit(discovery_manifest, *, model: str, config: dict,
        feature_columns: list[str]) -> dict:
    """Stage 1 -- fit on discovery data only.  Returns a model artifact.
    (Skeleton: the real estimator is wired in Phase 3.)"""
    return {
        "model": model,
        "config_hash": _config_hash(config),
        "feature_columns": list(feature_columns),
        "n_discovery": int(len(discovery_manifest)),
        "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "_estimator": None,  # Phase 3: the fitted estimator
    }


def predict(model: dict, confirmation_manifest) -> dict:
    """Stage 2 -- locked predictions for confirmation conditions, made
    BEFORE unblinding.  (Skeleton: Phase 3 wires the estimator.)"""
    if model.get("_estimator") is None:
        raise LockError("confirmation interface skeleton: no estimator "
                        "attached yet (Phase 3); predictions cannot be made")
    return {
        "model": model["model"],
        "config_hash": model["config_hash"],
        "feature_columns": model["feature_columns"],
        "n_confirmation": int(len(confirmation_manifest)),
        "predictions": None,
    }


def evaluate_locked_predictions(predictions: dict, observed) -> dict:
    """Stage 3 -- one-shot evaluation of the LOCKED predictions against the
    observed confirmation outcomes.  The lock protocol allows exactly one
    call per prediction lock; the caller owns that discipline."""
    if predictions.get("predictions") is None:
        raise LockError("predictions were never made; nothing to evaluate")
    return {
        "model": predictions["model"],
        "config_hash": predictions["config_hash"],
        "n_confirmation": predictions["n_confirmation"],
        "evaluated": True,
        "observed_n": int(len(observed)),
    }


def write_lock(predictions: dict, path: Path) -> Path:
    """Persist the locked prediction (pre-registration artifact)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in predictions.items() if k != "_estimator"}
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False),
                    encoding="utf-8")
    return path


def read_lock(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
