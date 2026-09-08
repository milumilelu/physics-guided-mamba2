"""Fail-closed registration identity and job accounting."""
import hashlib
import json


def calibration_fingerprint(config):
    # The pointer is filled AFTER registration; all other settings remain bound.
    value = {k: v for k, v in config.items() if k != 'solver_registration_run'}
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def complete_jobs(expected, observed, errors):
    return (not errors and len(expected) == len(set(expected))
            and len(observed) == len(expected) and set(observed) == set(expected))
