"""Sealed candidate selection: choices are hashed before measured labels attach.

The selector sees only predictions and candidate identities. `seal_choices`
writes the choice table and its SHA256; `evaluate_sealed` refuses to join the
measured labels until the file hash matches the sealed value and the candidate
set is unchanged. This is the counterexample-checked firewall for E04/E09.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .grouping import require


def choice_hash(frame):
    payload = json.dumps(frame.to_dict('records'), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def seal_choices(path, frame):
    """Write choices + hash BEFORE any measured label is joined."""
    frame = frame.copy()
    require(not any(str(c).lower().startswith(('measured', 'true', 'observed'))
                    for c in frame.columns), 'Choice table must not contain measured labels')
    digest = choice_hash(frame)
    frame.to_csv(path, index=False)
    Path(str(path) + '.sha256').write_text(digest, encoding='utf-8')
    return digest


def evaluate_sealed(path, measured, join_key):
    """Verify the seal, then join measured outcomes by key; regret per pair row."""
    path = Path(path)
    frame = pd.read_csv(path)
    digest = choice_hash(frame)
    sealed = Path(str(path) + '.sha256').read_text(encoding='utf-8').strip()
    require(digest == sealed, 'Sealed choice table was modified after sealing')
    require(join_key in frame.columns and join_key in measured.columns,
            'Join key missing on one side')
    require(not set(measured.columns) & set(frame.columns) - {join_key},
            'Candidate table already contains measured columns')
    merged = frame.merge(measured, on=join_key, validate='many_to_one')
    require(len(merged) == len(frame), 'Sealed choice rows lost in join')
    return merged


def pairwise_regret(merged, objective_column, pair_id_column='pair_id',
                    candidate_column='candidate_id', chosen_column='chosen_id'):
    """regret = J(chosen) - min(J_i, J_j) per pair; equal weight per pair."""
    rows = []
    for pair, group in merged.groupby(pair_id_column):
        require(len(group) == 2, f'Pair {pair} does not have exactly two candidates')
        chosen = group[group[candidate_column] == group[chosen_column].iloc[0]]
        require(len(chosen) == 1, f'Chosen id not among candidates for pair {pair}')
        values = group[objective_column]
        rows.append({'pair_id': pair, 'regret': float(chosen[objective_column].iloc[0]
                                                      - values.min()),
                     'objective_chosen': float(chosen[objective_column].iloc[0]),
                     'objective_best': float(values.min())})
    return pd.DataFrame(rows)
