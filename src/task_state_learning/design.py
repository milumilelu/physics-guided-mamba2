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
import numpy as np

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


def family_pair_interval(differences,endpoints,level=.9833,bootstrap=5000,seed=20260907):
    """Resample history families, assigning pair weight count(i)*count(j).

    Conditional on the fixed fitted models, anchors and candidate-pair graph.
    Pairs sharing an endpoint are not treated as independent observations.
    """
    values=np.asarray(differences,dtype=float)
    links=np.asarray(endpoints)
    require(links.shape==(len(values),2) and len(values)>=2,'Pair endpoint shape mismatch')
    require(np.isfinite(values).all() and (links[:,0]!=links[:,1]).all(),'Invalid pair contrast')
    nodes,indices=np.unique(links,return_inverse=True)
    indices=indices.reshape(-1,2)
    rng=np.random.default_rng(seed);stats=[]
    for _ in range(bootstrap):
        counts=rng.multinomial(len(nodes),np.full(len(nodes),1/len(nodes)))
        weights=counts[indices[:,0]]*counts[indices[:,1]]
        if weights.sum():stats.append(float(np.average(values,weights=weights)))
    require(len(stats)>=.95*bootstrap,'Too few defined family-bootstrap replicates')
    alpha=(1-level)/2
    return {'mean':float(values.mean()),'low':float(np.quantile(stats,alpha)),
            'high':float(np.quantile(stats,1-alpha)),'n_families':len(nodes),
            'valid_bootstrap_replicates':len(stats),'resampling_unit':'history_family',
            'interval_scope':'conditional_on_fixed_models_anchors_and_pair_graph'}
