"""Explicit execution holds prevent concurrent launches of invalid pipelines."""
import json
from .artifacts import ROOT
from .grouping import require


def require_not_held(stage):
    path=ROOT/'config/task_state_v1/execution_hold.json'
    if path.exists():
        state=json.loads(path.read_text(encoding='utf-8'))
        require(stage not in state['held_stages'],f'Execution held for {stage}: '+state['reason'])
