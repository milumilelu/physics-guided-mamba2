"""Counterexample tests (runbook 22): cache reset, sealed design, firewall."""
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import torch
from src.task_state_learning.models import B1ClosureModel
from src.task_state_learning.benchmarks import load_terminal_only
from src.task_state_learning.design import seal_choices, evaluate_sealed


def sample(control_seed=3, length=40):
    gen = torch.Generator().manual_seed(control_seed)
    ctrl = torch.rand(1, length, 2, generator=gen, dtype=torch.float64) * 2 - 1
    dt = torch.rand(1, length, generator=gen, dtype=torch.float64) * .07 + .01
    mask = torch.ones(1, length, dtype=torch.bool)
    return ctrl, dt, mask


class MemoryReset(unittest.TestCase):
    def test_continuation_and_fresh_reset_differ(self):
        """A carried cache must change results; a fresh reset restores the baseline."""
        torch.manual_seed(0)
        for mid in ('GRU', 'MAMBA2'):
            model = B1ClosureModel(mid).double()
            ctrl, dt, mask = sample()
            with torch.no_grad():
                fresh = model(ctrl, dt, mask)
                state = model(ctrl[:, :17], dt[:, :17], mask[:, :17], return_state=True)
                continued = model(ctrl, dt, mask, initial_state=state)
            self.assertFalse(torch.allclose(fresh, continued, atol=1e-9),
                             f'{mid}: carried cache had no effect (reset semantics broken)')
            state2 = model(ctrl[:, :17], dt[:, :17], mask[:, :17], return_state=True)
            with torch.no_grad():
                rerun = model(ctrl[:, 17:], dt[:, 17:], mask[:, 17:],
                              initial_state=state2)
                fresh_tail = model(ctrl[:, 17:], dt[:, 17:], mask[:, 17:])
            self.assertFalse(torch.allclose(rerun, fresh_tail, atol=1e-9),
                             f'{mid}: tail run ignored the carried cache')


class SealedDesign(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_seal_rejects_modified_choices(self):
        frame = pd.DataFrame({'pair_id': [0, 0], 'candidate_id': ['A0', 'B1'],
                              'chosen_id': ['A0', 'A0']})
        path = Path(self.tmp.name) / 'choices.csv'
        seal_choices(path, frame)
        tampered = frame.copy()
        tampered.loc[0, 'chosen_id'] = 'B1'
        tampered.to_csv(path, index=False)
        with self.assertRaises(ValueError):
            evaluate_sealed(path, pd.DataFrame({'candidate_id': ['A0'], 'y': [1.0]}),
                            'candidate_id')

    def test_seal_rejects_label_columns(self):
        frame = pd.DataFrame({'pair_id': [0], 'candidate_id': ['A0'], 'chosen_id': ['A0'],
                              'measured_y': [1.0]})
        with self.assertRaises(ValueError):
            seal_choices(Path(self.tmp.name) / 'bad.csv', frame)

    def test_evaluator_rejects_measured_column_collision(self):
        frame = pd.DataFrame({'pair_id': [0], 'candidate_id': ['A0'], 'chosen_id': ['A0']})
        path = Path(self.tmp.name) / 'ok.csv'
        seal_choices(path, frame)
        with self.assertRaises(ValueError):
            evaluate_sealed(path, pd.DataFrame({'candidate_id': ['A0'], 'chosen_id': ['A0']}),
                            'candidate_id')

    def test_clean_join_passes(self):
        frame = pd.DataFrame({'pair_id': [0], 'candidate_id': ['A0'], 'chosen_id': ['A0']})
        path = Path(self.tmp.name) / 'ok.csv'
        seal_choices(path, frame)
        merged = evaluate_sealed(path, pd.DataFrame({'candidate_id': ['A0'], 'y': [2.0]}),
                                 'candidate_id')
        self.assertEqual(len(merged), 1)
        self.assertIn('y', merged.columns)


class SupervisionFirewall(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_loader_rejects_hidden_arrays(self):
        path = Path(self.tmp.name) / 'leak.npz'
        np.savez(path, control=np.zeros((2, 4, 2)), physical_dt=np.ones((2, 4)),
                 mask=np.ones((2, 4), bool), terminal_x=np.zeros((2, 2)),
                 family_id=np.array(['a', 'b']), hidden_q=np.zeros((2, 4)))
        with self.assertRaises(ValueError):
            load_terminal_only(path)

    def test_loader_accepts_contract_arrays(self):
        path = Path(self.tmp.name) / 'ok.npz'
        np.savez(path, control=np.zeros((2, 4, 2)), physical_dt=np.ones((2, 4)),
                 mask=np.ones((2, 4), bool), terminal_x=np.zeros((2, 2)),
                 family_id=np.array(['a', 'b']))
        values = load_terminal_only(path)
        self.assertEqual(set(values), {'control', 'physical_dt', 'mask', 'terminal_x', 'family_id'})


if __name__ == '__main__':
    unittest.main()
