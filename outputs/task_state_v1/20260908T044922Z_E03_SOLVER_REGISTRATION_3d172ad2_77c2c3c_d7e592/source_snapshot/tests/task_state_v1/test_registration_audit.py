import unittest
from src.task_state_learning.registration import calibration_fingerprint, complete_jobs


class RegistrationAuditTests(unittest.TestCase):
    def test_missing_duplicate_and_exception_fail_closed(self):
        self.assertTrue(complete_jobs(['a', 'b'], ['b', 'a'], []))
        self.assertFalse(complete_jobs(['a', 'b'], ['a'], []))
        self.assertFalse(complete_jobs(['a', 'b'], ['a', 'a'], []))
        self.assertFalse(complete_jobs(['a', 'b'], ['a', 'b'], ['worker failed']))

    def test_pointer_change_allowed_but_domain_change_rejected(self):
        cfg = {'parameter_bounds': {'x': [0, 1]}, 'solver_registration_run': None}
        digest = calibration_fingerprint(cfg)
        cfg['solver_registration_run'] = 'sealed-run'
        self.assertEqual(digest, calibration_fingerprint(cfg))
        cfg['parameter_bounds']['x'][1] = 2
        self.assertNotEqual(digest, calibration_fingerprint(cfg))
