import unittest

import numpy as np

from src.conical_dropout import repair_compact_dropouts


class TestCompactDropoutRepair(unittest.TestCase):
    def test_repairs_compact_downward_pit_without_mutating_input(self):
        y, x = np.mgrid[:101, :101]
        z = 0.001*x + 0.002*y
        z[(x-50)**2+(y-50)**2 <= 9] -= 3.0
        original = z.copy()
        repaired, mask, records, metrics = repair_compact_dropouts(
            z, np.ones_like(z, bool), dx_um=.5, dy_um=.5)
        np.testing.assert_array_equal(z, original)
        self.assertGreater(mask.sum(), 0)
        self.assertGreater(repaired[50, 50], z[50, 50])
        self.assertGreaterEqual(len(records), 1)
        self.assertEqual(metrics["status"], "PASS")

    def test_rejects_long_scan_groove(self):
        z = np.zeros((101, 101), dtype=float)
        z[49:52, 15:86] = -3.0
        repaired, mask, records, _ = repair_compact_dropouts(
            z, np.ones_like(z, bool), dx_um=.5, dy_um=.5)
        self.assertFalse(mask.any())
        self.assertEqual(records, [])
        np.testing.assert_array_equal(repaired, z)

    def test_never_repairs_boundary(self):
        z = np.zeros((101, 101), dtype=float)
        z[2:5, 40:44] = -4.0
        _, mask, _, _ = repair_compact_dropouts(
            z, np.ones_like(z, bool), dx_um=.5, dy_um=.5)
        self.assertFalse(mask.any())


if __name__ == "__main__":
    unittest.main()
