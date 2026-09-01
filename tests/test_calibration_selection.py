import unittest

from src.calibration_selection import select_calibration_samples


class TestCalibrationSelection(unittest.TestCase):
    def test_selection_covers_levels_before_filling_by_snr(self):
        rows = []
        for sample_id in range(1, 13):
            rows.append({
                "sample_id": sample_id,
                "q50_minus_q05": float(sample_id),
                "edge_energy": float(sample_id),
                "selected_valid_fraction": 1.0,
                "pass": "A" if sample_id <= 6 else "B",
                "frequency": str(sample_id % 3),
            })
        selected = select_calibration_samples(
            rows, fraction=0.25, minimum=4,
            factors=["pass", "frequency"],
            weights={"contrast_pctile": 0.4, "edge_energy_pctile": 0.4,
                     "valid_fraction_pctile": 0.2})
        self.assertEqual(len(selected), 4)
        self.assertEqual({row["pass"] for row in selected}, {"A", "B"})
        self.assertEqual({row["frequency"] for row in selected}, {"0", "1", "2"})


if __name__ == "__main__":
    unittest.main()
