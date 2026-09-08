import unittest
import numpy as np
import pandas as pd
from src.task_state_learning.grouping import components, nested_splits, validate_partition, assign_folds


def example():
    n = 30
    df = pd.DataFrame({"dataset_index": range(n), "session_id": ["s"]*n, "sample_id": range(n),
                       "session_role": ["formal"]*n, "shared_height_source_id": [f"source_{i}" for i in range(n)],
                       "pulse_duration_fs": np.arange(n)+100, "frequency_kHz": [10]*n,
                       "velocity_mm_s": [5]*n, "hatch_spacing_um": [2]*n, "pass_count": [1]*n})
    df.loc[1, "shared_height_source_id"] = "source_0"
    df.loc[2, "pulse_duration_fs"] = df.loc[1, "pulse_duration_fs"]
    df.loc[2, "pass_count"] = 4
    return df


class GroupingTests(unittest.TestCase):
    def test_transitive_source_and_family_excludes_N(self):
        x = components(example())
        self.assertEqual(x.component_id.iloc[:3].nunique(), 1)
        self.assertEqual(x.component_id.nunique(), 28)

    def test_row_reordering_and_outcome_changes_cannot_change_split(self):
        x = example(); a, _ = nested_splits(components(x))
        x = x.sample(frac=1, random_state=17); x["D"] = np.arange(len(x))*1000
        b, _ = nested_splits(components(x))
        pd.testing.assert_frame_equal(a[["component_id", "outer_fold"]], b[["component_id", "outer_fold"]])

    def test_nested_test_coverage(self):
        outer, inner = nested_splits(components(example()))
        self.assertEqual(len(outer), 30)
        self.assertEqual(len(inner), 120)
        for f in range(5):
            self.assertFalse(set(outer[outer.outer_fold.eq(f)].dataset_index) & set(inner[inner.outer_fold.eq(f)].dataset_index))

    def test_deliberate_group_leak_rejected(self):
        x = components(example())
        with self.assertRaisesRegex(ValueError, "Leakage"):
            validate_partition(x, [0], list(range(1, 30)))

    def test_duplicate_identity_rejected(self):
        x = example(); x.loc[1, "sample_id"] = 0
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            components(x)

    def test_missing_family_rejected(self):
        x = example(); x.loc[1, "velocity_mm_s"] = np.nan
        with self.assertRaisesRegex(ValueError, "Null"):
            components(x)

    def test_exact_decimal_not_approximate_family(self):
        x = example(); x.loc[4, "pulse_duration_fs"] = 103
        x["hatch_spacing_um"] = x.hatch_spacing_um.astype(float)
        x.loc[4, "hatch_spacing_um"] = 2.0001
        c = components(x)
        self.assertNotEqual(c.component_id.iloc[3], c.component_id.iloc[4])

    def test_insufficient_components_rejected(self):
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            assign_folds(components(example()).iloc[:3], 5)


if __name__ == "__main__":
    unittest.main()
