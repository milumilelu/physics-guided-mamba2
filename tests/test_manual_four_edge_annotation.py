import unittest

import pandas as pd

from src.manual_four_edge_annotation import (
    canonical_box_record,
    assign_annotation_values,
    annotation_is_complete,
    first_incomplete_index,
    local_extents_from_record,
)


class TestManualFourEdgeAnnotation(unittest.TestCase):
    def test_numeric_assignment_into_pandas_string_column(self):
        table=pd.DataFrame({"annotator_a_left_u_um":pd.Series([""],dtype="str")})
        assign_annotation_values(table,0,"annotator_a_",{"left_u_um":-111.25})
        self.assertEqual(table.at[0,"annotator_a_left_u_um"],-111.25)

    def test_rotated_box_round_trip(self):
        record=canonical_box_record(
            left_local_um=-101,right_local_um=99,
            top_local_um=-98,bottom_local_um=102,
            display_center_x_um=40,display_center_y_um=-15,theta_deg=-.7)
        restored=local_extents_from_record(
            record,display_center_x_um=40,display_center_y_um=-15,theta_deg=-.7)
        for actual,expected in zip(restored,(-101,99,-98,102)):
            self.assertAlmostEqual(actual,expected)
        self.assertAlmostEqual(record["width_um"],200)
        self.assertAlmostEqual(record["height_um"],200)

    def test_resume_starts_at_first_incomplete(self):
        rows=[{"annotator_a_state":"complete"},{"annotator_a_state":""},
              {"annotator_a_state":"unusable"}]
        self.assertEqual(first_incomplete_index(rows,"a"),1)
        self.assertFalse(annotation_is_complete(rows,"a"))
        self.assertTrue(annotation_is_complete(
            [{"annotator_a_state":"complete"},{"annotator_a_state":"unusable"}],"a"))


if __name__=="__main__":
    unittest.main()
