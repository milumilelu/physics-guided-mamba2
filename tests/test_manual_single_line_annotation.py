import unittest

import numpy as np

from src.manual_four_edge_annotation import local_extents_from_record
from src.manual_single_line_annotation import (
    DEFAULT_MINIMUM_ASPECT,
    RANGE_FIELDS,
    annotation_table_columns,
    canonical_view_pixels,
    elongated_box_record,
    elongation_is_suspicious,
    estimate_line_orientation,
    fit_reference_plane,
    plane_depth,
    rotated_crop_length_um,
)


def synthetic_strip(rows=64, cols=256, dx=0.28, dy=0.28,
                    a=0.01, b=-0.02, c=5.0, noise_sigma=0.005,
                    line_amplitude=12.0, line_sigma_um=2.5,
                    theta_deg=0.6, seed=7):
    """Strip geometry matched to the real single-line CAG groups
    (18 um tall strip, a trench narrower than the reference margin)."""
    rng = np.random.default_rng(seed)
    y, x = np.indices((rows, cols), dtype=float)
    x = (x-(cols-1)/2.0)*dx
    y = (y-(rows-1)/2.0)*dy
    theta = np.deg2rad(theta_deg)
    perpendicular = -np.sin(theta)*x+np.cos(theta)*y
    depth = line_amplitude*np.exp(-.5*(perpendicular/line_sigma_um)**2)
    z = a*x+b*y+c-depth+rng.normal(0.0, noise_sigma, size=(rows, cols))
    valid = np.ones_like(z, dtype=bool)
    return z, valid, dx, dy


class TestElongatedBoxRecord(unittest.TestCase):
    def test_derived_fields_and_round_trip(self):
        record = elongated_box_record(
            left_local_um=-100, right_local_um=100,
            top_local_um=-6, bottom_local_um=6,
            display_center_x_um=0, display_center_y_um=0, theta_deg=.6)
        self.assertAlmostEqual(record["long_axis_um"], 200)
        self.assertAlmostEqual(record["short_axis_um"], 12)
        self.assertAlmostEqual(record["aspect_ratio"], 200/12)
        restored = local_extents_from_record(
            record, display_center_x_um=0, display_center_y_um=0, theta_deg=.6)
        for actual, expected in zip(restored, (-100, 100, -6, 6)):
            self.assertAlmostEqual(actual, expected)

    def test_swapped_extents_are_sorted(self):
        record = elongated_box_record(
            left_local_um=100, right_local_um=-100,
            top_local_um=6, bottom_local_um=-6,
            display_center_x_um=3, display_center_y_um=-4, theta_deg=-.6)
        self.assertAlmostEqual(record["width_um"], 200)
        self.assertAlmostEqual(record["height_um"], 12)
        self.assertAlmostEqual(record["long_axis_um"], 200)

    def test_suspicious_aspect(self):
        record = elongated_box_record(
            left_local_um=-5, right_local_um=5, top_local_um=-4,
            bottom_local_um=4, display_center_x_um=0, display_center_y_um=0,
            theta_deg=0)
        self.assertTrue(elongation_is_suspicious(
            record, minimum_aspect=DEFAULT_MINIMUM_ASPECT))
        self.assertFalse(elongation_is_suspicious(record, minimum_aspect=1.0))
        self.assertFalse(elongation_is_suspicious({"aspect_ratio": ""}))


class TestTableColumns(unittest.TestCase):
    def test_columns_cover_range_fields(self):
        columns = annotation_table_columns("A")
        self.assertEqual(columns[:4], ["session_id", "sample_id",
                                       "measurement_id",
                                       "roi_within_measurement"])
        for field in RANGE_FIELDS:
            self.assertIn(f"annotator_a_{field}", columns)


class TestReferencePlane(unittest.TestCase):
    def test_recovers_clean_plane(self):
        z, valid, dx, dy = synthetic_strip(line_amplitude=0.0)
        fit = fit_reference_plane(z, valid, dx, dy)
        self.assertAlmostEqual(fit.a, 0.01, places=4)
        self.assertAlmostEqual(fit.b, -0.02, places=4)
        self.assertAlmostEqual(fit.c, 5.0, places=3)
        self.assertLess(fit.rmse_um, 0.02)

    def test_plane_ignores_machined_line(self):
        z, valid, dx, dy = synthetic_strip()
        fit = fit_reference_plane(z, valid, dx, dy)
        self.assertAlmostEqual(fit.a, 0.01, places=3)
        self.assertAlmostEqual(fit.b, -0.02, places=3)
        self.assertLess(fit.sigma_ref_um, 0.2)
        depth = plane_depth(z, valid, dx, dy, fit)
        self.assertTrue(np.nanmax(depth) > 8.0)
        self.assertTrue(np.all(np.isnan(depth[~valid])))

    def test_depth_negative_on_pileup(self):
        z, valid, dx, dy = synthetic_strip(line_amplitude=-4.0)
        fit = fit_reference_plane(z, valid, dx, dy)
        depth = plane_depth(z, valid, dx, dy, fit)
        # The frozen asymmetric clipping tolerates pile-up (+4 sigma), so the
        # plane rides above a wide pile-up; depth still turns negative there.
        self.assertTrue(np.nanmin(depth) < -2.0)


class TestLineOrientation(unittest.TestCase):
    def test_recovers_small_tilt(self):
        z, valid, dx, dy = synthetic_strip()
        fit = fit_reference_plane(z, valid, dx, dy)
        depth = plane_depth(z, valid, dx, dy, fit)
        estimate = estimate_line_orientation(
            depth, valid, dx, dy, sigma_ref_um=fit.sigma_ref_um)
        self.assertTrue(estimate.confident)
        self.assertAlmostEqual(estimate.theta_deg, 0.6, delta=0.15)
        self.assertGreater(estimate.signal_pixels, 100)

    def test_fallback_without_signal(self):
        z, valid, dx, dy = synthetic_strip(line_amplitude=0.0)
        fit = fit_reference_plane(z, valid, dx, dy)
        depth = plane_depth(z, valid, dx, dy, fit)
        estimate = estimate_line_orientation(
            depth, valid, dx, dy, sigma_ref_um=fit.sigma_ref_um)
        self.assertFalse(estimate.confident)
        self.assertEqual(estimate.theta_deg, 0.0)


class TestViewGeometry(unittest.TestCase):
    def test_crop_length_covers_rotated_strip(self):
        # The square canonical view must cover the full strip at any angle.
        self.assertAlmostEqual(rotated_crop_length_um(285.0, 18.0, 0.0), 289.0)
        self.assertAlmostEqual(rotated_crop_length_um(285.0, 18.0, 90.0), 289.0)
        long = rotated_crop_length_um(285.0, 18.0, 0.6)
        self.assertGreater(long, 289.0)
        self.assertLess(long, 292.0)

    def test_view_pixels_follow_four_edge_cap(self):
        self.assertEqual(canonical_view_pixels(289.3, 0.278657), 1000)
        self.assertEqual(canonical_view_pixels(10.0, 0.5), 20)
        with self.assertRaises(ValueError):
            canonical_view_pixels(0.0, 0.5)


if __name__ == "__main__":
    unittest.main()
