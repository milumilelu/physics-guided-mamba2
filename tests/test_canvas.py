import unittest

from src.canvas import available_centered_square_um, resolve_registered_grid


class TestCommonCanvas(unittest.TestCase):
    def test_off_center_sample_limits_canvas(self):
        size = available_centered_square_um(
            fov_width_um=704.0, fov_height_um=528.0,
            center_x_um=252.0, center_y_um=0.0, theta_deg=0.0)
        self.assertAlmostEqual(size, 200.0)

    def test_rotation_accounts_for_corner_projection(self):
        size = available_centered_square_um(
            fov_width_um=704.0, fov_height_um=528.0,
            center_x_um=0.0, center_y_um=0.0, theta_deg=45.0)
        self.assertAlmostEqual(size, 528.0/(2**0.5))

    def test_grid_is_not_finer_than_input(self):
        result = resolve_registered_grid(
            common_fov_um=310.0, preferred_size_um=300.0,
            minimum_size_um=260.0, coarsest_input_pixel_um=0.344174)
        self.assertEqual(result["status"], "PASS")
        self.assertGreaterEqual(result["pixel_um"], 0.344174)

    def test_below_minimum_stops(self):
        result = resolve_registered_grid(
            common_fov_um=205.0, preferred_size_um=300.0,
            minimum_size_um=260.0, coarsest_input_pixel_um=0.344174)
        self.assertEqual(result["status"], "STOP")


if __name__ == "__main__":
    unittest.main()
