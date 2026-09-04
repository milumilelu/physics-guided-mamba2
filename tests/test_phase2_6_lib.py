"""Phase 2.6 unit tests (synthetic + frozen repo inputs; CI-safe).

Covers the frozen §4 geometry/width implementation on synthetic profiles, the
frozen design-table grid, the frozen in-box number (101/200), the block
hatch shuffle, and the lambda*/peak descriptors.  Formal-input-dependent
tests (manifest, geometry CSVs, bridge) anchor via SkipTest until the
corresponding Task outputs exist.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "phase1_5"))
sys.path.insert(0, str(REPO / "experiments" / "phase2"))

_spec = importlib.util.spec_from_file_location(
    "phase2_6_lib", REPO / "experiments" / "phase2_6" / "_lib.py")
p26 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p26)


def _cfg():
    import yaml
    with open(REPO / "experiments" / "phase2_6" / "phase2_6_config.yaml",
              encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _groove(v: np.ndarray, depth: float = 18.0, sigma: float = 2.2,
            ridge: float = 0.4) -> np.ndarray:
    """Synthetic groove with two side uplifts (analytic check fixture)."""
    return (depth * np.exp(-(v ** 2) / (2 * sigma ** 2))
            - ridge * np.exp(-((v - 6.5) ** 2) / (2 * 0.5 ** 2))
            - ridge * np.exp(-((v + 6.5) ** 2) / (2 * 0.5 ** 2)))


class AxisFrameTests(unittest.TestCase):
    def test_axis_frame_orthonormal(self):
        for theta in (0.0, 30.0, -77.0, 90.0):
            t_hat, n_hat = p26.axis_frame(theta)
            p26.require(abs(np.hypot(*t_hat) - 1) < 1e-12, "t_hat not unit")
            p26.require(abs(np.hypot(*n_hat) - 1) < 1e-12, "n_hat not unit")
            p26.require(abs(float(np.dot(t_hat, n_hat))) < 1e-12, "not orthogonal")
        t0, n0 = p26.axis_frame(0.0)
        self.assertTrue(np.allclose(t0, [1, 0]) and np.allclose(n0, [0, 1]))
        t90, n90 = p26.axis_frame(90.0)
        self.assertTrue(np.allclose(t90, [0, 1]) and np.allclose(n90, [-1, 0]))

    def test_lateral_positions_match_row_centers(self):
        positions = p26.lateral_positions(4, 2.0)
        self.assertTrue(np.allclose(positions, [-3.0, -1.0, 1.0, 3.0]))


class WidthTests(unittest.TestCase):
    def test_section_features_analytic_groove(self):
        v = np.arange(-8.0, 8.0 + 1e-9, 0.5)
        profile = _groove(v)
        features = p26.section_features(profile, v, (0.2, 0.5, 0.8),
                                        affected_delta_um=0.10)
        p26.require(features["W20_um"] >= features["W50_um"]
                    >= features["W80_um"], "W20 >= W50 >= W80 violated")
        analytic = {"W20": 2 * 2.2 * np.sqrt(2 * np.log(5.0)),
                    "W50": 2 * 2.2 * np.sqrt(2 * np.log(2.0)),
                    "W80": 2 * 2.2 * np.sqrt(2 * np.log(1.25))}
        for key, value in analytic.items():
            p26.require(abs(features[f"{key}_um"] - value) / value < 0.02,
                        f"{key} {features[f'{key}_um']} vs analytic {value}")
        weq_analytic = 2.2 * np.sqrt(2 * np.pi)
        p26.require(abs(features["W_eq_um"] - weq_analytic) / weq_analytic < 0.02,
                    f"W_eq {features['W_eq_um']} vs analytic {weq_analytic}")
        p26.require(abs(features["D_max_um"] - 18.0) < 0.26, "D_max off")
        p26.require(features["n_runs_W50"] == 1, "single groove must give 1 run")
        self.assertFalse(features["censored_W50"])
        p26.require(features["ridge_left_um"] > 0.05
                    and features["ridge_right_um"] > 0.05,
                    "side uplifts must register as ridges (net of groove tail)")
        p26.require(abs(features["edge_asymmetry"]) < 1e-6,
                    "symmetric groove must give zero asymmetry")

    def test_section_features_truncated_profile_is_censored(self):
        # window [−3, 3] clips the 20%-level extent (analytic ±3.69) but not
        # the 50%/80%-level extents -> only W20 may report censoring
        v = np.arange(-3.0, 3.0 + 1e-9, 0.5)
        profile = _groove(v)
        features = p26.section_features(profile, v, (0.2, 0.5, 0.8),
                                        affected_delta_um=0.10)
        p26.require(bool(features["censored_W20"]),
                    "W20 run touching profile ends must be censored")
        self.assertFalse(bool(features["censored_W50"]))
        self.assertFalse(bool(features["censored_W80"]))

    def test_section_features_nan_profile(self):
        features = p26.section_features(np.full(64, np.nan),
                                        p26.lateral_positions(64, 0.278657),
                                        (0.2, 0.5, 0.8), affected_delta_um=0.1)
        self.assertTrue(np.isnan(features["W50_um"]))
        self.assertTrue(np.isnan(features["W_eq_um"]))

    def test_aggregate_identifiability_states(self):
        cfg = _cfg()["single_line"]
        base = pd.DataFrame({"n_above_threshold": [10] * 25, "W50_um": np.full(25, 12.0),
                             "censored_W50": [False] * 25, "W20_um": np.full(25, 15.0),
                             "censored_W20": [False] * 25, "W80_um": np.full(25, 8.0),
                             "censored_W80": [False] * 25, "W_eq_um": np.full(25, 5.5),
                             "D_max_um": np.full(25, 18.0)})
        row = p26.aggregate_line(base, min_sections=20,
                                 censored_frac_limit=cfg["censored_frac_W50_uncertain_above"])
        self.assertEqual(row["width_identifiability"], "estimable")
        censored = base.copy()
        censored.loc[:14, "censored_W50"] = True
        row = p26.aggregate_line(censored, min_sections=20,
                                 censored_frac_limit=cfg["censored_frac_W50_uncertain_above"])
        self.assertEqual(row["width_identifiability"], "right_censored")
        row = p26.aggregate_line(base.iloc[:10], min_sections=20,
                                 censored_frac_limit=cfg["censored_frac_W50_uncertain_above"])
        self.assertEqual(row["width_identifiability"], "insufficient_sections")
        p26.require(row["median_W50_um"] == 12.0, "median from uncensored sections")

    def test_censored_sections_excluded_from_median(self):
        cfg = _cfg()["single_line"]
        values = np.concatenate([np.full(5, 20.0), np.full(20, 12.0)])
        censor = np.array([True] * 5 + [False] * 20)
        frame = pd.DataFrame({"n_above_threshold": [10] * 25, "W50_um": values,
                              "censored_W50": censor, "W20_um": values + 3,
                              "censored_W20": censor, "W80_um": values - 4,
                              "censored_W80": censor, "W_eq_um": np.full(25, 5.5),
                              "D_max_um": np.full(25, 18.0)})
        row = p26.aggregate_line(frame, min_sections=20,
                                 censored_frac_limit=cfg["censored_frac_W50_uncertain_above"])
        p26.require(abs(row["median_W50_um"] - 12.0) < 1e-12,
                    "censored sections must not enter the line median")


class ExtentTests(unittest.TestCase):
    def test_extent_and_stable_region(self):
        s_scan = np.arange(-140.0, 140.0 + 1e-9, 1.0)
        online = (np.abs(s_scan) <= 95.0)
        start, end = p26.line_extent(s_scan, online, min_run_um=3.0,
                                     merge_gap_um=5.0)
        p26.require(abs(start + 95.0) < 1e-9 and abs(end - 95.0) < 1e-9,
                    f"extent {start}..{end} != -95..95")
        stable = p26.stable_region(start, end, pad_low=0.15, pad_high=0.15)
        p26.require(abs((stable[0] - start) - 0.15 * (end - start)) < 1e-9,
                    "stable region pad_low wrong")
        sections = p26.section_positions(stable, 2.0)
        p26.require(len(sections) >= 20, "central 70% of 190um must give >= 20 sections at 2um")
        p26.require(sections.min() >= stable[0] - 1e-9
                    and sections.max() <= stable[1] + 1e-9, "sections outside stable")

    def test_isolated_detections_discarded(self):
        s_scan = np.arange(-50.0, 50.0 + 1e-9, 1.0)
        online = (np.abs(s_scan) <= 30.0) | (np.abs(s_scan - 45.0) < 0.5)
        start, end = p26.line_extent(s_scan, online, min_run_um=3.0,
                                     merge_gap_um=5.0)
        p26.require(abs(start + 30.0) < 1e-9 and abs(end - 30.0) < 1e-9,
                    "isolated far detection must be discarded")


class LambdaStarTests(unittest.TestCase):
    @staticmethod
    def _long(values: list[tuple[float, float, float]]) -> pd.DataFrame:
        rows = []
        for dataset_index, triple in enumerate(values):
            for lam, energy, modes in triple:
                rows.append({"dataset_index": dataset_index, "bin": len(rows),
                             "lambda_geo_um": lam, "energy": energy,
                             "n_modes": modes})
        return pd.DataFrame(rows)

    def test_lambda_star_window_and_guard(self):
        lam = np.geomspace(0.7, 160.0, 24)
        energy = np.exp(-((np.log(lam) - np.log(12.0)) ** 2) / (2 * 0.3 ** 2))
        energy /= energy.sum()
        good = self._long([(tuple(zip(lam, energy, [30] * 24)))])
        frame = p26.lambda_star_4_32(good, window_um=(4.0, 32.0), guard=0.10)
        p26.require(bool(frame.loc[0, "lambda_star_valid"]), "peak-in-window must be valid")
        p26.require(abs(frame.loc[0, "lambda_star_4_32_um"] - 12.0) / 12.0 < 0.05,
                    "lambda* must sit near the energy pile-up")
        energy_far = np.exp(-((np.log(lam) - np.log(90.0)) ** 2) / (2 * 0.3 ** 2))
        energy_far /= energy_far.sum()
        bad = self._long([(tuple(zip(lam, energy_far, [30] * 24)))])
        frame = p26.lambda_star_4_32(bad, window_um=(4.0, 32.0), guard=0.10)
        p26.require(not bool(frame.loc[0, "lambda_star_valid"]),
                    "window energy below guard must give NA")
        self.assertTrue(np.isnan(frame.loc[0, "lambda_star_4_32_um"]))

    def test_peak_gates(self):
        lam = np.geomspace(0.7, 160.0, 24)
        inside = (lam >= 4.0) & (lam < 32.0)
        energy = np.where(inside, 0.0, 1.0)
        peak_index = int(np.flatnonzero(inside)[np.argmax(
            np.abs(np.log(lam[inside]) - np.log(12.0)))])
        energy[peak_index] = 10.0
        energy /= energy.sum()
        modes = np.where(np.arange(24) == peak_index, 30, 30)
        good = self._long([(tuple(zip(lam, energy, modes)))])
        frame = p26.lambda_peak_4_32(good, window_um=(4.0, 32.0),
                                     n_modes_min=20, share_min=0.20)
        p26.require(bool(frame.loc[0, "lambda_peak_valid"]),
                    "dominant in-window peak with modes>=30 must be valid")
        modes_low = np.where(np.arange(24) == peak_index, 10, 30)
        low = self._long([(tuple(zip(lam, energy, modes_low)))])
        frame = p26.lambda_peak_4_32(low, window_um=(4.0, 32.0),
                                     n_modes_min=20, share_min=0.20)
        p26.require(not bool(frame.loc[0, "lambda_peak_valid"]),
                    "low-mode peak bin must be dropped (no forced peak)")


class FrozenInputTests(unittest.TestCase):
    def test_design_table_grid_frozen(self):
        design = pd.read_csv(REPO / "氧化锆" / "氧化锆_line_design.csv",
                             encoding="gb18030")
        p26.require(len(design) == 120, "design rows != 120")
        p26.require(design["加工顺序"].tolist() == list(range(1, 121)),
                    "加工顺序 != 1..120")
        for column, expected in (("脉宽_fs", (223, 500, 1000, 2000, 4000)),
                                 ("频率_kHz", (2, 5, 10, 20, 40)),
                                 ("速度_mm/s", (5, 10, 15, 20, 25)),
                                 ("重复扫描次数", (1, 2, 3, 4, 5))):
            p26.require(sorted(design[column].unique().tolist()) == list(expected),
                        f"{column} grid drifted")
        p26.require(not design.duplicated().any(), "design rows must be unique")

    def test_cag_pitch_and_field_frozen(self):
        cag = REPO / "氧化锆" / "120组直线.cag"
        if not cag.exists():
            raise unittest.SkipTest("120组直线.cag not present")
        from src.io_cag import CagHeightReader
        reader = CagHeightReader(cag)
        try:
            hm = reader.read_height_map(13)
        finally:
            reader.close()
        p26.require(hm.shape == (64, 1024), f"shape {hm.shape} != (64, 1024)")
        p26.require(abs(hm.dx_um - 0.278657) < 1e-9
                    and abs(hm.dy_um - 0.278657) < 1e-9, "pixel pitch drifted")
        p26.require(float(np.mean(hm.valid_mask)) == 1.0, "valid ratio != 1.0")

    def test_in_box_coverage_frozen_101(self):
        manifest_path = REPO / "outputs" / "phase2" / "manifest" / "phase2_manifest.csv"
        if not manifest_path.exists():
            raise unittest.SkipTest("phase2 manifest not present")
        manifest = pd.read_csv(manifest_path)
        mask = p26.in_box_mask(manifest, _cfg()["bridge"]["box"])
        p26.require(int(mask.sum()) == 101,
                    f"in-box coverage {int(mask.sum())} != frozen 101/200")

    def test_shuffle_block_structure(self):
        manifest_path = REPO / "outputs" / "phase2" / "manifest" / "phase2_manifest.csv"
        if not manifest_path.exists():
            raise unittest.SkipTest("phase2 manifest not present")
        manifest = pd.read_csv(manifest_path)
        shuffled = p26.shuffle_h_by_block(
            manifest, unit_columns=("session_id", "base_condition_group"),
            seed=20260904)
        unit = ["session_id", "base_condition_group"]
        p26.require(shuffled.groupby([manifest["session_id"],
                                      manifest["base_condition_group"]])
                    .nunique().le(1).all(),
                    "shuffled hatch must stay constant within a unit")
        for session, block in manifest.groupby("session_id"):
            original = (block.groupby(unit[1])["hatch_spacing_um"].first()
                        .value_counts().sort_index())
            after = (shuffled[block.index].groupby(
                manifest.loc[block.index, "base_condition_group"])
                .first().value_counts().sort_index())
            p26.require(bool(original.equals(after)),
                        f"{session}: unit-level h multiset must be preserved")
        again = p26.shuffle_h_by_block(
            manifest, unit_columns=("session_id", "base_condition_group"),
            seed=20260904)
        p26.require(bool(np.allclose(shuffled.to_numpy(dtype=float),
                                     again.to_numpy(dtype=float))),
                    "shuffled-h null must be deterministic under a fixed seed")

    def test_manifest_exists(self):
        path = REPO / "outputs" / "phase2_6" / "single_line" / "single_line_manifest.csv"
        if not path.exists():
            raise unittest.SkipTest("Task 15 output not present yet")
        frame = pd.read_csv(path, encoding="utf-8-sig")
        p26.require(len(frame) == 120 and frame["single_line_id"].is_unique,
                    "manifest must hold 120 unique rows")
        required = ["pulse_duration_fs", "frequency_kHz", "velocity_mm_s",
                    "pass_count", "power_W_or_proxy", "pixel_size_um",
                    "line_scan_direction", "measurement_orientation_theta_deg",
                    "processing_date_or_batch", "height_data_type",
                    "background_correction_status", "valid_mask_status",
                    "hatch_spacing_um", "mapping_provenance", "exclusion_note"]
        p26.require(all(column in frame.columns for column in required),
                    "manifest lacks required fields")
        p26.require(frame["hatch_spacing_um"].isna().all(),
                    "single-line hatch must be NA")

    def test_geometry_outputs_states(self):
        path = REPO / "outputs" / "phase2_6" / "single_line" / "single_line_geometry.csv"
        if not path.exists():
            raise unittest.SkipTest("Task 16 output not present yet")
        geometry = pd.read_csv(path, encoding="utf-8-sig")
        p26.require(geometry["width_identifiability"]
                    .isin(["estimable", "right_censored",
                           "insufficient_sections"]).all(),
                    "width_identifiability states drifted")
        p26.require((geometry["median_W20_um"]
                     >= geometry["median_W50_um"]).all()
                    and (geometry["median_W50_um"]
                         >= geometry["median_W80_um"]).all(),
                    "median W ordering W20 >= W50 >= W80 violated")
        p26.require((geometry["median_W_eq_um"] > 0).all(), "W_eq must be > 0")
        p26.require((geometry["n_sections_used"]
                     >= _cfg()["single_line"]["min_sections"]).all(),
                    "every line must hold >= 20 sections")


if __name__ == "__main__":
    unittest.main()
