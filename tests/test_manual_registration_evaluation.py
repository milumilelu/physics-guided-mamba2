"""Tests for the manual_v1 registration contract (plan §4 WP6).

Covers, in order of the plan's numbered list:

1. canonical (u,v) <-> raw (x,y) round trip;
2. manual edge midpoint equals the saved centre;
3. automatic theoretical four-edge signs and the y-down convention;
4. one-to-one merge with missing/duplicate hard failure;
5. paired slot order/separation gate;
6. hard failure when a single sample's centre source differs from the
   manual edges (sample-wise centre-source mixing is forbidden);
7. tagged (manual_v1) pipeline outputs can never overwrite the legacy
   archive;
8. the approval renderer refuses to write PASS;
9. manual-centre resampling and final leveling on a synthetic plane;
10. end-to-end smoke test on one real CAG sample (skipped when the raw
    data file is absent).
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

from src.data_contracts import HeightMap
from src.leveling import fit_outer_reference_plane
from src.manual_four_edge_annotation import canonical_box_record
from src.manual_registration_evaluation import (
    ManualRegistrationError,
    auto_theoretical_edges,
    band_label,
    check_paired_measurements,
    manual_center_from_edges,
    manual_v1_record,
    merge_one_to_one,
    paired_failure_keys,
    render_approval_text,
    resolve_pipeline_paths,
    uv_to_xy,
    validate_manual_identity,
    xy_to_uv,
)
from src.resampling import resample_center_crop, resample_to_canonical

REPO = Path(__file__).resolve().parents[1]
REAL_CAG = REPO / "氧化锆/pass实验数据/120正式.cag"


def annotation_row(**overrides) -> dict:
    """A synthetic but fully consistent annotation row (theta = -0.7 deg)."""
    base = canonical_box_record(
        left_local_um=-100.5, right_local_um=99.5,
        top_local_um=-101.0, bottom_local_um=101.5,
        display_center_x_um=12.0, display_center_y_um=3.0, theta_deg=-0.7)
    row = {
        "session_id": "zro2_120_formal",
        "measurement_id": "1",
        "sample_id": "1",
        "roi_within_measurement": "single",
        "annotator_a_state": "complete",
    }
    for key, value in base.items():
        row[f"annotator_a_{key}"] = value
    row.update(overrides)
    return row


GATE_CFG = {
    "observed_width_um": [180.0, 220.0],
    "observed_height_um": [180.0, 220.0],
    "require_all_complete": True,
    "allow_unusable": False,
    "require_paired_order": True,
    "minimum_paired_center_separation_um": 300.0,
}


class TestCanonicalTransforms(unittest.TestCase):
    """Plan item 1: (u,v) <-> (x,y) round trip at several angles."""

    def test_round_trip(self):
        for theta in (-45.0, -0.7, 0.0, 12.3, 44.9):
            for u, v in ((0.0, 0.0), (11.5, -3.25), (-140.0, 120.0)):
                x, y = uv_to_xy(u, v, theta)
                back_u, back_v = xy_to_uv(x, y, theta)
                self.assertAlmostEqual(back_u, u, places=9)
                self.assertAlmostEqual(back_v, v, places=9)

    def test_matches_annotation_convention(self):
        # uv_to_xy must reproduce the exact convention used by the frozen
        # annotation tool (src.manual_four_edge_annotation).
        row = annotation_row()
        x, y = uv_to_xy(row["annotator_a_center_u_um"],
                        row["annotator_a_center_v_um"], -0.7)
        self.assertAlmostEqual(x, row["annotator_a_center_x_um"], places=9)
        self.assertAlmostEqual(y, row["annotator_a_center_y_um"], places=9)


class TestManualEdgeMidpoint(unittest.TestCase):
    """Plan item 2: edge midpoint equals the saved centre."""

    def test_midpoint_is_center(self):
        row = annotation_row()
        identity = validate_manual_identity(row, -0.7)
        mid_u, mid_v = manual_center_from_edges(
            identity["left_u_um"], identity["right_u_um"],
            identity["top_v_um"], identity["bottom_v_um"])
        self.assertAlmostEqual(mid_u, identity["center_u_um"], places=9)
        self.assertAlmostEqual(mid_v, identity["center_v_um"], places=9)

    def test_identity_rejects_tampered_center(self):
        row = annotation_row()
        row["annotator_a_center_u_um"] = (
            float(row["annotator_a_center_u_um"]) + 5.0)
        with self.assertRaises(ManualRegistrationError):
            validate_manual_identity(row, -0.7)


class TestAutoTheoreticalEdges(unittest.TestCase):
    """Plan item 3: signs and y-down convention of the theoretical edges."""

    def test_edge_ordering(self):
        edges = auto_theoretical_edges(10.0, -4.0)
        self.assertLess(edges["left_u_um"], edges["right_u_um"])
        self.assertLess(edges["top_v_um"], edges["bottom_v_um"])
        self.assertAlmostEqual(edges["right_u_um"]-edges["left_u_um"], 200.0)
        self.assertAlmostEqual(edges["bottom_v_um"]-edges["top_v_um"], 200.0)

    def test_y_down_convention(self):
        # +v points down the image: the top edge is v - 100 and maps to a
        # raw y 100 um ABOVE the centre (smaller image y).
        center_u, center_v = 0.0, 0.0
        edges = auto_theoretical_edges(center_u, center_v)
        top_x, top_y = uv_to_xy(center_u, edges["top_v_um"], 0.0)
        bottom_x, bottom_y = uv_to_xy(center_u, edges["bottom_v_um"], 0.0)
        self.assertLess(top_y, bottom_y)
        self.assertAlmostEqual(top_y, -100.0)
        self.assertAlmostEqual(bottom_y, 100.0)
        # and at a non-zero session angle the same holds in the rotated frame
        theta = -0.7
        self.assertLess(
            xy_to_uv(*uv_to_xy(center_u, edges["top_v_um"], theta), theta)[1],
            xy_to_uv(*uv_to_xy(center_u, edges["bottom_v_um"], theta),
                     theta)[1])

    def test_band_labels(self):
        self.assertEqual(band_label(1.9, close_um=2.0, moderate_um=5.0),
                         "close")
        self.assertEqual(band_label(2.0, close_um=2.0, moderate_um=5.0),
                         "close")
        self.assertEqual(band_label(4.9, close_um=2.0, moderate_um=5.0),
                         "moderate")
        self.assertEqual(band_label(5.1, close_um=2.0, moderate_um=5.0),
                         "large")


class TestOneToOneMerge(unittest.TestCase):
    """Plan item 4: missing and duplicate keys hard-fail."""

    def setUp(self):
        self.manual = [{"session_id": "s", "sample_id": str(i)}
                       for i in (1, 2, 3)]
        self.auto = [{"session_id": "s", "sample_id": str(i),
                      "center_x_um": 0.0, "center_y_um": 0.0,
                      "status": "PASS"} for i in (1, 2, 3)]

    def test_clean_merge(self):
        pairs = merge_one_to_one(self.manual, self.auto)
        self.assertEqual(len(pairs), 3)

    def test_missing_sample_fails(self):
        self.auto = self.auto[:2]
        with self.assertRaises(ManualRegistrationError):
            merge_one_to_one(self.manual, self.auto)

    def test_duplicate_sample_fails(self):
        self.auto.append(dict(self.auto[0]))
        with self.assertRaises(ManualRegistrationError):
            merge_one_to_one(self.manual, self.auto)


class TestPairedGate(unittest.TestCase):
    """Plan item 5: paired slot order and separation gate."""

    @staticmethod
    def pair_rows(du: float, dv: float = 0.0) -> list[dict]:
        left = annotation_row(sample_id="1", roi_within_measurement="slot_1")
        right = annotation_row(sample_id="2",
                               roi_within_measurement="slot_2")
        for field in ("center_u_um", "center_x_um"):
            right[f"annotator_a_{field}"] = (
                float(left[f"annotator_a_{field}"]) + du)
        for field in ("center_v_um", "center_y_um"):
            right[f"annotator_a_{field}"] = (
                float(left[f"annotator_a_{field}"]) + dv)
        return [left, right]

    def test_good_pair_passes(self):
        self.assertEqual(check_paired_measurements(self.pair_rows(400.0)), [])

    def test_wrong_order_fails(self):
        issues = check_paired_measurements(self.pair_rows(-400.0))
        self.assertTrue(issues and "not right of slot_1" in issues[0])

    def test_insufficient_separation_fails(self):
        issues = check_paired_measurements(self.pair_rows(250.0))
        self.assertTrue(any("separation" in issue for issue in issues))

    def test_incomplete_pair_fails(self):
        issues = check_paired_measurements(self.pair_rows(400.0)[:1])
        self.assertTrue(any("incomplete pair" in issue for issue in issues))

    def test_failure_keys_are_structured(self):
        """Row-level flagging must use keys, not parsed issue strings."""
        self.assertEqual(paired_failure_keys(self.pair_rows(400.0)), set())
        self.assertEqual(
            paired_failure_keys(self.pair_rows(250.0)),
            {("zro2_120_formal", 1)})

    def test_duplicate_slot_raises_no_unboundlocal(self):
        """A duplicated slot must be a controlled data-contract issue.

        Regression: ``issues`` used to be initialised *after* the grouping
        loop, so a duplicate slot raised ``UnboundLocalError`` instead of
        reporting the offending pair.
        """
        duplicate = annotation_row(sample_id="3",
                                   roi_within_measurement="slot_1")
        issues = check_paired_measurements(self.pair_rows(400.0) + [duplicate])
        self.assertTrue(any("duplicate slot_1 annotation" in issue
                            for issue in issues),
                        f"unexpected issues: {issues}")


class TestNoSamplewiseCenterMixing(unittest.TestCase):
    """Plan item 6: a centre not derived from the manual edges hard-fails."""

    def test_foreign_center_rejected(self):
        row = annotation_row()
        row["annotator_a_center_u_um"] = 42.0  # centre from elsewhere
        with self.assertRaises(ManualRegistrationError):
            manual_v1_record(
                row, theta_deg=-0.7, d4_transform="identity",
                source_sha256="X"*64, config_sha256="Y"*64,
                gate_cfg=GATE_CFG, fov_width_um=700.0, fov_height_um=530.0,
                paired_gate_ok=True)

    def test_consistent_row_passes_gates(self):
        row = annotation_row()
        record = manual_v1_record(
            row, theta_deg=-0.7, d4_transform="identity",
            source_sha256="X"*64, config_sha256="Y"*64,
            gate_cfg=GATE_CFG, fov_width_um=700.0, fov_height_um=530.0,
            paired_gate_ok=True)
        self.assertEqual(record["status"], "PASS")
        self.assertEqual(record["registration_method"],
                         "manual_four_edge_a_v1")
        self.assertEqual(record["evidence_level"], 3)

    def test_out_of_gate_size_stops(self):
        row = annotation_row()
        record = manual_v1_record(
            row, theta_deg=-0.7, d4_transform="identity",
            source_sha256="X"*64, config_sha256="Y"*64,
            gate_cfg=GATE_CFG, fov_width_um=700.0, fov_height_um=530.0,
            paired_gate_ok=True)
        self.assertEqual(record["status"], "PASS")
        narrow = annotation_row()
        # shrink the box to ~150 um -> outside the frozen gate
        left = float(narrow["annotator_a_left_u_um"])
        right = float(narrow["annotator_a_right_u_um"])
        narrow["annotator_a_right_u_um"] = left + 0.75*(right-left)
        narrow["annotator_a_width_um"] = (
            float(narrow["annotator_a_right_u_um"])
            - float(narrow["annotator_a_left_u_um"]))
        narrow["annotator_a_center_u_um"] = (
            float(narrow["annotator_a_left_u_um"])
            + float(narrow["annotator_a_right_u_um"])) / 2.0
        x, y = uv_to_xy(float(narrow["annotator_a_center_u_um"]),
                        float(narrow["annotator_a_center_v_um"]), -0.7)
        narrow["annotator_a_center_x_um"] = x
        narrow["annotator_a_center_y_um"] = y
        record = manual_v1_record(
            narrow, theta_deg=-0.7, d4_transform="identity",
            source_sha256="X"*64, config_sha256="Y"*64,
            gate_cfg=GATE_CFG, fov_width_um=700.0, fov_height_um=530.0,
            paired_gate_ok=True)
        self.assertEqual(record["status"], "STOP")


class TestTaggedOutputsNeverOverwrite(unittest.TestCase):
    """Plan item 7: manual_v1 outputs live only under the tag directory."""

    def setUp(self):
        self.root = Path("outputs/rectangle_registration")

    def test_tagged_paths_are_disjoint_from_legacy(self):
        legacy = resolve_pipeline_paths(self.root, None)
        tagged = resolve_pipeline_paths(self.root, "manual_v1")
        legacy_paths = {legacy.resampling_dir, legacy.registered_h_reg_dir,
                        legacy.registered_h_200_dir,
                        legacy.registered_masks_dir, legacy.metrics_dir,
                        legacy.registration_metrics_csv,
                        legacy.common_fov_summary_json,
                        legacy.resampling_summary_json}
        tagged_paths = {tagged.resampling_dir, tagged.registered_h_reg_dir,
                        tagged.registered_h_200_dir,
                        tagged.registered_masks_dir, tagged.metrics_dir,
                        tagged.registration_metrics_csv,
                        tagged.common_fov_summary_json,
                        tagged.resampling_summary_json}
        self.assertFalse(legacy_paths & tagged_paths)
        for path in tagged_paths:
            self.assertIn(str(self.root / "manual_v1"), str(path))

    def test_default_paths_are_the_legacy_locations(self):
        paths = resolve_pipeline_paths(self.root, None)
        self.assertEqual(paths.registration_metrics_csv,
                         self.root / "metrics/registration_metrics.csv")
        self.assertEqual(paths.common_fov_summary_json,
                         self.root / "resampling/common_fov_summary.json")


class TestApprovalCannotPass(unittest.TestCase):
    """Plan item 8: the approval renderer refuses to write PASS."""

    def test_pass_is_refused(self):
        with self.assertRaises(ManualRegistrationError):
            render_approval_text(status="PASS", decision="X", body_lines=[])

    def test_pending_and_blocked_allowed(self):
        for status in ("PENDING", "BLOCKED"):
            text = render_approval_text(
                status=status, decision="AWAITING_REVIEW", body_lines=["a"])
            self.assertIn(f"Status: {status}", text)
            self.assertNotIn("Status: PASS", text)
            self.assertIn("forbidden from marking it PASS", text)


class TestSyntheticPlaneResampling(unittest.TestCase):
    """Plan item 9: manual-centre resampling + final leveling on a plane."""

    def test_flat_after_final_leveling(self):
        # 700 x 530 um raw map, tilted plane, no depression: after
        # coarse levelling, resampling at a manual centre and final
        # outer-reference leveling the H_200 must be ~flat everywhere.
        pixels = 400
        axis = (np.arange(pixels, dtype=float)+0.5)*1.75-350.0
        xx, yy = np.meshgrid(axis, axis*530.0/700.0)
        plane = 0.002*xx + 0.001*yy + 5.0
        noise = np.random.default_rng(42).normal(0.0, 0.002, plane.shape)
        raw = HeightMap(z=plane+noise, valid_mask=np.ones_like(plane, bool),
                        dx_um=1.75, dy_um=1.75, x_um=axis+350.0,
                        y_um=axis*530.0/700.0+265.0, metadata={})
        coarse = resample_to_canonical(
            raw, plane=(0.002, 0.001, 5.0), center_x_um=12.0,
            center_y_um=3.0, theta_deg=-0.7, length_um=300.0, pixels=871,
            minimum_mask_weight=0.99, order=1)
        frame_width = coarse.width_um/2.0-120.0
        fit = fit_outer_reference_plane(
            coarse, frame_width_um=frame_width, sigma_low=2.0,
            sigma_high=3.0, max_iterations=10,
            minimum_reference_valid_fraction=0.20)
        self.assertEqual(fit.status, "PASS")
        x = coarse.x_um-(coarse.x_um[0]+coarse.x_um[-1])/2
        y = coarse.y_um-(coarse.y_um[0]+coarse.y_um[-1])/2
        final_z = coarse.z-(fit.a*x[None, :]+fit.b*y[:, None]+fit.c)
        self.assertLess(float(np.nanstd(final_z)), 0.01)
        h200 = resample_center_crop(
            HeightMap(z=np.where(coarse.valid_mask, final_z, np.nan),
                      valid_mask=coarse.valid_mask, dx_um=coarse.dx_um,
                      dy_um=coarse.dy_um, x_um=coarse.x_um, y_um=coarse.y_um,
                      metadata={}),
            length_um=200.0, pixels=580, minimum_mask_weight=0.99)
        self.assertEqual(h200.valid_fraction, 1.0)
        self.assertLess(float(np.nanstd(h200.z)), 0.01)


@unittest.skipUnless(REAL_CAG.exists(), "real CAG data not available")
class TestRealCagEndToEndSmoke(unittest.TestCase):
    """Plan item 10: one real sample through the manual_v1 path."""

    def test_sample_1_register_and_level(self):
        from src.io_cag import CagHeightReader
        import csv as _csv
        planes_path = (REPO / "outputs/rectangle_registration/metrics"
                       / "coarse_leveling_metrics.csv")
        with planes_path.open("r", encoding="utf-8-sig") as stream:
            plane = next(row for row in _csv.DictReader(stream)
                         if row["session_id"] == "zro2_120_formal"
                         and row["measurement_id"] == "1")
        # frozen manual centre of zro2_120_formal sample 1 (theta -0.7)
        with CagHeightReader(REAL_CAG) as reader:
            raw = reader.read_height_map(1)
        coarse = resample_to_canonical(
            raw, plane=tuple(float(plane[k]) for k in ("a", "b", "c")),
            center_x_um=11.047183507313195, center_y_um=3.3496569435690215,
            theta_deg=-0.7000000000025182, length_um=300.0, pixels=871,
            minimum_mask_weight=0.99, order=1)
        self.assertGreater(coarse.valid_fraction, 0.99)
        fit = fit_outer_reference_plane(
            coarse, frame_width_um=300.0/2.0-120.0, sigma_low=2.0,
            sigma_high=3.0, max_iterations=10,
            minimum_reference_valid_fraction=0.20)
        self.assertEqual(fit.status, "PASS")
        h200 = resample_center_crop(
            coarse, length_um=200.0, pixels=580, minimum_mask_weight=0.99)
        self.assertEqual(h200.valid_fraction, 1.0)
        self.assertTrue(np.isfinite(h200.z[h200.valid_mask]).all())


if __name__ == "__main__":
    unittest.main()
