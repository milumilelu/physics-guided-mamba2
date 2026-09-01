"""CAG decoding: sentinels, rounding, and the LUT offset.

Tests that need a real container are skipped when the data is not present, but
the end-to-end acceptance run must execute them on the real files.
"""

from __future__ import annotations

import struct
import unittest
from pathlib import Path

import numpy as np

from src.data_contracts import HeightMap
from src.io_cag import (INVALID_SENTINEL, LUT_BYTES, VK4_KEYS,
                        CagHeightReader, derive_lut_bytes,
                        raw_to_micrometres, score_lut_candidate,
                        verify_lut_offset)

REPO = Path(__file__).resolve().parent.parent
REAL_CAG = REPO / "氧化锆" / "pass实验数据" / "20补充pass.cag"


class TestRounding(unittest.TestCase):
    """KEYENCE rounds half away from zero; numpy and round() do not."""

    def test_half_up_at_three_decimals(self):
        # 30.0405 um -> 30.041, not the banker's 30.040
        self.assertEqual(raw_to_micrometres(np.array([300405]), 100)[0], 30.041)
        self.assertEqual(raw_to_micrometres(np.array([300415]), 100)[0], 30.042)

    def test_exact_values_are_untouched(self):
        got = raw_to_micrometres(np.array([0, 1, 309467, 1_000_000]), 100)
        np.testing.assert_allclose(got, [0.0, 0.0, 30.947, 100.0], atol=1e-9)

    def test_banker_rounding_would_differ(self):
        """Documents the trap: np.round gives 30.040 where KEYENCE gives 30.041."""
        raw = np.array([300405])
        keyence = raw_to_micrometres(raw, 100)[0]
        naive = np.round(raw * 100 * 1e-6, 3)[0]
        self.assertEqual(keyence, 30.041)
        self.assertEqual(naive, 30.040)
        self.assertNotEqual(keyence, naive)

    def test_sentinel_becomes_nan(self):
        raw = np.array([100, INVALID_SENTINEL, 0xFF123456, 200])
        invalid = raw >= INVALID_SENTINEL
        z = raw_to_micrometres(raw, 100, invalid)
        self.assertTrue(np.isnan(z[1]))
        self.assertTrue(np.isnan(z[2]))
        self.assertFalse(np.isnan(z[0]))
        self.assertEqual(z[3], 0.02)

    def test_sentinel_never_enters_statistics(self):
        raw = np.array([100, INVALID_SENTINEL, 200])
        invalid = raw >= INVALID_SENTINEL
        z = raw_to_micrometres(raw, 100, invalid)
        self.assertEqual(np.nanmin(z), 0.01)
        self.assertEqual(np.nanmax(z), 0.02)


class TestLutVerification(unittest.TestCase):
    """The 776-byte offset must be a derived result, not a comment."""

    def _fake_blob(self, width=64, height=48, z_step_pm=100, lut=LUT_BYTES,
                   trailing_section=True):
        """Synthesise a VK4 blob whose height section holds a smooth surface.

        The surface is a ramp across x plus two smooth terms in y.  The ramp
        matters: a surface that is periodic in x wraps seamlessly, so a
        misaligned read would leave no trace and the test would prove nothing.
        """
        ii = np.linspace(0.0, 1.0, height, dtype=np.float64)[:, None]
        jj = np.linspace(0.0, 1.0, width, dtype=np.float64)[None, :]
        surface = (25.0 + 20.0 * jj
                   + 6.0 * np.sin(2 * np.pi * ii)
                   + 3.0 * np.cos(4 * np.pi * ii))
        raw = (surface * 1e4 / z_step_pm).astype("<u4")
        payload = raw.tobytes()

        # Layout must match what CagHeightReader.read_raw expects:
        #   0..12   "VK4_" + 8 bytes
        #   12..84  18-entry offset table
        #   84..    meas_conds, date at +4, XYZ pitches at +42*4
        #   then the height section: 20-byte header, palette, samples
        #   then a trailing section, so the palette can be derived as the gap
        blob = bytearray(b"VK4_" + b"\x00" * 8)
        blob += b"\x00" * (18 * 4)
        meas_conds = len(blob)
        blob += b"\x00" * 4
        blob += struct.pack("<6I", 2026, 5, 28, 10, 56, 18)
        blob += b"\x00" * (42 * 4 - 4 - 24)
        blob += struct.pack("<3I", 344174, 344174, z_step_pm)
        height_offset = len(blob)
        blob += struct.pack("<5I", width, height, 32, 842220365, len(payload))
        blob += b"\xff" * lut
        blob += payload
        offsets = [0] * 18
        offsets[0] = meas_conds
        offsets[6] = height_offset
        if trailing_section:
            offsets[9] = len(blob)          # color_peak_thumb
            blob += b"\x00" * 64
        blob[12:12 + 18 * 4] = struct.pack("<18I", *offsets)
        return bytes(blob), height_offset, z_step_pm

    # ---- primary: the palette length is derived, not assumed ------------- #
    def test_palette_is_derived_from_the_section_table(self):
        blob, ho, _ = self._fake_blob()
        got = derive_lut_bytes(blob, ho, expected=LUT_BYTES)
        self.assertTrue(got["ok"], msg=str(got))
        self.assertEqual(got["lut_bytes"], LUT_BYTES)
        self.assertEqual(got["gap_bytes"], LUT_BYTES)
        self.assertEqual(got["next_section_offset"],
                         ho + 20 + got["data_bytes"] + LUT_BYTES)

    def test_derivation_tracks_a_different_palette_length(self):
        """The derivation must measure the gap, not agree with a constant."""
        blob, ho, _ = self._fake_blob(lut=600)
        self.assertEqual(derive_lut_bytes(blob, ho, expected=None)["lut_bytes"],
                         600)
        mismatched = derive_lut_bytes(blob, ho, expected=LUT_BYTES)
        self.assertFalse(mismatched["ok"])
        self.assertIn("!= expected", mismatched["reason"])

    def test_derivation_falls_back_to_the_blob_end(self):
        """Without a trailing section the gap is still well defined."""
        blob, ho, _ = self._fake_blob(trailing_section=False)
        got = derive_lut_bytes(blob, ho, expected=LUT_BYTES)
        self.assertTrue(got["ok"], msg=str(got))
        self.assertTrue(got["next_section_at_blob_end"])

    def test_implausible_gap_is_rejected(self):
        blob, ho, _ = self._fake_blob(trailing_section=False)
        got = derive_lut_bytes(blob, ho, expected=None, max_plausible=8)
        self.assertFalse(got["ok"])
        self.assertIn("plausibility", got["reason"])

    # ---- corroborating: morphology agrees and its limits are reported ---- #
    def test_correct_offset_has_no_seam(self):
        blob, ho, z_step = self._fake_blob()
        result = verify_lut_offset(blob, ho, z_step, LUT_BYTES)
        self.assertTrue(result["passed"], msg=str(result))
        self.assertEqual(result["best_lut"], LUT_BYTES)
        self.assertLess(result["seam_ratio"], 2.0)
        self.assertTrue(result["seam_conclusive_on_this_sample"])
        self.assertGreater(result["seam_headroom"], 10.0)

    def test_zero_offset_is_rejected(self):
        blob, ho, z_step = self._fake_blob()
        result = verify_lut_offset(blob, ho, z_step, 0)
        self.assertFalse(result["passed"], msg=str(result))
        self.assertFalse(result["structural"]["ok"])

    def test_headroom_is_reported_when_the_field_edges_are_level(self):
        """A flat field cannot be verified by seam strength, and says so.

        Real machined samples look like this: the rectangle sits in the middle
        and both edges of the field are untouched surface at the same height,
        so the wrap jump is no larger than an ordinary machining step.  The
        ranking still picks the right offset but the check is not conclusive,
        and the report has to say that instead of implying a safe margin.
        """
        blob, ho, z_step = self._fake_blob()
        flat = bytearray(blob)
        width, height = 64, 48
        start = ho + 20 + LUT_BYTES
        rng = np.random.default_rng(7)
        plateau = (30_000 + rng.integers(-25, 26, size=(height, width))
                   ).astype("<u4")
        plateau[:, 24:40] += 1_500          # a steep step in the middle only
        flat[start:start + plateau.nbytes] = plateau.tobytes()
        result = verify_lut_offset(bytes(flat), ho, z_step, LUT_BYTES)
        self.assertTrue(result["passed"], msg=str(result))
        self.assertTrue(result["structural"]["ok"])
        self.assertFalse(result["seam_has_power_on_this_sample"])
        self.assertFalse(result["seam_conclusive_on_this_sample"])
        self.assertLess(result["seam_headroom"], 10.0)

    def test_a_powerless_morphology_cannot_veto_the_structure(self):
        """When the seam cannot discriminate, the section table still decides.

        Failing here would be worse than useless: it would let a known-blind
        check override a measurement that is exact by construction.
        """
        blob, ho, z_step = self._fake_blob()
        flat = bytearray(blob)
        start = ho + 20 + LUT_BYTES
        rng = np.random.default_rng(11)
        noise = (30_000 + rng.integers(-25, 26, size=(48, 64))).astype("<u4")
        flat[start:start + noise.nbytes] = noise.tobytes()
        result = verify_lut_offset(bytes(flat), ho, z_step, LUT_BYTES)
        self.assertTrue(result["passed"], msg=str(result))
        self.assertFalse(result["seam_has_power_on_this_sample"])
        # and the structure is what carried the decision
        self.assertEqual(result["structural"]["lut_bytes"], LUT_BYTES)

    def test_whole_row_shift_is_caught_by_the_invalid_count(self):
        """A shift of exactly one row leaves no seam at all.

        The seam metric alone cannot see it, so the scanner also refuses any
        candidate that drags extra sentinel-valued palette bytes into the map.
        """
        blob, ho, z_step = self._fake_blob(width=64, height=48)
        shift = LUT_BYTES - 64 * 4          # one row of 64 uint32 samples
        wrong = score_lut_candidate(blob, ho, z_step, shift)
        right = score_lut_candidate(blob, ho, z_step, LUT_BYTES)
        # the seam is invisible...
        self.assertLess(wrong["seam_ratio"], 2.0, msg=str(wrong))
        # ...but the dragged-in palette row is not
        self.assertGreater(wrong["invalid_fraction"],
                           right["invalid_fraction"] + 1e-3)
        self.assertAlmostEqual(wrong["invalid_fraction"], 64 / (64 * 48),
                               places=9)

    def test_one_sample_shift_produces_a_large_seam(self):
        blob, ho, z_step = self._fake_blob()
        right = score_lut_candidate(blob, ho, z_step, LUT_BYTES)
        wrong = score_lut_candidate(blob, ho, z_step, LUT_BYTES - 4)
        self.assertGreater(wrong["seam_ratio"], 10 * right["seam_ratio"],
                           msg=f"{wrong} vs {right}")


class TestRealContainer(unittest.TestCase):
    """End-to-end checks against a real measurement file."""

    @unittest.skipUnless(REAL_CAG.is_file(), "20补充pass.cag not available")
    def test_data_names_cover_every_measurement(self):
        with CagHeightReader(REAL_CAG) as reader:
            groups = reader.groups
            self.assertEqual(len(groups), 10)
            self.assertEqual(len(reader.data_names), 10)
            self.assertEqual(reader.data_names[1], "1 2")

    @unittest.skipUnless(REAL_CAG.is_file(), "20补充pass.cag not available")
    def test_read_height_map_obeys_the_contract(self):
        with CagHeightReader(REAL_CAG) as reader:
            hm = reader.read_height_map(1)
        self.assertIsInstance(hm, HeightMap)
        self.assertEqual(hm.shape, (1536, 2048))
        self.assertAlmostEqual(hm.dx_um, 0.344174, places=9)
        self.assertAlmostEqual(hm.dy_um, 0.344174, places=9)
        # every masked-out pixel is NaN, every masked-in pixel is finite
        self.assertTrue(np.all(np.isnan(hm.z[~hm.valid_mask])))
        self.assertTrue(np.all(np.isfinite(hm.z[hm.valid_mask])))
        self.assertEqual(hm.metadata["mask_source"], "cag_raw_sentinel")
        self.assertFalse(hm.metadata["mask_is_fabricated"])

    @unittest.skipUnless(REAL_CAG.is_file(), "20补充pass.cag not available")
    def test_first_pixels_match_the_official_csv_row(self):
        """Independent cross-check of the LUT offset and the rounding rule."""
        csv_path = (REPO / "氧化锆" / "pass实验数据" / "csv文件" / "20补充pass"
                    / "1 2_高度.csv")
        if not csv_path.is_file():
            self.skipTest("official CSV not available")
        with open(csv_path, "r", encoding="gbk") as stream:
            lines = stream.read().splitlines()
        caption = lines.index('"高度"')
        first = [float(v.strip('"')) for v in lines[caption + 1].split(",")[:4]]
        with CagHeightReader(REAL_CAG) as reader:
            hm = reader.read_height_map(1)
        np.testing.assert_allclose(hm.z[0, :4], first, atol=1e-9)

    @unittest.skipUnless(REAL_CAG.is_file(), "20补充pass.cag not available")
    def test_lut_offset_verifies_on_real_data(self):
        with CagHeightReader(REAL_CAG, verify_lut=True) as reader:
            reader.read_height_map(1)
            check = reader.lut_checks[1]
        self.assertTrue(check["passed"], msg=str(check))
        self.assertTrue(check["structural"]["ok"], msg=str(check))

    @unittest.skipUnless(REAL_CAG.is_file(), "20补充pass.cag not available")
    def test_palette_derives_to_776_on_real_data(self):
        """The palette length is measured from the container, not assumed."""
        with CagHeightReader(REAL_CAG) as reader:
            blob = reader.archive.read(reader._vk4[1])
        offsets = dict(zip(VK4_KEYS, struct.unpack_from("<18I", blob, 12)))
        got = derive_lut_bytes(blob, offsets["height"], expected=LUT_BYTES)
        self.assertTrue(got["ok"], msg=str(got))
        self.assertEqual(got["lut_bytes"], LUT_BYTES)
        self.assertEqual(got["width"], 2048)
        self.assertEqual(got["height"], 1536)


class TestNoFillingInProductionPath(unittest.TestCase):
    """The old fill-invalid helper must not survive in src/."""

    def test_src_does_not_reference_filling(self):
        offenders = []
        for path in sorted((REPO / "src").glob("*.py")):
            text = path.read_text(encoding="utf-8")
            for token in ("_fill_invalid", "fill_invalid"):
                if token in text:
                    offenders.append(f"{path.name}:{token}")
        self.assertEqual(offenders, [], msg=f"filling found in src/: {offenders}")


if __name__ == "__main__":
    unittest.main()
