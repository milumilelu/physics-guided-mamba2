"""src.spectrum / src.composition parity tests vs the frozen phase2_5 and
phase1_5 _lib originals (WP1 migration protocol), plus unit tests for the
new Fourier-phase realization helpers."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "phase1_5"))

import _lib as l15  # noqa: E402
from src import composition as scomp  # noqa: E402
from src import spectrum as sspec  # noqa: E402


def _load(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p25 = _load("phase2_5_lib_spec_parity", "experiments/phase2_5/_lib.py")

BANDS = [(8.0, 16.0, "8_16"), (16.0, 32.0, "16_32"), (32.0, 64.0, "32_64"),
         (64.0, np.inf, "64_inf")]


def _batch(n: int = 3, s: int = 40, seed: int = 5):
    rng = np.random.default_rng(seed)
    x = np.arange(s)
    field = np.empty((n, s, s))
    for i in range(n):
        fx, fy = np.meshgrid(x, x)
        field[i] = (np.sin(2 * np.pi * fx / 7.0) * 0.4
                    + np.cos(2 * np.pi * fy / 11.0) * 0.3
                    + rng.normal(scale=0.05, size=(s, s)))
    return field


class DctGridParity(unittest.TestCase):
    def test_dct_lambda_grid_identical(self):
        for shape in ((40, 40), (160, 160), (17, 23)):
            np.testing.assert_array_equal(
                sspec.dct_lambda_grid(shape, 0.5),
                l15.dct_lambda_grid(shape, 0.5))


class RadialSpectrumParity(unittest.TestCase):
    def test_radial_spectrum_identical(self):
        R = _batch()
        a, ea = p25.radial_spectrum(R, 0.5, 12, 0.7, 160.0)
        b, eb = sspec.radial_spectrum(R, 0.5, 12, 0.7, 160.0)
        self.assertEqual(sorted(a.keys()), sorted(b.keys()))
        for key in a:
            np.testing.assert_array_equal(np.asarray(a[key]),
                                          np.asarray(b[key]))
        np.testing.assert_array_equal(ea, eb)

    def test_spectrum_descriptors_identical(self):
        R = _batch()
        out, _ = p25.radial_spectrum(R, 0.5, 12, 0.7, 160.0)
        a = p25.spectrum_descriptors(out["energy_fraction"],
                                     out["lambda_geo_um"])
        b = sspec.spectrum_descriptors(out["energy_fraction"],
                                       out["lambda_geo_um"])
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])

    def test_directional_band_metrics_identical(self):
        R = _batch(n=2, s=160)  # full grid so all four bands are non-empty
        a = p25.directional_band_metrics(R, 0.5, BANDS, 18)
        b = sspec.directional_band_metrics(R, 0.5, BANDS, 18)
        self.assertTrue(a.equals(b))


class CompositionParity(unittest.TestCase):
    def test_five_part_composition_identical(self):
        R = _batch(n=2, seed=9)
        pa, da = p25.five_part_composition(R, 0.5)
        pb, db = scomp.five_part_composition(R, 0.5)
        for key in pa:
            np.testing.assert_array_equal(pa[key], pb[key])
        np.testing.assert_array_equal(da, db)
        closed = np.sum([pb[k] for k in pb], axis=0)
        np.testing.assert_allclose(closed, 1.0, rtol=0, atol=1e-12)

    def test_zero_replacement_identical(self):
        p = np.array([[0.5, 0.5, 0.0, 0.0, 0.0],
                      [0.2, 0.2, 0.2, 0.2, 0.2]])
        a = p25.apply_zero_replacement(p, 1e-12, 1e-8)
        b = scomp.apply_zero_replacement(p, 1e-12, 1e-8)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    def test_ilr_matrix_roundtrip_and_aitchison(self):
        np.testing.assert_array_equal(scomp.ilr_matrix(),
                                      p25.ilr_matrix())
        np.testing.assert_array_equal(scomp.ILR_A, p25.ILR_A)
        p = np.array([[0.1, 0.2, 0.3, 0.25, 0.15],
                      [0.3, 0.1, 0.1, 0.2, 0.3]])
        z = scomp.ilr_transform(p)
        np.testing.assert_allclose(scomp.ilr_inverse(z), p, atol=1e-12)
        np.testing.assert_array_equal(scomp.aitchison_distance(p, p + 1e-9),
                                      p25.aitchison_distance(p, p + 1e-9))


class PhaseOnlyHelpers(unittest.TestCase):
    def test_phase_only_field_flattens_amplitude_and_real(self):
        rng = np.random.default_rng(2)
        f = rng.normal(size=(32, 32))
        q = sspec.phase_only_field(f)
        self.assertEqual(q.shape, f.shape)
        self.assertTrue(np.isfinite(q).all())
        F = np.fft.fft2(q)
        np.testing.assert_allclose(np.abs(F[1:, :]), 1.0, atol=1e-8)
        self.assertAlmostEqual(abs(F[0, 0]), 0.0, places=12)

    def test_shift_invariance_and_self_distance(self):
        rng = np.random.default_rng(4)
        f = rng.normal(size=(32, 32))
        q = sspec.phase_only_field(f)
        self.assertAlmostEqual(
            sspec.shift_invariant_phase_distance(q, q, max_shift_px=4),
            0.0, places=12)
        qs = np.roll(np.roll(q, 3, axis=0), -2, axis=1)
        self.assertAlmostEqual(
            sspec.shift_invariant_phase_distance(q, qs, max_shift_px=4),
            0.0, places=12)

    def test_large_shift_beyond_window_not_invariant(self):
        rng = np.random.default_rng(6)
        f = rng.normal(size=(32, 32))
        q = sspec.phase_only_field(f)
        qs = np.roll(np.roll(q, 13, axis=0), 13, axis=1)
        self.assertGreater(
            sspec.shift_invariant_phase_distance(q, qs, max_shift_px=4),
            1e-3)


if __name__ == "__main__":
    unittest.main()
