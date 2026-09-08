"""Canonical observer adapter, with explicit validity instead of epsilon labels."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy.fft import dctn
from src import composition, spectrum
from .grouping import require

BANDS = [(8., 16., "8_16"), (16., 32., "16_32")]
PARTS = ["lt8", "8_16", "16_32", "32_64", "64_inf"]


class CanonicalObserver:
    def __init__(self, zero_threshold=1e-10, replacement_delta=1e-6):
        self.zero_threshold = zero_threshold
        self.replacement_delta = replacement_delta

    def __call__(self, height, valid_mask):
        h, v = np.asarray(height, dtype=np.float64), np.asarray(valid_mask, dtype=bool)
        require(h.ndim == 3 and h.shape[1:] == (160, 160) and h.shape == v.shape, "Observer requires N x 160 x 160")
        rows, validity = [], []
        for i in range(len(h)):
            row = {"dataset_index": i, "D": np.nan, "A_med": np.nan,
                   **{f"ilr_z{j}": np.nan for j in range(1, 5)},
                   **{f"{name}_{band}": np.nan for band in ["8_16", "16_32"] for name in ["A2", "entropy"]}}
            finite = v[i].any() and np.isfinite(h[i][v[i]]).all()
            if not finite:
                rows.append(row)
                validity.append({"dataset_index": i, "target_block": "ALL", "band_valid": False,
                                 "band_energy": np.nan, "undefined_reason": "EMPTY_OR_NONFINITE_VALID_HEIGHT"})
                continue
            med = np.median(h[i][v[i]])
            residual = h[i] - med
            row.update(D=-med, A_med=np.sqrt(np.mean(residual[v[i]]**2)))
            # Historical spectral code needs a full rectangular grid. Do not invent missing pixels.
            if not v[i].all():
                rows.append(row)
                validity.append({"dataset_index": i, "target_block": "SPECTRAL", "band_valid": False,
                                 "band_energy": np.nan, "undefined_reason": "INCOMPLETE_GRID_NO_REGISTERED_IMPUTATION"})
                continue
            batch = residual[None, :, :]
            coefficients = dctn(batch, axes=(1, 2), norm="ortho")
            coefficients[:, 0, 0] = 0
            energy = float((coefficients**2).sum())
            ok = energy > 0 and np.isfinite(energy)
            validity.append({"dataset_index": i, "target_block": "ILR", "band_valid": bool(ok),
                             "band_energy": energy, "undefined_reason": "" if ok else "ZERO_NON_DC_ENERGY"})
            if ok:
                p, _ = composition.five_part_composition(batch, .5)
                vector = np.column_stack([p[key] for key in PARTS])
                vector, replaced = composition.apply_zero_replacement(vector, self.zero_threshold, self.replacement_delta)
                z = composition.ilr_transform(vector)[0]
                row.update({f"ilr_z{j+1}": float(z[j]) for j in range(4)})
                row["composition_replacement_used"] = bool(replaced.any())
            else:
                row["composition_replacement_used"] = False
            freq = np.fft.fftfreq(160, d=.5)
            fx, fy = np.meshgrid(freq, freq)
            radial = np.hypot(fx, fy)
            lam = np.full(radial.shape, np.inf)
            np.divide(1, radial, out=lam, where=radial > 0)
            window = np.outer(np.hanning(160), np.hanning(160))
            power = np.abs(np.fft.fft2((residual-residual.mean())*window))**2 / (window**2).sum()
            for lo, hi, band in BANDS:
                band_energy = float(power[(lam >= lo) & (lam < hi)].sum())
                band_ok = band_energy > 0 and np.isfinite(band_energy)
                validity.append({"dataset_index": i, "target_block": band, "band_valid": bool(band_ok),
                                 "band_energy": band_energy, "undefined_reason": "" if band_ok else "ZERO_DIRECTION_BAND_ENERGY"})
                if band_ok:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", RuntimeWarning)
                        d = spectrum.directional_band_metrics(batch, .5, [(lo, hi, band)], 18).iloc[0]
                    row[f"A2_{band}"] = float(d.A2)
                    row[f"entropy_{band}"] = float(d.angular_entropy)
            rows.append(row)
        return pd.DataFrame(rows), pd.DataFrame(validity)
