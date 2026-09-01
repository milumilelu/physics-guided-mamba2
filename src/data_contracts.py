"""The one data object every stage of the pipeline exchanges.

``HeightMap`` is deliberately strict: it refuses to be constructed from data
that violates the contract.  The point is to make it impossible to carry a
silently-filled height map past the parser without noticing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["HeightMap", "ContractViolation"]


class ContractViolation(ValueError):
    """Raised when a candidate height map breaks the H_raw data contract."""


@dataclass(frozen=True)
class HeightMap:
    """A height field together with the mask that says where it is real.

    Rules enforced on construction:

    * ``z`` and ``valid_mask`` have identical shapes;
    * ``valid_mask`` is boolean;
    * every masked-out sample is ``NaN`` in ``z`` -- never a filled value;
    * every masked-in sample is finite;
    * ``dx_um`` / ``dy_um`` are strictly positive;
    * ``x_um`` / ``y_um`` match the matrix dimensions and increase monotonically.
    """

    z: np.ndarray
    valid_mask: np.ndarray
    dx_um: float
    dy_um: float
    x_um: np.ndarray
    y_um: np.ndarray
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        z = np.asarray(self.z, dtype=np.float64)
        mask = np.asarray(self.valid_mask)

        if z.ndim != 2:
            raise ContractViolation(f"z must be 2-D, got shape {z.shape}")
        if mask.shape != z.shape:
            raise ContractViolation(
                f"valid_mask shape {mask.shape} != z shape {z.shape}")
        if mask.dtype != np.bool_:
            raise ContractViolation(
                f"valid_mask must be bool, got {mask.dtype}")

        if not np.all(np.isnan(z[~mask])):
            bad = int(np.count_nonzero(~np.isnan(z[~mask])))
            raise ContractViolation(
                f"{bad} masked-out pixel(s) are not NaN; the contract forbids "
                f"carrying filled values inside a HeightMap")
        good = z[mask]
        if good.size and not np.all(np.isfinite(good)):
            bad = int(np.count_nonzero(~np.isfinite(good)))
            raise ContractViolation(
                f"{bad} masked-in pixel(s) are not finite")

        if not (self.dx_um > 0) or not np.isfinite(self.dx_um):
            raise ContractViolation(f"dx_um must be finite and > 0, got {self.dx_um}")
        if not (self.dy_um > 0) or not np.isfinite(self.dy_um):
            raise ContractViolation(f"dy_um must be finite and > 0, got {self.dy_um}")

        x = np.asarray(self.x_um, dtype=np.float64)
        y = np.asarray(self.y_um, dtype=np.float64)
        if x.shape != (z.shape[1],):
            raise ContractViolation(
                f"x_um length {x.shape} != width {z.shape[1]}")
        if y.shape != (z.shape[0],):
            raise ContractViolation(
                f"y_um length {y.shape} != height {z.shape[0]}")
        if x.size > 1 and not np.all(np.diff(x) > 0):
            raise ContractViolation("x_um must be strictly increasing")
        if y.size > 1 and not np.all(np.diff(y) > 0):
            raise ContractViolation("y_um must be strictly increasing")

        object.__setattr__(self, "z", z)
        object.__setattr__(self, "valid_mask", mask)
        object.__setattr__(self, "x_um", x)
        object.__setattr__(self, "y_um", y)

    # ------------------------------------------------------------------ #
    @property
    def shape(self) -> tuple[int, int]:
        return self.z.shape

    @property
    def n_valid(self) -> int:
        return int(self.valid_mask.sum())

    @property
    def n_invalid(self) -> int:
        return int((~self.valid_mask).sum())

    @property
    def valid_fraction(self) -> float:
        return float(self.valid_mask.mean())

    @property
    def width_um(self) -> float:
        return float(self.x_um[-1] - self.x_um[0] + self.dx_um)

    @property
    def height_um(self) -> float:
        return float(self.y_um[-1] - self.y_um[0] + self.dy_um)

    @property
    def mask_is_fabricated(self) -> bool:
        """True when the source carried no mask information at all.

        Such a HeightMap may be used as processing input, but it can never be
        quoted as evidence about mask semantics.
        """
        return bool(self.metadata.get("mask_is_fabricated", False))

    def summary(self) -> dict:
        valid = self.z[self.valid_mask]
        return {
            "shape": list(self.shape),
            "dx_um": self.dx_um,
            "dy_um": self.dy_um,
            "fov_um": [round(self.width_um, 3), round(self.height_um, 3)],
            "valid_fraction": round(self.valid_fraction, 6),
            "n_invalid": self.n_invalid,
            "z_min": None if valid.size == 0 else round(float(valid.min()), 4),
            "z_max": None if valid.size == 0 else round(float(valid.max()), 4),
            "mask_is_fabricated": self.mask_is_fabricated,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"HeightMap({self.shape[1]}x{self.shape[0]}, "
                f"dx={self.dx_um:.6f}um, valid={self.valid_fraction:.4f}, "
                f"invalid={self.n_invalid})")
