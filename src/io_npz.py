"""Native intermediate format for height maps.

A plain numeric CSV cannot carry the validity mask, and carrying the mask in a
second CSV invites the two files drifting apart.  ``.npz`` keeps height, mask,
pixel pitch and metadata in one container that round-trips exactly.

Heights are stored as float64 even though float32 would halve the file.  The
reason is the CAG/CSV equivalence gate: the raw decoder produces values that
are byte-identical to the official KEYENCE export, and the gate compares them
at ``atol=1e-9`` µm.  float32 has a relative resolution of 6e-8, which at a
100 µm span is a 6e-6 µm error -- four orders of magnitude larger than the
tolerance.  Caching in float32 would turn a real disagreement into a rounding
argument, so the cache is bit-exact instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data_contracts import HeightMap

__all__ = ["save_height_npz", "load_height_npz"]

FORMAT_VERSION = 1


def save_height_npz(path: Path, hm: HeightMap) -> Path:
    """Store a HeightMap.  Height is float64, mask bool, metadata JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        height=hm.z.astype(np.float64),
        valid_mask=hm.valid_mask.astype(bool),
        x_um=hm.x_um.astype(np.float64),
        y_um=hm.y_um.astype(np.float64),
        dx_um=np.float64(hm.dx_um),
        dy_um=np.float64(hm.dy_um),
        format_version=np.int64(FORMAT_VERSION),
        metadata_json=np.array(json.dumps(hm.metadata, ensure_ascii=False,
                                          default=str)),
    )
    return path


def load_height_npz(path: Path) -> HeightMap:
    """Read back a HeightMap written by :func:`save_height_npz`."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        version = int(data["format_version"])
        if version != FORMAT_VERSION:
            raise ValueError(f"{path}: npz format version {version} != "
                             f"{FORMAT_VERSION}")
        z = data["height"].astype(np.float64)
        mask = data["valid_mask"].astype(bool)
        dx_um = float(data["dx_um"])
        dy_um = float(data["dy_um"])
        x_um = data["x_um"].astype(np.float64)
        y_um = data["y_um"].astype(np.float64)
        metadata = json.loads(str(data["metadata_json"]))
    return HeightMap(z=z, valid_mask=mask, dx_um=dx_um, dy_um=dy_um,
                     x_um=x_um, y_um=y_um, metadata=metadata)
