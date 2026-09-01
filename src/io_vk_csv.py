"""Parser for KEYENCE ``ImageDataCsv`` height exports.

The file is a 17-line quoted metadata header (GBK, CRLF), a blank line, a
``"高度"`` caption, then the matrix with every value quoted to three decimals.

Important limitation, stated up front
-------------------------------------
This format carries **no** validity mask.  The parser therefore cannot know
which pixels were measured.  It marks the result with
``mask_source = unavailable`` and ``mask_is_fabricated = True`` so that no
downstream stage can mistake the all-valid mask for evidence.

Zero is a legitimate height and is never treated as "missing".
"""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .data_contracts import HeightMap

__all__ = ["parse_vk_csv", "VkCsvError"]

CAPTION = "高度"
HEADER_ROWS = 17

# value / divisor = micrometres.  Every divisor is an exact power of ten, so
# IEEE division is correctly rounded: a height written as "30958.000" in nm
# lands on exactly the same double as "30.958" in um.  Multiplying by 1e-3
# would miss by one ulp and would make two exports of one surface disagree.
UNIT_DIVISOR_TO_UM = {
    "um": 1.0, "μm": 1.0, "µm": 1.0,
    "nm": 1000.0,
    "mm": 0.001,
}

# Keys that must be present and parseable, or the file is not what we think.
REQUIRED_KEYS = ("水平", "垂直", "XY校准", "单位")


class VkCsvError(ValueError):
    """Raised when a file does not look like a KEYENCE ImageDataCsv."""


def _sniff_encoding(path: Path) -> str:
    for encoding in ("utf-8-sig", "gbk"):
        try:
            with path.open("r", encoding=encoding) as stream:
                stream.read(4096)
            return encoding
        except UnicodeDecodeError:
            continue
    raise VkCsvError(f"cannot decode {path} as utf-8-sig or gbk")


def _read_header(path: Path, encoding: str) -> tuple[dict[str, list[str]], int, str]:
    """Return the header mapping, the data start row, and the caption found."""
    header: dict[str, list[str]] = {}
    caption_row = None
    with path.open("r", encoding=encoding, newline="") as stream:
        reader = csv.reader(stream)
        for index, row in enumerate(reader):
            cells = [c.strip() for c in row if c.strip() != ""]
            if not cells:
                continue
            if cells[0] == CAPTION:
                caption_row = index
                break
            header[cells[0]] = cells[1:]
    if caption_row is None:
        raise VkCsvError(f"no {CAPTION!r} caption found in {path}")
    return header, caption_row + 1, encoding


def parse_vk_csv(path: Path, dtype=np.float64) -> HeightMap:
    """Read a KEYENCE height CSV into a ``HeightMap``.

    The returned mask is all-true because the format does not encode validity;
    ``metadata['mask_source']`` says so explicitly.  Do not use such a map to
    draw conclusions about missing data.
    """
    path = Path(path)
    if not path.is_file():
        raise VkCsvError(f"missing file: {path}")

    encoding = _sniff_encoding(path)
    header, skiprows, encoding = _read_header(path, encoding)

    missing = [k for k in REQUIRED_KEYS if k not in header or not header[k]]
    if missing:
        raise VkCsvError(f"{path}: header missing {missing}")

    try:
        width = int(header["水平"][0])
        height = int(header["垂直"][0])
    except (ValueError, IndexError) as exc:
        raise VkCsvError(f"{path}: unreadable 水平/垂直: {exc}") from exc

    xy_raw = header["XY校准"]
    if len(xy_raw) < 2:
        raise VkCsvError(f"{path}: XY校准 needs a value and a unit")
    try:
        xy_value = float(xy_raw[0])
    except ValueError as exc:
        raise VkCsvError(f"{path}: XY校准 value {xy_raw[0]!r} is not numeric") from exc
    xy_unit = xy_raw[1].strip().lower()
    if xy_unit not in UNIT_DIVISOR_TO_UM:
        raise VkCsvError(f"{path}: unknown XY unit {xy_unit!r}")
    pitch_um = xy_value / UNIT_DIVISOR_TO_UM[xy_unit]

    z_unit = header["单位"][0].strip().lower()
    if z_unit not in UNIT_DIVISOR_TO_UM:
        raise VkCsvError(f"{path}: unknown height unit {z_unit!r}")
    z_divisor = UNIT_DIVISOR_TO_UM[z_unit]
    z_scale = 1.0 / z_divisor

    frame = pd.read_csv(path, encoding=encoding, skiprows=skiprows,
                        header=None, dtype=dtype, na_filter=False)
    z = frame.to_numpy(dtype=np.float64)

    if z.shape != (height, width):
        raise VkCsvError(
            f"{path}: matrix is {z.shape[1]}x{z.shape[0]}, "
            f"header claims {width}x{height}")

    if z_divisor != 1.0:
        z = z / z_divisor

    # The format has no mask.  A height of exactly 0.0 is real data.
    valid_mask = np.ones_like(z, dtype=bool)
    dx_um = float(pitch_um)
    dy_um = float(pitch_um)
    x_um = (np.arange(width, dtype=np.float64) + 0.5) * dx_um
    y_um = (np.arange(height, dtype=np.float64) + 0.5) * dy_um

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)

    metadata = {
        "source_path": str(path),
        "source_format": "keyence_imagedata_csv",
        "encoding": encoding,
        "header": {k: v for k, v in header.items()},
        "measurement_datetime": header.get("测量日期", [""])[0],
        "data_name": header.get("测量数据名", [""])[0],
        "instrument": header.get("机型", [""])[0],
        "dx_um": dx_um,
        "dy_um": dy_um,
        "xy_calibration_raw": [xy_raw[0], xy_raw[1]],
        "z_unit_raw": header["单位"][0],
        "z_scale_to_um": z_scale,
        "width": width,
        "height": height,
        # ---- the honest part -------------------------------------------
        "mask_source": "unavailable",
        "mask_is_fabricated": True,
        "mask_note": ("KEYENCE ImageDataCsv carries no validity mask; the "
                      "all-true mask is a placeholder, not evidence"),
        "csv_sha256": digest.hexdigest(),
    }
    return HeightMap(z=z, valid_mask=valid_mask, dx_um=dx_um, dy_um=dy_um,
                     x_um=x_um, y_um=y_um, metadata=metadata)


def summarise(path: Path) -> dict:
    """Small helper used by inventory scripts."""
    hm = parse_vk_csv(path)
    out = hm.summary()
    out["source_path"] = str(path)
    out["data_name"] = hm.metadata.get("data_name", "")
    return out


_DIGITS = re.compile(r"\d+")


def natural_key(text: str) -> tuple:
    """Natural-number ordering, e.g. ``2_高度`` before ``10_高度``."""
    return tuple(int(p) if p.isdigit() else p for p in _DIGITS.split(text))
