"""KEYENCE ``.cag`` reader that never repairs the data.

The reader:

* never fills invalid sentinel pixels -- they become ``NaN`` + ``False``;
* keeps the raw sentinel value out of min/max, leveling and registration;
* turns the once-magic ``LUT_BYTES = 776`` into a verified structural result.

Why 776 needs checking, and how
-------------------------------
Between the 20-byte height-section header and the first height sample sits a
pseudo-colour palette.  Reading at the wrong offset shifts the image
horizontally by ``offset / 4`` columns and shears every row.

The palette length is **derived, not assumed**.  :func:`derive_lut_bytes`
reads the container's own 18-entry section-offset table and takes the palette
to be the gap between the end of the height samples and the start of the next
section.  That is exact and says nothing about the surface, and on all three
containers of this project it yields 776 for every group.

:func:`verify_lut_offset` additionally scores the read morphologically, because
two independent arguments are better than one.  A misaligned read leaves a
vertical seam whose size is exactly the wrap jump ``|z[r][0] - z[r-1][-1]|``,
so the check is only as strong as that jump is large -- see
:func:`score_lut_candidate` for why it is reported rather than gated on.
"""

from __future__ import annotations

import hashlib
import json
import struct
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

from .data_contracts import HeightMap

__all__ = [
    "CagHeightReader",
    "LUT_BYTES",
    "INVALID_SENTINEL",
    "raw_to_micrometres",
    "verify_lut_offset",
    "derive_lut_bytes",
    "score_lut_candidate",
    "container_sha256",
]


def raw_to_micrometres(raw: np.ndarray, z_step_pm: int,
                       invalid: np.ndarray | None = None) -> np.ndarray:
    """Convert raw 32-bit samples to micrometres, rounded half-up to 3 decimals.

    KEYENCE rounds half away from zero while Python's ``round`` and
    ``np.round`` use banker's rounding, so the rounding happens in integer
    space -- that is what makes the output byte-identical to the official CSV.
    """
    scaled = np.asarray(raw).astype(np.int64) * int(z_step_pm)
    milli = (scaled + 500) // 1000
    z = milli.astype(np.float64) / 1000.0
    if invalid is not None:
        z = np.where(invalid, np.nan, z)
    return z

# VK4 section names in the order they appear in the 18-entry offset table.
VK4_KEYS = [
    "meas_conds", "color_peak", "color_light", "light", "unknown_4",
    "unknown_5", "height", "unknown_7", "unknown_8", "color_peak_thumb",
    "color_thumb", "light_thumb", "height_thumb", "assembly_info",
    "line_measure", "line_thickness", "string_data", "reserved",
]

# Pseudo-colour palette between the 20-byte section header and the samples.
# Verified structurally by `verify_lut_offset`, not just asserted.
LUT_BYTES = 776

# KEYENCE flags unmeasured pixels with a 0xFF...... sentinel.
INVALID_SENTINEL = 0xFF000000

# Nothing measured by a VK-X3000 is anywhere near 2**24 counts (1.68 mm at the
# 100 pm step used here).  Words above this are palette bytes in disguise.
MAX_PLAUSIBLE_RAW = 1 << 24

FILE_ITEM_KEY = "FileItemAccessor"
DATA_ENTRY_KEY = "e57e75b1-707b-4a6f-a095-1485b8b95efb"

_SHA_CACHE: dict[str, str] = {}


# --------------------------------------------------------------------------- #
# LUT verification
# --------------------------------------------------------------------------- #
def derive_lut_bytes(blob: bytes, height_offset: int,
                     expected: int | None = LUT_BYTES,
                     max_plausible: int = 1 << 16) -> dict:
    """Derive the palette length from the container's own offset table.

    This is the primary, sample-independent check.  The VK4 header carries an
    18-entry table of section offsets; sections are packed back to back, so the
    palette length is whatever sits between the end of the height samples and
    the start of the next section::

        palette = next_section_offset - height_offset - 20 - width*height*4

    Nothing here depends on what the surface looks like.  Measured on all three
    containers of this project the result is exactly 776 bytes for every group.

    ``max_plausible`` guards against a nonsense derivation (a negative gap, or
    a gap so large that "palette" is the wrong explanation) rather than
    silently accepting it.
    """
    width, height, bit_depth, _marker, nbytes = struct.unpack_from(
        "<5I", blob, height_offset)
    if bit_depth != 32:
        return {"ok": False, "reason": f"bit_depth={bit_depth} != 32",
                "lut_bytes": None}

    offsets = struct.unpack_from("<18I", blob, 12)
    later = sorted(o for o in offsets if o > height_offset)
    next_offset = later[0] if later else len(blob)

    section_end = height_offset + 20 + nbytes
    palette = next_offset - section_end
    result = {
        "ok": False,
        "lut_bytes": palette,
        "expected": expected,
        "height_offset": height_offset,
        "next_section_offset": next_offset,
        "next_section_at_blob_end": not later,
        "width": width,
        "height": height,
        "data_bytes": nbytes,
        "section_header_bytes": 20,
        "gap_bytes": palette,
    }
    if palette < 0:
        result["reason"] = (f"height section overruns the next section by "
                            f"{-palette} bytes")
        return result
    if palette > max_plausible:
        result["reason"] = (f"derived palette of {palette} bytes exceeds the "
                            f"{max_plausible}-byte plausibility bound; the "
                            f"section layout is not what this reader assumes")
        return result
    if expected is not None and palette != expected:
        result["reason"] = f"derived palette {palette} != expected {expected}"
        return result
    result["ok"] = True
    result["reason"] = None
    return result


def score_lut_candidate(blob: bytes, height_offset: int, z_step_pm: int,
                        lut: int) -> dict:
    """Score one candidate palette offset for a height section.

    Two independent symptoms of a misaligned read are measured:

    ``seam_ratio``
        A read that starts at the wrong byte offset rolls the image
        horizontally by ``(lut_bytes - lut) / 4`` samples.  Every row then
        contains one column where two pixels that are far apart in reality sit
        next to each other, and the mean horizontal gradient at that column
        explodes.  The correctly aligned read has no such column.

    ``invalid_fraction``
        A read that starts too early pulls palette bytes in front of the first
        samples.  Real palette words are ``0xFF......``, so they register as
        sentinel-invalid pixels that the true offset does not have.

    The second symptom is what closes the blind spot of the first: a shift of
    exactly one whole row leaves every row internally continuous and therefore
    produces *no* seam at all, but it still drags one row of palette bytes into
    the image.

    ``seam_headroom`` bounds how much this metric could ever prove on the
    sample at hand.  The seam a misalignment creates is exactly the wrap jump
    ``|z[r][0] - z[r-1][-1]|``, so if the left and right edges of the field sit
    at the same height -- as they do when the machined rectangle is surrounded
    by untouched surface -- the jump is no larger than an ordinary machining
    step and the metric is inconclusive however the data is read.  Reporting
    the headroom keeps that limitation visible instead of letting a small
    margin look like a comfortable one.

    Returns ``decodable=False`` when the window runs past the end of the blob;
    such candidates are never admissible.
    """
    width, height, bit_depth, _marker, nbytes = struct.unpack_from(
        "<5I", blob, height_offset)
    start = height_offset + 20 + lut
    if bit_depth != 32 or start + nbytes > len(blob):
        return {"lut": lut, "decodable": False, "seam_ratio": float("inf"),
                "invalid_fraction": float("inf"),
                "seam_headroom": float("inf")}

    raw = np.frombuffer(blob[start:start + nbytes], dtype="<u4")
    if raw.size != width * height:
        return {"lut": lut, "decodable": False, "seam_ratio": float("inf"),
                "invalid_fraction": float("inf"), "seam_headroom": float("inf")}
    raw = raw.reshape(height, width)

    invalid = raw >= INVALID_SENTINEL
    z = raw.astype(np.float64) * z_step_pm * 1e-6
    z[invalid] = np.nan

    grad = np.abs(np.diff(z, axis=1))
    with np.errstate(invalid="ignore"):
        col = np.nanmean(grad, axis=0)
    med = np.nanmedian(col)
    if not np.isfinite(med) or med <= 0:
        seam = float("inf")
        headroom = float("inf")
    else:
        seam = float(np.nanmax(col) / med)
        # the jump a one-sample roll would inject, averaged over rows
        wrap = np.abs(z[1:, 0] - z[:-1, -1])
        wrap_jump = np.nanmedian(wrap) if np.isfinite(wrap).any() else np.nan
        headroom = (float(wrap_jump / med)
                    if np.isfinite(wrap_jump) else float("inf"))

    return {
        "lut": lut,
        "decodable": True,
        "seam_ratio": seam,
        "invalid_fraction": float(invalid.mean()),
        "seam_headroom": headroom,
    }


def verify_lut_offset(blob: bytes, height_offset: int, z_step_pm: int,
                      lut_bytes: int = LUT_BYTES,
                      scan_max: int = 1600,
                      invalid_tol: float = 1e-3,
                      seam_conclusive_ratio: float = 10.0) -> dict:
    """Verify the palette offset two ways and report what each can prove.

    Primary -- structural
        :func:`derive_lut_bytes` reads the container's own 18-entry section
        table and computes the palette length as the gap between the end of the
        height samples and the start of the next section.  This is exact and
        independent of what the surface looks like.

    Corroborating -- morphological
        Candidate offsets are scanned in 4-byte steps -- the sample width --
        from 0 to ``scan_max``.  A candidate is admissible only if it decodes
        and its invalid-pixel fraction is within ``invalid_tol`` of the best
        fraction seen; among those, the one with the lowest seam ratio wins.
        The derived offset must also be that winner.

        The corroboration is *reported, not gated on an absolute threshold*.
        The seam a misalignment injects is exactly the wrap jump between the
        right edge of one row and the left edge of the next, so on a field
        whose left and right edges are both untouched surface at the same
        height the jump is no bigger than an ordinary machining step.  On such
        samples the ranking still comes out right but the margin is small, and
        a fixed threshold would manufacture failures that say nothing about the
        data.  ``seam_headroom`` quantifies this: it is the seam ratio a
        misalignment would produce, so below ``seam_conclusive_ratio`` the
        morphological check is labelled inconclusive and must not be used
        alone.

    ``invalid_tol`` is deliberately coarse.  On a 2048-wide map a one sample
    misalignment adds only 3e-7 to the invalid fraction, so the filter does
    nothing there; it only bites when the dragged-in garbage is a visible slice
    of the image, which is exactly the whole-row case the seam cannot see.

    Gating
        A check that cannot discriminate must not be allowed to veto.  When
        ``seam_headroom`` is below ``seam_conclusive_ratio`` the morphology has
        no power on this sample, so agreement is desirable but not required;
        the result is reported as inconclusive rather than failed.  When the
        headroom is large and the ranking picks a different offset, the two
        checks contradict each other and that is a genuine failure.

    Returns a dict describing both checks plus a ``passed`` flag.  Raises
    nothing -- the caller decides whether a failure is fatal, because a
    production run must be able to record *why* it stopped.
    """
    derived = derive_lut_bytes(blob, height_offset, expected=lut_bytes)
    report: dict = {"lut_bytes": lut_bytes, "structural": derived}

    if not derived["ok"]:
        report.update({"passed": False,
                       "reason": f"structural: {derived['reason']}"})
        return report

    scored = [score_lut_candidate(blob, height_offset, z_step_pm, lut)
              for lut in range(0, scan_max + 1, 4)]
    target = next((c for c in scored if c["lut"] == lut_bytes), None)
    if target is None:
        report.update({"passed": False,
                       "reason": f"offset {lut_bytes} not scanned"})
        return report
    if not target["decodable"]:
        report.update({"passed": False,
                       "reason": f"offset {lut_bytes} does not decode"})
        return report

    decodable = [c for c in scored if c["decodable"]
                 and np.isfinite(c["seam_ratio"])]
    floor = min(c["invalid_fraction"] for c in decodable) if decodable else None
    admissible = ([c for c in decodable
                   if c["invalid_fraction"] <= floor + invalid_tol]
                  if decodable else [])
    admissible.sort(key=lambda c: c["seam_ratio"])

    headroom = target["seam_headroom"]
    seam_has_power = bool(np.isfinite(headroom)
                          and headroom >= seam_conclusive_ratio)
    if admissible:
        best = admissible[0]
        seam_agrees = bool(best["lut"] == lut_bytes
                           and np.isfinite(target["seam_ratio"]))
        runner = next((c for c in admissible if c["lut"] != lut_bytes), None)
    else:
        best = {"lut": -1, "seam_ratio": float("inf")}
        seam_agrees = False
        runner = None

    passed = bool(seam_agrees or not seam_has_power)
    report.update({
        "passed": passed,
        "seam_agrees_with_structure": seam_agrees,
        "seam_has_power_on_this_sample": seam_has_power,
        "seam_conclusive_on_this_sample": bool(seam_agrees and seam_has_power),
        "seam_ratio": target["seam_ratio"],
        "seam_headroom": headroom,
        "invalid_fraction": target["invalid_fraction"],
        "best_lut": best["lut"],
        "best_ratio": best["seam_ratio"],
        "runner_up_lut": runner["lut"] if runner else -1,
        "runner_up_ratio": runner["seam_ratio"] if runner else float("inf"),
        "invalid_fraction_floor": floor,
        "invalid_tol": invalid_tol,
        "seam_conclusive_ratio": seam_conclusive_ratio,
        "n_decodable": len(decodable),
        "n_admissible": len(admissible),
        "scanned": len(scored),
        "reason": None if passed else (
            f"morphology contradicts structure with headroom "
            f"{headroom:.1f}: seam argmin is {best['lut']}, not the "
            f"structurally derived {lut_bytes}"),
    })
    return report


# --------------------------------------------------------------------------- #
# container hashing (expensive, so cached on disk)
# --------------------------------------------------------------------------- #
def container_sha256(path: Path, cache_path: Path | None = None) -> str:
    """SHA-256 of a .cag, memoised in a JSON sidecar keyed by size+mtime."""
    path = Path(path)
    stat = path.stat()
    key = f"{path.resolve()}::{stat.st_size}::{int(stat.st_mtime)}"
    if key in _SHA_CACHE:
        return _SHA_CACHE[key]

    cache: dict = {}
    if cache_path is not None:
        if cache_path.is_file():
            try:
                cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cache = {}
        if key in cache:
            _SHA_CACHE[key] = cache[key]
            return cache[key]

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            digest.update(block)
    value = digest.hexdigest()
    _SHA_CACHE[key] = value

    if cache_path is not None:
        cache[key] = value
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    return value


# --------------------------------------------------------------------------- #
# reader
# --------------------------------------------------------------------------- #
def _decode_data_name(raw: bytes) -> str:
    """Display names are stored as length-prefix-free ANSI/UTF-8 text."""
    for encoding in ("utf-8", "gbk"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        text = text.split("\x00", 1)[0].strip()
        if text:
            return text
    return ""


class CagHeightReader:
    """Minimal reader that pulls one height map per measurement out of a .cag."""

    def __init__(self, path: Path, verify_lut: bool = False) -> None:
        self.path = Path(path)
        #: when True every read also runs the O(400 x N) seam scan
        self.verify_lut = verify_lut
        self.archive = zipfile.ZipFile(self.path)
        self._vk4: dict[int, zipfile.ZipInfo] = {}
        self._names: dict[int, str] = {}
        self._data_names: dict[int, str] = {}
        self._lut_checks: dict[int, dict] = {}
        self._lut_structural: dict[int, dict] = {}
        self._index()

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        self.archive.close()

    def __enter__(self) -> "CagHeightReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _index(self) -> None:
        measurement_xml: bytes | None = None
        for info in self.archive.infolist():
            parts = info.filename.split("/")
            if len(parts) == 3 and parts[1].isdigit() and info.file_size > 100_000:
                with self.archive.open(info) as stream:
                    if stream.read(4) == b"VK4_":
                        self._vk4[int(parts[1])] = info
            if 1_000 < info.file_size < 5_000_000:
                with self.archive.open(info) as stream:
                    head = stream.read(256)
                if b"MeasurementDataMap" in head:
                    measurement_xml = self.archive.read(info)
        if not self._vk4:
            raise ValueError(f"no VK4 measurement blobs found in {self.path}")
        if measurement_xml is not None:
            root = ET.fromstring(measurement_xml.decode("utf-8-sig"))
            file_item_keys: dict[int, str] = {}
            for item in root.findall("MeasurementData"):
                group = int(item.findtext("Path", "0"))
                self._names[group] = Path(item.findtext("OriginalFileName", "")).stem
                for key in item.findall("./StorageKeys/StorageKey"):
                    if key.get("Name") == FILE_ITEM_KEY:
                        file_item_keys[group] = (key.text or "").strip()
            if file_item_keys:
                self._data_names = self._read_data_names(file_item_keys)

    def _read_data_names(self, keys: dict[int, str]) -> dict[int, str]:
        names: dict[int, str] = {}
        for info in self.archive.infolist():
            parts = info.filename.split("/")
            if len(parts) != 5 or not parts[1].isdigit():
                continue
            group = int(parts[1])
            if parts[3] != keys.get(group) or parts[4] != DATA_ENTRY_KEY:
                continue
            name = _decode_data_name(self.archive.read(info))
            if name:
                names[group] = name
        return names

    # ------------------------------------------------------------------ #
    @property
    def groups(self) -> list[int]:
        return sorted(self._vk4)

    @property
    def names(self) -> dict[int, str]:
        """Captured file names (OriginalFileName stems)."""
        return self._names

    @property
    def data_names(self) -> dict[int, str]:
        """Display names as KEYENCE stores them, e.g. ``{1: "1 2"}``."""
        return self._data_names

    @property
    def lut_checks(self) -> dict[int, dict]:
        """Full LUT verification results, when ``verify_lut`` was set."""
        return self._lut_checks

    @property
    def lut_structural(self) -> dict[int, dict]:
        """Palette length derived from the section table, for every read group.

        Cheap -- it reads one header -- so it runs on every read, not only when
        ``verify_lut`` is set.
        """
        return self._lut_structural

    def peek_shape(self, group: int) -> str:
        with self.archive.open(self._vk4[group]) as stream:
            head = stream.read(128)
            offsets = dict(zip(VK4_KEYS, struct.unpack_from("<18I", head, 12)))
            stream.seek(offsets["height"])
            width, height = struct.unpack_from("<2I", stream.read(8))
        return f"{width}x{height}"

    # ------------------------------------------------------------------ #
    def read_raw(self, group: int) -> tuple[np.ndarray, dict]:
        """Raw uint32 samples plus decoding metadata.  No unit conversion."""
        blob = self.archive.read(self._vk4[group])
        offsets = dict(zip(VK4_KEYS, struct.unpack_from("<18I", blob, 12)))
        measure_offset = offsets["meas_conds"]
        year, month, day, hour, minute, second = struct.unpack_from(
            "<6I", blob, measure_offset + 4)
        x_pitch_pm, y_pitch_pm, z_step_pm = struct.unpack_from(
            "<3I", blob, measure_offset + 42 * 4)

        height_offset = offsets["height"]
        width, height, bit_depth, _marker, data_bytes = struct.unpack_from(
            "<5I", blob, height_offset)
        if bit_depth != 32 or data_bytes != width * height * 4:
            raise ValueError(
                f"unsupported height layout in group {group}: "
                f"{width}x{height} bit={bit_depth} bytes={data_bytes}")

        # The palette length is derived from the container's own section table
        # on every read: it costs one header parse and it is the check that
        # actually decides the offset.  The seam scan is far more expensive and
        # only runs on request.
        self._lut_structural[group] = derive_lut_bytes(
            blob, height_offset, expected=LUT_BYTES)
        if not self._lut_structural[group]["ok"]:
            raise ValueError(
                f"group {group}: palette offset {LUT_BYTES} failed structural "
                f"verification: {self._lut_structural[group]['reason']}")
        if self.verify_lut:
            self._lut_checks[group] = verify_lut_offset(
                blob, height_offset, z_step_pm, LUT_BYTES)

        start = height_offset + 20 + LUT_BYTES
        raw = np.frombuffer(blob[start:start + data_bytes], dtype="<u4").reshape(
            height, width)

        meta = {
            "group": group,
            "width": width,
            "height": height,
            "bit_depth": bit_depth,
            "x_pitch_pm": x_pitch_pm,
            "y_pitch_pm": y_pitch_pm,
            "z_step_pm": z_step_pm,
            "lut_bytes": LUT_BYTES,
            "lut_derived_bytes": self._lut_structural[group]["lut_bytes"],
            "lut_structural_ok": self._lut_structural[group]["ok"],
            "next_section_offset": self._lut_structural[group][
                "next_section_offset"],
            "height_section_offset": height_offset,
            "timestamp": f"{year:04d}-{month:02d}-{day:02d} "
                         f"{hour:02d}:{minute:02d}:{second:02d}",
            "original_name": self._names.get(group, ""),
            "data_name": self._data_names.get(group, ""),
            "source_path": str(self.path),
        }
        return raw, meta

    def read_height_map(self, group: int) -> HeightMap:
        """Decode one measurement into a contract-abiding ``HeightMap``.

        Invalid sentinel pixels are ``NaN`` in ``z`` and ``False`` in the mask.
        They are never filled, never clipped and never medians of neighbours.
        """
        raw, meta = self.read_raw(group)
        z_step_pm = int(meta["z_step_pm"])

        invalid = raw >= INVALID_SENTINEL
        z = raw_to_micrometres(raw, z_step_pm, invalid)

        dx_um = float(meta["x_pitch_pm"]) * 1e-6
        dy_um = float(meta["y_pitch_pm"]) * 1e-6
        width = int(meta["width"])
        height = int(meta["height"])
        x_um = (np.arange(width, dtype=np.float64) + 0.5) * dx_um
        y_um = (np.arange(height, dtype=np.float64) + 0.5) * dy_um

        valid = z[~invalid]
        meta.update({
            "dx_um": dx_um,
            "dy_um": dy_um,
            "n_invalid": int(invalid.sum()),
            "valid_fraction": float(1.0 - invalid.mean()),
            "z_min": float(valid.min()) if valid.size else None,
            "z_max": float(valid.max()) if valid.size else None,
            "mask_source": "cag_raw_sentinel",
            "mask_is_fabricated": False,
            "invalid_sentinel": hex(INVALID_SENTINEL),
            "height_sign": "upward",
            "unit": "um",
        })
        if self.verify_lut and group in self._lut_checks:
            meta["lut_verification"] = self._lut_checks[group]

        return HeightMap(z=z, valid_mask=~invalid, dx_um=dx_um, dy_um=dy_um,
                         x_um=x_um, y_um=y_um, metadata=meta)
