#!/usr/bin/env python3
"""Batch-export height maps from a KEYENCE .cag container to CSV files.

The output reproduces the ``ImageDataCsv`` layout written by KEYENCE VK analyser
software: a 17-line metadata header (GBK encoded, CRLF line endings), a blank
line, a ``"高度"`` caption, then the height matrix with every value quoted and
formatted to three decimals.

Decoding notes (verified pixel-exact against official KEYENCE exports):
  * A ``.cag`` is a ZIP container; each measurement ``Path/<n>/`` holds one VK4
    blob whose 32-bit height samples are embedded in the blob itself.
  * The height section starts at ``height_offset + 20 + LUT_BYTES`` where
    ``LUT_BYTES = 776`` is a pseudo-colour lookup table that follows the
    20-byte section header.  Reading at ``height_offset + 20`` shifts the image
    by 194 columns.
  * Physical height in micrometres is ``raw_uint32 * z_step_pm * 1e-6``.

Usage
-----
    # one container
    python export_height_csv.py --cag 氧化锆/pass实验数据/120正式.cag \
        --output-dir 氧化锆/pass实验数据/csv文件/120正式

    # every container under a directory tree (mirrors the source layout)
    python export_height_csv.py \
        --batch-root "E:/博士课题资料/光机所实验原始数据" \
        --output-root "E:/博士课题资料/csv高度"

    # preview the plan first, or export a subset as a bare numeric matrix
    python export_height_csv.py --batch-root ... --output-root ... --dry-run
    python export_height_csv.py --cag ... --groups 1-5,10 --format plain
"""

from __future__ import annotations

import argparse
import csv
import struct
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

# VK4 section names in the order they appear in the 18-entry offset table.
VK4_KEYS = [
    "meas_conds", "color_peak", "color_light", "light", "unknown_4",
    "unknown_5", "height", "unknown_7", "unknown_8", "color_peak_thumb",
    "color_thumb", "light_thumb", "height_thumb", "assembly_info",
    "line_measure", "line_thickness", "string_data", "reserved",
]

# Pseudo-colour lookup table that sits between the 20-byte height section
# header and the first height sample.  Confirmed constant across every group of
# both 120正式.cag and 20补充pass.cag.
LUT_BYTES = 776

# KEYENCE flags unmeasured pixels with a 0xFF...... sentinel.
INVALID_SENTINEL = 0xFF000000


class CagHeightReader:
    """Minimal reader that pulls one height map per measurement out of a .cag."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.archive = zipfile.ZipFile(path)
        self._vk4: dict[int, zipfile.ZipInfo] = {}
        self._names: dict[int, str] = {}
        self._index()

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
            # Measurement blobs live at <root>/<group>/<uuid>; metadata sits
            # deeper (5+ segments) or at the archive root.
            # VK4 blob size tracks the resolution: ~568 KB for the older
            # 1024x64 line scans, ~9.8 MB for 1024x768, ~38 MB for 2048x1536.
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
            for item in root.findall("MeasurementData"):
                group = int(item.findtext("Path", "0"))
                self._names[group] = Path(item.findtext("OriginalFileName", "")).stem

    @property
    def groups(self) -> list[int]:
        return sorted(self._vk4)

    def read_height(self, group: int, fill_invalid: bool = True
                    ) -> tuple[np.ndarray, dict[str, object]]:
        """Return the height map in micrometres plus its metadata."""
        info = self._vk4[group]
        blob = self.archive.read(info)
        offsets = dict(zip(VK4_KEYS, struct.unpack_from("<18I", blob, 12)))
        measure_offset = offsets["meas_conds"]
        year, month, day, hour, minute, second = struct.unpack_from(
            "<6I", blob, measure_offset + 4
        )
        x_pitch_pm, y_pitch_pm, z_step_pm = struct.unpack_from(
            "<3I", blob, measure_offset + 42 * 4
        )

        height_offset = offsets["height"]
        width, height, bit_depth, _marker, data_bytes = struct.unpack_from(
            "<5I", blob, height_offset
        )
        if bit_depth != 32 or data_bytes != width * height * 4:
            raise ValueError(
                f"unsupported height layout in group {group}: "
                f"{width}x{height} bit={bit_depth} bytes={data_bytes}"
            )

        start = height_offset + 20 + LUT_BYTES
        raw = np.frombuffer(blob[start:start + data_bytes], dtype="<u4").reshape(
            height, width
        )

        invalid = raw >= INVALID_SENTINEL
        z_um = _to_micrometres(raw, z_step_pm)

        if fill_invalid and invalid.any():
            z_um = _fill_invalid(z_um, invalid)

        meta = {
            "group": group,
            "width": width,
            "height": height,
            "dx_nm": x_pitch_pm * 1e-3,
            "dy_nm": y_pitch_pm * 1e-3,
            "min": float(np.nanmin(z_um)),
            "max": float(np.nanmax(z_um)),
            "invalid": int(invalid.sum()),
            "timestamp": f"{year:04d}-{month:02d}-{day:02d} "
                         f"{hour:02d}:{minute:02d}:{second:02d}",
            "original_name": self._names.get(group, ""),
        }
        return z_um, meta


def _to_micrometres(raw: np.ndarray, z_step_pm: int) -> np.ndarray:
    """Convert raw 32-bit samples to micrometres, rounded half-up to 3 decimals.

    KEYENCE rounds half away from zero, while Python's ``round`` and
    ``np.round`` use banker's rounding.  Doing the rounding in integer space
    reproduces the official export byte for byte and avoids float drift.
    """
    scaled = raw.astype(np.int64) * int(z_step_pm)      # value * 1e-6 um
    milli = (scaled + 500) // 1000                      # half-up to 0.001 um
    return milli.astype(np.float64) / 1000.0


def _fill_invalid(z: np.ndarray, invalid: np.ndarray) -> np.ndarray:
    """Replace sentinel pixels with the median of their valid 8-neighbourhood."""
    out = z.copy()
    out[invalid] = np.nan
    idx = np.argwhere(invalid)
    h, w = invalid.shape
    for r, c in idx:
        r0, r1 = max(0, r - 1), min(h, r + 2)
        c0, c1 = max(0, c - 1), min(w, c + 2)
        patch = out[r0:r1, c0:c1]
        good = patch[~np.isnan(patch)]
        out[r, c] = float(np.median(good)) if good.size else 0.0
    return out


HEADER_FIELDS = (
    ("机型", "VK-X3000 Series"),
    ("文件类型", "ImageDataCsv"),
    ("文件版本", "1000"),
    ("测量尺寸", "高清"),
    ("测量模式", "表面形状"),
    ("扫描模式", "激光共聚焦"),
    ("镜头倍率", "20"),
    ("输出图像数据", "高度"),
    ("单位", "μm"),
    ("基准数据名称", ""),
)


def write_keyence_csv(path: Path, z: np.ndarray, meta: dict[str, object],
                      data_name: str) -> None:
    """Write a height map using KEYENCE's own ImageDataCsv layout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    xy_nm = meta["dx_nm"]
    header = [
        ["测量日期", str(meta["timestamp"])],
        ["机型", "VK-X3000 Series"],
        ["文件类型", "ImageDataCsv"],
        ["文件版本", "1000"],
        ["测量数据名", data_name],
        ["测量尺寸", "高清"],
        ["测量模式", "表面形状"],
        ["扫描模式", "激光共聚焦"],
        ["镜头倍率", "20"],
        ["XY校准", f"{xy_nm:.3f}", "nm"],
        ["输出图像数据", "高度"],
        ["水平", str(meta["width"])],
        ["垂直", str(meta["height"])],
        ["最小值", f"{meta['min']:.3f}"],
        ["最大值", f"{meta['max']:.3f}"],
        ["单位", "μm"],
        ["基准数据名称", ""],
    ]

    with open(path, "w", encoding="gbk", newline="") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_ALL, lineterminator="\r\n")
        writer.writerows(header)
        fh.write("\r\n")
        writer.writerow(["高度"])
        for row in z:
            writer.writerow([f"{v:.3f}" for v in row])


def write_plain_csv(path: Path, z: np.ndarray) -> None:
    """Write a bare numeric matrix, no metadata header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, z, delimiter=",", fmt="%.3f",
               newline="\r\n", encoding="utf-8")


def parse_groups(text: str, available: list[int], strict: bool = True) -> list[int]:
    """Parse 'all' or a comma list of numbers/ranges such as '1-5,10'.

    With ``strict=False`` (batch mode) groups missing from a container are
    dropped instead of aborting the run.
    """
    text = text.strip()
    if text.lower() in {"all", "*", ""}:
        return available
    wanted: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            wanted.extend(range(int(lo), int(hi) + 1))
        else:
            wanted.append(int(chunk))
    present = set(available)
    missing = [g for g in wanted if g not in present]
    if missing and strict:
        raise SystemExit(f"groups not present in the container: {missing}")
    return [g for g in wanted if g in present]


def export_container(cag_path: Path, out_dir: Path, args,
                     verbose: bool = True) -> tuple[int, int, str]:
    """Export every selected group of one container.

    Returns (group count, bytes written, "WxH").  Failures inside a single
    container never abort a batch run.
    """
    with CagHeightReader(cag_path) as reader:
        groups = parse_groups(args.groups, reader.groups,
                              strict=not args.batch_root)
        if not groups:
            return 0, 0, "-"
        probe = reader.read_height(groups[0])[1]
        shape = f"{probe['width']}x{probe['height']}"
        if args.dry_run:
            return len(groups), 0, shape

        out_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        started = time.time()
        for n, group in enumerate(groups, 1):
            stem = f"{group:03d}"
            if args.naming == "original" and reader._names.get(group):
                stem = reader._names[group]
            out_path = out_dir / f"{stem}{args.suffix}"
            if args.skip_existing and out_path.exists():
                continue
            t0 = time.time()
            z, meta = reader.read_height(group)
            if args.format == "keyence":
                write_keyence_csv(out_path, z, meta, stem)
            else:
                write_plain_csv(out_path, z)
            size = out_path.stat().st_size
            total += size
            if verbose:
                print(f"  [{n:>3}/{len(groups)}] group {group:>3}  "
                      f"{meta['width']}x{meta['height']}  "
                      f"z {meta['min']:7.3f}~{meta['max']:7.3f} um  "
                      f"invalid={meta['invalid']:<5} {size / 1e6:6.1f} MB  "
                      f"{time.time() - t0:5.1f}s")
        if verbose:
            print(f"  -> {len(groups)} file(s), {total / 1e6:.0f} MB, "
                  f"{time.time() - started:.0f}s")
        return len(groups), total, shape


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch-export height maps from KEYENCE .cag files to CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  single container\n"
               "    export_height_csv.py --cag a.cag --output-dir out\n"
               "  every container under a directory tree\n"
               "    export_height_csv.py --batch-root E:/原始数据 "
               "--output-root E:/csv高度\n")
    parser.add_argument("--cag", type=Path,
                        help="path to a single .cag container")
    parser.add_argument("--output-dir", type=Path,
                        help="directory that receives the CSV files")
    parser.add_argument("--batch-root", type=Path,
                        help="recursively export every .cag below this directory")
    parser.add_argument("--output-root", type=Path,
                        help="root for batch output; the source layout is "
                             "mirrored, one folder per container")
    parser.add_argument("--groups", default="all",
                        help="'all' or a selection such as '1-5,10,120'")
    parser.add_argument("--format", choices=("keyence", "plain"), default="keyence",
                        help="keyence reproduces the official ImageDataCsv "
                             "layout; plain writes a bare numeric matrix")
    parser.add_argument("--naming", choices=("index", "original"), default="index",
                        help="index -> 001_高度.csv; original -> the embedded "
                             "VK4 file name")
    parser.add_argument("--suffix", default="_高度.csv",
                        help="file name suffix (default: _高度.csv)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="do not overwrite files that already exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="scan containers and report the plan without "
                             "writing anything")
    args = parser.parse_args()

    batch = bool(args.batch_root)
    if batch and not args.output_root:
        parser.error("--batch-root requires --output-root")
    if not batch and not (args.cag and args.output_dir):
        parser.error("either --cag/--output-dir or --batch-root/--output-root "
                     "is required")

    if batch:
        if not args.batch_root.is_dir():
            raise SystemExit(f"batch root not a directory: {args.batch_root}")
        containers = sorted(args.batch_root.rglob("*.cag"))
        if not containers:
            raise SystemExit(f"no .cag files under {args.batch_root}")
        print(f"batch root : {args.batch_root}")
        print(f"output root: {args.output_root}")
        print(f"containers : {len(containers)}")
        print(f"format     : {args.format}"
              f"{'  [dry run]' if args.dry_run else ''}\n")
    else:
        if not args.cag.exists():
            raise SystemExit(f"CAG not found: {args.cag}")
        containers = [args.cag]
        print(f"container : {args.cag}")
        print(f"output    : {args.output_dir}")
        print(f"format    : {args.format}"
              f"{'  [dry run]' if args.dry_run else ''}\n")

    grand_groups = 0
    grand_bytes = 0
    failures: list[tuple[str, str]] = []
    started = time.time()

    for idx, cag in enumerate(containers, 1):
        if batch:
            rel = cag.relative_to(args.batch_root)
            out_dir = args.output_root / rel.parent / rel.stem
        else:
            out_dir = args.output_dir

        print(f"[{idx}/{len(containers)}] {cag.name}")
        try:
            n, size, shape = export_container(cag, out_dir, args, verbose=not batch)
        except Exception as exc:                      # keep the batch alive
            failures.append((str(cag), f"{type(exc).__name__}: {exc}"))
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            continue
        grand_groups += n
        grand_bytes += size
        print(f"  {shape}  {n} group(s)"
              + ("" if args.dry_run else f"  {size / 1e6:.0f} MB"))

    print(f"\n{'planned' if args.dry_run else 'exported'}: "
          f"{grand_groups} height map(s) from {len(containers) - len(failures)}"
          f"/{len(containers)} container(s)"
          + ("" if args.dry_run else f", {grand_bytes / 1e9:.2f} GB")
          + f"  [{time.time() - started:.0f}s]")
    if failures:
        print("\nfailures:")
        for path, reason in failures:
            print(f"  {path}\n      {reason}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
