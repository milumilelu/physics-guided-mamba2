#!/usr/bin/env python3
"""Batch-export height maps from a KEYENCE .cag container to CSV files.

The output reproduces the ``ImageDataCsv`` layout written by KEYENCE VK analyser
software: a 17-line metadata header (GBK encoded, CRLF line endings), a blank
line, a ``"高度"`` caption, then the height matrix with every value quoted and
formatted to three decimals.

Decoding is delegated entirely to :mod:`src.io_cag`.  This module owns only the
CSV *writing*; it no longer contains a second, divergent copy of the decoder.

Invalid pixels
--------------
KEYENCE flags unmeasured pixels with a ``0xFF......`` sentinel.  What those
pixels should become in a CSV is a policy choice, and the two choices exist for
different purposes:

``--invalid-policy preserve_nan`` (default)
    Invalid pixels are written as empty fields.  The file then states
    "this pixel was not measured" instead of inventing a number.  Use this for
    anything scientific.

``--invalid-policy keyence_compat``
    Invalid pixels are replaced by the median of their valid 8-neighbourhood,
    which is what the original version of this script did unconditionally.  It
    exists only to reproduce legacy files or to study how filling perturbs
    downstream statistics.  Every file written this way is a *derived*
    artefact and must never be treated as measured data or as an independent
    check on the decoder.

The filler lives here, outside ``src/``, so that the production path
(``src/io_cag.py``) cannot reach it.  ``tests/test_io_cag.py`` enforces that.

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
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.io_cag import CagHeightReader  # noqa: E402  (re-exported for scripts)

__all__ = ["CagHeightReader", "write_keyence_csv", "write_plain_csv",
           "export_container", "main"]


def fill_invalid(z: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Replace invalid pixels with the median of their valid 8-neighbourhood.

    Compatibility only.  The result is a *derived* surface: the filled pixels
    look like measurements but are interpolations, and no downstream stage may
    treat them as evidence.
    """
    invalid = ~valid_mask
    out = np.where(valid_mask, z, np.nan)
    height, width = invalid.shape
    for row, col in np.argwhere(invalid):
        r0, r1 = max(0, row - 1), min(height, row + 2)
        c0, c1 = max(0, col - 1), min(width, col + 2)
        patch = out[r0:r1, c0:c1]
        good = patch[~np.isnan(patch)]
        out[row, col] = float(np.median(good)) if good.size else 0.0
    return out


EMPTY = ""


def _fmt_row(z_row: np.ndarray, mask_row: np.ndarray,
             policy: str) -> list[str]:
    if policy == "preserve_nan":
        return [EMPTY if not ok else f"{v:.3f}"
                for v, ok in zip(z_row, mask_row)]
    return [f"{v:.3f}" for v in z_row]


def write_keyence_csv(path: Path, z: np.ndarray, meta: dict,
                      data_name: str, valid_mask: np.ndarray,
                      policy: str = "preserve_nan") -> None:
    """Write a height map using KEYENCE's own ImageDataCsv layout.

    ``最小值``/``最大值`` are taken over valid pixels only; a sentinel value
    must never reach the summary statistics.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    xy_nm = meta["dx_um"] * 1e3
    valid = z[valid_mask]
    z_min = float(valid.min()) if valid.size else float("nan")
    z_max = float(valid.max()) if valid.size else float("nan")
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
        ["最小值", f"{z_min:.3f}"],
        ["最大值", f"{z_max:.3f}"],
        ["单位", "μm"],
        ["基准数据名称", ""],
    ]

    with open(path, "w", encoding="gbk", newline="") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_ALL, lineterminator="\r\n")
        writer.writerows(header)
        fh.write("\r\n")
        writer.writerow(["高度"])
        for z_row, mask_row in zip(z, valid_mask):
            writer.writerow(_fmt_row(z_row, mask_row, policy))


def write_plain_csv(path: Path, z: np.ndarray, valid_mask: np.ndarray,
                    policy: str = "preserve_nan") -> None:
    """Write a bare numeric matrix, no metadata header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\r\n")
        for z_row, mask_row in zip(z, valid_mask):
            writer.writerow(_fmt_row(z_row, mask_row, policy))


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


_UNSAFE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def build_stem(args, reader, group: int) -> str:
    """Build the file stem for one measurement group.

    ``data`` uses the name KEYENCE itself stores in the container (the same
    string it writes to the CSV ``测量数据名`` field), falling back to the
    captured file name and finally to the group index.  The token order is
    preserved: ``14 13`` stays ``14 13``, because that is the physical slot
    order of a serpentine scan.
    """
    if args.naming == "data":
        name = reader.data_names.get(group, "")
        if name:
            return _UNSAFE.sub("_", name).strip().rstrip(".") or f"{group:03d}"
    if args.naming in ("data", "original") and reader.names.get(group):
        return reader.names[group]
    return f"{group:03d}"


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
        shape = reader.peek_shape(groups[0])
        if args.dry_run:
            if verbose:
                named = sum(1 for g in groups if reader.data_names.get(g))
                preview = ", ".join(
                    f"{build_stem(args, reader, g)}{args.suffix}"
                    for g in groups[:3])
                print(f"  {len(groups)} group(s), {shape}, "
                      f"{named} named in container")
                print(f"  e.g. {preview}")
            return len(groups), 0, shape

        out_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        started = time.time()
        for n, group in enumerate(groups, 1):
            stem = build_stem(args, reader, group)
            out_path = out_dir / f"{stem}{args.suffix}"
            if args.skip_existing and out_path.exists():
                continue
            t0 = time.time()
            hm = reader.read_height_map(group)
            z = hm.z
            mask = hm.valid_mask
            if args.invalid_policy == "keyence_compat":
                z = fill_invalid(z, mask)
                mask = np.ones_like(mask)
            meta = dict(hm.metadata)
            meta["dx_um"] = hm.dx_um
            meta["dy_um"] = hm.dy_um
            meta["width"] = hm.z.shape[1]
            meta["height"] = hm.z.shape[0]
            if args.format == "keyence":
                write_keyence_csv(out_path, z, meta, stem, mask,
                                  args.invalid_policy)
            else:
                write_plain_csv(out_path, z, mask, args.invalid_policy)
            size = out_path.stat().st_size
            total += size
            if verbose:
                valid = z[mask]
                print(f"  [{n:>3}/{len(groups)}] group {group:>3}  "
                      f"{meta['width']}x{meta['height']}  "
                      f"z {valid.min():7.3f}~{valid.max():7.3f} um  "
                      f"invalid={hm.n_invalid:<5} {size / 1e6:6.1f} MB  "
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
    parser.add_argument("--naming", choices=("data", "original", "index"),
                        default="data",
                        help="data -> the name stored in the container, e.g. "
                             "'1 2' (default, matches the official export); "
                             "original -> the captured VK4 file name, e.g. "
                             "MeasureData20260528105621; index -> 001")
    parser.add_argument("--invalid-policy",
                        choices=("preserve_nan", "keyence_compat"),
                        default="preserve_nan",
                        help="preserve_nan writes empty fields for unmeasured "
                             "pixels; keyence_compat median-fills them for "
                             "format reproduction only (derived artefact)")
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

    if args.invalid_policy == "keyence_compat":
        print("WARNING: --invalid-policy keyence_compat median-fills "
              "unmeasured pixels.\n"
              "         The output is a DERIVED artefact.  It must not be used "
              "as pipeline\n"
              "         input, nor as independent evidence that the decoder is "
              "correct.\n")

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
