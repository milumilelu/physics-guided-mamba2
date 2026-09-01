"""WP1 - build the height source manifest and validate file naming / counts.

The manifest is driven by the CAG containers, never by filename guessing:
for every measurement group the KEYENCE data name is read out of the .cag and
turned into the expected CSV stem.  Files that cannot be explained this way
are reported as extras and cause a hard stop.

Outputs
    config/height_source_manifest.csv
    <output-dir>/phase0/height_file_inventory.csv
    <output-dir>/phase0/height_inventory_validation.json
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import pathlib
import re
import statistics
import sys

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.io_cag import CagHeightReader  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sha256_of(path: pathlib.Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def read_design_sample_ids(path: pathlib.Path, column: str,
                           encoding: str) -> list[int]:
    with path.open("r", encoding=encoding, newline="") as stream:
        reader = csv.DictReader(stream)
        if column not in (reader.fieldnames or []):
            raise SystemExit(f"design table {path} has no column {column!r}")
        return [int(row[column]) for row in reader]


def cluster_by_mtime(records: list[dict], gap_s: float) -> None:
    """Split the mtime series into batches and record each file's own pace.

    `nearest_gap_s` is the smaller of the gaps to the previous and next file in
    the same directory.  A batch export produces ~1 s gaps; a human clicking
    through KEYENCE produces tens of seconds.
    """
    ordered = sorted(records, key=lambda r: r["mtime_epoch"])
    cluster = 0
    for index, record in enumerate(ordered):
        if index and (record["mtime_epoch"] - ordered[index - 1]["mtime_epoch"]) > gap_s:
            cluster += 1
        record["cluster"] = cluster

    for index, record in enumerate(ordered):
        gaps = []
        if index:
            gaps.append(record["mtime_epoch"] - ordered[index - 1]["mtime_epoch"])
        if index + 1 < len(ordered):
            gaps.append(ordered[index + 1]["mtime_epoch"] - record["mtime_epoch"])
        record["nearest_gap_s"] = round(min(gaps), 3) if gaps else float("inf")

    for cid in {r["cluster"] for r in ordered}:
        members = sorted(r["mtime_epoch"] for r in ordered if r["cluster"] == cid)
        gaps = [b - a for a, b in zip(members, members[1:])]
        med = statistics.median(gaps) if gaps else 0.0
        for r in ordered:
            if r["cluster"] == cid:
                r["cluster_median_gap_s"] = med


def classify_provenance(records: list[dict], rules: list[dict]) -> None:
    """Match each file against the pre-registered signature rules."""
    for record in records:
        delta = record["mtime_minus_ctime_s"]
        record["source_rule"] = ""
        record["csv_source_type"] = "unknown"
        record["csv_export_tool"] = ""
        record["evidence_grade"] = ""
        for rule in rules:
            low, high = rule["mtime_minus_ctime_s"]
            if not (low <= delta <= high):
                continue
            need_gap = rule.get("require_nearest_gap_ge_s")
            if need_gap is not None and record["nearest_gap_s"] < need_gap:
                continue
            need_member = rule.get("require_cluster_contains")
            if need_member is not None:
                peers = [r for r in records if r["cluster"] == record["cluster"]]
                if not any(p["source_rule"] == need_member for p in peers):
                    continue
            record["source_rule"] = rule["name"]
            record["csv_source_type"] = rule["source_type"]
            record["csv_export_tool"] = rule["export_tool"]
            record["evidence_grade"] = rule["evidence_grade"]
            break


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/rectangle_registration.yaml")
    parser.add_argument("--output-dir", default=None,
                        help="defaults to paths.outputs_root from the config")
    parser.add_argument("--dry-run", action="store_true",
                        help="scan and validate but write nothing")
    parser.add_argument("--skip-hashes", action="store_true",
                        help="skip SHA-256 (NOT allowed for formal acceptance)")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(pathlib.Path(args.config).read_bytes())
    paths = cfg["paths"]
    outputs_root = pathlib.Path(args.output_dir or paths["outputs_root"])
    phase0 = outputs_root / cfg["output_layout"]["phase0"]

    naming = cfg["naming"]
    fixture_root = (REPO_ROOT / paths["fixture_root"]).resolve()
    single_re = re.compile(naming["single_pattern"])
    paired_re = re.compile(naming["paired_pattern"])
    excluded_dirs = {(REPO_ROOT / p).resolve()
                     for p in naming.get("exclude_dirs", [])}
    excluded_prefixes = tuple(naming.get("exclude_dir_prefixes", []))

    prov = cfg["provenance"]
    rules = prov["signature_rules"]

    # ---------------------------------------------------------------- sessions
    session_rows: list[dict] = []
    inventory: list[dict] = []
    blockers: list[str] = []
    warnings: list[str] = []

    with (REPO_ROOT / paths["session_manifest"]).open(
            "r", encoding="utf-8-sig", newline="") as stream:
        sessions = list(csv.DictReader(stream))

    for session in sessions:
        sid = session["session_id"]
        expected = cfg["expected_counts"]["per_session"][sid]
        n_meas = int(expected["measurements"])
        n_rect = int(expected["rectangles_per_measurement"])

        design_ids = read_design_sample_ids(
            REPO_ROOT / session["design_path"],
            cfg["design"]["sample_id_column"],
            cfg["design"]["encoding"],
        )
        if len(design_ids) != int(expected["samples"]):
            blockers.append(
                f"{sid}: design table has {len(design_ids)} rows, "
                f"expected {expected['samples']}")

        # ---- CAG data names (the only source of truth for slot -> sample)
        cag_path = REPO_ROOT / session["cag_path"]
        with CagHeightReader(cag_path) as reader:
            groups = reader.groups
            data_names = reader.data_names

        if len(groups) != n_meas:
            blockers.append(f"{sid}: CAG has {len(groups)} groups, expected {n_meas}")
        missing_names = [g for g in groups if g not in data_names]
        if missing_names:
            blockers.append(
                f"{sid}: {len(missing_names)} measurement(s) have no CAG data name")

        # ---- files present on disk
        subdir = session.get("csv_subdir", "")
        if not subdir:
            blockers.append(f"{sid}: session manifest has no csv_subdir")
            continue
        csv_dir = (REPO_ROOT / paths["csv_root"]) / subdir
        if not csv_dir.is_dir():
            blockers.append(f"{sid}: csv directory missing: {csv_dir}")
            continue

        on_disk = sorted(p for p in csv_dir.iterdir() if p.is_file())
        scanned = [p for p in on_disk
                   if p.suffix.lower() == ".csv"
                   and not p.name.startswith(excluded_prefixes)
                   and p.parent not in excluded_dirs]
        fixture_subdir = session.get("fixture_subdir", "").strip()
        fixture_dir = fixture_root / fixture_subdir if fixture_subdir else None
        fixture_scanned = (
            sorted(p for p in fixture_dir.iterdir()
                   if p.is_file() and p.suffix.lower() == ".csv")
            if fixture_dir is not None and fixture_dir.is_dir() else []
        )

        # ---- pair each measurement with its expected file
        matched: dict[pathlib.Path, int] = {}
        explained_primary: set[pathlib.Path] = set()
        explained_fixtures: set[pathlib.Path] = set()
        for group in groups:
            tokens = data_names.get(group, "").split()
            if len(tokens) != n_rect:
                blockers.append(
                    f"{sid} measurement {group}: data name "
                    f"{data_names.get(group, '')!r} has {len(tokens)} tokens, "
                    f"expected {n_rect}")
                continue
            stem = naming["paired_token_separator"].join(tokens)
            expected_path = csv_dir / f"{stem}{naming['file_suffix']}"
            fixture_path = ((fixture_dir / expected_path.name)
                            if fixture_dir is not None else None)
            if expected_path.is_file():
                explained_primary.add(expected_path)
            if fixture_path is not None and fixture_path.is_file():
                explained_fixtures.add(fixture_path)
                chosen_path = fixture_path
            elif expected_path.is_file():
                chosen_path = expected_path
            else:
                blockers.append(f"{sid} measurement {group}: missing {expected_path.name}")
                continue
            matched[chosen_path] = group

        extras = [p for p in scanned if p not in explained_primary]
        extras.extend(p for p in fixture_scanned if p not in explained_fixtures)
        for path in extras:
            blockers.append(f"{sid}: file not explained by any CAG data name: "
                            f"{path.name}")

        # ---- per-file facts
        records: list[dict] = []
        for path, group in sorted(matched.items(), key=lambda kv: kv[1]):
            stat = path.stat()
            tokens = data_names[group].split()
            record = {
                "session_id": sid,
                "measurement_id": group,
                "n_rectangles": n_rect,
                "slot_1_sample_id": int(tokens[0]),
                "slot_2_sample_id": int(tokens[1]) if n_rect == 2 else "",
                "cag_data_name": data_names[group],
                "cag_path": session["cag_path"],
                "csv_path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "csv_bytes": stat.st_size,
                "mtime_utc": dt.datetime.fromtimestamp(
                    stat.st_mtime, dt.timezone.utc).isoformat(timespec="seconds"),
                "mtime_epoch": stat.st_mtime,
                "mtime_minus_ctime_s": round(stat.st_mtime - stat.st_ctime, 3),
                "csv_sha256": "" if args.skip_hashes else sha256_of(path),
            }
            # cross-check the filename against the data name
            if path.name != f"{data_names[group]}{naming['file_suffix']}":
                blockers.append(f"{sid} measurement {group}: filename "
                                f"{path.name!r} != data name {data_names[group]!r}")
            records.append(record)

        cluster_by_mtime(records, prov["cluster_gap_s"])
        classify_provenance(records, rules)
        for record in records:
            source_path = (REPO_ROOT / record["csv_path"]).resolve()
            if fixture_root in source_path.parents:
                record["source_rule"] = "user_provided_keyence_fixture"
                record["csv_source_type"] = "keyence_official_export"
                record["csv_export_tool"] = "KEYENCE VK-X3000 software"
                record["evidence_grade"] = "user_attested_independent"

        unknown = [r for r in records if r["csv_source_type"] == "unknown"]
        if unknown:
            blockers.append(
                f"{sid}: {len(unknown)} file(s) with undetermined provenance "
                f"(e.g. {unknown[0]['csv_path']})")

        # ---- sample id set must equal the design set
        derived: list[int] = []
        for record in records:
            derived.append(record["slot_1_sample_id"])
            if record["slot_2_sample_id"] != "":
                derived.append(record["slot_2_sample_id"])
        if len(derived) != len(set(derived)):
            dupes = sorted({i for i in derived if derived.count(i) > 1})
            blockers.append(f"{sid}: duplicate sample_id from CAG data names: {dupes}")
        if sorted(derived) != sorted(design_ids):
            blockers.append(
                f"{sid}: CAG sample ids != design table sample ids "
                f"(missing {sorted(set(design_ids) - set(derived))[:5]}, "
                f"unexpected {sorted(set(derived) - set(design_ids))[:5]})")

        counts = {t: sum(1 for r in records if r["csv_source_type"] == t)
                  for t in ("keyence_official_export", "cag_decoder_derived", "unknown")}
        official = counts["keyence_official_export"]
        need = cfg["equivalence"]["fixtures_per_session_min"]
        if official < need:
            warnings.append(
                f"{sid}: only {official} official export(s), need >= {need} "
                f"for the WP3 equivalence gate -> WP3 will STOP")

        session_rows.append({
            "session_id": sid,
            "measurements_expected": n_meas,
            "measurements_matched": len(matched),
            "samples_expected": int(expected["samples"]),
            "samples_derived": len(derived),
            "official_exports": official,
            "decoder_derived": counts["cag_decoder_derived"],
            "unknown": counts["unknown"],
        })
        inventory.extend(records)

    # ---------------------------------------------------------------- write out
    manifest_columns = [
        "session_id", "measurement_id", "n_rectangles",
        "slot_1_sample_id", "slot_2_sample_id", "cag_data_name", "cag_path",
        "csv_path", "csv_source_type", "csv_export_tool", "csv_export_timestamp",
        "csv_sha256", "csv_bytes", "source_rule", "evidence_grade",
        "expected_width", "expected_height", "expected_dx_um", "expected_dy_um",
        "provenance_status",
    ]
    for record in inventory:
        record.setdefault("csv_export_timestamp", record["mtime_utc"])
        record["expected_width"] = 2048
        record["expected_height"] = 1536
        record["expected_dx_um"] = 0.344174
        record["expected_dy_um"] = 0.344174
        record["provenance_status"] = "registered"

    manifest_path = REPO_ROOT / paths["height_source_manifest"]
    if not args.dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=manifest_columns,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(inventory)

        phase0.mkdir(parents=True, exist_ok=True)
        inv_columns = manifest_columns + [
            "mtime_utc", "mtime_minus_ctime_s", "nearest_gap_s",
            "cluster", "cluster_median_gap_s"]
        with (phase0 / "height_file_inventory.csv").open(
                "w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=inv_columns,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(inventory)

    total_meas = sum(r["measurements_matched"] for r in session_rows)
    total_samples = sum(r["samples_derived"] for r in session_rows)
    expected = cfg["expected_counts"]
    if total_meas != expected["measurements_total"]:
        blockers.append(f"measurement total {total_meas} != "
                        f"{expected['measurements_total']}")
    if total_samples != expected["samples_total"]:
        blockers.append(f"sample total {total_samples} != {expected['samples_total']}")

    decision = "PASS" if not blockers else "STOP"
    report = {
        "stage": "WP1_height_source_manifest",
        "decision": decision,
        "dry_run": args.dry_run,
        "hashes_skipped": args.skip_hashes,
        "config": args.config,
        "generated_at_local": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "sessions": session_rows,
        "totals": {
            "measurements": total_meas,
            "measurements_expected": expected["measurements_total"],
            "samples": total_samples,
            "samples_expected": expected["samples_total"],
        },
        "provenance_counts": {
            t: sum(1 for r in inventory if r["csv_source_type"] == t)
            for t in ("keyence_official_export", "cag_decoder_derived", "unknown")
        },
        "blockers": blockers,
        "warnings": warnings,
    }

    if not args.dry_run:
        phase0.mkdir(parents=True, exist_ok=True)
        (phase0 / "height_inventory_validation.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"decision={decision}  measurements={total_meas}  samples={total_samples}")
    for row in session_rows:
        print(f"  {row['session_id']:<20} meas={row['measurements_matched']:>3}"
              f"  samples={row['samples_derived']:>3}"
              f"  official={row['official_exports']:>2}"
              f"  derived={row['decoder_derived']:>3}"
              f"  unknown={row['unknown']}")
    if args.skip_hashes:
        print("  WARNING: --skip-hashes used, not valid for formal acceptance")
    for warning in warnings:
        print(f"  WARNING: {warning}")
    for blocker in blockers:
        print(f"  BLOCKER: {blocker}")
    return 0 if decision == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
