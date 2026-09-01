"""WP2 acceptance probe -- decode real measurements from every session.

The unit tests already prove the decoder against a synthetic blob and against
one measurement of one container.  This script is the other half of the WP2
gate: it opens **all three** ``.cag`` containers and decodes real measurements
through the production code path, then reports shape, pixel pitch, valid
fraction and invalid count.  Any session that fails to decode stops the run.

The probe also re-checks the slot mapping inside every container, because
``60Pass组.cag`` is scanned in a serpentine pattern and 12 of its 30 data
names are reversed (``14 13`` rather than ``13 14``).  A slot rule derived
arithmetically would look perfectly healthy here while silently swapping
sample identity, so the check compares the *multiset* of parsed ids against
the design table and reports the reversed ones explicitly.

Nothing is written back into the input tree.  The probe is read-only.

Outputs
    <output-dir>/phase0/session_decode_probe.json
    <output-dir>/phase0/session_decode_probe.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import pathlib
import platform
import sys

import numpy as np
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_contracts import ContractViolation  # noqa: E402
from src.io_cag import CagHeightReader  # noqa: E402
from src.provenance import parse_data_name  # noqa: E402

PROBE_STAGE = "WP2_decode_probe"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sha256_of(path: pathlib.Path, chunk: int = 8 << 20) -> str:
    import hashlib

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


def pick_groups(groups: list[int], limit: int) -> list[int]:
    """Evenly spaced groups covering the session, always including both ends."""
    if limit <= 0 or limit >= len(groups):
        return groups
    if limit == 1:
        return [groups[len(groups) // 2]]
    idx = np.linspace(0, len(groups) - 1, limit).round().astype(int)
    return sorted({groups[int(i)] for i in idx})


def read_session_manifest(path: pathlib.Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


# --------------------------------------------------------------------------- #
# per-session work
# --------------------------------------------------------------------------- #
def probe_session(session: dict, cfg: dict, repo_root: pathlib.Path,
                  max_groups: int, verify_lut: bool) -> dict:
    sid = session["session_id"]
    cag_path = repo_root / session["cag_path"]
    expected_groups = int(session["rois_per_measurement"])
    record: dict = {
        "session_id": sid,
        "cag_path": session["cag_path"],
        "cag_exists": cag_path.is_file(),
        "mapping_rule": session["mapping_rule"],
        "errors": [],
        "warnings": [],
        "measurements": [],
    }

    if not record["cag_exists"]:
        record["errors"].append(f"CAG container is missing: {cag_path}")
        record["decoded"] = 0
        return record

    record["cag_bytes"] = cag_path.stat().st_size
    if verify_lut:
        record["cag_sha256"] = sha256_of(cag_path)

    design_ids = read_design_sample_ids(
        repo_root / session["design_path"],
        cfg["design"]["sample_id_column"],
        cfg["design"]["encoding"],
    )
    record["design_rows"] = len(design_ids)

    with CagHeightReader(cag_path, verify_lut=verify_lut) as reader:
        groups = reader.groups
        record["n_groups"] = len(groups)
        record["n_data_names"] = len(reader.data_names)

        if not groups:
            record["errors"].append("no VK4 height blobs found in container")
            record["decoded"] = 0
            return record

        # ---- slot mapping: read every data name, never derive it ----------
        parsed: dict[int, list[int]] = {}
        reversed_names: list[dict] = []
        for group in groups:
            name = reader.data_names.get(group, "")
            if not name:
                record["errors"].append(f"group {group}: no data name in CAG")
                continue
            try:
                tokens = parse_data_name(name)
            except ValueError as exc:
                record["errors"].append(f"group {group}: {exc}")
                continue
            if len(tokens) != expected_groups:
                record["errors"].append(
                    f"group {group}: data name {name!r} has {len(tokens)} "
                    f"tokens, session expects {expected_groups}")
                continue
            parsed[group] = tokens
            if len(tokens) == 2 and tokens[0] > tokens[1]:
                reversed_names.append({"group": group, "data_name": name,
                                       "slot_1_sample_id": tokens[0],
                                       "slot_2_sample_id": tokens[1]})

        record["n_data_names_parsed"] = len(parsed)
        record["reversed_data_names"] = reversed_names

        flat = sorted(t for tokens in parsed.values() for t in tokens)
        if flat != sorted(design_ids):
            record["errors"].append(
                f"CAG sample ids != design table sample ids "
                f"(missing {sorted(set(design_ids) - set(flat))[:5]}, "
                f"unexpected {sorted(set(flat) - set(design_ids))[:5]})")
        if len(flat) != len(set(flat)):
            record["errors"].append("duplicate sample ids across groups")

        # ---- decode a spread of real measurements -------------------------
        selected = pick_groups(groups, max_groups)
        record["groups_selected"] = selected

        for group in selected:
            entry: dict = {"group": group,
                           "data_name": reader.data_names.get(group, "")}
            entry["slot_1_sample_id"] = parsed.get(group, [None])[0]
            entry["slot_2_sample_id"] = (parsed[group][1]
                                         if group in parsed
                                         and len(parsed[group]) > 1 else None)
            try:
                hm = reader.read_height_map(group)
            except Exception as exc:  # noqa: BLE001 - the probe must not abort
                entry["decoded"] = False
                entry["error"] = f"{type(exc).__name__}: {exc}"
                record["errors"].append(f"group {group}: {entry['error']}")
                record["measurements"].append(entry)
                continue

            valid = hm.z[hm.valid_mask]
            entry.update({
                "decoded": True,
                "width": hm.z.shape[1],
                "height": hm.z.shape[0],
                "dx_um": hm.dx_um,
                "dy_um": hm.dy_um,
                "z_step_pm": hm.metadata.get("z_step_pm"),
                "n_valid": int(hm.n_valid),
                "n_invalid": int(hm.n_invalid),
                "valid_fraction": round(hm.valid_fraction, 6),
                "z_min_um": round(float(valid.min()), 3) if valid.size else None,
                "z_max_um": round(float(valid.max()), 3) if valid.size else None,
                "z_mean_um": round(float(valid.mean()), 3) if valid.size else None,
                "mask_source": hm.metadata.get("mask_source"),
                "mask_is_fabricated": bool(hm.mask_is_fabricated),
                "timestamp": hm.metadata.get("timestamp"),
                "original_name": hm.metadata.get("original_name", ""),
            })
            structural = reader.lut_structural.get(group, {})
            entry["lut_derived_bytes"] = structural.get("lut_bytes")
            entry["lut_structural_ok"] = bool(structural.get("ok"))
            if not structural.get("ok"):
                record["errors"].append(
                    f"group {group}: palette offset failed structural "
                    f"verification: {structural.get('reason')}")

            if verify_lut and group in reader.lut_checks:
                lut = reader.lut_checks[group]
                entry["lut_verified"] = bool(lut["passed"])
                entry["lut_seam_ratio"] = round(lut["seam_ratio"], 3)
                headroom = lut["seam_headroom"]
                entry["lut_seam_headroom"] = (round(headroom, 3)
                                              if np.isfinite(headroom) else None)
                entry["lut_seam_conclusive"] = bool(
                    lut["seam_conclusive_on_this_sample"])
                if not lut["passed"]:
                    record["errors"].append(
                        f"group {group}: palette offset {lut['lut_bytes']} "
                        f"failed verification: {lut['reason']}")
                if not lut["seam_conclusive_on_this_sample"]:
                    record["warnings"].append(
                        f"group {group}: seam check is inconclusive on this "
                        f"sample (headroom {headroom:.2f} < "
                        f"{lut['seam_conclusive_ratio']}); the palette offset "
                        f"rests on the structural derivation alone")
            if hm.mask_is_fabricated:
                record["errors"].append(
                    f"group {group}: mask was fabricated, CAG must carry the "
                    f"raw sentinel mask")
            record["measurements"].append(entry)

    record["decoded"] = sum(1 for m in record["measurements"]
                            if m.get("decoded"))
    return record


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/rectangle_registration.yaml")
    parser.add_argument("--output-dir", default=None,
                        help="defaults to paths.outputs_root from the config")
    parser.add_argument("--groups", type=int, default=3,
                        help="measurements to decode per session; 0 = all")
    parser.add_argument("--verify-lut", action="store_true",
                        help="also run structural LUT verification per group")
    parser.add_argument("--hash-containers", action="store_true",
                        help="SHA-256 every container (slow, 9.5 GB for 120)")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(pathlib.Path(args.config).read_text(encoding="utf-8"))
    repo_root = REPO_ROOT / cfg["paths"]["repo_root"]
    out_root = pathlib.Path(args.output_dir or cfg["paths"]["outputs_root"])
    if not out_root.is_absolute():
        out_root = repo_root / out_root
    phase0 = out_root / cfg["output_layout"]["phase0"]
    phase0.mkdir(parents=True, exist_ok=True)

    sessions = read_session_manifest(repo_root / cfg["paths"]["session_manifest"])
    print(f"[probe] {len(sessions)} sessions, up to "
          f"{args.groups or 'all'} measurements each")

    records = []
    for session in sessions:
        print(f"[probe] {session['session_id']} ...", flush=True)
        record = probe_session(session, cfg, repo_root, args.groups,
                               args.verify_lut)
        if args.hash_containers and record["cag_exists"]:
            record["cag_sha256"] = sha256_of(repo_root / session["cag_path"])
        records.append(record)
        status = "ok" if not record["errors"] else f"{len(record['errors'])} error(s)"
        print(f"[probe]   groups={record.get('n_groups')} "
              f"decoded={record.get('decoded')} {status}", flush=True)
        for measurement in record["measurements"]:
            if not measurement.get("decoded"):
                print(f"[probe]   ! group {measurement['group']}: "
                      f"{measurement.get('error')}", flush=True)
                continue
            print(f"[probe]   group {measurement['group']:>3} "
                  f"{measurement['width']}x{measurement['height']} "
                  f"pitch={measurement['dx_um']:.6f} "
                  f"valid={measurement['valid_fraction']:.4f} "
                  f"invalid={measurement['n_invalid']}", flush=True)

    blockers = [f"{r['session_id']}: {e}"
                for r in records for e in r["errors"]]
    decision = "STOP" if blockers else "PASS"

    report = {
        "stage": PROBE_STAGE,
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "config": args.config,
        "config_sha256": sha256_of(pathlib.Path(args.config)),
        "groups_per_session": args.groups,
        "verify_lut": args.verify_lut,
        "sessions": records,
        "blockers": blockers,
        "decision": decision,
    }

    json_path = phase0 / "session_decode_probe.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                         encoding="utf-8")

    columns = ["session_id", "group", "data_name", "slot_1_sample_id",
               "slot_2_sample_id", "width", "height", "dx_um", "dy_um",
               "z_step_pm", "n_valid", "n_invalid", "valid_fraction",
               "z_min_um", "z_max_um", "z_mean_um", "mask_source",
               "mask_is_fabricated", "timestamp", "original_name",
               "lut_structural_ok", "lut_derived_bytes",
               "lut_verified", "lut_seam_ratio",
               "lut_seam_headroom", "lut_seam_conclusive",
               "decoded", "error"]
    csv_path = phase0 / "session_decode_probe.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns,
                                extrasaction="ignore")
        writer.writeheader()
        for record in records:
            if not record["measurements"]:
                writer.writerow({"session_id": record["session_id"],
                                 "decoded": False,
                                 "error": "; ".join(record["errors"]) or
                                          "no measurement decoded"})
            for measurement in record["measurements"]:
                writer.writerow({"session_id": record["session_id"],
                                 **measurement})

    print(f"\n[probe] decision={decision}")
    print(f"[probe] wrote {json_path}")
    print(f"[probe] wrote {csv_path}")
    for blocker in blockers:
        print(f"[probe] BLOCKER {blocker}")
    return 0 if decision == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
