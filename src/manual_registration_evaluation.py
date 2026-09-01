"""Manual (annotator A) four-edge registration: freeze checks, manual vs
automatic consistency evaluation, and manual_v1 record construction.

Scientific wording rule (frozen in ``config/manual_registration_v1.yaml``):
the single annotator's four-edge boxes are Level 3 evidence.  Differences
between a manual centre and an automatic centre are *consistency
observations* ("disagreement", "offset") -- never "absolute error",
"accuracy" or "ground truth".

Coordinate conventions (shared with :mod:`src.resampling` and
:mod:`src.manual_four_edge_annotation`):

* raw/canonical image coordinates ``(x, y)``: origin at the field-of-view
  centre, ``+x`` to the right, ``+y`` DOWNWARD (image row direction).
* session-rotated coordinates ``(u, v)``::

      u =  x*cos(t) + y*sin(t)
      v = -x*sin(t) + y*cos(t)
      x =  u*cos(t) - v*sin(t)
      y =  u*sin(t) + v*cos(t)

* an edge box in ``(u, v)`` has ``left_u < right_u`` and ``top_v <
  bottom_v``; ``top_v`` is the image-UPPER edge because ``+v`` points down.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .canvas import available_centered_square_um

__all__ = [
    "ManualRegistrationError",
    "uv_to_xy",
    "xy_to_uv",
    "manual_center_from_edges",
    "auto_theoretical_edges",
    "validate_manual_identity",
    "manual_geometry_gate",
    "check_paired_measurements",
    "center_within_fov",
    "box_corner_margin_um",
    "merge_one_to_one",
    "evaluate_pair",
    "band_label",
    "summarize_axis",
    "summarize_radial",
    "band_fractions",
    "manual_v1_record",
    "render_approval_text",
    "PipelinePaths",
    "resolve_pipeline_paths",
    "APPROVAL_ALLOWED_STATUSES",
]


class ManualRegistrationError(RuntimeError):
    """Hard failure of a frozen manual-registration contract."""


# ---------------------------------------------------------------------------
# Canonical transforms
# ---------------------------------------------------------------------------

def uv_to_xy(u: float, v: float, theta_deg: float) -> tuple[float, float]:
    """Session-rotated ``(u, v)`` -> raw image ``(x, y)`` (µm)."""
    theta = math.radians(theta_deg)
    cosine, sine = math.cos(theta), math.sin(theta)
    return (u*cosine - v*sine, u*sine + v*cosine)


def xy_to_uv(x: float, y: float, theta_deg: float) -> tuple[float, float]:
    """Raw image ``(x, y)`` -> session-rotated ``(u, v)`` (µm)."""
    theta = math.radians(theta_deg)
    cosine, sine = math.cos(theta), math.sin(theta)
    return (x*cosine + y*sine, -x*sine + y*cosine)


def manual_center_from_edges(left_u_um: float, right_u_um: float,
                             top_v_um: float,
                             bottom_v_um: float) -> tuple[float, float]:
    """Centre of a manual box = midpoint of its four edges."""
    return ((left_u_um+right_u_um)/2.0, (top_v_um+bottom_v_um)/2.0)


def auto_theoretical_edges(center_u_um: float, center_v_um: float,
                           nominal_half_um: float = 100.0) -> dict[str, float]:
    """Theoretical four edges of a nominal box around an automatic centre.

    Sign conventions (y-down image): ``left = u - h < right = u + h`` and
    ``top = v - h < bottom = v + h``; the top edge is the one with the
    *smaller* ``v`` because ``+v`` points down the image.
    """
    return {
        "left_u_um": center_u_um - nominal_half_um,
        "right_u_um": center_u_um + nominal_half_um,
        "top_v_um": center_v_um - nominal_half_um,
        "bottom_v_um": center_v_um + nominal_half_um,
    }


# ---------------------------------------------------------------------------
# Frozen-table identity and geometry gates
# ---------------------------------------------------------------------------

def validate_manual_identity(row: dict, theta_deg: float, *,
                             prefix: str = "annotator_a_",
                             tolerance_um: float = 1e-6) -> dict:
    """Validate all internal identities of one manual annotation row.

    Raises :class:`ManualRegistrationError` on any violation; returns the
    parsed numeric record otherwise.
    """
    def field(name: str) -> float:
        raw = row.get(prefix+name, "")
        if raw in ("", None):
            raise ManualRegistrationError(
                f"missing manual field {prefix}{name} for "
                f"({row.get('session_id')}, {row.get('sample_id')})")
        return float(raw)

    state = str(row.get(prefix+"state", "")).strip()
    if state != "complete":
        raise ManualRegistrationError(
            f"annotator state is {state!r}, not 'complete'")

    left, right = field("left_u_um"), field("right_u_um")
    top, bottom = field("top_v_um"), field("bottom_v_um")
    center_u, center_v = field("center_u_um"), field("center_v_um")
    center_x, center_y = field("center_x_um"), field("center_y_um")
    width, height = field("width_um"), field("height_um")

    if not left < right:
        raise ManualRegistrationError(f"left_u {left} >= right_u {right}")
    if not top < bottom:
        raise ManualRegistrationError(f"top_v {top} >= bottom_v {bottom}")
    if width <= 0 or height <= 0:
        raise ManualRegistrationError(f"non-positive box size {width}x{height}")

    mid_u, mid_v = manual_center_from_edges(left, right, top, bottom)
    if (abs(mid_u-center_u) > tolerance_um
            or abs(mid_v-center_v) > tolerance_um):
        raise ManualRegistrationError(
            "saved centre is not the edge midpoint "
            f"({mid_u - center_u:+.3e}, {mid_v - center_v:+.3e} um)")
    if (abs((right-left)-width) > tolerance_um
            or abs((bottom-top)-height) > tolerance_um):
        raise ManualRegistrationError("saved width/height disagrees with edges")

    derived_x, derived_y = uv_to_xy(center_u, center_v, theta_deg)
    if (abs(derived_x-center_x) > tolerance_um
            or abs(derived_y-center_y) > tolerance_um):
        raise ManualRegistrationError(
            "(u,v)->(x,y) identity under session theta fails by "
            f"({derived_x-center_x:+.3e}, {derived_y-center_y:+.3e} um)")

    return {
        "left_u_um": left, "right_u_um": right,
        "top_v_um": top, "bottom_v_um": bottom,
        "center_u_um": center_u, "center_v_um": center_v,
        "center_x_um": center_x, "center_y_um": center_y,
        "width_um": width, "height_um": height,
    }


def manual_geometry_gate(width_um: float, height_um: float,
                         gate_cfg: dict) -> tuple[bool, str]:
    """Mechanical width/height gate from the frozen config."""
    width_range = gate_cfg["observed_width_um"]
    height_range = gate_cfg["observed_height_um"]
    if not (float(width_range[0]) <= width_um <= float(width_range[1])):
        return False, (f"observed width {width_um:.2f} um outside "
                       f"{width_range}")
    if not (float(height_range[0]) <= height_um <= float(height_range[1])):
        return False, (f"observed height {height_um:.2f} um outside "
                       f"{height_range}")
    return True, ""


def _paired_issues_by_key(rows: list[dict], *,
                          prefix: str,
                          minimum_separation_um: float,
                          require_slot1_left: bool
                          ) -> list[tuple[tuple[str, int], str]]:
    """Return ``((session_id, measurement_id), message)`` for every failure."""
    found: list[tuple[tuple[str, int], str]] = []
    grouped: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        roi = str(row["roi_within_measurement"])
        if roi not in {"slot_1", "slot_2"}:
            continue
        key = (row["session_id"], int(row["measurement_id"]))
        slot = grouped.setdefault(key, {})
        if roi in slot:
            found.append((key, f"{key}: duplicate {roi} annotation "
                               f"(sample_id={row.get('sample_id')})"))
            continue
        slot[roi] = row
    for key in sorted(grouped):
        pair = grouped[key]
        if set(pair) != {"slot_1", "slot_2"}:
            found.append((key, f"{key}: incomplete pair {sorted(pair)}"))
            continue
        first, second = pair["slot_1"], pair["slot_2"]
        u1 = float(first[prefix+"center_u_um"])
        v1 = float(first[prefix+"center_v_um"])
        u2 = float(second[prefix+"center_u_um"])
        v2 = float(second[prefix+"center_v_um"])
        separation = math.hypot(u2-u1, v2-v1)
        if require_slot1_left and not u2 > u1:
            found.append((key, f"{key}: slot_2 is not right of slot_1 "
                               f"(du={u2-u1:+.3f} um)"))
        if separation < minimum_separation_um:
            found.append((key, f"{key}: paired centre separation "
                               f"{separation:.3f} um < "
                               f"{minimum_separation_um} um"))
    return found


def check_paired_measurements(rows: list[dict], *,
                              prefix: str = "annotator_a_",
                              minimum_separation_um: float = 300.0,
                              require_slot1_left: bool = True) -> list[str]:
    """Validate paired measurements: slot 1 left of slot 2, separation gate.

    ``rows`` are annotation rows with ``roi_within_measurement`` in
    ``{"slot_1", "slot_2"}``.  Returns a list of human-readable issues
    (empty when every pair passes).
    """
    return [message for _, message in _paired_issues_by_key(
        rows, prefix=prefix,
        minimum_separation_um=minimum_separation_um,
        require_slot1_left=require_slot1_left)]


def paired_failure_keys(rows: list[dict], *,
                        prefix: str = "annotator_a_",
                        minimum_separation_um: float = 300.0,
                        require_slot1_left: bool = True
                        ) -> set[tuple[str, int]]:
    """``(session_id, measurement_id)`` keys whose paired gate fails.

    Structured counterpart of :func:`check_paired_measurements`; callers that
    need to flag individual rows must use this instead of string-matching the
    human-readable messages.
    """
    return {key for key, _ in _paired_issues_by_key(
        rows, prefix=prefix,
        minimum_separation_um=minimum_separation_um,
        require_slot1_left=require_slot1_left)}


def center_within_fov(*, fov_width_um: float, fov_height_um: float,
                      center_x_um: float, center_y_um: float) -> bool:
    return (abs(center_x_um) <= fov_width_um/2.0
            and abs(center_y_um) <= fov_height_um/2.0)


def box_corner_margin_um(*, left_u_um: float, right_u_um: float,
                         top_v_um: float, bottom_v_um: float,
                         theta_deg: float, fov_width_um: float,
                         fov_height_um: float) -> float:
    """Signed margin of the manual box corners inside the raw FOV (µm).

    Positive: fully inside; negative: the deepest corner overshoot.  This is
    a QA *observation* (edge-of-field sessions can legitimately clip the
    annotated rectangle corners); the hard gate is centre-within-FOV.
    """
    worst = float("inf")
    for u, v in ((left_u_um, top_v_um), (right_u_um, top_v_um),
                 (right_u_um, bottom_v_um), (left_u_um, bottom_v_um)):
        x, y = uv_to_xy(u, v, theta_deg)
        worst = min(worst, fov_width_um/2.0-abs(x), fov_height_um/2.0-abs(y))
    return worst


# ---------------------------------------------------------------------------
# One-to-one merge and per-sample evaluation
# ---------------------------------------------------------------------------

def merge_one_to_one(manual_rows: list[dict], auto_rows: list[dict], *,
                     manual_key_fields: dict | None = None,
                     auto_key_fields: dict | None = None) -> list[tuple[dict, dict]]:
    """Join two tables one-to-one on ``(session_id, sample_id)``.

    Hard-fails on duplicate or missing keys in either table.
    """
    def key(row: dict) -> tuple[str, str]:
        return (str(row["session_id"]), str(row["sample_id"]))

    manual_by_key: dict[tuple, dict] = {}
    for row in manual_rows:
        k = key(row)
        if k in manual_by_key:
            raise ManualRegistrationError(f"duplicate manual key {k}")
        manual_by_key[k] = row
    auto_by_key: dict[tuple, dict] = {}
    for row in auto_rows:
        k = key(row)
        if k in auto_by_key:
            raise ManualRegistrationError(f"duplicate automatic key {k}")
        auto_by_key[k] = row

    missing_in_auto = sorted(set(manual_by_key) - set(auto_by_key))
    if missing_in_auto:
        raise ManualRegistrationError(
            f"samples missing from automatic table: {missing_in_auto[:5]}")
    missing_in_manual = sorted(set(auto_by_key) - set(manual_by_key))
    if missing_in_manual:
        raise ManualRegistrationError(
            f"samples missing from manual table: {missing_in_manual[:5]}")

    return [(manual_by_key[k], auto_by_key[k])
            for k in sorted(manual_by_key)]


def evaluate_pair(manual: dict, auto: dict, theta_deg: float, *,
                  manual_prefix: str = "annotator_a_",
                  nominal_half_um: float = 100.0) -> dict:
    """Manual vs automatic consistency for one sample (µm).

    The automatic centre ``(x, y)`` is rotated into the session canonical
    frame before differencing, so deltas are directly comparable to the
    manual ``(u, v)`` edges.
    """
    auto_u, auto_v = xy_to_uv(float(auto["center_x_um"]),
                              float(auto["center_y_um"]), theta_deg)
    manual_u = float(manual[manual_prefix+"center_u_um"])
    manual_v = float(manual[manual_prefix+"center_v_um"])
    delta_u = auto_u - manual_u
    delta_v = auto_v - manual_v
    radial = math.hypot(delta_u, delta_v)
    edges = auto_theoretical_edges(auto_u, auto_v, nominal_half_um)
    return {
        "auto_center_u_um": auto_u,
        "auto_center_v_um": auto_v,
        "delta_u_um": delta_u,
        "delta_v_um": delta_v,
        "center_disagreement_um": radial,
        "auto_left_u_um": edges["left_u_um"],
        "auto_right_u_um": edges["right_u_um"],
        "auto_top_v_um": edges["top_v_um"],
        "auto_bottom_v_um": edges["bottom_v_um"],
        "edge_diff_left_um": edges["left_u_um"]
        - float(manual[manual_prefix+"left_u_um"]),
        "edge_diff_right_um": edges["right_u_um"]
        - float(manual[manual_prefix+"right_u_um"]),
        "edge_diff_top_um": edges["top_v_um"]
        - float(manual[manual_prefix+"top_v_um"]),
        "edge_diff_bottom_um": edges["bottom_v_um"]
        - float(manual[manual_prefix+"bottom_v_um"]),
    }


def band_label(distance_um: float, *, close_um: float,
               moderate_um: float) -> str:
    """Consistency band label (not an error threshold)."""
    if distance_um <= close_um:
        return "close"
    if distance_um <= moderate_um:
        return "moderate"
    return "large"


def summarize_axis(values: list[float]) -> dict:
    ordered = sorted(values)
    count = len(ordered)

    def quantile(fraction: float) -> float:
        if count == 0:
            return float("nan")
        position = fraction*(count-1)
        lower = int(math.floor(position))
        upper = min(lower+1, count-1)
        weight = position-lower
        return ordered[lower]*(1.0-weight)+ordered[upper]*weight

    median = quantile(0.5)
    mad = (quantile(0.5) if count == 0 else
           _median([abs(v-median) for v in values]))
    return {"median_um": median, "mad_um": mad,
            "q05_um": quantile(0.05), "q95_um": quantile(0.95)}


def summarize_radial(values: list[float]) -> dict:
    ordered = sorted(values)
    count = len(ordered)

    def quantile(fraction: float) -> float:
        if count == 0:
            return float("nan")
        position = fraction*(count-1)
        lower = int(math.floor(position))
        upper = min(lower+1, count-1)
        weight = position-lower
        return ordered[lower]*(1.0-weight)+ordered[upper]*weight

    return {"median_um": quantile(0.5), "q90_um": quantile(0.90),
            "q95_um": quantile(0.95),
            "max_um": ordered[-1] if ordered else float("nan")}


def band_fractions(distances: list[float], *, close_um: float,
                   moderate_um: float) -> dict:
    count = len(distances)
    if count == 0:
        return {"frac_le_close": float("nan"),
                "frac_close_to_moderate": float("nan"),
                "frac_above_moderate": float("nan")}
    close = sum(d <= close_um for d in distances)
    moderate = sum(close_um < d <= moderate_um for d in distances)
    large = sum(d > moderate_um for d in distances)
    return {"frac_le_close": close/count,
            "frac_close_to_moderate": moderate/count,
            "frac_above_moderate": large/count}


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        return float("nan")
    if count % 2:
        return ordered[count//2]
    return (ordered[count//2-1]+ordered[count//2])/2.0


# ---------------------------------------------------------------------------
# manual_v1 record construction (WP3)
# ---------------------------------------------------------------------------

def manual_v1_record(annotation_row: dict, *, theta_deg: float,
                     d4_transform: str, source_sha256: str,
                     config_sha256: str, gate_cfg: dict,
                     fov_width_um: float, fov_height_um: float,
                     paired_gate_ok: bool,
                     registration_method: str = "manual_four_edge_a_v1",
                     identity_tolerance_um: float = 1e-6) -> dict:
    """Build one manual_v1 registration row from a frozen annotation row.

    Hard-fails (raises) when the row violates any frozen identity contract --
    in particular a centre that does not come from the manual four edges
    (sample-wise centre-source mixing is forbidden).
    """
    identity = validate_manual_identity(
        annotation_row, theta_deg, tolerance_um=identity_tolerance_um)
    geometry_ok, geometry_reason = manual_geometry_gate(
        identity["width_um"], identity["height_um"], gate_cfg)
    fov_ok = center_within_fov(
        fov_width_um=fov_width_um, fov_height_um=fov_height_um,
        center_x_um=identity["center_x_um"], center_y_um=identity["center_y_um"])
    available = available_centered_square_um(
        fov_width_um=fov_width_um, fov_height_um=fov_height_um,
        center_x_um=identity["center_x_um"],
        center_y_um=identity["center_y_um"], theta_deg=theta_deg)
    corner_margin = box_corner_margin_um(
        left_u_um=identity["left_u_um"], right_u_um=identity["right_u_um"],
        top_v_um=identity["top_v_um"], bottom_v_um=identity["bottom_v_um"],
        theta_deg=theta_deg, fov_width_um=fov_width_um,
        fov_height_um=fov_height_um)

    passed = geometry_ok and paired_gate_ok and fov_ok
    warnings: list[str] = []
    if not geometry_ok:
        warnings.append(geometry_reason)
    if not paired_gate_ok:
        warnings.append("paired slot order/separation gate failed")
    if not fov_ok:
        warnings.append("manual centre outside measured field of view")
    if corner_margin < 0.0:
        warnings.append(
            f"manual box corner exceeds raw FOV by {-corner_margin:.2f} um "
            "(edge-of-field observation)")

    return {
        "session_id": annotation_row["session_id"],
        "measurement_id": int(annotation_row["measurement_id"]),
        "sample_id": int(annotation_row["sample_id"]),
        "roi_within_measurement": annotation_row["roi_within_measurement"],
        "theta_session_deg": theta_deg,
        "d4_transform_session": d4_transform,
        "manual_left_u_um": identity["left_u_um"],
        "manual_right_u_um": identity["right_u_um"],
        "manual_top_v_um": identity["top_v_um"],
        "manual_bottom_v_um": identity["bottom_v_um"],
        "manual_center_u_um": identity["center_u_um"],
        "manual_center_v_um": identity["center_v_um"],
        "manual_center_x_um": identity["center_x_um"],
        "manual_center_y_um": identity["center_y_um"],
        # standard downstream column names (scripts 05/06/07 contract):
        "center_x_um": identity["center_x_um"],
        "center_y_um": identity["center_y_um"],
        "manual_width_um": identity["width_um"],
        "manual_height_um": identity["height_um"],
        "registration_method": registration_method,
        "evidence_level": 3,
        "source_table": "manual_four_edge_validation_frozen.csv",
        "source_sha256": source_sha256,
        "config_sha256": config_sha256,
        "geometry_gate_pass": geometry_ok,
        "paired_gate_pass": paired_gate_ok,
        "center_within_fov": fov_ok,
        "box_corner_margin_um": corner_margin,
        "available_centered_square_um": available,
        "status": "PASS" if passed else "STOP",
        "warning": "; ".join(warnings),
    }


# ---------------------------------------------------------------------------
# Approval file guard (WP5/WP6: scripts must never approve)
# ---------------------------------------------------------------------------

APPROVAL_ALLOWED_STATUSES = ("PENDING", "BLOCKED")


def render_approval_text(*, status: str, decision: str,
                         body_lines: list[str]) -> str:
    """Render a Phase A approval document.  Refuses ``PASS`` outright."""
    if status not in APPROVAL_ALLOWED_STATUSES:
        raise ManualRegistrationError(
            f"scripts are forbidden from writing approval status {status!r}; "
            "only PENDING/BLOCKED may be written automatically")
    lines = ["# Phase A Approval (manual_v1)", "", f"Status: {status}", ""]
    lines.extend(body_lines)
    lines.append("")
    lines.append("This file must be reviewed and changed by a human; scripts "
                 "are forbidden from marking it PASS.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Versioned pipeline output paths (WP4)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelinePaths:
    """All writable locations of one pipeline run (default or tagged)."""

    resampling_dir: Path
    registered_h_reg_dir: Path
    registered_h_200_dir: Path
    registered_masks_dir: Path
    metrics_dir: Path

    @property
    def sample_fov_diagnostics_csv(self) -> Path:
        return self.resampling_dir / "sample_fov_diagnostics.csv"

    @property
    def session_canvas_csv(self) -> Path:
        return self.resampling_dir / "session_canvas.csv"

    @property
    def common_fov_summary_json(self) -> Path:
        return self.resampling_dir / "common_fov_summary.json"

    @property
    def resampling_summary_json(self) -> Path:
        return self.resampling_dir / "resampling_summary.json"

    @property
    def registration_metrics_csv(self) -> Path:
        return self.metrics_dir / "registration_metrics.csv"


def resolve_pipeline_paths(outputs_root: Path,
                           tag: str | None = None) -> PipelinePaths:
    """Output locations for a pipeline run.

    ``tag=None`` reproduces the legacy v2 archive layout exactly (used by
    the unmodified default invocation); a tag such as ``manual_v1`` moves
    every writable artefact under ``<outputs_root>/<tag>/`` so old results
    can never be overwritten.
    """
    root = Path(outputs_root)
    base = root if tag is None else root/tag
    return PipelinePaths(
        resampling_dir=base/"resampling",
        registered_h_reg_dir=base/"registered/H_reg",
        registered_h_200_dir=base/"registered/H_200",
        registered_masks_dir=base/"registered/masks",
        metrics_dir=base/"metrics",
    )
