#!/usr/bin/env python3
"""Interactive blinded elongated-rectangle annotation of single-line ranges.

Build the table first with ``scripts/33_build_single_line_annotation_table.py``.
The view freezes the display rotation per measurement, so the machined line
appears horizontal and the annotator marks its extent with one elongated
rectangle (ends = machining range, sides = line boundaries).  No automatic
line boundary is ever displayed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Button, RectangleSelector

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.io_cag import CagHeightReader  # noqa: E402
from src.manual_four_edge_annotation import (  # noqa: E402
    annotation_is_complete,
    assign_annotation_values,
    first_incomplete_index,
    local_extents_from_record,
)
from src.manual_single_line_annotation import (  # noqa: E402
    DEFAULT_MINIMUM_ASPECT,
    RANGE_FIELDS,
    elongated_box_record,
    elongation_is_suspicious,
)
from src.resampling import resample_to_canonical  # noqa: E402

BASE_FIELDS = ("session_id", "sample_id", "measurement_id",
               "roi_within_measurement")
VIEW_REQUIRED = ("measurement_id", "cag_path", "plane_a", "plane_b", "plane_c",
                 "theta_line_deg", "crop_center_x_um", "crop_center_y_um",
                 "crop_length_um", "crop_pixels")


class SingleLineAnnotator:
    def __init__(self, *, annotator: str, table_path: Path,
                 view_manifest_path: Path,
                 minimum_aspect: float = DEFAULT_MINIMUM_ASPECT) -> None:
        self.annotator = annotator.lower()
        self.prefix = f"annotator_{self.annotator}_"
        self.table_path = table_path
        self.minimum_aspect = float(minimum_aspect)
        self.table = pd.read_csv(table_path, encoding="utf-8-sig",
                                 keep_default_na=False)
        self._validate_table()
        for column in self.table.columns:
            if column.startswith("annotator_"):
                self.table[column] = self.table[column].astype(object)
        manifest = pd.read_csv(view_manifest_path, encoding="utf-8-sig",
                               keep_default_na=False)
        missing = sorted(set(VIEW_REQUIRED)-set(manifest.columns))
        if missing:
            raise ValueError(
                "view manifest is stale; rerun the table builder: "
                f"missing {missing}")
        self.views = {int(row["measurement_id"]): row
                      for row in manifest.to_dict("records")}
        unknown = sorted(set(
            self.table["measurement_id"].astype(int))-set(self.views))
        if unknown:
            raise ValueError(f"table rows missing from view manifest: {unknown}")
        self.index = first_incomplete_index(
            self.table.to_dict("records"), self.annotator)
        self.reader: CagHeightReader | None = None
        self.reader_cag: str | None = None
        self.selector: RectangleSelector | None = None
        self.current_extents: tuple[float, float, float, float] | None = None
        self.saved_extents: tuple[float, float, float, float] | None = None
        self.local = None
        self.show_depth = False
        self.contrast_options = ((1, 99), (2, 98), (5, 95), (.5, 99.5))
        self.contrast_index = 1
        self.dirty = False
        self.current_comment = ""
        # Matplotlib widgets must be kept alive.  Without these references the
        # Button instances can be garbage-collected and stop receiving clicks.
        self.buttons: list[Button] = []
        self.fig, self.ax = plt.subplots(figsize=(12.6, 6.2))
        self.fig.canvas.manager.set_window_title(
            f"Single-line range annotation {self.annotator.upper()}")
        self.fig.subplots_adjust(left=.06, right=.97, bottom=.20, top=.88)
        self.status = self.fig.text(.06, .028, "", fontsize=9)
        self._build_controls()
        self._install_keyboard_shortcuts()
        self.fig.canvas.mpl_connect("close_event", self._on_close)
        self._load_current()

    def _install_keyboard_shortcuts(self) -> None:
        actions = {"a": self._previous, "d": self._next,
                   "s": self._save_next, "z": self._restore,
                   "r": self._clear, "u": self._unusable,
                   "c": self._contrast, "h": self._height_depth}
        try:
            window = self.fig.canvas.manager.window
        except AttributeError:
            self.fig.canvas.mpl_connect("key_press_event", self._on_key)
            return

        def invoke(event, action):
            widget_class = str(event.widget.winfo_class())
            if widget_class in {"Entry", "TEntry"}:
                return None
            action()
            return "break"

        for key, action in actions.items():
            window.bind(f"<KeyPress-{key}>",
                        lambda event, callback=action: invoke(event, callback))
            window.bind(f"<KeyPress-{key.upper()}>",
                        lambda event, callback=action: invoke(event, callback))

    def _focus_canvas(self) -> None:
        try:
            self.fig.canvas.get_tk_widget().focus_set()
        except AttributeError:
            pass

    def _validate_table(self) -> None:
        required = set(BASE_FIELDS)
        required.update(self.prefix+field for field in RANGE_FIELDS)
        missing = sorted(required-set(self.table.columns))
        if missing:
            raise ValueError(
                "annotation table schema is stale; rerun the table builder: "
                f"missing {missing}"
            )

    def _button(self, label: str, x: float, callback, width: float = .105) -> None:
        axis = self.fig.add_axes([x, .115, width, .045])
        button = Button(axis, label)
        button.on_clicked(callback)
        self.buttons.append(button)

    def _build_controls(self) -> None:
        controls = [
            ("Prev [A]", .04, self._previous),
            ("Save+Next [S]", .155, self._save_next),
            ("Next [D]", .30, self._next),
            ("Restore [Z]", .415, self._restore),
            ("Clear [R]", .53, self._clear),
            ("Unusable [U]", .645, self._unusable),
            ("Contrast [C]", .76, self._contrast),
            ("Height/Depth [H]", .875, self._height_depth),
        ]
        for label, x, callback in controls:
            self._button(label, x, callback, .105 if x != .155 else .135)

    def _row(self) -> pd.Series:
        return self.table.iloc[self.index]

    def _view(self) -> pd.Series:
        return self.views[int(self._row()["measurement_id"])]

    def _reader_for(self, cag_path: str) -> CagHeightReader:
        if self.reader_cag != cag_path:
            if self.reader is not None:
                self.reader.close()
            path = Path(cag_path)
            if not path.is_absolute():
                path = REPO/path
            self.reader = CagHeightReader(path)
            self.reader_cag = cag_path
        return self.reader

    def _annotation_record(self, row: pd.Series) -> dict:
        return {
            field: row.get(self.prefix+field, "") for field in RANGE_FIELDS
        }

    def _load_current(self) -> None:
        row = self._row()
        view = self._view()
        hm = self._reader_for(str(view["cag_path"])).read_height_map(
            int(row["measurement_id"]))
        plane = tuple(float(view[key]) for key in ("plane_a", "plane_b", "plane_c"))
        theta = float(view["theta_line_deg"])
        self.local = resample_to_canonical(
            hm, plane=plane,
            center_x_um=float(view["crop_center_x_um"]),
            center_y_um=float(view["crop_center_y_um"]),
            theta_deg=theta, length_um=float(view["crop_length_um"]),
            pixels=int(view["crop_pixels"]), minimum_mask_weight=.99, order=1,
            metadata={"purpose": "blinded_single_line_range_annotation"})
        record = self._annotation_record(row)
        self.saved_extents = local_extents_from_record(
            record, display_center_x_um=float(view["crop_center_x_um"]),
            display_center_y_um=float(view["crop_center_y_um"]),
            theta_deg=theta)
        self.current_extents = self.saved_extents
        self.current_comment = str(record.get("comment", ""))
        draft = self._draft_for_current()
        restored_draft = False
        if self.saved_extents is None and draft is not None:
            self.current_extents = tuple(float(value) for value in draft["extents"])
            self.current_comment = str(draft.get("comment", ""))
            restored_draft = True
        self.dirty = restored_draft
        self._draw()
        self._focus_canvas()
        if restored_draft:
            self._set_status(
                "Recovered your UNSAVED draft box after the previous error. "
                "Inspect it and press S to save."
            )

    def _draft_for_current(self) -> dict | None:
        path = self.table_path.with_name(
            f"single_line_annotation_draft_{self.annotator}.json")
        if not path.exists():
            return None
        try:
            draft = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        row = self._row()
        if (str(draft.get("annotator", "")).lower() == self.annotator
                and int(draft.get("measurement_id", -1))
                == int(row["measurement_id"])
                and draft.get("extents") is not None):
            return draft
        return None

    def _values(self) -> np.ndarray:
        height = self.local.z
        if not self.show_depth:
            return height
        finite = height[self.local.valid_mask]
        return float(np.median(finite))-height

    def _describe_box(self, extents: tuple[float, float, float, float]
                      ) -> tuple[str, bool]:
        left, right, top, bottom = extents
        long_axis, short_axis = sorted(
            (right-left, bottom-top), reverse=True)
        aspect = long_axis/short_axis if short_axis > 0 else float("inf")
        text = f"{long_axis:.2f} x {short_axis:.2f} um (aspect {aspect:.1f})"
        return text, aspect < self.minimum_aspect

    def _draw(self) -> None:
        if self.selector is not None:
            self.selector.set_active(False)
        self.ax.clear()
        values = self._values()
        finite = values[self.local.valid_mask]
        qlo, qhi = self.contrast_options[self.contrast_index]
        lo, hi = np.percentile(finite, (qlo, qhi))
        axis = self.local.x_um
        self.ax.imshow(
            values, extent=(axis[0], axis[-1], axis[-1], axis[0]),
            cmap="viridis", vmin=lo, vmax=hi, interpolation="nearest"
        )
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("canonical u (um; along the line after rotation)")
        self.ax.set_ylabel("canonical v (um; +down)")
        row = self._row()
        state = str(row.get(self.prefix+"state", "")) or "incomplete"
        completed = int(self.table[self.prefix+"state"].astype(str).isin(
            ["complete", "unusable"]).sum())
        self.ax.set_title(
            f"Single-line range annotator {self.annotator.upper()} | "
            f"completed={completed}/{len(self.table)} | row={self.index+1} | "
            f"{row['session_id']} sample {int(row['sample_id'])}\n"
            f"state={state} | elongated box: ends = machining range, "
            f"sides = line boundaries"
        )
        self.selector = RectangleSelector(
            self.ax, self._on_select, useblit=True, button=[1],
            minspanx=2, minspany=2, spancoords="data", interactive=True,
            props={"facecolor": "none", "edgecolor": "red", "linewidth": 1.8},
            handle_props={"markeredgecolor": "red", "markerfacecolor": "white"},
        )
        if self.current_extents is not None:
            description, suspicious = self._describe_box(self.current_extents)
            self._set_status(
                f"box {description} | "
                "S save+next; A/D navigate; C contrast; H height/depth"
                + (" | WARNING: aspect below "
                   f"{self.minimum_aspect:g}, check the box" if suspicious else "")
            )
        else:
            self._set_status(
                "Drag one elongated rectangle along the machined line. "
                "Cover the continuous machining range; ignore isolated spatter."
            )
        self.fig.canvas.draw_idle()

    def _on_select(self, click, release) -> None:
        if None in (click.xdata, click.ydata, release.xdata, release.ydata):
            return
        left, right = sorted((float(click.xdata), float(release.xdata)))
        top, bottom = sorted((float(click.ydata), float(release.ydata)))
        self.current_extents = (left, right, top, bottom)
        self.dirty = True
        self._focus_canvas()
        description, suspicious = self._describe_box(self.current_extents)
        self._set_status(
            f"UNSAVED box {description}; "
            "drag handles to refine, then S"
            + (" | WARNING: too square for a single line" if suspicious else "")
        )
        self._save_draft()

    def _set_status(self, text: str) -> None:
        self.status.set_text(text)
        self.fig.canvas.draw_idle()

    def _save_draft(self) -> None:
        row = self._row()
        payload = {
            "annotator": self.annotator,
            "measurement_id": int(row["measurement_id"]),
            "extents": self.current_extents,
            "comment": self.current_comment,
        }
        path = self.table_path.with_name(
            f"single_line_annotation_draft_{self.annotator}.json")
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def _atomic_write(self) -> None:
        temporary = self.table_path.with_suffix(".csv.tmp")
        self.table.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(self.table_path)

    def _save(self, state: str = "complete") -> bool:
        if state == "complete" and self.current_extents is None:
            self._set_status("No rectangle to save. Drag a box or press U for unusable.")
            return False
        values = {field: "" for field in RANGE_FIELDS}
        if state == "complete":
            view = self._view()
            left, right, top, bottom = self.current_extents
            values.update(elongated_box_record(
                left_local_um=left, right_local_um=right,
                top_local_um=top, bottom_local_um=bottom,
                display_center_x_um=float(view["crop_center_x_um"]),
                display_center_y_um=float(view["crop_center_y_um"]),
                theta_deg=float(view["theta_line_deg"])
            ))
        else:
            values.update({"state": "unusable",
                           "timestamp_utc": datetime.now(timezone.utc).isoformat()})
        values["comment"] = self.current_comment
        assign_annotation_values(self.table, self.index, self.prefix, values)
        try:
            self._atomic_write()
        except PermissionError:
            self._set_status("Save failed: close the CSV in Excel and press S again.")
            return False
        self.saved_extents = self.current_extents if state == "complete" else None
        self.dirty = False
        return True

    def _move(self, step: int) -> None:
        if self.dirty:
            self._set_status("Current box is unsaved. Press S, Z, or R before navigating.")
            return
        target = self.index+step
        if 0 <= target < len(self.table):
            self.index = target
            self._load_current()
        else:
            self._set_status("Reached the end of the single-line table.")

    def _advance_to_next_incomplete(self) -> None:
        states = self.table[self.prefix+"state"].astype(str)
        incomplete = ~states.isin(["complete", "unusable"])
        candidates = np.flatnonzero(incomplete.to_numpy())
        later = candidates[candidates > self.index]
        if later.size:
            self.index = int(later[0])
            self._load_current()
            return
        if candidates.size:
            self.index = int(candidates[0])
            self._load_current()
            return
        self._set_status("All samples are saved. Closing annotation window.")
        self.fig.canvas.draw_idle()
        plt.close(self.fig)

    def _previous(self, _event=None) -> None:
        self._move(-1)

    def _next(self, _event=None) -> None:
        self._move(1)

    def _save_next(self, _event=None) -> None:
        if self._save("complete"):
            self._advance_to_next_incomplete()

    def _unusable(self, _event=None) -> None:
        if not self.current_comment:
            self.current_comment = "machined line range not reliably visible"
        if self._save("unusable"):
            self._advance_to_next_incomplete()

    def _restore(self, _event=None) -> None:
        self.current_extents = self.saved_extents
        self.dirty = False
        self._draw()

    def _clear(self, _event=None) -> None:
        self.current_extents = None
        self.dirty = True
        self._draw()
        self._set_status("UNSAVED: rectangle cleared. Drag a new box or press Z.")

    def _contrast(self, _event=None) -> None:
        self.contrast_index = (self.contrast_index+1) % len(self.contrast_options)
        self._draw()

    def _height_depth(self, _event=None) -> None:
        self.show_depth = not self.show_depth
        self._draw()

    def _on_key(self, event) -> None:
        key = (event.key or "").lower()
        actions = {"a": self._previous, "d": self._next,
                   "s": self._save_next, "z": self._restore,
                   "r": self._clear, "u": self._unusable,
                   "c": self._contrast, "h": self._height_depth}
        if key in actions:
            actions[key]()

    def _on_close(self, _event=None) -> None:
        if self.reader is not None:
            self.reader.close()

    def run(self) -> None:
        plt.show()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator", required=True, choices=("A", "B", "a", "b"))
    parser.add_argument("--table", type=Path, default=(
        REPO/"annotations/single_line_range_annotation.csv"))
    parser.add_argument("--view-manifest", type=Path, default=(
        REPO/"annotations/single_line_view_manifest.csv"))
    parser.add_argument("--min-aspect-warning", type=float,
                        default=DEFAULT_MINIMUM_ASPECT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--render-smoke", type=Path,
                        help="Render the first resume sample and exit without saving annotations.")
    parser.add_argument("--self-test-save", action="store_true",
                        help="Exercise a real CAG save against an isolated temporary CSV.")
    parser.add_argument("--review", action="store_true",
                        help="Open even when this annotator has completed every row.")
    args = parser.parse_args()
    if args.check:
        table = pd.read_csv(args.table, encoding="utf-8-sig",
                            keep_default_na=False)
        required = set(BASE_FIELDS)
        required.update(
            f"annotator_{args.annotator.lower()}_{field}"
            for field in RANGE_FIELDS)
        missing = sorted(required-set(table.columns))
        if missing:
            raise SystemExit(f"annotation schema check failed: {missing}")
        manifest = pd.read_csv(args.view_manifest, encoding="utf-8-sig",
                               keep_default_na=False)
        missing = sorted(set(VIEW_REQUIRED)-set(manifest.columns))
        if missing:
            raise SystemExit(f"view manifest check failed: {missing}")
        print(f"annotation_ui=OK annotator={args.annotator.upper()} "
              f"rows={len(table)} views={len(manifest)}")
        return 0
    if args.self_test_save:
        with tempfile.TemporaryDirectory(prefix="single_line_save_test_") as directory:
            temporary_table = Path(directory) / args.table.name
            shutil.copy2(args.table, temporary_table)
            shutil.copy2(args.view_manifest, Path(directory) / args.view_manifest.name)
            app = SingleLineAnnotator(
                annotator=args.annotator, table_path=temporary_table,
                view_manifest_path=Path(directory) / args.view_manifest.name,
                minimum_aspect=args.min_aspect_warning)
            app.current_extents = (-100.0, 100.0, -6.0, 6.0)
            if not app._save("complete"):
                raise SystemExit("isolated save self-test failed")
            saved = pd.read_csv(temporary_table, encoding="utf-8-sig",
                                keep_default_na=False)
            prefix = f"annotator_{args.annotator.lower()}_"
            assert saved.at[app.index, prefix+"state"] == "complete"
            assert abs(float(saved.at[app.index, prefix+"long_axis_um"])-200.0) < 1e-9
            assert abs(float(saved.at[app.index, prefix+"short_axis_um"])-12.0) < 1e-9
            assert abs(float(saved.at[app.index, prefix+"aspect_ratio"])-200.0/12.0) < 1e-9
            app._on_close()
            plt.close(app.fig)
        print("annotation_save=OK isolated_table=true")
        return 0
    records = pd.read_csv(args.table, encoding="utf-8-sig",
                          keep_default_na=False).to_dict("records")
    if annotation_is_complete(records, args.annotator) and not args.review:
        print(f"All {len(records)} samples are already complete for annotator "
              f"{args.annotator.upper()}.")
        return 0
    app = SingleLineAnnotator(
        annotator=args.annotator, table_path=args.table,
        view_manifest_path=args.view_manifest,
        minimum_aspect=args.min_aspect_warning)
    if args.render_smoke:
        args.render_smoke.parent.mkdir(parents=True, exist_ok=True)
        app.fig.savefig(args.render_smoke, dpi=140)
        app._on_close()
        plt.close(app.fig)
        print(args.render_smoke)
        return 0
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
