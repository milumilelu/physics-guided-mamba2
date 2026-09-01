#!/usr/bin/env python3
"""Interactive blinded rectangle annotation for the frozen validation list."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Button, RectangleSelector

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.io_cag import CagHeightReader  # noqa: E402
from src.manual_four_edge_annotation import (  # noqa: E402
    ANNOTATION_FIELDS,
    assign_annotation_values,
    annotation_is_complete,
    canonical_box_record,
    first_incomplete_index,
    local_extents_from_record,
)
from src.resampling import resample_to_canonical  # noqa: E402


class FourEdgeAnnotator:
    def __init__(self, *, annotator: str, table_path: Path,
                 crop_um: float = 400.0) -> None:
        self.annotator = annotator.lower()
        self.prefix = f"annotator_{self.annotator}_"
        self.table_path = table_path
        self.crop_um = float(crop_um)
        self.table = pd.read_csv(table_path, keep_default_na=False)
        self._validate_table()
        for column in self.table.columns:
            if column.startswith("annotator_"):
                self.table[column] = self.table[column].astype(object)
        self.sessions = {
            row["session_id"]: row for row in pd.read_csv(
                REPO / "config/session_manifest.csv",
                keep_default_na=False
            ).to_dict("records")
        }
        planes = pd.read_csv(REPO / "config/frozen/measurement_planes_160.csv")
        self.planes = {
            (row["session_id"], int(row["measurement_id"])): row
            for row in planes.to_dict("records")
        }
        geometry = pd.read_csv(REPO / "annotations/session_geometry.csv")
        self.theta = {
            row["session_id"]: float(row["theta_session_deg"])
            for row in geometry.to_dict("records")
        }
        views = pd.read_csv(REPO / "annotations/sample_view_manifest.csv")
        self.views = {
            (row["session_id"], int(row["sample_id"])): row
            for row in views.to_dict("records")
        }
        self.index = first_incomplete_index(
            self.table.to_dict("records"), self.annotator)
        self.reader: CagHeightReader | None = None
        self.reader_session: str | None = None
        self.selector: RectangleSelector | None = None
        self.current_extents: tuple[float, float, float, float] | None = None
        self.saved_extents: tuple[float, float, float, float] | None = None
        self.display_center = (0.0, 0.0)
        self.local = None
        self.show_depth = False
        self.contrast_options = ((1, 99), (2, 98), (5, 95), (.5, 99.5))
        self.contrast_index = 1
        self.dirty = False
        self.current_comment = ""
        # Matplotlib widgets must be kept alive.  Without these references the
        # Button instances can be garbage-collected and stop receiving clicks.
        self.buttons: list[Button] = []
        self.fig, self.ax = plt.subplots(figsize=(11.8, 8.4))
        self.fig.canvas.manager.set_window_title(
            f"Four-edge blind annotation {self.annotator.upper()}")
        self.fig.subplots_adjust(left=.07, right=.97, bottom=.18, top=.91)
        self.status = self.fig.text(.07, .025, "", fontsize=9)
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
        required = {"session_id", "sample_id", "measurement_id",
                    "roi_within_measurement"}
        required.update(self.prefix+field for field in ANNOTATION_FIELDS)
        missing = sorted(required-set(self.table.columns))
        if missing:
            raise ValueError(
                "annotation table schema is stale; regenerate the empty table: "
                f"missing {missing}"
            )

    def _button(self, label: str, x: float, callback, width: float = .105) -> None:
        axis = self.fig.add_axes([x, .105, width, .042])
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

    def _reader_for(self, sid: str) -> CagHeightReader:
        if self.reader_session != sid:
            if self.reader is not None:
                self.reader.close()
            self.reader = CagHeightReader(REPO / self.sessions[sid]["cag_path"])
            self.reader_session = sid
        return self.reader

    def _display_origin(self, row: pd.Series) -> tuple[float, float]:
        if row["roi_within_measurement"] == "single":
            return 0.0, 0.0
        view = self.views[(row["session_id"], int(row["sample_id"]))]
        return (
            (float(view["center_search_x_min_um"])
             + float(view["center_search_x_max_um"]))/2,
            (float(view["center_search_y_min_um"])
             + float(view["center_search_y_max_um"]))/2,
        )

    def _annotation_record(self, row: pd.Series) -> dict:
        return {
            field: row.get(self.prefix+field, "") for field in ANNOTATION_FIELDS
        }

    def _load_current(self) -> None:
        row = self._row()
        sid = str(row["session_id"])
        measurement = int(row["measurement_id"])
        hm = self._reader_for(sid).read_height_map(measurement)
        plane_row = self.planes[(sid, measurement)]
        plane = tuple(float(plane_row[key]) for key in ("a", "b", "c"))
        self.display_center = self._display_origin(row)
        pixels = min(1000, int(np.floor(
            self.crop_um/max(hm.dx_um, hm.dy_um))))
        self.local = resample_to_canonical(
            hm, plane=plane,
            center_x_um=self.display_center[0],
            center_y_um=self.display_center[1],
            theta_deg=self.theta[sid], length_um=self.crop_um,
            pixels=pixels, minimum_mask_weight=.99, order=1,
            metadata={"purpose": "blinded_manual_four_edge_annotation"}
        )
        record = self._annotation_record(row)
        self.saved_extents = local_extents_from_record(
            record, display_center_x_um=self.display_center[0],
            display_center_y_um=self.display_center[1],
            theta_deg=self.theta[sid]
        )
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
            f"manual_annotation_draft_{self.annotator}.json")
        if not path.exists():
            return None
        try:
            draft = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        row = self._row()
        if (str(draft.get("annotator", "")).lower() == self.annotator
                and str(draft.get("session_id", "")) == str(row["session_id"])
                and int(draft.get("sample_id", -1)) == int(row["sample_id"])
                and draft.get("extents") is not None):
            return draft
        return None

    def _values(self) -> np.ndarray:
        height = self.local.z
        if not self.show_depth:
            return height
        finite = height[self.local.valid_mask]
        return float(np.median(finite))-height

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
        self.ax.set_xlabel("canonical u (um; +right)")
        self.ax.set_ylabel("canonical v (um; +down)")
        row = self._row()
        state = str(row.get(self.prefix+"state", "")) or "incomplete"
        completed = int(self.table[self.prefix+"state"].astype(str).isin(
            ["complete", "unusable"]).sum())
        self.ax.set_title(
            f"Blind annotator {self.annotator.upper()} | "
            f"completed={completed}/{len(self.table)} | row={self.index+1} | "
            f"{row['session_id']} sample {int(row['sample_id'])}\n"
            f"state={state} | drag the four visible processing boundaries"
        )
        self.selector = RectangleSelector(
            self.ax, self._on_select, useblit=True, button=[1],
            minspanx=5, minspany=5, spancoords="data", interactive=True,
            props={"facecolor": "none", "edgecolor": "red", "linewidth": 1.8},
            handle_props={"markeredgecolor": "red", "markerfacecolor": "white"},
        )
        if self.current_extents is not None:
            left, right, top, bottom = self.current_extents
            self.selector.extents = (left, right, top, bottom)
            self._set_status(
                f"box {right-left:.2f} x {bottom-top:.2f} um | "
                "S save+next; A/D navigate; C contrast; H height/depth"
            )
        else:
            self._set_status(
                "Drag one rectangle around the four physical boundaries. "
                "No automatic center/status is displayed."
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
        self._set_status(
            f"UNSAVED box {right-left:.2f} x {bottom-top:.2f} um; "
            "drag handles to refine, then S"
        )
        self._save_draft()

    def _set_status(self, text: str) -> None:
        self.status.set_text(text)
        self.fig.canvas.draw_idle()

    def _save_draft(self) -> None:
        row = self._row()
        payload = {
            "annotator": self.annotator, "session_id": row["session_id"],
            "sample_id": int(row["sample_id"]), "extents": self.current_extents,
            "comment": self.current_comment,
        }
        path = self.table_path.with_name(f"manual_annotation_draft_{self.annotator}.json")
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def _atomic_write(self) -> None:
        temporary = self.table_path.with_suffix(".csv.tmp")
        self.table.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(self.table_path)

    def _save(self, state: str = "complete") -> bool:
        if state == "complete" and self.current_extents is None:
            self._set_status("No rectangle to save. Drag a box or press U for unusable.")
            return False
        values = {field: "" for field in ANNOTATION_FIELDS}
        if state == "complete":
            left, right, top, bottom = self.current_extents
            values.update(canonical_box_record(
                left_local_um=left, right_local_um=right,
                top_local_um=top, bottom_local_um=bottom,
                display_center_x_um=self.display_center[0],
                display_center_y_um=self.display_center[1],
                theta_deg=self.theta[str(self._row()["session_id"])]
            ))
        else:
            from datetime import datetime, timezone
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
            self._set_status("Reached the end of the frozen validation list.")

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
            self.current_comment = "manual edge not reliably visible"
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
        REPO / "annotations/manual_four_edge_validation.csv"))
    parser.add_argument("--crop-um", type=float, default=400.0)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--render-smoke", type=Path,
                        help="Render the first resume sample and exit without saving annotations.")
    parser.add_argument("--self-test-save", action="store_true",
                        help="Exercise a real CAG save against an isolated temporary CSV.")
    parser.add_argument("--self-test-hotkey", action="store_true",
                        help="Exercise the Tk S shortcut against an isolated temporary CSV.")
    parser.add_argument("--review", action="store_true",
                        help="Open even when this annotator has completed every row.")
    args = parser.parse_args()
    if args.check:
        table = pd.read_csv(args.table, keep_default_na=False)
        required = {f"annotator_{args.annotator.lower()}_{field}"
                    for field in ANNOTATION_FIELDS}
        missing = sorted(required-set(table.columns))
        if missing:
            raise SystemExit(f"schema check failed: {missing}")
        print(f"annotation_ui=OK annotator={args.annotator.upper()} rows={len(table)}")
        return 0
    if args.self_test_save:
        with tempfile.TemporaryDirectory(prefix="four_edge_save_test_") as directory:
            temporary_table = Path(directory) / args.table.name
            shutil.copy2(args.table, temporary_table)
            app = FourEdgeAnnotator(
                annotator=args.annotator, table_path=temporary_table,
                crop_um=args.crop_um)
            app.current_extents = (-111.25, 92.5, -99.75, 102.25)
            if not app._save("complete"):
                raise SystemExit("isolated save self-test failed")
            saved = pd.read_csv(temporary_table, keep_default_na=False)
            prefix = f"annotator_{args.annotator.lower()}_"
            assert saved.at[0, prefix+"state"] == "complete"
            assert abs(float(saved.at[0, prefix+"left_u_um"])+111.25) < 1e-9
            app._on_close()
            plt.close(app.fig)
        print("annotation_save=OK isolated_table=true")
        return 0
    if args.self_test_hotkey:
        with tempfile.TemporaryDirectory(prefix="four_edge_hotkey_test_") as directory:
            temporary_table = Path(directory) / args.table.name
            shutil.copy2(args.table, temporary_table)
            app = FourEdgeAnnotator(
                annotator=args.annotator, table_path=temporary_table,
                crop_um=args.crop_um)
            tested_index = app.index
            app.current_extents = (-111.25, 92.5, -99.75, 102.25)
            app.dirty = True
            window = app.fig.canvas.manager.window

            def press_and_close():
                app._focus_canvas()
                window.event_generate("<KeyPress-s>")
                window.after(300, lambda: plt.close(app.fig))

            window.after(200, press_and_close)
            app.run()
            saved = pd.read_csv(temporary_table, keep_default_na=False)
            prefix = f"annotator_{args.annotator.lower()}_"
            assert saved.at[tested_index, prefix+"state"] == "complete"
        print("annotation_hotkey_s=OK isolated_table=true")
        return 0
    records = pd.read_csv(args.table, keep_default_na=False).to_dict("records")
    if annotation_is_complete(records, args.annotator) and not args.review:
        print(f"All {len(records)} samples are already complete for annotator "
              f"{args.annotator.upper()}.")
        return 0
    app = FourEdgeAnnotator(
        annotator=args.annotator, table_path=args.table,
        crop_um=args.crop_um)
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
