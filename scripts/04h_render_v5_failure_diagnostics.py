#!/usr/bin/env python3
"""Render representative v5 failures without changing registration results."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.io_cag import CagHeightReader  # noqa: E402


def rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    config = yaml.safe_load(
        (REPO / "config/rectangle_registration.yaml").read_text(encoding="utf-8")
    )
    root = REPO / config["paths"]["outputs_root"]
    sessions = {r["session_id"]: r for r in rows(REPO / config["paths"]["session_manifest"])}
    planes = {(r["session_id"], int(r["measurement_id"])): r for r in rows(root / "metrics/coarse_leveling_metrics.csv")}
    versions = {
        label: {(r["session_id"], int(r["sample_id"])): r for r in rows(root / "registration" / name)}
        for label, name in (
            ("v2", "translation_metrics.csv"),
            ("v3", "translation_metrics_v3.csv"),
            ("v4", "translation_metrics_v4.csv"),
            ("v5", "translation_metrics_v5.csv"),
        )
    }
    selected = [("zro2_120_formal", sample) for sample in (6, 12, 24, 30, 100, 118)]
    figure, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    colors = {"v2": "cyan", "v3": "lime", "v4": "orange", "v5": "red"}
    reader = CagHeightReader(REPO / sessions["zro2_120_formal"]["cag_path"])
    try:
        for ax, key in zip(axes.flat, selected):
            current = versions["v5"][key]
            measurement = int(current["measurement_id"])
            hm = reader.read_height_map(measurement)
            plane = planes[(key[0], measurement)]
            x = hm.x_um-hm.width_um/2
            y = hm.y_um-hm.height_um/2
            a, b, c = (float(plane[k]) for k in ("a", "b", "c"))
            z = hm.z-(a*x[None, :]+b*y[:, None]+c)
            cx, cy = float(current["center_x_um"]), float(current["center_y_um"])
            ix = (x >= cx-150)&(x <= cx+150)
            iy = (y >= cy-150)&(y <= cy+150)
            crop = z[np.ix_(iy, ix)]
            lo, hi = np.nanpercentile(crop, [2, 98])
            ax.imshow(crop, extent=(x[ix][0], x[ix][-1], y[iy][-1], y[iy][0]),
                      cmap="viridis", vmin=lo, vmax=hi, interpolation="nearest")
            theta = float(current["theta_session_deg"])
            square = Rectangle((cx-100, cy-100), 200, 200, fill=False,
                               edgecolor=colors["v5"], linewidth=1.5)
            square.set_transform(Affine2D().rotate_deg_around(cx, cy, theta)+ax.transData)
            ax.add_patch(square)
            for label, table in versions.items():
                row = table[key]
                ax.plot(float(row["center_x_um"]), float(row["center_y_um"]),
                        marker="+", markersize=10, markeredgewidth=1.5,
                        color=colors[label], label=label)
            ax.set_title(
                f"sample {key[1]} | uCI={float(current['bootstrap_u_ci_span_um']):.1f}, "
                f"vCI={float(current['bootstrap_v_ci_span_um']):.1f}\n"
                f"evidence={float(current['joint_evidence_total']):.1f}; "
                f"u-multi={current['u_multimodal']}"
            )
            ax.set_xlabel("x (um; +right)")
            ax.set_ylabel("y (um; +down)")
            ax.legend(loc="lower right", fontsize=8, ncols=2)
    finally:
        reader.close()
    output = root / "qa/v5_failure_diagnostics.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
