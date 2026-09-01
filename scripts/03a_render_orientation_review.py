#!/usr/bin/env python3
"""Render representative raw-axis views for the WP7 D4 manual gate."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.io_cag import CagHeightReader  # noqa: E402


def rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def rectangle(center_x: float, center_y: float, width: float, height: float,
              theta_deg: float) -> np.ndarray:
    corners = np.array([
        [-width/2, -height/2], [width/2, -height/2],
        [width/2, height/2], [-width/2, height/2],
        [-width/2, -height/2],
    ])
    theta = np.deg2rad(theta_deg)
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    return corners @ rotation.T + [center_x, center_y]


def main() -> int:
    config = yaml.safe_load((REPO / "config/rectangle_registration.yaml")
                            .read_text(encoding="utf-8"))
    root = REPO / "outputs/rectangle_registration"
    sessions = rows(REPO / config["paths"]["session_manifest"])
    fits = rows(root / "geometry/theta_sample_distribution.csv")
    planes = rows(root / "metrics/coarse_leveling_metrics.csv")
    plane_by_key = {(row["session_id"], int(row["measurement_id"])): row
                    for row in planes}
    figure, axes = plt.subplots(1, len(sessions), figsize=(15, 5),
                                constrained_layout=True)
    for axis, session in zip(axes, sessions):
        sid = session["session_id"]
        fit = max((row for row in fits if row["session_id"] == sid
                   and row["status"] == "PASS"),
                  key=lambda row: float(row["quality_score"]))
        measurement_id = int(fit["measurement_id"])
        plane = plane_by_key[(sid, measurement_id)]
        with CagHeightReader(REPO / session["cag_path"]) as reader:
            hm = reader.read_height_map(measurement_id)
        x = hm.x_um - hm.width_um / 2.0
        y = hm.y_um - hm.height_um / 2.0
        a, b, c = (float(plane[name]) for name in ("a", "b", "c"))
        z = hm.z - (a*x[None, :] + b*y[:, None] + c)
        cx, cy = float(fit["center_x_um"]), float(fit["center_y_um"])
        columns = (x >= cx-145) & (x <= cx+145)
        selected_rows = (y >= cy-145) & (y <= cy+145)
        crop = z[np.ix_(selected_rows, columns)]
        valid = hm.valid_mask[np.ix_(selected_rows, columns)]
        vmin, vmax = np.quantile(crop[valid], [0.01, 0.99])
        image = axis.imshow(
            np.where(valid, crop, np.nan), cmap="viridis", vmin=vmin, vmax=vmax,
            origin="upper", extent=[x[columns][0], x[columns][-1],
                                    y[selected_rows][-1], y[selected_rows][0]])
        box = rectangle(cx, cy, float(fit["width_um"]),
                        float(fit["height_um"]), float(fit["theta_deg"]))
        axis.plot(box[:, 0], box[:, 1], color="white", linewidth=1.4)
        axis.set_title(f"{sid}\nsample {fit['sample_id']}, theta={float(fit['theta_deg']):.2f} deg")
        axis.set_xlabel("raw +X → (image right), µm")
        axis.set_ylabel("raw +Y → (image down), µm")
        axis.annotate("+X", xy=(0.25, 0.08), xytext=(0.08, 0.08),
                      xycoords="axes fraction", color="red",
                      arrowprops={"arrowstyle": "->", "color": "red"})
        axis.annotate("+Y", xy=(0.08, 0.08), xytext=(0.08, 0.25),
                      xycoords="axes fraction", color="red",
                      arrowprops={"arrowstyle": "->", "color": "red"})
        figure.colorbar(image, ax=axis, shrink=0.72, label="coarse-levelled height (µm)")
    figure.suptitle("WP7 D4 review — diagnostic only, not canonical H_reg", fontsize=13)
    output = root / "geometry/orientation_review.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)

    review = (
        "# WP7 D4 人工确认\n\n"
        "本图仅用于方向复核，不是 canonical `H_reg`。原始坐标定义为图像向右是 "
        "+X、向下是 +Y。\n\n"
        "请针对每个 session 记录：canonical +y（长扫描方向）在原图中指向上/下/左/右，"
        "canonical +x（相邻扫描线/hatch 方向）在原图中指向何方，以及依据（显微镜方向、"
        "加工路径或实验记录）。不得用‘哪边更深’决定方向。\n\n"
        "连续角度已经独立估计；D4 未确认前脚本不会输出 `H_reg`。\n"
    )
    (root / "geometry/ORIENTATION_REVIEW.md").write_text(review, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
