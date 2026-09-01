#!/usr/bin/env python3
"""Render every v6 non-PASS sample for method-level diagnosis."""

from __future__ import annotations

import csv
import sys
from contextlib import ExitStack
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.io_cag import CagHeightReader  # noqa: E402


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    config = yaml.safe_load((REPO / "config/rectangle_registration.yaml").read_text(encoding="utf-8"))
    root = REPO / config["paths"]["outputs_root"]
    sessions = {r["session_id"]: r for r in read_csv(REPO / config["paths"]["session_manifest"])}
    planes = {(r["session_id"], int(r["measurement_id"])): r for r in read_csv(root / "metrics/coarse_leveling_metrics.csv")}
    names = {
        "v3": "translation_metrics_v3.csv",
        "v4": "translation_metrics_v4.csv",
        "v5": "translation_metrics_v5.csv",
        "v6": "translation_metrics_v6.csv",
    }
    versions = {label: {(r["session_id"], int(r["sample_id"])): r for r in read_csv(root / "registration" / name)} for label, name in names.items()}
    selected = [r for r in versions["v6"].values() if r["status"] != "PASS"]
    selected.sort(key=lambda r: (r["session_id"], int(r["sample_id"])))
    figure, axes = plt.subplots(2, 5, figsize=(22, 9), constrained_layout=True)
    colors = {"v3": "lime", "v4": "orange", "v5": "cyan", "v6": "red"}
    with ExitStack() as stack:
        readers = {sid: stack.enter_context(CagHeightReader(REPO / spec["cag_path"])) for sid, spec in sessions.items()}
        for ax, current in zip(axes.flat, selected):
            key = (current["session_id"], int(current["sample_id"]))
            measurement = int(current["measurement_id"])
            hm = readers[key[0]].read_height_map(measurement)
            x = hm.x_um-hm.width_um/2; y = hm.y_um-hm.height_um/2
            plane = planes[(key[0], measurement)]
            a,b,c=(float(plane[k]) for k in ("a","b","c"))
            z=hm.z-(a*x[None,:]+b*y[:,None]+c)
            cx,cy=float(current["center_x_um"]),float(current["center_y_um"])
            ix=(x>=cx-145)&(x<=cx+145); iy=(y>=cy-145)&(y<=cy+145)
            crop=z[np.ix_(iy,ix)]; lo,hi=np.nanpercentile(crop,[2,98])
            ax.imshow(crop,extent=(x[ix][0],x[ix][-1],y[iy][-1],y[iy][0]),cmap="viridis",vmin=lo,vmax=hi,interpolation="nearest")
            theta=float(current["theta_session_deg"])
            square=Rectangle((cx-100,cy-100),200,200,fill=False,edgecolor="red",linewidth=1.6)
            square.set_transform(Affine2D().rotate_deg_around(cx,cy,theta)+ax.transData); ax.add_patch(square)
            for label,table in versions.items():
                row=table[key]
                ax.plot(float(row["center_x_um"]),float(row["center_y_um"]),marker="+",markersize=10,markeredgewidth=1.4,color=colors[label],label=label)
            ax.set_title(f"{key[0].replace('zro2_','')} s{key[1]} | {current['status']}\nE={float(current['joint_evidence_total']):.1f}, uCI={float(current['influence_u_ci_span_um']):.1f}, vCI={float(current['influence_v_ci_span_um']):.1f}")
            ax.set_xlabel("x (um; +right)"); ax.set_ylabel("y (um; +down)")
            ax.legend(loc="lower right",fontsize=7,ncols=2)
    for ax in axes.flat[len(selected):]:
        ax.axis("off")
    output=root / "qa/v6_nonpass_diagnostics.png"
    output.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(output,dpi=160); plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
