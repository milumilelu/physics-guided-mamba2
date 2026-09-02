#!/usr/bin/env python3
"""Phase 1 minimal exploration: depth-shape decomposition and basic morphology.

Implements 任务说明/第一阶段探索.md strictly as a descriptive study:
    manifest -> atlas -> depth -> absolute PCA -> residual PCA
    -> cluster bootstrap stability -> raw/repaired sensitivity
    -> repeatability sentinel (DOE 49/50) -> N=1..4 trajectory atlas.

Descriptive only: no regression, no prediction, no causal claims.
All outputs land in outputs/phase1_minimal/ (fixed 14-file list).
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parents[1]

SESSION_ROLES = {
    "zro2_120_formal": "formal",
    "zro2_60_pass": "pass_main",
    "zro2_20_supplement": "pass_supplement",
}
PAIRED_SESSIONS = ("zro2_60_pass", "zro2_20_supplement")
FORMAL_SESSION = "zro2_120_formal"

ROLE_COLORS = {
    "formal": "tab:blue",
    "pass_main": "tab:orange",
    "pass_supplement": "tab:green",
}

MANIFEST_COLUMNS = [
    "dataset_index", "session_id", "measurement_id", "sample_id",
    "shared_height_source_id", "roi_within_measurement", "session_role",
    "design_group", "processing_order", "x_position_um", "y_position_um",
    "compressor_steps", "pulse_duration_fs",
    "pulse_duration_calibration_id", "pulse_duration_calibration_version",
    "frequency_kHz", "hatch_spacing_um", "pass_count", "velocity_mm_s",
    "valid_fraction", "repair_fraction",
    "median_depth_um", "residual_Sq_um",
]
DEPTH_COLUMNS = ("median_depth_um", "residual_Sq_um")
IDENTITY_COLUMNS = [c for c in MANIFEST_COLUMNS if c not in DEPTH_COLUMNS]

EXPECTED_OUTPUTS = [
    "exploration_manifest.csv",
    "dataset_atlas_global_scale.png",
    "dataset_atlas_individual_scale.png",
    "depth_distribution.png",
    "depth_vs_process.png",
    "absolute_pca_evr.png",
    "residual_pca_evr.png",
    "absolute_pca_modes.png",
    "residual_pca_modes.png",
    "pca_reconstruction_rmse.png",
    "bootstrap_pca_stability.csv",
    "raw_repaired_sensitivity.csv",
    "repeatability_sentinel_49_50.png",
    "pass_trajectory_N1_N4.png",
]


def log(message: str = "") -> None:
    print(message, flush=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(f"HARD ASSERTION FAILED: {message}")


def read_csv_smart(path: Path, **kwargs) -> pd.DataFrame:
    """utf-8 first, gb18030 fallback (design sheets are GBK-encoded)."""
    try:
        return pd.read_csv(path, encoding="utf-8", **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gb18030", **kwargs)


# --------------------------------------------------------------------------- #
# Step 1: exploration manifest
# --------------------------------------------------------------------------- #

def load_design_table(cfg: dict, session_id: str) -> pd.DataFrame:
    path = REPO / cfg["design_tables"][session_id]
    encoding = cfg["design_tables"]["encoding"]
    raw = pd.read_csv(path, encoding=encoding)
    expected = ["加工顺序", "中心X", "中心Y", "脉宽", "频率", "线间距", "次数", "速度"]
    require(list(raw.columns) == expected,
            f"{session_id} design columns {list(raw.columns)} != {expected}")
    d = raw.rename(columns={
        "加工顺序": "processing_order", "中心X": "center_x_mm",
        "中心Y": "center_y_mm", "脉宽": "compressor_steps",
        "频率": "frequency_kHz", "线间距": "hatch_spacing_mm",
        "次数": "pass_count", "速度": "velocity_mm_s"})
    for col in ("processing_order", "compressor_steps", "frequency_kHz",
                "pass_count", "velocity_mm_s"):
        d[col] = d[col].astype(int)
    d["x_position_um"] = (d["center_x_mm"] * 1000.0).round(4)
    d["y_position_um"] = (d["center_y_mm"] * 1000.0).round(4)
    d["hatch_spacing_um"] = (d["hatch_spacing_mm"] * 1000.0).round(4)
    return d


def combo_key(row) -> tuple:
    return (int(row["compressor_steps"]), int(row["frequency_kHz"]),
            round(float(row["hatch_spacing_um"]), 3),
            int(row["velocity_mm_s"]))


def build_manifest(cfg: dict) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    log("== Step 1: exploration manifest ==")
    index = read_csv_smart(REPO / cfg["paths"]["dataset_index_csv"])
    require(len(index) == 200, f"index rows {len(index)} != 200 (A1)")
    require(not index.duplicated(["session_id", "sample_id"]).any(),
            "(session_id, sample_id) not unique (A1)")
    require((index["status"] == "PASS").all(), "non-PASS rows in index (A5)")
    require((index["valid_fraction"] >= 0.99).all(),
            "valid_fraction < 0.99 in index (A5)")

    views = read_csv_smart(REPO / cfg["paths"]["sample_view_manifest"])
    key = ["session_id", "measurement_id", "sample_id"]
    man = index.merge(views[key + ["shared_height_source_id",
                                   "roi_within_measurement"]],
                      on=key, how="left", validate="one_to_one")
    require(man["shared_height_source_id"].notna().all(),
            "view manifest join incomplete (A2)")
    n_sources = man["shared_height_source_id"].nunique()
    require(n_sources == 160, f"unique shared sources {n_sources} != 160 (A2)")
    log(f"  index 200 rows, join 200/200, unique shared sources = {n_sources}")

    sessions = read_csv_smart(REPO / cfg["paths"]["session_manifest"])
    require(set(sessions["session_id"]) == set(SESSION_ROLES),
            "session manifest sessions mismatch")

    designs: dict[str, pd.DataFrame] = {}
    frames = []
    for session_id, srow in sessions.set_index("session_id").iterrows():
        d = load_design_table(cfg, session_id)
        designs[session_id] = d
        expected_rows = int(srow["expected_design_rows"])
        require(len(d) == expected_rows,
                f"{session_id} design rows {len(d)} != {expected_rows} (A3)")

        steps = set(d["compressor_steps"])
        calib = cfg["pulse_duration_calibration"]["table_fs_to_steps"]
        steps_to_fs = {int(v): int(k) for k, v in calib.items()}
        missing = steps - set(steps_to_fs)
        require(not missing,
                f"{session_id} compressor_steps not in calibration: {missing} (A3)")
        d = d.copy()
        d["pulse_duration_fs"] = d["compressor_steps"].map(steps_to_fs).astype(int)

        role = SESSION_ROLES[session_id]
        if session_id == FORMAL_SESSION:
            require(srow["mapping_rule"] == "one_to_one_measurement_id",
                    "formal mapping_rule unexpected")
            require((d["processing_order"].to_numpy()
                     == np.arange(1, len(d) + 1)).all(),
                    "formal processing_order not 1..N sequence")
            d["design_group"] = ""
        else:
            require(srow["mapping_rule"] == "paired_slot_from_cag_data_name",
                    f"{session_id} mapping_rule unexpected")
            require(srow["rois_per_measurement"] == 2,
                    f"{session_id} expected 2 ROIs per measurement")
            combos: dict[tuple, list[int]] = {}
            for _, row in d.iterrows():
                combos.setdefault(combo_key(row), []).append(int(row["pass_count"]))
            ordered = sorted(combos)
            prefix = "T" if role == "pass_main" else "S"
            group_of = {c: f"{prefix}{i + 1:02d}" for i, c in enumerate(ordered)}
            d = d.copy()
            d["design_group"] = [group_of[combo_key(r)] for _, r in d.iterrows()]

            for m_id, grp in man[man["session_id"] == session_id].groupby(
                    "measurement_id"):
                m_id = int(m_id)
                sids = sorted(grp["sample_id"].astype(int).tolist())
                require(sids == [2 * m_id - 1, 2 * m_id],
                        f"{session_id} m{m_id:03d} samples {sids} != "
                        f"{[2 * m_id - 1, 2 * m_id]} (A4)")
                slots = set(grp["roi_within_measurement"])
                require(slots == {"slot_1", "slot_2"},
                        f"{session_id} m{m_id:03d} slot labels {slots} (A4)")
                r1 = d[d["processing_order"] == 2 * m_id - 1].iloc[0]
                r2 = d[d["processing_order"] == 2 * m_id].iloc[0]
                require(combo_key(r1) == combo_key(r2),
                        f"{session_id} m{m_id:03d} paired design rows differ in "
                        f"base condition (A4)")
                require(int(r1["pass_count"]) != int(r2["pass_count"]),
                        f"{session_id} m{m_id:03d} paired design rows share "
                        f"pass_count (A4)")
            if role == "pass_supplement":
                passes = set(d["pass_count"])
                require(passes <= {5, 6},
                        f"supplement pass values {passes} outside {{5,6}} (A4)")

            n_groups = d["design_group"].nunique()
            group_passes = d.groupby("design_group")["pass_count"].apply(
                lambda s: set(int(x) for x in s))
            if role == "pass_main":
                require(n_groups == 15,
                        f"60-pass trajectories {n_groups} != 15 (A7)")
                require(all(p == {1, 2, 3, 4} for p in group_passes),
                        "60-pass trajectory pass sets != {1,2,3,4} (A7)")
                log(f"  60-pass: {n_groups} trajectories x pass {{1,2,3,4}} (A7)")
            else:
                require(n_groups == 10,
                        f"supplement groups {n_groups} != 10 (A4)")
                require(all(p == {5, 6} for p in group_passes),
                        "supplement group pass sets != {5,6} (A4)")

        sel_man = man[man["session_id"] == session_id]
        rows = sel_man.merge(d, left_on="sample_id",
                             right_on="processing_order",
                             how="left", validate="one_to_one")
        require(len(rows) == len(sel_man),
                f"{session_id} design join row count mismatch (A2)")
        require(rows["processing_order"].notna().all(),
                f"{session_id} sample_id without design row")
        frames.append(rows)

    man = pd.concat(frames, ignore_index=True)
    man = man.sort_values("dataset_index").reset_index(drop=True)
    require(len(man) == 200, f"manifest rows {len(man)} != 200")
    man["session_role"] = man["session_id"].map(SESSION_ROLES)
    man["pulse_duration_calibration_id"] = cfg["pulse_duration_calibration"]["id"]
    man["pulse_duration_calibration_version"] = \
        cfg["pulse_duration_calibration"]["version"]

    sent = cfg["sentinel"]
    d_formal = designs[sent["session"]]
    a = d_formal[d_formal["processing_order"] == sent["processing_orders"][0]].iloc[0]
    b = d_formal[d_formal["processing_order"] == sent["processing_orders"][1]].iloc[0]
    fields = ["compressor_steps", "frequency_kHz", "hatch_spacing_um",
              "pass_count", "velocity_mm_s"]
    require(all(a[f] == b[f] for f in fields),
            "sentinel 49/50 design rows differ (A6)")
    for f, v in sent["expected_combo"].items():
        require(float(a[f]) == float(v),
                f"sentinel row 49 field {f}={a[f]} != expected {v} (A6)")
    log(f"  sentinel 49/50 identical on {fields} (A6): "
        f"{int(a['compressor_steps'])} steps, {int(a['frequency_kHz'])} kHz, "
        f"{a['hatch_spacing_um']} um hatch, pass {int(a['pass_count'])}, "
        f"{int(a['velocity_mm_s'])} mm/s")

    man = man[IDENTITY_COLUMNS]
    return man, designs


# --------------------------------------------------------------------------- #
# Step 2: data assembly
# --------------------------------------------------------------------------- #

def load_heights(cfg: dict) -> dict:
    log("== Step 2: data assembly ==")
    npz_path = REPO / cfg["paths"]["dataset_npz"]
    data = np.load(npz_path)
    for k in ("height_raw", "height_repaired", "valid_mask", "repair_mask",
              "session_id", "sample_id"):
        require(k in data.files, f"NPZ missing array {k}")
    require(data["height_raw"].shape == (200, 160, 160),
            f"height_raw shape {data['height_raw'].shape} != (200,160,160)")
    require(data["height_repaired"].shape == (200, 160, 160),
            "height_repaired shape mismatch")
    require(data["valid_mask"].shape == (200, 160, 160),
            "valid_mask shape mismatch")

    H = data["height_raw"].astype(np.float64)
    Hrep = data["height_repaired"].astype(np.float64)
    V = data["valid_mask"].astype(bool)

    for name, arr in (("height_raw", H), ("height_repaired", Hrep)):
        bad_inside = int(np.count_nonzero(~np.isfinite(arr[V])))
        require(bad_inside == 0, f"{name}: {bad_inside} non-finite valid pixels")
        bad_outside = int(np.count_nonzero(np.isfinite(arr[~V])))
        log(f"  {name}: non-finite valid pixels = 0, "
            f"filled pixels outside mask = {bad_outside} (treated as NaN)")
    require(np.array_equal(data["valid_mask"], V),
            "valid_mask inconsistent")
    log(f"  valid fraction per ROI: min={V.reshape(200, -1).mean(1).min():.4f}, "
        f"mean={V.mean():.4f}")

    # NaN-filled views (contract: mask-out pixels are never trusted).
    Hnan = np.where(V, H, np.nan)
    Hrep_nan = np.where(V, Hrep, np.nan)

    common = V.reshape(200, -1).all(axis=0)
    coverage = float(common.mean())
    min_cov = float(cfg["analysis"]["common_mask_min_coverage"])
    cols = None
    if coverage >= min_cov:
        mode = "restricted"
        cols = np.flatnonzero(common)
        log(f"  analysis mask: common-valid pixels = {int(common.sum())}/25600 "
            f"({coverage:.4f} >= {min_cov}); PCA restricted to common mask")
    else:
        mode = "imputed"
        log(f"  analysis mask: common coverage {coverage:.4f} < {min_cov}; "
            "per-pixel mean imputation over valid samples")
    return {
        "H3d": H, "Hrep3d": Hrep, "V3d": V,
        "Hnan": Hnan, "Hrep_nan": Hrep_nan,
        "npz_session_id": data["session_id"].astype(str),
        "npz_sample_id": data["sample_id"].astype(np.int64),
        "mode": mode, "common": common, "cols": cols,
    }


def assemble_matrices(bundle: dict, man: pd.DataFrame, cfg: dict) -> dict:
    """Build analysis-basis matrices for raw and repaired heights."""
    require((bundle["npz_session_id"] == man["session_id"].to_numpy()).all()
            and (bundle["npz_sample_id"]
                 == man["sample_id"].to_numpy(np.int64)).all(),
            "NPZ row order does not match manifest (session_id, sample_id)")
    log("  NPZ row order == manifest order (session_id, sample_id) verified")

    Hnan, Hrep_nan = bundle["Hnan"], bundle["Hrep_nan"]

    def basis(mat_nan: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mat2d = to_2d(mat_nan)
        if bundle["mode"] == "restricted":
            cols = bundle["common"]
            X = mat2d[:, cols]
            require(np.isfinite(X).all(), "NaN inside analysis mask")
            return X, cols
        # imputed branch: fill NaN with per-pixel mean over valid samples
        counts = np.sum(np.isfinite(mat2d), axis=0)
        sums = np.nansum(mat2d, axis=0)
        with np.errstate(invalid="ignore"):
            fill = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
        X = np.where(np.isfinite(mat2d), mat2d, fill[None, :])
        return X, np.ones(mat2d.shape[1], dtype=bool)

    X_raw, cols_raw = basis(Hnan)
    X_rep, cols_rep = basis(Hrep_nan)
    require(np.array_equal(cols_raw, cols_rep),
            "raw and repaired analysis masks differ")
    log(f"  analysis matrix: {X_raw.shape[0]} x {X_raw.shape[1]} "
        f"(mode={bundle['mode']})")
    bundle["X_raw"], bundle["X_rep"], bundle["cols"] = X_raw, X_rep, cols_raw
    return bundle


def residualize(Hnan: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R = H - per-sample valid-median (NaN-aware, keeps 3-D grid shape)."""
    med = np.nanmedian(Hnan, axis=(1, 2))
    R = Hnan - med[:, None, None]
    return R, med


def to_2d(mat: np.ndarray) -> np.ndarray:
    return mat.reshape(mat.shape[0], -1)


# --------------------------------------------------------------------------- #
# Step 3: data atlas
# --------------------------------------------------------------------------- #

def _masked(img: np.ndarray, valid: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_where(~valid, img)


def plot_atlas(bundle: dict, man: pd.DataFrame, out_dir: Path, cfg: dict,
               global_scale: bool) -> None:
    H3d, V3d = bundle["H3d"], bundle["V3d"]
    dpi = int(cfg["plot"]["dpi"])
    name = ("dataset_atlas_global_scale.png" if global_scale
            else "dataset_atlas_individual_scale.png")
    if global_scale:
        vals = H3d[V3d]
        vmin, vmax = np.percentile(vals, 1), np.percentile(vals, 99)
    fig, axes = plt.subplots(10, 20, figsize=(17.0, 9.4), dpi=dpi)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.82")
    order = man["dataset_index"].to_numpy()
    roles = man["session_role"].to_numpy()
    for k, ax in enumerate(axes.ravel()):
        i = order[k]
        img = _masked(H3d[i], V3d[i])
        if global_scale:
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            finite = img.compressed()
            if finite.size:
                lo, hi = np.percentile(finite, 1), np.percentile(finite, 99)
                if hi <= lo:
                    hi = lo + 1e-9
                ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi)
            else:
                ax.imshow(np.zeros((4, 4)), cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0.03, 0.90, str(int(i)), transform=ax.transAxes,
                fontsize=4.5, color="black", alpha=0.75)
    # session row labels on the left margin
    bounds = []
    start = 0
    for k in range(1, 201):
        if k == 200 or roles[order[k]] != roles[order[start]]:
            bounds.append((start, k, roles[order[start]]))
            start = k
    for lo, hi, role in bounds:
        r0, r1 = lo // 20, (hi - 1) // 20
        axes[(r0 + r1) // 2, 0].text(
            -0.35, 0.5, role, transform=axes[(r0 + r1) // 2, 0].transAxes,
            rotation=90, va="center", ha="center", fontsize=7)
    scale = "shared colour scale (1-99 pct, um)" if global_scale \
        else "per-image colour scale (1-99 pct, um)"
    fig.suptitle(f"200 ROI height maps (height_raw, 80x80 um) - {scale}",
                 fontsize=11)
    fig.subplots_adjust(wspace=0.02, hspace=0.04, left=0.035, right=0.99,
                        top=0.955, bottom=0.01)
    fig.savefig(out_dir / name, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {name}")


# --------------------------------------------------------------------------- #
# Step 4: depth
# --------------------------------------------------------------------------- #

def depth_stats(bundle: dict, man: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    R3, med = residualize(bundle["Hnan"])
    with np.errstate(invalid="ignore"):
        sq = np.sqrt(np.nanmean(R3 ** 2, axis=(1, 2)))
    require(np.isfinite(med).all() and np.isfinite(sq).all(),
            "non-finite depth/Sq statistics")
    bundle["R3_full"] = R3
    return -med, sq


def plot_depth_figs(bundle: dict, man: pd.DataFrame, out_dir: Path,
                    cfg: dict) -> None:
    dpi = int(cfg["plot"]["dpi"])
    depth = man["median_depth_um"].to_numpy()
    sq = man["residual_Sq_um"].to_numpy()
    roles = man["session_role"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.6), dpi=dpi)
    axes[0, 0].hist(depth, bins=40, color="tab:blue", alpha=0.8)
    axes[0, 0].set_xlabel("median depth  d_i = -median(H_i)  [um]")
    axes[0, 0].set_ylabel("ROIs")
    axes[0, 0].set_title("Depth distribution (all 200 ROIs)")
    data_by_role = [depth[roles == r] for r in
                    ("formal", "pass_main", "pass_supplement")]
    axes[0, 1].boxplot(data_by_role, tick_labels=["formal\n(120)", "pass N1-4\n(60)",
                                                  "pass N5-6\n(20)"],
                       showmeans=True)
    axes[0, 1].set_ylabel("median depth [um]")
    axes[0, 1].set_title("Depth by session")
    axes[1, 0].hist(sq, bins=40, color="tab:red", alpha=0.8)
    axes[1, 0].set_xlabel("residual Sq  [um]")
    axes[1, 0].set_ylabel("ROIs")
    axes[1, 0].set_title("Residual RMS roughness distribution")
    sq_by_role = [sq[roles == r] for r in
                  ("formal", "pass_main", "pass_supplement")]
    axes[1, 1].boxplot(sq_by_role, tick_labels=["formal\n(120)", "pass N1-4\n(60)",
                                                "pass N5-6\n(20)"],
                       showmeans=True)
    axes[1, 1].set_ylabel("residual Sq [um]")
    axes[1, 1].set_title("Residual Sq by session")
    fig.tight_layout()
    fig.savefig(out_dir / "depth_distribution.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log("  wrote depth_distribution.png")

    fig, axes = plt.subplots(1, 5, figsize=(21.0, 4.6), dpi=dpi)
    proc_cols = ["pulse_duration_fs", "frequency_kHz", "hatch_spacing_um",
                 "velocity_mm_s", "pass_count"]
    for ax, col in zip(axes, proc_cols):
        for role in ("formal", "pass_main", "pass_supplement"):
            sel = roles == role
            ax.scatter(man.loc[sel, col], depth[sel], s=16, alpha=0.65,
                       color=ROLE_COLORS[role], label=role, edgecolors="none")
        ax.set_xlabel(col)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("median depth [um]")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Depth vs process parameters (descriptive scatter, no fit)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "depth_vs_process.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log("  wrote depth_vs_process.png")


# --------------------------------------------------------------------------- #
# Steps 5-6: PCA (absolute + residual)
# --------------------------------------------------------------------------- #

def plot_evr(pca: PCA, path: Path, title: str, dpi: int) -> None:
    evr = pca.explained_variance_ratio_[:10]
    ks = np.arange(1, len(evr) + 1)
    fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=dpi)
    ax.bar(ks, evr * 100, color="tab:blue", alpha=0.75, label="EVR")
    ax.plot(ks, np.cumsum(evr) * 100, "o-", color="tab:red", ms=4,
            label="cumulative")
    for k, v in zip(ks, evr):
        ax.text(k, v * 100 + 1.0, f"{v * 100:.1f}", ha="center", fontsize=7)
    ax.set_xticks(ks)
    ax.set_xlabel("principal component")
    ax.set_ylabel("explained variance ratio [%]")
    ax.set_ylim(0, 105)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def to_mode_image(comp: np.ndarray, cols: np.ndarray | None) -> np.ndarray:
    if cols is None:
        full = comp.copy()
    else:
        full = np.full(25600, np.nan)
        full[cols] = comp
    img = full.reshape(160, 160)
    idx = np.nanargmax(np.abs(img))
    if img.flat[idx] < 0:
        img = -img
    return img


def plot_modes(pca: PCA, cols: np.ndarray | None, path: Path, title: str,
               dpi: int, cmap_name: str) -> None:
    comps = pca.components_[:6]
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 9.0), dpi=dpi)
    for k, ax in enumerate(axes.ravel()):
        img = to_mode_image(comps[k], cols)
        vmax = np.nanmax(np.abs(img))
        im = ax.imshow(img, cmap=plt.get_cmap(cmap_name).copy(),
                       vmin=-vmax, vmax=vmax)
        evr = pca.explained_variance_ratio_[k] * 100
        ax.set_title(f"PC{k + 1}  EVR={evr:.2f}%", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="um")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def recon_rmse_curve(pca: PCA, X: np.ndarray, ks: list[int]) -> np.ndarray:
    X0 = X - pca.mean_
    rmse = []
    for k in ks:
        Vt = pca.components_[:k]
        approx = (X0 @ Vt.T) @ Vt
        rmse.append(float(np.sqrt(np.mean((X0 - approx) ** 2))))
    return np.asarray(rmse)


def run_pca_block(X: np.ndarray, cols: np.ndarray | None, label: str,
                  out_dir: Path, cfg: dict) -> dict:
    n_pcs = int(cfg["analysis"]["n_pcs"])
    dpi = int(cfg["plot"]["dpi"])
    cmap = cfg["plot"]["diverging_cmap"]
    pca = PCA(n_components=n_pcs, svd_solver="full").fit(X)
    evr = pca.explained_variance_ratio_
    log(f"  [{label}] EVR PC1-PC10: "
        + " ".join(f"{v * 100:.2f}" for v in evr[:5])
        + f" ... cum10={evr.sum() * 100:.2f}%")
    scores = pca.transform(X)
    kind = label.split()[0].lower()  # absolute / residual
    plot_evr(pca, out_dir / f"{kind}_pca_evr.png",
             f"{label} - explained variance ratio (PC1-PC10)", dpi)
    plot_modes(pca, cols, out_dir / f"{kind}_pca_modes.png",
               f"{label} - spatial modes PC1-PC6", dpi, cmap)
    return {"pca": pca, "scores": scores, "evr": evr}


# --------------------------------------------------------------------------- #
# Step 7: cluster bootstrap stability
# --------------------------------------------------------------------------- #

def principal_angles(U_ref: np.ndarray, U_boot: np.ndarray) -> np.ndarray:
    s = np.linalg.svd(U_ref.T @ U_boot, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def gram_pca(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact PCA via the n x n Gram matrix (fast when n << n_features).

    Returns (components_ (k, F) rows, evr (k,)) equivalent to
    sklearn.decomposition.PCA(svd_solver="full") up to component sign.
    """
    n = X.shape[0]
    Xc = X - X.mean(axis=0, keepdims=True)
    G = (Xc @ Xc.T) / n
    w, U = np.linalg.eigh(G)
    order = np.argsort(w)[::-1][:k]
    top = w[order]
    s = np.sqrt(np.maximum(top, 0.0) * n)
    safe = np.where(s > 0, s, 1.0)
    Vt = (Xc.T @ U[:, order]) / safe[None, :]  # (F, k), orthonormal columns
    evr = top / np.trace(G)
    return Vt.T, evr


def cluster_bootstrap(bundle: dict, man: pd.DataFrame, ref_pca: PCA,
                      out_dir: Path, cfg: dict, n_rep: int) -> None:
    log(f"== Step 7: residual PCA cluster bootstrap (B={n_rep}) ==")
    rng = np.random.default_rng(int(cfg["random_seed"]))
    X_res = bundle["X_res"]
    cluster_key = cfg["bootstrap"]["cluster_key"]
    k_max = int(cfg["bootstrap"]["subspace_k_max"])

    clusters = [g.to_numpy() for _, g in
                man.groupby(cluster_key)["dataset_index"]]
    sizes = pd.Series([len(c) for c in clusters]).value_counts().to_dict()
    log(f"  {len(clusters)} clusters, size distribution {sizes}")

    rows = []
    evr_boot = np.zeros((n_rep, 3))
    for b in range(n_rep):
        pick = rng.integers(0, len(clusters), size=len(clusters))
        idx = np.concatenate([clusters[p] for p in pick])
        comps_b, evr_b = gram_pca(X_res[idx], k_max)
        evr_boot[b] = evr_b[:3]
        for k in range(1, k_max + 1):
            ang = principal_angles(ref_pca.components_[:k].T,
                                   comps_b[:k].T)
            rows.append((b + 1, k, float(ang[-1]), ""))
    df = pd.DataFrame(rows, columns=["replicate", "k",
                                     "max_principal_angle_deg", "note"])
    summary = []
    for k in range(1, k_max + 1):
        vals = df.loc[df["k"] == k, "max_principal_angle_deg"].to_numpy()
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        summary.append((k, med, q3 - q1, vals.min(), vals.max()))
    for k, med, iqr, lo, hi in summary:
        df.loc[len(df)] = ("summary", k, med,
                           f"IQR={iqr:.2f} deg, range={lo:.2f}..{hi:.2f} deg")
    df.to_csv(out_dir / "bootstrap_pca_stability.csv", index=False)
    log("  principal-angle stability (median [IQR], deg):")
    for k, med, iqr, lo, hi in summary:
        log(f"    k={k}: {med:6.2f} [{iqr:5.2f}]  range {lo:6.2f}..{hi:6.2f}")
    for i in range(3):
        q5, med, q95 = np.percentile(evr_boot[:, i], [5, 50, 95])
        log(f"    EVR PC{i + 1} bootstrap: median {med * 100:.2f}%, "
            f"90% interval [{q5 * 100:.2f}, {q95 * 100:.2f}]%")
    log("  wrote bootstrap_pca_stability.csv")


# --------------------------------------------------------------------------- #
# Step 8: raw vs repaired sensitivity
# --------------------------------------------------------------------------- #

def repaired_sensitivity(bundle: dict, man: pd.DataFrame, raw_res: dict,
                         raw_abs: dict, out_dir: Path, cfg: dict) -> pd.DataFrame:
    log("== Step 8: raw vs repaired sensitivity ==")
    X_rep = bundle["X_rep"]
    n_pcs = int(cfg["analysis"]["n_pcs"])
    k_max = int(cfg["bootstrap"]["subspace_k_max"])

    R3rep, _ = residualize(bundle["Hrep_nan"])
    bundle["R3_rep_full"] = R3rep
    R_rep2 = to_2d(R3rep)
    if bundle["mode"] == "restricted":
        R_rep_an = R_rep2[:, bundle["cols"]]
    else:
        counts = np.sum(np.isfinite(R_rep2), axis=0)
        sums = np.nansum(R_rep2, axis=0)
        fill = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
        R_rep_an = np.where(np.isfinite(R_rep2), R_rep2, fill[None, :])

    pca_rep_res = PCA(n_components=n_pcs, svd_solver="full").fit(R_rep_an)
    pca_rep_abs = PCA(n_components=n_pcs, svd_solver="full").fit(X_rep)
    evr_rr = pca_rep_res.explained_variance_ratio_
    evr_ra = pca_rep_abs.explained_variance_ratio_

    rows = []
    for i in range(n_pcs):
        rows.append((f"evr_absolute_pc{i + 1}", "", raw_abs["evr"][i], evr_ra[i]))
    for i in range(n_pcs):
        rows.append((f"evr_residual_pc{i + 1}", "", raw_res["evr"][i], evr_rr[i]))
    for k in range(1, k_max + 1):
        ang = principal_angles(raw_res["pca"].components_[:k].T,
                               pca_rep_res.components_[:k].T)[-1]
        rows.append((f"subspace_angle_k{k}",
                     "max principal angle raw vs repaired residual subspace",
                     ang, np.nan))

    traj = man[(man["session_role"] == "pass_main")]
    t_idx = traj["dataset_index"].to_numpy()
    s_raw = raw_res["scores"][t_idx, :3]
    s_rep = pca_rep_res.transform(R_rep_an)[t_idx, :3]
    for j in range(3):
        corr = float(np.corrcoef(s_raw[:, j], s_rep[:, j])[0, 1])
        rows.append(("traj_score_corr", f"PC{j + 1} over 60 pass ROIs",
                     corr, np.nan))

    # per-trajectory (N4 - N1) displacement vectors, raw space vs repaired space
    cosines = []
    for gname, grp in traj.groupby("design_group"):
        g = grp.sort_values("pass_count")
        require(int(g["pass_count"].iloc[0]) == 1
                and int(g["pass_count"].iloc[-1]) == 4,
                f"trajectory {gname} missing N=1 or N=4 endpoint")
        i1, i4 = g["dataset_index"].iloc[0], g["dataset_index"].iloc[-1]
        j1 = int(np.flatnonzero(t_idx == i1)[0])
        j4 = int(np.flatnonzero(t_idx == i4)[0])
        v_raw = s_raw[j4] - s_raw[j1]
        v_rep = s_rep[j4] - s_rep[j1]
        denom = np.linalg.norm(v_raw) * np.linalg.norm(v_rep)
        if denom > 0:
            cosines.append(float(v_raw @ v_rep / denom))
    cosines = np.asarray(cosines)
    for stat, val in (("min", cosines.min()), ("median", np.median(cosines)),
                      ("max", cosines.max())):
        rows.append(("traj_displacement_cos", f"{stat} over 15 trajectories "
                     f"(N4-N1 vector, raw-space vs repaired-space)",
                     val, np.nan))

    df = pd.DataFrame(rows, columns=["metric_family", "detail",
                                     "raw_value", "repaired_value"])
    df.to_csv(out_dir / "raw_repaired_sensitivity.csv", index=False)

    d_evr = np.abs(raw_res["evr"][:10] - evr_rr[:10])
    log(f"  max |EVR residual raw-repaired| PC1-10: {d_evr.max() * 100:.2f} pp "
        f"(PC{int(np.argmax(d_evr)) + 1})")
    angles = df.loc[df["metric_family"].str.startswith("subspace_angle"),
                    "raw_value"].to_numpy()
    log(f"  residual subspace angles raw vs repaired (k=1..{k_max}): "
        + " ".join(f"{a:.1f}" for a in angles) + " deg")
    log(f"  trajectory score correlations PC1-3: "
        + " ".join(f"{v:.3f}" for v in
                   df.loc[df["metric_family"] == "traj_score_corr",
                          "raw_value"]))
    log(f"  (N4-N1) displacement cosine raw vs repaired: "
        f"min={cosines.min():.3f}, median={np.median(cosines):.3f}, "
        f"max={cosines.max():.3f}")
    log("  wrote raw_repaired_sensitivity.csv")
    return df


# --------------------------------------------------------------------------- #
# Step 9: repeatability sentinel (DOE 49/50)
# --------------------------------------------------------------------------- #

def sentinel_analysis(bundle: dict, man: pd.DataFrame, out_dir: Path,
                      cfg: dict) -> dict:
    log("== Step 9: repeatability sentinel (DOE 49/50) ==")
    sent = cfg["sentinel"]
    sel = man[(man["session_id"] == sent["session"])
              & (man["processing_order"].isin(sent["processing_orders"]))]
    require(len(sel) == 2, f"sentinel rows found {len(sel)} != 2")
    i_a, i_b = sel["dataset_index"].to_numpy()

    fields = ["compressor_steps", "frequency_kHz", "hatch_spacing_um",
              "pass_count", "velocity_mm_s"]
    for f in fields:
        require(sel[f].nunique() == 1, f"sentinel manifest field {f} differs")

    H3d, V3d = bundle["H3d"], bundle["V3d"]
    R3 = bundle["R3_full"]
    M = V3d[i_a] & V3d[i_b]
    abs_rmse = float(np.sqrt(np.mean((H3d[i_a][M] - H3d[i_b][M]) ** 2)))
    d_diff = float(abs(sel["median_depth_um"].iloc[0]
                       - sel["median_depth_um"].iloc[1]))
    shape_rmse = float(np.sqrt(np.mean((R3[i_a][M] - R3[i_b][M]) ** 2)))

    # ordinary-pair contrast on the analysis basis (residual-shape RMSE)
    X_res = bundle["X_res"]
    gram = X_res @ X_res.T
    sqn = (X_res ** 2).sum(axis=1)
    d2 = np.clip(sqn[:, None] + sqn[None, :] - 2 * gram, 0.0, None)
    n_feat = X_res.shape[1]
    rmse_mat = np.sqrt(d2 / n_feat)
    iu = np.triu_indices(200, k=1)
    pair_rmse = rmse_mat[iu]
    pair_a, pair_b = iu
    same_source = (man["shared_height_source_id"].to_numpy()[pair_a]
                   == man["shared_height_source_id"].to_numpy()[pair_b])
    is_sentinel = ((pair_a == i_a) & (pair_b == i_b)) | \
                  ((pair_a == i_b) & (pair_b == i_a))
    ordinary = pair_rmse[~same_source & ~is_sentinel]
    n_shared = int(same_source.sum())
    pct = float(np.mean(ordinary < shape_rmse) * 100)

    log(f"  sentinel dataset_index {i_a}/{i_b}, common-valid pixels {int(M.sum())}")
    log(f"  absolute-height RMSE : {abs_rmse:.4f} um")
    log(f"  depth difference     : {d_diff:.4f} um")
    log(f"  residual-shape RMSE  : {shape_rmse:.4f} um")
    log(f"  ordinary pairs (excl. {n_shared} shared-source pairs and the "
        f"sentinel): n={ordinary.size}")
    log(f"  ordinary residual-shape RMSE: median={np.median(ordinary):.4f}, "
        f"P5={np.percentile(ordinary, 5):.4f}, "
        f"P95={np.percentile(ordinary, 95):.4f} um")
    log(f"  sentinel at percentile {pct:.3f} of ordinary pairs")

    dpi = int(cfg["plot"]["dpi"])
    fig = plt.figure(figsize=(14.0, 8.8), dpi=dpi)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0])
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.82")
    shared_vmin = min(np.nanmin(H3d[i_a]), np.nanmin(H3d[i_b]))
    shared_vmax = max(np.nanmax(H3d[i_a]), np.nanmax(H3d[i_b]))
    for col, (i, label) in enumerate(
            ((i_a, f"DOE order {sent['processing_orders'][0]}"),
             (i_b, f"DOE order {sent['processing_orders'][1]}"))):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(_masked(H3d[i], V3d[i]), cmap=cmap,
                       vmin=shared_vmin, vmax=shared_vmax)
        ax.set_title(f"{label}\nheight_raw [um]", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    ax = fig.add_subplot(gs[0, 2])
    diff = np.where(M, H3d[i_a] - H3d[i_b], np.nan)
    vmax = np.nanmax(np.abs(diff))
    im = ax.imshow(diff, cmap=plt.get_cmap(cfg["plot"]["diverging_cmap"]).copy(),
                   vmin=-vmax, vmax=vmax)
    ax.set_title("height difference (A - B)\ncommon-valid pixels", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="um")

    ax = fig.add_subplot(gs[1, :])
    ax.hist(ordinary, bins=80, color="tab:gray", alpha=0.85,
            label=f"ordinary pairs (n={ordinary.size})")
    ax.axvline(shape_rmse, color="tab:red", lw=2,
               label=f"sentinel 49/50 residual-shape RMSE = {shape_rmse:.3f} um "
                     f"(pct {pct:.2f})")
    ax.set_xlabel("residual-shape RMSE between sample pairs [um]")
    ax.set_ylabel("pairs")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Single-pair repeatability sentinel vs ordinary pair distances",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "repeatability_sentinel_49_50.png", dpi=dpi,
                bbox_inches="tight")
    plt.close(fig)
    log("  wrote repeatability_sentinel_49_50.png")
    return {"abs_rmse": abs_rmse, "depth_diff": d_diff,
            "shape_rmse": shape_rmse, "pct": pct}


# --------------------------------------------------------------------------- #
# Step 10: N=1-4 trajectory atlas
# --------------------------------------------------------------------------- #

def trajectory_figs(bundle: dict, man: pd.DataFrame, raw_res: dict,
                    out_dir: Path, cfg: dict) -> dict:
    log("== Step 10: N=1-4 trajectory atlas ==")
    dpi = int(cfg["plot"]["dpi"])
    traj = man[man["session_role"] == "pass_main"]
    idx = traj["dataset_index"].to_numpy()
    S = raw_res["scores"][idx, :3]
    groups = sorted(traj["design_group"].unique())
    require(len(groups) == 15, f"trajectory count {len(groups)} != 15")
    cmap = plt.get_cmap("tab20")

    fig = plt.figure(figsize=(17.0, 6.2), dpi=dpi)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 2.55])
    planes = [(0, 1, "PC1", "PC2"), (0, 2, "PC1", "PC3")]
    for pi, (a, b, la, lb) in enumerate(planes):
        ax = fig.add_subplot(gs[0, pi])
        for gi, gname in enumerate(groups):
            g = traj[traj["design_group"] == gname].sort_values("pass_count")
            pts = S[[int(np.flatnonzero(idx == i)[0]) for i in
                     g["dataset_index"]]]
            color = cmap(gi % 20)
            ax.plot(pts[:, a], pts[:, b], "-o", color=color, ms=4,
                    lw=1.0, alpha=0.85)
        ax.set_xlabel(la + f" ({raw_res['evr'][a] * 100:.1f}% EVR)")
        ax.set_ylabel(lb + f" ({raw_res['evr'][b] * 100:.1f}% EVR)")
        ax.set_title(f"{la}-{lb} plane", fontsize=10)
        ax.grid(alpha=0.25)
    gs_r = gs[0, 2].subgridspec(3, 5)
    for gi, gname in enumerate(groups):
        g = traj[traj["design_group"] == gname].sort_values("pass_count")
        pts = S[[int(np.flatnonzero(idx == i)[0]) for i in g["dataset_index"]]]
        ax = fig.add_subplot(gs_r[gi // 5, gi % 5])
        ax.plot(pts[:, 0], pts[:, 1], "-o", color=cmap(gi % 20), ms=3, lw=0.9)
        for n, (x, y) in zip(g["pass_count"], pts[:, :2]):
            ax.annotate(f"N{n}", (x, y), fontsize=5,
                        textcoords="offset points", xytext=(2, 2))
        ax.set_title(f"{gname}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.tick_params(labelsize=5)
    fig.suptitle("60-pass trajectories in residual-PCA space (N=1..4, "
                 "cross-sectional pseudo-temporal)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "pass_trajectory_N1_N4.png", dpi=dpi,
                bbox_inches="tight")
    plt.close(fig)
    log("  wrote pass_trajectory_N1_N4.png")

    # descriptive displacement statistics
    step_angles = []
    endpoints = []
    for gname in groups:
        g = traj[traj["design_group"] == gname].sort_values("pass_count")
        pts = S[[int(np.flatnonzero(idx == i)[0]) for i in g["dataset_index"]]]
        steps = np.diff(pts, axis=0)
        for s1, s2 in zip(steps[:-1], steps[1:]):
            n1, n2 = np.linalg.norm(s1), np.linalg.norm(s2)
            if n1 > 0 and n2 > 0:
                c = float(np.clip(s1 @ s2 / (n1 * n2), -1, 1))
                step_angles.append(np.degrees(np.arccos(c)))
        endpoints.append(pts[-1] - pts[0])
    endpoints = np.asarray(endpoints)
    cos_mat = []
    for i, j in itertools.combinations(range(len(endpoints)), 2):
        a, b = endpoints[i], endpoints[j]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 0 and nb > 0:
            cos_mat.append(float(a @ b / (na * nb)))
    cos_mat = np.asarray(cos_mat)
    stats = {
        "step_turn_median_deg": float(np.median(step_angles)),
        "endpoint_cos_mean": float(cos_mat.mean()),
        "endpoint_cos_min": float(cos_mat.min()),
        "endpoint_cos_max": float(cos_mat.max()),
        "endpoint_cos_frac_gt_0.7": float(np.mean(cos_mat > 0.7)),
    }
    log(f"  consecutive-step turning angle: median {stats['step_turn_median_deg']:.1f} deg")
    log(f"  (N4-N1) endpoint vectors, pairwise cosine over 15 trajectories: "
        f"mean={stats['endpoint_cos_mean']:.3f}, "
        f"min={stats['endpoint_cos_min']:.3f}, max={stats['endpoint_cos_max']:.3f}, "
        f"frac>0.7={stats['endpoint_cos_frac_gt_0.7']:.2f}")
    return stats


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name(
        "phase1_config.yaml")))
    parser.add_argument("--quick", action="store_true",
                        help="bootstrap smoke mode (few replicates)")
    args = parser.parse_args(argv)

    with open(REPO / args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    out_dir = REPO / cfg["paths"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    n_boot = (int(cfg["bootstrap"]["n_replicates_quick"]) if args.quick
              else int(cfg["bootstrap"]["n_replicates"]))

    log(f"Phase 1 minimal exploration  (mode={'quick' if args.quick else 'full'}, "
        f"seed={cfg['random_seed']})")

    # Step 1
    man, designs = build_manifest(cfg)
    man.to_csv(out_dir / "exploration_manifest.csv", index=False)
    log("  wrote exploration_manifest.csv (identity columns)")

    # Step 2
    bundle = load_heights(cfg)
    assemble_matrices(bundle, man, cfg)

    # Step 3: atlas
    log("== Step 3: data atlas ==")
    plot_atlas(bundle, man, out_dir, cfg, global_scale=True)
    plot_atlas(bundle, man, out_dir, cfg, global_scale=False)

    # Step 4: depth
    log("== Step 4: depth statistics ==")
    depth, sq = depth_stats(bundle, man)
    man = man.copy()
    man["median_depth_um"] = np.round(depth, 4)
    man["residual_Sq_um"] = np.round(sq, 4)
    man = man[MANIFEST_COLUMNS]
    man.to_csv(out_dir / "exploration_manifest.csv", index=False)
    log("  manifest augmented with median_depth_um / residual_Sq_um")
    log(f"  depth [um]: min={depth.min():.2f}, median={np.median(depth):.2f}, "
        f"max={depth.max():.2f}")
    log(f"  residual Sq [um]: min={sq.min():.3f}, median={np.median(sq):.3f}, "
        f"max={sq.max():.3f}")
    plot_depth_figs(bundle, man, out_dir, cfg)

    # Step 5: absolute PCA
    log("== Step 5: absolute PCA ==")
    raw_abs = run_pca_block(bundle["X_raw"], bundle["cols"],
                            "Absolute PCA on H_i", out_dir, cfg)
    d = -np.nanmedian(bundle["Hnan"], axis=(1, 2))
    corr_pc1_d = float(np.corrcoef(raw_abs["scores"][:, 0], d)[0, 1])
    log(f"  corr(PC1, d_i) = {corr_pc1_d:.4f}")

    # Step 6: residual PCA
    log("== Step 6: residual PCA ==")
    R3, _ = residualize(bundle["Hnan"])
    bundle["R3_full"] = R3
    R2 = to_2d(R3)
    if bundle["mode"] == "restricted":
        R_an = R2[:, bundle["cols"]]
    else:
        counts = np.sum(np.isfinite(R2), axis=0)
        sums = np.nansum(R2, axis=0)
        fill = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
        R_an = np.where(np.isfinite(R2), R2, fill[None, :])
    bundle["X_res"] = R_an
    raw_res = run_pca_block(R_an, bundle["cols"], "Residual PCA on R_i",
                            out_dir, cfg)

    # reconstruction RMSE (both curves, one figure)
    ks = list(range(1, int(cfg["analysis"]["n_pcs"]) + 1))
    rmse_abs = recon_rmse_curve(raw_abs["pca"], bundle["X_raw"], ks)
    rmse_res = recon_rmse_curve(raw_res["pca"], bundle["X_res"], ks)
    fig, ax = plt.subplots(figsize=(7.0, 4.8),
                           dpi=int(cfg["plot"]["dpi"]))
    ax.plot(ks, rmse_abs, "o-", label="absolute PCA (on centred H)")
    ax.plot(ks, rmse_res, "s-", label="residual PCA (on centred R)")
    ax.set_xticks(ks)
    ax.set_xlabel("number of components used for reconstruction")
    ax.set_ylabel("reconstruction RMSE [um]")
    ax.set_title("Reconstruction RMSE vs number of components")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pca_reconstruction_rmse.png",
                dpi=int(cfg["plot"]["dpi"]), bbox_inches="tight")
    plt.close(fig)
    log("  wrote pca_reconstruction_rmse.png")
    log(f"  RMSE k=1..5 absolute: " + " ".join(f"{v:.3f}" for v in rmse_abs[:5]))
    log(f"  RMSE k=1..5 residual: " + " ".join(f"{v:.3f}" for v in rmse_res[:5]))

    # depth dominance summary quantity (descriptive)
    Hc = bundle["X_raw"] - bundle["X_raw"].mean(axis=0, keepdims=True)
    Rc = bundle["X_res"] - bundle["X_res"].mean(axis=0, keepdims=True)
    var_h, var_r = float(Hc.var()), float(Rc.var())
    depth_share = 1.0 - var_r / var_h
    log(f"  variance decomposition (analysis basis): "
        f"Var(H)={var_h:.4f} um^2, Var(R)={var_r:.4f} um^2, "
        f"depth share = 1 - Var(R)/Var(H) = {depth_share:.4f}")

    # Step 7: bootstrap
    cluster_bootstrap(bundle, man, raw_res["pca"], out_dir, cfg, n_boot)

    # Step 8: sensitivity
    repaired_sensitivity(bundle, man, raw_res, raw_abs, out_dir, cfg)

    # Step 9: sentinel
    sentinel_stats = sentinel_analysis(bundle, man, out_dir, cfg)

    # Step 10: trajectories
    traj_stats = trajectory_figs(bundle, man, raw_res, out_dir, cfg)

    # Step 11: summary
    log("")
    log("=" * 74)
    log("PHASE 1 DESCRIPTIVE SUMMARY (no regression, no causal claims)")
    log("=" * 74)
    log(f"Q1 depth dominance           : depth share of absolute-height "
        f"variance = {depth_share:.4f} (Var(H)={var_h:.3f}, "
        f"Var(R)={var_r:.3f} um^2); corr(PC1_abs, d) = {corr_pc1_d:.4f}")
    log(f"Q2 residual structure        : residual PC1-PC10 EVR = "
        + " ".join(f"{v * 100:.1f}" for v in raw_res["evr"][:10])
        + f" % (cum {raw_res['evr'].sum() * 100:.1f}%); "
        f"recon RMSE k=1 -> {rmse_res[0]:.3f} um, k=10 -> {rmse_res[-1]:.3f} um")
    log(f"Q3 bootstrap stability       : see principal-angle table above "
        f"(B={n_boot}, cluster={cfg['bootstrap']['cluster_key']})")
    sens = pd.read_csv(out_dir / "raw_repaired_sensitivity.csv")
    ang = sens[sens["metric_family"].str.startswith("subspace_angle")][
        "raw_value"].to_numpy()
    rep_pc1 = float(sens.loc[sens["metric_family"] == "evr_residual_pc1",
                             "repaired_value"].iloc[0])
    log(f"Q4 raw vs repaired           : residual subspace angles k=1..6 = "
        + " ".join(f"{a:.1f}" for a in ang)
        + f" deg; residual EVR PC1 raw={raw_res['evr'][0] * 100:.2f}% "
        f"vs repaired={rep_pc1 * 100:.2f}%")
    log(f"Q5 N=1-4 trajectories        : endpoint-vector cosine mean="
        f"{traj_stats['endpoint_cos_mean']:.3f} "
        f"[{traj_stats['endpoint_cos_min']:.3f}, "
        f"{traj_stats['endpoint_cos_max']:.3f}], "
        f"frac>0.7={traj_stats['endpoint_cos_frac_gt_0.7']:.2f}; "
        f"median step turning {traj_stats['step_turn_median_deg']:.1f} deg")
    log(f"Sentinel 49/50               : abs RMSE "
        f"{sentinel_stats['abs_rmse']:.4f} um, depth diff "
        f"{sentinel_stats['depth_diff']:.4f} um, residual-shape RMSE "
        f"{sentinel_stats['shape_rmse']:.4f} um "
        f"(percentile {sentinel_stats['pct']:.3f} of ordinary pairs)")

    missing = [f for f in EXPECTED_OUTPUTS if not (out_dir / f).exists()]
    require(not missing, f"missing outputs: {missing}")
    log("")
    log(f"SELF-CHECK: all {len(EXPECTED_OUTPUTS)} expected outputs present in "
        f"{out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
