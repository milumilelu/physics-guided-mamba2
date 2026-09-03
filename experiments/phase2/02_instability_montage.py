#!/usr/bin/env python3
"""Phase 2 experiment 02: instability montage (Phase 2A-2, two-round blinded).

Round 1 (BLIND): for every selected sample a 3x3 diagnostic page showing only
morphology (absolute height, residual, four DCT bands, valid/repair mask,
profiles, radial PSD), labelled with the anonymous AUDIT-xx code assigned in
01. No process metadata, no LOCO rank, no sample identity.
Round 2 (UNBLIND): 4x3 pages that add process/morphology neighbours and the
full metadata panel; a _groupscale variant shares one colour scale across the
selected pool for the residual/band panels. Own-scale and group-scale pages
must never be mixed (细则 §4); panel A absolute height stays own-scale in
both, because sample depths differ by ~60 um and a shared scale would flatten
them.

The manual review happens in `instability_manual_review.csv`; this script
writes its template and performs no automatic classification.

Seed offsets: none (no stochastic step).
"""

from __future__ import annotations

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy.fft import dctn  # noqa: E402
from scipy.ndimage import label as ndlabel  # noqa: E402

import _lib as p2

EXPECTED = ["instability_manual_review.csv",
            "instability_montage_round1_blind.pdf",
            "round1", "round2", "README.md"]

CHECKLIST = ("edge_contamination;large_area_dropout;repair_driven_feature;"
             "large_pit;ridge;periodic_stripe;anisotropic_texture;"
             "low_frequency_waviness;localized_collapse;multi_lobe_morphology")

README = f"""# instability montage (Phase 2A-2, two-round blinded)

- `round1/AUDIT-xx_blind.png`: blinded morphology-only pages. Do the FIRST
  review pass using ONLY these pages (fill the `blind_*` columns).
- `round2/sample_<ddd>_unblind.png`: unblind pages with neighbours + metadata
  (`blind_*` conclusions may then be confirmed or revised in the unblind
  columns). `*_groupscale.png` shares one colour scale across the selected
  pool for the residual/DCT-band panels; panel A (absolute height) stays
  own-scale in both variants because sample depths differ by ~60 um.
- `morphology_pattern` uses the checklist terms, `;`-separated:
  {CHECKLIST}
No automatic classification exists in this script (细则 §4).
"""


def _p2p98(a: np.ndarray) -> tuple[float, float]:
    return tuple(np.percentile(a, [2, 98]))


def _imshow(ax, img, extent, vmin, vmax, title):
    im = ax.imshow(img, origin="lower", extent=extent, vmin=vmin, vmax=vmax,
                   cmap="RdBu_r")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def _psd_loglog(R2d: np.ndarray, pixel_um: float):
    lam = p2.l15.dct_lambda_grid(R2d.shape, pixel_um)
    power = dctn(R2d, norm="ortho") ** 2
    edges = np.logspace(np.log10(2.0), np.log10(160.0), 25)
    centers = np.sqrt(edges[:-1] * edges[1:])
    which = np.digitize(lam.ravel(), edges) - 1
    flat_p, flat_w = power.ravel(), which
    ok = (flat_w >= 0) & (flat_w < len(centers)) & np.isfinite(lam.ravel())
    mean_p = np.array([flat_p[ok & (flat_w == b)].mean()
                       if np.any(ok & (flat_w == b)) else np.nan
                       for b in range(len(centers))])
    return centers, mean_p


def main() -> int:
    cfg, quick = p2.load_config(__doc__)
    t0 = time.time()
    out = p2.output_dir(cfg, "instability")
    (out / "round1").mkdir(exist_ok=True)
    (out / "round2").mkdir(exist_ok=True)
    p2.log("== Phase 2 / 02: instability montage (two-round blinded) ==")
    frozen = p2.l15.load_frozen(cfg)
    man = p2.read_manifest(cfg, require_loco=True)
    inv = pd.read_csv(out / "instability_inventory.csv")
    sel = pd.read_csv(out / "instability_selected.csv")
    p2.require(len(sel) >= 1, "empty selection pool")

    R3 = frozen["R"]
    fields = p2.l15.multiscale_fields(R3, cfg)
    n_px = int(np.sqrt(fields["total"].shape[1]))
    bands = ["DCT_8_16", "DCT_16_32", "DCT_32_64", "DCT_64_inf"]
    band_img = {b: fields[b].reshape(-1, n_px, n_px) for b in bands}
    npz = np.load(p2.l15.REPO / cfg["paths"]["dataset_npz"])
    x_um = npz["x_um"].astype(float)
    y_um = npz["y_um"].astype(float)
    extent = [x_um[0], x_um[-1], y_um[0], y_um[-1]]

    idx = sel["dataset_index"].to_numpy(int)
    stack_r = np.concatenate([R3[i].ravel() for i in idx])
    group_lim = {"R": _p2p98(stack_r)}
    for b in bands:
        group_lim[b] = _p2p98(np.concatenate([band_img[b][i].ravel()
                                              for i in idx]))

    # neighbour tables for round 2 (k=5, phys process / descriptor space)
    dcols = cfg["instability"]["descriptor_cols"]
    Zd = p2.robust_z(inv[dcols].to_numpy(float))
    Zp = p2.zscore(man[p2.PROC_PHYS_COLS].to_numpy(float))
    Dd = np.sqrt(((Zd[:, None, :] - Zd[None, :, :]) ** 2).sum(-1))
    Dp = np.sqrt(((Zp[:, None, :] - Zp[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(Dd, np.inf)
    np.fill_diagonal(Dp, np.inf)
    Dm_tot = p2.pairwise_gram_rmse(fields["total"])

    desc = inv.set_index("dataset_index")
    review_rows = []
    with PdfPages(out / "instability_montage_round1_blind.pdf") as pdf:
        for _, srow in sel.iterrows():
            i = int(srow["dataset_index"])
            anon = str(srow["anon_code"])
            H = frozen["Hnan"][i]
            R = R3[i]
            meta = man.iloc[i]
            rep_mask = npz["repair_mask"][i]
            n_rep = int(rep_mask.sum())
            lab, n_lab = ndlabel(rep_mask)
            largest = int(np.bincount(lab.ravel())[1:].max()) if n_lab else 0

            # ---------------- round 1: blind (morphology only) --------------
            fig, axes = plt.subplots(3, 3, figsize=(11, 10))
            vmin, vmax = _p2p98(H[np.isfinite(H)])
            _imshow(axes[0, 0], H, extent, vmin, vmax,
                    "A absolute height [um]")
            _imshow(axes[0, 1], R, extent, *_p2p98(R), "B residual R [um]")
            for j, b in enumerate(bands):
                pos = [(0, 2), (1, 0), (1, 1), (1, 2)][j]
                _imshow(axes[pos[0], pos[1]], band_img[b][i], extent,
                        *_p2p98(band_img[b][i]), f"{'CDEF'[j]} {b} [um]")
            _imshow(axes[2, 0], frozen["V"][i].astype(float) - rep_mask,
                    extent, -0.5, 1.5, "G valid (1) / repaired (-1) px")
            axes[2, 1].plot(x_um, R[80, :], lw=0.8, label="row y=40um")
            axes[2, 1].plot(y_um, R[:, 80], lw=0.8, label="col x=40um")
            axes[2, 1].legend(fontsize=6)
            axes[2, 1].set_title("H central profiles [um]", fontsize=8)
            axes[2, 1].grid(alpha=0.25)
            axes[2, 1].tick_params(labelsize=6)
            centers, mean_p = _psd_loglog(R, float(cfg["scales"]["pixel_um"]))
            axes[2, 2].loglog(centers, mean_p, lw=0.9)
            axes[2, 2].set_title("I radial PSD (DCT bins)", fontsize=8)
            axes[2, 2].set_xlabel("lambda [um]", fontsize=7)
            axes[2, 2].grid(alpha=0.25, which="both")
            axes[2, 2].tick_params(labelsize=6)
            fig.suptitle(f"{anon}  (round 1 blind — identity hidden)",
                         fontsize=11)
            fig.tight_layout()
            fig.savefig(out / "round1" / f"{anon}_blind.png",
                        dpi=cfg["plot"]["dpi"])
            pdf.savefig(fig)
            plt.close(fig)

            # ---------------- round 2: unblind -------------------------------
            for scale_tag in ("ownscale", "groupscale"):
                fig, axes = plt.subplots(4, 3, figsize=(13, 14))
                _imshow(axes[0, 0], H, extent, *_p2p98(H[np.isfinite(H)]),
                        "A absolute height [um] (own scale)")
                panels = [("B residual R", R, "R"),
                          ("C DCT_8_16", band_img["DCT_8_16"][i], "DCT_8_16"),
                          ("D DCT_16_32", band_img["DCT_16_32"][i], "DCT_16_32"),
                          ("E DCT_32_64", band_img["DCT_32_64"][i], "DCT_32_64"),
                          ("F DCT_64_inf", band_img["DCT_64_inf"][i], "DCT_64_inf")]
                pos = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
                for (title, arr, key), (r, c) in zip(panels, pos):
                    lim = group_lim[key] if scale_tag == "groupscale" \
                        else _p2p98(arr)
                    _imshow(axes[r, c], arr, extent, *lim,
                            title + (" [group scale]"
                                     if scale_tag == "groupscale" else ""))
                _imshow(axes[2, 0], frozen["V"][i].astype(float) - rep_mask,
                        extent, -0.5, 1.5,
                        f"G valid / repaired (n={n_rep}, max comp={largest}px)")
                axes[2, 1].plot(x_um, R[80, :], lw=0.8, label="row y=40um")
                axes[2, 1].plot(y_um, R[:, 80], lw=0.8, label="col x=40um")
                axes[2, 1].legend(fontsize=6)
                axes[2, 1].set_title("H central profiles [um]", fontsize=8)
                axes[2, 1].grid(alpha=0.25)
                axes[2, 1].tick_params(labelsize=6)
                centers, mean_p = _psd_loglog(R, float(cfg["scales"]["pixel_um"]))
                axes[2, 2].loglog(centers, mean_p, lw=0.9)
                axes[2, 2].set_title("I radial PSD (DCT bins)", fontsize=8)
                axes[2, 2].set_xlabel("lambda [um]", fontsize=7)
                axes[2, 2].grid(alpha=0.25, which="both")
                axes[2, 2].tick_params(labelsize=6)

                lines_j = ["J nearest process neighbours (phys space, k=5):"]
                for j in np.argsort(Dp[i])[:5]:
                    lines_j.append(
                        f"  #{int(j)} {man.iloc[j]['session_id']}"
                        f"/s{int(man.iloc[j]['sample_id'])} "
                        f"Dproc={Dp[i, j]:.2f} Dmorph_tot={Dm_tot[i, j]:.2f}um")
                axes[3, 0].axis("off")
                axes[3, 0].text(0.0, 1.0, "\n".join(lines_j), fontsize=7,
                                family="monospace", va="top")
                lines_k = ["K nearest morphology neighbours (descriptor, k=5):"]
                for j in np.argsort(Dd[i])[:5]:
                    lines_k.append(
                        f"  #{int(j)} {man.iloc[j]['session_id']}"
                        f"/s{int(man.iloc[j]['sample_id'])} "
                        f"Dmorph_desc={Dd[i, j]:.2f} "
                        f"Dmorph_tot={Dm_tot[i, j]:.2f}um")
                axes[3, 1].axis("off")
                axes[3, 1].text(0.0, 1.0, "\n".join(lines_k), fontsize=7,
                                family="monospace", va="top")
                lines_l = [
                    f"L sample #{i} {meta['session_id']} "
                    f"s{int(meta['sample_id'])} "
                    f"po{int(meta['processing_order'])} "
                    f"role={meta['session_role']} "
                    f"grp={meta['base_condition_group']}",
                    f"  process: tau={meta['pulse_duration_fs']:.0f}fs "
                    f"f={meta['frequency_kHz']:g}kHz "
                    f"h={meta['hatch_spacing_um']:g}um "
                    f"N={int(meta['pass_count'])} "
                    f"v={meta['velocity_mm_s']:g}mm/s",
                    f"  reparam(proxy): Ep={meta['pulse_energy_proxy_uJ']:.1f}uJ "
                    f"dx={meta['scan_spacing_um']:.2f}um "
                    f"nA={meta['areal_pulse_density_per_mm2']:.3g}/mm2 "
                    f"DE={meta['areal_dose_proxy_J_per_mm2']:.3g}J/mm2",
                    f"  depth={meta['median_depth_um']:.2f}um "
                    f"Sq={desc.loc[i, 'Sq_um']:.3f}um "
                    f"repair={meta['repair_fraction']:.4f} "
                    f"plane_rmse={meta['plane_rmse_um']:.3f}um",
                    f"  LOCO(total,pc1)={meta['phase1_global_loco_angle_deg']:.1f}deg "
                    f"(rank {int(meta['phase1_global_loco_rank'])}) "
                    f"| consensus={float(srow['A_consensus']):.1f}",
                    "  LOCO by field (pc1): "
                    + ", ".join(f"{f}={desc.loc[i, f'loco_{f}_pc1_deg']:.1f}"
                                for f in ("total", "DCT_8_16", "DCT_16_32",
                                          "DCT_32_64", "DCT_64_inf")) + " deg",
                    f"  reasons: {srow['selection_reason']}",
                ]
                for b in bands:
                    lines_l.append(f"    {b}: rms={desc.loc[i, f'rms_{b}_um']:.3f}um "
                                   f"Efrac={desc.loc[i, f'E_{b}_frac']:.3f}")
                axes[3, 2].axis("off")
                axes[3, 2].text(0.0, 1.0, "\n".join(lines_l), fontsize=6.5,
                                family="monospace", va="top")
                fig.suptitle(f"sample #{i} ({anon}) — round 2 unblind "
                             f"[{scale_tag}]", fontsize=11)
                fig.tight_layout()
                suffix = "" if scale_tag == "ownscale" else "_groupscale"
                fig.savefig(out / "round2" / f"sample_{i:03d}_unblind{suffix}.png",
                            dpi=cfg["plot"]["dpi"])
                plt.close(fig)

            review_rows.append({
                "dataset_index": i, "anon_code": anon, "reviewer": "",
                "blind_morphology_pattern": "", "blind_artifact_suspected": "",
                "blind_notes": "", "unblind_artifact_suspected": "",
                "artifact_reason": "", "morphology_pattern": "",
                "confidence": "", "notes": ""})
            p2.log(f"  [{anon}] sample #{i} pages done ({len(sel)} total)")

    pd.DataFrame(review_rows).to_csv(out / "instability_manual_review.csv",
                                     index=False)
    (out / "README.md").write_text(README, encoding="utf-8")
    missing = [f for f in EXPECTED if not (out / f).exists()]
    p2.require(not missing, f"missing outputs: {missing}")
    p2.log(f"02 done in {time.time() - t0:.1f}s; all outputs present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
