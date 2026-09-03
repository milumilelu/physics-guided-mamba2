#!/usr/bin/env python3
"""Phase 2.5 Task 11: directional spectrum + stripe-phenotype validation.

Per-sample, per-band directional metrics from a windowed 2D FFT (DC removed,
separable Hann, window-energy normalization; image-frame orientation):
A2 (second angular moment), wave-vector angle, real-space stripe angle
(= wave-vector + 90 deg mod 180), angular entropy.

G2a phenotype validation ONLY: does A2_8_16 separate the blind-audit
"periodic stripe" samples? The audit set is an enriched selection — it never
estimates population stripe prevalence. All process modelling on the full 200
samples happens in Task 12 (G2b).

Seed offsets: stripe-label permutation = seed + 150 (Monte-Carlo permutation,
(1+b)/(1+n) correction).
"""

from __future__ import annotations

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _lib as p25

EXPECTED = ["directional_metrics.csv", "directional_spectrum_long.csv",
            "stripe_validation.csv", "stripe_validation.png",
            "anisotropy_vs_E8.png", "orientation_histogram.png",
            "example_directional_psd", "README.md"]

README = """# directional spectrum (Task 11)

- `directional_metrics.csv`: A2 / theta_k / theta_stripe / angular_entropy per
  sample x band. Orientation is IMAGE-FRAME relative — no scan/hatch pose
  metadata exists, so angles must not be interpreted as hatch-relative.
- `stripe_validation.csv`: G2a phenotype validation on the 28-sample enriched
  audit set (labels = blind pattern contains "periodic stripe"). AUROC /
  rank-biserial / permutation p. **Enriched selection — never read as
  population stripe prevalence.** p > 0.05 means INCONCLUSIVE_AT_AUDIT_SIZE,
  not "metric invalid".
- `directional_spectrum_long.csv`: per-sample x band x 36 folded-orientation
  bins of normalized power (figure input).
"""


def _auroc_perm(scores: np.ndarray, labels: np.ndarray, n_perm: int,
                seed: int) -> dict:
    x = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    p25.require(n_pos > 0 and n_neg > 0, "degenerate validation labels")
    order = np.argsort(np.argsort(x)) + 1                    # rank 1..n
    auc = float((order[y].sum() - n_pos * (n_pos + 1) / 2)
                / (n_pos * n_neg))
    rb = 2 * auc - 1                                          # rank-biserial
    rng = np.random.default_rng(seed)
    b = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        op = np.argsort(np.argsort(x)) + 1
        a = float((op[yp].sum() - yp.sum() * (yp.sum() + 1) / 2)
                  / (yp.sum() * (~yp).sum()))
        if abs(a - 0.5) >= abs(auc - 0.5):
            b += 1
    return {"auc": auc, "rank_biserial": rb,
            "p_perm": (1 + b) / (1 + n_perm),
            "n_positive": n_pos, "n_negative": n_neg}


def main() -> int:
    cfg, quick = p25.load_config(__doc__)
    t0 = time.time()
    out = p25.output_dir(cfg, "directional_spectrum")
    (out / "example_directional_psd").mkdir(exist_ok=True)
    seed = int(cfg["random_seed"])
    p25.log("== Phase 2.5 / 11: directional spectrum ==")
    frozen = p25.l15.load_frozen(cfg)
    R = frozen["R"]
    pixel = float(cfg["scales"]["pixel_um"])
    bands = [(float(lo), (float(hi) if hi < 1e8 else 1e9), name)
             for lo, hi, name in
             ((8, 16, "8_16"), (16, 32, "16_32"), (32, 64, "32_64"),
              (64, 1e9, "64_inf"))]

    metrics = p25.directional_band_metrics(R, pixel, bands,
                                           int(cfg["directional"]["theta_bins"]))
    metrics.to_csv(out / "directional_metrics.csv", index=False)
    p25.require(np.isfinite(metrics[["A2", "theta_k_deg", "theta_stripe_deg",
                                     "angular_entropy"]].to_numpy()).all(),
                "non-finite directional metrics")
    p25.log(f"  directional metrics: {len(metrics)} rows "
            f"(A2_8_16 median "
            f"{metrics[metrics.band == '8_16']['A2'].median():.3f})")

    # long angular distributions (36 folded-orientation bins) for figures
    from scipy.fft import fft2
    nx, ny = R.shape[1:]
    win = np.hanning(nx)[:, None] * np.hanning(ny)[None, :]
    win_energy = (win ** 2).sum()
    kx = np.fft.fftfreq(nx, d=pixel)
    ky = np.fft.fftfreq(ny, d=pixel)
    KX, KY = np.meshgrid(kx, ky)   # 'xy': KX along axis1 (x), consistent with _lib
    lam = np.divide(1.0, np.hypot(KX, KY), where=np.hypot(KX, KY) > 0,
                    out=np.full((nx, ny), np.inf))
    theta = np.arctan2(KY, KX)
    long_parts = []
    edges = np.linspace(0.0, np.pi, int(cfg["directional"]["theta_bins"]) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:]) * 180.0 / np.pi
    for i in range(R.shape[0]):
        x = R[i] - R[i].mean()
        P = np.abs(fft2(x * win)) ** 2 / win_energy
        for lo, hi, name in bands:
            msk = np.isfinite(lam) & (lam >= lo) & \
                ((lam < hi) if hi < 1e8 else True)
            th = np.mod(theta[msk], np.pi)
            pw = P[msk]
            hist, _ = np.histogram(th, bins=edges, weights=pw)
            q = hist / max(hist.sum(), 1e-300)
            long_parts.append(pd.DataFrame({
                "dataset_index": i, "band": name,
                "orientation_bin": np.arange(len(centers)),
                "orientation_center_deg": centers, "power_fraction": q}))
    pd.concat(long_parts).to_csv(out / "directional_spectrum_long.csv",
                                 index=False)

    # ---- G2a: blind stripe validation ---------------------------------------
    rev_path = (p25.l15.REPO / "outputs/phase2/instability/盲评"
                / "instability_manual_review_completed.csv")
    rev = pd.read_csv(rev_path)
    token = cfg["directional"]["blind_stripe_token"]
    lab = rev["blind_morphology_pattern"].astype(str) \
        .str.contains(token, regex=False)
    val_idx = rev["dataset_index"].astype(int).to_numpy()
    m = metrics[(metrics.band == "8_16")
                & (metrics.dataset_index.isin(val_idx))] \
        .set_index("dataset_index").loc[val_idx]
    labels = lab.to_numpy()
    scores = m["A2"].to_numpy()
    stats = _auroc_perm(scores, labels,
                        int(cfg["directional"]["permutation_n"]),
                        seed + 150)
    val = pd.DataFrame([{"metric": "A2_8_16", "token": token, **stats}])
    val.to_csv(out / "stripe_validation.csv", index=False)
    state = ("VALIDATED" if stats["p_perm"] <= cfg["gates"]["G2a_validation_p"]
             and abs(stats["rank_biserial"]) > 0 else
             "INCONCLUSIVE_AT_AUDIT_SIZE")
    p25.log(f"  G2a: AUROC={stats['auc']:.3f} rank-biserial="
            f"{stats['rank_biserial']:+.3f} p={stats['p_perm']:.4f} "
            f"(n_pos={stats['n_positive']}) -> {state}")

    # ---- figures -------------------------------------------------------------
    comp = pd.read_csv(p25.output_dir(cfg, "spectral_composition")
                       / "spectral_composition.csv")
    a2 = metrics[metrics.band == "8_16"].set_index("dataset_index")["A2"]
    e8 = comp.set_index("dataset_index")["p_8_16"]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    e8v = e8.loc[val_idx].to_numpy()
    a2v = a2.loc[val_idx].to_numpy()
    ax.scatter(e8v[~labels], a2v[~labels], s=18, c="0.5",
               label="no blind stripe")
    ax.scatter(e8v[labels], a2v[labels], s=26, c="tab:red",
               label=f"blind stripe (n={int(labels.sum())})")
    ax.set_xlabel("E_8-16 fraction (clean non-DC)")
    ax.set_ylabel("A2_8_16 (image frame)")
    ax.set_title("Anisotropy vs 8-16 um energy fraction (audit subset)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "anisotropy_vs_E8.png", dpi=cfg["plot"]["dpi"])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    data = [scores[labels], scores[~labels]]
    ax.boxplot(data, tick_labels=[f"stripe (n={int(labels.sum())})",
                                  f"other (n={int((~labels).sum())})"],
               widths=0.5)
    ax.set_ylabel("A2_8_16")
    ax.set_title(f"G2a: AUROC={stats['auc']:.3f}, perm p={stats['p_perm']:.4f}"
                 f"\n{state} (enriched audit set)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "stripe_validation.png", dpi=cfg["plot"]["dpi"])
    plt.close(fig)

    lng = pd.concat(long_parts)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6),
                             subplot_kw={"projection": "polar"})
    for ax, (lo, hi, name) in zip(axes, bands):
        sub = lng[lng.band == name].groupby("orientation_bin")[
            "power_fraction"].mean()
        th = centers * np.pi / 180.0
        ax.plot(np.concatenate([th, th + np.pi]),
                np.concatenate([sub.to_numpy(), sub.to_numpy()]), lw=1.0)
        ax.set_title(f"{name} um", fontsize=9, pad=10)
    fig.suptitle("Population-mean orientation distribution (image frame)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "orientation_histogram.png", dpi=cfg["plot"]["dpi"])
    plt.close(fig)

    for i in (int(val_idx[0]), 0):
        x = R[i] - R[i].mean()
        P = np.abs(fft2(x * win)) ** 2 / win_energy
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
        for ax, (lo, hi, name) in zip(axes, bands):
            msk = np.isfinite(lam) & (lam >= lo) & \
                ((lam < hi) if hi < 1e8 else True)
            img = np.zeros((nx, ny))
            img[msk] = P[msk]
            img = np.fft.fftshift(img)
            ax.imshow(np.log10(img + 1e-12), origin="lower", cmap="viridis")
            ax.set_title(f"#{i} {name} um (log PSD)", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(out / "example_directional_psd" / f"sample_{i:03d}.png",
                    dpi=cfg["plot"]["dpi"])
        plt.close(fig)

    (out / "README.md").write_text(README, encoding="utf-8")
    missing = [f for f in EXPECTED if not (out / f).exists()]
    p25.require(not missing, f"missing outputs: {missing}")
    p25.log(f"11 done in {time.time() - t0:.1f}s; all {len(EXPECTED)} outputs present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
