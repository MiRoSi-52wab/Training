### SAME THING AS visualize_dataset
### only difference is the color range on the color bar

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ============================================================
# CONFIGURATION — only edit these two lines
# ============================================================
DATASET_PATH = "data/raw/dataset_v1.h5"
N_SAMPLES    = 4          # number of random samples to display
# ============================================================

FIELD_LABELS = {
    "eps_star":   [r"$\varepsilon_{11}$", r"$\varepsilon_{22}$", r"$\gamma_{12}$"],
    "tau_star":   [r"$\tau_{11}$",        r"$\tau_{22}$",        r"$\tau_{12}$"],
    "sigma_star": [r"$\sigma_{11}$",      r"$\sigma_{22}$",      r"$\sigma_{12}$"],
}

CMAPS = {
    "phase":      "gray",
    "C_field":    "viridis",
    "eps_star":   "RdBu_r",
    "tau_star":   "RdBu_r",
    "sigma_star": "RdBu_r",
}


def _symmetric_clim(data: np.ndarray):
    """Return (-v, v) colormap limits centred on zero."""
    v = np.abs(data).max()
    return -v, v


def print_dataset_summary(f: h5py.File, path: str) -> None:
    """Print metadata and basic statistics to the terminal."""
    meta = f["metadata"]
    n      = meta.attrs.get("n_samples", "?")
    N      = meta.attrs.get("N", "?")
    n_tr   = len(meta.attrs.get("train_idx", []))
    n_val  = len(meta.attrs.get("val_idx", []))
    n_te   = len(meta.attrs.get("test_idx", []))

    print("=" * 60)
    print(f"Dataset : {path}")
    print(f"Samples : {n}  (train={n_tr}, val={n_val}, test={n_te})")
    print(f"Grid    : {N}×{N}")
    print("-" * 60)
    for key in ("eps_bar", "eps_star", "tau_star", "sigma_star"):
        arr = f[key][:]
        print(f"{key:12s}  shape={arr.shape}  "
              f"min={arr.min():.4e}  max={arr.max():.4e}  "
              f"mean={arr.mean():.4e}")
    print(f"{'n_iter':12s}  "
          f"min={f['n_iter'][:].min()}  "
          f"max={f['n_iter'][:].max()}  "
          f"mean={f['n_iter'][:].mean():.1f}")
    conv = f["converged"][:].mean() * 100
    print(f"{'converged':12s}  {conv:.1f}%")
    vf = f["phase"][:].mean(axis=(-2, -1))
    print(f"{'vol. frac.':12s}  min={vf.min():.3f}  max={vf.max():.3f}  mean={vf.mean():.3f}")
    print("=" * 60)


def plot_sample(f: h5py.File, idx: int, fig_idx: int) -> None:
    """
    Plot one sample: phase, C₁₁₁₁, and all three field arrays (eps, tau, sigma).
    Layout: 3 rows × 4 columns
      Row 0: phase | eps_11 | tau_11 | sigma_11
      Row 1: C_1111| eps_22 | tau_22 | sigma_22
      Row 2: eps_bar text | eps_12 | tau_12 | sigma_12
    """
    phase      = f["phase"][idx]              # (N, N)
    C_field    = f["C_field"][idx, 0, 0]     # C_1111  (N, N)
    eps_bar    = f["eps_bar"][idx]            # (3,)
    eps_star   = f["eps_star"][idx]          # (3, N, N)
    tau_star   = f["tau_star"][idx]          # (3, N, N)
    sigma_star = f["sigma_star"][idx]        # (3, N, N)
    n_iter     = f["n_iter"][idx]
    converged  = f["converged"][idx]

    fields = [eps_star, tau_star, sigma_star]
    field_keys = ["eps_star", "tau_star", "sigma_star"]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"Sample #{idx}  |  "
        r"$\bar{\varepsilon}$"
        f"=[{eps_bar[0]:.4f}, {eps_bar[1]:.4f}, {eps_bar[2]:.4f}]  |  "
        f"iter={n_iter}  conv={'✓' if converged else '✗'}",
        fontsize=12, fontweight="bold"
    )

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    def _imshow(ax, data, title, cmap, clim=None):
        im = ax.imshow(data.T, origin="lower", cmap=cmap,
                       vmin=clim[0] if clim else None,
                       vmax=clim[1] if clim else None)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Column 0: geometry
    _imshow(fig.add_subplot(gs[0, 0]), phase,   "Phase (inclusion=1)",  CMAPS["phase"])
    _imshow(fig.add_subplot(gs[1, 0]), C_field, r"$C_{1111}(x)$",       CMAPS["C_field"])

    # eps_bar text panel
    ax_text = fig.add_subplot(gs[2, 0])
    ax_text.axis("off")
    info = (
        r"$\bar{\varepsilon}_{11}$" + f" = {eps_bar[0]:.5f}\n"
        r"$\bar{\varepsilon}_{22}$" + f" = {eps_bar[1]:.5f}\n"
        r"$\bar{\gamma}_{12}$"      + f" = {eps_bar[2]:.5f}\n\n"
        f"VF = {phase.mean():.3f}\n"
        f"iters = {n_iter}\n"
        f"conv = {'yes' if converged else 'NO'}"
    )
    ax_text.text(0.5, 0.5, info, ha="center", va="center",
                 transform=ax_text.transAxes, fontsize=9,
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # Columns 1-3: field components
    for col, (fdata, fkey) in enumerate(zip(fields, field_keys), start=1):
        labels = FIELD_LABELS[fkey]
        cmap   = CMAPS[fkey]
        for row in range(3):
            ax = fig.add_subplot(gs[row, col])
            clim = _symmetric_clim(fdata[row])
            _imshow(ax, fdata[row], labels[row], cmap, clim)

    # Column headers
    col_titles = [r"$\varepsilon^*(x)$  [strain]",
                  r"$\tau^*(x)$  [polariz. stress]",
                  r"$\sigma^*(x)$  [stress]"]
    for col, title in enumerate(col_titles, start=1):
        fig.text(
            (col + 0.5) / 4, 0.97, title,
            ha="center", va="top", fontsize=10, fontstyle="italic"
        )

    plt.savefig(
        Path(DATASET_PATH).parent.parent / "visualize" /
        f"sample_{idx:04d}.png",
        dpi=120, bbox_inches="tight"
    )
    plt.show()


def plot_dataset_overview(f: h5py.File) -> None:
    """
    Single-figure overview:
      - Histogram of inclusion volume fractions
      - Histogram of iteration counts
      - Distribution of eps_bar components
      - Mean ± std of tau_star across all samples
    """
    vf     = f["phase"][:].mean(axis=(-2, -1))
    niters = f["n_iter"][:]
    eps_b  = f["eps_bar"][:]          # (N_samples, 3)
    tau    = f["tau_star"][:]         # (N_samples, 3, N, N)

    tau_mean = tau.mean(axis=(-2, -1))  # spatial mean per sample, (N_s, 3)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Dataset Overview", fontsize=13, fontweight="bold")

    # Row 0 ----------------------------------------------------------------
    axes[0, 0].hist(vf, bins=30, color="steelblue", edgecolor="white")
    axes[0, 0].set_xlabel("Inclusion volume fraction")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Volume fraction distribution")

    axes[0, 1].hist(niters, bins=range(niters.min(), niters.max() + 2),
                    color="salmon", edgecolor="white", align="left")
    axes[0, 1].set_xlabel("FFT iterations to convergence")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Iteration count distribution")

    labels_eb = [r"$\bar{\varepsilon}_{11}$",
                 r"$\bar{\varepsilon}_{22}$",
                 r"$\bar{\gamma}_{12}$"]
    for k, (col, lbl) in enumerate(zip(["steelblue", "salmon", "seagreen"], labels_eb)):
        axes[0, 2].hist(eps_b[:, k], bins=40, alpha=0.65, label=lbl, color=col)
    axes[0, 2].set_xlabel("Strain component value")
    axes[0, 2].set_ylabel("Count")
    axes[0, 2].set_title(r"$\bar{\varepsilon}$ distribution")
    axes[0, 2].legend(fontsize=8)

    # Row 1 ----------------------------------------------------------------
    labels_tau = [r"$\langle\tau_{11}\rangle$",
                  r"$\langle\tau_{22}\rangle$",
                  r"$\langle\tau_{12}\rangle$"]
    for k, (col, lbl) in enumerate(zip(["steelblue", "salmon", "seagreen"], labels_tau)):
        axes[1, k].hist(tau_mean[:, k], bins=40, color=col, edgecolor="white")
        axes[1, k].set_xlabel("Spatial mean value")
        axes[1, k].set_ylabel("Count")
        axes[1, k].set_title(f"Sample-wise spatial mean of {lbl}")
        axes[1, k].axvline(0, color="k", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(
        Path(DATASET_PATH).parent.parent / "visualize" / "overview.png",
        dpi=120, bbox_inches="tight"
    )
    plt.show()


def main():
    path = Path(DATASET_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with h5py.File(path, "r") as f:
        n_total = f["metadata"].attrs.get("n_samples", len(f["phase"]))

        print_dataset_summary(f, str(path))

        # Pick N_SAMPLES random indices
        rng = np.random.default_rng(0)
        indices = rng.choice(n_total, size=min(N_SAMPLES, n_total), replace=False)
        indices = sorted(indices)

        print(f"\nPlotting {len(indices)} sample(s): {indices}")
        for i, idx in enumerate(indices):
            plot_sample(f, int(idx), i)

        print("\nPlotting dataset overview…")
        plot_dataset_overview(f)

    print("Done. Figures saved to data/visualize/")


if __name__ == "__main__":
    main()
