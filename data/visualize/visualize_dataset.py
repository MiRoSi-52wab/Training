"""
Visualize samples from a generated HDF5 micromechanics dataset.

Change DATASET_PATH, N_SAMPLES and BASIS below, then run:
  python data/visualize/visualize_dataset.py
"""

import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from pathlib import Path

# Make utils/ importable when run as a script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.notation import convert_fields, LABELS

# ============================================================
# CONFIGURATION — only edit these three lines
# ============================================================
DATASET_PATH = "data/raw/dataset_v1.h5"
N_SAMPLES    = 4           # number of random samples to display
BASIS        = "voigt"     # "voigt" or "mandel"
# ============================================================

# Pull display labels for the chosen basis.
FIELD_LABELS   = {k: LABELS[BASIS][k] for k in ("eps_star", "tau_star", "sigma_star")}
EPS_BAR_LABELS = LABELS[BASIS]["eps_bar"]

CMAPS = {
    "phase":      "gray",
    "C_field":    "viridis",
    "eps_star":   "RdBu_r",
    "tau_star":   "RdBu_r",
    "sigma_star": "RdBu_r",
}


def _centered_norm(data: np.ndarray, center: float) -> mcolors.TwoSlopeNorm:
    """
    Build a TwoSlopeNorm that places `center` at the midpoint of the colormap.
    vmin and vmax are chosen symmetrically so the colormap diverges evenly
    from the center value, showing how much each pixel deviates from it.
    A small epsilon guards against degenerate (constant) fields.
    """
    eps = 1e-12
    deviation = max(abs(float(data.max()) - center),
                    abs(float(data.min()) - center),
                    eps)
    vmin = center - deviation
    vmax = center + deviation
    # TwoSlopeNorm requires vmin < vcenter < vmax strictly
    return mcolors.TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)


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
    print(f"Basis   : {BASIS}")
    print("-" * 60)

    # Load every component-bearing field once, optionally convert to Mandel.
    # All arrays here have a batch axis in front, so the component axis is 1.
    batch_fields = {key: f[key][:] for key in ("eps_bar", "eps_star", "tau_star", "sigma_star")}
    batch_fields = convert_fields(batch_fields, "voigt", BASIS,
                                   strain_axis=1, stress_axis=1)
    for key, arr in batch_fields.items():
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
    C_field    = f["C_field"][idx, 0, 0]     # C_1111  (N, N) — invariant under Voigt/Mandel
    eps_bar    = f["eps_bar"][idx]            # (3,)
    eps_star   = f["eps_star"][idx]          # (3, N, N)
    tau_star   = f["tau_star"][idx]          # (3, N, N)
    sigma_star = f["sigma_star"][idx]        # (3, N, N)
    n_iter     = f["n_iter"][idx]
    converged  = f["converged"][idx]

    # Convert from Voigt (the storage basis) to the requested display basis.
    _converted = convert_fields(
        {"eps_bar": eps_bar, "eps_star": eps_star,
         "tau_star": tau_star, "sigma_star": sigma_star},
        "voigt", BASIS,
    )
    eps_bar    = _converted["eps_bar"]
    eps_star   = _converted["eps_star"]
    tau_star   = _converted["tau_star"]
    sigma_star = _converted["sigma_star"]

    fields     = [eps_star, tau_star, sigma_star]
    field_keys = ["eps_star", "tau_star", "sigma_star"]

    fig = plt.figure(figsize=(16, 10))
    basis_tag = "" if BASIS == "voigt" else f"  |  basis: {BASIS}"
    fig.suptitle(
        f"Sample #{idx}  |  "
        r"$\bar{\varepsilon}$"
        f"=[{eps_bar[0]:.4f}, {eps_bar[1]:.4f}, {eps_bar[2]:.4f}]  |  "
        f"iter={n_iter}  conv={'✓' if converged else '✗'}{basis_tag}",
        fontsize=12, fontweight="bold"
    )

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    def _imshow(ax, data, title, cmap, norm=None):
        im = ax.imshow(data.T, origin="lower", cmap=cmap, norm=norm)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Column 0: geometry (no diverging norm needed)
    _imshow(fig.add_subplot(gs[0, 0]), phase,   "Phase (inclusion=1)",  CMAPS["phase"])
    _imshow(fig.add_subplot(gs[1, 0]), C_field, r"$C_{1111}(x)$",       CMAPS["C_field"])

    # Colormap centers per field type and component:
    #   eps_star  → center on eps_bar[component]  (prescribed macroscopic strain)
    #   tau_star  → center on spatial mean of that component
    #   sigma_star→ center on spatial mean of that component
    centers = {
        "eps_star":   [float(eps_bar[a])             for a in range(3)],
        "tau_star":   [float(tau_star[a].mean())     for a in range(3)],
        "sigma_star": [float(sigma_star[a].mean())   for a in range(3)],
    }

    # eps_bar text panel
    ax_text = fig.add_subplot(gs[2, 0])
    ax_text.axis("off")
    info = (
        EPS_BAR_LABELS[0] + f" = {eps_bar[0]:.5f}\n" +
        EPS_BAR_LABELS[1] + f" = {eps_bar[1]:.5f}\n" +
        EPS_BAR_LABELS[2] + f" = {eps_bar[2]:.5f}\n\n" +
        f"VF = {phase.mean():.3f}\n" +
        f"iters = {n_iter}\n" +
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
            ax   = fig.add_subplot(gs[row, col])
            norm = _centered_norm(fdata[row], centers[fkey][row])
            _imshow(ax, fdata[row], labels[row], cmap, norm)

    # Column headers
    col_titles = [r"$\varepsilon^*(x)$  [strain]",
                  r"$\tau^*(x)$  [polariz. stress]",
                  r"$\sigma^*(x)$  [stress]"]
    for col, title in enumerate(col_titles, start=1):
        fig.text(
            (col + 0.5) / 4, 0.97, title,
            ha="center", va="top", fontsize=10, fontstyle="italic"
        )

    suffix = "" if BASIS == "voigt" else f"_{BASIS}"
    plt.savefig(
        Path(DATASET_PATH).parent.parent / "visualize" /
        f"sample_{idx:04d}{suffix}.png",
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

    # Convert to chosen basis (component axis = 1 for batched arrays).
    _converted = convert_fields(
        {"eps_bar": eps_b, "tau_star": tau},
        "voigt", BASIS, strain_axis=1, stress_axis=1,
    )
    eps_b = _converted["eps_bar"]
    tau   = _converted["tau_star"]

    tau_mean = tau.mean(axis=(-2, -1))  # spatial mean per sample, (N_s, 3)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    basis_tag = "" if BASIS == "voigt" else f"  (basis: {BASIS})"
    fig.suptitle(f"Dataset Overview{basis_tag}", fontsize=13, fontweight="bold")

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

    labels_eb = EPS_BAR_LABELS
    for k, (col, lbl) in enumerate(zip(["steelblue", "salmon", "seagreen"], labels_eb)):
        axes[0, 2].hist(eps_b[:, k], bins=40, alpha=0.65, label=lbl, color=col)
    axes[0, 2].set_xlabel("Strain component value")
    axes[0, 2].set_ylabel("Count")
    axes[0, 2].set_title(r"$\bar{\varepsilon}$ distribution")
    axes[0, 2].legend(fontsize=8)

    # Row 1 ----------------------------------------------------------------
    labels_tau = [rf"$\langle$ {lbl} $\rangle$" for lbl in FIELD_LABELS["tau_star"]]
    for k, (col, lbl) in enumerate(zip(["steelblue", "salmon", "seagreen"], labels_tau)):
        axes[1, k].hist(tau_mean[:, k], bins=40, color=col, edgecolor="white")
        axes[1, k].set_xlabel("Spatial mean value")
        axes[1, k].set_ylabel("Count")
        axes[1, k].set_title(f"Sample-wise spatial mean of {lbl}")
        axes[1, k].axvline(0, color="k", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    suffix = "" if BASIS == "voigt" else f"_{BASIS}"
    plt.savefig(
        Path(DATASET_PATH).parent.parent / "visualize" / f"overview{suffix}.png",
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
