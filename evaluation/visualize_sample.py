"""
Field visualization for a single sample: microstructure, FFT solution,
KAN-FNO prediction, and per-pixel error colormaps.

Layout (4 rows × 3 columns):
  Row 0: microstructure C₁₁₁₁(x)  |  applied ε̄ text  |  rel-L2 error scalar
  Row 1: FFT solution  ε₁₁ / ε₂₂ / 2ε₁₂
  Row 2: KAN-FNO pred  ε₁₁ / ε₂₂ / 2ε₁₂   (same color range as Row 1)
  Row 3: |error|       ε₁₁ / ε₂₂ / 2ε₁₂   (sequential colormap, own scale)

Usage (CLI):
    python -m evaluation.visualize_sample \
        --checkpoint /path/to/best_checkpoint.pt \
        --data       /path/to/dataset_v3.h5 \
        --sample_idx 0          # index within the test split (default: 0)
        --save /path/to/out.png

Importable:
    from evaluation.visualize_sample import visualize
    visualize(checkpoint_path, data_path, sample_idx=0, save_path="sample.png")
"""

import argparse
import numpy as np
import torch
from pathlib import Path

from symbolic.recover import load_model_from_checkpoint
from datasets.micromechanics import MicromechanicsDataset


COMP_LABELS = [r"$\varepsilon_{11}$", r"$\varepsilon_{22}$", r"$2\varepsilon_{12}$"]


def visualize(
    checkpoint_path: str,
    data_path: str,
    sample_idx: int = 0,
    split: str = "test",
    save_path: str = None,
) -> dict:
    """
    Plot microstructure + FFT + KAN-FNO + error for one sample.

    Args:
        checkpoint_path: Path to best_checkpoint.pt from Trainer.
        data_path:       Path to dataset_v3.h5.
        sample_idx:      Index within the chosen split (0 = first test sample).
        split:           'train', 'val', or 'test'.
        save_path:       If given, save PNG there; otherwise shown interactively.

    Returns:
        dict with 'rel_l2', 'eps_bar', 'ctrl'.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        raise ImportError("matplotlib is required for visualization.")

    model, config, epoch = load_model_from_checkpoint(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    use_ckpt = bool(config.get("use_checkpointing", False))

    dataset = MicromechanicsDataset(data_path, split=split)
    if sample_idx >= len(dataset):
        raise IndexError(f"sample_idx={sample_idx} >= split size {len(dataset)}")

    batch = dataset[sample_idx]
    batch = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v)
             for k, v in batch.items()}

    with torch.no_grad():
        eps_pred = model(batch["C_field"], batch["eps_bar"],
                         use_checkpointing=use_ckpt)

    # Move everything to numpy, squeeze batch dim
    eps_fft  = batch["eps_star"].squeeze(0).cpu().numpy()   # (3, N, N)
    eps_kan  = eps_pred.squeeze(0).cpu().numpy()            # (3, N, N)
    C_field  = batch["C_field"].squeeze(0).cpu().numpy()    # (3, 3, N, N)
    eps_bar  = batch["eps_bar"].squeeze(0).cpu().numpy()    # (3,)

    error    = eps_kan - eps_fft                            # (3, N, N)
    abs_err  = np.abs(error)

    ref_norm  = np.linalg.norm(eps_fft.reshape(-1))
    pred_norm = np.linalg.norm(error.reshape(-1))
    rel_l2    = float(pred_norm / max(ref_norm, 1e-12))

    # Stiffness map: C₁₁₁₁(x) — distinguishes inclusion (high) from matrix (low)
    C11 = C_field[0, 0]                                     # (N, N)

    ctrl = model.tau_theta.ctrl.detach().cpu().numpy()

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 12))
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Row 0: microstructure, ε̄ text, summary
    ax_micro = fig.add_subplot(gs[0, 0])
    im = ax_micro.imshow(C11, cmap="gray", origin="lower")
    plt.colorbar(im, ax=ax_micro, fraction=0.046, pad=0.04)
    ax_micro.set_title(r"Microstructure  $C_{1111}(x)$", fontsize=10)
    ax_micro.axis("off")

    ax_text = fig.add_subplot(gs[0, 1])
    ax_text.axis("off")
    loading_txt = (
        f"Applied macroscopic strain  ε̄\n"
        f"  ε₁₁ = {eps_bar[0]:+.5f}\n"
        f"  ε₂₂ = {eps_bar[1]:+.5f}\n"
        f"2ε₁₂ = {eps_bar[2]:+.5f}\n\n"
        f"Loading: random (all components\nindependently ∼ U[−0.01, +0.01])\n\n"
        f"Model epoch: {epoch}\n"
        f"ctrl: [{ctrl[0]:+.3f}, {ctrl[1]:+.3f}, {ctrl[2]:+.3f}]"
    )
    ax_text.text(0.05, 0.95, loading_txt, transform=ax_text.transAxes,
                 fontsize=9, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    ax_sum = fig.add_subplot(gs[0, 2])
    ax_sum.axis("off")
    verdict = "PASS ✓" if rel_l2 < 1e-3 else "FAIL ✗"
    summary_txt = (
        f"Relative L2 field error\n"
        f"  ‖ε_KAN − ε_FFT‖ / ‖ε_FFT‖\n\n"
        f"  {rel_l2:.4%}\n\n"
        f"  Threshold: 0.1%\n"
        f"  {verdict}"
    )
    color = "lightgreen" if rel_l2 < 1e-3 else "lightsalmon"
    ax_sum.text(0.05, 0.95, summary_txt, transform=ax_sum.transAxes,
                fontsize=10, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor=color, alpha=0.9))

    # Rows 1–3: field colormaps
    row_labels = ["FFT ground truth", "KAN-FNO prediction", "Absolute error"]
    fields     = [eps_fft, eps_kan, abs_err]
    cmaps      = ["RdBu_r", "RdBu_r", "Reds"]

    for row_i, (label, field, cmap) in enumerate(zip(row_labels, fields, cmaps)):
        for col_i in range(3):
            ax = fig.add_subplot(gs[row_i + 1, col_i])
            comp = field[col_i]

            if row_i < 2:
                # Symmetric range for strain fields
                vmax = max(np.abs(eps_fft[col_i]).max(), 1e-10)
                vmin = -vmax
            else:
                vmin = 0.0
                vmax = max(abs_err[col_i].max(), 1e-10)

            im = ax.imshow(comp, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            title = f"{label}\n{COMP_LABELS[col_i]}" if col_i == 0 else COMP_LABELS[col_i]
            ax.set_title(title, fontsize=9)
            ax.axis("off")

    fig.suptitle(
        f"Study 2 — Field visualization  (sample {sample_idx} of {split} split, epoch {epoch})",
        fontsize=12, y=1.01,
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)
    return {"rel_l2": rel_l2, "eps_bar": eps_bar, "ctrl": ctrl}


def _parse_args():
    p = argparse.ArgumentParser(description="Visualize one sample: FFT vs KAN-FNO.")
    p.add_argument("--checkpoint",  required=True, help="Path to best_checkpoint.pt")
    p.add_argument("--data",        required=True, help="Path to dataset_v3.h5")
    p.add_argument("--sample_idx",  type=int, default=0,
                   help="Index within the split (default: 0 = first test sample)")
    p.add_argument("--split",       default="test", choices=["train", "val", "test"])
    p.add_argument("--save",        default=None,
                   help="Output PNG path. If omitted, shows interactively.")
    return p.parse_args()


def main():
    args = _parse_args()
    save = args.save or str(
        Path(args.checkpoint).parent / f"sample_vis_{args.split}_{args.sample_idx}.png"
    )
    visualize(args.checkpoint, args.data,
              sample_idx=args.sample_idx, split=args.split, save_path=save)


if __name__ == "__main__":
    main()
