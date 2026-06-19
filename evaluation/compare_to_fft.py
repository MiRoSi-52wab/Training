"""
Compare trained KANTauTheta model against FFT ground truth on the test split.

Loads best_checkpoint.pt, runs the LSFNO forward pass on every test sample,
and reports per-sample relative L2 field errors against the FFT-converged eps_star.

Usage (CLI):
    python -m evaluation.compare_to_fft \
        --checkpoint /path/to/best_checkpoint.pt \
        --data      /path/to/dataset_v3.h5 \
        --plot

Importable:
    from evaluation.compare_to_fft import evaluate
    results = evaluate(checkpoint_path, data_path, plot=True)
"""

import argparse
import numpy as np
import torch
from pathlib import Path

from symbolic.recover import load_model_from_checkpoint
from datasets.micromechanics import MicromechanicsDataset


def evaluate(
    checkpoint_path: str,
    data_path: str,
    split: str = "test",
    batch_size: int = 16,
    plot: bool = False,
) -> dict:
    """
    Run the trained LSFNO on the test split and compare to FFT ground truth.

    Returns dict with keys:
        rel_l2_per_sample  — (N,) array of per-sample relative L2 field errors
        mean_rel_l2        — scalar mean
        median_rel_l2      — scalar median
        p90_rel_l2         — 90th percentile
        max_rel_l2         — worst-case sample
        passed             — bool: mean error < 0.1%
    """
    model, config, epoch = load_model_from_checkpoint(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    use_checkpointing = bool(config.get("use_checkpointing", False))

    dataset = MicromechanicsDataset(data_path, split=split)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    errors = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            eps_pred = model(
                batch["C_field"], batch["eps_bar"],
                use_checkpointing=use_checkpointing,
            )
            eps_star = batch["eps_star"]
            # Relative L2 per sample: ||eps_pred - eps_star||_F / ||eps_star||_F
            diff  = (eps_pred - eps_star).reshape(eps_pred.shape[0], -1)
            ref   = eps_star.reshape(eps_star.shape[0], -1)
            rel_l2 = (diff.norm(dim=1) / ref.norm(dim=1).clamp(min=1e-12)).cpu().numpy()
            errors.append(rel_l2)

    errors = np.concatenate(errors)

    mean_e   = float(errors.mean())
    median_e = float(np.median(errors))
    p90_e    = float(np.percentile(errors, 90))
    max_e    = float(errors.max())
    passed   = mean_e < 1e-3

    print(f"\n{'='*60}")
    print(f"  Model vs FFT comparison — epoch {epoch}  ({split} split)")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")
    print(f"  Samples:    {len(errors)}")
    print(f"  Mean   rel-L2:  {mean_e:.4%}")
    print(f"  Median rel-L2:  {median_e:.4%}")
    print(f"  p90    rel-L2:  {p90_e:.4%}")
    print(f"  Max    rel-L2:  {max_e:.4%}")
    print(f"\n  Passed (mean < 0.1%): {'✓' if passed else '✗'}")
    print(f"{'='*60}\n")

    if plot:
        _plot(errors, epoch, checkpoint_path)

    return {
        "rel_l2_per_sample": errors,
        "mean_rel_l2":       mean_e,
        "median_rel_l2":     median_e,
        "p90_rel_l2":        p90_e,
        "max_rel_l2":        max_e,
        "passed":            passed,
        "epoch":             epoch,
        "n_samples":         len(errors),
    }


def _plot(errors: np.ndarray, epoch: int, checkpoint_path: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.hist(errors * 100, bins=40, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.axvline(errors.mean() * 100, color="red",    ls="--", lw=1.5, label=f"Mean {errors.mean():.3%}")
    ax.axvline(0.1,                 color="orange",  ls=":",  lw=1.5, label="0.1% threshold")
    ax.set_xlabel("Relative L2 field error (%)")
    ax.set_ylabel("Count")
    ax.set_title(f"Error distribution — epoch {epoch}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(np.sort(errors) * 100, np.linspace(0, 100, len(errors)), "b-", lw=1.5)
    ax.axhline(90, color="gray",   ls="--", lw=0.8, label="p90")
    ax.axhline(99, color="gray",   ls=":",  lw=0.8, label="p99")
    ax.axvline(0.1, color="orange", ls=":", lw=1.5, label="0.1% threshold")
    ax.set_xlabel("Relative L2 field error (%)")
    ax.set_ylabel("Cumulative %")
    ax.set_title("CDF of per-sample errors")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"KAN-FNO vs FFT  (epoch {epoch})", fontsize=12, y=1.01)
    plt.tight_layout()

    out = Path(checkpoint_path).parent / f"model_vs_fft_epoch{epoch}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Plot saved to {out}")
    plt.show()


def _parse_args():
    p = argparse.ArgumentParser(description="Compare trained KAN model to FFT ground truth.")
    p.add_argument("--checkpoint", required=True, help="Path to best_checkpoint.pt")
    p.add_argument("--data",       required=True, help="Path to dataset_v3.h5")
    p.add_argument("--split",      default="test", choices=["train", "val", "test"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--plot",       action="store_true")
    return p.parse_args()


def main():
    args = _parse_args()
    evaluate(args.checkpoint, args.data, split=args.split,
             batch_size=args.batch_size, plot=args.plot)


if __name__ == "__main__":
    main()
