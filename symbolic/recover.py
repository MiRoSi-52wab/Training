"""
Symbolic recovery for Study 2: extract the trained KANTauTheta control
points, verify they match the exact x² representation, and fit the learned
edge function φ(x) against a candidate symbolic library.

This is the final step of the linear sanity check.  A successful run shows:
  • ctrl ≈ [1, −1, 1]  to 3–4 decimal places
  • φ(x) ≈ x²  with max deviation < 1e-3
  • Best symbolic fit: x²  with R² > 0.9999

Usage:
    python -m symbolic.recover \\
        --checkpoint runs/linear_prototype_kan_shared/best_checkpoint.pt \\
        --plot

Importable form (e.g. from a notebook):
    from symbolic.recover import recover
    results = recover("runs/.../best_checkpoint.pt", plot=True)
"""

import argparse
import numpy as np
import torch
from pathlib import Path


# ── Candidate symbolic library ────────────────────────────────────────────────
# Each entry is a function f: np.ndarray → np.ndarray.
# The fitter solves  φ_learned(x) ≈ a·f(x) + b  in least squares.
CANDIDATE_LIBRARY = {
    "x²":    lambda x: x ** 2,
    "x":     lambda x: x,
    "|x|":   lambda x: np.abs(x),
    "const": lambda x: np.ones_like(x),
    "x³":    lambda x: x ** 3,
    "x⁴":    lambda x: x ** 4,
}

EXACT_CTRL = np.array([1.0, -1.0, 1.0])


# ── Checkpoint loading ────────────────────────────────────────────────────────

def load_model_from_checkpoint(checkpoint_path: str):
    """
    Load an LSFNO with KANTauTheta from a .pt file written by Trainer.

    Returns:
        (model, config, epoch) where model is in eval mode on CPU.
    """
    from models.kan_tau_theta import KANTauTheta
    from models.ls_fno import LSFNO

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    dim    = int(config.get("dim", 2))
    n_comp = 3 if dim == 2 else 6
    tau_theta = KANTauTheta(
        R=float(config.get("R", 1.0)),
        shared=bool(config.get("shared", True)),
        trainable=True,
        n_comp=n_comp,
    )
    model = LSFNO.from_config(config, tau_theta=tau_theta)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config, int(ckpt["epoch"])


# ── Core recovery function ────────────────────────────────────────────────────

def recover(
    checkpoint_path: str,
    plot: bool = False,
    n_grid: int = 300,
    delta_ctrl_threshold: float = 0.01,
    r2_threshold: float = 0.9999,
) -> dict:
    """
    Full symbolic recovery pipeline for one checkpoint.

    Args:
        checkpoint_path:      Path to a .pt file saved by Trainer.
        plot:                 If True, save and display a φ(x) comparison plot.
        n_grid:               Number of points in [-1, 1] for evaluating φ(x).
        delta_ctrl_threshold: Max allowed |ctrl − [1,−1,1]| for PASS verdict.
        r2_threshold:         Min R² for the x² candidate for PASS verdict.

    Returns:
        dict with keys:
            ctrl          — trained control points (numpy array)
            delta_ctrl    — per-point absolute deviation from [1,−1,1]
            delta_phi_max — max pointwise |φ_learned(x) − x²| over [-1,1]
            fits          — {candidate_name: {'a','b','r2','rmse'}}
            best_fit      — name of the best-fitting candidate
            best_r2       — R² of the best fit
            passed        — bool: all Study 2 criteria met
            epoch         — training epoch of the checkpoint
    """
    model, config, epoch = load_model_from_checkpoint(checkpoint_path)
    shared = bool(config.get("shared", True))

    ctrl_raw = model.tau_theta.ctrl.detach().cpu()

    # ── Control-point report ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Symbolic Recovery — epoch {epoch}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    if shared:
        ctrl = ctrl_raw.numpy()                    # (3,)
        # The Γ operator kills uniform fields, so φ(x)+C gives identical training loss for
        # any constant C. The null-mode offset is φ(0) = c₀/4 + c₁/2 + c₂/4.
        # Subtract it before comparing to exact [1,−1,1] so that [1.5,−0.5,1.5] → [1,−1,1].
        null_offset = ctrl[0] / 4.0 + ctrl[1] / 2.0 + ctrl[2] / 4.0
        ctrl_normalized = ctrl - null_offset   # projects out the Γ-null mode
        delta_ctrl = np.abs(ctrl_normalized - EXACT_CTRL)
        print(f"\n  Control points (shared across all edges):")
        print(f"    Trained:     [{ctrl[0]:+.6f}, {ctrl[1]:+.6f}, {ctrl[2]:+.6f}]")
        print(f"    Null offset: φ(0) = {null_offset:+.6f}  (Γ-invisible constant)")
        print(f"    Normalized:  [{ctrl_normalized[0]:+.6f}, {ctrl_normalized[1]:+.6f}, {ctrl_normalized[2]:+.6f}]")
        print(f"    Exact:       [{EXACT_CTRL[0]:+.6f}, {EXACT_CTRL[1]:+.6f}, {EXACT_CTRL[2]:+.6f}]")
        print(f"    δ_ctrl:      [{delta_ctrl[0]:.2e}, {delta_ctrl[1]:.2e}, {delta_ctrl[2]:.2e}]"
              f"  max = {delta_ctrl.max():.2e}")
    else:
        ctrl_np = ctrl_raw.numpy()                 # (3, n, n)
        n = ctrl_np.shape[1]
        delta_ctrl_all = np.abs(ctrl_np - EXACT_CTRL[:, None, None])
        delta_ctrl = delta_ctrl_all.max(axis=(1, 2))   # (3,) worst across edges
        ctrl = ctrl_np.mean(axis=(1, 2))               # mean for summary
        print(f"\n  Control points (independent per edge, {n}×{n} = {n*n} sets):")
        print(f"    Mean trained:  [{ctrl[0]:+.6f}, {ctrl[1]:+.6f}, {ctrl[2]:+.6f}]")
        print(f"    Exact:         [{EXACT_CTRL[0]:+.6f}, {EXACT_CTRL[1]:+.6f}, {EXACT_CTRL[2]:+.6f}]")
        print(f"    Max δ_ctrl:    [{delta_ctrl[0]:.2e}, {delta_ctrl[1]:.2e}, {delta_ctrl[2]:.2e}]"
              f"  max = {delta_ctrl.max():.2e}")

    # ── Reconstruct φ(x) on a grid ────────────────────────────────────────────
    x_np  = np.linspace(-1.0, 1.0, n_grid)
    x_t   = torch.from_numpy(x_np).double()

    if shared:
        with torch.no_grad():
            phi_learned = model.tau_theta._phi(x_t).numpy()
    else:
        # For independent edges: evaluate on a (1,1,1,n_grid) shaped input
        # and recover the (0,0) edge as a representative.
        x_in = x_t.reshape(1, 1, 1, n_grid)
        with torch.no_grad():
            phi_all = model.tau_theta._phi(x_in)   # (1, n, n, n_grid)
        phi_learned = phi_all[0, 0, 0].numpy()     # edge (0,0)
        print(f"    (φ(x) plot shows edge (0,0); all {n*n} edges have similar shape)")

    phi_exact = x_np ** 2
    # Compare with the null-offset removed — the physically meaningful difference
    phi_learned_centered = phi_learned - null_offset if shared else phi_learned
    delta_phi_max = float(np.abs(phi_learned_centered - phi_exact).max())

    print(f"\n  φ(x) reconstruction over x ∈ [-1, 1]:")
    print(f"    Null offset removed: {null_offset:+.4f}")
    print(f"    max |φ_centered(x) − x²| = {delta_phi_max:.2e}")

    # ── Fit to candidate library ──────────────────────────────────────────────
    print(f"\n  Candidate library fit  φ(x) ≈ a·f(x) + b:")
    print(f"    {'Candidate':<10}  {'a':>10}  {'b':>12}  {'R²':>10}  {'RMSE':>10}")

    fits = {}
    best_name, best_r2 = None, -np.inf
    for name, fn in CANDIDATE_LIBRARY.items():
        f = fn(x_np)
        A = np.stack([f, np.ones_like(f)], axis=1)
        coeffs, *_ = np.linalg.lstsq(A, phi_learned, rcond=None)
        a, b = float(coeffs[0]), float(coeffs[1])
        pred = a * f + b
        ss_res = float(np.sum((phi_learned - pred) ** 2))
        ss_tot = float(np.sum((phi_learned - phi_learned.mean()) ** 2))
        r2   = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        rmse = float(np.sqrt(ss_res / n_grid))
        fits[name] = {"a": a, "b": b, "r2": r2, "rmse": rmse}
        print(f"    {name:<10}  {a:>10.4f}  {b:>12.6f}  {r2:>10.6f}  {rmse:>10.2e}")
        if r2 > best_r2:
            best_r2, best_name = r2, name

    # ── Pass / fail verdict ───────────────────────────────────────────────────
    x2_r2 = fits["x²"]["r2"]
    passed = (
        delta_ctrl.max() < delta_ctrl_threshold
        and x2_r2 >= r2_threshold
        and best_name == "x²"
    )

    print(f"\n  Best symbolic fit: {best_name}  (R² = {best_r2:.6f})")
    print(f"\n  Study 2 criteria:")
    print(f"    δ_ctrl.max() < {delta_ctrl_threshold}  →  {delta_ctrl.max():.2e}  "
          f"{'✓' if delta_ctrl.max() < delta_ctrl_threshold else '✗'}")
    print(f"    R²(x²) ≥ {r2_threshold}         →  {x2_r2:.6f}  "
          f"{'✓' if x2_r2 >= r2_threshold else '✗'}")
    print(f"    best_fit == 'x²'        →  {best_name}  "
          f"{'✓' if best_name == 'x²' else '✗'}")
    verdict = "PASS ✓" if passed else "FAIL ✗"
    print(f"\n  Overall verdict: {verdict}")
    print(f"{'='*60}\n")

    if plot:
        _plot(x_np, phi_learned, phi_learned_centered, phi_exact,
              ctrl, ctrl_normalized, null_offset, delta_ctrl, epoch, checkpoint_path)

    return {
        "ctrl":          ctrl,
        "delta_ctrl":    delta_ctrl,
        "delta_phi_max": delta_phi_max,
        "fits":          fits,
        "best_fit":      best_name,
        "best_r2":       best_r2,
        "passed":        passed,
        "epoch":         epoch,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot(x, phi_learned, phi_learned_centered, phi_exact,
          ctrl, ctrl_normalized, null_offset, delta_ctrl, epoch, checkpoint_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left panel: centered learned vs exact φ(x)
    ax = axes[0]
    ax.plot(x, phi_exact,           "k-",  lw=2, label="Exact:  $x^2$")
    ax.plot(x, phi_learned,         "r:",  lw=1.5,
            label=f"Learned (raw): ctrl=[{ctrl[0]:.3f}, {ctrl[1]:.3f}, {ctrl[2]:.3f}]")
    ax.plot(x, phi_learned_centered,"r--", lw=2,
            label=f"Learned (−{null_offset:+.3f}): ctrl=[{ctrl_normalized[0]:.3f}, "
                  f"{ctrl_normalized[1]:.3f}, {ctrl_normalized[2]:.3f}]")
    ax.set_xlabel("x")
    ax.set_ylabel("φ(x)")
    ax.set_title(f"Edge function — epoch {epoch}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right panel: residual after null-offset removal
    residual = phi_learned_centered - phi_exact
    ax = axes[1]
    ax.plot(x, residual, "b-", lw=1.5)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.fill_between(x, residual, alpha=0.15, color="blue")
    ax.set_xlabel("x")
    ax.set_ylabel("$\\phi_{\\mathrm{centered}}(x) - x^2$")
    ax.set_title(f"Residual after null-offset removal  (max = {np.abs(residual).max():.2e})")
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Study 2 — Symbolic Recovery  (epoch {epoch})", fontsize=12, y=1.01)
    plt.tight_layout()

    out = Path(checkpoint_path).parent / f"symbolic_recovery_epoch{epoch}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Plot saved to {out}")
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Study 2 symbolic recovery.")
    p.add_argument("--checkpoint", required=True,
                   help="Path to a .pt checkpoint file written by Trainer.")
    p.add_argument("--plot", action="store_true",
                   help="Save and display a φ(x) vs x² comparison plot.")
    p.add_argument("--n_grid", type=int, default=300,
                   help="Grid points for φ(x) evaluation (default 300).")
    p.add_argument("--delta_ctrl", type=float, default=0.01,
                   help="Max |ctrl−[1,−1,1]| for PASS verdict (default 0.01).")
    p.add_argument("--r2_min", type=float, default=0.9999,
                   help="Min R²(x²) for PASS verdict (default 0.9999).")
    return p.parse_args()


def main():
    args = _parse_args()
    recover(
        args.checkpoint,
        plot=args.plot,
        n_grid=args.n_grid,
        delta_ctrl_threshold=args.delta_ctrl,
        r2_threshold=args.r2_min,
    )


if __name__ == "__main__":
    main()
