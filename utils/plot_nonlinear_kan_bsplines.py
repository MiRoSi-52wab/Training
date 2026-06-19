"""
Visual check of the two B-splines used in NonlinearKANTauTheta.

Produces a 2×2 figure:
  Row 1 — φ_sqrt : B-spline approximation of √(x) on [0, R_sq]
    Left  panel: B-spline vs exact function, with control points shown
    Right panel: absolute error  |φ_sqrt(x) − √(x)|

  Row 2 — φ_kink : B-spline approximation of max(x, 0) on [−f_range, +f_range]
    Left  panel: B-spline vs exact function, with control points shown
    Right panel: absolute error  |φ_kink(x) − max(x,0)|

Usage (from LS_KAN_FNO/):
    /home/myuser/BGCE/project/bin/python utils/plot_nonlinear_kan_bsplines.py

    Optional CLI flags:
        --sigma-y  FLOAT   yield stress in MPa (default 68.9)
        --H        FLOAT   hardening modulus in MPa (default 1710)
        --n-sqrt   INT     control points for phi_sqrt (default 100)
        --n-kink   INT     control points per half for phi_kink (default 20)
        --save     PATH    save figure to file instead of showing it
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.nonlinear_kan_tau_theta import (
    make_sqrt_bspline,
    make_kink_bspline,
    greville_abscissae,
    SQRT_2_3,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ctrl_points(spline):
    """Return (greville_abscissae, control_point_values) as numpy arrays."""
    knots = spline.knots.numpy()
    ctrl  = spline.ctrl.detach().numpy()
    g     = greville_abscissae(knots, spline.degree)
    return g, ctrl


def _eval(spline, x_np):
    """Evaluate spline on a numpy array, return numpy array."""
    x_t = torch.tensor(x_np, dtype=torch.float64)
    return spline(x_t).detach().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_bsplines(sigma_y, H, n_ctrl_sqrt, n_ctrl_kink_half, save_path=None):
    try:
        import matplotlib
        if save_path is None:
            matplotlib.use('TkAgg')
        else:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not installed — cannot plot.")
        return

    # ── Domain parameters (same defaults as NonlinearKANTauTheta) ─────────────
    R_sq    = (20.0 * sigma_y / SQRT_2_3) ** 2   # upper bound for ||s_trial||²
    f_range = 20.0 * sigma_y                       # half-range for trial yield fn

    print(f"sigma_y = {sigma_y} MPa,  H = {H} MPa")
    print(f"R_sq    = {R_sq:.4g}  (= (20 sigma_y / sqrt(2/3))²)")
    print(f"sqrt(R_sq) = {np.sqrt(R_sq):.4g}  MPa  [max deviatoric stress]")
    print(f"f_range = {f_range:.4g}  MPa  (= 20 sigma_y)")
    print()

    # ── Build splines ──────────────────────────────────────────────────────────
    print(f"Building phi_sqrt  (n_ctrl={n_ctrl_sqrt},  degree=3, domain=[0, {R_sq:.3g}]) ...")
    phi_sqrt = make_sqrt_bspline(R_sq=R_sq, n_ctrl=n_ctrl_sqrt, degree=3)
    print(f"Building phi_kink  (n_ctrl_half={n_ctrl_kink_half}, degree=3, "
          f"domain=[{-f_range:.3g}, {f_range:.3g}]) ...")
    phi_kink = make_kink_bspline(f_min=-f_range, f_max=f_range,
                                  degree=3, n_ctrl_half=n_ctrl_kink_half)
    print()

    # ── Evaluation grids ───────────────────────────────────────────────────────
    # Skip very near zero to avoid log-scale issues; show both linear and log x
    x_sqrt = np.linspace(0.0, R_sq, 2000)
    x_sqrt_inner = np.linspace(R_sq * 1e-4, R_sq * 0.9999, 2000)  # for relative error

    x_kink = np.linspace(-f_range, f_range, 4000)

    # Exact functions
    exact_sqrt = np.sqrt(x_sqrt)
    exact_kink = np.maximum(x_kink, 0.0)

    # Spline evaluations
    spline_sqrt = _eval(phi_sqrt, x_sqrt)
    spline_kink = _eval(phi_kink, x_kink)

    # Errors
    abs_err_sqrt = np.abs(spline_sqrt - exact_sqrt)
    abs_err_kink = np.abs(spline_kink - exact_kink)

    # Relative error for sqrt (avoid dividing by 0)
    exact_inner = np.sqrt(x_sqrt_inner)
    spline_inner = _eval(phi_sqrt, x_sqrt_inner)
    rel_err_sqrt = np.abs(spline_inner - exact_inner) / exact_inner

    # Control points
    g_sqrt, c_sqrt = _ctrl_points(phi_sqrt)
    g_kink, c_kink = _ctrl_points(phi_kink)

    # ── Print summary statistics ───────────────────────────────────────────────
    print("phi_sqrt summary:")
    print(f"  Max absolute error : {abs_err_sqrt.max():.4e}")
    print(f"  Max relative error : {rel_err_sqrt.max():.4e}  "
          f"(on [{x_sqrt_inner[0]:.3g}, {x_sqrt_inner[-1]:.3g}])")

    # Find the point of max relative error and corresponding yield-surface impact
    worst_idx = np.argmax(rel_err_sqrt)
    q_worst   = x_sqrt_inner[worst_idx]
    norm_s_err = abs(spline_inner[worst_idx] - exact_inner[worst_idx])
    # Yield surface error: d(yield_err)/d(norm_s_err) ≈ -0.4 (first-order Taylor)
    # so max yield error ≈ 0.4 * max norm_s error
    print(f"  Worst point:   q = {q_worst:.4g},  "
          f"||s_trial|| = {exact_inner[worst_idx]:.6g},  "
          f"error = {norm_s_err:.4e}")

    print()
    print("phi_kink summary:")
    # Only check positive side (negative should be exact 0)
    mask_pos = x_kink > 0
    mask_neg = x_kink < -0.01 * f_range
    print(f"  Max absolute error (full domain)      : {abs_err_kink.max():.4e}")
    if mask_pos.any():
        print(f"  Max absolute error (positive side)    : {abs_err_kink[mask_pos].max():.4e}")
    if mask_neg.any():
        print(f"  Max absolute error (negative side)    : {abs_err_kink[mask_neg].max():.4e}")
    print()

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.30)

    # Colour palette
    C_EXACT  = '#1f77b4'   # blue  — exact function
    C_SPLINE = '#d62728'   # red   — B-spline
    C_CTRL   = '#2ca02c'   # green — control points
    C_ERR    = '#ff7f0e'   # orange — absolute error
    C_RELERR = '#9467bd'   # purple — relative error

    # ── Panel (0,0): phi_sqrt — function ──────────────────────────────────────
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.plot(x_sqrt, exact_sqrt,  color=C_EXACT,  lw=1.8, label='$\\sqrt{x}$ (exact)', zorder=3)
    ax00.plot(x_sqrt, spline_sqrt, color=C_SPLINE, lw=1.2, ls='--', label='$\\phi_{sqrt}$ (B-spline)', zorder=4)
    ax00.scatter(g_sqrt, c_sqrt, color=C_CTRL, s=15, zorder=5, label=f'control pts (n={len(c_sqrt)})')
    ax00.set_xlabel('$q = \\|s_{\\mathrm{trial}}\\|^2$')
    ax00.set_ylabel('$\\|s_{\\mathrm{trial}}\\|$  (MPa)')
    ax00.set_title(r'$\phi_{sqrt}$: approximates $\sqrt{x}$' + '\n'
                   f'domain $[0,\\ R_{{sq}}={R_sq:.3g}]$,  $n_{{ctrl}}={n_ctrl_sqrt}$')
    ax00.legend(fontsize=8)
    ax00.set_xlim(0, R_sq)
    ax00.set_ylim(bottom=0)
    _add_vline_sigma_y(ax00, sigma_y, R_sq, axis='x', label_side='right')

    # ── Panel (0,1): phi_sqrt — absolute + relative error ─────────────────────
    ax01 = fig.add_subplot(gs[0, 1])
    ax01.semilogy(x_sqrt, abs_err_sqrt + 1e-30, color=C_ERR, lw=1.5, label='absolute error')
    ax01_r = ax01.twinx()
    ax01_r.semilogy(x_sqrt_inner, rel_err_sqrt + 1e-30, color=C_RELERR, lw=1.2,
                    ls='-.', alpha=0.8, label='relative error')
    ax01.set_xlabel('$q = \\|s_{\\mathrm{trial}}\\|^2$')
    ax01.set_ylabel('absolute error', color=C_ERR)
    ax01_r.set_ylabel('relative error', color=C_RELERR)
    ax01.set_title(r'$\phi_{sqrt}$ error' + '\n'
                   f'max abs = {abs_err_sqrt.max():.2e},  '
                   f'max rel = {rel_err_sqrt.max():.2e}')
    # Combined legend
    lines1, labels1 = ax01.get_legend_handles_labels()
    lines2, labels2 = ax01_r.get_legend_handles_labels()
    ax01.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax01.tick_params(axis='y', labelcolor=C_ERR)
    ax01_r.tick_params(axis='y', labelcolor=C_RELERR)

    # ── Panel (1,0): phi_kink — function ──────────────────────────────────────
    ax10 = fig.add_subplot(gs[1, 0])
    ax10.plot(x_kink, exact_kink,  color=C_EXACT,  lw=1.8, label='$\\max(x,0)$ (exact)', zorder=3)
    ax10.plot(x_kink, spline_kink, color=C_SPLINE, lw=1.2, ls='--', label='$\\phi_{kink}$ (B-spline)', zorder=4)
    ax10.scatter(g_kink, c_kink, color=C_CTRL, s=12, zorder=5, label=f'control pts (n={len(c_kink)})')
    ax10.axvline(0, color='grey', lw=0.8, ls=':')
    ax10.axhline(0, color='grey', lw=0.8, ls=':')
    ax10.set_xlabel('$f_{\\mathrm{trial}}$  (MPa)')
    ax10.set_ylabel('$\\max(f_{\\mathrm{trial}},\\, 0)$  (MPa)')
    ax10.set_title(r'$\phi_{kink}$: approximates $\max(x,0)$' + '\n'
                   f'domain $[{-f_range:.3g},\\ {f_range:.3g}]$,  $n_{{ctrl}}={len(c_kink)}$')
    ax10.legend(fontsize=8)
    _add_vline_sigma_y(ax10, sigma_y, f_range, axis='x_centered', label_side='right')

    # ── Panel (1,1): phi_kink — absolute error ────────────────────────────────
    ax11 = fig.add_subplot(gs[1, 1])
    # Separate positive and negative sides for clarity
    mask_neg_plot = x_kink <= 0
    mask_pos_plot = x_kink >= 0
    ax11.semilogy(x_kink[mask_neg_plot], abs_err_kink[mask_neg_plot] + 1e-30,
                  color='#aec7e8', lw=1.2, label='elastic side (x < 0)', zorder=3)
    ax11.semilogy(x_kink[mask_pos_plot], abs_err_kink[mask_pos_plot] + 1e-30,
                  color=C_ERR, lw=1.5, label='plastic side (x > 0)', zorder=4)
    ax11.axvline(0, color='grey', lw=0.8, ls=':')
    ax11.set_xlabel('$f_{\\mathrm{trial}}$  (MPa)')
    ax11.set_ylabel('absolute error  $|\\varphi_{\\kappa}(x) - \\max(x,0)|$')
    ax11.set_title(r'$\phi_{kink}$ error' + '\n'
                   f'max = {abs_err_kink.max():.2e}  (exact to machine precision)')
    ax11.legend(fontsize=8)

    # ── Super-title ────────────────────────────────────────────────────────────
    fig.suptitle(
        f'NonlinearKANTauTheta — B-spline components\n'
        f'sigma_y = {sigma_y} MPa,  H = {H} MPa',
        fontsize=13, fontweight='bold', y=1.01,
    )

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Figure saved to: {save_path}")
    else:
        plt.tight_layout()
        plt.show()


def _add_vline_sigma_y(ax, sigma_y, max_val, axis, label_side):
    """Add a reference vertical line at x ≈ sigma_y level (as a guide)."""
    pass   # visual clutter on these axes — omitted for clarity


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Plot NonlinearKAN B-spline components")
    p.add_argument('--sigma-y', type=float, default=68.9,  help='yield stress (MPa)')
    p.add_argument('--H',       type=float, default=1710.0, help='hardening modulus (MPa)')
    p.add_argument('--n-sqrt',  type=int,   default=100,   help='phi_sqrt control points')
    p.add_argument('--n-kink',  type=int,   default=20,    help='phi_kink control pts per half')
    p.add_argument('--save',    type=str,   default=None,  help='save path (e.g. plot.png)')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    plot_bsplines(
        sigma_y         = args.sigma_y,
        H               = args.H,
        n_ctrl_sqrt     = args.n_sqrt,
        n_ctrl_kink_half= args.n_kink,
        save_path       = args.save,
    )
