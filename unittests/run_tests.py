"""
Run analytical test cases against any solver.

Usage
-----
# Test the FFT solver (default, no extra arguments needed):
  python -m unittests.run_tests

# Also save comparison plots for each case:
  python -m unittests.run_tests --visualize

# Display scalars and plots in Mandel basis (solver still runs in Voigt):
  python -m unittests.run_tests --basis mandel --visualize

# Test the analytic LS-FNO (Yarotsky, no training needed) with dynamic depth:
  python -m unittests.run_tests --model ls_fno --mode solve --visualize

# Test LS-FNO with fixed K iterations (forward mode):
  python -m unittests.run_tests --model ls_fno --mode forward --visualize

# Test a trained LS-FNO checkpoint:
  python -m unittests.run_tests --model ls_fno --checkpoint models/checkpoints/ls_fno.pt

Solver interface contract
-------------------------
Any solver must be a callable with signature:
    result = solver(C_field: np.ndarray, eps_bar: np.ndarray) -> dict
where:
    C_field   — (3, 3, N, N) float64 Voigt stiffness field
    eps_bar   — (3,)         float64 macroscopic strain (engineering shear)
    result    — dict with required keys:
                  'eps_star'   : (3, N, N) float64
                  'tau_star'   : (3, N, N) float64
                  'sigma_star' : (3, N, N) float64
                and optional keys (used for CLI output and plots):
                  'n_iter'    : int  — iterations to convergence
                  'converged' : bool — whether the tolerance was reached

For the FFT solver this is just generation.fft_solver.solve().
For neural models, use the --model flag which wraps the forward/solve pass.
"""

import sys
import argparse
import numpy as np
import h5py
from pathlib import Path
from typing import Callable

# ─────────────────────────────────────────────────────────────────────────────
# Test thresholds — edit here to tighten or relax checks globally
# ─────────────────────────────────────────────────────────────────────────────
# Maximum coefficient of variation of the dominant strain component inside the
# Eshelby inclusion.  0.20 accounts for Gibbs oscillations at the discrete
# staircase boundary of the circle; shear loading requires slightly more margin
# than normal loading because the shear field has no Poisson coupling to soften
# the gradient at the boundary.
ESHELBY_CV_TOL = 0.20

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generation.fft_solver import solve as fft_solve
from utils.notation import (
    convert_fields,
    voigt_to_mandel_strain,
    INV_SQRT2,
    SQRT2,
    LABELS,
)
from utils.config_loader import load_config

# ── Load FFT solver runtime config once at import time ────────────────────────
try:
    _FFT_CFG_PATH = Path(__file__).resolve().parents[1] / "configs" / "fft_solver.yaml"
    _FFT_CFG = load_config(_FFT_CFG_PATH)
except Exception:
    _FFT_CFG = {}   # fall back to fft_solver.py defaults if yaml is unavailable

_FFT_TOL      = float(_FFT_CFG.get("tol",      1e-5))
_FFT_MAX_ITER = int(  _FFT_CFG.get("max_iter", 2000))


# ─────────────────────────────────────────────────────────────────────────────
# Solver wrappers
# ─────────────────────────────────────────────────────────────────────────────

def fft_solver_wrapper(C_field: np.ndarray, eps_bar: np.ndarray) -> dict:
    """Wrap fft_solve to match the standard solver interface.

    Convergence settings are read from configs/fft_solver.yaml so they stay
    in sync with ls_fno.yaml for the iteration-count comparison study.
    """
    result = fft_solve(
        C_field.astype(np.float64), eps_bar.astype(np.float64),
        tol=_FFT_TOL, max_iter=_FFT_MAX_ITER,
    )
    return {
        "eps_star":   result["eps_star"],
        "tau_star":   result["tau_star"],
        "sigma_star": result["sigma_star"],
        "n_iter":     result["n_iter"],
        "converged":  result["converged"],
    }



def load_nn_solver(model_name: str, checkpoint: str, mode: str = "solve") -> Callable:
    """
    Load an LS-FNO model and return a numpy-compatible solver wrapper.

    Args:
        model_name:  Currently only 'ls_fno' is supported.
        checkpoint:  Path to a .pt state-dict file, or 'analytic' / None for
                     the parameter-free Yarotsky construction (no training needed).
        mode:        'forward' (fixed K iterations) or 'solve' (dynamic depth,
                     residual-based stopping — matches fft_solver behaviour).
    """
    if model_name != "ls_fno":
        raise NotImplementedError(
            f"Model '{model_name}' is not supported. "
            "Use '--model ls_fno' or '--model fft_solver'."
        )

    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required: pip install torch")

    from models.ls_fno import LSFNO

    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "ls_fno.yaml"
    cfg = load_config(cfg_path)

    model = LSFNO.from_config(cfg)

    # Consistency check: model grid size must match the experiment config.
    cfg_N     = int(cfg.get("N", cfg.get("grid_size", -1)))
    model_N   = model.gamma_hat_M.shape[-1]
    if cfg_N != -1 and cfg_N != model_N:
        raise ValueError(
            f"Grid size mismatch: experiment N={cfg_N} but model was built "
            f"with N={model_N}.  Check configs/experiment.yaml and ls_fno.yaml."
        )

    ckpt_path = None
    if checkpoint and checkpoint.lower() not in ("none", "analytic"):
        ckpt_path = Path(checkpoint)

    if ckpt_path is not None and ckpt_path.exists():
        state = torch.load(str(ckpt_path), map_location="cpu")
        model.load_state_dict(state)
        print(f"  Loaded checkpoint: {ckpt_path}")
    elif ckpt_path is not None:
        print(f"  WARNING: checkpoint not found ({ckpt_path}); "
              "using analytic (Yarotsky) weights.")

    model.eval()

    def nn_solver(C_field: np.ndarray, eps_bar: np.ndarray) -> dict:
        C_t  = torch.from_numpy(C_field.astype(np.float32)).unsqueeze(0)  # (1,3,3,N,N)
        eb_t = torch.from_numpy(eps_bar.astype(np.float32)).unsqueeze(0)  # (1,3)

        with torch.no_grad():
            if mode == "solve":
                out       = model.solve(C_t, eb_t)
                eps_V     = out["eps_star"][0].numpy()   # (3, N, N)
                n_iter    = out["n_iter"]
                converged = out["converged"]
            else:  # forward — fixed K iterations
                eps_V     = model(C_t, eb_t)[0].numpy()
                n_iter    = model.K + 1   # embedding + K layers
                converged = True          # fixed-depth always completes

        # Derive tau and sigma from eps_star in Voigt notation.
        #   sigma = C : eps   (independent of reference stiffness — always correct)
        #   tau   = (C - C⁰) : eps
        #
        # C⁰ must use the same alpha0 estimation as fft_solver.solve() does,
        # so that tau is comparable to the stored test-case references.
        # fft_solver estimates: alpha0 = (C_1111.max() + C_1111.min()) / 2
        # model.alpha_0 is calibrated for a specific κ range and differs
        # per-sample (e.g. κ=1 homogeneous → alpha0≈1.35, κ=10 → alpha0≈7.4,
        # but model.alpha_0=9.81 for all). Using model.alpha_0 here would give
        # a systematically wrong tau for every test case with a different κ.
        C00        = C_field[0, 0]                                    # C_1111 (N, N)
        alpha0_ref = (float(C00.max()) + float(C00.min())) / 2.0     # matches fft_solver
        C0_V       = alpha0_ref * np.diag([1.0, 1.0, 0.5])
        eps_64     = eps_V.astype(np.float64)
        dC         = C_field.astype(np.float64) - C0_V[:, :, np.newaxis, np.newaxis]
        tau_star   = np.einsum("abxy,bxy->axy", dC, eps_64)
        sigma_star = np.einsum("abxy,bxy->axy", C_field.astype(np.float64), eps_64)

        return {
            "eps_star":   eps_64,
            "tau_star":   tau_star,
            "sigma_star": sigma_star,
            "n_iter":     n_iter,
            "converged":  converged,
        }

    return nn_solver


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def relative_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    """Relative L2 error: ‖pred - ref‖₂ / ‖ref‖₂. Returns 0 if ref is zero."""
    denom = np.linalg.norm(ref)
    if denom < 1e-14:
        return float(np.linalg.norm(pred - ref))
    return float(np.linalg.norm(pred - ref) / denom)


def max_abs_error(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.abs(pred - ref).max())


def phase_mean(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean of (3,N,N) field over pixels where mask is True. Returns (3,)."""
    return field[:, mask].mean(axis=1)


def uniformity_cv(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Coefficient of variation (std/|mean|) of each component inside `mask`.
    Small CV → field is approximately uniform inside that region.
    Returns (3,) array; components where mean ≈ 0 return std instead.
    """
    vals = field[:, mask]    # (3, n_pixels)
    mu   = vals.mean(axis=1)
    std  = vals.std(axis=1)
    cv   = np.where(np.abs(mu) > 1e-12, std / np.abs(mu), std)
    return cv


# ─────────────────────────────────────────────────────────────────────────────
# Per-case tests
# ─────────────────────────────────────────────────────────────────────────────

def _test_homogeneous(pred: dict, grp: h5py.Group, tol: float) -> dict:
    """
    Checks for homogeneous test cases:
    1. ε(x) = ε̄ everywhere          (relative L2 error < tol)
    2. τ(x) = 0 everywhere           (max absolute error < tol × ‖ε̄‖)
    3. Solver converges in ≤ 3 iters (informational)
    """
    eps_bar = grp["eps_bar"][:]
    ana_eps = grp["analytical/eps_star"][:]
    ana_tau = grp["analytical/tau_star"][:]

    err_eps = relative_l2(pred["eps_star"], ana_eps)
    err_tau = max_abs_error(pred["tau_star"], ana_tau)

    # τ reference is zero, so use ‖ε̄‖ to normalise
    eps_scale = float(np.linalg.norm(eps_bar))
    tau_pass  = err_tau < tol * eps_scale
    eps_pass  = err_eps < tol

    return {
        "pass":    eps_pass and tau_pass,
        "details": {
            "rel_L2_eps":  f"{err_eps:.2e}  ({'✓' if eps_pass else '✗'} < {tol:.0e})",
            "max_abs_tau": f"{err_tau:.2e}  ({'✓' if tau_pass else '✗'} < {tol*eps_scale:.0e})",
        },
        "scalars": {
            "eps_bar":        eps_bar,
            "rel_L2_eps":     err_eps,
            "max_abs_tau":    err_tau,
        },
    }


def _test_laminate(pred: dict, grp: h5py.Group, tol: float) -> dict:
    """
    Checks for laminate test cases:
    1. ε₁₁(x) is uniform (CV < tol)
    2. Mean ε₂₂ per phase matches analytical prediction
    3. σ₂₂ is uniform (CV < tol) — isostress condition
    4. σ₁₂ is uniform (CV < tol) — isostress condition
    5. Overall relative L2 error on τ < tol
    """
    phase = grp["phase"][:].astype(bool)
    inc, mat = phase, ~phase

    ana = grp["analytical"]

    # Uniformity checks
    cv_eps11  = uniformity_cv(pred["eps_star"][0:1], np.ones(phase.shape, bool))[0]
    cv_sig22  = uniformity_cv(pred["sigma_star"][1:2], np.ones(phase.shape, bool))[0]
    cv_sig12  = uniformity_cv(pred["sigma_star"][2:3], np.ones(phase.shape, bool))[0]

    # Phase-averaged ε₂₂
    eps22_inc_pred = float(pred["eps_star"][1, inc].mean())
    eps22_mat_pred = float(pred["eps_star"][1, mat].mean())
    eps22_inc_ref  = float(ana.attrs["eps22_inc"])
    eps22_mat_ref  = float(ana.attrs["eps22_mat"])

    err_eps22_inc = abs(eps22_inc_pred - eps22_inc_ref) / (abs(eps22_inc_ref) + 1e-14)
    err_eps22_mat = abs(eps22_mat_pred - eps22_mat_ref) / (abs(eps22_mat_ref) + 1e-14)

    # σ₂₂ reference and predicted means
    sigma22_ref  = float(ana.attrs["sigma22_bar"])
    sigma22_pred = float(pred["sigma_star"][1].mean())

    # Overall τ error
    err_tau = relative_l2(pred["tau_star"], ana["tau_star"][:])

    passes = {
        "eps11_uniform": cv_eps11  < tol,
        "sig22_uniform": cv_sig22  < tol,
        "sig12_uniform": cv_sig12  < tol,
        "eps22_inc":     err_eps22_inc < tol,
        "eps22_mat":     err_eps22_mat < tol,
        "tau_L2":        err_tau   < tol,
    }

    return {
        "pass": all(passes.values()),
        "details": {
            "CV(ε₁₁) in domain":  f"{cv_eps11:.2e}  ({'✓' if passes['eps11_uniform'] else '✗'} < {tol:.0e})",
            "CV(σ₂₂) in domain":  f"{cv_sig22:.2e}  ({'✓' if passes['sig22_uniform'] else '✗'} < {tol:.0e})",
            "CV(σ₁₂) in domain":  f"{cv_sig12:.2e}  ({'✓' if passes['sig12_uniform'] else '✗'} < {tol:.0e})",
            "rel err ε₂₂ inc":    f"{err_eps22_inc:.2e}  ({'✓' if passes['eps22_inc'] else '✗'} < {tol:.0e})",
            "rel err ε₂₂ mat":    f"{err_eps22_mat:.2e}  ({'✓' if passes['eps22_mat'] else '✗'} < {tol:.0e})",
            "rel L2 τ":           f"{err_tau:.2e}  ({'✓' if passes['tau_L2'] else '✗'} < {tol:.0e})",
        },
        "scalars": {
            "sigma22_bar_analytical": sigma22_ref,
            "sigma22_bar_predicted":  sigma22_pred,
            "eps22_inc_analytical":   eps22_inc_ref,
            "eps22_inc_predicted":    eps22_inc_pred,
            "eps22_mat_analytical":   eps22_mat_ref,
            "eps22_mat_predicted":    eps22_mat_pred,
            "gam12_inc_analytical":   float(ana.attrs.get("gam12_inc", 0.0)),
            "gam12_inc_predicted":    float(pred["eps_star"][2, inc].mean()),
            "gam12_mat_analytical":   float(ana.attrs.get("gam12_mat", 0.0)),
            "gam12_mat_predicted":    float(pred["eps_star"][2, mat].mean()),
        },
    }


def _test_eshelby(pred: dict, grp: h5py.Group, tol: float, basis: str = "voigt") -> dict:
    """
    Checks for the single circular inclusion (Eshelby) test cases:

    1. Dominant-component CV inside inclusion < 0.15
       Only the loading component (argmax |ε̄|) is checked.  Off-axis
       components (e.g. ε₂₂ under e11 load) legitimately vary inside a
       periodic inclusion due to near-field images, and their near-zero
       mean makes the standard CV formula blow up.

    2. Mean strain inside inclusion ≈ A : ε̄ (relative error < tol)
       The dilute Eshelby formula is only approximate for a periodic cell,
       so tol is set generously (default 0.15).

    3. Stress magnitude higher inside inclusion than in matrix (qualitative).

    4. rel L2 of τ vs dilute approximation (informational).
    """
    phase = grp["phase"][:].astype(bool)
    inc, mat = phase, ~phase
    ana = grp["analytical"]
    eps_bar = grp["eps_bar"][:]

    # Dominant loading component — the one where |ε̄| is largest.
    dominant = int(np.argmax(np.abs(eps_bar)))
    dom_field = pred["eps_star"][dominant:dominant+1]   # (1, N, N)
    cv_dom = float(uniformity_cv(dom_field, inc)[0])

    eps_inc_pred = phase_mean(pred["eps_star"], inc)
    eps_mat_pred = phase_mean(pred["eps_star"], mat)
    eps_inc_ref  = ana["eps_inc_mean"][:]

    rel_err_mean = np.linalg.norm(eps_inc_pred - eps_inc_ref) / (np.linalg.norm(eps_inc_ref) + 1e-14)

    sig_inc = np.linalg.norm(pred["sigma_star"][:, inc], axis=0).mean()
    sig_mat = np.linalg.norm(pred["sigma_star"][:, mat], axis=0).mean()
    stress_higher = sig_inc > sig_mat

    err_tau = relative_l2(pred["tau_star"], ana["tau_star"][:])

    comp_name = LABELS[basis]["eps_star_text"][dominant]
    passes = {
        "dominant_strain_uniform":  cv_dom < ESHELBY_CV_TOL,
        "mean_eps_inc_vs_eshelby":  rel_err_mean < tol,
        "stress_higher_in_inc":     stress_higher,
    }

    return {
        "pass": all(passes.values()),
        "details": {
            f"CV({comp_name}) inside inc (dominant)":
                f"{cv_dom:.3f}  ({'✓' if passes['dominant_strain_uniform'] else '✗'} < {ESHELBY_CV_TOL})",
            "rel err mean ε_inc":
                f"{rel_err_mean:.2e}  ({'✓' if passes['mean_eps_inc_vs_eshelby'] else '✗'} < {tol:.0e})",
            "‖σ‖_inc > ‖σ‖_mat":
                f"{sig_inc:.3e} > {sig_mat:.3e}  ({'✓' if stress_higher else '✗'})",
            "rel L2 τ (informational)": f"{err_tau:.2e}",
        },
        "scalars": {
            "eps_inc_eshelby": eps_inc_ref,
            "eps_inc_fft":     eps_inc_pred,
            "eps_mat_fft":     eps_mat_pred,
            "eps_bar":         eps_bar,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Value comparison printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_scalar_comparison(kind: str, scalars: dict, basis: str = "voigt") -> None:
    """
    Print a table of analytical vs predicted values for the test case.

    The pass/fail checks always run in Voigt internally; this printer only
    affects how scalar quantities are *displayed*.  In Mandel:
      - shear-strain values (γ₁₂ → √2·ε₁₂) are multiplied by 1/√2
      - strain vectors are converted via voigt_to_mandel_strain
      - normal components and stress means (σ̄₂₂) are unchanged
    """
    shear_label = LABELS[basis]["shear_strain"]   # "γ₁₂" or "√2·ε₁₂"
    eps_components = LABELS[basis]["eps_star_text"]

    basis_tag = "" if basis == "voigt" else f" (basis: {basis})"
    print(f"    {'─'*60}")
    print(f"    Analytical vs Predicted values:{basis_tag}")

    if kind == "homo":
        eps_bar = scalars["eps_bar"]
        if basis == "mandel":
            eps_bar = voigt_to_mandel_strain(eps_bar)
        print(f"      ε̄ = [{eps_bar[0]:.5f}, {eps_bar[1]:.5f}, {eps_bar[2]:.5f}]")
        print(f"      rel L2(ε)  = {scalars['rel_L2_eps']:.2e}  (should be 0)")
        print(f"      max|τ|     = {scalars['max_abs_tau']:.2e}  (should be 0)")

    elif kind == "laminate":
        # Shear-strain scalars stored as γ₁₂; rescale to √2·ε₁₂ for Mandel display.
        shear_scale = INV_SQRT2 if basis == "mandel" else 1.0
        print(f"      {'Quantity':<26}  {'Analytical':>14}  {'Predicted':>14}")
        print(f"      {'─'*56}")
        pairs = [
            ("σ̄₂₂",                       "sigma22_bar_analytical",  "sigma22_bar_predicted",  1.0),
            ("ε₂₂ (inc)",                  "eps22_inc_analytical",    "eps22_inc_predicted",    1.0),
            ("ε₂₂ (mat)",                  "eps22_mat_analytical",    "eps22_mat_predicted",    1.0),
            (f"{shear_label} (inc)",       "gam12_inc_analytical",    "gam12_inc_predicted",    shear_scale),
            (f"{shear_label} (mat)",       "gam12_mat_analytical",    "gam12_mat_predicted",    shear_scale),
        ]
        for name, key_a, key_p, scale in pairs:
            a = scalars[key_a] * scale
            p = scalars[key_p] * scale
            print(f"      {name:<26}  {a:>14.6e}  {p:>14.6e}")

    elif kind == "eshelby":
        eps_bar  = scalars["eps_bar"]
        eps_ref  = scalars["eps_inc_eshelby"]
        eps_pred = scalars["eps_inc_fft"]
        eps_mat  = scalars["eps_mat_fft"]
        if basis == "mandel":
            eps_bar  = voigt_to_mandel_strain(eps_bar)
            eps_ref  = voigt_to_mandel_strain(eps_ref)
            eps_pred = voigt_to_mandel_strain(eps_pred)
            eps_mat  = voigt_to_mandel_strain(eps_mat)
        comp_names = eps_components
        print(f"      ε̄ = [{eps_bar[0]:.5f}, {eps_bar[1]:.5f}, {eps_bar[2]:.5f}]")
        print(f"      {'Comp':<10}  {'Eshelby (dilute)':>18}  {'FFT mean (inc)':>16}  {'FFT mean (mat)':>16}")
        print(f"      {'─'*64}")
        for i, cname in enumerate(comp_names):
            print(f"      {cname:<10}  {eps_ref[i]:>18.6e}  {eps_pred[i]:>16.6e}  {eps_mat[i]:>16.6e}")


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def _visualize_case(pred: dict, grp: h5py.Group, case_name: str, passed: bool,
                    save_dir: Path, basis: str = "voigt",
                    test_result: dict = None, solver_tag: str = "") -> None:
    """
    Save a comparison figure: analytical vs predicted eps_star (all 3 components).

    Layout: 3 rows (ε₁₁, ε₂₂, shear) × 4 columns
      Col 0 row 0  : phase image
      Col 0 rows 1-2: solver info (n_iter, converged) + metrics table
      Cols 1-3     : analytical | predicted | error

    The basis only affects what is *displayed* — the underlying solver always
    runs in Voigt.  In Mandel mode the shear row is rescaled by 1/√2 (the strain
    rule) and labels show √2·ε₁₂ instead of γ₁₂.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.colors as mcolors

    phase    = grp["phase"][:].astype(bool)
    eps_bar  = grp["eps_bar"][:]
    ana_eps  = grp["analytical/eps_star"][:].astype(float)
    pred_eps = pred["eps_star"]

    if basis == "mandel":
        eps_bar  = voigt_to_mandel_strain(eps_bar)
        ana_eps  = voigt_to_mandel_strain(ana_eps)
        pred_eps = voigt_to_mandel_strain(pred_eps)

    diff_eps    = pred_eps - ana_eps
    comp_labels = LABELS[basis]["eps_star"]
    n_iter      = pred.get("n_iter")
    converged   = pred.get("converged")

    def _centered_cmap(data, center=0.0):
        eps = 1e-12
        dev = max(abs(float(data.max()) - center), abs(float(data.min()) - center), eps)
        return mcolors.TwoSlopeNorm(vcenter=center, vmin=center - dev, vmax=center + dev)

    # Title: include iteration count if available
    iter_tag  = ""
    if n_iter is not None:
        conv_sym = "✓" if converged else "✗"
        iter_tag = f"   |   iter={n_iter}  conv={conv_sym}"
    basis_tag = "" if basis == "voigt" else f"   |   basis: {basis}"

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        f"Test: {case_name}   |   "
        r"$\bar{\varepsilon}$"
        f"=[{eps_bar[0]:.4f}, {eps_bar[1]:.4f}, {eps_bar[2]:.4f}]   |   "
        f"{'PASS ✓' if passed else 'FAIL ✗'}{iter_tag}{basis_tag}",
        fontsize=12, fontweight="bold",
    )

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.35)

    def _show(ax, data, title, cmap="RdBu_r", norm=None):
        im = ax.imshow(data.T, origin="lower", cmap=cmap, norm=norm)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format="%.2e")

    for row, label in enumerate(comp_labels):
        # Column 0: phase image for row 0 only; rows 1-2 use a merged panel below.
        if row == 0:
            _show(fig.add_subplot(gs[0, 0]),
                  phase.astype(float), "Phase (inclusion=1)", cmap="gray")

        center = float(eps_bar[row])
        _show(fig.add_subplot(gs[row, 1]), ana_eps[row],
              f"Analytical {label}", norm=_centered_cmap(ana_eps[row], center))
        _show(fig.add_subplot(gs[row, 2]), pred_eps[row],
              f"Predicted {label}",  norm=_centered_cmap(pred_eps[row], center))
        _show(fig.add_subplot(gs[row, 3]), diff_eps[row],
              f"Error (pred − ana) {label}", norm=_centered_cmap(diff_eps[row], 0.0))

    # ── Info + metrics panel spanning rows 1-2 of column 0 ───────────────────
    ax_info = fig.add_subplot(gs[1:3, 0])
    ax_info.axis("off")
    info_lines = []
    if n_iter is not None:
        conv_str = "yes ✓" if converged else "NO ✗"
        info_lines += [f"n_iter:    {n_iter}", f"converged: {conv_str}", ""]
    if test_result is not None and "details" in test_result:
        info_lines += ["Metrics", "─" * 26]
        for metric, value in test_result["details"].items():
            info_lines += [f"{metric}:", f"  {value}"]
    if info_lines:
        facecolor = "palegreen" if passed else "lightsalmon"
        ax_info.text(0.5, 0.5, "\n".join(info_lines),
                     ha="center", va="center",
                     transform=ax_info.transAxes,
                     fontsize=7.5, family="monospace",
                     bbox=dict(boxstyle="round", facecolor=facecolor, alpha=0.85))

    # Column headers
    for col, title in enumerate(["", "Analytical", "Predicted", "Error (pred − ana)"], 0):
        if col == 0:
            continue
        fig.text((col + 0.5) / 4, 0.97, title,
                 ha="center", va="top", fontsize=10, fontstyle="italic")

    save_dir.mkdir(parents=True, exist_ok=True)
    basis_suffix  = "" if basis == "voigt" else f"_{basis}"
    solver_suffix = f"_{solver_tag}" if solver_tag else ""
    out = save_dir / f"{case_name}{solver_suffix}{basis_suffix}.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"    → Plot saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main test runner
# ─────────────────────────────────────────────────────────────────────────────

CASE_TESTERS = {
    "homo":     _test_homogeneous,
    "laminate": _test_laminate,
    "eshelby":  _test_eshelby,
}


def run_all_tests(solver: Callable,
                  test_file: str = "unittests/test_cases.h5",
                  solver_name: str = "fft_solver",
                  visualize: bool = False,
                  save_dir: str = "unittests/plots",
                  basis: str = "voigt",
                  solver_tag: str = "") -> bool:
    """
    Run all test cases and print a summary table.
    Returns True if all tests pass, False otherwise.

    The `basis` argument only affects how scalar values and the visualization
    plots are displayed.  The solver always runs in Voigt internally; pass/fail
    metrics are basis-invariant ratios so they don't change.
    """
    path = Path(test_file)
    if not path.exists():
        print(f"ERROR: test file not found — {path}")
        print("Run: python -m unittests.generate_test_cases")
        return False

    out_dir = Path(save_dir)
    total = 0
    passed = 0
    failed_cases = []

    col_w = 28

    print("=" * 70)
    print(f"  Test suite: {path.name}   Solver: {solver_name}   Basis: {basis}")
    print("=" * 70)

    with h5py.File(path, "r") as f:
        for case_name in sorted(f.keys()):
            grp  = f[case_name]
            kind = case_name.split("_")[0]
            tol  = float(grp.attrs["tolerance"])
            desc = grp.attrs["description"]

            C_field = grp["C_field"][:].astype(np.float64)
            eps_bar = grp["eps_bar"][:].astype(np.float64)

            pred = solver(C_field, eps_bar)

            tester = CASE_TESTERS.get(kind)
            if tester is None:
                print(f"  [{case_name}]  — no tester for kind='{kind}', skipping")
                continue

            # Only the eshelby tester customises labels by basis; others ignore it.
            if kind == "eshelby":
                result = tester(pred, grp, tol, basis=basis)
            else:
                result = tester(pred, grp, tol)
            total += 1
            status = "PASS" if result["pass"] else "FAIL"
            if result["pass"]:
                passed += 1
            else:
                failed_cases.append(case_name)

            print(f"\n  {'─'*66}")
            print(f"  {case_name:<{col_w}}  [{status}]  (tol={tol:.0e})")
            print(f"  {desc[:66]}")
            for metric, value in result["details"].items():
                print(f"    {metric:<35}  {value}")

            n_iter_val = pred.get("n_iter")
            converged_val = pred.get("converged")
            if n_iter_val is not None:
                conv_str = "yes" if converged_val else "NO"
                print(f"    {'n_iter':<35}  {n_iter_val}  (converged: {conv_str})")

            _print_scalar_comparison(kind, result["scalars"], basis=basis)

            if visualize:
                _visualize_case(pred, grp, case_name, result["pass"], out_dir,
                                basis=basis, test_result=result,
                                solver_tag=solver_tag)

    print(f"\n{'=' * 70}")
    print(f"  Results: {passed}/{total} passed")
    if failed_cases:
        print(f"  Failed:  {', '.join(failed_cases)}")
    print("=" * 70)

    return passed == total


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Run analytical test cases against a solver.")
    p.add_argument("--model",      default="fft_solver",
                   help="Solver to test: 'fft_solver' or 'ls_fno'.")
    p.add_argument("--checkpoint", default=None,
                   help="Path to a .pt checkpoint file, or 'analytic' for the "
                        "parameter-free Yarotsky construction (default when omitted).")
    p.add_argument("--mode",       default="solve", choices=["forward", "solve"],
                   help="LS-FNO inference mode: 'forward' (fixed K iters) or "
                        "'solve' (dynamic, residual-based stopping). "
                        "Ignored for fft_solver.")
    p.add_argument("--test_file",  default="unittests/test_cases.h5",
                   help="Path to the HDF5 test-cases file.")
    p.add_argument("--visualize",  action="store_true",
                   help="Save a comparison plot (analytical vs predicted) for each test case.")
    p.add_argument("--save_dir",   default="unittests/plots",
                   help="Directory for comparison plots (used with --visualize).")
    p.add_argument("--basis",      default="voigt", choices=["voigt", "mandel"],
                   help="Notation for displaying scalars and plots. "
                        "The solver always runs in Voigt internally.")
    args = p.parse_args()

    if args.model == "fft_solver":
        solver = fft_solver_wrapper
        solver_name = "FFT (Moulinec-Suquet)"
        solver_tag  = "fft"
    else:
        ckpt = args.checkpoint or "analytic"
        solver = load_nn_solver(args.model, ckpt, mode=args.mode)
        solver_name = f"{args.model} [{args.mode}] [{ckpt}]"
        solver_tag  = f"{args.model}_{args.mode}"

    all_passed = run_all_tests(
        solver, args.test_file, solver_name,
        visualize=args.visualize, save_dir=args.save_dir, basis=args.basis,
        solver_tag=solver_tag,
    )
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()