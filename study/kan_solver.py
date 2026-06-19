"""
KAN-based nonlinear LS solver for the comparative study.

Implements the Moulinec-Suquet nonlinear algorithm (Algorithm 8) identically to
solve_nonlinear() in generation/nonlinear_fft_solver.py, with a single change:

  _radial_return_2d    →    _radial_return_2d_kan

The KAN radial return uses B-spline approximations for the two non-polynomial
operations:
  np.sqrt(1.5 * s²)        →  phi_sqrt(1.5 * s²)     [B-spline #1]
  np.where(f_trial > 0)    →  phi_kink(f_trial)       [B-spline #2]

All other steps (trial stress, deviatoric decomposition, radial correction,
polarisation stress, FFT update) are bit-for-bit identical to the FFT solver.

This isolation means that the only source of difference between the FFT result
and the KAN result is the B-spline approximation error in phi_sqrt and phi_kink.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from typing import Optional, Dict, Any, List

from generation.fft_solver import _build_green_operator
from generation.nonlinear_fft_solver import _extract_lame, _INPLANE_IDX
from models.nonlinear_kan_tau_theta import make_sqrt_bspline, make_kink_bspline

SQRT_2_3 = float(np.sqrt(2.0 / 3.0))


# ---------------------------------------------------------------------------
# Torch → numpy bridge
# ---------------------------------------------------------------------------

def _make_numpy_fn(spline):
    """Wrap BSpline1D to accept / return numpy float64 arrays of any shape."""
    def fn(x_np: np.ndarray) -> np.ndarray:
        x = torch.tensor(x_np.astype(np.float64))
        with torch.no_grad():
            y = spline(x)
        return y.numpy().astype(np.float64)
    return fn


# ---------------------------------------------------------------------------
# Radial return — 2D plane strain, B-spline version
# ---------------------------------------------------------------------------

def _radial_return_2d_kan(
    eps_v:      np.ndarray,   # (3, N, N)
    eps_n_v:    np.ndarray,   # (3, N, N)
    sigma_n:    np.ndarray,   # (4, N, N)  [σ₁₁, σ₂₂, σ₃₃, σ₁₂]
    p_n:        np.ndarray,   # (N, N)
    phase:      np.ndarray,   # (N, N) bool — True = elastic inclusion
    C_field:    np.ndarray,   # (3, 3, N, N)
    lam:        np.ndarray,   # (N, N)
    mu:         np.ndarray,   # (N, N)
    sigma_0:    float,
    H:          float,
    phi_sqrt_fn,               # callable: (N,N) → (N,N)
    phi_kink_fn,               # callable: (N,N) → (N,N)
) -> tuple:
    """
    Radial return for 2D generalised plane strain using B-splines.

    Physics is identical to _radial_return_2d():
      - Full 3D von Mises criterion (σ₃₃ tracked via plane-strain constraint)
      - Same trial stress, same deviatoric decomposition, same radial correction

    Only difference: phi_sqrt_fn and phi_kink_fn replace np.sqrt and np.where.
    """
    deps = eps_v - eps_n_v    # (3, N, N)

    # ── Trial stress ──────────────────────────────────────────────────────────
    ds_ip = np.einsum('ab...,b...->a...', C_field, deps)   # in-plane increment
    ds_33 = lam * (deps[0] + deps[1])                      # out-of-plane (ε₃₃=0)

    st_11 = sigma_n[0] + ds_ip[0]
    st_22 = sigma_n[1] + ds_ip[1]
    st_33 = sigma_n[2] + ds_33
    st_12 = sigma_n[3] + ds_ip[2]

    # ── Deviatoric decomposition (full 3D for correct von Mises) ─────────────
    tr   = st_11 + st_22 + st_33
    phyd = tr / 3.0
    s11  = st_11 - phyd
    s22  = st_22 - phyd
    s33  = st_33 - phyd
    s12  = st_12           # shear is already deviatoric

    # s:s — standard double-contraction (σ₁₂ appears once in 2D cross-section)
    s2 = s11**2 + s22**2 + s33**2 + 2.0 * s12**2
    q  = 1.5 * s2   # = sigma_eq² (von Mises)

    # ── B-spline #1: σ_eq ≈ √(q) ─────────────────────────────────────────────
    sigma_eq = phi_sqrt_fn(q).clip(0.0)   # (N, N)

    # ── Yield function ────────────────────────────────────────────────────────
    # Only matrix voxels can yield — but phi_kink handles the elastic case
    # automatically (f_plus = 0 when f_trial ≤ 0).
    # We still respect the phase mask: force f_trial = -1 for elastic inclusions
    # so phi_kink returns 0 there regardless of B-spline accuracy.
    f_trial_raw = sigma_eq - (sigma_0 + H * p_n)
    f_trial     = np.where(phase, -1.0, f_trial_raw)   # force elastic for inclusions

    # ── B-spline #2: f_plus ≈ max(f_trial, 0) ────────────────────────────────
    f_plus = phi_kink_fn(f_trial)   # (N, N), ≥ 0

    # ── Plastic increment ─────────────────────────────────────────────────────
    delta_p = f_plus / (3.0 * mu + H)   # = 0 in elastic case (f_plus = 0)

    # ── Radial correction ─────────────────────────────────────────────────────
    safe_seq = np.where(sigma_eq > 1e-15, sigma_eq, 1.0)
    scale    = 1.0 - 3.0 * mu * delta_p / safe_seq

    sigma_new = np.stack([
        scale * s11 + phyd,    # σ₁₁
        scale * s22 + phyd,    # σ₂₂
        scale * s33 + phyd,    # σ₃₃
        scale * s12,           # σ₁₂
    ], axis=0)                 # (4, N, N)

    p_new = p_n + delta_p
    return sigma_new, p_new


# ---------------------------------------------------------------------------
# B-spline domain sizing
# ---------------------------------------------------------------------------

def _estimate_domains(C_field, eps_bar_path, sigma_0, H):
    """
    Estimate safe B-spline domains for phi_sqrt and phi_kink.

    phi_sqrt domain: [0, R_sq]  where R_sq ≥ max(q) = max(1.5 * s²)
    phi_kink domain: [−f_range, +f_range]  where f_range ≥ max|f_trial|

    Uses conservative bounds based on material stiffness and loading amplitude.
    """
    eps_max = float(np.abs(eps_bar_path).max()) + 1e-12
    C_max   = float(C_field[0, 0].max())

    # Upper bound on deviatoric trial stress magnitude
    # (s ≈ 2μ·ε_dev; safe upper bound: C_max * eps_max * sqrt(3))
    sigma_dev_max = C_max * eps_max * float(np.sqrt(3.0))

    # Upper bound on q = 1.5 * s²
    R_sq = max(
        (4.0 * sigma_dev_max)**2,          # generous safety factor
        (20.0 * sigma_0 / SQRT_2_3)**2,    # standard default
    )

    # Upper bound on |f_trial| = |σ_eq − threshold|
    p_max_est = eps_max / SQRT_2_3
    f_max_est = sigma_dev_max + abs(H) * p_max_est
    f_range   = max(2.0 * f_max_est, 20.0 * sigma_0)

    return R_sq, f_range


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_nonlinear_kan(
    C_field:      np.ndarray,
    phase:        np.ndarray,
    eps_bar_path: np.ndarray,
    sigma_0:      float,
    H:            float,
    alpha0:       Optional[float] = None,
    tol:          float = 1e-4,
    max_iter:     int   = 1000,
    discretization: str = 'exact',
    verbose:      bool  = False,
    n_ctrl_sqrt:  int   = 100,
    n_ctrl_kink_half: int = 20,
    degree:       int   = 3,
) -> Dict[str, Any]:
    """
    Moulinec-Suquet nonlinear LS solver with B-spline constitutive update.

    Identical interface and return dict as solve_nonlinear() in
    generation/nonlinear_fft_solver.py. Only the constitutive update
    (_radial_return_2d_kan) uses B-splines instead of exact functions.

    Currently supports 2D only (n_comp = 3).

    Parameters
    ----------
    C_field       : (3, 3, N, N) float64 Voigt stiffness field
    phase         : (N, N) bool — True = elastic inclusion
    eps_bar_path  : (n_steps, 3) macroscopic strain path (total, not incremental)
    sigma_0       : initial yield stress (matrix phase)
    H             : isotropic hardening modulus (0 = perfect plasticity)
    alpha0        : reference stiffness (defaults to Moulinec-Suquet optimal)
    tol           : LS convergence tolerance
    max_iter      : maximum inner iterations per step
    discretization: 'exact' or 'staggered' (Willot 2015)
    verbose       : print progress
    n_ctrl_sqrt   : control points for phi_sqrt
    n_ctrl_kink_half : control points per half-domain for phi_kink
    degree        : B-spline degree

    Returns
    -------
    Same dict as solve_nonlinear(): eps_history, sigma_history, p_history,
    macro_stress_history, n_iter_history, residuals_history, converged_history,
    eps_star, sigma_star, tau_star, p_star, alpha0.
    """
    n_comp = C_field.shape[0]
    dim    = C_field.ndim - 2
    N      = C_field.shape[2]

    assert dim == 2 and n_comp == 3, \
        "solve_nonlinear_kan currently supports 2D only (n_comp=3)"
    assert C_field.dtype == np.float64

    eps_bar_path = np.asarray(eps_bar_path, dtype=np.float64)
    n_steps = eps_bar_path.shape[0]

    # ── Reference stiffness ────────────────────────────────────────────────────
    if alpha0 is None:
        C00    = C_field[0, 0]
        alpha0 = (float(C00.max()) + float(C00.min())) / 2.0

    Gamma_hat = _build_green_operator(N, alpha0, discretization)   # (3,3,N,N)
    C0_voigt  = alpha0 * np.diag([1.0, 1.0, 0.5])
    fft_axes  = (-2, -1)
    dc_idx    = (slice(None), 0, 0)   # [:, 0, 0]
    ip_idx    = _INPLANE_IDX[2]       # [0, 1, 3]

    lam, mu = _extract_lame(C_field, dim)
    phase   = np.asarray(phase, dtype=bool)

    # ── B-spline construction ─────────────────────────────────────────────────
    R_sq, f_range = _estimate_domains(C_field, eps_bar_path, sigma_0, H)

    if verbose:
        print(f"  KAN B-spline domains: R_sq={R_sq:.3g}, f_range={f_range:.3g}")
        print(f"  Building phi_sqrt (n_ctrl={n_ctrl_sqrt}) ...")

    phi_sqrt = make_sqrt_bspline(
        R_sq=R_sq, n_ctrl=n_ctrl_sqrt, degree=degree, trainable=False
    )
    phi_kink = make_kink_bspline(
        f_min=-f_range, f_max=f_range,
        degree=degree, n_ctrl_half=n_ctrl_kink_half, trainable=False,
    )
    phi_sqrt_fn = _make_numpy_fn(phi_sqrt)
    phi_kink_fn = _make_numpy_fn(phi_kink)

    # ── Initial state ──────────────────────────────────────────────────────────
    eps_n   = np.zeros((3, N, N), dtype=np.float64)
    sigma_n = np.zeros((4, N, N), dtype=np.float64)
    p_n     = np.zeros((N, N),    dtype=np.float64)

    eps_history:          List = []
    sigma_history:        List = []
    p_history:            List = []
    macro_stress_history: List = []
    n_iter_history:       List = []
    residuals_history:    List = []
    converged_history:    List = []

    # ── Load-step loop ─────────────────────────────────────────────────────────
    for step in range(n_steps):
        eps_bar_v = eps_bar_path[step]

        if verbose:
            print(f"\n=== KAN Step {step+1}/{n_steps}  E = {eps_bar_v} ===")

        eps_v = np.empty((3, N, N), dtype=np.float64)
        for a in range(3):
            eps_v[a] = eps_bar_v[a]

        sigma_v, p_v = _radial_return_2d_kan(
            eps_v, eps_n, sigma_n, p_n, phase,
            C_field, lam, mu, sigma_0, H, phi_sqrt_fn, phi_kink_fn)

        sigma_ip = sigma_v[ip_idx]
        tau_v    = sigma_ip - np.einsum('ab,b...->a...', C0_voigt, eps_v)

        step_residuals: List[float] = []
        step_converged = False

        # ── Inner LS loop ──────────────────────────────────────────────────────
        for k in range(max_iter):
            tau_prev = tau_v

            tau_hat = np.fft.fftn(tau_v, axes=fft_axes)
            eps_hat = -np.einsum('ab...,b...->a...', Gamma_hat, tau_hat)
            eps_hat[dc_idx] = eps_bar_v * float(N**2)
            eps_v = np.real(np.fft.ifftn(eps_hat, axes=fft_axes))

            sigma_v, p_v = _radial_return_2d_kan(
                eps_v, eps_n, sigma_n, p_n, phase,
                C_field, lam, mu, sigma_0, H, phi_sqrt_fn, phi_kink_fn)

            sigma_ip = sigma_v[ip_idx]
            tau_v    = sigma_ip - np.einsum('ab,b...->a...', C0_voigt, eps_v)

            dtau     = tau_v - tau_prev
            tau_norm = float(np.sqrt(
                np.sum(tau_v[:2]**2) + 2.0 * np.sum(tau_v[2:]**2)))
            dtau_norm = float(np.sqrt(
                np.sum(dtau[:2]**2) + 2.0 * np.sum(dtau[2:]**2)))
            res = dtau_norm / tau_norm if tau_norm > 0.0 else 0.0
            step_residuals.append(res)

            if verbose and k % 50 == 0:
                print(f"  KAN iter {k:>4}/{max_iter}  res={res:.3e}")

            if res < tol:
                step_converged = True
                if verbose:
                    print(f"  KAN converged in {k+1} iters  res={res:.3e}")
                break

        macro_stress = sigma_ip.mean(axis=(-2, -1))

        eps_history.append(eps_v.copy())
        sigma_history.append(sigma_v.copy())
        p_history.append(p_v.copy())
        macro_stress_history.append(macro_stress)
        n_iter_history.append(len(step_residuals))
        residuals_history.append(step_residuals)
        converged_history.append(step_converged)

        eps_n   = eps_v.copy()
        sigma_n = sigma_v.copy()
        p_n     = p_v.copy()

    sigma_ip_final = sigma_n[ip_idx]
    tau_final      = sigma_ip_final - np.einsum('ab,b...->a...', C0_voigt, eps_n)

    return {
        'eps_history':          eps_history,
        'sigma_history':        sigma_history,
        'p_history':            p_history,
        'macro_stress_history': macro_stress_history,
        'n_iter_history':       n_iter_history,
        'residuals_history':    residuals_history,
        'converged_history':    converged_history,
        'eps_star':   eps_n,
        'sigma_star': sigma_ip_final,
        'tau_star':   tau_final,
        'p_star':     p_n,
        'alpha0':     alpha0,
    }
