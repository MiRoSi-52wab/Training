"""
Moulinec-Suquet FFT-based iterative solver — nonlinear case.

Von Mises plasticity with isotropic hardening.  Implements Algorithm (8) from:

  Moulinec & Suquet (1994) — "A fast numerical method for computing the
  linear and nonlinear mechanical properties of composites."

The only step that differs from the linear algorithm (5) is step (a):
  the local constitutive update via radial return at each voxel, using the
  local elastic stiffness of that voxel (not the reference material c⁰).

Two nested loops:
  Outer — load steps: applies a sequence of macroscopic strains E_0, E_1, …
  Inner — Lippmann-Schwinger iterations (LS): converges at each load step.

Supports both 2D (generalized plane strain) and 3D.

2D generalized plane strain note
---------------------------------
The FFT solver operates on the 3-component in-plane strain [ε₁₁, ε₂₂, γ₁₂]
exactly as in the linear case.  Because plasticity requires the full 3D von
Mises criterion, the out-of-plane stress σ₃₃ is tracked as an auxiliary
state variable (it satisfies ε₃₃ = 0 but σ₃₃ ≠ 0 once plastic flow begins).
σ₃₃ does not enter the Green operator or polarisation computation.

Voigt conventions (same as fft_solver.py)
------------------------------------------
2D:  n_comp = 3   [ε₁₁, ε₂₂, γ₁₂]   /   [σ₁₁, σ₂₂, σ₁₂]
3D:  n_comp = 6   [ε₁₁, ε₂₂, ε₃₃, γ₂₃, γ₁₃, γ₁₂]   /   [σ₁₁, σ₂₂, σ₃₃, σ₂₃, σ₁₃, σ₁₂]

Engineering shear: γᵢⱼ = 2εᵢⱼ,  C_shear * γᵢⱼ = σᵢⱼ  (C[2,2]=μ in 2D).

Public API
----------
solve_nonlinear(C_field, phase, eps_bar_path, sigma_0, H, …)
  → dict with per-step history and final converged fields.
"""

import numpy as np
from typing import Optional, Dict, Any, List

# Green operator builders are identical to the linear case — reuse them.
from generation.fft_solver import (
    _build_green_operator,
    _build_green_operator_3d,
)


# ---------------------------------------------------------------------------
# Helpers: extract isotropic Lamé fields from C_field
# ---------------------------------------------------------------------------

def _extract_lame(C_field: np.ndarray, dim: int):
    """
    Extract per-voxel Lamé parameters λ and μ from an isotropic Voigt stiffness
    field.

    2D: C[0,1] = λ,  C[2,2] = μ
    3D: C[0,1] = λ,  C[3,3] = μ
    """
    lam = C_field[0, 1]             # (N,…)  Lamé λ
    mu  = C_field[2, 2] if dim == 2 else C_field[3, 3]   # (N,…)  shear μ
    return lam, mu


# ---------------------------------------------------------------------------
# Radial return — 2D generalized plane strain
# ---------------------------------------------------------------------------

def _radial_return_2d(
    eps_v: np.ndarray,      # (3, N, N)  current strain iterate [ε₁₁, ε₂₂, γ₁₂]
    eps_n_v: np.ndarray,    # (3, N, N)  strain at previous load step
    sigma_n: np.ndarray,    # (4, N, N)  stress at previous step [σ₁₁, σ₂₂, σ₃₃, σ₁₂]
    p_n: np.ndarray,        # (N, N)     accumulated plastic strain
    phase: np.ndarray,      # (N, N)     bool: True = elastic inclusion
    C_field: np.ndarray,    # (3,3,N,N)  local elastic stiffness
    lam: np.ndarray,        # (N, N)     Lamé λ (per voxel)
    mu: np.ndarray,         # (N, N)     shear μ (per voxel)
    sigma_0: float,
    H: float,
) -> tuple:
    """
    Vectorised radial return for 2D generalised plane strain.

    Returns
    -------
    sigma_new : (4, N, N)  updated stress [σ₁₁, σ₂₂, σ₃₃, σ₁₂]
    p_new     : (N, N)     updated accumulated plastic strain
    """
    deps = eps_v - eps_n_v    # (3, N, N)  strain increment

    # --- elastic trial stress ---
    # In-plane increment via full C_field
    ds_ip = np.einsum('ab...,b...->a...', C_field, deps)  # (3,N,N): [Δσ₁₁, Δσ₂₂, Δσ₁₂]

    # Out-of-plane: Δσ₃₃ = λ(Δε₁₁ + Δε₂₂)  (plane-strain: Δε₃₃ = 0)
    ds_33 = lam * (deps[0] + deps[1])

    st_11 = sigma_n[0] + ds_ip[0]
    st_22 = sigma_n[1] + ds_ip[1]
    st_33 = sigma_n[2] + ds_33
    st_12 = sigma_n[3] + ds_ip[2]

    # --- deviatoric decomposition (full 3D for correct von Mises) ---
    tr    = st_11 + st_22 + st_33
    phyd  = tr / 3.0

    s11 = st_11 - phyd
    s22 = st_22 - phyd
    s33 = st_33 - phyd
    s12 = st_12    # shear (stress, not engineering shear)

    # von Mises equivalent: σ_eq = sqrt(3/2 * s:s)
    # s:s = s₁₁² + s₂₂² + s₃₃² + 2·s₁₂²  (σ₁₂ appears once in 2D cross-section)
    s2       = s11**2 + s22**2 + s33**2 + 2.0 * s12**2
    sigma_eq = np.sqrt(1.5 * s2)

    f_trial = sigma_eq - (sigma_0 + H * p_n)

    # Plastic: only matrix voxels (phase=False) with f_trial > 0
    plastic = (~phase) & (f_trial > 0.0)

    # Δp = f_trial / (3μ + H)  using local μ
    delta_p  = np.where(plastic, f_trial / (3.0 * mu + H), 0.0)

    # Radial scaling: s_new = scale · s_trial,  scale = 1 − 3μΔp/σ_eq
    safe_seq = np.where(sigma_eq > 0.0, sigma_eq, 1.0)
    scale    = np.where(plastic, 1.0 - 3.0 * mu * delta_p / safe_seq, 1.0)

    sigma_new = np.stack([
        scale * s11 + phyd,    # σ₁₁
        scale * s22 + phyd,    # σ₂₂
        scale * s33 + phyd,    # σ₃₃
        scale * s12,           # σ₁₂
    ], axis=0)                 # (4, N, N)

    p_new = p_n + delta_p
    return sigma_new, p_new


# ---------------------------------------------------------------------------
# Radial return — 3D
# ---------------------------------------------------------------------------

def _radial_return_3d(
    eps_v: np.ndarray,      # (6, N,N,N)  [ε₁₁, ε₂₂, ε₃₃, γ₂₃, γ₁₃, γ₁₂]
    eps_n_v: np.ndarray,    # (6, N,N,N)
    sigma_n: np.ndarray,    # (6, N,N,N)  [σ₁₁, σ₂₂, σ₃₃, σ₂₃, σ₁₃, σ₁₂]
    p_n: np.ndarray,        # (N,N,N)
    phase: np.ndarray,      # (N,N,N)  True = elastic
    C_field: np.ndarray,    # (6,6,N,N,N)
    mu: np.ndarray,         # (N,N,N)
    sigma_0: float,
    H: float,
) -> tuple:
    """
    Vectorised radial return for 3D.

    Returns
    -------
    sigma_new : (6, N,N,N)
    p_new     : (N,N,N)
    """
    deps         = eps_v - eps_n_v
    sigma_trial  = sigma_n + np.einsum('ab...,b...->a...', C_field, deps)

    # Voigt 3D: normals = indices 0,1,2;  shears = indices 3,4,5
    tr   = sigma_trial[0] + sigma_trial[1] + sigma_trial[2]
    phyd = tr / 3.0

    s = sigma_trial.copy()
    s[0] -= phyd
    s[1] -= phyd
    s[2] -= phyd
    # shear components are deviatoric already

    # s:s — shear entries carry a factor-2 in the Frobenius double-contraction
    # In Mandel-like counting: normals contribute s²; each shear pair contributes 2s²
    s2       = s[0]**2 + s[1]**2 + s[2]**2 + 2.0*(s[3]**2 + s[4]**2 + s[5]**2)
    sigma_eq = np.sqrt(1.5 * s2)

    f_trial  = sigma_eq - (sigma_0 + H * p_n)
    plastic  = (~phase) & (f_trial > 0.0)
    delta_p  = np.where(plastic, f_trial / (3.0 * mu + H), 0.0)

    safe_seq = np.where(sigma_eq > 0.0, sigma_eq, 1.0)
    scale    = np.where(plastic, 1.0 - 3.0 * mu * delta_p / safe_seq, 1.0)

    sigma_new    = np.empty_like(sigma_trial)
    sigma_new[0] = scale * s[0] + phyd
    sigma_new[1] = scale * s[1] + phyd
    sigma_new[2] = scale * s[2] + phyd
    sigma_new[3] = scale * s[3]
    sigma_new[4] = scale * s[4]
    sigma_new[5] = scale * s[5]

    p_new = p_n + delta_p
    return sigma_new, p_new


# ---------------------------------------------------------------------------
# In-plane stress extraction (2D needs to skip σ₃₃ at index 2)
# ---------------------------------------------------------------------------

_INPLANE_IDX = {2: np.array([0, 1, 3]), 3: np.arange(6)}


# ---------------------------------------------------------------------------
# Main nonlinear solver
# ---------------------------------------------------------------------------

def solve_nonlinear(
    C_field: np.ndarray,
    phase: np.ndarray,
    eps_bar_path: np.ndarray,
    sigma_0: float,
    H: float,
    alpha0: Optional[float] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    discretization: str = 'exact',
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Moulinec-Suquet nonlinear FFT scheme (Algorithm 8) for von Mises plasticity
    with isotropic hardening.

    Parameters
    ----------
    C_field : (n_comp, n_comp, N[, N, N]) float64
        Elastic stiffness field.  2D: (3,3,N,N).  3D: (6,6,N,N,N).
        Must be the LOCAL elastic stiffness at every voxel (used for the trial
        stress in the radial return as well as the reference material choice).
    phase : (N[, N, N]) bool
        True  = elastic inclusion (no plasticity).
        False = elasto-plastic matrix (von Mises + isotropic hardening).
    eps_bar_path : (n_steps, n_comp) float64
        Macroscopic strain at each load step (TOTAL, not incremental).
        E.g. for a uniaxial ramp to 1 %: linspace(0, 0.01, 20) for component 0.
    sigma_0 : float
        Initial yield stress of the matrix phase.
    H : float
        Isotropic hardening modulus (set to 0 for perfect plasticity).
    alpha0 : float, optional
        Reference stiffness of the homogeneous comparison medium c⁰.
        Defaults to (C₁₁₁₁.max + C₁₁₁₁.min) / 2  (Moulinec-Suquet choice).
    tol : float
        Relative convergence tolerance for the inner LS loop.
        Criterion: ‖Δτ‖_F / ‖τ‖_F < tol  (same Frobenius norm as fft_solver.py).
    max_iter : int
        Maximum inner iterations per load step.
    discretization : str
        'exact' (standard Fourier) or 'staggered' (Willot 2015).
    verbose : bool
        Print step and iteration progress.

    Returns
    -------
    dict with keys:

    Per-step history (lists of length n_steps):
      'eps_history'        : (n_comp, N,…) converged strain field
      'sigma_history'      : (n_stress, N,…) converged full stress
                             2D: 4 components [σ₁₁, σ₂₂, σ₃₃, σ₁₂]
                             3D: 6 components [σ₁₁, σ₂₂, σ₃₃, σ₂₃, σ₁₃, σ₁₂]
      'p_history'          : (N,…) converged accumulated plastic strain
      'macro_stress_history': (n_comp,) spatial-mean in-plane stress at each step
      'n_iter_history'     : int — inner iterations used
      'residuals_history'  : list[float] — inner residual sequence
      'converged_history'  : bool

    Final state (= last element of histories):
      'eps_star'    : (n_comp, N,…) final strain
      'sigma_star'  : (n_comp, N,…) final in-plane stress (compatible with linear solver)
      'tau_star'    : (n_comp, N,…) final polarisation stress
      'p_star'      : (N,…) final accumulated plastic strain
      'alpha0'      : float
    """
    # ── Dimensionality ────────────────────────────────────────────────────────
    n_comp = C_field.shape[0]
    dim    = C_field.ndim - 2
    N      = C_field.shape[2]

    assert dim in (2, 3), f"Expected 4-D or 5-D C_field, got {C_field.ndim}-D"
    assert n_comp == (3 if dim == 2 else 6), \
        f"n_comp={n_comp} inconsistent with dim={dim}"
    assert C_field.dtype == np.float64, "C_field must be float64"

    # 2D stores σ₃₃ as auxiliary → n_stress = 4; 3D n_stress = n_comp = 6
    n_stress   = 4 if dim == 2 else 6
    ip_idx     = _INPLANE_IDX[dim]   # indices into sigma array giving in-plane components

    eps_bar_path = np.asarray(eps_bar_path, dtype=np.float64)
    assert eps_bar_path.ndim == 2 and eps_bar_path.shape[1] == n_comp, \
        f"eps_bar_path must be (n_steps, {n_comp}), got {eps_bar_path.shape}"
    n_steps = eps_bar_path.shape[0]

    # ── Reference stiffness ───────────────────────────────────────────────────
    if alpha0 is None:
        C00    = C_field[0, 0]
        alpha0 = (float(C00.max()) + float(C00.min())) / 2.0

    # ── Green operator and reference stiffness tensor ─────────────────────────
    if dim == 2:
        Gamma_hat = _build_green_operator(N, alpha0, discretization)   # (3,3,N,N)
        C0_diag   = np.array([1.0, 1.0, 0.5])
    else:
        Gamma_hat = _build_green_operator_3d(N, alpha0, discretization)  # (6,6,N,N,N)
        C0_diag   = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])

    C0_voigt = alpha0 * np.diag(C0_diag)    # (n_comp, n_comp)
    fft_axes = tuple(range(-dim, 0))         # (-2,-1) or (-3,-2,-1)
    dc_idx   = (slice(None),) + (0,) * dim   # [:,0,0] or [:,0,0,0]

    # ── Material fields ───────────────────────────────────────────────────────
    lam, mu = _extract_lame(C_field, dim)
    phase   = np.asarray(phase, dtype=bool)

    # ── Initial state (stress-free, undeformed) ───────────────────────────────
    eps_n   = np.zeros((n_comp,)   + (N,) * dim, dtype=np.float64)
    sigma_n = np.zeros((n_stress,) + (N,) * dim, dtype=np.float64)
    p_n     = np.zeros((N,) * dim,               dtype=np.float64)

    # ── History containers ────────────────────────────────────────────────────
    eps_history:         List = []
    sigma_history:       List = []
    p_history:           List = []
    macro_stress_history:List = []
    n_iter_history:      List = []
    residuals_history:   List = []
    converged_history:   List = []

    spatial_axes = tuple(range(1, 1 + dim))   # axes over which to take spatial mean

    # ── Load-step loop ────────────────────────────────────────────────────────
    for step in range(n_steps):
        eps_bar_v = eps_bar_path[step]    # (n_comp,)

        if verbose:
            print(f"\n=== Step {step+1}/{n_steps}  E = {eps_bar_v} ===")

        # Initialise inner loop: ε⁰_{n+1}(x) = E_{n+1}  (Algorithm 8, init)
        eps_v = np.empty((n_comp,) + (N,) * dim, dtype=np.float64)
        for a in range(n_comp):
            eps_v[a] = eps_bar_v[a]

        # Constitutive update for the initial guess → first polarisation τ⁰
        if dim == 2:
            sigma_v, p_v = _radial_return_2d(
                eps_v, eps_n, sigma_n, p_n, phase, C_field, lam, mu, sigma_0, H)
        else:
            sigma_v, p_v = _radial_return_3d(
                eps_v, eps_n, sigma_n, p_n, phase, C_field, mu, sigma_0, H)

        sigma_ip = sigma_v[ip_idx]    # in-plane stress (n_comp, N,…)
        tau_v    = sigma_ip - np.einsum('ab,b...->a...', C0_voigt, eps_v)

        step_residuals: List[float] = []
        step_converged = False

        # ── Inner LS loop ─────────────────────────────────────────────────────
        for k in range(max_iter):
            tau_prev = tau_v

            # Steps (c)–(e): FFT → Γ⁰ → enforce mean → iFFT
            tau_hat = np.fft.fftn(tau_v, axes=fft_axes)
            eps_hat = -np.einsum('ab...,b...->a...', Gamma_hat, tau_hat)
            eps_hat[dc_idx] = eps_bar_v * float(N ** dim)
            eps_v = np.real(np.fft.ifftn(eps_hat, axes=fft_axes))

            # Step (a): radial return with updated strain
            if dim == 2:
                sigma_v, p_v = _radial_return_2d(
                    eps_v, eps_n, sigma_n, p_n, phase,
                    C_field, lam, mu, sigma_0, H)
            else:
                sigma_v, p_v = _radial_return_3d(
                    eps_v, eps_n, sigma_n, p_n, phase,
                    C_field, mu, sigma_0, H)

            # Step (b): polarisation stress τ = σ − c⁰ : ε
            sigma_ip = sigma_v[ip_idx]
            tau_v    = sigma_ip - np.einsum('ab,b...->a...', C0_voigt, eps_v)

            # Convergence: Frobenius norm of Δτ / τ  (same criterion as fft_solver.py)
            dtau      = tau_v - tau_prev
            tau_norm  = float(np.sqrt(
                np.sum(tau_v[:dim]**2) + 2.0 * np.sum(tau_v[dim:]**2)))
            dtau_norm = float(np.sqrt(
                np.sum(dtau[:dim]**2)  + 2.0 * np.sum(dtau[dim:]**2)))
            res = dtau_norm / tau_norm if tau_norm > 0.0 else 0.0
            step_residuals.append(res)

            if verbose and k % 50 == 0:
                print(f"  iter {k:>4}/{max_iter}  res={res:.3e}")

            if res < tol:
                step_converged = True
                if verbose:
                    print(f"  Converged in {k+1} iters  res={res:.3e}")
                break

        # ── Store converged state ─────────────────────────────────────────────
        macro_stress = sigma_ip.mean(axis=spatial_axes)   # (n_comp,)

        eps_history.append(eps_v.copy())
        sigma_history.append(sigma_v.copy())
        p_history.append(p_v.copy())
        macro_stress_history.append(macro_stress)
        n_iter_history.append(len(step_residuals))
        residuals_history.append(step_residuals)
        converged_history.append(step_converged)

        # Advance state to next step
        eps_n   = eps_v.copy()
        sigma_n = sigma_v.copy()
        p_n     = p_v.copy()

    # ── Final fields ──────────────────────────────────────────────────────────
    sigma_ip_final = sigma_n[ip_idx]
    tau_final      = sigma_ip_final - np.einsum('ab,b...->a...', C0_voigt, eps_n)

    return {
        # Per-step histories
        'eps_history':          eps_history,
        'sigma_history':        sigma_history,
        'p_history':            p_history,
        'macro_stress_history': macro_stress_history,
        'n_iter_history':       n_iter_history,
        'residuals_history':    residuals_history,
        'converged_history':    converged_history,
        # Final state
        'eps_star':   eps_n,
        'sigma_star': sigma_ip_final,
        'tau_star':   tau_final,
        'p_star':     p_n,
        'alpha0':     alpha0,
    }
