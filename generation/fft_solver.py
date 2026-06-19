"""
Moulinec-Suquet FFT-based iterative solver for periodic composites.

Implements the Basic Scheme from:
  Moulinec & Suquet (1994) — "A fast numerical method for computing the
  linear and nonlinear mechanical properties of composites."

Supports both 2D (plane-strain) and 3D problems.  The dimensionality is
detected automatically from the shape of C_field:

  2D: C_field (3, 3, N, N)     — 3 independent strain components (ε₁₁, ε₂₂, γ₁₂)
  3D: C_field (6, 6, N, N, N)  — 6 independent strain components (ε₁₁…ε₃₃, γ₂₃, γ₁₃, γ₁₂)

The solver finds the strain field ε(x) satisfying:
  ε(x) = ε̄ − Γ⁰ : (C(x) − C⁰) : ε(x)

Public API
----------
solve(C_field, eps_bar, alpha0=None, tol=1e-6, max_iter=1000)
  → dict with keys: eps_star, tau_star, sigma_star, n_iter, residuals, converged, alpha0
"""

import numpy as np
from typing import Optional, Dict, Any


# ---------------------------------------------------------------------------
# 2D Voigt index helpers (plane-strain, 3 independent components)
# ---------------------------------------------------------------------------
# Voigt pairs: 0↔(0,0), 1↔(1,1), 2↔(0,1)=(1,0)
_VOIGT_I_2D      = [0, 1, 0]
_VOIGT_J_2D      = [0, 1, 1]
_VOIGT_FACTOR_2D = [1.0, 1.0, 2.0]  # engineering shear factors

# ---------------------------------------------------------------------------
# 3D Voigt index helpers (full 3D, 6 independent components)
# ---------------------------------------------------------------------------
# Voigt pairs: 0↔(0,0), 1↔(1,1), 2↔(2,2), 3↔(1,2), 4↔(0,2), 5↔(0,1)
_VOIGT_I_3D      = [0, 1, 2, 1, 0, 0]
_VOIGT_J_3D      = [0, 1, 2, 2, 2, 1]
_VOIGT_FACTOR_3D = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]


def _strain_voigt_to_tensor(e_v: np.ndarray) -> np.ndarray:
    """(3, ...) 2D Voigt strain → (2, 2, ...) tensor strain."""
    shape = e_v.shape[1:]
    e = np.zeros((2, 2) + shape, dtype=e_v.dtype)
    e[0, 0] = e_v[0]
    e[1, 1] = e_v[1]
    e[0, 1] = e_v[2] / 2.0
    e[1, 0] = e_v[2] / 2.0
    return e


def _strain_tensor_to_voigt(e: np.ndarray) -> np.ndarray:
    """(2, 2, ...) tensor strain → (3, ...) 2D Voigt strain."""
    return np.stack([e[0, 0], e[1, 1], e[0, 1] + e[1, 0]], axis=0)


# ---------------------------------------------------------------------------
# Eshelby-Green operators in Fourier space
# ---------------------------------------------------------------------------

def _build_green_operator(N: int, alpha0: float,
                          discretization: str = 'exact') -> np.ndarray:
    """
    2D Eshelby-Green operator in Voigt notation, shape (3, 3, N, N).

    Γ̂_ab includes 1/α₀ so that the polarization stress τ = (C−C⁰):ε can be
    passed directly (C⁰ = α₀ I_sym).

    Args:
        discretization: 'exact' (standard Fourier) or 'staggered' (Willot 2015
                        rotated grid with effective frequencies).
    """
    freq = np.fft.fftfreq(N)
    xi_x, xi_y = np.meshgrid(freq, freq, indexing='ij')

    if discretization == 'staggered':
        xi_eff_x = np.sin(np.pi * xi_x) * np.cos(np.pi * xi_y)
        xi_eff_y = np.sin(np.pi * xi_y) * np.cos(np.pi * xi_x)
        xi = np.stack([xi_eff_x, xi_eff_y], axis=0)
        xi_norm2 = xi[0]**2 + xi[1]**2
    else:
        xi = np.stack([xi_x, xi_y], axis=0)
        xi_norm2 = xi_x**2 + xi_y**2

    safe = np.where(xi_norm2 == 0, 1.0, xi_norm2)

    def xip(p, q):   return xi[p] * xi[q] / safe
    def xip4(p,q,r,s): return xi[p]*xi[q]*xi[r]*xi[s] / safe**2
    def dlt(a, b):   return float(a == b)

    Gamma = np.zeros((3, 3, N, N), dtype=np.float64)
    for a in range(3):
        i, j = _VOIGT_I_2D[a], _VOIGT_J_2D[a]
        for b in range(3):
            k, l = _VOIGT_I_2D[b], _VOIGT_J_2D[b]
            G = (dlt(i,k)*xip(j,l) + dlt(i,l)*xip(j,k)
                 + dlt(j,k)*xip(i,l) + dlt(j,l)*xip(i,k)) / 2.0 \
                - xip4(i, j, k, l)
            Gamma[a, b] = G * _VOIGT_FACTOR_2D[a] * _VOIGT_FACTOR_2D[b] / alpha0

    Gamma[:, :, 0, 0] = 0.0
    return Gamma


def _build_green_operator_3d(N: int, alpha0: float,
                             discretization: str = 'exact') -> np.ndarray:
    """
    3D Eshelby-Green operator in Voigt notation, shape (6, 6, N, N, N).

    Same formula as the 2D version — dimension-agnostic four-index tensor —
    extended to three frequency directions and six Voigt components.
    Includes 1/α₀ (same convention as the 2D version).

    Args:
        discretization: 'exact' (standard Fourier) or 'staggered' (Willot 2015
                        rotated grid with effective frequencies).
    """
    freq = np.fft.fftfreq(N)
    xi_x, xi_y, xi_z = np.meshgrid(freq, freq, freq, indexing='ij')

    if discretization == 'staggered':
        xi_eff_x = np.sin(np.pi * xi_x) * np.cos(np.pi * xi_y) * np.cos(np.pi * xi_z)
        xi_eff_y = np.sin(np.pi * xi_y) * np.cos(np.pi * xi_x) * np.cos(np.pi * xi_z)
        xi_eff_z = np.sin(np.pi * xi_z) * np.cos(np.pi * xi_x) * np.cos(np.pi * xi_y)
        xi = np.stack([xi_eff_x, xi_eff_y, xi_eff_z], axis=0)
        xi_norm2 = xi[0]**2 + xi[1]**2 + xi[2]**2
    else:
        xi = np.stack([xi_x, xi_y, xi_z], axis=0)
        xi_norm2 = xi_x**2 + xi_y**2 + xi_z**2

    safe = np.where(xi_norm2 == 0, 1.0, xi_norm2)

    def xip(p, q):      return xi[p] * xi[q] / safe
    def xip4(p,q,r,s):  return xi[p]*xi[q]*xi[r]*xi[s] / safe**2
    def dlt(a, b):      return float(a == b)

    Gamma = np.zeros((6, 6, N, N, N), dtype=np.float64)
    for a in range(6):
        i, j = _VOIGT_I_3D[a], _VOIGT_J_3D[a]
        for b in range(6):
            k, l = _VOIGT_I_3D[b], _VOIGT_J_3D[b]
            G = (dlt(i,k)*xip(j,l) + dlt(i,l)*xip(j,k)
                 + dlt(j,k)*xip(i,l) + dlt(j,l)*xip(i,k)) / 2.0 \
                - xip4(i, j, k, l)
            Gamma[a, b] = G * _VOIGT_FACTOR_3D[a] * _VOIGT_FACTOR_3D[b] / alpha0

    Gamma[:, :, 0, 0, 0] = 0.0
    return Gamma


# ---------------------------------------------------------------------------
# Main solver  (2D and 3D unified)
# ---------------------------------------------------------------------------

def solve(C_field: np.ndarray,
          eps_bar: np.ndarray,
          alpha0: Optional[float] = None,
          tol: float = 1e-6,
          max_iter: int = 1000,
          discretization: str = 'exact', 
          verbose = False) -> Dict[str, Any]:
    """
    Moulinec-Suquet Basic Scheme — works for both 2D and 3D.

    Dimensionality is detected from C_field.ndim:
      ndim == 4  →  2D, n_comp = 3,  spatial axes = (-2, -1)
      ndim == 5  →  3D, n_comp = 6,  spatial axes = (-3, -2, -1)

    Args:
        C_field:         (3,3,N,N) or (6,6,N,N,N) Voigt stiffness field.
        eps_bar:         (3,) or (6,) macroscopic strain; 2D also accepts (2,2) tensor.
        alpha0:          Reference stiffness. Defaults to (C₁₁₁₁.max + C₁₁₁₁.min) / 2.
        tol:             Relative convergence tolerance: ‖τ_k − τ_{k−1}‖_F / ‖τ_k‖_F < tol,
                         where ‖·‖_F is the Frobenius (tensor) norm.  This matches the
                         Mandel L2 norm used in LS-FNO so iteration counts are comparable.
        max_iter:        Maximum number of iterations.
        discretization:  'exact' (standard Fourier) or 'staggered' (Willot 2015
                         rotated grid). Must match the value used in LS-FNO for
                         fair iteration-count comparison.

    Returns:
        dict with keys:
          'eps_star'   : (n_comp, N[, N, N])  converged strain field (Voigt)
          'tau_star'   : (n_comp, N[, N, N])  polarization stress (Voigt)
          'sigma_star' : (n_comp, N[, N, N])  stress field (Voigt)
          'n_iter'     : int
          'residuals'  : list[float]
          'converged'  : bool
          'alpha0'     : float
    """
    # ── Detect dimensionality ────────────────────────────────────────────────
    n_comp = C_field.shape[0]
    dim    = C_field.ndim - 2            # 4-2=2  or  5-2=3
    N      = C_field.shape[2]
    assert dim in (2, 3), f"C_field must be 4-D (2D) or 5-D (3D), got {C_field.ndim}-D"
    expected_n_comp = 3 if dim == 2 else 6
    assert n_comp == expected_n_comp, \
        f"dim={dim} requires n_comp={expected_n_comp}, got n_comp={n_comp}"
    assert C_field.shape == (n_comp, n_comp) + (N,) * dim, \
        f"C_field shape mismatch: expected {(n_comp, n_comp) + (N,)*dim}, got {C_field.shape}"

    # ── Normalise eps_bar ────────────────────────────────────────────────────
    if dim == 2 and isinstance(eps_bar, np.ndarray) and eps_bar.shape == (2, 2):
        eps_bar_v = _strain_tensor_to_voigt(eps_bar)
    else:
        eps_bar_v = np.asarray(eps_bar, dtype=np.float64).ravel()
    assert len(eps_bar_v) == n_comp, \
        f"eps_bar must have {n_comp} components for dim={dim}"

    # ── Reference stiffness ──────────────────────────────────────────────────
    if alpha0 is None:
        C00 = C_field[0, 0]
        alpha0 = (float(C00.max()) + float(C00.min())) / 2.0

    # ── Green operator and helper constants ──────────────────────────────────
    if dim == 2:
        Gamma_hat = _build_green_operator(N, alpha0, discretization)        # (3,3,N,N)
        C0_diag   = np.array([1.0, 1.0, 0.5])
    else:
        Gamma_hat = _build_green_operator_3d(N, alpha0, discretization)     # (6,6,N,N,N)
        C0_diag   = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])

    C0_voigt = alpha0 * np.diag(C0_diag)
    # (C − C⁰) is constant throughout the loop — compute once
    dC = C_field - C0_voigt.reshape((n_comp, n_comp) + (1,) * dim)

    fft_axes = tuple(range(-dim, 0))               # (-2,-1) or (-3,-2,-1)
    dc_idx   = (slice(None),) + (0,) * dim         # [:,0,0] or [:,0,0,0]

    # ── Initialise strain field ──────────────────────────────────────────────
    eps_v = np.zeros((n_comp,) + (N,) * dim, dtype=np.float64)
    for a in range(n_comp):
        eps_v[a] = eps_bar_v[a]

    residuals: list = []
    converged = False

    # Initial polarization stress τ₀ = (C − C⁰) : ε̄.
    # The loop then mirrors the LS-FNO solve() structure exactly:
    #   1. save τ_prev, 2. apply Green update (ε → ε'), 3. compute τ', 4. check residual.
    # This avoids the off-by-one that arises when τ is re-computed from the same ε at
    # the start of the loop (which gives res=0 on the first pass and forces a k>0 guard).
    tau_v = np.einsum('ab...,b...->a...', dC, eps_v)

    for k in range(max_iter):
        tau_prev = tau_v

        # FFT → Green operator → enforce mean → iFFT
        tau_hat = np.fft.fftn(tau_v, axes=fft_axes)
        eps_hat = -np.einsum('ab...,b...->a...', Gamma_hat, tau_hat)
        eps_hat[dc_idx] = eps_bar_v * float(N ** dim)   # enforce ⟨ε⟩ = ε̄
        eps_v = np.real(np.fft.ifftn(eps_hat, axes=fft_axes))

        tau_v = np.einsum('ab...,b...->a...', dC, eps_v)
        dtau  = tau_v - tau_prev
        # Frobenius norm of the polarization tensor — matches the Mandel L2 norm
        # used by LS-FNO so that the two solvers share an identical stopping criterion.
        # In Voigt, shear components store the actual stress τ_ij (not the engineering
        # factor 2τ_ij), so the Frobenius norm is:
        #   ‖τ‖_F² = Σ_{normal} τᵢ² + 2·Σ_{shear} τᵢ²   (i.e. ‖τ_V[:dim]‖² + 2·‖τ_V[dim:]‖²)
        tau_norm = float(np.sqrt(np.sum(tau_v[:dim] ** 2) + 2.0 * np.sum(tau_v[dim:] ** 2)))
        dtau_norm = float(np.sqrt(np.sum(dtau[:dim] ** 2) + 2.0 * np.sum(dtau[dim:] ** 2)))
        res = dtau_norm / tau_norm if tau_norm > 0 else 0.0
        residuals.append(res)

        if res < tol:
            converged = True
            if verbose == True:
                    print(f"----- FFT converged in {len(residuals)} iterations! -----")
            break

        if verbose and k % 50 == 0:
                print(f"Iteration FFT-solver: {k}/{max_iter}  residual: {res:.3e}")

    n_iter = len(residuals)

    # ── Final fields ─────────────────────────────────────────────────────────
    tau_v   = np.einsum('ab...,b...->a...', dC, eps_v)
    sigma_v = np.einsum('ab,b...->a...', C0_voigt, eps_v) + tau_v

    return {
        'eps_star':   eps_v,
        'tau_star':   tau_v,
        'sigma_star': sigma_v,
        'n_iter':     n_iter,
        'residuals':  residuals,
        'converged':  converged,
        'alpha0':     alpha0,
    }


# ---------------------------------------------------------------------------
# Effective stiffness (post-processing)
# ---------------------------------------------------------------------------

def effective_stiffness(C_field: np.ndarray) -> np.ndarray:
    """
    Compute the effective Voigt stiffness tensor by solving n_comp independent
    load cases (one unit strain per component).

    Works for both 2D (returns 3×3) and 3D (returns 6×6).
    """
    n_comp = C_field.shape[0]
    C_eff  = np.zeros((n_comp, n_comp), dtype=np.float64)
    for a in range(n_comp):
        eps_bar       = np.zeros(n_comp)
        eps_bar[a]    = 1.0
        result        = solve(C_field, eps_bar)
        spatial_axes  = tuple(range(1, result['sigma_star'].ndim))
        C_eff[:, a]   = result['sigma_star'].mean(axis=spatial_axes)
    return C_eff
