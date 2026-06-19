"""
Replication of §5.2.1: Effective stiffness of a 3D sphere-in-cube composite.

Paper: Nguyen & Schneider (2025), "Universal Fourier Neural Operators for
Micromechanics", arXiv:2507.12233v2.

Geometry:
  Single sphere (r = 10 mm) at the centre of a cube (L = 32 mm) → VF ≈ 12.78%.
  Discretized on a 32×32×32 staggered grid (Willot 2015).

Sweep:
  κ ∈ {12, 24, 48, 96}  — stiffness contrast E_inc / E_mat.
  m ∈ {7, 9, 11}        — Yarotsky depth controlling τ_θ approximation accuracy.

Output:
  C₁₁, C₁₂, C₄₄ [GPa] for the FFT reference solver and each LS-FNO variant,
  plus average iteration count per effective-stiffness column.

Note on parametric sweeps vs YAML:
  YAML configs are designed for reproducible single-experiment runs.  For a
  sweep like this one (loop over κ and m), all parameters live here as Python
  constants at the top of the file — no YAML needed or used.

Usage (from project root):
  python replicate/paper2_section521.py
"""

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from generation.microstructure import isotropic_stiffness_voigt_3d, build_C_field
from generation.fft_solver import solve as fft_solve
from models.ls_fno import LSFNO, YarotskyTauTheta
from utils.config_loader import compute_alpha_bounds


# ─────────────────────────────────────────────────────────────────────────────
# §5.2.1 parameters  (all in one place — change nothing else for replication)
# ─────────────────────────────────────────────────────────────────────────────

# Grid
N    = 32        # voxels per side  (32 voxels ≡ 32 mm → 1 mm / voxel)
DIM  = 3
N_COMP = 6       # 3D Voigt: (ε₁₁, ε₂₂, ε₃₃, γ₂₃, γ₁₃, γ₁₂)

# Material
E_MATRIX     = 3.0     # matrix Young's modulus [GPa]
NU_MATRIX    = 0.3     # matrix Poisson's ratio
NU_INCLUSION = 0.22    # inclusion Poisson's ratio
KAPPAS: List[int] = [48]   # κ = E_inc / E_mat

# Geometry: single sphere centred in the cube
SPHERE_RADIUS = 10.0   # pixels (= mm, since 1 mm/voxel)
VF_ANALYTICAL = (4.0 / 3.0 * np.pi * SPHERE_RADIUS**3) / N**3   # ≈ 0.1278

# LS-FNO architecture
M_DEPTHS: List[int] = [11]   # FNO7 / FNO9 / FNO11
CUTOFF_M = 1.0                      # strain clipping bound M for τ_θ
DEPTH_K  = 4                        # FNO layers used by forward(); solve() ignores it

# Macroscopic strain magnitude — paper §5.2.1: "six independent macroscopic
# strains of magnitude 0.1%".  Both FFT and FNO receive this value directly;
# no normalisation or scaling is applied.
#
# The FNO's Yarotsky τ_θ previously required a workaround here because float32
# catastrophic cancellation in q((T+ε)/2M) − q(T/2M) created a residual noise
# floor at ~1.5e-5 when ε ≈ 0.001.  That bug is fixed by computing the Yarotsky
# arithmetic in float64 inside YarotskyTauTheta.forward (ls_fno.py), so both
# solvers can now use the same physical 0.1% strain directly.
EPS_PHYSICAL = 1e-3      # 0.1 %

# Solver (shared between FFT and LS-FNO so iteration counts are comparable)
TOL      = 1e-5
MAX_ITER = 2000
DISC     = 'staggered'   # Willot (2015) rotated staggered grid


# ─────────────────────────────────────────────────────────────────────────────
# Microstructure
# ─────────────────────────────────────────────────────────────────────────────

def make_centered_sphere(N: int, r: float) -> np.ndarray:
    """
    Single sphere of radius r (pixels) at the centre of an N³ domain.

    Returns (N, N, N) bool array; True = inclusion.
    The centre is placed at voxel coordinate (N-1)/2 on each axis so that the
    sphere sits exactly in the middle of the periodic cell (not shifted by ±0.5
    due to integer vs half-integer grid conventions).
    """
    c = (N - 1) / 2.0
    xs, ys, zs = np.mgrid[0:N, 0:N, 0:N]
    return ((xs - c)**2 + (ys - c)**2 + (zs - c)**2) <= r**2


# ─────────────────────────────────────────────────────────────────────────────
# Effective-stiffness wrappers
# ─────────────────────────────────────────────────────────────────────────────

def effective_stiffness_fft(
    C_field: np.ndarray,
    alpha0: float,
) -> Tuple[np.ndarray, List[int]]:
    """
    6×6 effective stiffness via 6 load cases at EPS_PHYSICAL = 0.1% strain.

    Uses explicit alpha0 = (α⁻ + α⁺)/2 from true tensor eigenvalue bounds
    (not the C₁₁₁₁ heuristic inside fft_solve).

    Returns:
        C_eff:   (6, 6) effective stiffness [same units as C_field].
        n_iters: list[int] — iteration count for each of the 6 load cases.
    """
    C_eff   = np.zeros((N_COMP, N_COMP))
    n_iters: List[int] = []
    for a in range(N_COMP):
        eps_bar    = np.zeros(N_COMP)
        eps_bar[a] = EPS_PHYSICAL
        result     = fft_solve(
            C_field, eps_bar,
            alpha0=alpha0, tol=TOL, max_iter=MAX_ITER, discretization=DISC, verbose = True
        )
        spatial_axes  = tuple(range(1, result['sigma_star'].ndim))
        C_eff[:, a]   = result['sigma_star'].mean(axis=spatial_axes) / EPS_PHYSICAL
        n_iters.append(result['n_iter'])
    return C_eff, n_iters


def effective_stiffness_lsfno(
    model: LSFNO,
    C_field_t: torch.Tensor,
) -> Tuple[np.ndarray, List[int]]:
    """
    6×6 effective stiffness via 6 LS-FNO load cases at EPS_PHYSICAL = 0.1% strain.

    Runs model.solve() (dynamic depth) so iteration counts are directly
    comparable to the FFT reference.  The Yarotsky τ_θ now computes in float64
    internally (ls_fno.py), so 0.1% strain converges without any scaling workaround.

    Args:
        model:      Pre-built LSFNO instance for this (κ, m) combination.
        C_field_t:  (6, 6, N, N, N) Voigt stiffness field as a float32 tensor.

    Returns:
        C_eff:   (6, 6) effective stiffness.
        n_iters: list[int] — iteration count for each of the 6 load cases.
    """
    C_batch = C_field_t.unsqueeze(0)    # (1, 6, 6, N, N, N) — batch of size 1
    C_eff   = np.zeros((N_COMP, N_COMP))
    n_iters: List[int] = []

    for a in range(N_COMP):
        eps_bar       = torch.zeros(1, N_COMP)
        eps_bar[0, a] = EPS_PHYSICAL
        result        = model.solve(C_batch, eps_bar, verbose=True)
        eps_star      = result['eps_star']           # (1, 6, N, N, N) Voigt strain
        # σ(x) = C(x) : ε*(x)  —  field-wise Voigt contraction
        sigma         = torch.einsum('bijxyz,bjxyz->bixyz', C_batch, eps_star)
        C_eff[:, a]   = sigma.mean(dim=(-3, -2, -1))[0].cpu().numpy() / EPS_PHYSICAL
        n_iters.append(result['n_iter'])

    return C_eff, n_iters


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    torch.set_grad_enabled(False)

    # ── Header ────────────────────────────────────────────────────────────────
    print("=" * 78)
    print("  §5.2.1  Effective stiffness — 3D sphere-in-cube composite")
    print("=" * 78)
    print(f"  Grid:      {N}³   disc: {DISC}")
    print(f"  Geometry:  r = {SPHERE_RADIUS:.0f} px  in  L = {N} px cube"
          f"  →  VF_analytical ≈ {VF_ANALYTICAL:.4f}")
    print(f"  Material:  E_mat = {E_MATRIX} GPa,  ν_mat = {NU_MATRIX},"
          f"  ν_inc = {NU_INCLUSION},  κ ∈ {KAPPAS}")
    print(f"  Solver:    tol = {TOL:.0e},  max_iter = {MAX_ITER}")
    print()

    # Fixed microstructure — same for all κ
    phase     = make_centered_sphere(N, SPHERE_RADIUS)
    vf_actual = float(phase.mean())
    print(f"  Voxel VF: {vf_actual:.4f}  (analytical: {VF_ANALYTICAL:.4f})")
    print()

    # ── Table header ──────────────────────────────────────────────────────────
    hdr_fmt = "{:>4}  {:>8}  {:>12}  {:>12}  {:>12}  {:>10}"
    row_fmt = "{:>4}  {:>8}  {:>12.4f}  {:>12.4f}  {:>12.4f}  {:>10.1f}"
    hdr = hdr_fmt.format("κ", "Model", "C₁₁ [GPa]", "C₁₂ [GPa]", "C₄₄ [GPa]", "avg iter")
    print(hdr)
    print("-" * len(hdr))

    # ── Sweep over κ ──────────────────────────────────────────────────────────
    for kappa in KAPPAS:
        E_inc = E_MATRIX * kappa

        # Build stiffness field for this κ
        C_mat   = isotropic_stiffness_voigt_3d(E_MATRIX, NU_MATRIX)
        C_inc   = isotropic_stiffness_voigt_3d(E_inc,    NU_INCLUSION)
        C_field = build_C_field(phase, C_mat, C_inc)   # (6, 6, 32, 32, 32)

        # Optimal α₀ from true spectral bounds (eq. 2.19 of the paper)
        alpha_minus, alpha_plus = compute_alpha_bounds(
            E_MATRIX, NU_MATRIX, NU_INCLUSION, kappa, dim=DIM
        )
        alpha0 = (alpha_minus + alpha_plus) / 2.0

        # ── FFT reference ─────────────────────────────────────────────────
        C_eff_fft, iters_fft = effective_stiffness_fft(C_field, alpha0)
        print(row_fmt.format(
            kappa, "FFT",
            C_eff_fft[0, 0], C_eff_fft[0, 1], C_eff_fft[3, 3],
            float(np.mean(iters_fft)),
        ))

        # ── LS-FNO (m = 7, 9, 11) ─────────────────────────────────────────
        C_field_t = torch.from_numpy(C_field).float()
        for m in M_DEPTHS:
            model = LSFNO(
                grid_size      = N,
                depth_K        = DEPTH_K,
                alpha_minus    = alpha_minus,
                alpha_plus     = alpha_plus,
                tol            = TOL,
                max_iter       = MAX_ITER,
                tau_theta      = YarotskyTauTheta(depth_m=m, cutoff_M=CUTOFF_M),
                dim            = DIM,
                discretization = DISC,
            )
            C_eff_fno, iters_fno = effective_stiffness_lsfno(model, C_field_t)
            print(row_fmt.format(
                kappa, f"FNO{m}",
                C_eff_fno[0, 0], C_eff_fno[0, 1], C_eff_fno[3, 3],
                float(np.mean(iters_fno)),
            ))

        print()   # blank line between κ blocks

    print("=" * 78)


if __name__ == "__main__":
    main()
