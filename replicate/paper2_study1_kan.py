"""
Study 1: KAN-exact τ_θ vs FFT reference and Yarotsky FNO variants.

Reference: LS_KAN_FNO/KAN_Architecture_Theory.md (§ Study 1)

What this script does
---------------------
Reproduces the §5.2.1 table from paper2_section521.py, but adds the KAN-exact
model alongside FFT, FNO7, FNO9, FNO11.  Since the KAN-exact τ_θ computes
T:ε algebraically (up to float64 machine ε), the expected outcomes are:

  1.  Effective stiffness (C₁₁, C₁₂, C₄₄): KAN-exact == FFT to 12+ digits.
  2.  Iteration count: KAN-exact matches FFT exactly (Δ = 0) for every load
      case at every κ — unlike FNO7/FNO9/FNO11 which may differ by 1–15 iter.
  3.  Strain-magnitude sweep (D2 diagnostic): KAN-exact iteration count is
      strictly flat across ε̄ ∈ {1e-4, 1e-3, 1e-2, 5e-1}, confirming α_eff = 0.

Architecture
------------
  KAN-exact:   KANTauTheta(R=1.0, shared=True, trainable=False)
               Control points fixed at [1, −1, 1] → φ(x) = x² exactly.
               324 parameters if independent edges; 3 if shared.
               No ridge-function clipping.

Sweep
-----
  κ ∈ {12, 96}     — one easy (κ=12) and one hard (κ=96) contrast.
  m ∈ {7, 9, 11}   — Yarotsky depths for FNO comparison.
  ε̄ ∈ {1e-4, 1e-3, 1e-2, 5e-1} — strain sweep for the D2 check.

Usage (from project root):
    python replicate/paper2_study1_kan.py
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from generation.microstructure import isotropic_stiffness_voigt_3d, build_C_field
from generation.fft_solver import solve as fft_solve
from models.ls_fno import LSFNO, YarotskyTauTheta
from models.kan_tau_theta import KANTauTheta
from utils.config_loader import compute_alpha_bounds


# ─────────────────────────────────────────────────────────────────────────────
# Parameters  (mirror paper2_section521.py conventions)
# ─────────────────────────────────────────────────────────────────────────────

N             = 32
DIM           = 3
N_COMP        = 6
E_MATRIX      = 3.0
NU_MATRIX     = 0.3
NU_INCLUSION  = 0.22
SPHERE_RADIUS = 10.0
EPS_PHYSICAL  = 1e-3      # 0.1% — six independent macroscopic strains

TOL      = 1e-5
MAX_ITER = 2000
DISC     = 'staggered'
DEPTH_K  = 4
CUTOFF_M = 1.0    # Yarotsky clipping bound (also used as R for KAN for fair comparison)

KAPPAS         = [12, 48]
YAROTSKY_DEPTHS = [11]

# D2 strain sweep (Study 1 key diagnostic — must be flat for KAN, drifts for FNO)
STRAIN_MAGS = [1e-4, 1e-3, 1e-2, 5e-1]
SWEEP_LOAD_CASE = 0  # ε̄₁₁ — normal case; shows the largest drift for FNO11 at high κ


# ─────────────────────────────────────────────────────────────────────────────
# Microstructure helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_centered_sphere(N: int, r: float) -> np.ndarray:
    c = (N - 1) / 2.0
    xs, ys, zs = np.mgrid[0:N, 0:N, 0:N]
    return ((xs - c)**2 + (ys - c)**2 + (zs - c)**2) <= r**2


def make_materials(kappa: float):
    """
    Returns (C_field_f64, C_field_t_f32, alpha_m, alpha_p, alpha0).
    C_field_f64 is the numpy stiffness field shared between all solvers.
    """
    phase   = make_centered_sphere(N, SPHERE_RADIUS)
    C_mat   = isotropic_stiffness_voigt_3d(E_MATRIX, NU_MATRIX)
    C_inc   = isotropic_stiffness_voigt_3d(E_MATRIX * kappa, NU_INCLUSION)
    C_field = build_C_field(phase, C_mat, C_inc)
    C_field_t = torch.from_numpy(C_field).float()
    alpha_m, alpha_p = compute_alpha_bounds(
        E_MATRIX, NU_MATRIX, NU_INCLUSION, kappa, dim=DIM
    )
    alpha0 = (alpha_m + alpha_p) / 2.0
    return C_field, C_field_t, alpha_m, alpha_p, alpha0


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def make_fno(alpha_m, alpha_p, depth_m: int) -> LSFNO:
    return LSFNO(
        grid_size=N, depth_K=DEPTH_K,
        alpha_minus=alpha_m, alpha_plus=alpha_p,
        tol=TOL, max_iter=MAX_ITER,
        tau_theta=YarotskyTauTheta(depth_m=depth_m, cutoff_M=CUTOFF_M),
        dim=DIM, discretization=DISC,
    )


def make_kan(alpha_m, alpha_p, shared: bool = True) -> LSFNO:
    """
    KAN-exact: B-spline edges with fixed exact-x² control points.
    R = CUTOFF_M so that the pre-scaling is identical to the Yarotsky construction.
    """
    return LSFNO(
        grid_size=N, depth_K=DEPTH_K,
        alpha_minus=alpha_m, alpha_plus=alpha_p,
        tol=TOL, max_iter=MAX_ITER,
        tau_theta=KANTauTheta(R=CUTOFF_M, shared=shared,
                              trainable=False, n_comp=N_COMP),
        dim=DIM, discretization=DISC,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Effective stiffness computation (matches paper2_section521.py)
# ─────────────────────────────────────────────────────────────────────────────

def effective_stiffness_fft(
    C_field: np.ndarray,
    alpha0: float,
) -> Tuple[np.ndarray, List[int]]:
    """6×6 effective stiffness from 6 FFT load cases.  Returns (C_eff, n_iters)."""
    C_eff   = np.zeros((N_COMP, N_COMP))
    n_iters: List[int] = []
    for a in range(N_COMP):
        eps_bar    = np.zeros(N_COMP)
        eps_bar[a] = EPS_PHYSICAL
        result = fft_solve(
            C_field, eps_bar,
            alpha0=alpha0, tol=TOL, max_iter=MAX_ITER, discretization=DISC,
        )
        spatial_axes = tuple(range(1, result['sigma_star'].ndim))
        C_eff[:, a]  = result['sigma_star'].mean(axis=spatial_axes) / EPS_PHYSICAL
        n_iters.append(result['n_iter'])
    return C_eff, n_iters


def effective_stiffness_model(
    model: LSFNO,
    C_field_t: torch.Tensor,
) -> Tuple[np.ndarray, List[int]]:
    """6×6 effective stiffness from 6 model load cases.  Returns (C_eff, n_iters)."""
    C_batch = C_field_t.unsqueeze(0)
    C_eff   = np.zeros((N_COMP, N_COMP))
    n_iters: List[int] = []
    for a in range(N_COMP):
        eps_bar       = torch.zeros(1, N_COMP)
        eps_bar[0, a] = EPS_PHYSICAL
        result        = model.solve(C_batch, eps_bar)
        eps_star      = result['eps_star']
        sigma         = torch.einsum('bijxyz,bjxyz->bixyz', C_batch, eps_star)
        C_eff[:, a]   = sigma.mean(dim=(-3, -2, -1))[0].cpu().numpy() / EPS_PHYSICAL
        n_iters.append(result['n_iter'])
    return C_eff, n_iters


# ─────────────────────────────────────────────────────────────────────────────
# D2 strain-magnitude sweep
# ─────────────────────────────────────────────────────────────────────────────

def strain_sweep(
    C_field: np.ndarray,
    C_field_t: torch.Tensor,
    alpha0: float,
    alpha_m: float,
    alpha_p: float,
    kappa: int,
) -> Dict:
    """
    Run load case SWEEP_LOAD_CASE at each strain magnitude for FFT, KAN-exact,
    and FNO7/FNO11.

    Returns {model_name: {eps_mag: n_iter}}.
    LS is linear → FFT and KAN-exact counts must be flat.  FNO drifts with ε̄.
    """
    C_batch  = C_field_t.unsqueeze(0)
    kan_model  = make_kan(alpha_m, alpha_p)
    fno7_model = make_fno(alpha_m, alpha_p, 7)
    fno11_model = make_fno(alpha_m, alpha_p, 11)

    results: Dict[str, Dict] = {
        'FFT': {}, 'KAN-exact': {}, 'FNO7': {}, 'FNO11': {}
    }

    for eps_mag in STRAIN_MAGS:
        # FFT
        eps_np = np.zeros(N_COMP); eps_np[SWEEP_LOAD_CASE] = eps_mag
        r_fft  = fft_solve(C_field, eps_np, alpha0=alpha0,
                           tol=TOL, max_iter=MAX_ITER, discretization=DISC)
        results['FFT'][eps_mag] = {
            'n_iter': r_fft['n_iter'],
            'converged': r_fft['converged'],
        }

        # LS-FNO variants (KAN and Yarotsky)
        eps_t = torch.zeros(1, N_COMP); eps_t[0, SWEEP_LOAD_CASE] = float(eps_mag)
        for name, model in [('KAN-exact', kan_model),
                            ('FNO7',      fno7_model),
                            ('FNO11',     fno11_model)]:
            r = model.solve(C_batch, eps_t)
            results[name][eps_mag] = {
                'n_iter': r['n_iter'],
                'converged': r['converged'],
            }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Printing helpers
# ─────────────────────────────────────────────────────────────────────────────

LOAD_NAMES = [
    'ε̄₁₁ (normal x)', 'ε̄₂₂ (normal y)', 'ε̄₃₃ (normal z)',
    'γ̄₂₃ (shear yz)',  'γ̄₁₃ (shear xz)',  'γ̄₁₂ (shear xy)',
]


def print_effective_stiffness_row(
    label: str,
    kappa: int,
    C_eff: np.ndarray,
    n_iters: List[int],
    C_ref: np.ndarray,
) -> None:
    avg_iter = float(np.mean(n_iters))
    err_C11  = abs(C_eff[0, 0] - C_ref[0, 0])
    print(f"  {kappa:>4}  {label:<12}  "
          f"{C_eff[0,0]:>10.4f}  {C_eff[0,1]:>10.4f}  {C_eff[3,3]:>10.4f}  "
          f"{avg_iter:>10.1f}  {err_C11:>12.2e}")


def print_per_load_case(label: str, n_iters_model: List[int],
                         n_iters_fft: List[int]) -> None:
    print(f"\n  {label} vs FFT — per-load-case iteration count:")
    print(f"  {'#':>2}  {'Load case':<22}  {'FFT':>6}  {label:>12}  {'Δ':>5}")
    print(f"  {'─'*56}")
    for a, (nf, nm) in enumerate(zip(n_iters_fft, n_iters_model)):
        delta = nf - nm
        marker = "" if delta == 0 else (" ←FNO faster" if delta > 0 else " ←FNO slower")
        print(f"  {a:>2}  {LOAD_NAMES[a]:<22}  {nf:>6}  {nm:>12}  {delta:>5}{marker}")
    avg_f = np.mean(n_iters_fft)
    avg_m = np.mean(n_iters_model)
    print(f"  {'Average':>28}  {avg_f:>6.1f}  {avg_m:>12.1f}  {avg_f-avg_m:>5.1f}")


def print_strain_sweep(kappa: int, sweep_results: Dict) -> None:
    names = ['FFT', 'KAN-exact', 'FNO7', 'FNO11']
    print(f"\n  D2 strain-magnitude sweep  (κ={kappa}, load case {SWEEP_LOAD_CASE} = ε̄₁₁)")
    print(f"  {'ε̄':>10}  " + "  ".join(f"{'N_iter':>10}" for _ in names))
    print(f"  {'':>10}  " + "  ".join(f"{n:>10}" for n in names))
    print(f"  {'─'*62}")

    ref_iters = {n: None for n in names}
    for eps_mag in STRAIN_MAGS:
        row = f"  {eps_mag:>10.0e}  "
        for name in names:
            ni = sweep_results[name][eps_mag]['n_iter']
            conv = "✓" if sweep_results[name][eps_mag]['converged'] else "✗"
            if ref_iters[name] is None:
                ref_iters[name] = ni
            drift = ni - ref_iters[name]
            drift_str = f" ({drift:+d})" if drift != 0 else "     "
            row += f"{ni:>7}{conv}{drift_str}  "
        print(row)

    print()
    print("  Drift is relative to the first row (ε̄=1e-4).")
    print("  FFT and KAN-exact must show 0 drift (LS is linear).")
    print("  FNO drift reveals strain-scale-dependent α_eff.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    torch.set_grad_enabled(False)

    print("=" * 78)
    print("  Study 1: KAN-exact τ_θ vs FFT and Yarotsky FNO variants")
    print("  Ref: KAN_Architecture_Theory.md")
    print(f"  Grid: {N}³   disc: {DISC}   tol: {TOL:.0e}   ε̄_physical: {EPS_PHYSICAL:.0e}")
    print(f"  κ ∈ {KAPPAS}   Yarotsky depths ∈ {YAROTSKY_DEPTHS}")
    print(f"  KAN: shared edges, R={CUTOFF_M}, fixed exact control points")
    print("=" * 78)

    # ── KAN parameter count overview ──────────────────────────────────────────
    print()
    print("  Parameter counts for τ_θ:")
    for depth in YAROTSKY_DEPTHS:
        print(f"    FNO{depth}  (Yarotsky):    0 trainable  "
              f"(analytic, depth-{depth} ReLU construction)")
    kan_s = KANTauTheta(R=CUTOFF_M, shared=True,  trainable=False, n_comp=N_COMP)
    kan_i = KANTauTheta(R=CUTOFF_M, shared=False, trainable=False, n_comp=N_COMP)
    print(f"    KAN-exact (shared):   {kan_s.n_ctrl_points()} control points  "
          f"(all {N_COMP}×{N_COMP} edges share one B-spline)")
    print(f"    KAN-exact (indep):    {kan_i.n_ctrl_points()} control points  "
          f"(one B-spline per (i,j) edge)")

    # ── Effective stiffness table ─────────────────────────────────────────────
    print()
    hdr = (f"  {'κ':>4}  {'Model':<12}  "
           f"{'C₁₁ [GPa]':>10}  {'C₁₂ [GPa]':>10}  {'C₄₄ [GPa]':>10}  "
           f"{'avg iter':>10}  {'|ΔC₁₁|':>12}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for kappa in KAPPAS:
        C_field, C_field_t, alpha_m, alpha_p, alpha0 = make_materials(kappa)
        gamma = (alpha_p - alpha_m) / (alpha_p + alpha_m)
        print(f"\n  κ={kappa}   α₀={alpha0:.4f}   γ_theoretical={gamma:.6f}")

        # ── FFT reference ─────────────────────────────────────────────────
        C_fft, iters_fft = effective_stiffness_fft(C_field, alpha0)
        print_effective_stiffness_row("FFT", kappa, C_fft, iters_fft, C_fft)

        # ── KAN-exact ─────────────────────────────────────────────────────
        kan = make_kan(alpha_m, alpha_p)
        C_kan, iters_kan = effective_stiffness_model(kan, C_field_t)
        print_effective_stiffness_row("KAN-exact", kappa, C_kan, iters_kan, C_fft)

        # ── Yarotsky FNO variants ─────────────────────────────────────────
        for depth in YAROTSKY_DEPTHS:
            fno = make_fno(alpha_m, alpha_p, depth)
            C_fno, iters_fno = effective_stiffness_model(fno, C_field_t)
            print_effective_stiffness_row(
                f"FNO{depth}", kappa, C_fno, iters_fno, C_fft
            )

        # ── Per-load-case breakdown ────────────────────────────────────────
        print_per_load_case("KAN-exact", iters_kan, iters_fft)

        # ── D2 strain sweep ────────────────────────────────────────────────
        print(f"\n  Running D2 strain sweep for κ={kappa} …")
        sweep = strain_sweep(C_field, C_field_t, alpha0, alpha_m, alpha_p, kappa)
        print_strain_sweep(kappa, sweep)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 78)
    print("  Study 1 expected outcomes:")
    print()
    print("  ✓  |ΔC₁₁| for KAN-exact ≈ machine epsilon (~1e-13 GPa)")
    print("     (Yarotsky FNO variants have |ΔC₁₁| at the Yarotsky approximation level)")
    print()
    print("  ✓  Iteration count: KAN-exact matches FFT exactly (Δ=0 every load case)")
    print("     (FNO variants show Δ≠0 at high κ, especially for normal load cases)")
    print()
    print("  ✓  D2 drift for KAN-exact: 0 across all ε̄ (LS is linear, B-spline exact)")
    print("     (FNO7/FNO11 drift by 5–15 iterations at κ=96 over 4 orders of magnitude)")
    print("=" * 78)


if __name__ == "__main__":
    main()
