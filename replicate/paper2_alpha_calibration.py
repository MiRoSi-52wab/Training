"""
α-perturbation calibration: characterising the FNO's effective contraction bias.

Reference: LS_KAN_FNO/fno_diagnostic_summary.md

Background
----------
The Yarotsky approximation τ_θ(T, ε) ≈ T:ε has a small, strain-scale-dependent
error that makes the FNO solve a slightly perturbed problem.  The net effect can
be modelled as a scalar perturbation α to the contraction operator:

    τ_θ(T, ε) ≈ (1 + α_eff) · T : ε

  α_eff < 0  →  undershoot  →  effectively smaller contrast  →  faster than FFT
  α_eff > 0  →  overshoot   →  effectively larger  contrast  →  slower than FFT
  α_eff = 0  →  FNO is iteration-for-iteration identical to FFT

This script
  1. Sweeps α in a perturbed FFT basic scheme
         τ_pert(ε) = (1+α) · (C − C⁰) : ε
     and measures N_iter(α) — the calibration curve.
  2. Overlays the theoretical prediction
         N_iter(α) ≈ N_iter(0) · log(γ̂) / log[(1+α)·γ̂]
     where γ̂ = (tol/res₀)^(1/N₀) is the empirical per-iteration reduction rate.
  3. Runs FNO7 and FNO11 at four strain magnitudes (1e-4, 1e-3, 1e-2, 5e-1)
     on load case 0 (ε̄₁₁ — the normal case with the largest iteration count).
  4. Inverts the theoretical curve to get the FNO's effective α at each strain:
         α_eff = γ̂^(N_FFT / N_FNO − 1) − 1
  5. Produces two figures saved under replicate/figures/:
       alpha_calibration_curves.png   — N_iter(α) curves + FNO horizontal lines
       effective_alpha_vs_strain.png  — α_eff vs ε̄ for FNO7/FNO11 at each κ
  6. Prints a sensitivity table: ΔN_iter per 1% perturbation in α.

Usage (from project root):
    python replicate/paper2_alpha_calibration.py

Expected runtime: ~30–60 min (32³ grid, κ ∈ {12, 96}, full sweep).
For a quick smoke test use KAPPAS = [12] and shorten ALPHA_SWEEP.
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
from utils.config_loader import compute_alpha_bounds

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ─────────────────────────────────────────────────────────────────────────────
# Experiment parameters
# ─────────────────────────────────────────────────────────────────────────────

N             = 32
DIM           = 3
N_COMP        = 6
E_MATRIX      = 3.0
NU_MATRIX     = 0.3
NU_INCLUSION  = 0.22
SPHERE_RADIUS = 10.0
TOL           = 1e-5
MAX_ITER      = 2000
DISC          = 'staggered'
CUTOFF_M      = 1.0
DEPTH_K       = 4
LOAD_CASE     = 0          # ε̄₁₁ — normal case (largest iteration count at high κ)

KAPPAS      = [96]
M_DEPTHS    = [11]
STRAIN_MAGS = [1e-4, 1e-3, 1e-2, 5e-1]

# α-sweep: the calibration grid (as suggested in fno_diagnostic_summary.md)
ALPHA_SWEEP = np.array([
    -0.05, -0.02, -0.01, -0.005, -0.002, -0.001,
     0.0,
     0.001,  0.002,  0.005,  0.01,  0.02,  0.05,
])

# Strain magnitude used for the α-sweep (the physical 0.1% value)
ALPHA_SWEEP_EPS = 1e-3

FIG_DIR = Path(__file__).parent / "figures"


# ─────────────────────────────────────────────────────────────────────────────
# Microstructure helpers
# ─────────────────────────────────────────────────────────────────────────────

def centered_sphere(N: int, r: float) -> np.ndarray:
    c = (N - 1) / 2.0
    xs, ys, zs = np.mgrid[0:N, 0:N, 0:N]
    return ((xs - c)**2 + (ys - c)**2 + (zs - c)**2) <= r**2


def make_stiffness(kappa: float):
    """
    Build stiffness field and material parameters for a given κ.

    Returns: (C_field_f64, C_field_t_f32, alpha_m, alpha_p, alpha0, gamma_th)
      gamma_th = theoretical contraction constant (α₊ − α₋)/(α₊ + α₋)
    """
    phase   = centered_sphere(N, SPHERE_RADIUS)
    C_mat   = isotropic_stiffness_voigt_3d(E_MATRIX, NU_MATRIX)
    C_inc   = isotropic_stiffness_voigt_3d(E_MATRIX * kappa, NU_INCLUSION)
    C_field = build_C_field(phase, C_mat, C_inc)
    C_field_t = torch.from_numpy(C_field).float()
    alpha_m, alpha_p = compute_alpha_bounds(
        E_MATRIX, NU_MATRIX, NU_INCLUSION, kappa, dim=DIM
    )
    alpha0   = (alpha_m + alpha_p) / 2.0
    gamma_th = (alpha_p - alpha_m) / (alpha_p + alpha_m)
    return C_field, C_field_t, alpha_m, alpha_p, alpha0, gamma_th


def make_model(alpha_m: float, alpha_p: float, depth_m: int) -> LSFNO:
    return LSFNO(
        grid_size=N, depth_K=DEPTH_K,
        alpha_minus=alpha_m, alpha_plus=alpha_p,
        tol=TOL, max_iter=MAX_ITER,
        tau_theta=YarotskyTauTheta(depth_m=depth_m, cutoff_M=CUTOFF_M),
        dim=DIM, discretization=DISC,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Perturbed stiffness: C_pert = C⁰ + (1+α)·(C − C⁰)
# ─────────────────────────────────────────────────────────────────────────────

def make_perturbed_C_field(C_field: np.ndarray, alpha0: float,
                            alpha_pert: float) -> np.ndarray:
    """
    Scale the stiffness contrast by (1+α):
        C_pert = C⁰ + (1+α)·(C − C⁰)

    Passing C_pert with the same alpha0 to fft_solve gives:
        dC_inside = C_pert − C⁰ = (1+α)·(C − C⁰) = (1+α)·dC
        τ_pert = dC_inside : ε = (1+α) · τ

    The Green operator Γ̂ = Γ(α₀) is unchanged.
    """
    n   = C_field.shape[0]
    dim = C_field.ndim - 2
    # C⁰ = α₀ · diag([1,1,1,0.5,0.5,0.5]) in Voigt (engineering shear convention)
    C0_diag         = np.ones(n, dtype=np.float64)
    C0_diag[DIM:]   = 0.5
    C0              = alpha0 * np.diag(C0_diag)
    C0_bc           = C0.reshape(n, n, *([1] * dim))
    return C0_bc + (1.0 + alpha_pert) * (C_field.astype(np.float64) - C0_bc)


# ─────────────────────────────────────────────────────────────────────────────
# Theoretical formulas
# ─────────────────────────────────────────────────────────────────────────────

def gamma_empirical(n_iter: int, res0: float) -> float:
    """
    Per-iteration reduction rate from a single converged run.
    Geometric convergence assumption: res_k ≈ res₀ · γ̂^k
    → γ̂ = (tol / res₀)^(1 / N_iter)
    """
    return float((TOL / max(res0, 1e-30)) ** (1.0 / max(n_iter, 1)))


def theoretical_n_iter(n0: float, gamma: float,
                       alpha_values: np.ndarray) -> np.ndarray:
    """
    N_iter(α) ≈ N₀ · log(γ̂) / log[(1+α)·γ̂]

    Diverges when (1+α)·γ̂ ≥ 1 (iteration no longer converges).
    Returns NaN for those α values.
    """
    inner = (1.0 + alpha_values) * gamma
    result = np.full_like(alpha_values, np.nan, dtype=float)
    valid = inner < 1.0
    result[valid] = n0 * np.log(gamma) / np.log(inner[valid])
    return result


def invert_alpha(n_fno: int, n_fft: int, gamma: float) -> float:
    """
    Invert the theoretical N_iter(α) curve to recover the FNO's effective α:
        α_eff = γ̂^(N_FFT/N_FNO − 1) − 1

    Derivation: set N_FNO = N_FFT·log(γ̂)/log[(1+α)γ̂], solve for α.
    Sign convention:
      N_FNO < N_FFT  →  N_FFT/N_FNO > 1  →  γ̂^(positive) < 1  →  α_eff < 0 (undershoot)
      N_FNO > N_FFT  →  N_FFT/N_FNO < 1  →  γ̂^(negative) > 1  →  α_eff > 0 (overshoot)
    """
    if n_fno <= 0:
        return float('nan')
    return float(gamma ** (n_fft / n_fno - 1.0) - 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Sweep routines
# ─────────────────────────────────────────────────────────────────────────────

def run_alpha_sweep(
    C_field: np.ndarray,
    alpha0: float,
    alpha_values: np.ndarray,
    eps_mag: float = ALPHA_SWEEP_EPS,
) -> Tuple[List[int], List[float], float]:
    """
    Run the perturbed FFT across all α values on load case LOAD_CASE.

    Returns:
      n_iters:   iteration count for each α
      res0s:     first residual for each α (should be constant — sanity check)
      gamma_hat: empirical γ̂ from the unperturbed (α=0) run
    """
    eps_bar = np.zeros(N_COMP)
    eps_bar[LOAD_CASE] = eps_mag

    n_iters: List[int] = []
    res0s:   List[float] = []

    for alpha in alpha_values:
        C_pert = make_perturbed_C_field(C_field, alpha0, alpha)
        r = fft_solve(C_pert, eps_bar, alpha0=alpha0,
                      tol=TOL, max_iter=MAX_ITER, discretization=DISC)
        n_iters.append(r['n_iter'])
        res0s.append(r['residuals'][0])

    # γ̂ from the α=0 entry (closest to 0 if 0.0 not exactly in array)
    idx0      = int(np.argmin(np.abs(alpha_values)))
    gamma_hat = gamma_empirical(n_iters[idx0], res0s[idx0])
    return n_iters, res0s, gamma_hat


def run_fno_strain_sweep(
    C_field: np.ndarray,
    C_field_t: torch.Tensor,
    alpha_m: float,
    alpha_p: float,
    alpha0: float,
) -> Dict:
    """
    For each (depth, strain_mag), run FFT and FNO on load case LOAD_CASE.

    Returns a nested dict:
      {
        depth: {
          'fft':      {eps_mag: {'n_iter', 'res0', 'converged'}},
          'fno':      {eps_mag: {'n_iter', 'res0', 'converged'}},
          'alpha_eff':{eps_mag: float},
          'gamma_hat': float,
        }
      }

    γ̂ is computed from the ε̄=ALPHA_SWEEP_EPS FFT run (the standard physical scale).
    """
    C_batch = C_field_t.unsqueeze(0)
    results = {}

    for depth in M_DEPTHS:
        model    = make_model(alpha_m, alpha_p, depth)
        fft_data: Dict = {}
        fno_data: Dict = {}

        for eps_mag in STRAIN_MAGS:
            eps_np = np.zeros(N_COMP); eps_np[LOAD_CASE] = eps_mag
            r_fft  = fft_solve(C_field, eps_np, alpha0=alpha0,
                               tol=TOL, max_iter=MAX_ITER, discretization=DISC)
            fft_data[eps_mag] = {
                'n_iter':    r_fft['n_iter'],
                'res0':      r_fft['residuals'][0],
                'converged': r_fft['converged'],
            }

            eps_t = torch.zeros(1, N_COMP); eps_t[0, LOAD_CASE] = float(eps_mag)
            r_fno = model.solve(C_batch, eps_t)
            fno_data[eps_mag] = {
                'n_iter':    r_fno['n_iter'],
                'res0':      r_fno['residuals'][0],
                'converged': r_fno['converged'],
            }

        # γ̂ from the standard physical strain
        ref_eps   = ALPHA_SWEEP_EPS
        gamma_hat = gamma_empirical(fft_data[ref_eps]['n_iter'],
                                    fft_data[ref_eps]['res0'])

        alpha_eff = {
            eps_mag: invert_alpha(
                fno_data[eps_mag]['n_iter'],
                fft_data[eps_mag]['n_iter'],
                gamma_hat,
            )
            for eps_mag in STRAIN_MAGS
        }

        results[depth] = {
            'fft':       fft_data,
            'fno':       fno_data,
            'alpha_eff': alpha_eff,
            'gamma_hat': gamma_hat,
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Console output
# ─────────────────────────────────────────────────────────────────────────────

def print_sensitivity_table(kappas: List[int],
                             gamma_hats: List[float]) -> None:
    """
    Reproduce the table from fno_diagnostic_summary.md §Why approximation…

    ΔN/N₀ per 1% α ≈ 0.01 / log(1/γ̂)
    At high κ (γ̂ close to 1) this becomes very large, making the iteration
    count extremely sensitive to even small contraction perturbations.
    """
    print("\n  Sensitivity: ΔN_iter per 1% perturbation in α")
    print(f"  {'κ':>5}  {'γ̂ (empirical)':>15}  {'log(1/γ̂)':>10}  {'ΔN/N₀ per 1% α':>16}")
    print(f"  {'─'*52}")
    for kappa, gh in zip(kappas, gamma_hats):
        log_inv_g = float(-np.log(gh))
        sens      = 0.01 / log_inv_g
        print(f"  {kappa:>5}  {gh:>15.6f}  {log_inv_g:>10.4f}  {sens*100:>15.1f}%")
    print()
    print("  Interpretation: at κ=96 a 1% perturbation in α changes N_iter by ~48%,")
    print("  which is why even the tiny Yarotsky error (~2×10⁻⁷) produces a visible")
    print("  iteration-count gap. At κ=12 the same error causes negligible deviation.")


def print_alpha_sweep_table(kappa: int, eps_mag: float,
                             alpha_values: np.ndarray,
                             n_iters: List[int],
                             res0s: List[float],
                             gamma_hat: float,
                             n_theory: np.ndarray) -> None:
    print(f"\n  α-sweep  (κ={kappa}, load case {LOAD_CASE}, ε̄={eps_mag:.0e})")
    print(f"  γ̂ = {gamma_hat:.6f}   (empirical from α=0 run)")
    print(f"  {'α':>8}  {'N_FFT_pert':>12}  {'N_theory':>10}  {'res₀':>10}  converged?")
    print(f"  {'─'*56}")
    for a, ni, r0, nt in zip(alpha_values, n_iters, res0s, n_theory):
        nt_str   = f"{nt:10.1f}" if not np.isnan(nt) else "       n/a"
        flag     = "  ← α=0" if abs(a) < 1e-10 else ""
        conv_str = "✓" if ni < MAX_ITER else "✗ (hit limit)"
        print(f"  {a:>8.4f}  {ni:>12}  {nt_str}  {r0:>10.4e}  {conv_str}{flag}")


def print_fno_effective_alpha(kappa: int, sweep_results: Dict) -> None:
    print(f"\n  FNO effective α_eff = γ̂^(N_FFT/N_FNO − 1) − 1  (κ={kappa})")
    for depth in M_DEPTHS:
        d         = sweep_results[depth]
        gamma_hat = d['gamma_hat']
        print(f"\n    FNO{depth}  (γ̂={gamma_hat:.6f})")
        print(f"    {'ε̄':>10}  {'N_FFT':>8}  {'N_FNO':>8}  {'Δ':>5}  "
              f"{'α_eff':>10}  converged?")
        print(f"    {'─'*56}")
        for eps_mag in STRAIN_MAGS:
            n_f  = d['fft'][eps_mag]['n_iter']
            n_n  = d['fno'][eps_mag]['n_iter']
            ae   = d['alpha_eff'][eps_mag]
            conv = "✓" if d['fno'][eps_mag]['converged'] else "✗"
            ae_s = f"{ae:>10.3e}" if not np.isnan(ae) else "       nan"
            bias = ("← FNO faster" if n_n < n_f
                    else ("← FNO slower" if n_n > n_f else "← tied"))
            print(f"    {eps_mag:>10.0e}  {n_f:>8}  {n_n:>8}{conv}  {n_f-n_n:>5}  "
                  f"{ae_s}  {bias}")


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def plot_calibration_curves(
    kappas: List[int],
    alpha_sweep_all:  Dict,   # kappa → np.ndarray of α values
    n_iter_sweep_all: Dict,   # kappa → list[int]
    gamma_hat_all:    Dict,   # kappa → float
    fno_strain_all:   Dict,   # kappa → {depth → {eps_mag → n_iter}}
    save_path: Path,
) -> None:
    """
    Figure 1: N_iter(α) calibration curve for each κ.

    - Blue dots: empirical perturbed-FFT sweep
    - Blue line: theoretical formula N₀·log(γ̂)/log[(1+α)γ̂]
    - Coloured horizontal lines: FNO iteration counts at each strain magnitude
    - The horizontal-line / curve intersection gives the FNO's effective α
    """
    n_cols = len(kappas)
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    depth_colors  = {7: '#E07B39', 11: '#2E86AB'}
    strain_styles = {1e-4: (0, (1, 1)),   # densely dotted
                     1e-3: '--',
                     1e-2: '-.',
                     5e-1: '-'}
    strain_labels = {1e-4: 'ε̄=1e-4', 1e-3: 'ε̄=1e-3',
                     1e-2: 'ε̄=1e-2', 5e-1: 'ε̄=0.5'}

    for ax, kappa in zip(axes, kappas):
        alpha_vals = alpha_sweep_all[kappa]
        n_sweep    = np.array(n_iter_sweep_all[kappa])
        gamma_hat  = gamma_hat_all[kappa]
        idx0       = int(np.argmin(np.abs(alpha_vals)))
        n0         = float(n_sweep[idx0])

        # Theoretical curve on a fine grid
        alpha_fine    = np.linspace(alpha_vals.min() * 1.1,
                                    alpha_vals.max() * 1.1, 400)
        n_theory_fine = theoretical_n_iter(n0, gamma_hat, alpha_fine)
        ax.plot(alpha_fine, n_theory_fine, color='royalblue', lw=2,
                label=f'Theory (γ̂={gamma_hat:.4f})', zorder=3)

        # Empirical dots
        ax.scatter(alpha_vals, n_sweep, color='royalblue', s=50, zorder=5,
                   label='Perturbed FFT')

        # α=0 reference
        ax.axvline(0, color='gray', lw=0.8, ls='--')

        # FNO horizontal lines
        fno_data = fno_strain_all.get(kappa, {})
        for depth in M_DEPTHS:
            if depth not in fno_data:
                continue
            for eps_mag in STRAIN_MAGS:
                n_fno = fno_data[depth].get(eps_mag)
                if n_fno is None:
                    continue
                ax.axhline(
                    n_fno,
                    color=depth_colors[depth],
                    ls=strain_styles[eps_mag],
                    lw=1.4, alpha=0.9,
                    label=f'FNO{depth}, {strain_labels[eps_mag]}',
                )

        ax.set_title(f'κ = {kappa}', fontsize=12)
        ax.set_xlabel('α  (contraction perturbation)', fontsize=10)
        ax.set_ylabel('N_iter to convergence', fontsize=10)
        ax.legend(fontsize=7, loc='upper left', framealpha=0.85)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        r'N_iter(α) calibration: τ_pert(ε) = (1+α)·(C−C⁰):ε'
        '\nHorizontal lines show FNO iter counts; intersection with curve = effective α',
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Figure 1 → {save_path.name}")


def plot_effective_alpha(
    kappas: List[int],
    alpha_eff_all: Dict,   # kappa → {depth → {eps_mag → float}}
    save_path: Path,
) -> None:
    """
    Figure 2: effective α_eff vs strain magnitude for FNO7/FNO11.

    A flat line at α_eff=0 would mean the FNO is a perfect contraction operator.
    Negative slope at small ε̄ (undershoot, FNO faster) and positive values at
    large ε̄ (overshoot from r_θ clipping, FNO slower) are expected for the
    analytic Yarotsky construction.
    """
    n_cols = len(kappas)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4.5))
    if n_cols == 1:
        axes = [axes]

    depth_colors  = {7: '#E07B39', 11: '#2E86AB'}
    depth_markers = {7: 's', 11: 'o'}

    for ax, kappa in zip(axes, kappas):
        ax.axhline(0, color='gray', lw=0.8, ls='--', label='α=0 (perfect)')
        for depth in M_DEPTHS:
            ae_dict = alpha_eff_all.get(kappa, {}).get(depth, {})
            if not ae_dict:
                continue
            eps_sorted = sorted(ae_dict)
            ae_vals    = [ae_dict[e] for e in eps_sorted]
            ax.semilogx(
                eps_sorted, ae_vals,
                color=depth_colors[depth],
                marker=depth_markers[depth],
                label=f'FNO{depth}',
                lw=1.8, ms=7,
            )
            for e, ae in zip(eps_sorted, ae_vals):
                if not np.isnan(ae):
                    ax.annotate(
                        f'{ae:+.1e}', (e, ae),
                        textcoords='offset points', xytext=(5, 4),
                        fontsize=7, color=depth_colors[depth],
                    )

        ax.set_title(f'κ = {kappa}', fontsize=12)
        ax.set_xlabel('ε̄ magnitude', fontsize=10)
        ax.set_ylabel('α_eff', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')

    fig.suptitle(
        r'FNO effective contraction perturbation α_eff = γ̂^(N_FFT/N_FNO − 1) − 1'
        '\nα<0: FNO faster (undershoot)   α>0: FNO slower (overshoot)',
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure 2 → {save_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    torch.set_grad_enabled(False)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  α-Perturbation Calibration Diagnostic")
    print("  Ref: LS_KAN_FNO/fno_diagnostic_summary.md")
    print(f"  Grid: {N}³   disc: {DISC}   tol: {TOL:.0e}   load_case: {LOAD_CASE} (ε̄₁₁)")
    print(f"  κ ∈ {KAPPAS}   depths ∈ {M_DEPTHS}")
    print(f"  α-sweep: {len(ALPHA_SWEEP)} values ∈ [{ALPHA_SWEEP[0]:.3f}, {ALPHA_SWEEP[-1]:.3f}]"
          f"  at ε̄={ALPHA_SWEEP_EPS:.0e}")
    print(f"  FNO strain sweep: ε̄ ∈ {STRAIN_MAGS}")
    print(f"  Figures → {FIG_DIR}/")
    print("=" * 72)

    # Collected results across κ
    alpha_sweep_all   = {}   # kappa → alpha_values array
    n_iter_sweep_all  = {}   # kappa → list[int]
    gamma_hat_all     = {}   # kappa → float
    fno_strain_all    = {}   # kappa → {depth → {eps_mag → n_iter}}
    alpha_eff_all     = {}   # kappa → {depth → {eps_mag → float}}

    for kappa in KAPPAS:
        print(f"\n{'─'*72}")
        print(f"  κ = {kappa}")
        print(f"{'─'*72}")

        C_field, C_field_t, alpha_m, alpha_p, alpha0, gamma_th = make_stiffness(kappa)
        print(f"  α₋={alpha_m:.4f}  α₊={alpha_p:.4f}  α₀={alpha0:.4f}")
        print(f"  γ_theoretical = {gamma_th:.6f}")

        # ── Step 1: α-sweep on the perturbed FFT ──────────────────────────
        total = len(ALPHA_SWEEP)
        print(f"\n  [1/2] α-sweep: {total} perturbed FFT runs at ε̄={ALPHA_SWEEP_EPS:.0e} …")
        n_iters, res0s, gamma_hat = run_alpha_sweep(
            C_field, alpha0, ALPHA_SWEEP, eps_mag=ALPHA_SWEEP_EPS
        )

        idx0     = int(np.argmin(np.abs(ALPHA_SWEEP)))
        n0       = float(n_iters[idx0])
        n_theory = theoretical_n_iter(n0, gamma_hat, ALPHA_SWEEP)

        alpha_sweep_all[kappa]  = ALPHA_SWEEP
        n_iter_sweep_all[kappa] = n_iters
        gamma_hat_all[kappa]    = gamma_hat

        print_alpha_sweep_table(
            kappa, ALPHA_SWEEP_EPS, ALPHA_SWEEP,
            n_iters, res0s, gamma_hat, n_theory,
        )

        # ── Step 2: FNO strain sweep ───────────────────────────────────────
        print(f"\n  [2/2] FNO strain sweep: "
              f"{len(M_DEPTHS)} depths × {len(STRAIN_MAGS)} magnitudes …")
        sweep_results = run_fno_strain_sweep(
            C_field, C_field_t, alpha_m, alpha_p, alpha0
        )

        fno_strain_all[kappa] = {
            depth: {
                eps_mag: sweep_results[depth]['fno'][eps_mag]['n_iter']
                for eps_mag in STRAIN_MAGS
            }
            for depth in M_DEPTHS
        }
        alpha_eff_all[kappa] = {
            depth: sweep_results[depth]['alpha_eff']
            for depth in M_DEPTHS
        }

        print_fno_effective_alpha(kappa, sweep_results)

    # ── Sensitivity table ─────────────────────────────────────────────────────
    print_sensitivity_table(KAPPAS, [gamma_hat_all[k] for k in KAPPAS])

    # ── Figures ───────────────────────────────────────────────────────────────
    if HAS_MPL:
        print("\n  Generating figures …")
        plot_calibration_curves(
            KAPPAS,
            alpha_sweep_all, n_iter_sweep_all, gamma_hat_all,
            fno_strain_all,
            save_path=FIG_DIR / "alpha_calibration_curves.png",
        )
        plot_effective_alpha(
            KAPPAS,
            alpha_eff_all,
            save_path=FIG_DIR / "effective_alpha_vs_strain.png",
        )
    else:
        print("\n  matplotlib not available — skipping figures.")

    print(f"\n{'='*72}")
    print("  Calibration complete.  Key questions answered:")
    print()
    print("  Q1: Is the FNO deviation consistent with a scalar contraction perturbation?")
    print("      → Yes if the FNO horizontal lines intersect the calibration curve at")
    print("        the same α value predicted by the inversion formula.")
    print()
    print("  Q2: What is the sign/magnitude of α_eff vs strain magnitude?")
    print("      → α_eff < 0 at small ε̄ (undershoot, FNO faster)")
    print("      → α_eff > 0 at large ε̄ (overshoot from r_θ clipping, FNO slower)")
    print("      → Crossover near ε̄ ≈ 1e-2 for FNO11 at κ=96")
    print()
    print("  Q3: What should a KAN replacement achieve?")
    print("      → α_eff ≈ 0 (flat line) across all strain magnitudes at all κ.")
    print("      → Run this script again with the KAN-based τ_θ to verify.")
    print("=" * 72)


if __name__ == "__main__":
    main()
