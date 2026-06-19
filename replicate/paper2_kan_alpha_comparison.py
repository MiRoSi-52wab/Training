"""
KAN vs FFT vs FNO — α-perturbation calibration comparison at κ = 48.

Identical problem formulation to paper2_alpha_calibration.py, extended to
include KAN-exact alongside FFT, FNO7, and FNO11.

Two figures are produced:
  kan_alpha_curves.png        — N_iter vs α_pert for all four models overlaid
                                on the theoretical calibration curve.
  kan_alpha_eff_vs_strain.png — effective α_eff vs ε̄ magnitude for all models.

Background
----------
The LS iteration converges at rate γ = (α₊−α₋)/(α₊+α₋).  Scaling the
stiffness contrast by (1+α):

    C_pert = C⁰ + (1+α)·(C − C⁰)   →   τ_pert = (1+α)·(C−C⁰):ε

changes the effective contraction constant to (1+α)·γ, giving:

    N_iter(α) ≈ N₀ · log(γ̂) / log[(1+α)·γ̂]

A solver whose τ_θ introduces an implicit bias α_eff (model error) solves an
effectively perturbed problem, shifting its N_iter off the theoretical curve.

Expected outcomes
-----------------
  FFT       : always follows the theoretical curve exactly (it IS the reference).
  KAN-exact : tracks the curve closely — B-spline control points [1,−1,1] give
              φ(x)=x² exactly, so α_eff ≈ 0 at all strain magnitudes.
  FNO7/11   : displaced from the curve by their Yarotsky approximation error.
              α_eff < 0 at small ε̄ (undershoot, fewer iterations than FFT);
              α_eff > 0 at large ε̄ (overshoot from r_θ clipping, more iterations).

Usage (from project root):
    python replicate/paper2_kan_alpha_comparison.py

Expected runtime: ~25–50 min (32³ grid, κ=48, 4 models × 17 total sweeps).
For a quick smoke test set ALPHA_SWEEP to 3–5 values and STRAIN_MAGS to [1e-3].
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

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ─────────────────────────────────────────────────────────────────────────────
# Parameters  (identical to paper2_alpha_calibration.py)
# ─────────────────────────────────────────────────────────────────────────────

N             = 32
DIM           = 3
N_COMP        = 6
E_MATRIX      = 3.0
NU_MATRIX     = 0.3
NU_INCLUSION  = 0.22
SPHERE_RADIUS = 10.0
TOL           = 1e-3
MAX_ITER      = 2000
DISC          = 'staggered'
CUTOFF_M      = 1.0
DEPTH_K       = 4
LOAD_CASE     = 0          # ε̄₁₁ — normal case (largest iteration count at high κ)

KAPPA           = 48
STRAIN_MAGS     = [1e-4, 1e-3, 1e-2, 5e-1]
ALPHA_SWEEP_EPS = 1e-3     # strain magnitude used for the α-sweep calibration

ALPHA_SWEEP = np.array([
    -0.05, -0.02, -0.01, -0.005, -0.002, -0.001,
     0.0,
     0.001,  0.002,  0.005,  0.01,  0.02,  0.05,
])

FIG_DIR = Path(__file__).parent / "figures"

MODEL_NAMES = ['FFT', 'KAN-exact', 'FNO7', 'FNO11']
MODEL_STYLE: Dict = {
    'FFT':       {'color': '#2E86AB', 'marker': 'o', 'ls': '-',  'lw': 2.0, 'ms': 7},
    'KAN-exact': {'color': '#28A745', 'marker': '^', 'ls': '-',  'lw': 2.0, 'ms': 8},
    'FNO7':      {'color': '#E07B39', 'marker': 's', 'ls': '--', 'lw': 1.6, 'ms': 7},
    'FNO11':     {'color': '#D62246', 'marker': 'D', 'ls': '-.', 'lw': 1.6, 'ms': 7},
}


# ─────────────────────────────────────────────────────────────────────────────
# Microstructure and model construction
# ─────────────────────────────────────────────────────────────────────────────

def centered_sphere(N: int, r: float) -> np.ndarray:
    c = (N - 1) / 2.0
    xs, ys, zs = np.mgrid[0:N, 0:N, 0:N]
    return ((xs - c)**2 + (ys - c)**2 + (zs - c)**2) <= r**2


def make_stiffness(kappa: float):
    """Returns (C_field_f64, alpha_m, alpha_p, alpha0, gamma_th)."""
    phase    = centered_sphere(N, SPHERE_RADIUS)
    C_mat    = isotropic_stiffness_voigt_3d(E_MATRIX, NU_MATRIX)
    C_inc    = isotropic_stiffness_voigt_3d(E_MATRIX * kappa, NU_INCLUSION)
    C_field  = build_C_field(phase, C_mat, C_inc)
    alpha_m, alpha_p = compute_alpha_bounds(
        E_MATRIX, NU_MATRIX, NU_INCLUSION, kappa, dim=DIM
    )
    alpha0   = (alpha_m + alpha_p) / 2.0
    gamma_th = (alpha_p - alpha_m) / (alpha_p + alpha_m)
    return C_field, alpha_m, alpha_p, alpha0, gamma_th


def make_models(alpha_m: float, alpha_p: float) -> Dict[str, LSFNO]:
    """Create KAN-exact, FNO7, FNO11 with identical α bounds for fair comparison."""
    common = dict(
        grid_size=N, depth_K=DEPTH_K,
        alpha_minus=alpha_m, alpha_plus=alpha_p,
        tol=TOL, max_iter=MAX_ITER,
        dim=DIM, discretization=DISC,
    )
    return {
        'KAN-exact': LSFNO(**common, tau_theta=KANTauTheta(
            R=CUTOFF_M, shared=True, trainable=False, n_comp=N_COMP)),
        'FNO7':  LSFNO(**common, tau_theta=YarotskyTauTheta(
            depth_m=7,  cutoff_M=CUTOFF_M)),
        'FNO11': LSFNO(**common, tau_theta=YarotskyTauTheta(
            depth_m=11, cutoff_M=CUTOFF_M)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Perturbed stiffness field
# ─────────────────────────────────────────────────────────────────────────────

def make_perturbed_C_field(C_field: np.ndarray, alpha0: float,
                            alpha_pert: float) -> np.ndarray:
    """
    C_pert = C⁰ + (1+α)·(C − C⁰)

    Keeps the reference medium C⁰ = α₀·diag([1,1,1,½,½,½]) unchanged so
    the Green operator Γ(α₀) stays the same.  Only the contrast is scaled:
        τ_pert = (C_pert − C⁰):ε = (1+α)·(C − C⁰):ε = (1+α)·τ
    """
    n   = C_field.shape[0]
    dim = C_field.ndim - 2
    C0_diag         = np.ones(n, dtype=np.float64)
    C0_diag[DIM:]   = 0.5
    C0              = alpha0 * np.diag(C0_diag)
    C0_bc           = C0.reshape(n, n, *([1] * dim))
    return C0_bc + (1.0 + alpha_pert) * (C_field.astype(np.float64) - C0_bc)


# ─────────────────────────────────────────────────────────────────────────────
# Theoretical formulas
# ─────────────────────────────────────────────────────────────────────────────

def gamma_empirical(n_iter: int, res0: float) -> float:
    """γ̂ = (tol/res₀)^(1/N) — empirical per-iteration reduction rate."""
    return float((TOL / max(res0, 1e-30)) ** (1.0 / max(n_iter, 1)))


def theoretical_n_iter(n0: float, gamma: float,
                       alpha_values: np.ndarray) -> np.ndarray:
    """N(α) ≈ N₀·log(γ̂)/log[(1+α)·γ̂].  Returns NaN where divergent."""
    inner  = (1.0 + alpha_values) * gamma
    result = np.full_like(alpha_values, np.nan, dtype=float)
    valid  = inner < 1.0
    result[valid] = n0 * np.log(gamma) / np.log(inner[valid])
    return result


def invert_alpha(n_model: int, n_fft: int, gamma: float) -> float:
    """α_eff = γ̂^(N_FFT/N_model − 1) − 1."""
    if n_model <= 0:
        return float('nan')
    return float(gamma ** (n_fft / n_model - 1.0) - 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Sweep routines
# ─────────────────────────────────────────────────────────────────────────────

def run_alpha_sweep(
    C_field: np.ndarray,
    alpha0: float,
    models: Dict[str, LSFNO],
    alpha_values: np.ndarray,
    eps_mag: float,
) -> Tuple[Dict[str, List[int]], float]:
    """
    Run all four models on the perturbed C_field for every α value.

    For each α:
      C_pert = C⁰ + (1+α)·(C−C⁰)
      All models (FFT, KAN, FNO7, FNO11) solve with this perturbed field.

    FFT uses the same α₀ (unchanged Green operator), so its N_iter follows
    the theoretical calibration curve.  KAN-exact should track it closely;
    FNO7/11 deviate by their respective α_eff.

    Returns:
      n_iters_all : {model_name: [n_iter per alpha_value]}
      gamma_hat   : estimated from the FFT run at the α-value closest to 0
    """
    eps_np = np.zeros(N_COMP); eps_np[LOAD_CASE] = eps_mag
    eps_t  = torch.zeros(1, N_COMP); eps_t[0, LOAD_CASE] = float(eps_mag)

    n_iters: Dict[str, List[int]] = {name: [] for name in MODEL_NAMES}
    res0_fft: List[float] = []

    for i, alpha in enumerate(alpha_values):
        print(f"    α = {alpha:+.4f}  ({i+1}/{len(alpha_values)}) …", end=" ", flush=True)

        C_pert   = make_perturbed_C_field(C_field, alpha0, alpha)
        C_pert_t = torch.from_numpy(C_pert).float().unsqueeze(0)  # (1,6,6,N,N,N)

        # FFT reference
        r_fft = fft_solve(C_pert, eps_np, alpha0=alpha0,
                          tol=TOL, max_iter=MAX_ITER, discretization=DISC)
        n_iters['FFT'].append(r_fft['n_iter'])
        res0_fft.append(r_fft['residuals'][0])

        # Learned models (KAN-exact, FNO7, FNO11)
        for name, model in models.items():
            r = model.solve(C_pert_t, eps_t)
            n_iters[name].append(r['n_iter'])

        fft_n = n_iters['FFT'][-1]
        kan_n = n_iters['KAN-exact'][-1]
        f7_n  = n_iters['FNO7'][-1]
        f11_n = n_iters['FNO11'][-1]
        print(f"FFT={fft_n}  KAN={kan_n}  FNO7={f7_n}  FNO11={f11_n}")

    idx0      = int(np.argmin(np.abs(alpha_values)))
    gamma_hat = gamma_empirical(n_iters['FFT'][idx0], res0_fft[idx0])
    return n_iters, gamma_hat


def run_strain_sweep(
    C_field: np.ndarray,
    alpha0: float,
    models: Dict[str, LSFNO],
    strain_mags: List[float],
) -> Dict[str, Dict[float, int]]:
    """
    Run all models at each strain magnitude on the unperturbed C_field.

    For a linear LS problem the convergence criterion is relative
    (‖τ_k−τ_{k-1}‖/‖τ_k‖ < tol), so FFT and KAN-exact iteration counts
    are scale-invariant in ε̄.  FNO7/11 drift because the Yarotsky absolute
    approximation error O(4^{-m}) has a strain-scale-dependent relative effect.

    Returns {model_name: {eps_mag: n_iter}}.
    """
    C_batch = torch.from_numpy(C_field).float().unsqueeze(0)
    n_iters: Dict[str, Dict] = {name: {} for name in MODEL_NAMES}

    for eps_mag in strain_mags:
        print(f"    ε̄ = {eps_mag:.0e} …", end=" ", flush=True)

        eps_np = np.zeros(N_COMP); eps_np[LOAD_CASE] = eps_mag
        eps_t  = torch.zeros(1, N_COMP); eps_t[0, LOAD_CASE] = float(eps_mag)

        r_fft = fft_solve(C_field, eps_np, alpha0=alpha0,
                          tol=TOL, max_iter=MAX_ITER, discretization=DISC)
        n_iters['FFT'][eps_mag] = r_fft['n_iter']

        for name, model in models.items():
            r = model.solve(C_batch, eps_t)
            n_iters[name][eps_mag] = r['n_iter']

        fft_n = n_iters['FFT'][eps_mag]
        kan_n = n_iters['KAN-exact'][eps_mag]
        f7_n  = n_iters['FNO7'][eps_mag]
        f11_n = n_iters['FNO11'][eps_mag]
        print(f"FFT={fft_n}  KAN={kan_n}  FNO7={f7_n}  FNO11={f11_n}")

    return n_iters


# ─────────────────────────────────────────────────────────────────────────────
# Console output
# ─────────────────────────────────────────────────────────────────────────────

def print_alpha_sweep_table(
    alpha_values: np.ndarray,
    n_iters_all: Dict[str, List[int]],
    gamma_hat: float,
    n_theory: np.ndarray,
) -> None:
    print(f"\n  α-sweep results  (κ={KAPPA}, load case {LOAD_CASE}, ε̄={ALPHA_SWEEP_EPS:.0e})")
    print(f"  γ̂ = {gamma_hat:.6f}   (empirical from α=0 FFT run)")
    w = 11
    header = f"  {'α':>8}  {'N_theory':>{w}}"
    for name in MODEL_NAMES:
        header += f"  {name:>{w}}"
    print(header)
    print(f"  {'─'*70}")
    for i, a in enumerate(alpha_values):
        nt_s = f"{n_theory[i]:>{w}.1f}" if not np.isnan(n_theory[i]) else f"{'n/a':>{w}}"
        row  = f"  {a:>8.4f}  {nt_s}"
        for name in MODEL_NAMES:
            row += f"  {n_iters_all[name][i]:>{w}}"
        row += "   ← α=0" if abs(a) < 1e-10 else ""
        print(row)


def print_strain_sweep_table(
    n_iters_all: Dict[str, Dict[float, int]],
    gamma_hat: float,
) -> None:
    print(f"\n  Strain sweep  (κ={KAPPA}, load case {LOAD_CASE})")
    print(f"  α_eff = γ̂^(N_FFT/N_model − 1) − 1   γ̂ = {gamma_hat:.6f}")
    print()
    # Header
    print(f"  {'ε̄':>10}  {'FFT':>6}", end="")
    for name in ['KAN-exact', 'FNO7', 'FNO11']:
        print(f"  {name:>10}  {'Δ':>5}  {'α_eff':>10}", end="")
    print()
    print(f"  {'─'*86}")
    for eps_mag in STRAIN_MAGS:
        n_fft = n_iters_all['FFT'][eps_mag]
        row   = f"  {eps_mag:>10.0e}  {n_fft:>6}"
        for name in ['KAN-exact', 'FNO7', 'FNO11']:
            n_m   = n_iters_all[name][eps_mag]
            ae    = invert_alpha(n_m, n_fft, gamma_hat)
            delta = n_fft - n_m
            ae_s  = f"{ae:>+10.3e}" if not np.isnan(ae) else f"{'nan':>10}"
            bias  = "faster" if delta > 0 else ("slower" if delta < 0 else "  tied")
            row  += f"  {n_m:>10}  {delta:>+5}  {ae_s}  ← {bias}"
        print(row)
    print()
    print("  Δ = N_FFT − N_model:  Δ>0 means model is faster, Δ<0 means slower.")
    print("  KAN-exact should have Δ=0 and α_eff≈0 at all strains.")


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def plot_n_iter_vs_alpha(
    alpha_values: np.ndarray,
    n_iters_all: Dict[str, List[int]],
    gamma_hat: float,
    save_path: Path,
) -> None:
    """
    N_iter vs α_pert for all four models on the same axes.

    The theoretical calibration curve (gray dashed) is the expected N_iter
    for a solver with α_eff = 0.  FFT dots lie exactly on it.  KAN-exact
    should track it closely.  FNO7/11 are displaced by their α_eff bias.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Theoretical reference curve
    idx0       = int(np.argmin(np.abs(alpha_values)))
    n0         = float(n_iters_all['FFT'][idx0])
    alpha_fine = np.linspace(alpha_values.min() * 1.15, alpha_values.max() * 1.15, 500)
    n_th_fine  = theoretical_n_iter(n0, gamma_hat, alpha_fine)
    ax.plot(alpha_fine, n_th_fine,
            color='silver', lw=2.5, ls='--', zorder=1,
            label=f'Theory  N₀·log(γ̂)/log[(1+α)γ̂]   γ̂={gamma_hat:.4f}')
    ax.axvline(0, color='lightgray', lw=0.8, ls=':', zorder=1)

    # All four models
    for name in MODEL_NAMES:
        st = MODEL_STYLE[name]
        y  = np.array(n_iters_all[name], dtype=float)
        ax.plot(alpha_values, y,
                color=st['color'], marker=st['marker'],
                ls=st['ls'], lw=st['lw'], ms=st['ms'],
                label=name, zorder=4, alpha=0.92)

    ax.set_title(
        f'N_iter vs α_pert — perturbed stiffness contrast  (κ={KAPPA})\n'
        f'C_pert = C⁰ + (1+α)·(C−C⁰)     ε̄ = {ALPHA_SWEEP_EPS:.0e},  '
        f'load case {LOAD_CASE} (ε̄₁₁)',
        fontsize=10,
    )
    ax.set_xlabel('α_pert  (stiffness perturbation factor)', fontsize=10)
    ax.set_ylabel('N iterations to convergence', fontsize=10)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure 1 → {save_path.name}")


def plot_alpha_eff_vs_strain(
    n_iters_all: Dict[str, Dict[float, int]],
    gamma_hat: float,
    save_path: Path,
) -> None:
    """
    Effective α_eff vs strain magnitude for all four models.

    FFT is the reference (α_eff = 0 everywhere by definition — flat blue line).
    KAN-exact should also be flat at 0 (exact T:ε computation, no bias).
    FNO7/11 show negative α at small ε̄ (undershoot → fewer iterations than FFT)
    and positive α at large ε̄ (overshoot from r_θ clipping → more iterations).
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(0, color='silver', lw=1.5, ls='--', zorder=1, label='α = 0  (ideal)')

    for name in MODEL_NAMES:
        if name == 'FFT':
            ae_vals = [0.0] * len(STRAIN_MAGS)
        else:
            ae_vals = [
                invert_alpha(n_iters_all[name][e], n_iters_all['FFT'][e], gamma_hat)
                for e in STRAIN_MAGS
            ]
        st = MODEL_STYLE[name]
        ax.semilogx(STRAIN_MAGS, ae_vals,
                    color=st['color'], marker=st['marker'],
                    ls=st['ls'], lw=st['lw'], ms=st['ms'],
                    label=name, zorder=3, alpha=0.92)
        for e, ae in zip(STRAIN_MAGS, ae_vals):
            if not np.isnan(ae) and abs(ae) > 1e-12:
                ax.annotate(
                    f'{ae:+.1e}', (e, ae),
                    textcoords='offset points', xytext=(6, 4),
                    fontsize=7, color=st['color'],
                )

    ax.set_title(
        f'Effective α_eff vs strain magnitude  (κ={KAPPA}, load case {LOAD_CASE})\n'
        r'α_eff = γ̂^(N_FFT / N_model − 1) − 1',
        fontsize=10,
    )
    ax.set_xlabel('ε̄ magnitude', fontsize=10)
    ax.set_ylabel('α_eff', fontsize=10)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')

    note = ('α_eff < 0 : model converges faster than FFT  |  '
            'α_eff > 0 : model converges slower than FFT')
    fig.text(0.5, -0.03, note, ha='center', fontsize=8.5, color='dimgray')
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
    print("  KAN vs FFT vs FNO — α Calibration Comparison")
    print(f"  κ = {KAPPA}   Grid: {N}³   disc: {DISC}   tol: {TOL:.0e}")
    print(f"  Models: {', '.join(MODEL_NAMES)}")
    print(f"  α-sweep: {len(ALPHA_SWEEP)} values ∈ [{ALPHA_SWEEP[0]:.3f}, {ALPHA_SWEEP[-1]:.3f}]"
          f"  at ε̄ = {ALPHA_SWEEP_EPS:.0e}  (load case {LOAD_CASE}: ε̄₁₁)")
    print(f"  Strain sweep: ε̄ ∈ {STRAIN_MAGS}")
    print(f"  Figures → {FIG_DIR}/")
    print("=" * 72)

    C_field, alpha_m, alpha_p, alpha0, gamma_th = make_stiffness(KAPPA)
    print(f"\n  α₋ = {alpha_m:.4f}   α₊ = {alpha_p:.4f}   α₀ = {alpha0:.4f}")
    print(f"  γ_theoretical = {gamma_th:.6f}")

    models = make_models(alpha_m, alpha_p)

    # ── Step 1: α-perturbation sweep ──────────────────────────────────────────
    n_runs_1 = len(ALPHA_SWEEP) * len(MODEL_NAMES)
    print(f"\n  [1/2] α-sweep: {len(ALPHA_SWEEP)} α values × {len(MODEL_NAMES)} models"
          f" = {n_runs_1} runs …")

    n_iters_alpha, gamma_hat = run_alpha_sweep(
        C_field, alpha0, models, ALPHA_SWEEP, ALPHA_SWEEP_EPS
    )

    idx0     = int(np.argmin(np.abs(ALPHA_SWEEP)))
    n0       = float(n_iters_alpha['FFT'][idx0])
    n_theory = theoretical_n_iter(n0, gamma_hat, ALPHA_SWEEP)
    print_alpha_sweep_table(ALPHA_SWEEP, n_iters_alpha, gamma_hat, n_theory)

    # ── Step 2: strain-magnitude sweep ────────────────────────────────────────
    n_runs_2 = len(STRAIN_MAGS) * len(MODEL_NAMES)
    print(f"\n  [2/2] Strain sweep: {len(STRAIN_MAGS)} magnitudes × {len(MODEL_NAMES)} models"
          f" = {n_runs_2} runs …")

    n_iters_strain = run_strain_sweep(C_field, alpha0, models, STRAIN_MAGS)
    print_strain_sweep_table(n_iters_strain, gamma_hat)

    # ── Figures ───────────────────────────────────────────────────────────────
    if HAS_MPL:
        print("  Generating figures …")
        plot_n_iter_vs_alpha(
            ALPHA_SWEEP, n_iters_alpha, gamma_hat,
            FIG_DIR / "kan_alpha_curves.png",
        )
        plot_alpha_eff_vs_strain(
            n_iters_strain, gamma_hat,
            FIG_DIR / "kan_alpha_eff_vs_strain.png",
        )
    else:
        print("  matplotlib not available — skipping figures.")

    print(f"\n{'='*72}")
    print("  Expected results:")
    print()
    print("  N_iter(α) plot [kan_alpha_curves.png]:")
    print("    FFT       — dots lie exactly on the theoretical curve by construction.")
    print("    KAN-exact — should track the curve closely (α_eff ≈ 0).")
    print("    FNO7/11   — displaced upward or downward from the curve by α_eff;")
    print("                the horizontal offset at α=0 is their intrinsic bias.")
    print()
    print("  α_eff vs ε̄ plot [kan_alpha_eff_vs_strain.png]:")
    print("    FFT       — flat at 0 (reference by definition).")
    print("    KAN-exact — flat near 0 (exact T:ε, no approximation error).")
    print("    FNO7/11   — negative at small ε̄, positive at large ε̄;")
    print("                crossover strain moves left (smaller) for deeper m.")
    print("=" * 72)


if __name__ == "__main__":
    main()
