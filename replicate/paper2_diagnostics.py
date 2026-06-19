"""
Diagnostic suite for §5.2.1 — FFT vs LS-FNO iteration-count comparison.

Addresses the checks listed in LSFNO_Load_Case_and_Extended_Debug.md.
Run this script to rule in/out systematic biases after the two main fixes
(loop-structure off-by-one and Voigt → Frobenius norm in fft_solver.py).

Checks
------
D1  Per-load-case iteration counts (normal vs shear breakdown)
D2  Strain-magnitude sweep — iteration count must be invariant (LS is linear)
D3  Initial residuals — both solvers must start from the same physical state
D4  Green operator consistency — same random field → same strain output
D5  α₀ consistency — formula value vs model value
D6  DC mode exactly zero in both Green operators

Usage (from project root):
    python replicate/paper2_diagnostics.py

Expected runtime: ~10–20 min (32³ grid, κ ∈ {12, 48, 96}, full sweep).
For a quick smoke test change KAPPAS to [12] and STRAIN_MAGS to [1e-3, 1e-2].
"""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from generation.microstructure import isotropic_stiffness_voigt_3d, build_C_field
from generation.fft_solver import (
    solve as fft_solve,
    _build_green_operator_3d,
)
from models.ls_fno import LSFNO, YarotskyTauTheta
from utils.config_loader import compute_alpha_bounds

# ── Parameters (mirror paper2_section521.py) ──────────────────────────────────
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
EPS_PHYSICAL  = 1e-3

KAPPAS = [12]
STRAIN_MAGS = [1e-4, 1e-3, 1e-2, 5e-1]   # D2 sweep magnitudes

LOAD_NAMES = [
    'ε̄₁₁ (normal x)',
    'ε̄₂₂ (normal y)',
    'ε̄₃₃ (normal z)',
    'γ̄₂₃ (shear yz)',
    'γ̄₁₃ (shear xz)',
    'γ̄₁₂ (shear xy)',
]

SQRT2 = np.sqrt(2.0)


# ── Geometry / material helpers ───────────────────────────────────────────────

def centered_sphere(N: int, r: float) -> np.ndarray:
    c = (N - 1) / 2.0
    xs, ys, zs = np.mgrid[0:N, 0:N, 0:N]
    return ((xs - c)**2 + (ys - c)**2 + (zs - c)**2) <= r**2


def make_stiffness(kappa: float):
    """Return (phase, C_field_f64, C_field_t_f32, alpha_minus, alpha_plus, alpha0)."""
    phase   = centered_sphere(N, SPHERE_RADIUS)
    C_mat   = isotropic_stiffness_voigt_3d(E_MATRIX, NU_MATRIX)
    C_inc   = isotropic_stiffness_voigt_3d(E_MATRIX * kappa, NU_INCLUSION)
    C_field = build_C_field(phase, C_mat, C_inc)
    C_field_t = torch.from_numpy(C_field).float()
    alpha_m, alpha_p = compute_alpha_bounds(
        E_MATRIX, NU_MATRIX, NU_INCLUSION, kappa, dim=DIM
    )
    alpha0 = (alpha_m + alpha_p) / 2.0
    return phase, C_field, C_field_t, alpha_m, alpha_p, alpha0


def make_model(alpha_m: float, alpha_p: float, depth_m: int = 7) -> LSFNO:
    return LSFNO(
        grid_size      = N,
        depth_K        = DEPTH_K,
        alpha_minus    = alpha_m,
        alpha_plus     = alpha_p,
        tol            = TOL,
        max_iter       = MAX_ITER,
        tau_theta      = YarotskyTauTheta(depth_m=depth_m, cutoff_M=CUTOFF_M),
        dim            = DIM,
        discretization = DISC,
    )


def run_pair(C_field, C_batch, model, alpha0, a: int, eps_mag: float = EPS_PHYSICAL):
    """Run one load case through both solvers and return result dicts."""
    eps_np    = np.zeros(N_COMP); eps_np[a] = eps_mag
    eps_t     = torch.zeros(1, N_COMP); eps_t[0, a] = eps_mag
    r_fft = fft_solve(C_field, eps_np, alpha0=alpha0,
                      tol=TOL, max_iter=MAX_ITER, discretization=DISC)
    r_fno = model.solve(C_batch, eps_t)
    return r_fft, r_fno


# ── D4 — Green operator consistency ─────────────────────────────────────────

def diag_d4_green(kappa: float = 12):
    """
    Apply both Green operators to the same random Voigt stress field and compare.

    FFT Γ (float64, includes 1/α₀, outputs Voigt strain) should match
    FNO Γ (float32, no 1/α₀, outputs Mandel strain) after appropriate rescaling.

    Conversion pipeline:
      Voigt stress τ_V
        → Mandel stress τ_M = τ_V * [1,1,1, √2,√2,√2]   (shear rows ×√2)
        → Mandel xi_M = τ_M / α₀
        → FNO apply_green → Mandel strain ε_M
        → Voigt engineering strain ε_V_fno = ε_M * [1,1,1, √2,√2,√2]   (shear ×√2)

    FFT:
      τ_V → +Γ_fft·τ_V → Voigt strain ε_V_fft  (without the solver's minus sign)

    These two should agree within float32 precision (~1e-5 relative).
    """
    _, _, _, alpha_m, alpha_p, alpha0 = make_stiffness(kappa)
    model = make_model(alpha_m, alpha_p)
    Gamma_fft = _build_green_operator_3d(N, alpha0, DISC)   # f64, includes 1/α₀

    rng   = np.random.default_rng(42)
    tau_V = rng.standard_normal((N_COMP, N, N, N))          # random Voigt stress

    # FFT side (float64, positive Γ)
    fft_ax   = (-3, -2, -1)
    tau_hat  = np.fft.fftn(tau_V, axes=fft_ax)
    eps_hat  = np.einsum('abxyz,bxyz->axyz', Gamma_fft, tau_hat)   # + sign
    eps_V_fft = np.real(np.fft.ifftn(eps_hat, axes=fft_ax))

    # FNO side (float32 model): τ_V → ξ_M → apply_green → ε_V
    tau_M  = tau_V.copy(); tau_M[DIM:] *= SQRT2             # Voigt → Mandel stress
    xi_M   = (tau_M / alpha0).astype(np.float32)
    xi_M_t = torch.from_numpy(xi_M).unsqueeze(0)           # (1, 6, N, N, N)
    eps_M_fno  = model._apply_green(xi_M_t)[0].numpy()     # Mandel strain
    eps_V_fno  = eps_M_fno.copy(); eps_V_fno[DIM:] *= SQRT2  # Mandel → Voigt

    diff    = eps_V_fft - eps_V_fno.astype(np.float64)
    ref     = float(np.linalg.norm(eps_V_fft))
    rel_err = float(np.linalg.norm(diff)) / max(ref, 1e-30)

    # Float32 vs float64 gives ~1e-7 per entry; accumulated over 32³×6 ≈ 2e5 entries
    # expect relative error ~1e-5 from precision alone.
    ok = rel_err < 5e-4
    print(f"\n  D4  Green operator consistency  (κ={kappa})")
    print(f"      ‖Γ_FFT·τ − Γ_FNO·ξ‖/‖Γ_FFT·τ‖ = {rel_err:.3e}  "
          f"{'✓ consistent (float32 precision expected)' if ok else '✗ MISMATCH'}")
    if not ok:
        # Component-wise breakdown to help locate mismatch
        for c in range(N_COMP):
            ce = float(np.linalg.norm(diff[c])) / max(float(np.linalg.norm(eps_V_fft[c])), 1e-30)
            print(f"      component {c}: rel_err = {ce:.3e}")


# ── D6 — DC mode exactly zero ────────────────────────────────────────────────

def diag_d6_dc(kappa: float = 12):
    """Γ̂(0) must be identically zero in both operators (not just small)."""
    _, _, _, alpha_m, alpha_p, alpha0 = make_stiffness(kappa)
    model     = make_model(alpha_m, alpha_p)
    Gamma_fft = _build_green_operator_3d(N, alpha0, DISC)

    dc_fft = float(np.abs(Gamma_fft[:, :, 0, 0, 0]).max())
    dc_fno = float(model.gamma_hat_M[:, :, 0, 0, 0].abs().max().item())

    print(f"\n  D6  DC mode Γ̂(0) == 0  (κ={kappa})")
    print(f"      FFT Γ̂(0) max = {dc_fft:.3e}  "
          f"{'✓ exactly 0.0' if dc_fft == 0.0 else '✗ NONZERO — leaks mean strain'}")
    print(f"      FNO Γ̂(0) max = {dc_fno:.3e}  "
          f"{'✓ exactly 0.0' if dc_fno == 0.0 else '✗ NONZERO — leaks mean strain'}")


# ── D5 — α₀ consistency ──────────────────────────────────────────────────────

def diag_d5_alpha0(kappa: float):
    """Print α₀ from formula and from model; warn if the C11-heuristic differs."""
    _, C_field, _, alpha_m, alpha_p, alpha0 = make_stiffness(kappa)
    model = make_model(alpha_m, alpha_p)

    C00 = C_field[0, 0]
    alpha0_c11 = (float(C00.max()) + float(C00.min())) / 2.0
    gamma = (alpha_p - alpha_m) / (alpha_p + alpha_m)

    print(f"\n  D5  α₀ consistency  (κ={kappa})")
    print(f"      α₋ = {alpha_m:.6f} GPa   (= 2μ_mat, min tensor eigenvalue of C)")
    print(f"      α₊ = {alpha_p:.6f} GPa   (= 3λ+2μ of inclusion)")
    print(f"      α₀ formula  = {alpha0:.6f} GPa  {'✓' if abs(alpha0 - model.alpha_0) < 1e-8 else '✗ MISMATCH'}")
    print(f"      α₀ model    = {model.alpha_0:.6f} GPa")
    print(f"      α₀ C11-heur = {alpha0_c11:.6f} GPa  "
          f"{'(matches formula ✓)' if abs(alpha0 - alpha0_c11) < 0.01 * alpha0 else '(differs from formula — FFT would use wrong α₀ without explicit arg)'}")
    print(f"      γ = {gamma:.6f}   (contraction constant; closer to 1 → slower convergence)")


# ── D1 + D3 — Per-load-case iterations and initial residuals ─────────────────

def diag_d1_d3_per_case(kappa: float):
    """
    D1: iteration counts per load case (normal vs shear breakdown).
    D3: first residual of both solvers — should match within ~1% for FNO11.

    If both solvers show the same normal/shear split (e.g. 220 vs 194 at κ=48),
    the difference is physical, not a bug.  If FNO consistently finishes earlier
    than FFT *within* the same load-case type, a systematic bias remains.
    """
    _, C_field, C_field_t, alpha_m, alpha_p, alpha0 = make_stiffness(kappa)
    C_batch = C_field_t.unsqueeze(0)
    model   = make_model(alpha_m, alpha_p)

    print(f"\n  D1 / D3  Per-load-case  (κ={kappa})")
    hdr = f"  {'#':>2}  {'Load case':<22}  {'FFT':>6}  {'FNO11':>6}  {'Δ':>5}  {'res₀_FFT':>10}  {'res₀_FNO':>10}  {'ratio':>6}"
    print(hdr)
    print(f"  {'-'*len(hdr.rstrip())}")

    fft_iters, fno_iters = [], []
    ratio_ok = []
    for a in range(N_COMP):
        rf, rn = run_pair(C_field, C_batch, model, alpha0, a)
        fi, ni = rf['n_iter'], rn['n_iter']
        r0f, r0n = rf['residuals'][0], rn['residuals'][0]
        ratio = r0f / r0n if r0n > 1e-30 else float('nan')
        ratio_ok.append(abs(ratio - 1.0) < 0.10)
        mismatch = " !" if abs(ratio - 1.0) > 0.10 else ""
        faster   = " ←FNO faster" if ni < fi else (" ←FNO slower" if ni > fi else "")
        print(f"  {a:>2}  {LOAD_NAMES[a]:<22}  {fi:>6}  {ni:>6}  {fi-ni:>5}"
              f"  {r0f:>10.4e}  {r0n:>10.4e}  {ratio:>6.3f}{mismatch}{faster}")
        fft_iters.append(fi)
        fno_iters.append(ni)

    print()
    print(f"  {'Average':>30}  {np.mean(fft_iters):>6.1f}  {np.mean(fno_iters):>6.1f}  "
          f"{np.mean(fft_iters) - np.mean(fno_iters):>5.1f}")
    print(f"  {'Normal avg (0–2)':>30}  {np.mean(fft_iters[:3]):>6.1f}  {np.mean(fno_iters[:3]):>6.1f}  "
          f"{np.mean(fft_iters[:3]) - np.mean(fno_iters[:3]):>5.1f}")
    print(f"  {'Shear  avg (3–5)':>30}  {np.mean(fft_iters[3:]):>6.1f}  {np.mean(fno_iters[3:]):>6.1f}  "
          f"{np.mean(fft_iters[3:]) - np.mean(fno_iters[3:]):>5.1f}")

    print(f"\n  D3  Initial-residual ratios within 10%: "
          f"{'✓ initial states agree' if all(ratio_ok) else '✗ initial-state mismatch — check τ₀/α₀ vs ξ₀'}")


# ── D2 — Strain-magnitude sweep ───────────────────────────────────────────────

def diag_d2_strain_sweep(kappa: float):
    """
    Run load case 0 (ε̄₁₁) at multiple strain magnitudes.

    Expected: iteration count is CONSTANT (±1–2) regardless of magnitude.
    If it drifts: a nonlinearity in τ_θ or T construction remains.

    Also checks FNO vs FFT difference at each magnitude.
    """
    _, C_field, C_field_t, alpha_m, alpha_p, alpha0 = make_stiffness(kappa)
    C_batch = C_field_t.unsqueeze(0)
    model   = make_model(alpha_m, alpha_p)

    print(f"\n  D2  Strain-magnitude sweep  (κ={kappa}, load case 0 = ε̄₁₁)")
    print(f"  {'ε̄ mag':>10}  {'FFT':>6}  {'FNO11':>6}  {'Δ':>5}  {'converged?':>14}")
    print(f"  {'-'*52}")

    ref_fft, ref_fno = None, None
    for mag in STRAIN_MAGS:
        rf, rn = run_pair(C_field, C_batch, model, alpha0, a=0, eps_mag=mag)
        fi, ni = rf['n_iter'], rn['n_iter']
        if ref_fft is None:
            ref_fft, ref_fno = fi, ni
        drift_fft = fi - ref_fft
        drift_fno = ni - ref_fno
        conv = ("FFT✓ FNO✓" if rf['converged'] and rn['converged'] else
                ("FFT✓ FNO✗" if rf['converged'] else "FFT✗ FNO✓"))
        drift_str = ""
        if abs(drift_fft) > 2 or abs(drift_fno) > 2:
            drift_str = f"  ← DRIFT fft:{drift_fft:+d} fno:{drift_fno:+d}"
        print(f"  {mag:>10.0e}  {fi:>6}  {ni:>6}  {fi-ni:>5}  {conv:>14}{drift_str}")

    print(f"\n  Iteration count should be constant (±1–2) across all magnitudes.")
    print(f"  Large drift indicates a remaining nonlinearity in τ_θ or T.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.set_grad_enabled(False)

    print("=" * 72)
    print("  LS-FNO vs FFT Diagnostic Suite")
    print("  Ref: LSFNO_Load_Case_and_Extended_Debug.md")
    print(f"  Grid: {N}³   disc: {DISC}   tol: {TOL:.0e}   max_iter: {MAX_ITER}")
    print("=" * 72)

    # κ-independent checks
    diag_d4_green(kappa=12)
    diag_d6_dc(kappa=12)

    # Per-κ checks
    for kappa in KAPPAS:
        sep = f"\n{'─' * 72}\n  κ = {kappa}\n{'─' * 72}"
        print(sep)
        diag_d5_alpha0(kappa)
        diag_d1_d3_per_case(kappa)
        diag_d2_strain_sweep(kappa)

    print(f"\n{'=' * 72}")
    print("  Diagnostic suite complete.")
    print("  Key results to look for:")
    print("   • D1 Δ column:  0 for all load cases → both fixes fully resolved the bias")
    print("   • D1 normal vs shear split: should be present in BOTH solvers (physical)")
    print("   • D2 iteration count constant across magnitudes → no residual nonlinearity")
    print("   • D3 ratio ≈ 1.000 → initial states identical (FNO11 τ_θ accurate)")
    print("   • D4 rel_err < 1e-4 → Green operators physically consistent")
    print("   • D6 exactly 0.0 → no DC leakage")
    print("=" * 72)


if __name__ == "__main__":
    main()
