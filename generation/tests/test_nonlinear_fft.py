"""
Smoke-tests for the nonlinear FFT solver.

Run from the project root:
    python -m misc.test_nonlinear_fft

Tests
-----
1. Below-yield elastic: nonlinear solver must match linear solver exactly.
2. Homogeneous perfect-plasticity: analytical σ_eq check.
3. Linear hardening composite (2D): convergence and plastic strain accumulation.
4. 3D sanity: runs without error, plastic voxels appear.
"""

import sys
import numpy as np
sys.path.insert(0, '.')

from generation.microstructure import (
    isotropic_stiffness_voigt,
    isotropic_stiffness_voigt_3d,
    build_C_field,
    random_disks,
    random_spheres,
)
from generation.fft_solver import solve as solve_linear
from generation.nonlinear_fft_solver import solve_nonlinear

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def check(name, cond, detail=""):
    tag = PASS if cond else FAIL
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    return cond


# ---------------------------------------------------------------------------
# Test 1 — Below-yield: nonlinear == linear
# ---------------------------------------------------------------------------
def test_elastic_regime():
    """When σ_0 is huge, no voxel ever yields.  Both solvers must agree."""
    print("\n[1] Below-yield elastic equivalence (2D)")
    N = 16
    rng = np.random.default_rng(0)
    phase = random_disks(N, n_disks=5, r_min=2, r_max=4, rng=rng)
    C_mat = isotropic_stiffness_voigt(1.0,  0.3)
    C_inc = isotropic_stiffness_voigt(10.0, 0.3)
    C_field = build_C_field(phase, C_mat, C_inc).astype(np.float64)

    eps_bar = np.array([0.005, 0.0, 0.0])

    # Linear solver reference
    lin = solve_linear(C_field, eps_bar)

    # Nonlinear solver with σ_0 >> max possible elastic stress
    sigma_0_huge = 1e9
    nl = solve_nonlinear(
        C_field, phase,
        eps_bar_path=eps_bar[None, :],   # single load step
        sigma_0=sigma_0_huge,
        H=0.0,
        tol=1e-6,
        max_iter=500,
    )

    eps_match   = np.allclose(nl['eps_star'],   lin['eps_star'],   atol=1e-8)
    sigma_match = np.allclose(nl['sigma_star'], lin['sigma_star'], atol=1e-8)
    p_zero      = float(nl['p_star'].max()) < 1e-15

    ok = check("eps fields match",    eps_match)
    ok = check("sigma fields match",  sigma_match) and ok
    ok = check("no plastic strain",   p_zero,
               f"max p = {nl['p_star'].max():.2e}") and ok
    return ok


# ---------------------------------------------------------------------------
# Test 2 — Homogeneous perfect plasticity: analytical σ_eq
# ---------------------------------------------------------------------------
def test_homogeneous_perfect_plasticity():
    """
    Homogeneous (no inclusions), all matrix, perfect plasticity (H=0).

    Under uniaxial macroscopic strain ε₁₁ = ε_bar (in-plane, ε₂₂=γ₁₂=0),
    for isotropic plane-strain:
        σ₁₁_trial = (λ+2μ)·ε_bar,  σ₂₂ = σ₃₃ = λ·ε_bar
        tr = (3λ+2μ)·ε_bar
        s₁₁ = 4μ/3·ε_bar,  s₂₂ = s₃₃ = −2μ/3·ε_bar
        σ_eq = 2μ·|ε_bar|

    Yield at σ_eq = σ_0  →  ε_yield = σ_0 / (2μ).
    After yielding, the deviatoric stress is radially projected back to the
    yield surface: σ_eq converges to σ_0 (perfect plasticity).
    """
    print("\n[2] Homogeneous perfect plasticity — σ_eq check (2D)")
    N  = 8   # small grid, uniform material → result is independent of N
    E, nu = 200.0, 0.3
    lam = E * nu / ((1+nu)*(1-2*nu))
    mu  = E / (2*(1+nu))

    C_mat   = isotropic_stiffness_voigt(E, nu)
    C_field = np.tile(C_mat[:, :, None, None], (1, 1, N, N)).astype(np.float64)
    phase   = np.zeros((N, N), dtype=bool)   # all matrix

    sigma_0 = 100.0   # MPa
    H       = 0.0

    eps_yield = sigma_0 / (2.0 * mu)
    eps_bar_max = 3.0 * eps_yield          # well past yield

    n_steps = 20
    eps_path = np.zeros((n_steps, 3), dtype=np.float64)
    eps_path[:, 0] = np.linspace(0.0, eps_bar_max, n_steps)

    result = solve_nonlinear(
        C_field, phase, eps_path, sigma_0=sigma_0, H=H,
        tol=1e-6, max_iter=2000,
    )

    all_conv = all(result['converged_history'])
    check("all steps converged", all_conv,
          f"{sum(result['converged_history'])}/{n_steps}")

    # At the final step (well past yield), σ_eq should equal σ_0
    sigma_v = result['sigma_history'][-1]   # (4, N, N)
    s11 = sigma_v[0]; s22 = sigma_v[1]; s33 = sigma_v[2]; s12 = sigma_v[3]
    tr   = s11 + s22 + s33
    ph   = tr / 3.0
    d11  = s11 - ph;  d22 = s22 - ph;  d33 = s33 - ph
    seq  = float(np.mean(np.sqrt(1.5*(d11**2 + d22**2 + d33**2 + 2*s12**2))))

    ok = check(f"final σ_eq ≈ σ_0 = {sigma_0}",
               abs(seq - sigma_0) / sigma_0 < 1e-3,
               f"σ_eq = {seq:.4f}")

    # Plastic strain should be positive everywhere
    p_pos = float(result['p_star'].min()) > 0.0
    ok = check("plastic strain positive", p_pos,
               f"min p = {result['p_star'].min():.4f}") and ok
    return ok


# ---------------------------------------------------------------------------
# Test 3 — Linear hardening composite 2D
# ---------------------------------------------------------------------------
def test_composite_2d_hardening():
    """
    Two-phase 2D composite: elastic fibers in an elasto-plastic matrix.
    Multi-step uniaxial loading.  Checks:
      - All steps converge.
      - Plastic strain is zero in fiber voxels.
      - Plastic strain accumulates in matrix voxels.
      - Macroscopic stress increases monotonically (hardening).
    """
    print("\n[3] Composite 2D — linear hardening, multi-step")
    N   = 32
    rng = np.random.default_rng(7)
    phase = random_disks(N, n_disks=8, r_min=2, r_max=5, rng=rng)

    E_m, nu_m = 68900.0, 0.35
    E_f, nu_f = 400000.0, 0.23
    sigma_0   = 68.9
    H         = 1710.0

    C_mat   = isotropic_stiffness_voigt(E_m, nu_m)
    C_inc   = isotropic_stiffness_voigt(E_f, nu_f)
    C_field = build_C_field(phase, C_mat, C_inc).astype(np.float64)

    n_steps = 10
    eps_path = np.zeros((n_steps, 3))
    eps_path[:, 0] = np.linspace(0.0, 0.01, n_steps)

    result = solve_nonlinear(
        C_field, phase, eps_path,
        sigma_0=sigma_0, H=H,
        tol=1e-4, max_iter=1000,
        verbose=False,
    )

    all_conv = all(result['converged_history'])
    check("all steps converged", all_conv)

    # Plastic strain in fiber voxels must be zero
    p_fiber  = result['p_star'][phase]
    p_matrix = result['p_star'][~phase]
    check("no plastic strain in fibers",  float(p_fiber.max()) < 1e-12,
          f"max p_fiber = {p_fiber.max():.2e}")
    check("plastic strain in matrix",     float(p_matrix.max()) > 0.0,
          f"max p_matrix = {p_matrix.max():.4f}")

    # Macroscopic σ₁₁ must be monotonically increasing (hardening material)
    macro_s11 = [s[0] for s in result['macro_stress_history']]
    monotone  = all(macro_s11[i] <= macro_s11[i+1] + 1e-6
                    for i in range(len(macro_s11)-1))
    check("macro σ₁₁ monotone (hardening)", monotone,
          f"σ₁₁ range: {macro_s11[0]:.2f} → {macro_s11[-1]:.2f}")

    return all_conv


# ---------------------------------------------------------------------------
# Test 4 — 3D sanity check
# ---------------------------------------------------------------------------
def test_3d_sanity():
    """Small 3D run — checks that no exception is raised and plastic strain appears."""
    print("\n[4] 3D sanity check")
    N   = 8
    rng = np.random.default_rng(42)
    phase = random_spheres(N, n_spheres=2, r_min=1, r_max=2, rng=rng)

    E_m, nu_m = 68900.0, 0.35
    E_f, nu_f = 400000.0, 0.23
    sigma_0   = 68.9
    H         = 1710.0

    C_mat   = isotropic_stiffness_voigt_3d(E_m, nu_m)
    C_inc   = isotropic_stiffness_voigt_3d(E_f, nu_f)
    C_field = build_C_field(phase, C_mat, C_inc).astype(np.float64)

    n_steps = 5
    eps_path = np.zeros((n_steps, 6))
    eps_path[:, 0] = np.linspace(0.0, 0.01, n_steps)

    result = solve_nonlinear(
        C_field, phase, eps_path,
        sigma_0=sigma_0, H=H,
        tol=1e-4, max_iter=500,
        verbose=False,
    )

    all_conv  = all(result['converged_history'])
    p_appears = float(result['p_star'].max()) > 0.0

    check("all steps converged", all_conv)
    check("plastic strain appears", p_appears,
          f"max p = {result['p_star'].max():.4f}")

    return all_conv and p_appears


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    results = [
        test_elastic_regime(),
        test_homogeneous_perfect_plasticity(),
        test_composite_2d_hardening(),
        test_3d_sanity(),
    ]
    n_pass = sum(results)
    n_total = len(results)
    print(f"\n{'='*50}")
    print(f"Results: {n_pass}/{n_total} tests passed")
    if n_pass < n_total:
        sys.exit(1)
