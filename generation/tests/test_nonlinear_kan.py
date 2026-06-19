"""
Unit tests for models/nonlinear_kan_tau_theta.py.

Tests are organised in three stages:
  Stage 1 (TestBSplines):          greville_abscissae, φ_sqrt, φ_kink
  Stage 2 (TestDeviatoric):        build_deviatoric_projector, extract_mu_from_C
  Stage 3 (TestNonlinearKANTau):   NonlinearKANTauTheta — elastic and plastic

Run from the project root:
    pytest generation/tests/test_nonlinear_kan.py -v

Material constants (Moulinec-Suquet 1994, matrix phase):
    E_m   = 68 900 MPa,  nu_m = 0.35
    sigma_y = 68.9 MPa,  H_lin = 1 710 MPa


/home/myuser/BGCE/project/bin/python -m pytest generation/tests/test_nonlinear_kan.py -v

"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import pytest

from models.nonlinear_kan_tau_theta import (
    greville_abscissae,
    BSpline1D,
    make_sqrt_bspline,
    make_kink_bspline,
    build_deviatoric_projector,
    extract_mu_from_C,
    NonlinearKANTauTheta,
    update_plastic_strain,
    SQRT_2_3,
    SQRT_EPS,
)

# ── shared material constants ─────────────────────────────────────────────────
E_m, nu_m  = 68_900.0, 0.35
sigma_y    = 68.9
H_lin      = 1_710.0

LAM_M = E_m * nu_m / ((1 + nu_m) * (1 - 2 * nu_m))
MU_M  = E_m / (2 * (1 + nu_m))

# 2D Mandel stiffness (C_M = D·C_V·D, D=diag(1,1,√2)):
#   C_M[2,2] = 2μ  (shear block doubled vs Voigt)
C_M_2D = np.array([
    [LAM_M + 2*MU_M, LAM_M,           0.0         ],
    [LAM_M,           LAM_M + 2*MU_M, 0.0         ],
    [0.0,             0.0,             2.0 * MU_M  ],
])

ALPHA_0 = MU_M                         # simple reference stiffness
C0_2D   = ALPHA_0 * np.eye(3)
R_SQ    = (20.0 * sigma_y / SQRT_2_3) ** 2
F_RANGE = 20.0 * sigma_y


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: B-spline components
# ─────────────────────────────────────────────────────────────────────────────

class TestBSplines:
    """Tests for greville_abscissae, BSpline1D, make_sqrt_bspline, make_kink_bspline."""

    def test_greville_degree1_uniform(self):
        """Degree-1 uniform B-spline: Greville abscissae are midpoints of spans."""
        # Uniform knots for a degree-1 B-spline with 4 control points:
        # t = [0, 0, 1/3, 2/3, 1, 1]  →  n_ctrl = 4
        knots = np.array([0.0, 0.0, 1/3, 2/3, 1.0, 1.0])
        degree = 1
        g = greville_abscissae(knots, degree)
        assert len(g) == 4
        # g[i] = mean(knots[i+1:i+2]) = knots[i+1]
        expected = knots[1:-1]
        np.testing.assert_allclose(g, expected, atol=1e-15)

    def test_greville_degree3_clamped(self):
        """Degree-3 clamped B-spline: first and last Greville abscissae at endpoints."""
        # Minimal degree-3 clamped spline: 4 control points
        # t = [0,0,0,0, 1,1,1,1]  →  n_ctrl = 4
        knots = np.array([0.0]*4 + [1.0]*4)
        degree = 3
        g = greville_abscissae(knots, degree)
        assert len(g) == 4
        assert abs(g[0] - 0.0) < 1e-15, f"g[0]={g[0]} should be 0"
        assert abs(g[-1] - 1.0) < 1e-15, f"g[-1]={g[-1]} should be 1"

    def test_knot_count_consistency(self):
        """make_sqrt_bspline must produce a knot vector of length n_ctrl + degree + 1."""
        for n_ctrl in [10, 20, 50]:
            for degree in [2, 3]:
                phi = make_sqrt_bspline(R_sq=1e4, n_ctrl=n_ctrl, degree=degree)
                assert phi.n_ctrl == n_ctrl, (
                    f"n_ctrl={phi.n_ctrl} but expected {n_ctrl} "
                    f"(degree={degree})"
                )
                K_expected = n_ctrl + degree + 1
                assert len(phi.knots) == K_expected, (
                    f"Knot count {len(phi.knots)} ≠ {K_expected}"
                )

    def test_phi_sqrt_accuracy_low_R(self):
        """φ_sqrt on [0, 1e4]: max relative error < 1e-3 on interior domain.

        Absolute error scales as √(R_sq) with domain size, so we test relative
        error which should be < 1e-5 with collocation initialisation.
        """
        R_sq  = 1e4
        phi   = make_sqrt_bspline(R_sq=R_sq, n_ctrl=50, degree=3)
        x     = torch.linspace(R_sq * 1e-3, R_sq * 0.99, 500, dtype=torch.float64)
        exact = torch.sqrt(x)
        rel_err = ((phi(x) - exact).abs() / exact.clamp(min=1e-15)).max().item()
        assert rel_err < 1e-3, f"φ_sqrt max relative error {rel_err:.2e} ≥ 1e-3"

    def test_phi_sqrt_accuracy_high_R(self):
        """φ_sqrt on benchmark domain [0, R_SQ]: max relative error < 1e-3.

        Collocation initialisation gives O(h^4) accuracy; relative error is
        independent of domain size (absolute error ∝ √R_sq cancels with exact).
        """
        phi   = make_sqrt_bspline(R_sq=R_SQ, n_ctrl=50, degree=3)
        x     = torch.linspace(R_SQ * 1e-3, R_SQ * 0.99, 500, dtype=torch.float64)
        exact = torch.sqrt(x)
        rel_err = ((phi(x) - exact).abs() / exact.clamp(min=1e-15)).max().item()
        assert rel_err < 1e-3, f"φ_sqrt max relative error {rel_err:.2e} ≥ 1e-3"

    def test_phi_sqrt_at_zero(self):
        """φ_sqrt(0) must equal 0."""
        phi = make_sqrt_bspline(R_sq=1e6, n_ctrl=50, degree=3)
        val = float(phi(torch.tensor([0.0], dtype=torch.float64)))
        assert abs(val) < 1e-10, f"φ_sqrt(0) = {val:.2e}"

    def test_phi_kink_accuracy(self):
        """φ_kink on [−F, +F]: max error vs max(x, 0) < 1e-8."""
        F   = 1_000.0
        phi = make_kink_bspline(f_min=-F, f_max=F, degree=3, n_ctrl_half=20)
        x   = torch.linspace(-F, F, 2000, dtype=torch.float64)
        err = (phi(x) - torch.relu(x)).abs().max().item()
        assert err < 1e-8, f"φ_kink max error {err:.2e} ≥ 1e-8"

    def test_phi_kink_at_zero(self):
        """φ_kink(0) = 0 (kink is at the origin, not a jump)."""
        phi = make_kink_bspline(f_min=-100.0, f_max=100.0, degree=3)
        val = float(phi(torch.tensor([0.0], dtype=torch.float64)))
        assert abs(val) < 1e-14, f"φ_kink(0) = {val:.2e}, expected 0"

    def test_phi_kink_non_negative(self):
        """φ_kink must be non-negative everywhere (approximates max(x,0))."""
        phi = make_kink_bspline(f_min=-200.0, f_max=200.0, degree=3)
        x   = torch.linspace(-200.0, 200.0, 1000, dtype=torch.float64)
        assert phi(x).min().item() >= -1e-14, "φ_kink returned a negative value"

    def test_phi_kink_zero_on_negative(self):
        """φ_kink(x) ≈ 0 for x < 0 (elastic regime)."""
        phi = make_kink_bspline(f_min=-200.0, f_max=200.0, degree=3)
        x   = torch.linspace(-200.0, -0.1, 500, dtype=torch.float64)
        err = phi(x).abs().max().item()
        assert err < 1e-8, f"φ_kink(x<0) max = {err:.2e}, expected ≈ 0"

    def test_bspline1d_shapes(self):
        """BSpline1D must handle arbitrary tensor shapes correctly."""
        phi = make_sqrt_bspline(R_sq=1e4, n_ctrl=30, degree=3)
        for shape in [(5,), (3, 4), (2, 3, 4, 5)]:
            x = torch.rand(*shape, dtype=torch.float64) * 1e4
            y = phi(x)
            assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"

    def test_bspline1d_dtype_preservation(self):
        """BSpline1D must return the same dtype as the input."""
        phi   = make_sqrt_bspline(R_sq=1e4, n_ctrl=30, degree=3)
        x_f32 = torch.rand(10, dtype=torch.float32) * 1e4
        x_f64 = torch.rand(10, dtype=torch.float64) * 1e4
        assert phi(x_f32).dtype == torch.float32
        assert phi(x_f64).dtype == torch.float64


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: deviatoric projector and μ extraction
# ─────────────────────────────────────────────────────────────────────────────

class TestDeviatoric:
    """Tests for build_deviatoric_projector and extract_mu_from_C."""

    def test_projector_2d_traceless(self):
        """2D: dev(σ) must have trace = σ₁₁-dev + σ₂₂-dev = 0."""
        P = build_deviatoric_projector(3)
        for _ in range(20):
            s = torch.randn(3, dtype=torch.float64)
            ds = P @ s
            trace_2d = float(ds[0] + ds[1])
            assert abs(trace_2d) < 1e-12, f"2D dev trace = {trace_2d:.2e}"

    def test_projector_2d_shear_unchanged(self):
        """2D: shear component is unchanged by the deviatoric projection."""
        P   = build_deviatoric_projector(3)
        sig = torch.tensor([100.0, 50.0, 30.0], dtype=torch.float64)
        s   = P @ sig
        assert abs(float(s[2] - sig[2])) < 1e-12

    def test_projector_3d_traceless(self):
        """3D: dev(σ) must have trace = 0."""
        P = build_deviatoric_projector(6)
        for _ in range(20):
            s  = torch.randn(6, dtype=torch.float64)
            ds = P @ s
            trace_3d = float(ds[0] + ds[1] + ds[2])
            assert abs(trace_3d) < 1e-12, f"3D dev trace = {trace_3d:.2e}"

    def test_projector_3d_shear_unchanged(self):
        """3D: shear components (indices 3,4,5) are unchanged by P_dev."""
        P   = build_deviatoric_projector(6)
        sig = torch.randn(6, dtype=torch.float64)
        s   = P @ sig
        assert torch.allclose(s[3:], sig[3:], atol=1e-12)

    def test_projector_is_idempotent(self):
        """P_dev is a projection: P_dev² = P_dev."""
        for n in (3, 6):
            P  = build_deviatoric_projector(n)
            P2 = P @ P
            assert torch.allclose(P, P2, atol=1e-12), f"P_dev not idempotent (n={n})"

    def test_extract_mu_2d(self):
        """extract_mu_from_C returns μ = C[:, 2, 2, ...] / 2 for 2D Mandel."""
        B, N = 2, 4
        # Uniform Mandel stiffness
        C = torch.tensor(C_M_2D, dtype=torch.float64).view(1, 3, 3, 1, 1)
        C = C.expand(B, 3, 3, N, N)
        mu = extract_mu_from_C(C, n_comp=3)
        assert mu.shape == (B, 1, N, N)
        expected = MU_M
        err = (mu - expected).abs().max().item()
        assert err < 1e-10, f"μ extraction error {err:.2e}"

    def test_extract_mu_heterogeneous(self):
        """extract_mu_from_C handles per-voxel heterogeneous stiffness."""
        B, N = 1, 4
        C = torch.zeros(B, 3, 3, N, N, dtype=torch.float64)
        C[:, 2, 2, :, :N//2] = 2.0 * MU_M     # left half: matrix
        C[:, 2, 2, :, N//2:] = 2.0 * 5.0 * MU_M  # right half: 5x stiffer
        mu = extract_mu_from_C(C, n_comp=3)
        assert mu.shape == (B, 1, N, N)
        assert torch.allclose(mu[0, 0, :, :N//2],
                              torch.full((N, N//2), MU_M,       dtype=torch.float64), atol=1e-10)
        assert torch.allclose(mu[0, 0, :, N//2:],
                              torch.full((N, N//2), 5.0*MU_M,  dtype=torch.float64), atol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: NonlinearKANTauTheta — forward pass
# ─────────────────────────────────────────────────────────────────────────────

def _make_model(sigma_y_val=sigma_y, H_val=H_lin, **kwargs):
    """Build a 2D NonlinearKANTauTheta with the benchmark material constants."""
    return NonlinearKANTauTheta(
        C0=C0_2D, sigma_y=sigma_y_val, H=H_val, n_comp=3,
        R_sq=R_SQ, f_range=F_RANGE,
        **kwargs,
    )


def _make_C_tensor(B=1, N=1):
    """(B, 3, 3, N, N) uniform Mandel stiffness tensor."""
    C = torch.tensor(C_M_2D, dtype=torch.float64).view(1, 3, 3, 1, 1)
    return C.expand(B, 3, 3, N, N).clone()


class TestNonlinearKANTau:
    """Tests for NonlinearKANTauTheta.forward()."""

    def test_output_shapes(self):
        """forward() must return (tau, sigma_new, p_new) with correct shapes."""
        model  = _make_model()
        B, N   = 2, 4
        C      = _make_C_tensor(B, N)
        eps    = torch.zeros(B, 3, N, N, dtype=torch.float64)
        ep_n   = torch.zeros(B, 3, N, N, dtype=torch.float64)
        p_n0   = torch.zeros(B, 1, N, N, dtype=torch.float64)

        tau, sig, p = model(eps, ep_n, p_n0, C)

        assert tau.shape == (B, 3, N, N)
        assert sig.shape == (B, 3, N, N)
        assert p.shape   == (B, 1, N, N)

    def test_zero_strain_gives_zero_stress(self):
        """ε = 0, ε_p_n = 0, p_n = 0  →  σ_new = 0, τ = 0, p_new = 0."""
        model = _make_model()
        C     = _make_C_tensor()
        z     = torch.zeros(1, 3, 1, 1, dtype=torch.float64)
        z1    = torch.zeros(1, 1, 1, 1, dtype=torch.float64)

        tau, sig, p = model(z, z, z1, C)

        assert tau.abs().max().item() < 1e-15
        assert sig.abs().max().item() < 1e-15
        assert p.abs().max().item()   < 1e-15

    def test_elastic_stress_equals_C_eps(self):
        """Below yield: σ_new = C:ε exactly (no plastic correction)."""
        model     = _make_model()
        C         = _make_C_tensor()
        eps_yield = sigma_y / (2.0 * MU_M)           # safe Mandel yield estimate
        eps_val   = 0.3 * eps_yield                   # 30% of yield — clearly elastic

        eps = torch.zeros(1, 3, 1, 1, dtype=torch.float64)
        eps[0, 0, 0, 0] = eps_val                     # ε₁₁ loading in Mandel

        ep_n = torch.zeros(1, 3, 1, 1, dtype=torch.float64)
        p_n0 = torch.zeros(1, 1, 1, 1, dtype=torch.float64)

        _, sig, p = model(eps, ep_n, p_n0, C)

        # Expected: σ = C_M · [ε_val, 0, 0]
        sig_expected = torch.tensor(C_M_2D, dtype=torch.float64) @ torch.tensor(
            [eps_val, 0.0, 0.0], dtype=torch.float64
        )
        err_sig = (sig[0, :, 0, 0] - sig_expected).abs().max().item()
        err_p   = p.abs().max().item()

        assert err_sig < 1e-8,  f"σ mismatch below yield: {err_sig:.2e}"
        assert err_p   < 1e-12, f"p should be 0 below yield: {err_p:.2e}"

    def test_elastic_plastic_strain_unchanged(self):
        """Below yield: plastic strain state ε_p_n is respected; p_new = p_n."""
        model   = _make_model()
        C       = _make_C_tensor()

        # Non-zero pre-existing plastic strain, but keep total strain elastic
        eps_val = 0.1 * sigma_y / (2.0 * MU_M)
        eps     = torch.zeros(1, 3, 1, 1, dtype=torch.float64)
        eps[0, 0, 0, 0] = eps_val
        ep_n    = torch.zeros(1, 3, 1, 1, dtype=torch.float64)
        ep_n[0, 0, 0, 0] = 0.02 * eps_val            # tiny pre-existing plastic strain
        p_n_val = torch.zeros(1, 1, 1, 1, dtype=torch.float64)
        p_n_val[0, 0, 0, 0] = 0.001                  # small pre-existing p

        # Elastic strain = eps - ep_n is well below yield
        _, sig, p = model(eps, ep_n, p_n_val, C)

        # p_new should equal p_n (no additional plastic increment)
        err_p = (p - p_n_val).abs().max().item()
        assert err_p < 1e-10, f"p_new changed below yield: Δp={err_p:.2e}"

    def test_plastic_stress_on_yield_surface(self):
        """Above yield: ‖dev(σ_new)‖ = √(2/3)·(σ_y + H·p_new) (yield surface)."""
        model     = _make_model()
        C         = _make_C_tensor()
        eps_yield = sigma_y / (2.0 * MU_M)
        eps_val   = 3.0 * eps_yield                   # 3× yield — clearly plastic

        eps = torch.zeros(1, 3, 1, 1, dtype=torch.float64)
        eps[0, 0, 0, 0] = eps_val
        ep_n = torch.zeros(1, 3, 1, 1, dtype=torch.float64)
        p_n0 = torch.zeros(1, 1, 1, 1, dtype=torch.float64)

        _, sig, p = model(eps, ep_n, p_n0, C)

        P_dev  = build_deviatoric_projector(3)
        s_vec  = P_dev @ sig[0, :, 0, 0]
        norm_s = float(torch.sqrt((s_vec**2).sum()))
        thresh = SQRT_2_3 * (sigma_y + H_lin * float(p[0, 0, 0, 0]))

        err = abs(norm_s - thresh)
        assert err < 1e-4, (
            f"Stress not on yield surface: ‖s‖={norm_s:.6f}, threshold={thresh:.6f}, "
            f"err={err:.2e}"
        )

    def test_plastic_strain_increases(self):
        """Above yield: p_new > p_n (plastic strain accumulates)."""
        model     = _make_model()
        C         = _make_C_tensor()
        eps_yield = sigma_y / (2.0 * MU_M)

        eps = torch.zeros(1, 3, 1, 1, dtype=torch.float64)
        eps[0, 0, 0, 0] = 3.0 * eps_yield
        ep_n = torch.zeros(1, 3, 1, 1, dtype=torch.float64)
        p_n0 = torch.zeros(1, 1, 1, 1, dtype=torch.float64)

        _, _, p = model(eps, ep_n, p_n0, C)
        assert float(p[0, 0, 0, 0]) > 0.0, "p_new should increase above yield"

    def test_perfect_plasticity_plateau(self):
        """With H=0: σ_eq = sigma_y regardless of how far past yield we are."""
        model     = _make_model(sigma_y_val=sigma_y, H_val=0.0)
        C         = _make_C_tensor()
        eps_yield = sigma_y / (2.0 * MU_M)

        P_dev = build_deviatoric_projector(3)
        for mult in [1.5, 3.0, 10.0]:
            eps = torch.zeros(1, 3, 1, 1, dtype=torch.float64)
            eps[0, 0, 0, 0] = mult * eps_yield
            ep_n = torch.zeros(1, 3, 1, 1, dtype=torch.float64)
            p_n0 = torch.zeros(1, 1, 1, 1, dtype=torch.float64)

            _, sig, _ = model(eps, ep_n, p_n0, C)

            s_vec  = P_dev @ sig[0, :, 0, 0]
            norm_s = float(torch.sqrt((s_vec**2).sum()))
            sigma_eq = norm_s / SQRT_2_3   # von Mises σ_eq = ‖s‖ / √(2/3)

            assert abs(sigma_eq - sigma_y) / sigma_y < 1e-4, (
                f"Perfect plasticity: σ_eq={sigma_eq:.4f} ≠ σ_y={sigma_y} "
                f"(ε={mult}·ε_yield)"
            )

    def test_tau_equals_sigma_minus_C0_eps(self):
        """τ = σ_new − C⁰:ε must hold exactly for the output."""
        model   = _make_model()
        B, N    = 2, 4
        C       = _make_C_tensor(B, N)
        eps     = torch.randn(B, 3, N, N, dtype=torch.float64) * sigma_y / (2.0 * MU_M)
        ep_n    = torch.zeros(B, 3, N, N, dtype=torch.float64)
        p_n0    = torch.zeros(B, 1, N, N, dtype=torch.float64)

        tau, sig, _ = model(eps, ep_n, p_n0, C)

        C0_t = torch.tensor(C0_2D, dtype=torch.float64)
        C0_eps = torch.einsum('ij,bjxy->bixy', C0_t, eps)
        tau_expected = sig - C0_eps

        err = (tau - tau_expected).abs().max().item()
        assert err < 1e-12, f"τ ≠ σ_new − C⁰:ε  (max diff = {err:.2e})"

    def test_high_yield_stress_no_plasticity(self):
        """With σ_y → ∞: no plastic correction for any finite strain."""
        model   = _make_model(sigma_y_val=1e15)
        C       = _make_C_tensor()
        eps_val = 10.0 * sigma_y / (2.0 * MU_M)   # large strain, but below huge yield

        eps = torch.zeros(1, 3, 1, 1, dtype=torch.float64)
        eps[0, 0, 0, 0] = eps_val
        ep_n = torch.zeros(1, 3, 1, 1, dtype=torch.float64)
        p_n0 = torch.zeros(1, 1, 1, 1, dtype=torch.float64)

        _, sig, p = model(eps, ep_n, p_n0, C)

        sig_expected = torch.tensor(C_M_2D, dtype=torch.float64) @ torch.tensor(
            [eps_val, 0.0, 0.0], dtype=torch.float64
        )
        err_sig = (sig[0, :, 0, 0] - sig_expected).abs().max().item()
        err_p   = p.abs().max().item()

        assert err_sig < 1e-8,  f"σ mismatch with huge σ_y: {err_sig:.2e}"
        assert err_p   < 1e-12, f"p should be 0 with huge σ_y: {err_p:.2e}"

    def test_spatial_independence(self):
        """
        NonlinearKANTauTheta is pointwise: each voxel must give the same result
        as running the operator on that voxel alone.
        """
        torch.manual_seed(42)
        model   = _make_model()
        B, N    = 1, 8
        C       = _make_C_tensor(B, N)
        eps_val = sigma_y / (2.0 * MU_M)
        eps     = (torch.rand(B, 3, N, N, dtype=torch.float64) - 0.5) * 2.0 * eps_val
        ep_n    = torch.zeros(B, 3, N, N, dtype=torch.float64)
        p_n0    = torch.zeros(B, 1, N, N, dtype=torch.float64)

        tau_full, sig_full, p_full = model(eps, ep_n, p_n0, C)

        # Compare voxel (0, 3, 5) run in isolation
        ix, iy = 3, 5
        eps_1   = eps[  :, :, ix:ix+1, iy:iy+1]
        ep_n_1  = ep_n[ :, :, ix:ix+1, iy:iy+1]
        p_n0_1  = p_n0[ :, :, ix:ix+1, iy:iy+1]
        C_1     = C[    :, :, :, ix:ix+1, iy:iy+1]

        tau_1, sig_1, p_1 = model(eps_1, ep_n_1, p_n0_1, C_1)

        err_tau = (tau_full[:, :, ix, iy] - tau_1[:, :, 0, 0]).abs().max().item()
        err_sig = (sig_full[:, :, ix, iy] - sig_1[:, :, 0, 0]).abs().max().item()

        assert err_tau < 1e-12, f"tau not pointwise-independent: {err_tau:.2e}"
        assert err_sig < 1e-12, f"sig not pointwise-independent: {err_sig:.2e}"

    def test_update_plastic_strain_consistency(self):
        """
        update_plastic_strain: eps_p_new = eps − C⁻¹:sigma_new.
        Verify against the direct formula for a purely elastic case.
        """
        model   = _make_model(sigma_y_val=1e15)  # purely elastic
        B, N    = 1, 4
        C       = _make_C_tensor(B, N)
        eps_val = 0.5 * sigma_y / (2.0 * MU_M)
        eps     = torch.zeros(B, 3, N, N, dtype=torch.float64)
        eps[:, 0] = eps_val
        ep_n    = torch.zeros(B, 3, N, N, dtype=torch.float64)
        p_n0    = torch.zeros(B, 1, N, N, dtype=torch.float64)

        _, sig, _ = model(eps, ep_n, p_n0, C)
        eps_p_recovered = update_plastic_strain(eps, sig, C)

        # Elastic: eps_p = 0 everywhere
        err = eps_p_recovered.abs().max().item()
        assert err < 1e-10, f"eps_p should be 0 in elastic case: {err:.2e}"
