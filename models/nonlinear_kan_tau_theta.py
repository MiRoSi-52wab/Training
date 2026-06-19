"""
Nonlinear KAN polarisation-stress operator for von Mises plasticity.

This is the drop-in replacement for KANTauTheta in the LS-FNO Fourier layer
when the material is elastoplastic (von Mises with isotropic hardening).

The operator computes:
    τ_θ(ε, ε_p_n, p_n, C) = σ_new(ε, ε_p_n, p_n, C) − C⁰ : ε

where σ_new is the radial-return stress update for von Mises plasticity.

Two non-polynomial operations are represented by analytically initialised
B-splines that can optionally be made trainable:
    φ_sqrt : ℝ≥0 → ℝ≥0    approximates √(x)       (collocation init)
    φ_kink : ℝ   → ℝ≥0    approximates max(x, 0)   (exact kink at x=0 via
                                                      knot of multiplicity = degree)

All other operations (Hooke's law, deviatoric projection, radial correction,
polarisation stress) are implemented as exact floating-point linear algebra.

Notation
--------
Internal state uses Mandel notation (same as ls_fno.py and kan_tau_theta.py):
    2D plane strain: [ε₁₁, ε₂₂, √2·ε₁₂]                  (n = 3)
    3D:              [ε₁₁, ε₂₂, ε₃₃, √2·ε₂₃, √2·ε₁₃, √2·ε₁₂]  (n = 6)

The √2 factor on shear components means the Mandel inner product equals the
full tensor double contraction: a:b = Σᵢ aᵢbᵢ.

In Mandel notation, the Mandel stiffness for an isotropic material has:
    C_M[shear, shear] = 2μ      (vs C_V[shear, shear] = μ in Voigt)

Notes on B-spline initialization accuracy
------------------------------------------
φ_sqrt approximates √(x) on [0, R_sq].  Simple Greville initialisation
(c_i = √(g_i)) gives only O(h²) accuracy — for large R_sq this produces
unacceptably large yield-surface residuals.  We instead solve the collocation
system B(g) @ ctrl = √(g) at the Greville abscissae, which gives the full
O(h^{degree+1}) accuracy of B-spline interpolation.  scipy is used for the
collocation solve; if unavailable a Greville fallback is used with a warning.

References
----------
    NONLINEAR_KAN_IMPLEMENTATION.md
    Moulinec & Suquet (1994), "A fast numerical method..."
    Simo & Hughes (1998), "Computational Inelasticity", Chapter 3
"""

import warnings
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

SQRT_2_3 = float(np.sqrt(2.0 / 3.0))
SQRT_EPS = 1e-15   # numerical floor for division (avoids 0/0 in elastic case)


# ─────────────────────────────────────────────────────────────────────────────
# Utility: Greville abscissae
# ─────────────────────────────────────────────────────────────────────────────

def greville_abscissae(knots: np.ndarray, degree: int) -> np.ndarray:
    """
    Greville abscissae for a B-spline of given degree.

        g[i] = mean(knots[i+1 : i+degree+1])   for i = 0 … n_ctrl−1

    where n_ctrl = len(knots) − degree − 1.

    Args:
        knots:  full clamped knot vector (1D numpy array)
        degree: polynomial degree

    Returns:
        g: (n_ctrl,) array of Greville abscissae
    """
    n_ctrl = len(knots) - degree - 1
    return np.array([
        np.mean(knots[i + 1: i + degree + 1])
        for i in range(n_ctrl)
    ])


# ─────────────────────────────────────────────────────────────────────────────
# BSpline1D: differentiable B-spline evaluator
# ─────────────────────────────────────────────────────────────────────────────

class BSpline1D(nn.Module):
    """
    Trainable 1-D B-spline with a fixed knot vector and learnable control points.

    Evaluates at arbitrary-shape input tensors using the vectorised Cox-de Boor
    recursion. The clamp-based 0/0 handling ensures correct gradients at knot
    boundaries. Inputs are clamped to the knot domain before evaluation.

    Args:
        knots:     (K,) full clamped knot vector (numpy array).
        ctrl_init: (n_ctrl,) initial control point values (numpy array).
        degree:    polynomial degree (default 3).
        trainable: if True, control points are nn.Parameters.
    """

    def __init__(
        self,
        knots: np.ndarray,
        ctrl_init: np.ndarray,
        degree: int = 3,
        trainable: bool = False,
    ):
        super().__init__()
        self.degree = degree
        self.n_ctrl = len(ctrl_init)
        assert len(knots) == self.n_ctrl + degree + 1, (
            f"Knot vector length {len(knots)} must equal n_ctrl + degree + 1 "
            f"= {self.n_ctrl + degree + 1}"
        )

        t = torch.tensor(knots,     dtype=torch.float64)
        c = torch.tensor(ctrl_init, dtype=torch.float64)

        self.register_buffer('knots', t)
        if trainable:
            self.ctrl = nn.Parameter(c)
        else:
            self.register_buffer('ctrl', c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the B-spline at x (arbitrary shape, any dtype).

        Returns a tensor with the same shape and dtype as x.
        """
        orig_dtype = x.dtype
        x = x.double()

        shape   = x.shape
        x_flat  = x.reshape(-1)          # (M,)
        M       = x_flat.shape[0]
        device  = x_flat.device

        # Clamp to the supported domain
        t_lo = self.knots[self.degree]
        t_hi = self.knots[-(self.degree + 1)]
        x_flat = x_flat.clamp(t_lo, t_hi)

        t = self.knots       # (K,)

        # ── Cox-de Boor degree-0 basis ────────────────────────────────────────
        t_l = t[:-1].unsqueeze(1)    # (K-1, 1)
        t_r = t[1: ].unsqueeze(1)    # (K-1, 1)
        x_e = x_flat.unsqueeze(0)    # (1, M)

        N_basis = ((x_e >= t_l) & (x_e < t_r)).to(torch.float64)  # (K-1, M)

        # Include x == t[-1] in the last NON-DEGENERATE interval.
        # For a clamped degree-d B-spline the last d+1 knots equal t_hi, so the
        # last non-degenerate span is at degree-0 index K-d-2 = n_ctrl-1.
        # Setting N_basis[-1] (the degenerate [t_hi,t_hi] interval) would be
        # zero'd out by the recursion's dL/dR=0 guards — wrong index.
        at_right = (x_flat >= t_hi)
        N_basis[self.n_ctrl - 1] = torch.where(
            at_right,
            torch.ones(M, dtype=torch.float64, device=device),
            N_basis[self.n_ctrl - 1],
        )

        # ── Recursion: degree 1 … degree ─────────────────────────────────────
        for d in range(1, self.degree + 1):
            n_out = N_basis.shape[0] - 1

            ti   = t[:n_out]
            ti_d  = t[d     : n_out + d    ]
            ti_d1 = t[d + 1 : n_out + d + 1]
            ti_1  = t[1     : n_out + 1    ]

            dL = (ti_d  - ti  ).unsqueeze(1)   # (n_out, 1)
            dR = (ti_d1 - ti_1).unsqueeze(1)   # (n_out, 1)

            x_b = x_flat.unsqueeze(0).expand(n_out, M)

            left = torch.where(
                dL > 0,
                (x_b - ti.unsqueeze(1)) / dL.clamp(min=1e-300) * N_basis[:n_out],
                torch.zeros(n_out, M, dtype=torch.float64, device=device),
            )
            right = torch.where(
                dR > 0,
                (ti_d1.unsqueeze(1) - x_b) / dR.clamp(min=1e-300) * N_basis[1: n_out + 1],
                torch.zeros(n_out, M, dtype=torch.float64, device=device),
            )
            N_basis = left + right   # (n_out, M)

        # N_basis: (n_ctrl, M)  — evaluate spline
        y_flat = (N_basis * self.ctrl.unsqueeze(1)).sum(dim=0)  # (M,)
        return y_flat.reshape(shape).to(orig_dtype)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"BSpline1D(n_ctrl={self.n_ctrl}, degree={self.degree}, "
            f"trainable={self.n_params() > 0})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Factory: φ_sqrt — B-spline for √(x) on [0, R_sq]
# ─────────────────────────────────────────────────────────────────────────────

def _collocation_ctrl(knots: np.ndarray, degree: int,
                      targets: np.ndarray) -> np.ndarray:
    """
    Solve B-spline collocation: find ctrl such that B(g[i]) = targets[i].

    Builds the basis matrix N[i,j] = N_j(g[i]) (where g are the Greville
    abscissae) using scipy.interpolate.BSpline, then solves the linear system.
    Falls back to returning targets directly (Greville approximation) if scipy
    is unavailable.

    Returns ctrl: (n_ctrl,) numpy array.
    """
    try:
        from scipy.interpolate import BSpline as SciB
        n_ctrl = len(targets)
        g = greville_abscissae(knots, degree)
        N_mat = np.zeros((n_ctrl, n_ctrl))
        for j in range(n_ctrl):
            c = np.zeros(n_ctrl); c[j] = 1.0
            N_mat[:, j] = SciB(knots, c, degree)(np.maximum(g, 0.0))
        return np.linalg.solve(N_mat, targets)
    except ImportError:
        warnings.warn(
            "scipy not found: using Greville initialisation for φ_sqrt "
            "(O(h²) accuracy). Install scipy for better initialisation.",
            RuntimeWarning, stacklevel=3,
        )
        return targets


def make_sqrt_bspline(
    R_sq:      float,
    n_ctrl:    int  = 50,
    degree:    int  = 3,
    trainable: bool = False,
) -> BSpline1D:
    """
    Build a BSpline1D that approximates √(x) on [0, R_sq].

    Knot placement: power-law (x_i ∝ (i/N)^4 · R_sq), denser near 0 to match
    the high curvature of √(x) at the origin.

    Control-point initialisation: B-spline collocation at the Greville
    abscissae (solves N @ ctrl = √(g)), giving O(h^{degree+1}) accuracy.
    Falls back to Greville approximation (O(h²)) if scipy is unavailable.

    With n_ctrl=100 and scipy the max relative error on [0, R_sq] is < 1e-3.

    Args:
        R_sq:      upper bound of the domain (≥ max expected ‖s_trial‖²).
        n_ctrl:    number of control points (default 50; 100 recommended
                   for better accuracy when R_sq is large).
        degree:    B-spline polynomial degree (default 3).
        trainable: if True, control points are learnable parameters.
    """
    # Number of interior knots.
    # Clamped spline: K = (d+1) + n_internal + (d+1) = n_ctrl + d + 1
    # → n_internal = n_ctrl - d - 1
    n_internal = n_ctrl - degree - 1
    assert n_internal >= 0, (
        f"n_ctrl={n_ctrl} too small for degree={degree}; need n_ctrl > degree+1"
    )

    # Power-law (k=4) internal knots: denser near 0 where √x has high curvature
    if n_internal > 0:
        u = np.linspace(0.0, 1.0, n_internal + 2)[1:-1]
        t_internal = (u ** 4) * R_sq
    else:
        t_internal = np.empty(0)

    knots = np.concatenate([
        np.zeros(degree + 1),          # clamped at 0
        t_internal,
        np.full(degree + 1, R_sq),     # clamped at R_sq
    ])

    g          = greville_abscissae(knots, degree)       # (n_ctrl,)
    g_safe     = np.maximum(g, 0.0)
    rhs        = np.sqrt(g_safe)                          # target: √(g_i)
    ctrl_init  = _collocation_ctrl(knots, degree, rhs)   # O(h^{d+1}) init

    return BSpline1D(knots, ctrl_init, degree=degree, trainable=trainable)


# ─────────────────────────────────────────────────────────────────────────────
# Factory: φ_kink — B-spline for max(x, 0) with exact kink at 0
# ─────────────────────────────────────────────────────────────────────────────

def make_kink_bspline(
    f_min:        float,
    f_max:        float,
    degree:       int  = 3,
    n_ctrl_half:  int  = 20,
    trainable:    bool = False,
) -> BSpline1D:
    """
    Build a BSpline1D that exactly represents max(x, 0) on [f_min, f_max].

    A knot of multiplicity = degree is placed at x = 0, enforcing C⁰ continuity
    (kink) at the yield surface — the exact mathematical structure of the
    elastic/plastic transition. Control points are initialised to c[i] = max(g[i], 0)
    via Greville interpolation, which makes the spline reproduce max(x, 0) exactly
    (because max(x,0) is piecewise linear, and Greville init gives exact
    representations of degree ≤ 1 polynomials via the linear precision property).

    Args:
        f_min:       lower bound (negative, e.g. −10*sigma_y).
        f_max:       upper bound (positive, e.g. +10*sigma_y).
        degree:      B-spline polynomial degree.
        n_ctrl_half: target number of control points per half of the domain.
        trainable:   if True, control points are learnable parameters.
    """
    assert f_min < 0.0 < f_max, "f_min must be negative and f_max positive"

    n_internal_each = max(n_ctrl_half - degree, 1)

    # Interior knots on each side (excluding the shared endpoints and the kink at 0)
    t_neg = np.linspace(f_min, 0.0,  n_internal_each + 2)[1:-1]
    t_pos = np.linspace(0.0,  f_max, n_internal_each + 2)[1:-1]

    knots = np.concatenate([
        np.full(degree + 1, f_min),   # clamped start
        t_neg,                         # internal knots on negative side
        np.zeros(degree),              # multiplicity = degree at 0 → C⁰ kink
        t_pos,                         # internal knots on positive side
        np.full(degree + 1, f_max),   # clamped end
    ])

    g         = greville_abscissae(knots, degree)
    ctrl_init = np.maximum(g, 0.0)   # linear precision → exact for max(x,0)

    return BSpline1D(knots, ctrl_init, degree=degree, trainable=trainable)


# ─────────────────────────────────────────────────────────────────────────────
# Deviatoric projector in Mandel notation
# ─────────────────────────────────────────────────────────────────────────────

def build_deviatoric_projector(n_comp: int) -> torch.Tensor:
    """
    Deviatoric projection matrix P_dev in Mandel notation.

        dev(σ) = σ − (tr σ / d) I     (d = spatial dimension)

    2D Mandel [σ₁₁, σ₂₂, √2·σ₁₂] (n=3):
        P_dev = diag-block([[ 1/2, -1/2], [-1/2, 1/2]], [1])

    3D Mandel [σ₁₁, σ₂₂, σ₃₃, √2·σ₂₃, √2·σ₁₃, √2·σ₁₂] (n=6):
        P_dev = diag-block([[2/3, -1/3, -1/3], ...], I₃)

    Returns:
        P_dev: (n, n) float64 tensor
    """
    if n_comp == 3:
        d, n_normal = 2, 2
    elif n_comp == 6:
        d, n_normal = 3, 3
    else:
        raise ValueError(f"n_comp must be 3 (2D) or 6 (3D), got {n_comp}")

    P = torch.eye(n_comp, dtype=torch.float64)
    for i in range(n_normal):
        for j in range(n_normal):
            P[i, j] = (1.0 - 1.0 / d) if i == j else (-1.0 / d)
    return P


# ─────────────────────────────────────────────────────────────────────────────
# Extract shear modulus from Mandel stiffness
# ─────────────────────────────────────────────────────────────────────────────

def extract_mu_from_C(C: torch.Tensor, n_comp: int) -> torch.Tensor:
    """
    Shear modulus μ per voxel from the Mandel stiffness tensor.

    In Mandel notation, C[shear_idx, shear_idx] = 2μ for an isotropic material:
        2D (n=3): shear_idx = 2,  μ = C[:, 2, 2, ...] / 2
        3D (n=6): shear_idx = 3,  μ = C[:, 3, 3, ...] / 2

    Args:
        C:      (B, n, n, *spatial) Mandel stiffness field.
        n_comp: 3 (2D) or 6 (3D).

    Returns:
        mu: (B, 1, *spatial) shear modulus per voxel.
    """
    shear_idx = {3: 2, 6: 3}.get(n_comp)
    if shear_idx is None:
        raise ValueError(f"n_comp must be 3 or 6, got {n_comp}")
    return C[:, shear_idx, shear_idx, ...].unsqueeze(1) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Main module: NonlinearKANTauTheta
# ─────────────────────────────────────────────────────────────────────────────

class NonlinearKANTauTheta(nn.Module):
    """
    KAN-based polarisation-stress operator for von Mises plasticity.

    Drop-in replacement for KANTauTheta in the LS-FNO Fourier layer for the
    nonlinear (elastoplastic) problem.

    Architecture:
        φ_sqrt : BSpline1D — approximates √(x), used to compute ‖dev(σ_trial)‖
        φ_kink : BSpline1D — approximates max(x, 0) with exact kink at x = 0
        All other steps: exact floating-point linear algebra

    The forward pass implements the 13-step radial-return algorithm in
    differentiable form so that gradients flow through both B-splines when
    trainable=True.

    Args:
        C0:               (n, n) numpy array — reference stiffness in Mandel.
                          Typically α₀·I where α₀ = (α⁺ + α⁻)/2.
        sigma_y:          Initial yield stress (MPa) of the plastic phase.
        H:                Isotropic hardening modulus (MPa).
        n_comp:           Mandel components: 3 for 2D, 6 for 3D.
        R_sq:             Upper bound on ‖s_trial‖² for φ_sqrt domain.
                          Set generously: e.g. (E_max * eps_max)² or
                          (20 * sigma_y / SQRT_2_3)².
        f_range:          Half-range for φ_kink: domain is [−f_range, +f_range].
                          Should be at least 10 * sigma_y.
        n_ctrl_sqrt:      Control points for φ_sqrt (default 100 for good
                          accuracy on large R_sq; 50 is acceptable if scipy
                          collocation is available and R_sq is modest).
        n_ctrl_kink_half: Control points per half-domain for φ_kink (default 20).
        degree:           B-spline degree for both splines (default 3).
        trainable:        If True, B-spline control points are nn.Parameters.
    """

    def __init__(
        self,
        C0:               np.ndarray,
        sigma_y:          float,
        H:                float,
        n_comp:           int   = 3,
        R_sq:             float = 1e8,
        f_range:          float = 1e6,
        n_ctrl_sqrt:      int   = 100,
        n_ctrl_kink_half: int   = 20,
        degree:           int   = 3,
        trainable:        bool  = False,
    ):
        super().__init__()
        self.n       = n_comp
        self.sigma_y = float(sigma_y)
        self.H       = float(H)

        # Reference stiffness C⁰ (n, n)
        self.register_buffer('C0', torch.tensor(C0, dtype=torch.float64))

        # Deviatoric projector P_dev (n, n)
        self.register_buffer('P_dev', build_deviatoric_projector(n_comp))

        # φ_sqrt: B-spline for √(x) on [0, R_sq]
        self.phi_sqrt = make_sqrt_bspline(
            R_sq=R_sq,
            n_ctrl=n_ctrl_sqrt,
            degree=degree,
            trainable=trainable,
        )

        # φ_kink: B-spline for max(x, 0) on [−f_range, +f_range]
        self.phi_kink = make_kink_bspline(
            f_min=-f_range,
            f_max=+f_range,
            degree=degree,
            n_ctrl_half=n_ctrl_kink_half,
            trainable=trainable,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _matvec(self, M: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Batched matrix-vector product:  y[b, i, *] = Σ_j M[..., i, j, *] · v[b, j, *]

        Handles:
            M: (n, n)           — uniform matrix, broadcast over batch + spatial
            M: (B, n, n, *sp)   — per-voxel matrix

        v: (B, n, *sp)   →   returns (B, n, *sp)
        """
        if M.dim() == 2:
            n_sp  = v.dim() - 2
            M_v   = M.view(1, self.n, self.n, *([1] * n_sp))
            v_exp = v.unsqueeze(1)                          # (B, 1, n, *sp)
            return (M_v * v_exp).sum(dim=2)
        else:
            return (M * v.unsqueeze(1)).sum(dim=2)          # (B, n, *sp)

    # ── forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        eps:     torch.Tensor,
        eps_p_n: torch.Tensor,
        p_n:     torch.Tensor,
        C:       torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Radial-return stress update and polarisation stress (13-step algorithm).

        Args:
            eps:     (B, n, *spatial)     current strain iterate (LS iteration)
            eps_p_n: (B, n, *spatial)     plastic strain from previous time step n
            p_n:     (B, 1, *spatial)     accumulated plastic strain from step n
            C:       (B, n, n, *spatial)  Mandel stiffness field per voxel

        Returns:
            tau:       (B, n, *spatial)  polarisation stress = sigma_new − C⁰:ε
            sigma_new: (B, n, *spatial)  updated Cauchy stress (on/inside yield surface)
            p_new:     (B, 1, *spatial)  updated accumulated plastic strain
        """
        orig_dtype = eps.dtype
        eps     = eps.double()
        eps_p_n = eps_p_n.double()
        p_n     = p_n.double()
        C       = C.double()

        # Step 1 — elastic strain
        eps_e = eps - eps_p_n                                  # (B, n, *sp)

        # Step 2 — trial stress
        sigma_trial = self._matvec(C, eps_e)                   # (B, n, *sp)

        # Step 3 — deviatoric trial stress
        s_trial = self._matvec(self.P_dev, sigma_trial)        # (B, n, *sp)

        # Step 4 — squared norm  ‖s_trial‖²  (Mandel inner product = Euclidean)
        q = (s_trial ** 2).sum(dim=1, keepdim=True)            # (B, 1, *sp)

        # Step 5 — ‖s_trial‖ via φ_sqrt  [B-spline #1]
        norm_s = self.phi_sqrt(q)                              # (B, 1, *sp) ≈ √q

        # Step 6 — yield threshold
        threshold = SQRT_2_3 * (self.sigma_y + self.H * p_n)  # (B, 1, *sp)

        # Step 7 — trial yield function
        f_trial = norm_s - threshold                           # (B, 1, *sp)

        # Step 8 — positive part via φ_kink  [B-spline #2]
        f_plus = self.phi_kink(f_trial)                        # (B, 1, *sp) ≈ max(f,0)

        # Step 9 — plastic multiplier
        mu          = extract_mu_from_C(C, self.n)            # (B, 1, *sp)
        denom       = 2.0 * mu + (2.0 / 3.0) * self.H        # (B, 1, *sp)
        delta_gamma = f_plus / denom                           # (B, 1, *sp), ≥ 0

        # Step 10 — radial correction scale (numerically safe)
        # When elastic: f_plus = 0 → delta_gamma = 0 → scale = 0  ✓
        # When plastic: norm_s ≥ threshold > 0                     ✓
        scale = (2.0 * mu * delta_gamma) / norm_s.clamp(min=SQRT_EPS)  # (B, 1, *sp)

        # Step 11 — updated stress
        sigma_new = sigma_trial - scale * s_trial              # (B, n, *sp)

        # Step 12 — updated accumulated plastic strain
        delta_p = SQRT_2_3 * delta_gamma                       # (B, 1, *sp)
        p_new   = p_n + delta_p                                # (B, 1, *sp)

        # Step 13 — polarisation stress
        C0_eps = self._matvec(self.C0, eps)                    # (B, n, *sp)
        tau    = sigma_new - C0_eps                            # (B, n, *sp)

        return tau.to(orig_dtype), sigma_new.to(orig_dtype), p_new.to(orig_dtype)

    # ── inspection ────────────────────────────────────────────────────────────

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        trainable = self.n_params() > 0
        return (
            f"NonlinearKANTauTheta(n={self.n}, sigma_y={self.sigma_y:.4g}, "
            f"H={self.H:.4g}, trainable={trainable}, "
            f"ctrl_sqrt={self.phi_sqrt.n_ctrl}, ctrl_kink={self.phi_kink.n_ctrl})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helper: recover updated plastic strain tensor from converged LS iterate
# ─────────────────────────────────────────────────────────────────────────────

def update_plastic_strain(
    eps:       torch.Tensor,
    sigma_new: torch.Tensor,
    C:         torch.Tensor,
) -> torch.Tensor:
    """
    Recover ε_p_new = ε − C⁻¹ : σ_new after the LS loop has converged.

    The elastic strain satisfies σ = C : ε_e, so ε_e = C⁻¹ : σ and
    ε_p = ε − ε_e.  Uses torch.linalg.solve for per-voxel linear solves.

    Args:
        eps:       (B, n, *spatial)     converged total strain
        sigma_new: (B, n, *spatial)     converged Cauchy stress
        C:         (B, n, n, *spatial)  Mandel stiffness field

    Returns:
        eps_p_new: (B, n, *spatial)
    """
    B, n = sigma_new.shape[:2]
    sp   = sigma_new.shape[2:]
    N_vox = int(np.prod(sp)) if sp else 1

    C_flat = C.reshape(B, n, n, N_vox).permute(0, 3, 1, 2).reshape(B * N_vox, n, n)
    s_flat = sigma_new.reshape(B, n, N_vox).permute(0, 2, 1).reshape(B * N_vox, n, 1)
    e_flat = eps.reshape(B, n, N_vox).permute(0, 2, 1).reshape(B * N_vox, n, 1)

    eps_e_flat = torch.linalg.solve(C_flat, s_flat)   # (B*N_vox, n, 1)
    eps_p_flat = e_flat - eps_e_flat                   # (B*N_vox, n, 1)

    return eps_p_flat.reshape(B, N_vox, n).permute(0, 2, 1).reshape(B, n, *sp)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test / quick verification
# ─────────────────────────────────────────────────────────────────────────────

def _verify_nonlinear_kan() -> None:
    """
    Quick verification of the NonlinearKANTauTheta module.

    Checks:
    1. Below yield: sigma_new = C:eps (purely elastic), p unchanged.
    2. Above yield: corrected stress lies on the yield surface.
    3. φ_sqrt accuracy: max relative error < 1e-3 on [R_sq*1e-4, R_sq*0.99].
    4. φ_kink: max error < 1e-8 on [f_min, f_max].
    """
    torch.set_grad_enabled(False)

    E, nu    = 68_900.0, 0.35
    lam      = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu       = E / (2 * (1 + nu))
    sigma_y  = 68.9
    H        = 1_710.0

    # Mandel stiffness (2D): C_M = D·C_V·D, D=diag(1,1,√2)
    C_np = np.array([
        [lam + 2*mu, lam,         0.0        ],
        [lam,         lam + 2*mu, 0.0        ],
        [0.0,         0.0,        2.0 * mu   ],
    ])
    alpha_0 = mu
    C0_np   = alpha_0 * np.eye(3)
    R_sq    = (20.0 * sigma_y / SQRT_2_3) ** 2
    f_range = 20.0 * sigma_y

    model = NonlinearKANTauTheta(
        C0=C0_np, sigma_y=sigma_y, H=H, n_comp=3,
        R_sq=R_sq, f_range=f_range, n_ctrl_sqrt=100,
    )
    print(model)

    C_t = torch.tensor(C_np, dtype=torch.float64).view(1, 3, 3, 1, 1)

    # ── Test 1: elastic regime ─────────────────────────────────────────────
    eps_yield = sigma_y / (2.0 * mu)
    eps_val   = 0.3 * eps_yield
    eps_1     = torch.tensor([[[[eps_val]], [[0.0]], [[0.0]]]], dtype=torch.float64)
    ep_n      = torch.zeros(1, 3, 1, 1, dtype=torch.float64)
    p_n0      = torch.zeros(1, 1, 1, 1, dtype=torch.float64)

    tau, sig, p = model(eps_1, ep_n, p_n0, C_t)

    sig_expected = torch.tensor(C_np, dtype=torch.float64) @ torch.tensor(
        [eps_val, 0.0, 0.0], dtype=torch.float64
    )
    err_sig = (sig[0, :, 0, 0] - sig_expected).abs().max().item()
    err_p   = p.abs().max().item()
    status  = '✓' if err_sig < 1e-8 and err_p < 1e-12 else '✗'
    print(f"  [1] Elastic: Δσ={err_sig:.2e}  Δp={err_p:.2e}  {status}")
    assert err_sig < 1e-8  and err_p < 1e-12, "Elastic test failed"

    # ── Test 2: plastic regime — stress on yield surface ──────────────────
    eps_2  = torch.tensor([[[[3.0 * eps_yield]], [[0.0]], [[0.0]]]], dtype=torch.float64)
    tau, sig, p = model(eps_2, ep_n, p_n0, C_t)

    P_dev = build_deviatoric_projector(3)
    s_vec = P_dev @ sig[0, :, 0, 0]
    norm_s_actual = float(torch.sqrt((s_vec**2).sum()))
    threshold_val = SQRT_2_3 * (sigma_y + H * float(p[0, 0, 0, 0]))
    err_yld = abs(norm_s_actual - threshold_val)
    status  = '✓' if err_yld < 1e-3 else '✗'
    print(f"  [2] Plastic: yield error={err_yld:.4e}  {status}")
    assert err_yld < 1e-3, f"Plastic yield-surface test failed: err={err_yld:.4e}"

    # ── Test 3: φ_sqrt relative accuracy ──────────────────────────────────
    phi_sqrt = make_sqrt_bspline(R_sq=R_sq, n_ctrl=100, degree=3)
    x_test   = torch.linspace(R_sq * 1e-4, R_sq * 0.99, 1000, dtype=torch.float64)
    y_exact  = torch.sqrt(x_test)
    rel_err  = ((phi_sqrt(x_test) - y_exact).abs() / y_exact).max().item()
    status   = '✓' if rel_err < 1e-3 else '✗'
    print(f"  [3] φ_sqrt max rel error: {rel_err:.2e}  {status}")
    assert rel_err < 1e-3, f"φ_sqrt relative error too large: {rel_err:.2e}"

    # ── Test 4: φ_kink accuracy ────────────────────────────────────────────
    phi_kink = make_kink_bspline(f_min=-f_range, f_max=f_range, degree=3)
    x_k      = torch.linspace(-f_range, f_range, 2000, dtype=torch.float64)
    err_kink = (phi_kink(x_k) - torch.relu(x_k)).abs().max().item()
    status   = '✓' if err_kink < 1e-8 else '✗'
    print(f"  [4] φ_kink  max error: {err_kink:.2e}  {status}")
    assert err_kink < 1e-8, f"φ_kink error too large: {err_kink:.2e}"

    print("\nAll NonlinearKANTauTheta verifications passed ✓")


if __name__ == "__main__":
    _verify_nonlinear_kan()
