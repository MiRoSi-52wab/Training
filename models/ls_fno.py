"""
LS-FNO: Lippmann-Schwinger Fourier Neural Operator.

Implements the analytic architecture from:
    Nguyen & Schneider (2025), "Universal Fourier Neural Operators for Micromechanics"
    arXiv:2507.12233v2

Architecture:  F_θ = P ∘ N^∘K ∘ E

    E (Embedding):  (ε̄, C)  →  (ε̄, T, ξ₀)       where ξ₀ = τ_θ(T, ε̄)
    N (FNO layer):  (ε̄, T, ξ) → (ε̄, T, ξ')       where ξ' = τ_θ(T, ε̄ − Γ:ξ)
    P (Projection): (ε̄, T, ξ) →  ε*               where ε* = ε̄ − Γ:ξ

Each FNO layer N is one Lippmann-Schwinger iteration; stacking K layers gives
K LS steps after the embedding, for a total of K+1 applications of Γ and τ_θ.

Internal state uses Mandel notation (required so that the tensor double-
contraction T : ε is an ordinary matrix-vector product, and Frobenius norms
are Euclidean — both properties are needed by the Yarotsky construction).
Inputs and outputs stay in Voigt notation with engineering shear, matching
the HDF5 dataset produced by generation/generate_dataset.py.

Replaceable component
---------------------
`YarotskyTauTheta` is the analytic (no parameters) double-contraction module.
In Phase 3, pass any `nn.Module` with the same forward signature to `LSFNO`
as the `tau_theta` argument to swap in a trainable KAN.

Requirements: PyTorch ≥ 1.7 (for torch.fft.fftn / torch.complex).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Yarotsky double-contraction operator  τ_θ(T, ε) ≈ T : ε
# ─────────────────────────────────────────────────────────────────────────────

class YarotskyTauTheta(nn.Module):
    """
    Analytic, parameter-free approximation of the tensor double-contraction.

    Given a normalised stiffness contrast T (a 3×3 matrix field in Mandel
    notation) and a strain field ε, computes ξ ≈ T : ε pointwise at every
    voxel using Yarotsky's ReLU construction:

        1.  q_θ(x) ≈ x²    on [-1, 1],  depth m,  error O(4^{-m})
        2.  m_θ(a, b) ≈ ab  for a, b ∈ [-M, M]
                = 2M² [q_θ((a+b)/(2M)) − q_θ(a/(2M)) − q_θ(b/(2M))]
                via the polarization identity ab = [(a+b)² − a² − b²]/2
        3.  r_θ(x) = clip(x, −M, M)
                = ReLU(x+M) − ReLU(x−M) − M
        4.  τ_θ(T, ε)_i = Σ_j m_θ(T_ij, r_θ(ε_j))   (matrix-vector product)

    Accuracy:  FNO7 (m=7): ~6×10⁻⁵,  FNO9 (m=9): ~4×10⁻⁶,  FNO11 (m=11): ~2×10⁻⁷.

    Small-strain note:  The calibration box is ‖ε‖ ≤ M.  If strain magnitudes
    are much smaller than M (e.g., eps_bar_scale=0.01 with M=1.0), accuracy
    degrades near zero — a known limitation the paper discusses in §5.2.4.
    Phase 3 (KAN replacement) addresses this directly.

    This class is the ONLY part that changes between Phase 2 and Phase 3.
    """

    def __init__(self, depth_m: int = 9, cutoff_M: float = 1.0):
        """
        Args:
            depth_m:  Depth of the Yarotsky square-function network.
                      Higher values give smaller approximation error (O(4^{-m})).
                      Practical choices: 7 (fast), 9 (default), 11 (accurate).
            cutoff_M: Strain clipping bound for the calibration box.
                      Match to the maximum expected ‖ε‖ in the dataset.
        """
        super().__init__()
        self.m = depth_m
        self.M = float(cutoff_M)

    # ── Yarotsky building blocks ──────────────────────────────────────────────

    def _q(self, x: torch.Tensor) -> torch.Tensor:
        """
        Approximate x² for x ∈ [-1, 1] using Yarotsky's recursive construction.

        Formula:  q_m(|x|) = |x| − Σ_{k=1}^m g^k(|x|) / 4^k
        where g is the tent function g(t) = 2t − 4·ReLU(t − 0.5) on [0, 1],
        and g^k denotes its k-fold composition.
        """
        t = F.relu(x) + F.relu(-x)         # t = |x|; maps [-1, 1] → [0, 1]
        out = t
        g = t
        for k in range(1, self.m + 1):
            g = 2.0 * g - 4.0 * F.relu(g - 0.5)   # tent function: [0,1] → [0,1]
            out = out - g / (4.0 ** k)
        return out

    def _r(self, x: torch.Tensor) -> torch.Tensor:
        """Clip x to [−M, M]: r(x) = ReLU(x+M) − ReLU(x−M) − M."""
        M = self.M
        return F.relu(x + M) - F.relu(x - M) - M

    def _m(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Approximate a·b for a, b ∈ [−M, M].
        Rescaled polarization identity: ab = 2M² [q((a+b)/2M) − q(a/2M) − q(b/2M)].
        All q inputs are in [−1, 1] so the Yarotsky construction is valid.
        """
        M = self.M
        return 2.0 * M ** 2 * (
            self._q((a + b) / (2.0 * M))
            - self._q(a / (2.0 * M))
            - self._q(b / (2.0 * M))
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, T: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """
        Pointwise double-contraction ξ_i = Σ_j m_θ(T_ij, r_θ(ε_j)).

        Args:
            T:   (B, n, n, *spatial) — normalised stiffness contrast in Mandel.
            eps: (B, n, *spatial)    — strain field in Mandel notation.

        Returns:
            xi:  (B, n, *spatial)    — polarisation stress ξ ≈ T : ε.

        Precision note:
            The Yarotsky product m_θ(a, b) computes q((a+b)/2M) − q(a/2M), a
            near-cancellation of two similar values when |b| ≪ M (e.g. 0.1%
            strain with M=1).  Float32 loses ~3 significant digits here, creating
            a noise floor that prevents convergence below ~1.5e-5.  The Yarotsky
            arithmetic is therefore performed in float64 and cast back to the
            caller's dtype on output.  The overhead is negligible — τ_θ is far
            cheaper than the FFT/Green-operator steps.
        """
        orig_dtype = T.dtype
        T   = T.double()
        eps = eps.double()

        # Clip strain components to calibration box [−M, M]
        eps_c = self._r(eps)

        # Expand ε_j along the row (i) axis to match T layout.
        # After expansion: eps_exp[b, i, j, ...] = eps_c[b, j, ...]
        eps_exp = eps_c.unsqueeze(1).expand_as(T)

        # Pairwise products: m_θ(T_ij, r_θ(ε_j)), then sum over j → ξ_i
        products = self._m(T, eps_exp)
        return products.sum(dim=2).to(orig_dtype)


# ─────────────────────────────────────────────────────────────────────────────
# LS-FNO
# ─────────────────────────────────────────────────────────────────────────────

class LSFNO(nn.Module):
    """
    Lippmann-Schwinger Fourier Neural Operator.

    Takes a Voigt stiffness field C(x) and a Voigt macroscopic strain ε̄ and
    returns the Voigt microscopic strain field ε*(x).

    Two operating modes — choose based on context:

    ``forward(C_field_V, eps_bar_V)``
        Runs exactly K iterations (fixed depth).  Use for training: the
        computation graph is predictable, memory usage is bounded, and
        gradients flow cleanly through the fixed loop.

    ``solve(C_field_V, eps_bar_V, tol, max_iter)``
        Runs until the polarisation-stress residual drops below ``tol``
        (or ``max_iter`` is reached).  Use for evaluation and the
        iteration-count study from §5.2.2 / Table 3 of the paper.
        Returns a dict with ``eps_star``, ``n_iter``, ``residuals``,
        and ``converged`` — matching the signature of ``fft_solver.solve()``.

    Internally the computation uses Mandel notation (required by the Yarotsky
    τ_θ construction and consistent with the paper's bounds).

    Args:
        grid_size:   Spatial grid size N.  The domain is N×N voxels.  The Green
                     operator is precomputed for this size and cached as a buffer.
        depth_K:     Number of Fourier neural layers used by ``forward()``.
        alpha_minus: Minimum eigenvalue of C(x) across the entire domain.
                     Together with alpha_plus determines α₀ = (α⁻ + α⁺)/2,
                     the optimal reference stiffness (Theorem 2.1).
        alpha_plus:  Maximum eigenvalue of C(x) across the entire domain.
        tol:         Convergence tolerance for ``solve()``.  Iteration stops
                     when the relative residual ‖ξ_k − ξ_{k−1}‖/‖ξ_k‖ < tol.
                     Default 1e-5 matches the paper's experiments (page 18).
        max_iter:    Hard cap on iterations for ``solve()``.
        tau_theta:   Replaceable τ_θ module (double-contraction operator).
                     Defaults to YarotskyTauTheta (analytic, no parameters).
                     Swap with a KAN module in Phase 3 — only this module
                     changes; the FNO layers and Green operator stay fixed.
    """

    def __init__(
        self,
        grid_size: int,
        depth_K: int = 4,
        alpha_minus: float = 1.0,
        alpha_plus: float = 10.0,
        tol: float = 1e-5,
        max_iter: int = 2000,
        tau_theta: Optional[nn.Module] = None,
        dim: int = 2,
        discretization: str = 'exact',
    ):
        super().__init__()
        assert dim in (2, 3), f"dim must be 2 or 3, got {dim}"
        assert discretization in ('exact', 'staggered'), \
            f"discretization must be 'exact' or 'staggered', got '{discretization}'"
        self.dim = dim
        self.n_comp = 3 if dim == 2 else 6
        self.n_normal = 2 if dim == 2 else 3
        self.K = depth_K
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.alpha_0 = float((alpha_minus + alpha_plus) / 2.0)
        self.discretization = discretization
        self.tau_theta = tau_theta if tau_theta is not None else YarotskyTauTheta()

        # Precompute the Mandel Green operator for this grid size.
        # Stored as a non-trainable buffer so it moves with the model to GPU.
        if dim == 2:
            gamma_hat = self._build_green_operator_mandel(grid_size, discretization)    # (3, 3, N, N)
        else:
            gamma_hat = self._build_green_operator_mandel_3d(grid_size, discretization) # (6, 6, N, N, N)
        self.register_buffer("gamma_hat_M", gamma_hat)

    # ── Green operator construction ───────────────────────────────────────────

    @staticmethod
    def _build_green_operator_mandel(N: int,
                                     discretization: str = 'exact') -> torch.Tensor:
        """
        Compute the Eshelby-Green operator Γ̂_M in Mandel notation, Fourier space.

        The operator maps polarisation stress τ_M → strain fluctuation ε_M via:
            (Γ̂_M : τ_M)_p = Σ_q Γ̂_M[p, q, kx, ky] · τ̂_M[q, kx, ky]

        Construction (three steps):
          1. 4-index tensor G_{ijkl}(ξ̂) from the Green formula:
               G = (δᵢₖξ̂ⱼξ̂ₗ + δᵢₗξ̂ⱼξ̂ₖ + δⱼₖξ̂ᵢξ̂ₗ + δⱼₗξ̂ᵢξ̂ₖ)/2 − ξ̂ᵢξ̂ⱼξ̂ₖξ̂ₗ
          2. Contract to Voigt 3×3 with engineering-shear symmetry factors
             (row and column factor 2 for the shear index, no 1/α₀):
               Γ_V[a, b] = G_{i(a)j(a)k(b)l(b)} × VF[a] × VF[b]
          3. Voigt → Mandel diagonal scaling (Green maps stress → strain):
               Γ_M[p, q] = (D_ε[p] / D_τ[q]) × Γ_V[p, q]
               D_ε = [1, 1, 1/√2]   (ε_M[2] = γ_V[2] / √2)
               D_τ = [1, 1, √2]     (τ_M[2] = τ_V[2] · √2)

        The 1/α₀ factor is NOT included here; it is absorbed into
        T = (C − C⁰)/α₀, the input to τ_θ.

        Args:
            discretization: 'exact' (standard Fourier) or 'staggered' (Willot 2015
                            rotated grid with effective frequencies).

        Returns:
            Tensor of shape (3, 3, N, N), dtype float32.
        """
        freq = np.fft.fftfreq(N)                               # in [−0.5, 0.5)
        xi_x, xi_y = np.meshgrid(freq, freq, indexing="ij")   # (N, N) each

        if discretization == 'staggered':
            xi_eff_x = np.sin(np.pi * xi_x) * np.cos(np.pi * xi_y)
            xi_eff_y = np.sin(np.pi * xi_y) * np.cos(np.pi * xi_x)
            xi = np.stack([xi_eff_x, xi_eff_y], axis=0)       # (2, N, N)
            xi_sq = xi[0]**2 + xi[1]**2
        else:
            xi = np.stack([xi_x, xi_y], axis=0)               # (2, N, N)
            xi_sq = xi_x**2 + xi_y**2                         # |ξ|²

        # Avoid division by zero at DC; the DC entry will be zeroed afterwards.
        safe = np.where(xi_sq == 0.0, 1.0, xi_sq)

        def xip(a, b):       # ξ_a · ξ_b / |ξ|²  =  ξ̂_a · ξ̂_b
            return xi[a] * xi[b] / safe

        def xip4(a, b, c, d):  # ξ̂_a · ξ̂_b · ξ̂_c · ξ̂_d
            return xi[a] * xi[b] * xi[c] * xi[d] / safe ** 2

        dlt = lambda a, b: float(a == b)

        # Voigt index map for 2D plane-strain:  0↔(0,0),  1↔(1,1),  2↔(0,1)
        VI = [0, 1, 0]
        VJ = [0, 1, 1]
        # Engineering-shear symmetry factors:
        #   row (output/strain): ×2 because γ₁₂ = 2ε₁₂
        #   col (input/stress) : ×2 because τ₁₂ = τ₂₁ (symmetry counting)
        VF = [1.0, 1.0, 2.0]

        Gamma_V = np.zeros((3, 3, N, N), dtype=np.float64)
        for a in range(3):
            i, j = VI[a], VJ[a]
            for b in range(3):
                k, l = VI[b], VJ[b]
                G = (
                    dlt(i, k) * xip(j, l)
                    + dlt(i, l) * xip(j, k)
                    + dlt(j, k) * xip(i, l)
                    + dlt(j, l) * xip(i, k)
                ) / 2.0 - xip4(i, j, k, l)
                Gamma_V[a, b] = G * VF[a] * VF[b]

        Gamma_V[:, :, 0, 0] = 0.0                             # Γ̂(0) = 0

        # Voigt → Mandel conversion for a stress→strain operator:
        #   Γ_M[p, q] = (D_ε[p] / D_τ[q]) × Γ_V[p, q]
        SQRT2 = np.sqrt(2.0)
        D_eps = np.array([1.0, 1.0, 1.0 / SQRT2])            # strain output factors
        D_tau = np.array([1.0, 1.0, SQRT2])                   # stress input factors
        scale = D_eps[:, None] / D_tau[None, :]               # (3, 3)
        Gamma_M = Gamma_V * scale[:, :, np.newaxis, np.newaxis]  # (3, 3, N, N)

        return torch.from_numpy(Gamma_M).float()

    @staticmethod
    def _build_green_operator_mandel_3d(N: int,
                                        discretization: str = 'exact') -> torch.Tensor:
        """
        Compute the Eshelby-Green operator Γ̂_M in Mandel notation for 3D, Fourier space.

        Same four-index tensor formula as the 2D version, extended to three frequency
        directions and six Voigt components.

        Voigt index map for 3D:  0↔(0,0), 1↔(1,1), 2↔(2,2), 3↔(1,2), 4↔(0,2), 5↔(0,1)
        Engineering-shear factors VF: [1,1,1,2,2,2]

        Voigt → Mandel scaling (stress→strain operator):
            D_ε = [1, 1, 1, 1/√2, 1/√2, 1/√2]
            D_τ = [1, 1, 1, √2,   √2,   √2  ]
            Γ_M[p, q] = (D_ε[p] / D_τ[q]) × Γ_V[p, q]

        Args:
            discretization: 'exact' (standard Fourier) or 'staggered' (Willot 2015
                            rotated grid with effective frequencies).

        Returns:
            Tensor of shape (6, 6, N, N, N), dtype float32.
        """
        freq = np.fft.fftfreq(N)
        xi_x, xi_y, xi_z = np.meshgrid(freq, freq, freq, indexing="ij")

        if discretization == 'staggered':
            xi_eff_x = np.sin(np.pi * xi_x) * np.cos(np.pi * xi_y) * np.cos(np.pi * xi_z)
            xi_eff_y = np.sin(np.pi * xi_y) * np.cos(np.pi * xi_x) * np.cos(np.pi * xi_z)
            xi_eff_z = np.sin(np.pi * xi_z) * np.cos(np.pi * xi_x) * np.cos(np.pi * xi_y)
            xi = np.stack([xi_eff_x, xi_eff_y, xi_eff_z], axis=0)   # (3, N, N, N)
            xi_sq = xi[0]**2 + xi[1]**2 + xi[2]**2
        else:
            xi = np.stack([xi_x, xi_y, xi_z], axis=0)   # (3, N, N, N)
            xi_sq = xi_x**2 + xi_y**2 + xi_z**2

        safe = np.where(xi_sq == 0.0, 1.0, xi_sq)

        def xip(a, b):         return xi[a] * xi[b] / safe
        def xip4(a, b, c, d):  return xi[a] * xi[b] * xi[c] * xi[d] / safe**2
        dlt = lambda a, b: float(a == b)

        VI = [0, 1, 2, 1, 0, 0]
        VJ = [0, 1, 2, 2, 2, 1]
        VF = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]

        Gamma_V = np.zeros((6, 6, N, N, N), dtype=np.float64)
        for a in range(6):
            i, j = VI[a], VJ[a]
            for b in range(6):
                k, l = VI[b], VJ[b]
                G = (
                    dlt(i, k) * xip(j, l)
                    + dlt(i, l) * xip(j, k)
                    + dlt(j, k) * xip(i, l)
                    + dlt(j, l) * xip(i, k)
                ) / 2.0 - xip4(i, j, k, l)
                Gamma_V[a, b] = G * VF[a] * VF[b]

        Gamma_V[:, :, 0, 0, 0] = 0.0  # Γ̂(0) = 0

        SQRT2 = np.sqrt(2.0)
        D_eps = np.array([1.0, 1.0, 1.0, 1.0/SQRT2, 1.0/SQRT2, 1.0/SQRT2])
        D_tau = np.array([1.0, 1.0, 1.0, SQRT2,      SQRT2,      SQRT2    ])
        scale = D_eps[:, None] / D_tau[None, :]                             # (6, 6)
        Gamma_M = Gamma_V * scale[:, :, np.newaxis, np.newaxis, np.newaxis] # (6,6,N,N,N)

        return torch.from_numpy(Gamma_M).float()

    # ── Fourier-space Green-operator application ──────────────────────────────

    def _apply_green(self, xi_M: torch.Tensor) -> torch.Tensor:
        """
        Compute Γ : ξ in Fourier space (all in Mandel notation).

            FFT(ξ)  →  Γ̂_M · ξ̂_M  →  iFFT  →  real part

        Args:
            xi_M: (B, n_comp, *spatial) — polarisation stress field in Mandel notation.

        Returns:
            (B, n_comp, *spatial) — strain fluctuation Γ:ξ in Mandel notation.
        """
        fft_dims = tuple(range(-self.dim, 0))
        einsum_str = "pqxy, bqxy -> bpxy" if self.dim == 2 else "pqxyz, bqxyz -> bpxyz"

        xi_hat = torch.fft.fftn(xi_M, dim=fft_dims)
        # Γ̂_M is real; multiply real and imaginary parts separately.
        r  = torch.einsum(einsum_str, self.gamma_hat_M, xi_hat.real)
        im = torch.einsum(einsum_str, self.gamma_hat_M, xi_hat.imag)
        return torch.fft.ifftn(torch.complex(r, im), dim=fft_dims).real

    # ── Voigt ↔ Mandel helpers ────────────────────────────────────────────────

    def _voigt_stiffness_to_mandel(self, C_V: torch.Tensor) -> torch.Tensor:
        """
        C_M = D · C_V · D  with D = diag(1,...,1, √2,...,√2).

        C_V: (B, n_comp, n_comp, *spatial) in Voigt notation.
        Returns: same shape in Mandel notation.
        """
        SQRT2 = 2.0 ** 0.5
        D = torch.ones(self.n_comp, dtype=C_V.dtype, device=C_V.device)
        D[self.n_normal:] = SQRT2
        D_outer = D[:, None] * D[None, :]                     # (n_comp, n_comp)
        shape = (1, self.n_comp, self.n_comp) + (1,) * self.dim
        return C_V * D_outer.reshape(shape)

    def _mandel_strain_to_voigt(self, eps_M: torch.Tensor) -> torch.Tensor:
        """
        ε_V[n_normal:] = ε_M[n_normal:] · √2   (Mandel → Voigt engineering shear).

        eps_M: (B, n_comp, *spatial) Mandel strain field.
        Returns: same shape Voigt strain field.
        """
        SQRT2 = 2.0 ** 0.5
        out = eps_M.clone()
        out[:, self.n_normal:] = eps_M[:, self.n_normal:] * SQRT2
        return out

    # ── Shared input preparation ──────────────────────────────────────────────

    def _to_mandel_state(
        self, C_field_V: torch.Tensor, eps_bar_V: torch.Tensor
    ):
        """
        Convert Voigt inputs to Mandel internal state for the LS iteration.

        Returns:
            T_M:           (B, n_comp, n_comp, *spatial) normalised stiffness contrast
                           T = (C_M − C⁰_M)/α₀
            eps_bar_field: (B, n_comp, *spatial) macroscopic strain broadcast to field shape
        """
        B = C_field_V.shape[0]
        N = C_field_V.shape[-1]
        assert N == self.gamma_hat_M.shape[-1], (
            f"Grid size mismatch: model built for N={self.gamma_hat_M.shape[-1]}, "
            f"got N={N}. Rebuild the model with the correct grid_size."
        )

        SQRT2 = 2.0 ** 0.5
        n = self.n_comp
        C_field_M = self._voigt_stiffness_to_mandel(C_field_V)
        eps_bar_M = eps_bar_V.clone()
        eps_bar_M[:, self.n_normal:] = eps_bar_M[:, self.n_normal:] / SQRT2
        alpha_0 = self.alpha_0
        C0_M = torch.eye(n, dtype=C_field_M.dtype, device=C_field_M.device) * alpha_0
        C0_M_bc = C0_M.reshape(1, n, n, *([1] * self.dim))
        T_M = (C_field_M - C0_M_bc) / alpha_0
        eps_bar_field = eps_bar_M.reshape(B, n, *([1] * self.dim)).expand(
            B, n, *([N] * self.dim)
        )
        return T_M, eps_bar_field

    # ── Forward pass (fixed depth, for training) ──────────────────────────────

    def _ls_layer(
        self, xi_M: torch.Tensor, T_M: torch.Tensor, eps_bar_field: torch.Tensor
    ) -> torch.Tensor:
        """One LS iteration: ξ_{k-1} ↦ ξ_k = τ_θ(T, ε̄ − Γ:ξ_{k-1}). Factored out
        of forward() so it can be wrapped in torch.utils.checkpoint.checkpoint."""
        gamma_xi = self._apply_green(xi_M)                       # Γ:ξ_{k-1}
        eps_M = eps_bar_field - gamma_xi                         # ε_k = ε̄ − Γ:ξ_{k-1}
        return self.tau_theta(T_M, eps_M)                        # ξ_k = τ_θ(T, ε_k)

    def forward(
        self, C_field_V: torch.Tensor, eps_bar_V: torch.Tensor,
        use_checkpointing: bool = False,
    ) -> torch.Tensor:
        """
        Run exactly K LS iterations and return the microscopic strain field ε*(x).

        Use for training: fixed depth gives a predictable computation graph,
        bounded memory, and clean gradient flow.

        Architecture:
            Embedding:    ξ₀ = τ_θ(T, broadcast(ε̄))
            Layer k=1..K: ε_k = ε̄ − Γ:ξ_{k−1};  ξ_k = τ_θ(T, ε_k)
            Projection:   ε* = ε̄ − Γ:ξ_K

        Args:
            C_field_V: (B, n_comp, n_comp, *spatial) stiffness field in Voigt notation.
            eps_bar_V: (B, n_comp) macroscopic strain in Voigt (engineering shear).
            use_checkpointing: if True, recompute each LS layer's activations
                during the backward pass (torch.utils.checkpoint) instead of
                storing them — trades ~30% more FLOPs for much lower training
                memory when K is large. No effect on the result. Off by default
                so existing callers (study/, replicate/, unittests/) are unaffected.

        Returns:
            eps_star_V: (B, n_comp, *spatial) strain field ε*(x) in Voigt notation.
        """
        T_M, eps_bar_field = self._to_mandel_state(C_field_V, eps_bar_V)

        xi_M = self.tau_theta(T_M, eps_bar_field)                # embedding: ξ₀

        for _ in range(self.K):
            if use_checkpointing and self.training:
                xi_M = checkpoint(self._ls_layer, xi_M, T_M, eps_bar_field,
                                  use_reentrant=False)
            else:
                xi_M = self._ls_layer(xi_M, T_M, eps_bar_field)

        gamma_xi = self._apply_green(xi_M)
        eps_star_M = eps_bar_field - gamma_xi                    # projection: ε* = ε̄ − Γ:ξ_K

        return self._mandel_strain_to_voigt(eps_star_M)          # (B, 3, N, N)

    # ── Solve (dynamic depth, for evaluation) ─────────────────────────────────

    def solve(
        self,
        C_field_V: torch.Tensor,
        eps_bar_V: torch.Tensor,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        verbose: bool = False,
    ) -> dict:
        """
        Run LS iterations until convergence and return a result dict.

        Use for the iteration-count study (§5.2.2 / Table 3 of the paper) and
        for side-by-side comparison with fft_solver.solve().  The output dict
        intentionally mirrors fft_solver.solve() so callers can swap between
        the two without code changes.

        Convergence criterion (same structure as fft_solver.py):
            res_k = ‖ξ_k − ξ_{k−1}‖_F / ‖ξ_k‖_F  <  tol

        Args:
            C_field_V: (B, n_comp, n_comp, *spatial) Voigt stiffness field.
            eps_bar_V: (B, n_comp) macroscopic strain in Voigt.
            tol:       Convergence tolerance (default: self.tol = 1e-5).
            max_iter:  Hard cap on iterations (default: self.max_iter = 2000).
            verbose:   If True, print iteration/residual progress every 50 steps.

        Returns:
            dict with:
                'eps_star'  : (B, n_comp, *spatial) Voigt strain field
                'n_iter'    : int — iterations performed
                'residuals' : list[float] — relative residual per iteration
                'converged' : bool — True if tol was reached before max_iter
        """
        _tol = self.tol if tol is None else float(tol)
        _max_iter = self.max_iter if max_iter is None else int(max_iter)

        T_M, eps_bar_field = self._to_mandel_state(C_field_V, eps_bar_V)

        xi_M = self.tau_theta(T_M, eps_bar_field)   # embedding: ξ₀

        residuals: list = []
        converged = False

        for iteration in range(_max_iter):
            xi_prev = xi_M                           # no clone needed; tau_theta returns a new tensor

            gamma_xi = self._apply_green(xi_M)
            eps_M = eps_bar_field - gamma_xi
            xi_M = self.tau_theta(T_M, eps_M)       # new tensor; xi_prev still holds previous value

            xi_norm = xi_M.norm()
            res = float((xi_M - xi_prev).norm() / xi_norm) if xi_norm > 0 else 0.0
            residuals.append(res)

            if res < _tol:
                converged = True
                if verbose == True:
                    print(f"----- FNO converged in {len(residuals)} iterations! -----")
                break

            if verbose and iteration % 50 == 0:
                print(f"Iteration LS-FNO: {iteration}/{_max_iter}  residual: {res:.3e}")

        gamma_xi = self._apply_green(xi_M)
        eps_star_M = eps_bar_field - gamma_xi

        return {
            "eps_star": self._mandel_strain_to_voigt(eps_star_M),
            "n_iter": len(residuals),
            "residuals": residuals,
            "converged": converged,
        }

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls, cfg: dict, tau_theta: Optional[nn.Module] = None
    ) -> "LSFNO":
        """
        Instantiate from a config dict loaded via utils.config_loader.load_config().

        Alpha bounds are derived from material parameters (E_matrix, nu_matrix,
        nu_inclusion, kappa) so they never go stale when the experiment changes.
        Explicit ``alpha_minus`` / ``alpha_plus`` keys are accepted as an override
        for backwards compatibility or for non-standard materials.

        Grid size is read from ``N`` (experiment convention) with ``grid_size``
        as a fallback (old single-file convention).
        """
        # ── Alpha bounds (derived, not stored in config) ──────────────────────
        if "alpha_minus" in cfg and "alpha_plus" in cfg:
            alpha_minus = float(cfg["alpha_minus"])
            alpha_plus  = float(cfg["alpha_plus"])
        elif all(k in cfg for k in ("E_matrix", "nu_matrix", "kappa")):
            from utils.config_loader import compute_alpha_bounds
            alpha_minus, alpha_plus = compute_alpha_bounds(
                cfg["E_matrix"],
                cfg.get("nu_matrix",    0.3),
                cfg.get("nu_inclusion", cfg.get("nu_matrix", 0.3)),
                cfg["kappa"],
                dim=int(cfg.get("dim", 2)),
            )
        else:
            raise ValueError(
                "from_config() requires either explicit 'alpha_minus'/'alpha_plus' keys "
                "or material parameters 'E_matrix', 'nu_matrix', 'kappa'."
            )

        # ── Grid size: experiment.yaml uses 'N'; legacy single-file used 'grid_size' ──
        grid_size_val = cfg.get("N", cfg.get("grid_size"))
        if grid_size_val is None:
            raise ValueError("from_config() requires 'N' or 'grid_size' in config.")
        grid_size = int(grid_size_val)

        tau = tau_theta or YarotskyTauTheta(
            depth_m=cfg.get("m", 9),
            cutoff_M=cfg.get("M", 1.0),
        )
        return cls(
            grid_size=grid_size,
            depth_K=cfg.get("K", 4),
            alpha_minus=alpha_minus,
            alpha_plus=alpha_plus,
            tol=cfg.get("tol", 1e-5),
            max_iter=cfg.get("max_iter", 2000),
            tau_theta=tau,
            dim=int(cfg.get("dim", 2)),
            discretization=str(cfg.get("discretization", "exact")),
        )

    @classmethod
    def from_data(
        cls,
        C_field_V: torch.Tensor,
        depth_K: int = 4,
        depth_m: int = 9,
        cutoff_M: float = 1.0,
        tau_theta: Optional[nn.Module] = None,
        discretization: str = 'exact',
    ) -> "LSFNO":
        """
        Instantiate with α₀ estimated from the C₁₁₁₁ component of a stiffness field.

        Uses the same heuristic as fft_solver.py:
            α₀ = (max(C₁₁₁₁) + min(C₁₁₁₁)) / 2.

        Dimensionality is inferred from shape:
            2D single (3,3,N,N), 2D batch (B,3,3,N,N),
            3D single (6,6,N,N,N), 3D batch (B,6,6,N,N,N).

        The distinguishing rule: a *single sample* has shape[0] == shape[1] == n_comp
        (3 or 6); a *batch* has shape[1] == shape[2] == n_comp.
        """
        s = C_field_V.shape
        # Single-sample: first two dims are equal and equal to n_comp (3 or 6)
        if s[0] == s[1] and s[0] in (3, 6):
            n_comp = s[0]
            N = s[-1]
            c11 = C_field_V[0, 0]
        # Batch: second and third dims are equal and equal to n_comp
        elif len(s) >= 4 and s[1] == s[2] and s[1] in (3, 6):
            n_comp = s[1]
            N = s[-1]
            c11 = C_field_V[:, 0, 0]
        else:
            raise ValueError(
                f"C_field_V shape {s} not recognised. Expected "
                f"(3,3,N,N), (B,3,3,N,N), (6,6,N,N,N), or (B,6,6,N,N,N)."
            )
        dim = 2 if n_comp == 3 else 3
        alpha_plus  = float(c11.max().item())
        alpha_minus = float(c11.min().item())
        tau = tau_theta or YarotskyTauTheta(depth_m=depth_m, cutoff_M=cutoff_M)
        return cls(
            grid_size=N,
            depth_K=depth_K,
            alpha_minus=alpha_minus,
            alpha_plus=alpha_plus,
            tau_theta=tau,
            dim=dim,
            discretization=discretization,
        )

    # ── Inspection helpers ────────────────────────────────────────────────────

    def n_params(self) -> int:
        """Number of trainable parameters (0 for the analytic construction)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"LSFNO(dim={self.dim}, K={self.K}, tol={self.tol:.1e}, α₀={self.alpha_0:.4g}, "
            f"N={self.gamma_hat_M.shape[-1]}, disc={self.discretization}, "
            f"τ_θ={self.tau_theta.__class__.__name__}, "
            f"params={self.n_params()})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity-check  (run with:  python -m models.ls_fno)
# ─────────────────────────────────────────────────────────────────────────────

def _self_test() -> None:
    """
    Verify the LS-FNO output against the NumPy FFT solver on a toy problem.

    Uses a homogeneous C_field (C(x) = const) where the exact answer is
    ε*(x) = ε̄ everywhere (no fluctuations), so any solver should reproduce ε̄.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from generation.fft_solver import solve as fft_solve

    torch.set_grad_enabled(False)

    N = 16
    B = 2
    E, nu = 1.0, 0.3
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    C_voigt = np.array([
        [lam + 2 * mu, lam,           0.0],
        [lam,           lam + 2 * mu, 0.0],
        [0.0,           0.0,          mu],
    ])

    # Homogeneous field
    C_np = np.broadcast_to(C_voigt[:, :, None, None], (3, 3, N, N)).copy()
    eps_bar_np = np.array([0.001, -0.0005, 0.0003])

    # FFT solver reference
    ref = fft_solve(C_np, eps_bar_np)
    eps_ref = ref["eps_star"]   # (3, N, N)

    # LS-FNO
    alpha_plus = float(C_np[0, 0].max())
    alpha_minus = float(C_np[0, 0].min())
    model = LSFNO(
        grid_size=N,
        depth_K=10,
        alpha_minus=alpha_minus,
        alpha_plus=alpha_plus,
        tau_theta=YarotskyTauTheta(depth_m=9, cutoff_M=1.0),
    )
    print(model)

    # Build batched tensors
    C_t = torch.from_numpy(C_np).float().unsqueeze(0).expand(B, -1, -1, -1, -1)
    eb_t = torch.from_numpy(eps_bar_np).float().unsqueeze(0).expand(B, -1)

    eps_pred = model(C_t, eb_t)    # (B, 3, N, N)

    # For a homogeneous field, ε*(x) = ε̄ everywhere (no fluctuations).
    eps_ref_t = torch.from_numpy(eps_ref).float()
    rel_err = (eps_pred[0] - eps_ref_t).norm() / (eps_ref_t.norm() + 1e-12)
    print(f"Homogeneous test: rel_err = {rel_err:.4e}")

    assert rel_err < 0.05, f"Relative error too large: {rel_err:.4e}"
    print("ls_fno.py self-test passed.")


if __name__ == "__main__":
    _self_test()
