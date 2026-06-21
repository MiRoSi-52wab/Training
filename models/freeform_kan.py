"""
FreeFormKANTauTheta: a generic multi-layer B-spline KAN that learns
τ(T, ε) = T:ε from data, with NO hard-coded polarization-identity scaffold.

This is the model for FREEFORM_KAN_CONTRACTION_DISCOVERY.md.  It is a drop-in
replacement for KANTauTheta: same forward(T, eps) → xi signature, same float64
internals, works inside LSFNO unchanged.

Architecture
------------
Input side:
  • Extract upper-triangular T (n×n) per voxel → n(n+1)/2 features
    (T is symmetric in Mandel notation, so only 6 independent values for n=3)
  • Flatten ε  (n)   per voxel → n   features
  • Concatenate: n(n+1)/2 + n features total  (9 for n=3, vs 12 before)
  • Optional element-wise scale (eps_input_scale) to bring ε to the same
    magnitude as T before the B-spline grid.
Core:
  • Stack of BSplineKANLayers: [n(n+1)/2+n] → [width₁] → ... → [n]
  • Each layer uses degree-k B-splines + a SiLU residual per (in→out) edge
Output:
  • Reshape → (B, n, *spatial): polarization stress ξ = τ_θ(T, ε)

Parameter count (n_comp=3, G=5, k=3, depth-1 width=[18]):
  Layer 1  (9→18): 18×9×8 spline + 18×9 base = 1296 + 162 = 1,458
  Layer 2  (18→3):  3×18×8 spline +  3×18 base =  432 +  54 =   486
  Total: 1,944 parameters
  (cf. old 12-input version: 2,430 params)

Theoretical minimum hidden width for 1-hidden-layer (depth-2 KAN) to represent
the full T:ε contraction via the polarization identity: 18 neurons.
Reasoning: n_comp=3 with symmetric T has 9 unique scalar products; each
needs 2 hidden neurons (sum + difference) → 9×2 = 18.
With 9 inputs the active-edge fraction in L1 is 36/162 = 22%,
vs 18/216 = 8% with the old 12-input version — easier to find by optimisation.

References
----------
FREEFORM_KAN_CONTRACTION_DISCOVERY.md
Liu et al. "KAN: Kolmogorov-Arnold Networks" (2024), arXiv:2404.19756
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# B-spline utility
# ─────────────────────────────────────────────────────────────────────────────

def _make_extended_grid(
    x_min: float,
    x_max: float,
    G: int,
    k: int,
    n_features: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build the extended (clamped) B-spline knot vector for G intervals and
    degree k.  Returns shape (n_features, G+2k+1): each feature row holds the
    same uniform extended grid.

    The first k knots are at x_min - k·h, ..., x_min (extended left),
    the last k knots are at x_max, ..., x_max + k·h (extended right),
    giving clamped B-splines that are nonzero at the boundary.
    """
    h = (x_max - x_min) / G
    grid_1d = torch.linspace(x_min - k * h, x_max + k * h, G + 2 * k + 1,
                              device=device, dtype=dtype)
    return grid_1d.unsqueeze(0).expand(n_features, -1).contiguous()


def _b_spline_basis(
    x: torch.Tensor,
    grid: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    Vectorized Cox–de Boor B-spline basis evaluation.

    Args:
        x:    (N, F) float64 — N samples, F features; each feature has its own grid.
        grid: (F, G+2k+1)   — extended knot vectors.
        k:    spline degree.

    Returns:
        B: (N, F, G+k) — B-spline basis values, one per edge.

    The B-splines are degree-k piecewise polynomials on the G interior
    intervals [grid[k], grid[G+k]].  Inputs are clamped before calling.
    """
    # Degree-0 indicator: B_{i,0} = 1  iff  grid[i] ≤ x < grid[i+1]
    # x_exp: (N, F, 1), g: (1, F, G+2k+1) → broadcast to (N, F, G+2k)
    x_exp = x.unsqueeze(-1)
    g = grid.unsqueeze(0)
    B = ((x_exp >= g[:, :, :-1]) & (x_exp < g[:, :, 1:])).to(x.dtype)
    # B: (N, F, G+2k)

    for d in range(1, k + 1):
        M = B.shape[-1]           # G+2k-d+1  (number of basis functions entering this step)
        # Slices from the extended grid, each shape (1, F, M-1):
        t_i   = g[:, :, :M - 1]          # t_0 … t_{M-2}
        t_id  = g[:, :, d:d + M - 1]     # t_d … t_{d+M-2}
        t_id1 = g[:, :, d + 1:d + M]     # t_{d+1} … t_{d+M-1}
        t_i1  = g[:, :, 1:M]             # t_1 … t_{M-1}

        denom_l = (t_id  - t_i ).clamp(min=1e-8)
        denom_r = (t_id1 - t_i1).clamp(min=1e-8)

        # B_{i,d-1}: columns 0..M-2 of current B → B[:, :, :M-1]
        # B_{i+1,d-1}: columns 1..M-1             → B[:, :, 1:]
        left  = (x_exp - t_i)   / denom_l * B[:, :, :M - 1]
        right = (t_id1 - x_exp) / denom_r * B[:, :, 1:]
        B = left + right           # (N, F, M-1)  i.e. one fewer column

    # Final shape: (N, F, G+k)
    return B


# ─────────────────────────────────────────────────────────────────────────────
# KAN Layer
# ─────────────────────────────────────────────────────────────────────────────

class BSplineKANLayer(nn.Module):
    """
    One KAN layer: n_in → n_out using B-spline activations + SiLU residual.

    For each (input i, output j) pair there is one learnable B-spline φ_{ij}:
        y_j = Σ_i [ φ_{ij}(x_i) + w_{ij} · SiLU(x_i) ]

    The residual term  w_{ij}·SiLU(x_i)  (from pykan) provides a nearly-linear
    initialisation and helps gradients flow early in training.

    Args:
        n_in, n_out: layer dimensions.
        grid_size:   G — number of B-spline intervals.
        k:           spline degree (3 = cubic).
        x_range:     domain used to build the knot vector; inputs outside this
                     range are clamped before evaluation.
        noise_scale: std of normal init for spline_weight (small → near-linear start).
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        grid_size: int = 5,
        k: int = 3,
        x_range: Tuple[float, float] = (-2.0, 2.0),
        noise_scale: float = 0.05,
    ):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.grid_size = grid_size
        self.k = k
        self.x_min = x_range[0]
        self.x_max = x_range[1]

        n_coeffs = grid_size + k   # basis functions per edge

        # Extended grid: (n_in, G+2k+1), registered so it moves with the model
        grid = _make_extended_grid(
            x_range[0], x_range[1], grid_size, k, n_in,
            device=torch.device("cpu"), dtype=torch.float64,
        )
        self.register_buffer("grid", grid)

        # Spline weights:  (n_out, n_in, G+k)
        self.spline_weight = nn.Parameter(
            torch.zeros(n_out, n_in, n_coeffs, dtype=torch.float64)
        )
        nn.init.normal_(self.spline_weight, mean=0.0, std=noise_scale)

        # Residual (SiLU) weights:  (n_out, n_in)
        self.base_weight = nn.Parameter(
            torch.empty(n_out, n_in, dtype=torch.float64)
        )
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, n_in) float64
        Returns: (N, n_out) float64
        """
        # Clamp to grid interior
        clamp_lo = self.grid[:, self.k].min()
        clamp_hi = self.grid[:, self.grid_size + self.k].min() - 1e-7
        x_c = x.clamp(clamp_lo, clamp_hi)

        # B-spline basis: (N, n_in, G+k)
        basis = _b_spline_basis(x_c, self.grid, self.k)

        # Spline contribution:  Σ_{i,k} basis[n,i,k] * weight[o,i,k]
        spline_out = torch.einsum("nik,oik->no", basis, self.spline_weight)

        # SiLU residual contribution: W @ SiLU(x)
        base_out = F.linear(F.silu(x), self.base_weight)

        return spline_out + base_out

    def n_params(self) -> int:
        return self.spline_weight.numel() + self.base_weight.numel()


# ─────────────────────────────────────────────────────────────────────────────
# Free-form KAN τ_θ
# ─────────────────────────────────────────────────────────────────────────────

class FreeFormKANTauTheta(nn.Module):
    """
    Drop-in replacement for KANTauTheta that learns τ(T, ε) = T:ε without the
    polarization-identity scaffold.

    The model only sees raw (T, ε) values per voxel — it has no hint that the
    target function is bilinear, no hardcoded s=(T+ε)/2R argument, and no
    prescribed combination formula.  Whether it discovers the bilinear structure
    is the empirical question of the FREEFORM_KAN_CONTRACTION_DISCOVERY study.

    Args:
        n_comp:          Mandel strain components: 3 for 2D, 6 for 3D.
        width:           Hidden layer widths.  E.g. [18] for depth-1 [12,18,3],
                         [18,12] for depth-2 [12,18,12,3].  input_dim and
                         output_dim are appended automatically.
        grid_size:       G — B-spline grid intervals per layer (default 5).
        k:               Spline degree (default 3 = cubic).
        x_range:         Domain for B-spline knots.  Should comfortably contain
                         all input values after eps_input_scale is applied.
        eps_input_scale: Pre-scale the ε features by this factor before the
                         first KAN layer.  With eps_bar_scale=0.01 and T~0.9,
                         setting eps_input_scale=100 brings both to the same
                         order of magnitude — makes better use of the B-spline
                         grid.  Set to 1.0 to disable.  This is a fixed
                         (non-trainable) normalisation, not a structural prior.
    """

    def __init__(
        self,
        n_comp: int = 3,
        width: List[int] = None,
        grid_size: int = 5,
        k: int = 3,
        x_range: Tuple[float, float] = (-2.0, 2.0),
        eps_input_scale: float = 100.0,
    ):
        super().__init__()
        self.n_comp = n_comp
        self.grid_size = grid_size
        self.k = k
        self.eps_input_scale = eps_input_scale

        # Upper-triangular indices of the n×n T matrix (row-major).
        # T is symmetric in Mandel notation, so we only feed the 6 independent
        # components (for n=3) rather than all 9.
        n = n_comp
        triu_idx = torch.tensor(
            [i * n + j for i in range(n) for j in range(i, n)],
            dtype=torch.long,
        )
        self.register_buffer("_triu_idx", triu_idx)

        if width is None:
            # Default depth-1: 18 = theoretical minimum for n_comp=3
            width = [2 * n_comp * (n_comp + 1) // 2]  # 2 × n_T_features

        input_dim  = n_comp * (n_comp + 1) // 2 + n_comp  # 6+3=9 for n=3
        output_dim = n_comp

        dims = [input_dim] + list(width) + [output_dim]
        self._dims = dims

        self.layers = nn.ModuleList([
            BSplineKANLayer(
                dims[i], dims[i + 1],
                grid_size=grid_size, k=k, x_range=x_range,
            )
            for i in range(len(dims) - 1)
        ])

    # ── helpers ──────────────────────────────────────────────────────────────

    @property
    def n_T_features(self) -> int:
        """Number of independent T components fed to the KAN (upper triangle)."""
        return self.n_comp * (self.n_comp + 1) // 2

    def _flatten_inputs(
        self, T: torch.Tensor, eps: torch.Tensor
    ) -> Tuple[torch.Tensor, int, torch.Size]:
        """
        Flatten T and ε over spatial dims, scale ε, and concatenate.

        T:   (B, n, n, *sp)  → T_triu: (B*N_sp, n(n+1)/2)   [upper triangle only]
        eps: (B, n, *sp)     → e_flat: (B*N_sp, n)           [× eps_input_scale]

        Returns (x, B, spatial) where x is (B*N_sp, n(n+1)/2 + n).
        T is symmetric in Mandel notation, so feeding only the 6 independent
        components (for n=3) avoids duplicating T₀₁/T₁₀, T₀₂/T₂₀, T₁₂/T₂₁.
        """
        B = T.shape[0]
        n = T.shape[1]
        spatial = T.shape[3:]
        N_sp = 1
        for s in spatial:
            N_sp *= s

        # Full flatten then select upper-triangular indices
        T_full = (
            T.reshape(B, n * n, N_sp)
             .permute(0, 2, 1)
             .reshape(B * N_sp, n * n)
        )
        T_triu = T_full[:, self._triu_idx]   # (B*N_sp, n(n+1)/2)

        e_flat = (
            eps.reshape(B, n, N_sp)
               .permute(0, 2, 1)
               .reshape(B * N_sp, n)
        )
        if self.eps_input_scale != 1.0:
            e_flat = e_flat * self.eps_input_scale

        x = torch.cat([T_triu, e_flat], dim=-1)   # (B*N_sp, n(n+1)/2 + n)
        return x, B, spatial

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, T: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """
        T:   (B, n, n, *spatial) — normalised stiffness contrast (Mandel)
        eps: (B, n, *spatial)    — strain field (Mandel)
        Returns xi: (B, n, *spatial) — learned polarization stress
        """
        orig_dtype = T.dtype
        T   = T.double()
        eps = eps.double()

        x, B, spatial = self._flatten_inputs(T, eps)
        N_sp = x.shape[0] // B
        n    = self.n_comp

        for layer in self.layers:
            x = layer(x)          # (B*N_sp, n)

        xi = (
            x.reshape(B, N_sp, n)
             .permute(0, 2, 1)
             .reshape(B, n, *spatial)
        )
        return xi.to(orig_dtype)

    # ── sparsity regularization ───────────────────────────────────────────────

    def sparsity_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        L1 regularization on edge function magnitudes (pykan Eq. 3.5).

        For every edge (i→j) in every layer, computes the mean absolute value
        of φᵢⱼ(xₙ) over the batch, then sums across all edges and layers:

            L_sparse = Σₗ Σᵢⱼ  (1/N Σₙ |φᵢⱼ⁽ˡ⁾(xₙ)|)

        This promotes group sparsity: edges that do not contribute to the task
        loss are pushed to φ ≡ 0 entirely (the L1 norm has constant gradient
        at any non-zero magnitude, unlike L2 which has diminishing gradient).

        Args:
            x: (N, input_dim) pre-flattened KAN input — T_triu concatenated
               with scaled ε.  Obtain via  x, _, _ = model._flatten_inputs(T, eps).
               Subsample to ~1024 voxels before calling for efficiency.

        Returns:
            Scalar tensor on the model's device / dtype.

        Usage in training loop::

            loss_task   = contraction_loss(xi_pred, xi_true)
            x_kan, _, _ = model._flatten_inputs(T_M, eps_M)
            idx          = torch.randperm(x_kan.shape[0])[:1024]
            loss_sparse  = model.sparsity_loss(x_kan[idx].detach())
            loss = loss_task + sparsity_lambda * loss_sparse
        """
        x = x.double()
        reg = x.new_zeros(())
        h   = x
        for layer in self.layers:
            clamp_lo = layer.grid[:, layer.k].min()
            clamp_hi = layer.grid[:, layer.grid_size + layer.k].min() - 1e-7
            h_c   = h.clamp(clamp_lo, clamp_hi)
            basis = _b_spline_basis(h_c, layer.grid, layer.k)  # (N, n_in, G+k)

            # Per-edge contributions: (N, n_out, n_in)
            per_edge = (
                torch.einsum("nik,oik->noi", basis, layer.spline_weight)
                + layer.base_weight.unsqueeze(0) * F.silu(h).unsqueeze(1)
            )
            reg = reg + per_edge.abs().mean(0).sum()   # mean over N, sum over edges
            h   = layer(h)                             # propagate to next layer
        return reg

    # ── diagnostics ──────────────────────────────────────────────────────────

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def layer_param_counts(self) -> List[int]:
        return [layer.n_params() for layer in self.layers]

    def __repr__(self) -> str:
        return (
            f"FreeFormKANTauTheta("
            f"dims={self._dims}, G={self.grid_size}, k={self.k}, "
            f"eps_scale={self.eps_input_scale}, "
            f"n_params={self.n_params():,})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Mandel conversion helpers (used by the training notebook)
# ─────────────────────────────────────────────────────────────────────────────

SQRT2 = 2.0 ** 0.5


def voigt_stiffness_to_mandel(C_V: torch.Tensor, n_normal: int = 2) -> torch.Tensor:
    """
    C_V: (B, n, n, *sp) Voigt stiffness  →  C_M Mandel stiffness (same shape).
    C_M[i,j] = C_V[i,j] × D[i] × D[j]  where D[k] = √2 if k ≥ n_normal else 1.
    """
    n = C_V.shape[1]
    D = torch.ones(n, dtype=C_V.dtype, device=C_V.device)
    D[n_normal:] = SQRT2
    D_outer = D[:, None] * D[None, :]           # (n, n)
    n_sp = C_V.dim() - 3                        # number of spatial dims
    shape = (1, n, n) + (1,) * n_sp
    return C_V * D_outer.reshape(shape)


def voigt_strain_to_mandel(eps_V: torch.Tensor, n_normal: int = 2) -> torch.Tensor:
    """
    eps_V: (B, n, *sp) Voigt strain (engineering shear)  →  Mandel (physical shear).
    Shear components are divided by √2.
    """
    eps_M = eps_V.clone()
    eps_M[:, n_normal:] = eps_M[:, n_normal:] / SQRT2
    return eps_M


def compute_T_mandel(
    C_field_V: torch.Tensor, alpha0: float, n_normal: int = 2
) -> torch.Tensor:
    """
    Compute normalised stiffness contrast T_M = (C_M − α₀ I) / α₀.
    C_field_V: (B, n, n, *sp) Voigt stiffness.
    Returns T_M: same shape in Mandel notation.
    """
    C_M = voigt_stiffness_to_mandel(C_field_V, n_normal)
    n   = C_M.shape[1]
    n_sp = C_M.dim() - 3
    C0  = torch.eye(n, dtype=C_M.dtype, device=C_M.device) * alpha0
    C0_bc = C0.reshape(1, n, n, *([1] * n_sp))
    return (C_M - C0_bc) / alpha0


def contraction_loss(
    xi_pred: torch.Tensor, xi_true: torch.Tensor
) -> torch.Tensor:
    """
    Relative L2 loss for the standalone regression: ‖ξ_pred − ξ_true‖ / ‖ξ_true‖.
    Both tensors: (N, n) or (B, n, *sp) — any shape, compared element-wise.
    """
    diff = xi_pred - xi_true
    ref  = xi_true.norm()
    return diff.norm() / ref.clamp(min=1e-30)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

def _verify() -> None:
    """
    Basic shape and gradient check.  Verifies the model runs and differentiates
    without error; does NOT check that it has learned anything useful (that
    requires training).
    """
    torch.manual_seed(0)
    B, n, N = 2, 3, 8
    T   = (torch.rand(B, n, n, N, N) - 0.5).double()
    eps = (torch.rand(B, n, N, N) - 0.5).double() * 0.02

    model = FreeFormKANTauTheta(n_comp=n, width=[18], grid_size=5, k=3)
    print(model)
    print(f"  Per-layer params: {model.layer_param_counts()}")
    assert model._dims[0] == n * (n + 1) // 2 + n, \
        f"Expected input_dim={n*(n+1)//2+n}, got {model._dims[0]}"

    xi = model(T, eps)
    assert xi.shape == (B, n, N, N), f"shape mismatch: {xi.shape}"

    # Check gradient flows
    loss = xi.mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads), "some parameter has None gradient"

    print(f"  Output shape: {xi.shape}  ✓")
    print(f"  All gradients present: ✓")
    print("FreeFormKANTauTheta self-test passed ✓")


if __name__ == "__main__":
    _verify()
