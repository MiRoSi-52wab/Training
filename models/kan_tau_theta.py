"""
KAN double-contraction module: τ_θ(T, ε) = T : ε via degree-2 B-spline edges.

This is the drop-in replacement for YarotskyTauTheta (ls_fno.py).  Pass an
instance as the `tau_theta` argument to LSFNO — everything else stays unchanged.

Theory
------
The double contraction T_ij · ε_j is recovered from the polarization identity:

    T_ij · ε_j  =  2R² [ φ(s_ij) − φ(a_ij) − φ(b_ij) ]

    s_ij = (T_ij + ε_j) / (2R)   ← "sum" argument
    a_ij =  T_ij         / (2R)   ← "T alone" argument
    b_ij =          ε_j  / (2R)   ← "ε alone" argument

φ is a degree-2 B-spline with the Bernstein basis on the domain [-1, 1]:

    B_0(x) = (1 − x)² / 4
    B_1(x) = (1 − x²) / 2
    B_2(x) = (1 + x)² / 4

    φ(x) = c₀·B_0(x) + c₁·B_1(x) + c₂·B_2(x)

Exact initialisation:  c₀ = 1,  c₁ = −1,  c₂ = 1.

Proof that this gives φ(x) = x² :

    1·(1-x)²/4  −  1·(1-x²)/2  +  1·(1+x)²/4
    = [(1-x)² + (1+x)²]/4  −  (1-x²)/2
    = [2 + 2x²]/4  −  (1-x²)/2
    = (1 + x²)/2  −  1/2 + x²/2
    = x²   ✓

Therefore  2R²[s² − a² − b²] = 2R²·[(T+ε)² − T² − ε²]/(4R²) = T·ε  exactly.

Consequences
------------
  • No approximation error: α_eff = 0 for all strain magnitudes at all κ.
  • No calibration-box / ridge-function clipping (r_θ is removed).
  • The Bernstein basis is a polynomial, so the formula is valid for |x| > 1 as
    well — the B-spline extrapolates exactly as x² outside [-1, 1].
  • With trainable control points the module can learn deviations from x²,
    enabling Study 2 (symbolic recovery) and Study 3 (suboptimal α₀ correction).

Parameter count
---------------
  shared=True  (Study 1, default): 3 control points total.
  shared=False (Studies 2 & 3):    3 × n_comp × n_comp control points
                                   = 108 for 3D Mandel (n_comp=6).

References
----------
  KAN_Architecture_Theory.md
  BSpline_Theory_Complete.md
"""

import torch
import torch.nn as nn


class KANTauTheta(nn.Module):
    """
    B-spline KAN double-contraction operator.

    Drop-in replacement for YarotskyTauTheta with an identical forward
    signature: forward(T, eps) → xi.

    Args:
        R:         Domain/scaling parameter.  Pre-scales inputs by 1/(2R) so
                   that T_ij/(2R) ∈ [-½, ½] and (T_ij+ε_j)/(2R) ∈ [-1, 1]
                   when |T_ij|, |ε_j| ≤ R.  Set R to match the Yarotsky
                   cutoff_M for a fair comparison (default: R=1.0).
        shared:    If True (Study 1, default): all n² edges share the same
                   3 control points.  If False (Studies 2 & 3): each (i,j)
                   edge has its own 3 control points.
        trainable: If False (default, Study 1): control points are frozen at
                   the exact-x² values — no parameters.  If True (Studies 2 &
                   3): control points are nn.Parameter.
        n_comp:    Number of Mandel strain components (3 for 2D, 6 for 3D).
    """

    def __init__(
        self,
        R: float = 1.0,
        shared: bool = True,
        trainable: bool = False,
        n_comp: int = 6,
    ):
        super().__init__()
        self.R      = float(R)
        self.n      = n_comp
        self._shared = shared

        # Exact B-spline control points for φ(x) = x² on [-1, 1].
        # The Bernstein basis on [-1,1] has B_0 = (1-x)²/4, B_1 = (1-x²)/2,
        # B_2 = (1+x)²/4, and the unique solution to Σ c_k B_k = x² is
        # c₀ = 1, c₁ = −1, c₂ = 1.
        if shared:
            exact = torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64)
        else:
            # Shape (3, n_comp, n_comp): ctrl[k, i, j] for edge (i,j)
            exact = torch.zeros(3, n_comp, n_comp, dtype=torch.float64)
            exact[0] =  1.0
            exact[1] = -1.0
            exact[2] =  1.0

        if trainable:
            self.ctrl = nn.Parameter(exact)
        else:
            self.register_buffer('ctrl', exact)

    # ── B-spline evaluation ───────────────────────────────────────────────────

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the B-spline φ(x) = c₀·B₀(x) + c₁·B₁(x) + c₂·B₂(x).

        Bernstein basis on [-1, 1]:
            B₀(x) = (1 − x)² / 4
            B₁(x) = (1 − x²) / 2
            B₂(x) = (1 + x)² / 4

        These are polynomials valid for all x ∈ ℝ; the [-1, 1] label just
        indicates where exact control points keep φ(x) = x² within machine ε.

        For the exact control points [1, -1, 1] the result is identically x².

        Args:
            x: any shape, float64
        Returns:
            φ(x): same shape as x
        """
        B0 = (1.0 - x) ** 2 * 0.25   # (1-x)² / 4
        B1 = (1.0 - x * x) * 0.5     # (1-x²) / 2
        B2 = (1.0 + x) ** 2 * 0.25   # (1+x)² / 4

        if self._shared:
            # ctrl: (3,) — scalars that broadcast against any x shape
            return self.ctrl[0] * B0 + self.ctrl[1] * B1 + self.ctrl[2] * B2

        # ctrl: (3, n, n) — one set per (i,j) edge.
        # x has shape (B, n, n, *spatial); need ctrl shaped (1, n, n, 1..1).
        n_sp  = x.dim() - 3                        # number of spatial dims
        view  = (1, self.n, self.n) + (1,) * n_sp
        c0 = self.ctrl[0].reshape(view)
        c1 = self.ctrl[1].reshape(view)
        c2 = self.ctrl[2].reshape(view)
        return c0 * B0 + c1 * B1 + c2 * B2

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, T: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """
        Pointwise double-contraction ξ_i = Σ_j T_ij · ε_j (exactly).

        Pipeline (per voxel, per batch element):
            Step 1 — pre-scale:   s_ij, a_ij, b_ij  (108 scalars per voxel)
            Step 2 — B-spline:    φ(s), φ(a), φ(b)
            Step 3 — polarize:    p_ij = 2R²[φ(s) − φ(a) − φ(b)] = T_ij·ε_j
            Step 4 — row sum:     ξ_i  = Σ_j p_ij

        All arithmetic is done in float64 (same as YarotskyTauTheta) and cast
        back to the caller's dtype on return.

        Args:
            T:   (B, n, n, *spatial) — normalised stiffness contrast in Mandel.
            eps: (B, n, *spatial)    — strain field in Mandel notation.

        Returns:
            xi:  (B, n, *spatial)    — polarisation stress ξ = T : ε (exactly).
        """
        orig_dtype = T.dtype
        T   = T.double()
        eps = eps.double()

        inv2R = 1.0 / (2.0 * self.R)
        coeff = 2.0 * self.R ** 2        # 2R²

        # Expand ε_j along the i-axis: eps_exp[b,i,j,...] = eps[b,j,...]
        eps_exp = eps.unsqueeze(1).expand_as(T)

        # Step 1 — pre-scaled arguments
        s = (T + eps_exp) * inv2R        # (T_ij + ε_j) / (2R)
        a = T             * inv2R        # T_ij / (2R)
        b = eps_exp       * inv2R        # ε_j  / (2R)

        # Steps 2 + 3 — B-spline evaluation and polarization recovery
        # p_ij = 2R²[φ(s_ij) − φ(a_ij) − φ(b_ij)] = T_ij · ε_j  (exactly)
        p = coeff * (self._phi(s) - self._phi(a) - self._phi(b))

        # Step 4 — row sum: ξ_i = Σ_j p_ij
        return p.sum(dim=2).to(orig_dtype)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def n_params(self) -> int:
        """Number of trainable parameters (0 if trainable=False)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def n_ctrl_points(self) -> int:
        """Total number of control points (trainable or fixed)."""
        return int(self.ctrl.numel())

    def __repr__(self) -> str:
        trainable = self.n_params() > 0
        mode = "shared" if self._shared else "independent"
        return (
            f"KANTauTheta(R={self.R}, n={self.n}, edges={mode}, "
            f"trainable={trainable}, ctrl_pts={self.n_ctrl_points()})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Self-test / quick verification
# ─────────────────────────────────────────────────────────────────────────────

def _verify_exact_representation() -> None:
    """
    Verify that KANTauTheta with exact initialisation computes T:ε up to
    float64 machine epsilon, and compare against YarotskyTauTheta at depth 11.

    Two tests:

    Test 1 — float64 I/O (exact representation claim):
        Pass float64 T and eps directly.  Expected error ~ machine epsilon (1e-15).
        This is the fundamental claim: the B-spline with control points [1,-1,1]
        computes x² algebraically, so the polarization identity gives T:ε exactly.

    Test 2 — float32 I/O (LSFNO pipeline simulation):
        Pass float32 T and eps (as LSFNO does in practice).  The KAN computes the
        product of the float32-truncated inputs in float64, then rounds the result
        back to float32.  Error vs the float32-truncated ground truth ~ machine
        epsilon; error vs the original float64 ground truth ~ ε_f32 ≈ 1.2e-7
        (from float32 input truncation, NOT from the KAN computation).

    FNO11 comparison:
        Uses float32 inputs and is compared against the float32-truncated ground
        truth so the comparison is fair.  Expected error ~ rms Yarotsky approx
        error per xi entry, which at these input magnitudes is ~ 6×10⁻⁵.
        (The "~2×10⁻⁷" in the YarotskyTauTheta docstring is the absolute error
        in the q(x)≈x² approximation, not the relative error in the product.)
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from models.ls_fno import YarotskyTauTheta

    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    B, n, N = 2, 6, 8
    # T in [-0.9, 0.9] and eps in [-0.01, 0.01] — representative values
    T   = (torch.rand(B, n, n, N, N, N, dtype=torch.float64) - 0.5) * 1.8
    eps = (torch.rand(B, n,    N, N, N, dtype=torch.float64) - 0.5) * 0.02

    kan     = KANTauTheta(R=1.0, shared=True,  trainable=False, n_comp=n)
    kan_ind = KANTauTheta(R=1.0, shared=False, trainable=False, n_comp=n)
    yar     = YarotskyTauTheta(depth_m=11, cutoff_M=1.0)

    # ── Test 1: float64 I/O ───────────────────────────────────────────────────
    xi_true_f64 = torch.einsum('bijxyz,bjxyz->bixyz', T, eps)
    xi_kan      = kan(T, eps)         # float64 in → float64 out (no round-trip)
    xi_kan_ind  = kan_ind(T, eps)

    norm64       = xi_true_f64.norm().item()
    err_kan      = (xi_kan     - xi_true_f64).norm().item() / norm64
    err_kan_ind  = (xi_kan_ind - xi_true_f64).norm().item() / norm64

    # ── Test 2: float32 I/O (fair FNO11 comparison) ──────────────────────────
    # Ground truth from float32-truncated inputs (what KAN actually received)
    xi_true_f32 = torch.einsum('bijxyz,bjxyz->bixyz',
                                T.float().double(), eps.float().double())
    xi_kan_f32_pipeline = kan(T.float(), eps.float()).double()
    xi_yar              = yar(T.float(), eps.float()).double()

    norm32       = xi_true_f32.norm().item()
    err_kan_f32  = (xi_kan_f32_pipeline - xi_true_f32).norm().item() / norm32
    err_yar      = (xi_yar              - xi_true_f32).norm().item() / norm32

    print("KANTauTheta verification")
    print()
    print("  Test 1 — float64 inputs (exact representation):")
    print(f"    KAN-exact (shared):      rel_err = {err_kan:.2e}"
          f"  ({'✓ ≤ 1e-13' if err_kan < 1e-13 else '✗ too large'})")
    print(f"    KAN-exact (independent): rel_err = {err_kan_ind:.2e}"
          f"  ({'✓ ≤ 1e-13' if err_kan_ind < 1e-13 else '✗ too large'})")
    print()
    print("  Test 2 — float32 inputs (LSFNO pipeline, vs float32-truncated truth):")
    print(f"    KAN-exact float32:       rel_err = {err_kan_f32:.2e}"
          f"  (float32 output rounding, expected ~ε_f32 ≈ 1.2e-7)")
    print(f"    FNO11 (Yarotsky):        rel_err = {err_yar:.2e}"
          f"  (Yarotsky approx error at these magnitudes)")

    assert err_kan     < 1e-13, f"KAN-exact (shared) float64 error too large: {err_kan:.2e}"
    assert err_kan_ind < 1e-13, f"KAN-exact (indep)  float64 error too large: {err_kan_ind:.2e}"
    assert err_kan_f32 < 2e-7,  f"KAN-exact float32 pipeline error too large: {err_kan_f32:.2e}"
    print("\nAll assertions passed ✓")


if __name__ == "__main__":
    _verify_exact_representation()
