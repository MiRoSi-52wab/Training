"""
Evaluation metrics for the trained LS-KAN-FNO: accuracy (reusing the training
losses as metrics) plus the non-optional contractivity check.

gamma_theta_empirical is the guide's "non-optional" check: if the trained
tau_theta is not a contraction (gamma_theta >= 2/(kappa+1)), iteration counts
explode and any accuracy/iteration-count comparison is meaningless. Always
run this before trusting other numbers from a trained checkpoint.
"""

import torch

from training.losses import directional_modulus_loss as rel_err_modulus  # noqa: F401
from training.losses import field_loss as rel_L2_field  # noqa: F401


def gamma_theta_bound(kappa: float) -> float:
    """Contractivity requirement from theory: gamma_theta < 2/(kappa+1)."""
    return 2.0 / (kappa + 1.0)


def _voigt_to_mandel_strain_torch(eps_v: torch.Tensor, n_normal: int) -> torch.Tensor:
    """Torch-native, grad-safe equivalent of utils.notation.voigt_to_mandel_strain
    (component axis fixed at 1, matching this project's (B, n_comp, ...) layout)."""
    out = eps_v.clone()
    out[:, n_normal:] = eps_v[:, n_normal:] * (1.0 / (2.0 ** 0.5))
    return out


def gamma_theta_empirical(model, batch: dict, n_pairs: int = 4096, seed: int = 0) -> dict:
    """
    Empirically estimate the Lipschitz constant of model.tau_theta with
    respect to its eps argument, using real (T, eps) values pulled from a
    data batch (per the guide: "sample random pairs (T, eps) and (T, eps')").

    T comes from the batch's real C_field (via the model's own
    _to_mandel_state). eps candidates come from the batch's real eps_star
    field (real, spatially-varying, converged strain values) rather than
    synthetic noise, so the pairs are physically representative. Each sampled
    voxel's T is paired with two independently-drawn eps voxels.

    tau_theta acts pointwise with no spatial coupling, so "voxel" here means
    any (batch, spatial-location) pair, flattened into one pool to sample from.

    Args:
        model:   an LSFNO instance (provides ._to_mandel_state and .tau_theta).
        batch:   dict with 'C_field', 'eps_bar', 'eps_star' (Voigt tensors, as
                 returned by MicromechanicsDataset / DataLoader).
        n_pairs: number of voxel pairs to sample.

    Returns:
        dict with 'max', 'p99', 'mean' empirical gamma_theta (float).
    """
    with torch.no_grad():
        T_M, _ = model._to_mandel_state(batch["C_field"], batch["eps_bar"])
        eps_star_M = _voigt_to_mandel_strain_torch(batch["eps_star"], model.n_normal)

        n = model.n_comp
        perm_T = (0,) + tuple(range(3, T_M.dim())) + (1, 2)
        T_flat = T_M.permute(*perm_T).reshape(-1, n, n)

        perm_eps = (0,) + tuple(range(2, eps_star_M.dim())) + (1,)
        eps_flat = eps_star_M.permute(*perm_eps).reshape(-1, n)

        m_total = T_flat.shape[0]
        idx_a = torch.randint(0, m_total, (n_pairs,))
        idx_b = torch.randint(0, m_total, (n_pairs,))

        # Treat the n_pairs samples as a fake "spatial" axis (length n_pairs,
        # batch size 1) so they pass through tau_theta's pointwise forward().
        T_in = T_flat[idx_a].permute(1, 2, 0).unsqueeze(0)        # (1, n, n, n_pairs)
        eps_a_in = eps_flat[idx_a].permute(1, 0).unsqueeze(0)     # (1, n, n_pairs)
        eps_b_in = eps_flat[idx_b].permute(1, 0).unsqueeze(0)     # (1, n, n_pairs)

        tau_a = model.tau_theta(T_in, eps_a_in).squeeze(0)        # (n, n_pairs)
        tau_b = model.tau_theta(T_in, eps_b_in).squeeze(0)
        eps_a_sq = eps_a_in.squeeze(0)
        eps_b_sq = eps_b_in.squeeze(0)

        # Already in Mandel notation, so the plain Euclidean norm over the
        # component axis IS the true tensor norm (no shear weighting needed).
        num_norm = (tau_a - tau_b).norm(dim=0)
        den_norm = (eps_a_sq - eps_b_sq).norm(dim=0)

        valid = den_norm > 1e-12
        ratios = num_norm[valid] / den_norm[valid]

        return {
            "max":  float(ratios.max()),
            "p99":  float(torch.quantile(ratios, 0.99)),
            "mean": float(ratios.mean()),
        }


def iteration_count(model, C_field_V: torch.Tensor, eps_bar_V: torch.Tensor,
                    tol: float = 1e-5, max_iter: int = 2000) -> dict:
    """
    Run model.solve() (dynamic depth, until convergence) for comparison
    against the FFT solver's own stored n_iter on the same samples.
    """
    with torch.no_grad():
        result = model.solve(C_field_V, eps_bar_V, tol=tol, max_iter=max_iter)
    return {"n_iter": result["n_iter"], "converged": result["converged"]}
