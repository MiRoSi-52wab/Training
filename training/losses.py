"""
Loss functions for training the trainable KANTauTheta inside LSFNO.

Both losses operate on Voigt-notation tensors (the format produced by
datasets/micromechanics.py and models/ls_fno.py's forward()). Getting the
*true* tensor (Frobenius) norm out of a Voigt vector with engineering shear
needs a component weighting; that weighting is folded in directly here rather
than routed through utils/notation.py, since that module is numpy-only and
would detach gradients if used on model predictions.
"""

import torch


def _strain_weight(n_comp: int, n_normal: int, device, dtype) -> torch.Tensor:
    """
    Per-component weight so that sum(weight * eps_v**2) over the component
    axis equals the true tensor Frobenius norm squared of a Voigt strain
    vector with engineering shear (gamma_ij = 2*eps_ij): normal components
    carry weight 1, shear components carry weight 1/2.
    """
    w = torch.ones(n_comp, device=device, dtype=dtype)
    w[n_normal:] = 0.5
    return w


def field_loss(eps_pred: torch.Tensor, eps_star: torch.Tensor, n_normal: int = 2) -> torch.Tensor:
    """
    Relative L2 error on the strain field, in the true tensor norm (not the
    naive Euclidean norm of the Voigt vector, which over/under-weights shear).

    Args:
        eps_pred, eps_star: (B, n_comp, *spatial) Voigt strain fields.
        n_normal: 2 for 2D (plane-strain), 3 for 3D. Pass model.n_normal.

    Returns:
        Scalar tensor: mean over the batch of the per-sample relative error.
    """
    n_comp = eps_pred.shape[1]
    w = _strain_weight(n_comp, n_normal, eps_pred.device, eps_pred.dtype)
    w = w.reshape(1, n_comp, *([1] * (eps_pred.dim() - 2)))

    diff_sq = (w * (eps_pred - eps_star) ** 2).flatten(1).sum(dim=1)
    ref_sq = (w * eps_star ** 2).flatten(1).sum(dim=1)
    return (diff_sq / ref_sq.clamp_min(1e-30)).sqrt().mean()


def directional_modulus_loss(
    eps_pred: torch.Tensor,
    C_field: torch.Tensor,
    eps_bar: torch.Tensor,
    sigma_star: torch.Tensor,
    n_normal: int = 2,
) -> torch.Tensor:
    """
    Relative error on a scalar "directional modulus"
        E_dir = (sigma : eps_bar) / (eps_bar : eps_bar)
    i.e. the effective stiffness seen along this sample's own loading
    direction. Adapted from the guide's full C_eff tensor (which needs an
    n_comp-load-case sweep we don't have per sample) to the single loading
    direction actually stored per sample.

    sigma:eps_bar is a stress-strain work pairing, where the Voigt
    engineering-shear convention already makes the plain dot product exact
    (no component weighting needed — this is the whole reason for the
    engineering-shear convention). eps_bar:eps_bar is a strain-strain pairing
    and does need the true-norm weighting from _strain_weight.

    Args:
        eps_pred:   (B, n_comp, *spatial) model's predicted strain field.
        C_field:    (B, n_comp, n_comp, *spatial) Voigt stiffness field (input).
        eps_bar:    (B, n_comp) macroscopic strain (Voigt).
        sigma_star: (B, n_comp, *spatial) ground-truth stress field.
        n_normal:   2 for 2D, 3 for 3D. Pass model.n_normal.

    Returns:
        Scalar tensor: mean over the batch of the relative error in E_dir.
    """
    sigma_pred = torch.einsum("bij...,bj...->bi...", C_field, eps_pred)

    spatial_mean_dims = tuple(range(2, eps_pred.dim()))
    sigma_pred_mean = sigma_pred.mean(dim=spatial_mean_dims)      # (B, n_comp)
    sigma_star_mean = sigma_star.mean(dim=spatial_mean_dims)      # (B, n_comp)

    work_pred = (sigma_pred_mean * eps_bar).sum(dim=1)
    work_star = (sigma_star_mean * eps_bar).sum(dim=1)

    n_comp = eps_bar.shape[1]
    w = _strain_weight(n_comp, n_normal, eps_bar.device, eps_bar.dtype)
    energy = (w * eps_bar ** 2).sum(dim=1).clamp_min(1e-30)

    e_dir_pred = work_pred / energy
    e_dir_star = work_star / energy

    return ((e_dir_pred - e_dir_star).abs() / e_dir_star.abs().clamp_min(1e-30)).mean()


def combined_loss(
    eps_pred: torch.Tensor,
    eps_star: torch.Tensor,
    C_field: torch.Tensor,
    eps_bar: torch.Tensor,
    sigma_star: torch.Tensor,
    n_normal: int = 2,
    lambda_field: float = 1.0,
    lambda_eff: float = 0.1,
):
    """
    L_total = lambda_field * field_loss + lambda_eff * directional_modulus_loss.

    Returns:
        (total, parts) where parts = {'field': ..., 'eff': ...} are detached
        scalars for logging (total keeps its graph for backward()).
    """
    l_field = field_loss(eps_pred, eps_star, n_normal)
    l_eff = directional_modulus_loss(eps_pred, C_field, eps_bar, sigma_star, n_normal)
    total = lambda_field * l_field + lambda_eff * l_eff
    return total, {"field": l_field.detach(), "eff": l_eff.detach()}
