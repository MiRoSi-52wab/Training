"""
Config loading with experiment-level inheritance.

Usage
-----
    from utils.config_loader import load_config, compute_alpha_bounds

    cfg = load_config("configs/ls_fno.yaml")
    # → merges configs/experiment.yaml (via the `experiment` key) then
    #   applies ls_fno.yaml overrides on top.

    alpha_minus, alpha_plus = compute_alpha_bounds(
        cfg["E_matrix"], cfg["nu_matrix"], cfg["nu_inclusion"], cfg["kappa"]
    )

Design
------
Parameters split into three categories:

  Problem definition  (experiment.yaml)
      N, kappa, E_matrix, nu_matrix, nu_inclusion, eps_bar_scale
      tol, max_iter  — evaluation stopping criterion, shared between all solvers

  Derived from problem definition  (computed, never stored)
      alpha_minus  = μ_matrix              (min eigenvalue of C_matrix in Voigt)
      alpha_plus   = 2(λ+μ)_inclusion      (max eigenvalue of C_inclusion in Voigt)

  Solver-specific  (individual config files)
      K, m, M              — LS-FNO architecture
      n_samples, seed, ... — data generation
      tol, max_iter        — data.yaml may OVERRIDE to tighter values for generation

Merging rule:  experiment.yaml is the base; the child config's values win on overlap.
The `experiment` key is stripped from the returned dict.
"""

from pathlib import Path
import yaml


def load_config(config_path, base_dir=None) -> dict:
    """
    Load a YAML config, transparently merging its parent experiment file.

    If the config contains an ``experiment`` key, that file is loaded first
    (relative to the config's own directory) and used as the base.  The child
    config's values then override the base, so per-solver or per-run settings
    win over shared defaults.

    Args:
        config_path: Path to the config YAML file.
        base_dir:    Ignored (reserved for future use).

    Returns:
        Merged dict with the ``experiment`` key removed.
    """
    config_path = Path(config_path)

    with open(config_path) as fh:
        raw = yaml.safe_load(fh) or {}

    if "experiment" not in raw:
        return raw

    exp_ref  = raw["experiment"]
    exp_path = config_path.parent / exp_ref

    with open(exp_path) as fh:
        base = yaml.safe_load(fh) or {}

    # Child overrides base; strip the reference key so callers never see it.
    merged = {**base, **raw}
    merged.pop("experiment")
    return merged


def compute_alpha_bounds(
    E_matrix: float,
    nu_matrix: float,
    nu_inclusion: float,
    kappa: float,
    dim: int = 2,
) -> tuple:
    """
    Compute the min/max stiffness eigenvalues for a two-phase composite.

    In Voigt notation with engineering shear, the stiffness matrix eigenvalues
    depend on the spatial dimension.  Across the two phases:

        alpha_minus = μ_matrix              (smallest eigenvalue — same in 2D and 3D)
        alpha_plus  = max eigenvalue of C_inclusion:
                      2D (plane-strain): 2(λ+μ)_inclusion
                      3D (full):         (3λ+2μ)_inclusion  = E_inclusion / (1 − 2ν)

    These determine the optimal reference stiffness
        α₀ = (alpha_minus + alpha_plus) / 2
    and the contraction constant γ = (α⁺ − α⁻) / (α⁺ + α⁻).

    Args:
        E_matrix:      Young's modulus of the matrix phase.
        nu_matrix:     Poisson's ratio of the matrix phase.
        nu_inclusion:  Poisson's ratio of the inclusion phase.
        kappa:         Stiffness contrast  E_inclusion / E_matrix.
        dim:           Spatial dimension: 2 (plane-strain) or 3 (full 3D).

    Returns:
        (alpha_minus, alpha_plus) as floats.

    Examples:
        2D, κ=10 (matches old ls_fno.yaml):
            >>> compute_alpha_bounds(1.0, 0.3, 0.3, 10.0, dim=2)
            (0.3846..., 19.23...)
        3D, κ=12 (paper §5.2, E=3 GPa):
            >>> compute_alpha_bounds(3.0, 0.3, 0.22, 12.0, dim=3)
            → alpha_minus = μ_matrix, alpha_plus = E_inc/(1−2·ν_inc)
    """
    def lame(E, nu):
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu  = E / (2.0 * (1.0 + nu))
        return lam, mu

    _,       mu_mat  = lame(E_matrix,             nu_matrix)
    lam_inc, mu_inc  = lame(E_matrix * kappa,      nu_inclusion)

    # alpha_minus is the smallest eigenvalue of C(x) viewed as a linear map on
    # Sym(d) — the deviatoric (shear) eigenvalue, which equals 2μ in both 2D and
    # 3D.  The Voigt 6×6 matrix has a smaller eigenvalue (μ, from the engineering-
    # shear block), but that is an artefact of the Voigt convention; the underlying
    # tensor eigenvalue is 2μ.  Using 2μ gives the paper's optimal α₀ and the
    # correct bound ‖T‖_op ≤ γ < 1 in Mandel notation.
    alpha_minus = float(2.0 * mu_mat)
    if dim == 2:
        alpha_plus = float(2.0 * (lam_inc + mu_inc))      # max eig in 2D (= 3K in plane-strain)
    else:
        alpha_plus = float(3.0 * lam_inc + 2.0 * mu_inc)  # max eig in 3D (= 3K)
    return alpha_minus, alpha_plus
