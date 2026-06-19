"""M&S 2D centered-disk microstructure geometry for the comparative study."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generation.microstructure import isotropic_stiffness_voigt, build_C_field

VF_DEFAULT = 0.475   # Moulinec & Suquet (1994) volume fraction

E_F_DEFAULT  = 400_000.0
NU_F_DEFAULT = 0.23
E_M_DEFAULT  =  68_900.0
NU_M_DEFAULT = 0.35


def make_ms_geometry(
    N: int,
    E_m: float  = E_M_DEFAULT,
    nu_m: float = NU_M_DEFAULT,
    E_f: float  = E_F_DEFAULT,
    nu_f: float = NU_F_DEFAULT,
    vf: float   = VF_DEFAULT,
):
    """
    Create the Moulinec & Suquet (1994) centered-disk two-phase unit cell.

    Parameters
    ----------
    N     : grid resolution (N × N voxels)
    E_m   : Young's modulus of the elastoplastic matrix (MPa)
    nu_m  : Poisson's ratio of the matrix
    E_f   : Young's modulus of the elastic fiber
    nu_f  : Poisson's ratio of the fiber
    vf    : fiber volume fraction (default 0.475)

    Returns
    -------
    phase   : (N, N) bool — True = elastic fiber, False = elastoplastic matrix
    C_field : (3, 3, N, N) float64 — Voigt stiffness field
    """
    r = np.sqrt(vf * N**2 / np.pi)
    xs, ys = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    cx, cy = (N - 1) / 2.0, (N - 1) / 2.0
    phase   = (xs - cx)**2 + (ys - cy)**2 <= r**2   # True = inside disk (fiber)

    C_m = isotropic_stiffness_voigt(E_m, nu_m)
    C_f = isotropic_stiffness_voigt(E_f, nu_f)
    C_field = build_C_field(phase, C_m, C_f)   # phase=True → C_f

    return phase, C_field


def compute_alpha_opt(C_field: np.ndarray) -> float:
    """Return Moulinec-Suquet optimal reference stiffness (C00_max + C00_min) / 2."""
    C00 = C_field[0, 0]
    return (float(C00.max()) + float(C00.min())) / 2.0
