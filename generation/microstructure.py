"""
Two-phase microstructure generation for 2D and 3D periodic composites.

2D generators (plane-strain):
  - random_disks:    non-overlapping circular inclusions on an N×N grid
  - random_ellipses: non-overlapping elliptic inclusions on an N×N grid

3D generators:
  - random_spheres:  non-overlapping spherical inclusions on an N×N×N grid

Each generator returns a boolean phase field (True = inclusion) plus the
corresponding Voigt stiffness field C(x):
  2D: (3, 3, N, N)    — components: (11,11), (22,22), (11,22), (12,12), …
  3D: (6, 6, N, N, N) — components: (11,11), (22,22), (33,33), (23,23), …

Voigt index mapping:
  2D: 0↔(1,1), 1↔(2,2), 2↔(1,2)
  3D: 0↔(1,1), 1↔(2,2), 2↔(3,3), 3↔(2,3), 4↔(1,3), 5↔(1,2)

Lamé parameters for an isotropic phase (E, ν):
  λ = E ν / ((1+ν)(1−2ν)),   μ = E / (2(1+ν))

2D plane-strain stiffness (3×3 Voigt):
  C_1111 = C_2222 = λ+2μ,  C_1122 = λ,  C_1212 = μ  (engineering shear)

3D full stiffness (6×6 Voigt):
  Normal block (3×3): diag = λ+2μ, off-diag = λ
  Shear block (3×3):  diag = μ  (engineering shear: σ₂₃ = μ·γ₂₃)
"""

import numpy as np
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Isotropic stiffness helpers
# ---------------------------------------------------------------------------

def lame_from_engineering(E: float, nu: float) -> Tuple[float, float]:
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lam, mu


def isotropic_stiffness_voigt(E: float, nu: float) -> np.ndarray:
    """Return (3, 3) Voigt stiffness for 2D plane-strain isotropic material."""
    lam, mu = lame_from_engineering(E, nu)
    C = np.array([
        [lam + 2*mu, lam,        0],
        [lam,        lam + 2*mu, 0],
        [0,          0,          mu],
    ])
    return C


def isotropic_stiffness_voigt_3d(E: float, nu: float) -> np.ndarray:
    """
    Return (6, 6) Voigt stiffness for a 3D isotropic material.

    Voigt order: (11, 22, 33, 23, 13, 12).
    Engineering shear convention: σ₂₃ = μ · γ₂₃, so C[3,3]=C[4,4]=C[5,5]=μ.
    """
    lam, mu = lame_from_engineering(E, nu)
    C = np.zeros((6, 6), dtype=np.float64)
    # Normal-normal block (3×3): λ off-diagonal, λ+2μ on diagonal
    for i in range(3):
        C[i, i] = lam + 2 * mu
        for j in range(3):
            if i != j:
                C[i, j] = lam
    # Shear-shear block (3×3 diagonal)
    for i in range(3, 6):
        C[i, i] = mu
    return C


def build_C_field(phase: np.ndarray,
                  C_matrix: np.ndarray,
                  C_inclusion: np.ndarray) -> np.ndarray:
    """
    Build a stiffness field from a boolean phase map.

    Works for both 2D and 3D:
      2D: phase (N, N),     C_matrix/inclusion (3, 3)  → C_field (3, 3, N, N)
      3D: phase (N, N, N),  C_matrix/inclusion (6, 6)  → C_field (6, 6, N, N, N)
    """
    n_comp = C_matrix.shape[0]
    spatial = phase.shape                          # (N, N) or (N, N, N)
    C_field = np.zeros((n_comp, n_comp) + spatial, dtype=np.float64)
    mask = phase.astype(bool)
    for i in range(n_comp):
        for j in range(n_comp):
            C_field[i, j] = np.where(mask, C_inclusion[i, j], C_matrix[i, j])
    return C_field


# ---------------------------------------------------------------------------
# Microstructure generators
# ---------------------------------------------------------------------------

def random_disks(N: int,
                 n_disks: int,
                 r_min: float,
                 r_max: float,
                 volume_fraction_target: Optional[float] = None,
                 max_attempts: int = 10_000,
                 rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Generate an N×N binary phase field with randomly placed non-overlapping disks.

    Placement uses a simple rejection sampler. Periodic boundary conditions are
    NOT enforced (disks are clipped at the boundary).

    Args:
        N:                   Grid size (pixels).
        n_disks:             Number of disks to attempt to place.
        r_min, r_max:        Radius range in pixels.
        volume_fraction_target: If given, stop when this VF is reached.
        max_attempts:        Maximum placement attempts per disk.
        rng:                 NumPy random generator (reproducible if seeded).

    Returns:
        phase: (N, N) bool array — True = inclusion.
    """
    if rng is None:
        rng = np.random.default_rng()

    phase = np.zeros((N, N), dtype=bool)
    centers = []
    radii = []

    xs, ys = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')

    for _ in range(n_disks):
        placed = False
        for _ in range(max_attempts):
            r = rng.uniform(r_min, r_max)
            cx = rng.uniform(0, N)
            cy = rng.uniform(0, N)

            # Check overlap with existing disks (with a small gap of 1 pixel)
            overlap = False
            for (ex, ey), er in zip(centers, radii):
                dist = np.hypot(cx - ex, cy - ey)
                if dist < r + er + 1.0:
                    overlap = True
                    break

            if not overlap:
                centers.append((cx, cy))
                radii.append(r)
                mask = (xs - cx)**2 + (ys - cy)**2 <= r**2
                phase |= mask
                placed = True
                break

        if not placed:
            continue  # skip disk if no valid position found

        if volume_fraction_target is not None:
            if phase.mean() >= volume_fraction_target:
                break

    return phase


def random_ellipses(N: int,
                    n_ellipses: int,
                    a_min: float,
                    a_max: float,
                    aspect_min: float = 0.5,
                    aspect_max: float = 2.0,
                    max_attempts: int = 10_000,
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Generate an N×N binary phase field with randomly placed non-overlapping ellipses.

    Each ellipse has a random semi-major axis `a`, aspect ratio (b/a), and
    rotation angle.

    Args:
        N:                       Grid size (pixels).
        n_ellipses:              Number of ellipses to attempt.
        a_min, a_max:            Semi-major axis range in pixels.
        aspect_min, aspect_max:  b/a aspect ratio range.
        max_attempts:            Maximum placement attempts per ellipse.
        rng:                     NumPy random generator.

    Returns:
        phase: (N, N) bool array.
    """
    if rng is None:
        rng = np.random.default_rng()

    phase = np.zeros((N, N), dtype=bool)
    xs, ys = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')

    ellipses = []  # (cx, cy, a, b, angle)

    for _ in range(n_ellipses):
        for _ in range(max_attempts):
            a = rng.uniform(a_min, a_max)
            aspect = rng.uniform(aspect_min, aspect_max)
            b = a * aspect
            angle = rng.uniform(0, np.pi)
            cx = rng.uniform(0, N)
            cy = rng.uniform(0, N)

            # Overlap check using bounding-box + ellipse distance heuristic
            overlap = False
            for (ex, ey, ea, eb, eangle) in ellipses:
                d = np.hypot(cx - ex, cy - ey)
                if d < max(a, b) + max(ea, eb) + 1.0:
                    overlap = True
                    break

            if not overlap:
                ellipses.append((cx, cy, a, b, angle))
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                dx = xs - cx
                dy = ys - cy
                xr = cos_a * dx + sin_a * dy
                yr = -sin_a * dx + cos_a * dy
                mask = (xr / a)**2 + (yr / b)**2 <= 1.0
                phase |= mask
                break

    return phase


def random_spheres(N: int,
                   n_spheres: int,
                   r_min: float,
                   r_max: float,
                   volume_fraction_target: Optional[float] = None,
                   max_attempts: int = 10_000,
                   rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Generate an N×N×N binary phase field with randomly placed non-overlapping spheres.

    Args:
        N:                   Grid size (voxels per side).
        n_spheres:           Number of spheres to attempt to place.
        r_min, r_max:        Radius range in voxels.
        volume_fraction_target: If given, stop when this VF is reached.
        max_attempts:        Maximum placement attempts per sphere.
        rng:                 NumPy random generator.

    Returns:
        phase: (N, N, N) bool array — True = inclusion.
    """
    if rng is None:
        rng = np.random.default_rng()

    phase = np.zeros((N, N, N), dtype=bool)
    centers: list = []
    radii:   list = []

    xs, ys, zs = np.meshgrid(
        np.arange(N), np.arange(N), np.arange(N), indexing='ij'
    )

    for _ in range(n_spheres):
        placed = False
        for _ in range(max_attempts):
            r  = rng.uniform(r_min, r_max)
            cx = rng.uniform(0, N)
            cy = rng.uniform(0, N)
            cz = rng.uniform(0, N)

            overlap = any(
                (cx-ex)**2 + (cy-ey)**2 + (cz-ez)**2 < (r + er + 1.0)**2
                for (ex, ey, ez), er in zip(centers, radii)
            )
            if not overlap:
                centers.append((cx, cy, cz))
                radii.append(r)
                mask = (xs-cx)**2 + (ys-cy)**2 + (zs-cz)**2 <= r**2
                phase |= mask
                placed = True
                break

        if not placed:
            continue

        if volume_fraction_target is not None and phase.mean() >= volume_fraction_target:
            break

    return phase


# ---------------------------------------------------------------------------
# Top-level factory
# ---------------------------------------------------------------------------

def generate_microstructure(N: int,
                             inclusion_type: str,
                             E_matrix: float,
                             nu_matrix: float,
                             E_inclusion: float,
                             nu_inclusion: float,
                             dim: int = 2,
                             rng: Optional[np.random.Generator] = None,
                             **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a random two-phase microstructure for 2D or 3D problems.

    Args:
        N:               Grid resolution (N×N for 2D, N×N×N for 3D).
        inclusion_type:  2D: 'disk' or 'ellipse'.  3D: 'sphere'.
        E_matrix:        Young's modulus of matrix.
        nu_matrix:       Poisson ratio of matrix.
        E_inclusion:     Young's modulus of inclusion.
        nu_inclusion:    Poisson ratio of inclusion.
        dim:             Spatial dimension: 2 or 3.
        rng:             NumPy random generator.
        **kwargs:        Forwarded to the geometry generator.

    Returns:
        phase:   2D→ (N, N) bool,    3D→ (N, N, N) bool  — True = inclusion.
        C_field: 2D→ (3, 3, N, N),  3D→ (6, 6, N, N, N)  — Voigt stiffness.
    """
    if rng is None:
        rng = np.random.default_rng()

    if dim == 2:
        if inclusion_type == 'disk':
            n     = kwargs.get('n_disks', 10)
            r_min = kwargs.get('r_min', 2.0)
            r_max = kwargs.get('r_max', 6.0)
            phase = random_disks(N, n, r_min, r_max, rng=rng, **{
                k: v for k, v in kwargs.items()
                if k in ('volume_fraction_target', 'max_attempts')
            })
        elif inclusion_type == 'ellipse':
            n     = kwargs.get('n_ellipses', 10)
            a_min = kwargs.get('a_min', 2.0)
            a_max = kwargs.get('a_max', 6.0)
            phase = random_ellipses(N, n, a_min, a_max, rng=rng, **{
                k: v for k, v in kwargs.items()
                if k in ('aspect_min', 'aspect_max', 'max_attempts')
            })
        else:
            raise ValueError(
                f"Unknown 2D inclusion_type '{inclusion_type}'. Use 'disk' or 'ellipse'."
            )
        C_mat = isotropic_stiffness_voigt(E_matrix, nu_matrix)
        C_inc = isotropic_stiffness_voigt(E_inclusion, nu_inclusion)

    elif dim == 3:
        if inclusion_type not in ('sphere', 'disk'):
            raise ValueError(
                f"Unknown 3D inclusion_type '{inclusion_type}'. Use 'sphere'."
            )
        n     = kwargs.get('n_spheres', kwargs.get('n_disks', 5))
        r_min = kwargs.get('r_min', 2.0)
        r_max = kwargs.get('r_max', 6.0)
        phase = random_spheres(N, n, r_min, r_max, rng=rng, **{
            k: v for k, v in kwargs.items()
            if k in ('volume_fraction_target', 'max_attempts')
        })
        C_mat = isotropic_stiffness_voigt_3d(E_matrix, nu_matrix)
        C_inc = isotropic_stiffness_voigt_3d(E_inclusion, nu_inclusion)

    else:
        raise ValueError(f"dim must be 2 or 3, got {dim}.")

    C_field = build_C_field(phase, C_mat, C_inc)
    return phase, C_field
