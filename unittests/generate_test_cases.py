"""
Generate analytical test cases for validating the FFT solver and all future models.

Supports both 2D (plane-strain) and 3D problems.  Dimensionality is read from the
``dim`` key in the config file (defaults to 2).

2D — three microstructure types × three loading directions = 9 test cases:

  Type 1 — Homogeneous (κ=1):
      Exact result: ε(x) = ε̄ everywhere, τ(x) = 0 everywhere.

  Type 2 — Periodic horizontal laminate:
      Exact result (piecewise constant analytical fields) from compatibility and
      equilibrium of a 1-D layered composite:
          ε₁₁(x) = ε̄₁₁  everywhere   (isostrain along layers)
          σ₂₂(x) = σ̄₂₂  everywhere   (isostress normal to layers)
          σ₁₂(x) = σ̄₁₂  everywhere   (isostress shear)

  Type 3 — Single circular inclusion (Eshelby):
      Approximate result from the dilute Eshelby inclusion theory:
          ε_inside ≈ A : ε̄   (uniform, A = [I + (κ-1)S]⁻¹)
      Valid for small volume fraction (r=8 on 64×64 → VF≈5%).

3D — two microstructure types × six loading directions = 12 test cases:

  Type 1 — Homogeneous (κ=1): exact, same principle as 2D.
  Type 2 — Periodic horizontal laminate (interface normal = e₂): exact analytical
      piecewise-constant fields from 3D isostrain/isostress conditions.

All material and grid parameters are read from configs/experiment.yaml so that
this file and run_tests.py always use the same physical problem definition.

Usage (from project root):
  python -m unittests.generate_test_cases
  python -m unittests.generate_test_cases --config configs/experiment.yaml
  python -m unittests.generate_test_cases --output unittests/test_cases.h5
"""

import sys
import argparse
import numpy as np
import h5py
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generation.microstructure import (
    isotropic_stiffness_voigt, isotropic_stiffness_voigt_3d,
    build_C_field, lame_from_engineering,
)
from utils.config_loader import load_config

# ── Test-specific constant (not a shared physics parameter) ───────────────────
# The Eshelby radius is fixed at 8 pixels to keep VF ≈ 5% regardless of grid
# size, which satisfies the dilute-limit assumption.  It is not driven by any
# physics config because the test's purpose is qualitative (uniform inclusion
# strain), not a quantitative match at a specific volume fraction.
ESHELBY_RADIUS = 8


# ─────────────────────────────────────────────────────────────────────────────
# Helper: reference stiffness α₀ = (C⁺₁₁₁₁ + C⁻₁₁₁₁) / 2
# ─────────────────────────────────────────────────────────────────────────────

def _alpha0(C_mat: np.ndarray, C_inc: np.ndarray) -> float:
    return (C_inc[0, 0] + C_mat[0, 0]) / 2.0


def _C0_voigt(alpha0: float) -> np.ndarray:
    """C⁰ = α₀ diag(1, 1, 0.5) in 2D Voigt with engineering shear."""
    return alpha0 * np.diag([1.0, 1.0, 0.5])


def _C0_voigt_3d(alpha0: float) -> np.ndarray:
    """C⁰ = α₀ diag(1, 1, 1, 0.5, 0.5, 0.5) in 3D Voigt with engineering shear."""
    return alpha0 * np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])


# ─────────────────────────────────────────────────────────────────────────────
# Analytical solutions
# ─────────────────────────────────────────────────────────────────────────────

def _homogeneous_solution(C_mat: np.ndarray, eps_bar: np.ndarray, N: int,
                          alpha0: float) -> dict:
    """
    Exact solution for a homogeneous domain.
    ε(x) = ε̄ everywhere → τ = (C_mat - C⁰) : ε̄ (uniform) → σ = C_mat : ε̄ (uniform).
    """
    eps_star   = np.zeros((3, N, N))
    tau_star   = np.zeros((3, N, N))
    sigma_star = np.zeros((3, N, N))

    C0 = _C0_voigt(alpha0)

    for a in range(3):
        eps_star[a]   = eps_bar[a]
        tau_star[a]   = np.dot(C_mat[a] - C0[a], eps_bar)
        sigma_star[a] = np.dot(C_mat[a], eps_bar)

    return {"eps_star": eps_star, "tau_star": tau_star, "sigma_star": sigma_star}


def _laminate_solution(C_mat: np.ndarray, C_inc: np.ndarray,
                       phase: np.ndarray, eps_bar: np.ndarray,
                       alpha0: float,
                       E_mat: float, E_inc: float, nu: float) -> dict:
    """
    Exact analytical solution for a horizontal periodic laminate
    (interface normal = e₂, i.e. layers are horizontal rows).

    Derivation
    ----------
    Compatibility  →  ε₁₁ = ε̄₁₁ everywhere (in-plane strain is uniform).
    Equilibrium    →  σ₂₂ = σ̄₂₂ = const, σ₁₂ = σ̄₁₂ = const.

    For isotropic phases with the SAME ν, λ/(λ+2μ) = ν/(1-ν) is phase-independent,
    which simplifies σ̄₂₂ to a closed form. σ̄₁₂ follows from the harmonic mean of μ.
    """
    N = phase.shape[0]
    lam_m, mu_m = lame_from_engineering(E_mat, nu)
    lam_i, mu_i = lame_from_engineering(E_inc, nu)

    M_m = lam_m + 2*mu_m   # C₂₂₂₂ of matrix
    M_i = lam_i + 2*mu_i   # C₂₂₂₂ of inclusion

    f = phase.mean()        # inclusion volume fraction

    A_mat = 1.0 / M_m
    A_inc = 1.0 / M_i

    B = nu / (1.0 - nu)    # ν/(1-ν) is the same for both phases (same ν)

    sigma22_bar = (eps_bar[1] + eps_bar[0] * B) / (f * A_inc + (1-f) * A_mat)
    sigma12_bar = eps_bar[2] / (f / mu_i + (1-f) / mu_m)

    eps22_inc = sigma22_bar * A_inc - eps_bar[0] * B
    eps22_mat = sigma22_bar * A_mat - eps_bar[0] * B
    gam12_inc = sigma12_bar / mu_i
    gam12_mat = sigma12_bar / mu_m

    eps_star   = np.zeros((3, N, N))
    sigma_star = np.zeros((3, N, N))

    inc = phase.astype(bool)
    mat = ~inc

    eps_star[0] = eps_bar[0]
    eps_star[1, inc] = eps22_inc
    eps_star[1, mat] = eps22_mat
    eps_star[2, inc] = gam12_inc
    eps_star[2, mat] = gam12_mat

    sigma_star[1] = sigma22_bar
    sigma_star[2] = sigma12_bar

    for p, (lam_p, mu_p, mask) in enumerate([
            (lam_i, mu_i, inc), (lam_m, mu_m, mat)]):
        eps22_p = eps22_inc if p == 0 else eps22_mat
        sigma_star[0, mask] = (lam_p + 2*mu_p) * eps_bar[0] + lam_p * eps22_p

    tau_star = np.zeros((3, N, N))
    C0 = _C0_voigt(alpha0)
    dC_inc = C_inc - C0
    dC_mat = C_mat - C0
    for a in range(3):
        tau_star[a, inc] = np.dot(dC_inc[a], [eps_bar[0], eps22_inc, gam12_inc])
        tau_star[a, mat] = np.dot(dC_mat[a], [eps_bar[0], eps22_mat, gam12_mat])

    return {
        "eps_star":   eps_star,
        "tau_star":   tau_star,
        "sigma_star": sigma_star,
        "sigma22_bar": sigma22_bar,
        "sigma12_bar": sigma12_bar,
        "eps22_inc":   eps22_inc,
        "eps22_mat":   eps22_mat,
        "gam12_inc":   gam12_inc,
        "gam12_mat":   gam12_mat,
    }


def _eshelby_concentration_tensor(kappa: float, nu: float) -> np.ndarray:
    """
    Dilute strain concentration tensor A = [I + (κ-1) S]⁻¹ for a circular
    inclusion in 2D plane strain with same Poisson ratio in both phases.

    Eshelby tensor for a circular cylinder (Mura 1987):
      S_1111 = S_2222 = (5-4ν) / (8(1-ν))
      S_1122 = S_2211 = (4ν-1) / (8(1-ν))
      S_1212              = (3-4ν) / (8(1-ν))

    In Voigt with engineering shear the (2,2) entry is 2×S_1212.
    """
    S = np.array([
        [(5 - 4*nu) / (8*(1 - nu)),  (4*nu - 1) / (8*(1 - nu)),  0.0              ],
        [(4*nu - 1) / (8*(1 - nu)),  (5 - 4*nu) / (8*(1 - nu)),  0.0              ],
        [0.0,                         0.0,              2*(3 - 4*nu) / (8*(1 - nu))],
    ])
    M = np.eye(3) + (kappa - 1) * S
    return np.linalg.inv(M)


def _eshelby_solution(C_mat: np.ndarray, C_inc: np.ndarray,
                      phase: np.ndarray, eps_bar: np.ndarray,
                      alpha0: float,
                      kappa: float, nu: float) -> dict:
    """
    Dilute Eshelby approximation for the mean strain inside a circular inclusion.

    This is an APPROXIMATE solution valid in the dilute limit (small VF).
    The key physics check is that the strain inside the inclusion is uniform.
    """
    N = phase.shape[0]
    A = _eshelby_concentration_tensor(kappa, nu)
    eps_inc_mean = A @ eps_bar

    inc = phase.astype(bool)
    mat = ~inc

    f = phase.mean()
    eps_mat_mean = (eps_bar - f * eps_inc_mean) / (1 - f)

    eps_star   = np.zeros((3, N, N))
    tau_star   = np.zeros((3, N, N))
    sigma_star = np.zeros((3, N, N))

    C0 = _C0_voigt(alpha0)
    dC_inc = C_inc - C0
    dC_mat = C_mat - C0

    for a in range(3):
        eps_star[a, inc] = eps_inc_mean[a]
        eps_star[a, mat] = eps_mat_mean[a]
        sigma_star[a, inc] = np.dot(C_inc[a], eps_inc_mean)
        sigma_star[a, mat] = np.dot(C_mat[a], eps_mat_mean)
        tau_star[a, inc] = np.dot(dC_inc[a], eps_inc_mean)
        tau_star[a, mat] = np.dot(dC_mat[a], eps_mat_mean)

    return {
        "eps_star":     eps_star,
        "tau_star":     tau_star,
        "sigma_star":   sigma_star,
        "eps_inc_mean": eps_inc_mean,
        "eps_mat_mean": eps_mat_mean,
        "A_tensor":     A,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3D Analytical solutions
# ─────────────────────────────────────────────────────────────────────────────

def _homogeneous_solution_3d(C_mat: np.ndarray, eps_bar: np.ndarray, N: int,
                              alpha0: float) -> dict:
    """
    Exact solution for a 3D homogeneous domain.
    ε(x) = ε̄ everywhere → τ = (C_mat − C⁰) : ε̄ (uniform) → σ = C_mat : ε̄.
    """
    eps_star   = np.zeros((6, N, N, N))
    tau_star   = np.zeros((6, N, N, N))
    sigma_star = np.zeros((6, N, N, N))
    C0 = _C0_voigt_3d(alpha0)
    for a in range(6):
        eps_star[a]   = eps_bar[a]
        tau_star[a]   = np.dot(C_mat[a] - C0[a], eps_bar)
        sigma_star[a] = np.dot(C_mat[a], eps_bar)
    return {"eps_star": eps_star, "tau_star": tau_star, "sigma_star": sigma_star}


def _laminate_solution_3d(C_mat: np.ndarray, C_inc: np.ndarray,
                           phase: np.ndarray, eps_bar: np.ndarray,
                           alpha0: float,
                           E_mat: float, E_inc: float, nu: float) -> dict:
    """
    Exact analytical solution for a 3D horizontal periodic laminate
    (interface normal = e₂; layers are planes of constant x₂).

    Voigt order: (11, 22, 33, 23, 13, 12).  Engineering shear convention.

    Isostrain components: ε₁₁, ε₃₃, γ₁₃  (indices 0, 2, 4)
    Isostress components: σ₂₂, σ₁₂, σ₂₃  (indices 1, 5, 3)

    For isotropic phases with the SAME ν, λ/(λ+2μ) = ν/(1-ν) is phase-independent,
    which yields closed-form averages (same logic as the 2D laminate).
    """
    N = phase.shape[0]
    lam_m, mu_m = lame_from_engineering(E_mat, nu)
    lam_i, mu_i = lame_from_engineering(E_inc, nu)
    M_m = lam_m + 2 * mu_m
    M_i = lam_i + 2 * mu_i
    f = phase.mean()
    B = nu / (1.0 - nu)   # λ/(λ+2μ) same for both phases when ν is shared

    e11, e22, e33 = eps_bar[0], eps_bar[1], eps_bar[2]
    g23, g13, g12 = eps_bar[3], eps_bar[4], eps_bar[5]

    harmonic_M  = f / M_i  + (1 - f) / M_m
    harmonic_mu = f / mu_i + (1 - f) / mu_m

    sigma22_bar = (e22 + (e11 + e33) * B) / harmonic_M
    sigma12_bar = g12 / harmonic_mu
    sigma23_bar = g23 / harmonic_mu

    eps22_inc = (sigma22_bar - lam_i * (e11 + e33)) / M_i
    eps22_mat = (sigma22_bar - lam_m * (e11 + e33)) / M_m
    gam12_inc = sigma12_bar / mu_i
    gam12_mat = sigma12_bar / mu_m
    gam23_inc = sigma23_bar / mu_i
    gam23_mat = sigma23_bar / mu_m

    inc = phase.astype(bool)
    mat = ~inc

    eps_star = np.zeros((6, N, N, N))
    eps_star[0] = e11          # isostrain ε₁₁
    eps_star[2] = e33          # isostrain ε₃₃
    eps_star[4] = g13          # isostrain γ₁₃
    eps_star[1, inc] = eps22_inc;  eps_star[1, mat] = eps22_mat
    eps_star[5, inc] = gam12_inc;  eps_star[5, mat] = gam12_mat
    eps_star[3, inc] = gam23_inc;  eps_star[3, mat] = gam23_mat

    sigma_star = np.zeros((6, N, N, N))
    sigma_star[1] = sigma22_bar    # isostress σ₂₂
    sigma_star[5] = sigma12_bar    # isostress σ₁₂
    sigma_star[3] = sigma23_bar    # isostress σ₂₃

    for lam_p, mu_p, mask, eps22_p in [
            (lam_i, mu_i, inc, eps22_inc),
            (lam_m, mu_m, mat, eps22_mat)]:
        sigma_star[0, mask] = (lam_p + 2*mu_p)*e11 + lam_p*eps22_p + lam_p*e33
        sigma_star[2, mask] = lam_p*e11 + lam_p*eps22_p + (lam_p + 2*mu_p)*e33
        sigma_star[4, mask] = mu_p * g13

    tau_star = np.zeros((6, N, N, N))
    C0 = _C0_voigt_3d(alpha0)
    dC_inc = C_inc - C0
    dC_mat = C_mat - C0
    for a in range(6):
        eps_inc = [e11, eps22_inc, e33, gam23_inc, g13, gam12_inc]
        eps_mat = [e11, eps22_mat, e33, gam23_mat, g13, gam12_mat]
        tau_star[a, inc] = np.dot(dC_inc[a], eps_inc)
        tau_star[a, mat] = np.dot(dC_mat[a], eps_mat)

    return {
        "eps_star":    eps_star,
        "tau_star":    tau_star,
        "sigma_star":  sigma_star,
        "sigma22_bar": sigma22_bar,
        "sigma12_bar": sigma12_bar,
        "sigma23_bar": sigma23_bar,
        "eps22_inc":   eps22_inc,
        "eps22_mat":   eps22_mat,
        "gam12_inc":   gam12_inc,
        "gam12_mat":   gam12_mat,
        "gam23_inc":   gam23_inc,
        "gam23_mat":   gam23_mat,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Microstructure builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_homogeneous(N: int, E_mat: float, nu: float) -> tuple:
    """All matrix — no inclusions (κ=1 for this test type)."""
    phase = np.zeros((N, N), dtype=bool)
    C_mat = isotropic_stiffness_voigt(E_mat, nu)
    C_inc = isotropic_stiffness_voigt(E_mat, nu)   # κ=1: same as matrix
    C_field = build_C_field(phase, C_mat, C_inc)
    return phase, C_field, C_mat, C_inc


def _build_laminate(N: int, E_mat: float, E_inc: float, nu: float) -> tuple:
    """Top half inclusion, bottom half matrix (horizontal layers)."""
    phase = np.zeros((N, N), dtype=bool)
    phase[:, N//2:] = True
    C_mat = isotropic_stiffness_voigt(E_mat, nu)
    C_inc = isotropic_stiffness_voigt(E_inc, nu)
    C_field = build_C_field(phase, C_mat, C_inc)
    return phase, C_field, C_mat, C_inc


def _build_eshelby(N: int, E_mat: float, E_inc: float, nu: float,
                   radius: int = ESHELBY_RADIUS) -> tuple:
    """Single circular inclusion at the centre."""
    phase = np.zeros((N, N), dtype=bool)
    xs, ys = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    cx, cy = N // 2, N // 2
    phase[(xs - cx)**2 + (ys - cy)**2 <= radius**2] = True
    C_mat = isotropic_stiffness_voigt(E_mat, nu)
    C_inc = isotropic_stiffness_voigt(E_inc, nu)
    C_field = build_C_field(phase, C_mat, C_inc)
    return phase, C_field, C_mat, C_inc


def _build_homogeneous_3d(N: int, E_mat: float, nu: float) -> tuple:
    """3D all-matrix domain — no inclusions."""
    phase = np.zeros((N, N, N), dtype=bool)
    C_mat = isotropic_stiffness_voigt_3d(E_mat, nu)
    C_inc = isotropic_stiffness_voigt_3d(E_mat, nu)   # κ=1
    C_field = build_C_field(phase, C_mat, C_inc)
    return phase, C_field, C_mat, C_inc


def _build_laminate_3d(N: int, E_mat: float, E_inc: float, nu: float) -> tuple:
    """3D horizontal laminate — inclusion occupies the x₂>N/2 half-space."""
    phase = np.zeros((N, N, N), dtype=bool)
    phase[:, N // 2:, :] = True   # interface normal = e₂
    C_mat = isotropic_stiffness_voigt_3d(E_mat, nu)
    C_inc = isotropic_stiffness_voigt_3d(E_inc, nu)
    C_field = build_C_field(phase, C_mat, C_inc)
    return phase, C_field, C_mat, C_inc


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 writer
# ─────────────────────────────────────────────────────────────────────────────

def _write_case(grp: h5py.Group, C_field, phase, eps_bar, solution: dict,
                case_type: str, description: str, tolerance: float,
                extra_scalars: dict = None) -> None:
    """Write one test case into an HDF5 group."""
    grp.attrs["description"] = description
    grp.attrs["type"]        = case_type
    grp.attrs["tolerance"]   = tolerance

    grp.create_dataset("C_field",  data=C_field.astype("float32"))
    grp.create_dataset("phase",    data=phase)
    grp.create_dataset("eps_bar",  data=eps_bar.astype("float32"))

    ana = grp.create_group("analytical")
    ana.create_dataset("eps_star",   data=solution["eps_star"].astype("float32"))
    ana.create_dataset("tau_star",   data=solution["tau_star"].astype("float32"))
    ana.create_dataset("sigma_star", data=solution["sigma_star"].astype("float32"))

    if extra_scalars:
        for k, v in extra_scalars.items():
            if isinstance(v, np.ndarray):
                ana.create_dataset(k, data=v.astype("float64"))
            else:
                ana.attrs[k] = float(v)


# ─────────────────────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────────────────────

def generate(config_path: str = None,
             output_path: str = "unittests/test_cases.h5") -> Path:
    """
    Generate analytical test cases and write them to an HDF5 file.

    Dimensionality is read from the ``dim`` key in the config (default 2).

    2D produces 9 cases (3 types × 3 loadings).
    3D produces 12 cases (2 types × 6 loadings).

    Args:
        config_path: Path to a config file (or configs/experiment.yaml by default).
        output_path: Destination HDF5 file.
    """
    # ── Load config ──────────────────────────────────────────────────────────
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "experiment.yaml"
    cfg = load_config(config_path)

    dim   = int(cfg.get("dim", 2))
    N     = int(cfg["N"])
    kappa = float(cfg["kappa"])
    E_mat = float(cfg["E_matrix"])
    E_inc = E_mat * kappa
    nu    = float(cfg["nu_matrix"])
    scale = float(cfg["eps_bar_scale"])

    print(f"  Config: dim={dim}, N={N}, κ={kappa}, E_mat={E_mat}, ν={nu}, scale={scale}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if dim == 2:
        _generate_2d(out, N, kappa, E_mat, E_inc, nu, scale)
    else:
        _generate_3d(out, N, kappa, E_mat, E_inc, nu, scale)

    print(f"\nTest cases saved → {out}  ({out.stat().st_size / 1e3:.1f} kB)")
    return out


def _generate_2d(out: Path, N, kappa, E_mat, E_inc, nu, scale) -> None:
    """Write the 9 two-dimensional test cases."""
    loading_cases = {
        "e11": np.array([scale, 0.0,   0.0  ]),
        "e22": np.array([0.0,   scale, 0.0  ]),
        "g12": np.array([0.0,   0.0,   scale]),
    }

    with h5py.File(out, "w") as f:
        f.attrs["dim"]     = 2
        f.attrs["N"]       = N
        f.attrs["E_mat"]   = E_mat
        f.attrs["E_inc"]   = E_inc
        f.attrs["nu"]      = nu
        f.attrs["kappa"]   = kappa
        f.attrs["scale"]   = scale
        f.attrs["n_cases"] = 9

        # ── Homogeneous (κ=1) ────────────────────────────────────────────────
        phase_h, C_h, C_mat_h, C_inc_h = _build_homogeneous(N, E_mat, nu)
        a0_h = _alpha0(C_mat_h, C_inc_h)

        for load_name, eps_bar in loading_cases.items():
            sol  = _homogeneous_solution(C_mat_h, eps_bar, N, a0_h)
            grp  = f.create_group(f"homo_{load_name}")
            _write_case(
                grp, C_h, phase_h, eps_bar, sol,
                case_type   = "exact",
                description = (f"Homogeneous domain (κ=1), loading={load_name}. "
                               "ε(x)=ε̄ everywhere, τ(x)=0 everywhere."),
                tolerance   = 1e-6,
            )
            print(f"  [homogeneous/{load_name}]  α₀={a0_h:.4f}")

        # ── Laminate ─────────────────────────────────────────────────────────
        phase_l, C_l, C_mat_l, C_inc_l = _build_laminate(N, E_mat, E_inc, nu)
        a0_l = _alpha0(C_mat_l, C_inc_l)

        for load_name, eps_bar in loading_cases.items():
            sol  = _laminate_solution(C_mat_l, C_inc_l, phase_l, eps_bar, a0_l,
                                      E_mat, E_inc, nu)
            grp  = f.create_group(f"laminate_{load_name}")
            extra = {k: v for k, v in sol.items()
                     if k not in ("eps_star", "tau_star", "sigma_star")}
            _write_case(
                grp, C_l, phase_l, eps_bar, sol,
                case_type   = "exact",
                description = (f"Horizontal laminate (VF=0.5, κ={kappa}), loading={load_name}. "
                               "Analytical piecewise-constant fields."),
                tolerance   = 0.02,
                extra_scalars = extra,
            )
            print(f"  [laminate/{load_name}]  "
                  f"ε₂₂_inc={sol['eps22_inc']:.4e}  ε₂₂_mat={sol['eps22_mat']:.4e}  "
                  f"σ̄₂₂={sol['sigma22_bar']:.4e}")

        # ── Eshelby ──────────────────────────────────────────────────────────
        phase_e, C_e, C_mat_e, C_inc_e = _build_eshelby(N, E_mat, E_inc, nu,
                                                          radius=ESHELBY_RADIUS)
        a0_e = _alpha0(C_mat_e, C_inc_e)
        vf_e = phase_e.mean()

        for load_name, eps_bar in loading_cases.items():
            sol  = _eshelby_solution(C_mat_e, C_inc_e, phase_e, eps_bar, a0_e,
                                     kappa, nu)
            grp  = f.create_group(f"eshelby_{load_name}")
            extra = {k: v for k, v in sol.items()
                     if k not in ("eps_star", "tau_star", "sigma_star")}
            _write_case(
                grp, C_e, phase_e, eps_bar, sol,
                case_type   = "approximate",
                description = (f"Single circular inclusion r={ESHELBY_RADIUS} "
                               f"(VF={vf_e:.3f}, κ={kappa}), loading={load_name}. "
                               "Dilute Eshelby approximation."),
                tolerance   = 0.15,
                extra_scalars = extra,
            )
            print(f"  [eshelby/{load_name}]  "
                  f"ε_inc≈{sol['eps_inc_mean']}  VF={vf_e:.3f}")


def _generate_3d(out: Path, N, kappa, E_mat, E_inc, nu, scale) -> None:
    """Write the 12 three-dimensional test cases (2 types × 6 loadings)."""
    z = 0.0
    s = scale
    loading_cases = {
        "e11": np.array([s, z, z, z, z, z]),
        "e22": np.array([z, s, z, z, z, z]),
        "e33": np.array([z, z, s, z, z, z]),
        "g23": np.array([z, z, z, s, z, z]),
        "g13": np.array([z, z, z, z, s, z]),
        "g12": np.array([z, z, z, z, z, s]),
    }

    with h5py.File(out, "w") as f:
        f.attrs["dim"]     = 3
        f.attrs["N"]       = N
        f.attrs["E_mat"]   = E_mat
        f.attrs["E_inc"]   = E_inc
        f.attrs["nu"]      = nu
        f.attrs["kappa"]   = kappa
        f.attrs["scale"]   = scale
        f.attrs["n_cases"] = 12

        # ── Homogeneous (κ=1) ────────────────────────────────────────────────
        phase_h, C_h, C_mat_h, C_inc_h = _build_homogeneous_3d(N, E_mat, nu)
        a0_h = _alpha0(C_mat_h, C_inc_h)

        for load_name, eps_bar in loading_cases.items():
            sol  = _homogeneous_solution_3d(C_mat_h, eps_bar, N, a0_h)
            grp  = f.create_group(f"homo_{load_name}")
            _write_case(
                grp, C_h, phase_h, eps_bar, sol,
                case_type   = "exact",
                description = (f"3D homogeneous domain (κ=1), loading={load_name}. "
                               "ε(x)=ε̄ everywhere, τ(x)=0 everywhere."),
                tolerance   = 1e-6,
            )
            print(f"  [3D homogeneous/{load_name}]  α₀={a0_h:.4f}")

        # ── 3D Laminate ───────────────────────────────────────────────────────
        phase_l, C_l, C_mat_l, C_inc_l = _build_laminate_3d(N, E_mat, E_inc, nu)
        a0_l = _alpha0(C_mat_l, C_inc_l)

        for load_name, eps_bar in loading_cases.items():
            sol  = _laminate_solution_3d(C_mat_l, C_inc_l, phase_l, eps_bar, a0_l,
                                         E_mat, E_inc, nu)
            grp  = f.create_group(f"laminate_{load_name}")
            extra = {k: v for k, v in sol.items()
                     if k not in ("eps_star", "tau_star", "sigma_star")}
            _write_case(
                grp, C_l, phase_l, eps_bar, sol,
                case_type   = "exact",
                description = (f"3D horizontal laminate (VF=0.5, κ={kappa}), loading={load_name}. "
                               "Analytical piecewise-constant fields (isostrain/isostress)."),
                tolerance   = 0.02,
                extra_scalars = extra,
            )
            print(f"  [3D laminate/{load_name}]  "
                  f"ε₂₂_inc={sol['eps22_inc']:.4e}  ε₂₂_mat={sol['eps22_mat']:.4e}  "
                  f"σ̄₂₂={sol['sigma22_bar']:.4e}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Generate analytical test cases from the active experiment config."
    )
    p.add_argument(
        "--config", default=None,
        help="Path to config file (default: configs/experiment.yaml). "
             "Material and grid parameters are read from here.",
    )
    p.add_argument(
        "--output", default="unittests/test_cases.h5",
        help="Output HDF5 file path (default: unittests/test_cases.h5).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print("Generating test cases …")
    generate(config_path=args.config, output_path=args.output)
