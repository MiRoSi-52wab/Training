"""
Scenario definitions for the comparative study.

27 simulation scenarios in 6 groups (A–F), each comparing the nonlinear
FFT solver against the nonlinear KAN-FNO with analytically initialised
B-spline edge functions.

Usage
-----
    from study.scenarios import SCENARIOS, list_scenarios

    params = SCENARIOS['A1']   # dict with all simulation parameters
    list_scenarios()            # print all available IDs and descriptions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

# ---------------------------------------------------------------------------
# Base material constants (Moulinec & Suquet 1994, eq. 9)
# ---------------------------------------------------------------------------
E_F_BASE   = 400_000.0   # elastic fiber Young's modulus (MPa)
NU_F_BASE  = 0.23        # elastic fiber Poisson's ratio

E_M_BASE   =  68_900.0   # elastoplastic matrix Young's modulus (MPa)
NU_M_BASE  = 0.35        # matrix Poisson's ratio

SY_BASE    =  68.9       # initial yield stress of matrix (MPa)
H_LIN      = 1_710.0     # isotropic hardening modulus, linear case (MPa)

N_DEFAULT  = 64          # grid resolution
N_STEPS    = 20          # load steps (default)
VF         = 0.475       # fiber volume fraction


# ---------------------------------------------------------------------------
# Loading-path helpers
# ---------------------------------------------------------------------------

def _uniaxial_x(eps_max: float, n_steps: int = N_STEPS) -> np.ndarray:
    """Uniaxial macroscopic strain in x-direction: E_xx ramp, E_yy=E_xy=0."""
    path = np.zeros((n_steps, 3))
    path[:, 0] = np.linspace(0.0, eps_max, n_steps + 1)[1:]
    return path


def _uniaxial_x_with_lateral(
    eps_max: float, nu_eff: float = NU_M_BASE, n_steps: int = N_STEPS
) -> np.ndarray:
    """Uniaxial x with macroscopic lateral contraction E_yy = -nu_eff * E_xx."""
    path = np.zeros((n_steps, 3))
    eps  = np.linspace(0.0, eps_max, n_steps + 1)[1:]
    path[:, 0] = eps
    path[:, 1] = -nu_eff * eps
    return path


def _equibiaxial(eps_max: float, n_steps: int = N_STEPS) -> np.ndarray:
    """Equibiaxial tension: E_xx = E_yy = ramp, E_xy = 0."""
    path = np.zeros((n_steps, 3))
    eps  = np.linspace(0.0, eps_max, n_steps + 1)[1:]
    path[:, 0] = eps
    path[:, 1] = eps
    return path


def _pure_shear(gamma_max: float, n_steps: int = N_STEPS) -> np.ndarray:
    """Pure shear: engineering shear E_xy = ramp, E_xx = E_yy = 0."""
    path = np.zeros((n_steps, 3))
    path[:, 2] = np.linspace(0.0, gamma_max, n_steps + 1)[1:]
    return path


# ---------------------------------------------------------------------------
# Scenario dictionary
# ---------------------------------------------------------------------------
# Each entry:
#   E_m, nu_m       — matrix material
#   E_f, nu_f       — fiber material
#   sigma_y, H      — von Mises + isotropic hardening
#   N               — grid resolution
#   vf              — fiber volume fraction
#   eps_bar_path    — (n_steps, 3) macroscopic strain path (Voigt: [ε₁₁, ε₂₂, γ₁₂])
#   alpha0_factor   — alpha0 = alpha_opt * factor (1.0 = M-S optimal)
#   max_iter        — maximum LS iterations per step
#   description     — human-readable label
# ---------------------------------------------------------------------------

SCENARIOS: dict = {}


def _add(sid: str, desc: str, **kwargs) -> None:
    base = dict(
        E_m=E_M_BASE, nu_m=NU_M_BASE,
        E_f=E_F_BASE, nu_f=NU_F_BASE,
        sigma_y=SY_BASE, H=H_LIN,
        N=N_DEFAULT, vf=VF,
        eps_bar_path=_uniaxial_x(0.01),
        alpha0_factor=1.0,
        max_iter=1000,
    )
    base.update(kwargs)
    base['description'] = desc
    SCENARIOS[sid] = base


# ── Group A — hardening law ────────────────────────────────────────────────

_add('A1', 'Linear hardening (M&S benchmark)',
     sigma_y=68.9, H=1_710.0)

_add('A2', 'Perfect plasticity (H=0)',
     sigma_y=68.9, H=0.0)

_add('A3', 'Higher yield stress (σ_y=150 MPa)',
     sigma_y=150.0, H=1_710.0)

_add('A4', 'Elastic only (σ_y→∞, never yields)',
     sigma_y=1e9, H=0.0)


# ── Group B — material contrast κ = E_f / E_m ─────────────────────────────
# κ fixed by choosing E_m; E_f = 400 000 MPa throughout.

_add('B1', 'Contrast κ=12  (E_m=33 333 MPa)',
     E_m=400_000.0 / 12.0)

_add('B2', 'Contrast κ=24  (E_m=16 667 MPa)',
     E_m=400_000.0 / 24.0)

_add('B3', 'Contrast κ=48  (E_m=8 333 MPa)',
     E_m=400_000.0 / 48.0)

_add('B4', 'Contrast κ=96  (E_m=4 167 MPa)',
     E_m=400_000.0 / 96.0)


# ── Group C — strain magnitude (B1 geometry, A1 plasticity) ────────────────

_add('C1', 'Small strain 0.1% (predominantly elastic)',
     E_m=400_000.0 / 12.0,
     eps_bar_path=_uniaxial_x(0.001))

_add('C2', 'Moderate strain 0.5%',
     E_m=400_000.0 / 12.0,
     eps_bar_path=_uniaxial_x(0.005))

_add('C3', 'Benchmark strain 1.0%',
     E_m=400_000.0 / 12.0,
     eps_bar_path=_uniaxial_x(0.010))

_add('C4', 'Large strain 5.0% (heavily plastic)',
     E_m=400_000.0 / 12.0,
     eps_bar_path=_uniaxial_x(0.050))


# ── Group D — loading direction (B1 geometry, A1 plasticity) ───────────────

_add('D1', 'Uniaxial tension (M&S benchmark)',
     E_m=400_000.0 / 12.0,
     eps_bar_path=_uniaxial_x(0.01))

_add('D2', 'Uniaxial with lateral contraction (ν_eff = 0.35)',
     E_m=400_000.0 / 12.0,
     eps_bar_path=_uniaxial_x_with_lateral(0.01, nu_eff=NU_M_BASE))

_add('D3', 'Equibiaxial tension',
     E_m=400_000.0 / 12.0,
     eps_bar_path=_equibiaxial(0.01))

_add('D4', 'Pure shear (γ_xy = 1%)',
     E_m=400_000.0 / 12.0,
     eps_bar_path=_pure_shear(0.01))


# ── Group E — α₀ sweep at κ = 12 (B1 geometry, A1 plasticity) ─────────────
# alpha0 = alpha_opt * alpha0_factor

_add('E1', 'α₀/α_opt = 0.25  (γ₀ ≈ 0.972,  ~7× more iters)',
     E_m=400_000.0 / 12.0,
     alpha0_factor=0.25)

_add('E2', 'α₀/α_opt = 0.50  (γ₀ ≈ 0.944,  ~4× more iters)',
     E_m=400_000.0 / 12.0,
     alpha0_factor=0.50)

_add('E3', 'α₀/α_opt = 1.00  (optimal baseline)',
     E_m=400_000.0 / 12.0,
     alpha0_factor=1.00)

_add('E4', 'α₀/α_opt = 1.50  (γ₀ ≈ 0.879,  ~1.3× more iters)',
     E_m=400_000.0 / 12.0,
     alpha0_factor=1.50)

_add('E5', 'α₀/α_opt = 2.00  (γ₀ ≈ 0.920,  ~2.3× more iters)',
     E_m=400_000.0 / 12.0,
     alpha0_factor=2.00)

_add('E6', 'α₀/α_opt = 4.00  (γ₀ ≈ 0.972,  ~7× more iters)',
     E_m=400_000.0 / 12.0,
     alpha0_factor=4.00)

_add('E7', 'α₀/α_opt = 8.00  (γ₀ > 1, divergence expected)',
     E_m=400_000.0 / 12.0,
     alpha0_factor=8.00,
     max_iter=5000)


# ── Group F — α₀ sweep at κ = 48 (B3 geometry, A1 plasticity) ─────────────

_add('F1', 'κ=48, α₀/α_opt = 1.00  (optimal)',
     E_m=400_000.0 / 48.0,
     alpha0_factor=1.00)

_add('F2', 'κ=48, α₀/α_opt = 1.50  (near stability boundary)',
     E_m=400_000.0 / 48.0,
     alpha0_factor=1.50)

_add('F3', 'κ=48, α₀/α_opt = 2.00  (very near boundary)',
     E_m=400_000.0 / 48.0,
     alpha0_factor=2.00)

_add('F4', 'κ=48, α₀/α_opt = 3.00  (divergence expected)',
     E_m=400_000.0 / 48.0,
     alpha0_factor=3.00,
     max_iter=5000)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def list_scenarios() -> None:
    """Print all scenario IDs and descriptions."""
    groups = {}
    for sid in sorted(SCENARIOS):
        g = sid[0]
        groups.setdefault(g, []).append(sid)

    group_desc = {
        'A': 'Hardening law',
        'B': 'Material contrast κ',
        'C': 'Applied strain magnitude',
        'D': 'Loading direction',
        'E': 'Reference stiffness α₀ at κ=12',
        'F': 'Reference stiffness α₀ at κ=48',
    }
    for g, ids in sorted(groups.items()):
        print(f"\nGroup {g} — {group_desc.get(g, '')}")
        for sid in ids:
            print(f"  {sid}: {SCENARIOS[sid]['description']}")


if __name__ == '__main__':
    list_scenarios()
    print(f"\nTotal: {len(SCENARIOS)} scenarios")
