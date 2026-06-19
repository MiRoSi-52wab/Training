"""
Voigt ↔ Mandel basis conversion for 2D plane-strain micromechanics.

The rest of this codebase stores tensor quantities in **Voigt notation with
engineering shear**:

    strain     ε_V = [ε₁₁, ε₂₂, γ₁₂]    with  γ₁₂ = 2·ε₁₂
    stress     σ_V = [σ₁₁, σ₂₂, σ₁₂]    (no factor on shear)
    stiffness  C_V — 3×3 matrix with C_V[2,2] = μ

The LS-FNO of Nguyen & Schneider — and most neural-operator papers in
micromechanics — is written in **Mandel notation** instead:

    strain     ε_M = [ε₁₁, ε₂₂, √2·ε₁₂]
    stress     σ_M = [σ₁₁, σ₂₂, √2·σ₁₂]
    stiffness  C_M — 3×3 matrix with C_M[2,2] = 2μ

Why Mandel matters for the LS-FNO:

  1. The tensor double contraction T:ε IS the matrix-vector product T·ε in
     Mandel.  In Voigt with engineering shear it isn't (the row/column factors
     differ for strain vs stress).
  2. The Frobenius norm is preserved: ‖ε‖²_tensor = ε₁₁² + ε₂₂² + 2·ε₁₂² is
     exactly ‖ε_M‖²_euclid, but ≠ ‖ε_V‖²_euclid.  The paper's bounds use the
     tensor / Frobenius norm everywhere.
  3. The stiffness matrix stays symmetric, which the proof of contraction
     of the neural Lippmann–Schwinger operator relies on.

This module provides the conversions in both directions, plus a small
`convert_fields` helper that takes a dict of named fields and dispatches the
right conversion based on the field's tensor kind (strain / stress / stiffness).

All conversions act on a chosen component axis (default: axis 0).  For a
batched dataset of shape (n_samples, 3, N, N), pass `axis=1`; for a batched
stiffness field of shape (n_samples, 3, 3, N, N), pass `axes=(1, 2)`.
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SQRT2     = float(np.sqrt(2.0))
INV_SQRT2 = 1.0 / SQRT2

# Diagonal "D" factor that turns a Voigt stiffness into a Mandel stiffness:
#   C_M[i,j] = D[i] · D[j] · C_V[i,j]      with  D = [1, 1, √2]
_D = np.array([1.0, 1.0, SQRT2], dtype=np.float64)


def _shear_slice(ndim: int, axis: int) -> tuple:
    """Build a slicer that selects the shear component (=2) along `axis`."""
    sl = [slice(None)] * ndim
    sl[axis] = 2
    return tuple(sl)


def _broadcast_D(ndim: int, axis: int) -> np.ndarray:
    """Reshape D = [1, 1, √2] so it broadcasts along the given axis only."""
    shape = [1] * ndim
    shape[axis] = 3
    return _D.reshape(shape)


# ─────────────────────────────────────────────────────────────────────────────
# Strain conversions
# ─────────────────────────────────────────────────────────────────────────────

def voigt_to_mandel_strain(eps_v: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Voigt (engineering shear) → Mandel strain.

    eps_v[..., 2, ...] is γ₁₂ = 2·ε₁₂.
    eps_m[..., 2, ...] is √2·ε₁₂   = γ₁₂ / √2.

    Args:
        eps_v: array with strain components along `axis`.
        axis:  position of the component axis. Use 0 for (3, ...) layout,
               1 for (n_samples, 3, ...).
    """
    eps_v = np.asarray(eps_v, dtype=np.float64)
    out = eps_v.copy()
    sl = _shear_slice(out.ndim, axis)
    out[sl] = eps_v[sl] * INV_SQRT2
    return out


def mandel_to_voigt_strain(eps_m: np.ndarray, axis: int = 0) -> np.ndarray:
    """Mandel → Voigt (engineering shear) strain."""
    eps_m = np.asarray(eps_m, dtype=np.float64)
    out = eps_m.copy()
    sl = _shear_slice(out.ndim, axis)
    out[sl] = eps_m[sl] * SQRT2
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Stress conversions  (polarization stress τ obeys the same rule)
# ─────────────────────────────────────────────────────────────────────────────

def voigt_to_mandel_stress(sig_v: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Voigt → Mandel stress.

    sig_v[..., 2, ...] is σ₁₂ (no factor).
    sig_m[..., 2, ...] is √2·σ₁₂.
    """
    sig_v = np.asarray(sig_v, dtype=np.float64)
    out = sig_v.copy()
    sl = _shear_slice(out.ndim, axis)
    out[sl] = sig_v[sl] * SQRT2
    return out


def mandel_to_voigt_stress(sig_m: np.ndarray, axis: int = 0) -> np.ndarray:
    """Mandel → Voigt stress."""
    sig_m = np.asarray(sig_m, dtype=np.float64)
    out = sig_m.copy()
    sl = _shear_slice(out.ndim, axis)
    out[sl] = sig_m[sl] * INV_SQRT2
    return out


# Polarization stress τ is just a stress field — same rules apply.
voigt_to_mandel_polarization = voigt_to_mandel_stress
mandel_to_voigt_polarization = mandel_to_voigt_stress


# ─────────────────────────────────────────────────────────────────────────────
# Stiffness conversions
# ─────────────────────────────────────────────────────────────────────────────

def voigt_to_mandel_stiffness(C_v: np.ndarray, axes=(0, 1)) -> np.ndarray:
    """
    Voigt → Mandel stiffness.

    C_M[..., i, j, ...] = D[i] · D[j] · C_V[..., i, j, ...]   with D = [1, 1, √2].

    Effect on each block:
        normal-normal (i, j ∈ {0, 1}):       unchanged
        shear row / column (one of i, j = 2): multiplied by √2
        (shear, shear) entry [2, 2]:          multiplied by 2

    Args:
        C_v:  array with the two stiffness component axes at `axes`.
        axes: two-tuple. Defaults to (0, 1) for a (3, 3, ...) layout. Use
              (1, 2) for batched stiffness of shape (n_samples, 3, 3, N, N).
    """
    C_v = np.asarray(C_v, dtype=np.float64)
    ax_i, ax_j = axes
    D_i = _broadcast_D(C_v.ndim, ax_i)
    D_j = _broadcast_D(C_v.ndim, ax_j)
    return C_v * D_i * D_j


def mandel_to_voigt_stiffness(C_m: np.ndarray, axes=(0, 1)) -> np.ndarray:
    """Mandel → Voigt stiffness."""
    C_m = np.asarray(C_m, dtype=np.float64)
    ax_i, ax_j = axes
    D_i = _broadcast_D(C_m.ndim, ax_i)
    D_j = _broadcast_D(C_m.ndim, ax_j)
    return C_m / (D_i * D_j)


# ─────────────────────────────────────────────────────────────────────────────
# Field-name dispatcher (works on any subset of standard field names)
# ─────────────────────────────────────────────────────────────────────────────

_FIELD_KIND = {
    "C_field":    "stiffness",
    "eps_bar":    "strain",
    "eps_star":   "strain",
    "tau_star":   "stress",
    "sigma_star": "stress",
}


def convert_fields(fields: dict,
                   from_basis: str,
                   to_basis: str,
                   strain_axis: int = 0,
                   stress_axis: int = 0,
                   stiffness_axes=(0, 1)) -> dict:
    """
    Convert a dict of named tensor fields between Voigt and Mandel.

    Recognised field names → tensor kind:
        C_field    → stiffness   (3, 3, ...)
        eps_bar    → strain      (3, ...)
        eps_star   → strain      (3, ...)
        tau_star   → stress      (3, ...)
        sigma_star → stress      (3, ...)

    Unknown keys and `None` values pass through unchanged.

    Args:
        fields:        dict of name → ndarray (or None).
        from_basis:    "voigt" or "mandel".
        to_basis:      "voigt" or "mandel".
        strain_axis:   component axis for strain fields (default 0).
        stress_axis:   component axis for stress fields (default 0).
        stiffness_axes: pair of component axes for stiffness fields (default (0, 1)).

    Returns:
        new dict with converted arrays (originals are not modified).
    """
    if from_basis == to_basis:
        return dict(fields)
    if {from_basis, to_basis} != {"voigt", "mandel"}:
        raise ValueError(
            f"Unknown basis pair: {from_basis} → {to_basis}. "
            f"Both must be 'voigt' or 'mandel'."
        )

    forward = (from_basis == "voigt")
    dispatch = {
        "strain":    (voigt_to_mandel_strain    if forward else mandel_to_voigt_strain,
                      {"axis": strain_axis}),
        "stress":    (voigt_to_mandel_stress    if forward else mandel_to_voigt_stress,
                      {"axis": stress_axis}),
        "stiffness": (voigt_to_mandel_stiffness if forward else mandel_to_voigt_stiffness,
                      {"axes": stiffness_axes}),
    }

    out = {}
    for name, value in fields.items():
        kind = _FIELD_KIND.get(name)
        if value is None or kind is None:
            out[name] = value
        else:
            fn, kwargs = dispatch[kind]
            out[name] = fn(value, **kwargs)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Display labels per basis (LaTeX for plots, plain unicode for console tables)
# ─────────────────────────────────────────────────────────────────────────────

LABELS = {
    "voigt": {
        # LaTeX — used by matplotlib titles
        "eps_star":   [r"$\varepsilon_{11}$", r"$\varepsilon_{22}$", r"$\gamma_{12}$"],
        "tau_star":   [r"$\tau_{11}$",        r"$\tau_{22}$",        r"$\tau_{12}$"],
        "sigma_star": [r"$\sigma_{11}$",      r"$\sigma_{22}$",      r"$\sigma_{12}$"],
        "eps_bar":    [r"$\bar{\varepsilon}_{11}$",
                       r"$\bar{\varepsilon}_{22}$",
                       r"$\bar{\gamma}_{12}$"],
        # Unicode — used by console prints
        "eps_star_text":   ["ε₁₁", "ε₂₂", "γ₁₂"],
        "tau_star_text":   ["τ₁₁", "τ₂₂", "τ₁₂"],
        "sigma_star_text": ["σ₁₁", "σ₂₂", "σ₁₂"],
        "shear_strain":    "γ₁₂",
        "shear_stress":    "σ₁₂",
    },
    "mandel": {
        "eps_star":   [r"$\varepsilon_{11}$", r"$\varepsilon_{22}$", r"$\sqrt{2}\,\varepsilon_{12}$"],
        "tau_star":   [r"$\tau_{11}$",        r"$\tau_{22}$",        r"$\sqrt{2}\,\tau_{12}$"],
        "sigma_star": [r"$\sigma_{11}$",      r"$\sigma_{22}$",      r"$\sqrt{2}\,\sigma_{12}$"],
        "eps_bar":    [r"$\bar{\varepsilon}_{11}$",
                       r"$\bar{\varepsilon}_{22}$",
                       r"$\sqrt{2}\,\bar{\varepsilon}_{12}$"],
        "eps_star_text":   ["ε₁₁", "ε₂₂", "√2·ε₁₂"],
        "tau_star_text":   ["τ₁₁", "τ₂₂", "√2·τ₁₂"],
        "sigma_star_text": ["σ₁₁", "σ₂₂", "√2·σ₁₂"],
        "shear_strain":    "√2·ε₁₂",
        "shear_stress":    "√2·σ₁₂",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Self-tests
# ─────────────────────────────────────────────────────────────────────────────

def _self_test() -> None:
    """Quick sanity checks; raises AssertionError on failure."""
    rng = np.random.default_rng(0)

    # Round-trip on strain / stress / stiffness with assorted shapes
    for shape in [(3,), (3, 5), (3, 5, 5)]:
        x = rng.standard_normal(shape)
        assert np.allclose(mandel_to_voigt_strain(voigt_to_mandel_strain(x)), x)
        assert np.allclose(mandel_to_voigt_stress(voigt_to_mandel_stress(x)), x)

    for shape in [(3, 3), (3, 3, 5, 5)]:
        C = rng.standard_normal(shape)
        assert np.allclose(mandel_to_voigt_stiffness(voigt_to_mandel_stiffness(C)), C)

    # Constitutive consistency: σ = C : ε in both bases must agree
    E, nu = 1.0, 0.3
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu  = E / (2 * (1 + nu))
    C_V = np.array([[lam + 2 * mu, lam,           0.0],
                    [lam,           lam + 2 * mu, 0.0],
                    [0.0,           0.0,          mu]])
    eps_V = np.array([0.003, -0.001, 0.0042])           # γ₁₂ = 0.0042

    sig_V = C_V @ eps_V
    sig_M_via_conv = voigt_to_mandel_stress(sig_V)
    sig_M_direct   = voigt_to_mandel_stiffness(C_V) @ voigt_to_mandel_strain(eps_V)
    assert np.allclose(sig_M_via_conv, sig_M_direct), \
        f"C_M·ε_M ({sig_M_direct}) ≠ Mandel-converted σ_V ({sig_M_via_conv})"

    # Frobenius-norm preservation: ‖ε‖²_tensor == ‖ε_M‖²_euclid
    eps_12_tensor = eps_V[2] / 2.0
    tensor_norm_sq = eps_V[0] ** 2 + eps_V[1] ** 2 + 2.0 * eps_12_tensor ** 2
    eps_M = voigt_to_mandel_strain(eps_V)
    mandel_norm_sq = float(np.sum(eps_M ** 2))
    assert np.isclose(tensor_norm_sq, mandel_norm_sq), \
        f"Frobenius mismatch: tensor={tensor_norm_sq}, mandel={mandel_norm_sq}"

    # Batched conversion via convert_fields with non-default axes
    batch = {
        "eps_star":  rng.standard_normal((4, 3, 6, 6)),     # axis=1 component
        "C_field":   rng.standard_normal((4, 3, 3, 6, 6)),  # axes=(1, 2)
    }
    out = convert_fields(batch, "voigt", "mandel",
                         strain_axis=1, stress_axis=1, stiffness_axes=(1, 2))
    back = convert_fields(out, "mandel", "voigt",
                          strain_axis=1, stress_axis=1, stiffness_axes=(1, 2))
    assert np.allclose(back["eps_star"], batch["eps_star"])
    assert np.allclose(back["C_field"],  batch["C_field"])

    print("notation.py self-test passed.")


if __name__ == "__main__":
    _self_test()
