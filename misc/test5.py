import sys
sys.path.insert(0, '.')
from replicate.paper2_alpha_calibration import (
    make_stiffness, make_model, make_perturbed_C_field,
    gamma_empirical, theoretical_n_iter, invert_alpha,
    run_alpha_sweep, run_fno_strain_sweep,
    print_sensitivity_table,
    KAPPAS, ALPHA_SWEEP, STRAIN_MAGS, M_DEPTHS
)
import numpy as np
print('imports OK')

# Quick sanity checks (no solver calls)
# 1. perturbed C_field with alpha=0 should round-trip to original
import numpy as np
from generation.microstructure import isotropic_stiffness_voigt_3d, build_C_field
from replicate.paper2_alpha_calibration import centered_sphere, N, SPHERE_RADIUS, E_MATRIX, NU_MATRIX, NU_INCLUSION
phase   = centered_sphere(N, SPHERE_RADIUS)
C_mat   = isotropic_stiffness_voigt_3d(E_MATRIX, NU_MATRIX)
C_inc   = isotropic_stiffness_voigt_3d(E_MATRIX * 12, NU_INCLUSION)
C_field = build_C_field(phase, C_mat, C_inc)
from utils.config_loader import compute_alpha_bounds
alpha_m, alpha_p = compute_alpha_bounds(E_MATRIX, NU_MATRIX, NU_INCLUSION, 12, dim=3)
alpha0 = (alpha_m + alpha_p) / 2.0
C_pert0 = make_perturbed_C_field(C_field, alpha0, 0.0)
assert np.allclose(C_pert0, C_field.astype('float64')), 'alpha=0 perturbation must be identity'
print('make_perturbed_C_field(alpha=0) == original  OK')

# 2. theoretical_n_iter at alpha=0 should return n0
gamma = 0.93
n_theory = theoretical_n_iter(100.0, gamma, np.array([0.0]))
assert abs(n_theory[0] - 100.0) < 0.1, f'expected 100, got {n_theory[0]}'
print('theoretical_n_iter(alpha=0) == N0  OK')

# 3. invert_alpha should be self-consistent with theoretical_n_iter
gamma = 0.979
n0 = 384.0
for alpha_true in [-0.01, -0.001, 0.0, 0.001, 0.01]:
    n_alpha = float(theoretical_n_iter(n0, gamma, np.array([alpha_true]))[0])
    alpha_recovered = invert_alpha(int(round(n_alpha)), int(n0), gamma)
    assert abs(alpha_recovered - alpha_true) < 1e-4 * max(abs(alpha_true), 1e-6) + 1e-6, \
        f'inversion failed: true={alpha_true}, recovered={alpha_recovered}'
print('invert_alpha round-trip  OK')
print()
print('All sanity checks passed.')