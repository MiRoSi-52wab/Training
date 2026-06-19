import sys; sys.path.insert(0, '.')
import numpy as np, torch
from generation.microstructure import isotropic_stiffness_voigt_3d, build_C_field
from models.ls_fno import LSFNO, YarotskyTauTheta
from utils.config_loader import compute_alpha_bounds
torch.set_grad_enabled(False)

N=32; r=10.0
c=(N-1)/2.0; xs,ys,zs=np.mgrid[0:N,0:N,0:N]
phase=((xs-c)**2+(ys-c)**2+(zs-c)**2)<=r**2

E_MAT=3.0; NU_MAT=0.3; NU_INC=0.22; kappa=12
lam_mat = E_MAT*NU_MAT/((1+NU_MAT)*(1-2*NU_MAT))
mu_mat  = E_MAT/(2*(1+NU_MAT))
E_inc   = E_MAT*kappa
lam_inc = E_inc*NU_INC/((1+NU_INC)*(1-2*NU_INC))
mu_inc  = E_inc/(2*(1+NU_INC))

print("=== Material eigenvalues ===")
print(f"  C_Voigt min eig  = mu_mat         = {mu_mat:.4f}  GPa")
print(f"  C_Mandel min eig = 2*mu_mat        = {2*mu_mat:.4f}  GPa  (used internally by FNO)")
print(f"  C_Voigt  max eig = 3*lam_inc+2*mu_inc = {3*lam_inc+2*mu_inc:.4f} GPa")
print(f"  C_Mandel max eig = 3*lam_inc+2*mu_inc = {3*lam_inc+2*mu_inc:.4f} GPa  (same)")

am_voigt, ap = compute_alpha_bounds(E_MAT, NU_MAT, NU_INC, kappa, dim=3)
am_mandel    = 2 * mu_mat
a0_voigt  = (am_voigt + ap) / 2
a0_mandel = (am_mandel + ap) / 2

print(f"\n=== alpha_0 comparison ===")
print(f"  compute_alpha_bounds gives alpha_minus = {am_voigt:.4f}  (Voigt min eig)")
print(f"  Mandel-correct alpha_minus             = {am_mandel:.4f}  (Mandel min eig)")
print(f"  alpha0 currently used                  = {a0_voigt:.4f}")
print(f"  alpha0 Mandel-correct                  = {a0_mandel:.4f}")
print(f"  gamma (current) = {(ap-am_voigt)/(ap+am_voigt):.4f}")
print(f"  gamma (correct) = {(ap-am_mandel)/(ap+am_mandel):.4f}")

# Build C_field and compute T in Mandel (as the FNO does it internally)
C_mat = isotropic_stiffness_voigt_3d(E_MAT, NU_MAT)
C_inc = isotropic_stiffness_voigt_3d(E_inc, NU_INC)
C_field = build_C_field(phase, C_mat, C_inc)
C_t = torch.from_numpy(C_field).float().unsqueeze(0)  # (1, 6, 6, 32, 32, 32)

# Replicate _to_mandel_state from the model using current a0_voigt
SQRT2 = 2.0**0.5
D = torch.ones(6); D[3:] = SQRT2
D_outer = D[:,None]*D[None,:]  # (6,6)
C_M = C_t * D_outer.reshape(1,6,6,1,1,1)

def check_T(alpha0, label):
    C0 = torch.eye(6) * alpha0
    T = (C_M - C0.reshape(1,6,6,1,1,1)) / alpha0  # (1,6,6,32,32,32)
    T_comp_max  = T.abs().max().item()
    # Per-voxel operator norm (largest singular value of 6x6 matrix at each voxel)
    T_flat = T[0].reshape(6, 6, -1).permute(2, 0, 1)  # (N^3, 6, 6)
    svd_vals = torch.linalg.svdvals(T_flat)  # (N^3, 6)
    op_norm_max = svd_vals[:,0].max().item()
    print(f"\n  [{label}]  alpha0={alpha0:.4f}")
    print(f"    max |T_ij| (component-wise) = {T_comp_max:.4f}  {'✓ <1' if T_comp_max<1 else '✗ >1!'}")
    print(f"    max ||T||_op (spectral norm) = {op_norm_max:.4f}  {'✓ <1' if op_norm_max<1 else '✗ >1!'}")

print(f"\n=== T field bounds (Mandel) ===")
check_T(a0_voigt,  "current  (alpha_minus=mu_mat)")
check_T(a0_mandel, "corrected (alpha_minus=2*mu_mat)")

# Check C0 identity: does alpha0*I6 give the right C0 for the basic scheme?
print(f"\n=== C0 identity check ===")
eps_test = torch.tensor([0.001, 0.0, 0.0, 0.0, 0.0, 0.0])
C0_current = a0_voigt * torch.eye(6)
C0_correct = a0_mandel * torch.eye(6)
print(f"  C0_current @ eps  = {(C0_current @ eps_test).tolist()}")
print(f"  a0_current * eps  = {(a0_voigt * eps_test).tolist()}")
print(f"  (both should match - checking C0=alpha0*I6 is correct)")

# Float32 vs float64 precision test for q subtraction
print(f"\n=== float32 catastrophic cancellation test ===")
T_val = 0.46; M = 1.0
for eps_val in [0.001, 0.01, 0.1]:
    a32 = torch.tensor(T_val, dtype=torch.float32)
    b32 = torch.tensor(eps_val, dtype=torch.float32)
    a64 = torch.tensor(T_val, dtype=torch.float64)
    b64 = torch.tensor(eps_val, dtype=torch.float64)
    # compute q at (a+b)/(2M) and a/(2M) in float32 vs float64
    x32_ab = (a32+b32)/(2*M); x32_a = a32/(2*M)
    x64_ab = (a64+b64)/(2*M); x64_a = a64/(2*M)
    q32 = x32_ab**2 - x32_a**2   # approx of derivative * b/2 (simplified)
    q64 = x64_ab**2 - x64_a**2
    diff32 = float(2*q32)  # m_theta approx
    diff64 = float(2*q64)
    expected = T_val * eps_val  # true product
    err32 = abs(diff32 - expected)/abs(expected)
    err64 = abs(diff64 - expected)/abs(expected)
    print(f"  eps={eps_val:.3f}: float32 rel_err={err32:.2e}  float64 rel_err={err64:.2e}  exact={expected:.4e}")