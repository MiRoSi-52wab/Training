import sys, builtins; sys.path.insert(0, '.')
import numpy as np, torch
from generation.microstructure import isotropic_stiffness_voigt_3d, build_C_field
from generation.fft_solver import solve as fft_solve
from models.ls_fno import LSFNO, YarotskyTauTheta
torch.set_grad_enabled(False)

N=32; r=10.0; c=(N-1)/2.0
xs,ys,zs = np.mgrid[0:N,0:N,0:N]
phase = ((xs-c)**2+(ys-c)**2+(zs-c)**2)<=r**2

E_MAT=3.0; NU_MAT=0.3; NU_INC=0.22; kappa=12
lam_m = E_MAT*NU_MAT/((1+NU_MAT)*(1-2*NU_MAT))
mu_m  = E_MAT/(2*(1+NU_MAT))
E_inc = E_MAT*kappa
lam_i = E_inc*NU_INC/((1+NU_INC)*(1-2*NU_INC))
mu_i  = E_inc/(2*(1+NU_INC))

C_field = build_C_field(phase,
    isotropic_stiffness_voigt_3d(E_MAT, NU_MAT),
    isotropic_stiffness_voigt_3d(E_inc, NU_INC))
C_t32 = torch.from_numpy(C_field).float().unsqueeze(0)
C_t64 = torch.from_numpy(C_field).double().unsqueeze(0)

alpha_plus = 3*lam_i + 2*mu_i          # same for Voigt and Mandel
am_voigt   = mu_m                       # current (wrong for Mandel)
am_mandel  = 2*mu_m                     # correct for Mandel
a0_voigt   = (am_voigt  + alpha_plus)/2
a0_mandel  = (am_mandel + alpha_plus)/2

_p = builtins.print

def run(label, alpha_minus, alpha_plus, dtype):
    a0 = (alpha_minus + alpha_plus)/2
    C_t = C_t64 if dtype == torch.float64 else C_t32
    tau = YarotskyTauTheta(depth_m=11, cutoff_M=1.0)
    if dtype == torch.float64:
        # need to rebuild tau with float64 — the _q ops are dtype-agnostic in torch
        pass
    model = LSFNO(grid_size=N, depth_K=4,
                  alpha_minus=float(alpha_minus), alpha_plus=float(alpha_plus),
                  tol=1e-5, max_iter=300, tau_theta=tau, dim=3, discretization='staggered')
    if dtype == torch.float64:
        model = model.double()
    eb = torch.zeros(1, 6, dtype=dtype); eb[0,0] = 1e-3
    builtins.print = lambda *a,**k: None
    res = model.solve(C_t.to(dtype), eb)
    builtins.print = _p
    rs = res['residuals']
    _p(f"  {label:45s}  a0={a0:.3f}  γ={(alpha_plus-alpha_minus)/(alpha_plus+alpha_minus):.4f}  "
       f"n_iter={res['n_iter']:>4}  conv={str(res['converged']):>5}  "
       f"final_res={rs[-1]:.3e}  "
       f"len(res) = {len(rs)}"
    )

    if len(res) >= 50:
        _p(f"res@50={rs[49]:.3e}")

    if len(res) >= 100:
        _p(f"res@100={rs[99]:.3e}")
    

print(f"{'label':45s}  {'info':40s}")
print("-"*100)
#run("float32  wrong  alpha_minus=mu",       am_voigt,  alpha_plus, torch.float32)
#run("float32  fixed  alpha_minus=2*mu",     am_mandel, alpha_plus, torch.float32)
run("float64  wrong  alpha_minus=mu",       am_voigt,  alpha_plus, torch.float64)
run("float64  fixed  alpha_minus=2*mu",     am_mandel, alpha_plus, torch.float64)