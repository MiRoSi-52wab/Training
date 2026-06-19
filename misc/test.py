import sys; sys.path.insert(0, '.')
import numpy as np, torch
from generation.microstructure import isotropic_stiffness_voigt_3d, build_C_field
from generation.fft_solver import solve as fft_solve
from models.ls_fno import LSFNO, YarotskyTauTheta
from utils.config_loader import compute_alpha_bounds
torch.set_grad_enabled(False)

N=32; r=10.0
c=(N-1)/2.0
xs,ys,zs=np.mgrid[0:N,0:N,0:N]
phase=((xs-c)**2+(ys-c)**2+(zs-c)**2)<=r**2

E_MAT=3.0; NU_MAT=0.3; NU_INC=0.22; kappa=12
C_mat=isotropic_stiffness_voigt_3d(E_MAT,NU_MAT)
C_inc=isotropic_stiffness_voigt_3d(E_MAT*kappa,NU_INC)
C_field=build_C_field(phase,C_mat,C_inc)
am,ap=compute_alpha_bounds(E_MAT,NU_MAT,NU_INC,kappa,dim=3)
a0=(am+ap)/2
print(f'alpha_minus={am:.4f} alpha_plus={ap:.4f} alpha0={a0:.4f}  gamma=(ap-am)/(ap+am)={(ap-am)/(ap+am):.4f}')

C_t=torch.from_numpy(C_field).float().unsqueeze(0)

# FFT reference, uniaxial x, scale 1e-3
eb=np.zeros(6); eb[0]=1e-3
rf=fft_solve(C_field,eb,alpha0=a0,tol=1e-5,max_iter=2000,discretization='staggered')
print(f'FFT  uniaxial-x: n_iter={rf["n_iter"]} conv={rf["converged"]} C11={rf["sigma_star"][0].mean()/1e-3:.4f}')

# FNO11 at several strain scales (problem is LINEAR -> C11 should be scale-invariant)
print()
print(f'{"scale":>8} {"n_iter":>7} {"conv":>5} {"final_res":>12} {"C11":>10} {"max|eps_loc|":>12}')
for scale in [1.0]:
    model=LSFNO(grid_size=N,depth_K=4,alpha_minus=am,alpha_plus=ap,tol=1e-5,
                max_iter=600,tau_theta=YarotskyTauTheta(depth_m=11,cutoff_M=1.0),
                dim=3,discretization='staggered')
    eb=torch.zeros(1,6); eb[0,0]=scale
    # silence the prints inside solve by monkeypatching builtins.print? simpler: capture
    #import io, contextlib
    print("solve")
    #buf=io.StringIO()
    #with contextlib.redirect_stdout(buf):
        
    res=model.solve(C_t,eb)
    eps_star=res['eps_star']
    sigma=torch.einsum('bijxyz,bjxyz->bixyz',C_t,eps_star)
    C11=(sigma.mean(dim=(-3,-2,-1))[0,0]/scale).item()
    max_eps=eps_star.abs().max().item()
    fr=res['residuals'][-1]
    print(f'{scale:>8.0e} {res["n_iter"]:>7} {str(res["converged"]):>5} {fr:>12.3e} {C11:>10.4f} {max_eps:>12.4e}')
                                                    