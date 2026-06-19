import sys; sys.path.insert(0, '.')
import builtins
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
C_t=torch.from_numpy(C_field).float().unsqueeze(0)

# FFT reference (uniaxial-x)
eb=np.zeros(6); eb[0]=1e-3
rf=fft_solve(C_field,eb,alpha0=a0,tol=1e-5,max_iter=2000,discretization='staggered')
C11_fft=rf['sigma_star'][0].mean()/1e-3
print(f"kappa={kappa}  FFT: n_iter={rf['n_iter']}  C11={C11_fft:.4f}")
print(f"{'scale':>7} {'n_iter':>7} {'conv':>5} {'final_res':>11} {'C11':>9} {'relerr':>9} {'max|eps|':>9}")

# suppress solve()'s internal prints
_p = builtins.print
for scale in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]:
    model=LSFNO(grid_size=N,depth_K=4,alpha_minus=am,alpha_plus=ap,tol=1e-5,
                max_iter=600,tau_theta=YarotskyTauTheta(depth_m=11,cutoff_M=1.0),
                dim=3,discretization='staggered')
    eb=torch.zeros(1,6); eb[0,0]=scale
    builtins.print=lambda *a,**k: None
    res=model.solve(C_t,eb)
    builtins.print=_p
    es=res['eps_star']
    sigma=torch.einsum('bijxyz,bjxyz->bixyz',C_t,es)
    C11=(sigma.mean(dim=(-3,-2,-1))[0,0]/scale).item()
    relerr=abs(C11-C11_fft)/abs(C11_fft)
    print(f"{scale:>7.2f} {res['n_iter']:>7} {str(res['converged']):>5} "
          f"{res['residuals'][-1]:>11.3e} {C11:>9.4f} {relerr:>9.2e} {es.abs().max().item():>9.4f}")