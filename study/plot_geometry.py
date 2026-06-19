"""
Visualise the M&S centered-disk microstructure geometry.

Produces a figure with:
  Row 0: phase map (fiber/matrix) | C₁₁₁₁ stiffness field
  Row 1: C₁₁₂₂ stiffness field   | C₁₂₁₂ (shear) stiffness field

Usage (from LS_KAN_FNO/ directory):
    python study/plot_geometry.py
    python study/plot_geometry.py --N 128
    python study/plot_geometry.py --kappa 12      # Group B contrast
    python study/plot_geometry.py --save geo.png
"""
"""
# Default (N=64, M&S parameters)
python study/plot_geometry.py

# Higher resolution
python study/plot_geometry.py --N 128

# Different contrast (Group B scenarios)
python study/plot_geometry.py --kappa 12    # B1
python study/plot_geometry.py --kappa 48    # B3

# Save to file
python study/plot_geometry.py --save geo.png
python study/plot_geometry.py --kappa 12 --N 128 --save geo_k12.png"""



import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from study.geometry import make_ms_geometry, VF_DEFAULT, E_F_DEFAULT, NU_F_DEFAULT
from study.geometry import E_M_DEFAULT, NU_M_DEFAULT


def parse_args():
    p = argparse.ArgumentParser(description="Plot M&S microstructure geometry")
    p.add_argument('--N',      type=int,   default=64,           help='Grid resolution (default 64)')
    p.add_argument('--vf',     type=float, default=VF_DEFAULT,   help='Fiber volume fraction (default 0.475)')
    p.add_argument('--E-m',    type=float, default=E_M_DEFAULT,  help='Matrix Young modulus (MPa)')
    p.add_argument('--nu-m',   type=float, default=NU_M_DEFAULT, help='Matrix Poisson ratio')
    p.add_argument('--E-f',    type=float, default=E_F_DEFAULT,  help='Fiber Young modulus (MPa)')
    p.add_argument('--nu-f',   type=float, default=NU_F_DEFAULT, help='Fiber Poisson ratio')
    p.add_argument('--kappa',  type=float, default=None,
                   help='Contrast κ = E_f/E_m — overrides --E-m (sets E_m = E_f / κ)')
    p.add_argument('--save',   type=str,   default=None,         help='Save figure to this path')
    return p.parse_args()


def main():
    args = parse_args()

    E_f  = args.E_f
    nu_f = args.nu_f
    E_m  = E_f / args.kappa if args.kappa is not None else args.E_m
    nu_m = args.nu_m
    N    = args.N
    vf   = args.vf

    kappa = E_f / E_m
    print(f"Geometry parameters:")
    print(f"  N = {N}×{N}  (voxels)")
    print(f"  V_f = {vf:.3f}  ({vf*100:.1f}%)")
    print(f"  Fiber : E_f = {E_f:.0f} MPa,  ν_f = {nu_f}")
    print(f"  Matrix: E_m = {E_m:.1f} MPa,  ν_m = {nu_m}")
    print(f"  Contrast κ = E_f / E_m = {kappa:.1f}")

    phase, C_field = make_ms_geometry(N=N, E_m=E_m, nu_m=nu_m,
                                       E_f=E_f, nu_f=nu_f, vf=vf)

    vf_actual = phase.mean()
    print(f"  Actual V_f (pixel count) = {vf_actual:.4f}")

    try:
        import matplotlib
        matplotlib.use('Agg' if args.save else 'TkAgg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not installed — cannot plot.")
        return

    fig = plt.figure(figsize=(13, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30)

    extent = [0, N, 0, N]

    # ── Panel (0,0): phase map ─────────────────────────────────────────────────
    ax00 = fig.add_subplot(gs[0, 0])
    cmap_phase = plt.cm.colors.ListedColormap(['#4e79a7', '#f28e2b'])  # matrix/fiber
    im00 = ax00.imshow(phase.T.astype(float), origin='lower',
                       extent=extent, cmap='coolwarm', vmin=0, vmax=1, aspect='equal')
    cb00 = plt.colorbar(im00, ax=ax00, fraction=0.046, pad=0.04, ticks=[0, 1])
    cb00.set_ticklabels(['Matrix\n(elastic-plastic)', 'Fiber\n(elastic)'])
    ax00.set_title(f'Phase map  (V_f = {vf_actual:.3f})')
    ax00.set_xlabel('x₁  (voxel)')
    ax00.set_ylabel('x₂  (voxel)')

    # ── Panel (0,1): C₁₁₁₁ ───────────────────────────────────────────────────
    ax01 = fig.add_subplot(gs[0, 1])
    im01 = ax01.imshow(C_field[0, 0].T, origin='lower', extent=extent,
                       cmap='viridis', aspect='equal')
    plt.colorbar(im01, ax=ax01, fraction=0.046, pad=0.04, label='MPa')
    ax01.set_title(f'$C_{{1111}}$ stiffness field\n'
                   f'min={C_field[0,0].min():.0f}  max={C_field[0,0].max():.0f} MPa')
    ax01.set_xlabel('x₁  (voxel)')
    ax01.set_ylabel('x₂  (voxel)')

    # ── Panel (1,0): C₁₁₂₂ ───────────────────────────────────────────────────
    ax10 = fig.add_subplot(gs[1, 0])
    im10 = ax10.imshow(C_field[0, 1].T, origin='lower', extent=extent,
                       cmap='viridis', aspect='equal')
    plt.colorbar(im10, ax=ax10, fraction=0.046, pad=0.04, label='MPa')
    ax10.set_title(f'$C_{{1122}}$ (= λ) stiffness field\n'
                   f'min={C_field[0,1].min():.0f}  max={C_field[0,1].max():.0f} MPa')
    ax10.set_xlabel('x₁  (voxel)')
    ax10.set_ylabel('x₂  (voxel)')

    # ── Panel (1,1): C₁₂₁₂ ───────────────────────────────────────────────────
    ax11 = fig.add_subplot(gs[1, 1])
    im11 = ax11.imshow(C_field[2, 2].T, origin='lower', extent=extent,
                       cmap='viridis', aspect='equal')
    plt.colorbar(im11, ax=ax11, fraction=0.046, pad=0.04, label='MPa')
    ax11.set_title(f'$C_{{1212}}$ (= μ) shear stiffness\n'
                   f'min={C_field[2,2].min():.0f}  max={C_field[2,2].max():.0f} MPa')
    ax11.set_xlabel('x₁  (voxel)')
    ax11.set_ylabel('x₂  (voxel)')

    fig.suptitle(
        f'M&S centered-disk geometry  —  N={N}×{N},  κ = E_f/E_m = {kappa:.1f}\n'
        f'Fiber: E={E_f:.0f} MPa, ν={nu_f}   '
        f'Matrix: E={E_m:.1f} MPa, ν={nu_m}   '
        f'V_f = {vf_actual:.3f}',
        fontsize=11, fontweight='bold',
    )

    if args.save:
        plt.savefig(args.save, bbox_inches='tight', dpi=150)
        print(f"Figure saved to: {args.save}")
    else:
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
