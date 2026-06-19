import sys; sys.path.insert(0, '.')
import ast, pathlib
src = pathlib.Path('replicate/paper2_kan_alpha_comparison.py').read_text()
ast.parse(src)
print('Syntax OK')
# check imports resolve
from models.kan_tau_theta import KANTauTheta
from models.ls_fno import LSFNO, YarotskyTauTheta
from generation.fft_solver import solve as fft_solve
from utils.config_loader import compute_alpha_bounds
print('Imports OK')
# check model construction path
from generation.microstructure import isotropic_stiffness_voigt_3d, build_C_field
import numpy as np, torch
print('All imports OK')