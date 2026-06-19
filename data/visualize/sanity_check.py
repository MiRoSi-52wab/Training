"""
This file is use to have sanity check about the Moulinec Solver. 

The single most critical check for correctness of the solver is the volume average constraint, which must hold exactly:
⟨ε(x)⟩_spatial = ε̄

This is enforced by construction (the ξ=0 mode override in the solver), 
so if your stored eps_star field satisfies this, 
the solver is working correctly.


If this prints something like 1e-7 or smaller, the solver is ground-truth quality. 
If it is larger, there is a bug in the mean-strain enforcement.
"""
# ============================================================
# CONFIGURATION — only edit these two lines
# ============================================================
DATASET_PATH = "data/raw/dataset_v1.h5"
# ============================================================

import h5py
import numpy as np
from pathlib import Path


path = Path(DATASET_PATH)
if not path.exists():
    raise FileNotFoundError(f"Dataset not found: {path}")

with h5py.File(path, "r") as f:
    eps_star = f["eps_star"][:]   # (N_samples, 3, N, N)
    eps_bar  = f["eps_bar"][:]    # (N_samples, 3)

# Spatial mean of each sample
eps_mean = eps_star.mean(axis=(-2, -1))  # (N_samples, 3)

# Max absolute error across all samples and components
error = np.abs(eps_mean - eps_bar).max()
print(f"Max |⟨ε⟩ - ε̄| = {error:.2e}")  # should be < 1e-6