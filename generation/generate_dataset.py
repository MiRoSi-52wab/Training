"""
Dataset generation orchestration script.

For each sample:
  1. Generate a random two-phase microstructure (disk/ellipse for 2D, sphere for 3D).
  2. Draw a random macroscopic strain ε̄ (Voigt, n_comp components).
  3. Run the Moulinec-Suquet FFT solver.
  4. Save to HDF5.

Usage (from project root, with venv activated):
  python -m generation.generate_dataset --config configs/data.yaml

Or directly:
  python generation/generate_dataset.py --config configs/data.yaml

Output: data/raw/dataset_<tag>.h5
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from generation.microstructure import generate_microstructure
from generation.fft_solver import solve
from utils.config_loader import load_config, compute_alpha_bounds


# ---------------------------------------------------------------------------
# Defaults (overridden by config yaml)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Dimensionality
    'dim': 2,                    # 2 (plane-strain) or 3 (full 3D)
    # Grid
    'N': 64,
    # Material contrast κ = E_inclusion / E_matrix
    'kappa': 10.0,
    'E_matrix': 1.0,
    'nu_matrix': 0.3,
    'nu_inclusion': 0.3,
    # Microstructure geometry
    'inclusion_type': 'disk',   # 2D: 'disk' or 'ellipse'; 3D: 'sphere'
    'n_inclusions': 10,
    'r_min': 2.0,
    'r_max': 6.0,
    # Strain loading
    # loading_mode controls how eps_bar is chosen per sample:
    #   'random'   — all n_comp components drawn uniformly from [-eps_bar_scale, +eps_bar_scale]
    #   'unit'     — cycles through unit directions: sample k gets scale * e_{k % n_comp}
    #                (balanced coverage of all n_comp directions, no extra config needed)
    #   'explicit' — every sample uses the fixed vector given by the 'eps_bar' key below
    'loading_mode': 'random',
    'eps_bar_scale': 0.01,       # magnitude for 'random' and 'unit' modes
    # 'eps_bar': [0.01, 0.0, 0.0],  # only used when loading_mode = 'explicit'
    # Dataset
    'n_samples': 2000,
    'val_fraction': 0.1,
    'test_fraction': 0.1,
    # Solver
    'tol': 1e-6,
    'max_iter': 1000,
    # Output
    'output_dir': 'data/raw',
    'tag': 'v1',
    # Reproducibility
    'seed': 42,
}


# ---------------------------------------------------------------------------
# Strain sampling
# ---------------------------------------------------------------------------

def sample_eps_bar(
    rng: np.random.Generator,
    scale: float,
    n_comp: int = 3,
    loading_mode: str = "random",
    sample_idx: int = 0,
    explicit_eps_bar: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Return a macroscopic strain vector ε̄ (Voigt, shape (n_comp,)).

    Args:
        rng:              NumPy random generator (used only in 'random' mode).
        scale:            Strain magnitude for 'random' and 'unit' modes.
        n_comp:           3 for 2D (plane-strain) or 6 for 3D.
        loading_mode:     One of:
            'random'    — all n_comp components drawn independently from [-scale, +scale].
            'unit'      — exactly one non-zero component, cycling: e_{sample_idx % n_comp} * scale.
                          Produces balanced coverage of all directions across the dataset.
            'explicit'  — returns explicit_eps_bar unchanged (must be provided).
        sample_idx:       Sample index used to select the active direction in 'unit' mode.
        explicit_eps_bar: Required when loading_mode='explicit'.

    Returns:
        (n_comp,) float64 array.
    """
    if loading_mode == "random":
        return rng.uniform(-scale, scale, size=n_comp)

    elif loading_mode == "unit":
        eps = np.zeros(n_comp, dtype=np.float64)
        eps[sample_idx % n_comp] = scale
        return eps

    elif loading_mode == "explicit":
        if explicit_eps_bar is None:
            raise ValueError(
                "loading_mode='explicit' requires an 'eps_bar' key in the config."
            )
        eps = np.asarray(explicit_eps_bar, dtype=np.float64).ravel()
        if len(eps) != n_comp:
            raise ValueError(
                f"Config 'eps_bar' has {len(eps)} components but n_comp={n_comp} "
                f"for dim={2 if n_comp == 3 else 3}."
            )
        return eps.copy()

    else:
        raise ValueError(
            f"Unknown loading_mode '{loading_mode}'. "
            "Choose 'random', 'unit', or 'explicit'."
        )


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------

def _create_dataset_file(path: Path, n_samples: int, N: int, dim: int = 2) -> h5py.File:
    """Create and initialise the HDF5 file with pre-allocated datasets."""
    n_comp  = 3 if dim == 2 else 6
    spatial = (N,) * dim

    f = h5py.File(path, 'w')

    meta = f.create_group('metadata')
    meta.attrs['N']         = N
    meta.attrs['dim']       = dim
    meta.attrs['n_samples'] = n_samples

    # Pre-allocate datasets (allows partial writes)
    f.create_dataset('C_field',    shape=(n_samples, n_comp, n_comp) + spatial, dtype='float32')
    f.create_dataset('phase',      shape=(n_samples,) + spatial,                dtype='bool')
    f.create_dataset('eps_bar',    shape=(n_samples, n_comp),                   dtype='float32')
    f.create_dataset('eps_star',   shape=(n_samples, n_comp) + spatial,         dtype='float32')
    f.create_dataset('tau_star',   shape=(n_samples, n_comp) + spatial,         dtype='float32')
    f.create_dataset('sigma_star', shape=(n_samples, n_comp) + spatial,         dtype='float32')
    f.create_dataset('n_iter',     shape=(n_samples,),                          dtype='int32')
    f.create_dataset('converged',  shape=(n_samples,),                          dtype='bool')

    return f


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate(config: dict, verbose: bool = True) -> Path:
    """
    Generate a full dataset according to `config`.

    Returns the path to the written HDF5 file.
    """
    dim          = int(config.get('dim', 2))
    n_comp       = 3 if dim == 2 else 6
    N            = config['N']
    n_samples    = config['n_samples']
    kappa        = config['kappa']
    E_matrix     = config['E_matrix']
    nu_matrix    = config['nu_matrix']
    nu_inclusion = config['nu_inclusion']
    E_inclusion  = E_matrix * kappa

    inclusion_type = config['inclusion_type']
    n_inclusions   = config['n_inclusions']
    r_min          = config['r_min']
    r_max          = config['r_max']

    eps_bar_scale    = config['eps_bar_scale']
    loading_mode     = config.get('loading_mode', 'random')
    explicit_eps_bar = config.get('eps_bar', None)

    if explicit_eps_bar is not None and loading_mode != 'explicit':
        import warnings
        warnings.warn(
            f"Config contains 'eps_bar' but loading_mode='{loading_mode}' — "
            f"the 'eps_bar' value will be ignored. "
            f"Set loading_mode: explicit if you want to use it.",
            UserWarning, stacklevel=2,
        )

    tol      = config['tol']
    max_iter = config['max_iter']

    # Theorem 2.1 reference stiffness: α₀ = (α⁻ + α⁺) / 2 from eigenvalue bounds.
    # Using this instead of the per-sample heuristic (C₁₁.max + C₁₁.min) / 2
    # ensures that the FFT solver and LSFNO use identical α₀, so their iteration
    # counts can be compared directly in the Study 4 evaluation table.
    alpha_minus, alpha_plus = compute_alpha_bounds(
        E_matrix, nu_matrix, nu_inclusion, kappa, dim=dim,
    )
    alpha0_t21 = (alpha_minus + alpha_plus) / 2.0

    seed = config['seed']
    rng  = np.random.default_rng(seed)

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = config['tag']
    out_path = output_dir / f'dataset_{tag}.h5'

    grid_str = '×'.join([str(N)] * dim)
    if verbose:
        print(f"Generating {n_samples} samples ({dim}D) → {out_path}")
        print(f"  Grid: {grid_str}  |  κ={kappa}  |  inclusion={inclusion_type}  |  loading={loading_mode}")

    # Build geometry kwargs once — they are the same for every sample.
    if dim == 2:
        if inclusion_type == 'disk':
            geom_kwargs: dict = {'n_disks': n_inclusions, 'r_min': r_min, 'r_max': r_max}
        else:
            geom_kwargs = {'n_ellipses': n_inclusions, 'a_min': r_min, 'a_max': r_max}
    else:
        geom_kwargs = {'n_spheres': n_inclusions, 'r_min': r_min, 'r_max': r_max}

    t0 = time.time()

    with _create_dataset_file(out_path, n_samples, N, dim=dim) as f:
        f['metadata'].attrs['loading_mode'] = loading_mode
        f['metadata'].attrs['alpha0']        = alpha0_t21
        f['metadata'].attrs['alpha_minus']   = alpha_minus
        f['metadata'].attrs['alpha_plus']    = alpha_plus
        n_failed = 0
        for idx in range(n_samples):
            phase, C_field = generate_microstructure(
                N=N,
                inclusion_type=inclusion_type,
                E_matrix=E_matrix,
                nu_matrix=nu_matrix,
                E_inclusion=E_inclusion,
                nu_inclusion=nu_inclusion,
                dim=dim,
                rng=rng,
                **geom_kwargs,
            )

            eps_bar = sample_eps_bar(
                rng, eps_bar_scale, n_comp=n_comp,
                loading_mode=loading_mode,
                sample_idx=idx,
                explicit_eps_bar=explicit_eps_bar,
            )

            result = solve(C_field, eps_bar, alpha0=alpha0_t21, tol=tol, max_iter=max_iter)

            if not result['converged']:
                n_failed += 1

            f['C_field'][idx]    = C_field.astype('float32')
            f['phase'][idx]      = phase
            f['eps_bar'][idx]    = eps_bar.astype('float32')
            f['eps_star'][idx]   = result['eps_star'].astype('float32')
            f['tau_star'][idx]   = result['tau_star'].astype('float32')
            f['sigma_star'][idx] = result['sigma_star'].astype('float32')
            f['n_iter'][idx]     = result['n_iter']
            f['converged'][idx]  = result['converged']

            if verbose and (idx + 1) % max(1, n_samples // 20) == 0:
                elapsed = time.time() - t0
                eta = elapsed / (idx + 1) * (n_samples - idx - 1)
                vf = phase.mean()
                print(f"  [{idx+1:>5}/{n_samples}]  "
                      f"VF={vf:.3f}  iter={result['n_iter']:>4}  "
                      f"conv={'Y' if result['converged'] else 'N'}  "
                      f"elapsed={elapsed:.1f}s  ETA={eta:.1f}s")

        # Write split index arrays as attributes
        all_idx = np.arange(n_samples)
        rng_split = np.random.default_rng(seed + 1)
        rng_split.shuffle(all_idx)
        n_test = int(n_samples * config['test_fraction'])
        n_val  = int(n_samples * config['val_fraction'])
        f['metadata'].attrs['test_idx']  = all_idx[:n_test].tolist()
        f['metadata'].attrs['val_idx']   = all_idx[n_test:n_test+n_val].tolist()
        f['metadata'].attrs['train_idx'] = all_idx[n_test+n_val:].tolist()

    elapsed = time.time() - t0
    if verbose:
        print(f"\nDone. {n_samples} samples in {elapsed:.1f}s "
              f"({elapsed/n_samples:.2f}s/sample).  "
              f"Failed convergence: {n_failed}/{n_samples}.")
        print(f"Saved to {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Generate LS-KAN-FNO micromechanics dataset.")
    p.add_argument('--config', type=str, default=None,
                   help="Path to data.yaml config file.")
    p.add_argument('--n_samples', type=int, default=None,
                   help="Override n_samples from config.")
    p.add_argument('--seed', type=int, default=None,
                   help="Override random seed.")
    p.add_argument('--tag', type=str, default=None,
                   help="Override output file tag.")
    p.add_argument('--quiet', action='store_true',
                   help="Suppress progress output.")
    return p.parse_args()


def main():
    args = _parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.config is not None:
        loaded = load_config(args.config)
        if loaded:
            config.update(loaded)

    # CLI overrides
    if args.n_samples is not None:
        config['n_samples'] = args.n_samples
    if args.seed is not None:
        config['seed'] = args.seed
    if args.tag is not None:
        config['tag'] = args.tag

    generate(config, verbose=not args.quiet)


if __name__ == '__main__':
    main()
