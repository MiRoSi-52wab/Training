"""
Study runner: execute one scenario and save results + figure.

Usage (from LS_KAN_FNO/ directory):
    from study.runner import run_scenario
    run_scenario('A1', output_dir='study_results', verbose=True)

Outputs per scenario  (written to  <output_dir>/<scenario_id>/):
    results.npz   — all numerical arrays
    summary.txt   — human-readable metrics
    plot.png      — 6-panel figure
"""

import sys
import time
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from study.geometry import make_ms_geometry, compute_alpha_opt
from study.scenarios import SCENARIOS
from study.kan_solver import solve_nonlinear_kan
from generation.nonlinear_fft_solver import solve_nonlinear


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _l2_rel(a: np.ndarray, b: np.ndarray, eps: float = 1e-30) -> float:
    """Relative L2 error ‖a − b‖ / ‖b‖."""
    denom = float(np.linalg.norm(b))
    if denom < eps:
        return float(np.linalg.norm(a - b))
    return float(np.linalg.norm(a - b)) / denom


def compute_metrics(fft_res: dict, kan_res: dict, eps_bar_path: np.ndarray) -> dict:
    """
    Compute per-step and final metrics comparing FFT and KAN results.

    Returns a dict with arrays of length n_steps (per-step) plus scalars.
    """
    n_steps = len(fft_res['eps_history'])
    eps_path = np.array([e[0].mean() for e in fft_res['eps_history']])  # Σ₁₁ path

    # ── Per-step scalars ──────────────────────────────────────────────────────
    macro_fft = np.array(fft_res['macro_stress_history'])   # (n_steps, 3)
    macro_kan = np.array(kan_res['macro_stress_history'])

    n_iter_fft = np.array(fft_res['n_iter_history'])
    n_iter_kan = np.array(kan_res['n_iter_history'])

    p_mean_fft = np.array([p.mean() for p in fft_res['p_history']])
    p_max_fft  = np.array([p.max()  for p in fft_res['p_history']])
    p_mean_kan = np.array([p.mean() for p in kan_res['p_history']])
    p_max_kan  = np.array([p.max()  for p in kan_res['p_history']])

    # ── Per-step L2 field errors ───────────────────────────────────────────────
    err_eps   = np.array([_l2_rel(kan_res['eps_history'][i],
                                   fft_res['eps_history'][i]) for i in range(n_steps)])
    err_sigma = np.array([_l2_rel(kan_res['sigma_history'][i][[0,1,3]],
                                   fft_res['sigma_history'][i][[0,1,3]]) for i in range(n_steps)])
    err_p     = np.array([_l2_rel(kan_res['p_history'][i],
                                   fft_res['p_history'][i]) for i in range(n_steps)])

    # ── Yield onset ───────────────────────────────────────────────────────────
    yield_step_fft = next((i for i, p in enumerate(fft_res['p_history']) if p.max() > 0), None)
    yield_step_kan = next((i for i, p in enumerate(kan_res['p_history']) if p.max() > 0), None)

    # ── Final-step metrics ────────────────────────────────────────────────────
    err_eps_final   = _l2_rel(kan_res['eps_star'],   fft_res['eps_star'])
    err_sigma_final = _l2_rel(kan_res['sigma_star'], fft_res['sigma_star'])
    err_p_final     = _l2_rel(kan_res['p_star'],     fft_res['p_star'])

    return dict(
        eps_path       = eps_path,
        macro_fft      = macro_fft,
        macro_kan      = macro_kan,
        n_iter_fft     = n_iter_fft,
        n_iter_kan     = n_iter_kan,
        p_mean_fft     = p_mean_fft,
        p_max_fft      = p_max_fft,
        p_mean_kan     = p_mean_kan,
        p_max_kan      = p_max_kan,
        err_eps        = err_eps,
        err_sigma      = err_sigma,
        err_p          = err_p,
        err_eps_final   = err_eps_final,
        err_sigma_final = err_sigma_final,
        err_p_final     = err_p_final,
        yield_step_fft  = yield_step_fft,
        yield_step_kan  = yield_step_kan,
        converged_fft   = np.array(fft_res['converged_history']),
        converged_kan   = np.array(kan_res['converged_history']),
        total_iters_fft = int(n_iter_fft.sum()),
        total_iters_kan = int(n_iter_kan.sum()),
    )


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_results(out_dir: Path, scenario_id: str, params: dict,
                 fft_res: dict, kan_res: dict, metrics: dict,
                 time_fft: float, time_kan: float) -> None:
    """Save results.npz and summary.txt to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── NumPy archive ─────────────────────────────────────────────────────────
    save_dict: dict = {}

    # Per-step histories (stack into arrays)
    for key in ('eps_history', 'p_history'):
        save_dict[f'fft_{key}'] = np.array(fft_res[key])
        save_dict[f'kan_{key}'] = np.array(kan_res[key])

    for key in ('sigma_history',):
        save_dict[f'fft_{key}'] = np.array(fft_res[key])
        save_dict[f'kan_{key}'] = np.array(kan_res[key])

    save_dict['fft_macro_stress'] = np.array(fft_res['macro_stress_history'])
    save_dict['kan_macro_stress'] = np.array(kan_res['macro_stress_history'])
    save_dict['fft_n_iter']       = np.array(fft_res['n_iter_history'])
    save_dict['kan_n_iter']       = np.array(kan_res['n_iter_history'])

    # Final fields
    for key in ('eps_star', 'sigma_star', 'tau_star', 'p_star'):
        save_dict[f'fft_{key}'] = fft_res[key]
        save_dict[f'kan_{key}'] = kan_res[key]

    # Metrics
    for k, v in metrics.items():
        if isinstance(v, (np.ndarray, int, float)) or v is None:
            save_dict[f'metric_{k}'] = v if v is not None else np.array(-1)

    save_dict['time_fft_s'] = np.array(time_fft)
    save_dict['time_kan_s'] = np.array(time_kan)
    save_dict['alpha0_fft'] = np.array(fft_res['alpha0'])
    save_dict['alpha0_kan'] = np.array(kan_res['alpha0'])

    np.savez_compressed(out_dir / 'results.npz', **save_dict)

    # ── Summary text ──────────────────────────────────────────────────────────
    m = metrics
    lines = [
        f"Scenario {scenario_id}: {params['description']}",
        f"  Grid N={params['N']},  V_f={params['vf']:.3f}",
        f"  E_m={params['E_m']:.0f} MPa,  E_f={params['E_f']:.0f} MPa,",
        f"  σ_y={params['sigma_y']:.1f} MPa,  H={params['H']:.0f} MPa",
        f"  α₀_factor={params['alpha0_factor']:.2f}",
        f"  α₀ (FFT) = {fft_res['alpha0']:.4g} MPa",
        "",
        "Timing:",
        f"  FFT solver : {time_fft:.2f} s",
        f"  KAN solver : {time_kan:.2f} s",
        "",
        "Iteration counts:",
        f"  Total FFT  : {m['total_iters_fft']}",
        f"  Total KAN  : {m['total_iters_kan']}",
        f"  Per-step FFT: {m['n_iter_fft'].tolist()}",
        f"  Per-step KAN: {m['n_iter_kan'].tolist()}",
        f"  Converged FFT: {m['converged_fft'].all()}",
        f"  Converged KAN: {m['converged_kan'].all()}",
        "",
        "Yield onset:",
        f"  FFT yield at step {m['yield_step_fft']}  "
        f"(KAN: step {m['yield_step_kan']})",
        "",
        "Final-step field errors (KAN vs FFT):",
        f"  err_eps   = {m['err_eps_final']:.3e}",
        f"  err_sigma = {m['err_sigma_final']:.3e}",
        f"  err_p     = {m['err_p_final']:.3e}",
        "",
        "Per-step L2 errors (max over steps):",
        f"  max err_eps   = {m['err_eps'].max():.3e}  (step {m['err_eps'].argmax()})",
        f"  max err_sigma = {m['err_sigma'].max():.3e}  (step {m['err_sigma'].argmax()})",
        f"  max err_p     = {m['err_p'].max():.3e}  (step {m['err_p'].argmax()})",
        "",
        "Macroscopic stress (final step):",
        f"  Σ_xx  FFT={m['macro_fft'][-1, 0]:.4g}  KAN={m['macro_kan'][-1, 0]:.4g} MPa",
        f"  Σ_yy  FFT={m['macro_fft'][-1, 1]:.4g}  KAN={m['macro_kan'][-1, 1]:.4g} MPa",
        f"  Σ_xy  FFT={m['macro_fft'][-1, 2]:.4g}  KAN={m['macro_kan'][-1, 2]:.4g} MPa",
    ]
    (out_dir / 'summary.txt').write_text('\n'.join(lines))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_plot(out_dir: Path, scenario_id: str, params: dict,
              fft_res: dict, kan_res: dict, metrics: dict) -> None:
    """
    Save a 3×2 figure showing:
      Row 0: macro stress-strain curve | iteration counts per step
      Row 1: FFT p_field (final) | KAN p_field (final)
      Row 2: |FFT−KAN| p_field error | L2 relative errors vs step
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.colors import LogNorm
    except ImportError:
        print("  [plot] matplotlib not available — skipping plot")
        return

    m = metrics

    fig = plt.figure(figsize=(14, 13))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.46, wspace=0.32)

    C_FFT = '#1f77b4'
    C_KAN = '#d62728'
    C_ERR = '#ff7f0e'

    N = fft_res['p_star'].shape[0]
    eps_bar = np.array(params['eps_bar_path'])

    # ── Panel (0,0): Macro stress-strain ─────────────────────────────────────
    ax00 = fig.add_subplot(gs[0, 0])
    E_xx = eps_bar[:, 0] * 100   # % strain
    ax00.plot(E_xx, m['macro_fft'][:, 0], color=C_FFT, lw=2.0, label='FFT  Σ_xx')
    ax00.plot(E_xx, m['macro_kan'][:, 0], color=C_KAN, lw=1.4, ls='--', label='KAN  Σ_xx')

    if abs(m['macro_fft'][:, 2]).max() > 1.0:   # shear case
        ax00.plot(E_xx, m['macro_fft'][:, 2], color=C_FFT, lw=1.4, ls=':',  label='FFT  Σ_xy')
        ax00.plot(E_xx, m['macro_kan'][:, 2], color=C_KAN, lw=1.0, ls='-.',  label='KAN  Σ_xy')

    if m['yield_step_fft'] is not None:
        ax00.axvline(E_xx[m['yield_step_fft']], color='grey', ls=':', lw=1.0,
                     label=f'yield onset (FFT step {m["yield_step_fft"]+1})')

    ax00.set_xlabel('Macroscopic strain $E_{xx}$ (%)')
    ax00.set_ylabel('Macroscopic stress (MPa)')
    ax00.set_title(f'{scenario_id} — Stress-strain curve')
    ax00.legend(fontsize=8)
    ax00.grid(alpha=0.3)

    # ── Panel (0,1): Iteration counts ─────────────────────────────────────────
    ax01 = fig.add_subplot(gs[0, 1])
    steps = np.arange(1, len(m['n_iter_fft']) + 1)
    ax01.step(steps, m['n_iter_fft'], color=C_FFT, lw=2.0, where='mid', label='FFT')
    ax01.step(steps, m['n_iter_kan'], color=C_KAN, lw=1.4, ls='--', where='mid', label='KAN')

    # Mark steps where convergence failed
    for solver, n_iter, converged, c in [
        ('FFT', m['n_iter_fft'], m['converged_fft'], C_FFT),
        ('KAN', m['n_iter_kan'], m['converged_kan'], C_KAN),
    ]:
        failed = np.where(~converged)[0]
        if len(failed):
            ax01.scatter(steps[failed], n_iter[failed], color=c,
                         marker='x', s=60, zorder=5, label=f'{solver} DID NOT CONVERGE')

    ax01.set_xlabel('Load step')
    ax01.set_ylabel('LS iterations')
    ax01.set_title('Iteration count per step')
    ax01.legend(fontsize=8)
    ax01.grid(alpha=0.3)

    total_diff = int(m['total_iters_kan']) - int(m['total_iters_fft'])
    ax01.text(0.98, 0.98,
              f'Total FFT: {m["total_iters_fft"]}\nTotal KAN: {m["total_iters_kan"]}'
              f'\nΔ = {total_diff:+d}',
              ha='right', va='top', transform=ax01.transAxes, fontsize=8,
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # ── Panels (1,0) and (1,1): Plastic strain fields ─────────────────────────
    p_fft = fft_res['p_star']
    p_kan = kan_res['p_star']
    p_all = np.concatenate([p_fft.ravel(), p_kan.ravel()])
    vmax  = float(p_all.max()) if p_all.max() > 1e-10 else 1.0

    for ax_idx, (p_field, label) in enumerate([(p_fft, 'FFT'), (p_kan, 'KAN')]):
        ax = fig.add_subplot(gs[1, ax_idx])
        im = ax.imshow(p_field.T, origin='lower', vmin=0, vmax=vmax,
                       cmap='plasma', aspect='equal')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='p')
        ax.set_title(f'{label} — plastic strain $p$ (final step)')
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')

    # ── Panel (2,0): |FFT − KAN| p-field error ────────────────────────────────
    ax20 = fig.add_subplot(gs[2, 0])
    p_diff = np.abs(p_fft - p_kan)
    if p_diff.max() > 0:
        im2 = ax20.imshow(p_diff.T, origin='lower', cmap='hot',
                          norm=LogNorm(vmin=max(p_diff.max()*1e-6, 1e-15),
                                       vmax=max(p_diff.max(), 1e-14)),
                          aspect='equal')
        plt.colorbar(im2, ax=ax20, fraction=0.046, pad=0.04, label='|ΔPAN − ΔFFT|')
    else:
        ax20.imshow(p_diff.T, origin='lower', cmap='hot', aspect='equal')
        ax20.text(0.5, 0.5, 'Identical (diff = 0)',
                  ha='center', va='center', transform=ax20.transAxes)
    ax20.set_title(f'|FFT − KAN| plastic strain error\n'
                   f'max = {p_diff.max():.2e},  rel L2 = {m["err_p_final"]:.2e}')
    ax20.set_xlabel('x₁')
    ax20.set_ylabel('x₂')

    # ── Panel (2,1): L2 errors vs step ────────────────────────────────────────
    ax21 = fig.add_subplot(gs[2, 1])
    steps = np.arange(1, len(m['err_eps']) + 1)

    ax21.semilogy(steps, m['err_eps']   + 1e-30, color='C0',  lw=1.6, label='ε (strain)')
    ax21.semilogy(steps, m['err_sigma'] + 1e-30, color='C1',  lw=1.6, label='σ (stress)')
    ax21.semilogy(steps, m['err_p']     + 1e-30, color='C2',  lw=1.6, label='p (plastic)')

    ax21.set_xlabel('Load step')
    ax21.set_ylabel('Relative L2 error  ‖KAN − FFT‖ / ‖FFT‖')
    ax21.set_title('Field errors (KAN vs FFT)')
    ax21.legend(fontsize=8)
    ax21.grid(alpha=0.3, which='both')

    # ── Suptitle ───────────────────────────────────────────────────────────────
    desc_short = textwrap.shorten(params['description'], width=70)
    fig.suptitle(
        f'Scenario {scenario_id} — {desc_short}\n'
        f'N={N}×{N},  σ_y={params["sigma_y"]:.1f} MPa,  H={params["H"]:.0f} MPa,  '
        f'α₀_factor={params["alpha0_factor"]:.2f}',
        fontsize=11, fontweight='bold',
    )

    plot_path = out_dir / 'plot.png'
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {plot_path}")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_scenario(
    scenario_id:  str,
    output_dir:   str  = 'study_results',
    verbose:      bool = False,
    tol:          float = 1e-4,
    override_params: dict = None,
) -> dict:
    """
    Run one scenario: FFT solver + KAN solver, compute metrics, save outputs.

    Parameters
    ----------
    scenario_id    : e.g. 'A1', 'B3', 'E7'
    output_dir     : root directory for outputs (a sub-directory per scenario)
    verbose        : print step-by-step progress
    tol            : LS convergence tolerance (overrides scenario default)
    override_params: dict of parameter overrides (e.g. {'N': 128})

    Returns
    -------
    dict with keys: 'fft', 'kan', 'metrics', 'params'
    """
    if scenario_id not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{scenario_id}'. "
            f"Available: {sorted(SCENARIOS.keys())}"
        )

    params = SCENARIOS[scenario_id].copy()
    if override_params:
        params.update(override_params)

    out_dir = Path(output_dir) / scenario_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Scenario {scenario_id}: {params['description']}")
    print(f"  N={params['N']}, E_m={params['E_m']:.0f}, "
          f"σ_y={params['sigma_y']:.1f}, H={params['H']:.0f}, "
          f"α₀_factor={params['alpha0_factor']:.2f}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    # ── Geometry ───────────────────────────────────────────────────────────────
    print("Building geometry ...")
    phase, C_field = make_ms_geometry(
        N    = params['N'],
        E_m  = params['E_m'],
        nu_m = params['nu_m'],
        E_f  = params['E_f'],
        nu_f = params['nu_f'],
        vf   = params['vf'],
    )

    eps_bar_path = np.asarray(params['eps_bar_path'], dtype=np.float64)
    sigma_0      = float(params['sigma_y'])
    H            = float(params['H'])

    # Compute optimal alpha0 and apply factor
    alpha_opt = compute_alpha_opt(C_field)
    alpha0    = alpha_opt * float(params['alpha0_factor'])
    print(f"  α_opt = {alpha_opt:.4g} MPa  →  α₀ = {alpha0:.4g} MPa")

    max_iter = int(params['max_iter'])

    # ── FFT solver ─────────────────────────────────────────────────────────────
    print("Running FFT solver ...")
    t0 = time.perf_counter()
    fft_res = solve_nonlinear(
        C_field      = C_field,
        phase        = phase,
        eps_bar_path = eps_bar_path,
        sigma_0      = sigma_0,
        H            = H,
        alpha0       = alpha0,
        tol          = tol,
        max_iter     = max_iter,
        discretization = 'exact',
        verbose      = verbose,
    )
    time_fft = time.perf_counter() - t0
    n_iter_fft = sum(fft_res['n_iter_history'])
    conv_fft   = all(fft_res['converged_history'])
    print(f"  FFT done in {time_fft:.2f}s — {n_iter_fft} total iters, "
          f"converged={conv_fft}")

    # ── KAN solver ─────────────────────────────────────────────────────────────
    print("Running KAN solver ...")
    t0 = time.perf_counter()
    kan_res = solve_nonlinear_kan(
        C_field      = C_field,
        phase        = phase,
        eps_bar_path = eps_bar_path,
        sigma_0      = sigma_0,
        H            = H,
        alpha0       = alpha0,
        tol          = tol,
        max_iter     = max_iter,
        discretization = 'exact',
        verbose      = verbose,
    )
    time_kan = time.perf_counter() - t0
    n_iter_kan = sum(kan_res['n_iter_history'])
    conv_kan   = all(kan_res['converged_history'])
    print(f"  KAN done in {time_kan:.2f}s — {n_iter_kan} total iters, "
          f"converged={conv_kan}")

    # ── Metrics ────────────────────────────────────────────────────────────────
    print("Computing metrics ...")
    metrics = compute_metrics(fft_res, kan_res, eps_bar_path)

    # ── Print summary ──────────────────────────────────────────────────────────
    print(f"\n  Final-step L2 errors (KAN vs FFT):")
    print(f"    ε  :  {metrics['err_eps_final']:.3e}")
    print(f"    σ  :  {metrics['err_sigma_final']:.3e}")
    print(f"    p  :  {metrics['err_p_final']:.3e}")
    print(f"  Macro Σ_xx: FFT={metrics['macro_fft'][-1, 0]:.4g}  "
          f"KAN={metrics['macro_kan'][-1, 0]:.4g} MPa")
    print(f"  Iter diff: FFT={metrics['total_iters_fft']}  "
          f"KAN={metrics['total_iters_kan']}  "
          f"Δ={metrics['total_iters_kan']-metrics['total_iters_fft']:+d}")

    # ── Save ───────────────────────────────────────────────────────────────────
    print("Saving results ...")
    save_results(out_dir, scenario_id, params, fft_res, kan_res, metrics,
                 time_fft, time_kan)
    save_plot(out_dir, scenario_id, params, fft_res, kan_res, metrics)

    print(f"  results.npz → {out_dir / 'results.npz'}")
    print(f"  summary.txt → {out_dir / 'summary.txt'}")

    return {'fft': fft_res, 'kan': kan_res, 'metrics': metrics, 'params': params}
