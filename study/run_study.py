"""
CLI entry point for the comparative study.

Usage (from LS_KAN_FNO/ directory):
    python study/run_study.py A1
    python study/run_study.py A1 B1 E3 --output results/
    python study/run_study.py --group A --output results/
    python study/run_study.py --all --output results/ --tol 1e-5
    python study/run_study.py --list

Options
-------
  Positional         : scenario IDs to run (e.g. A1 B2 E7)
  --group  G         : run all scenarios in group G (A|B|C|D|E|F)
  --all              : run all 27 scenarios
  --output DIR       : output directory (default: study_results)
  --tol    TOL       : LS convergence tolerance (default: 1e-4)
  --N      INT       : override grid resolution for all scenarios
  --verbose          : print iteration-level progress
  --list             : print all available scenario IDs and exit
"""

"""
# See all 27 scenario IDs
python study/run_study.py --list

# Run a single scenario
python study/run_study.py A1

# Run several scenarios
python study/run_study.py A1 A4 B1 E3

# Run an entire group
python study/run_study.py --group A

# Run all 27 scenarios
python study/run_study.py --all

# Custom output directory + tolerance
python study/run_study.py A1 B1 --output results/ --tol 1e-5

# Override grid resolution (e.g. N=128 for A2 perfect plasticity)
python study/run_study.py A2 --N 128

# Verbose iteration output
python study/run_study.py A1 --verbose
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study.scenarios import SCENARIOS, list_scenarios
from study.runner import run_scenario


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Run comparative study: nonlinear FFT vs KAN solver',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('scenarios', nargs='*',
                   help='Scenario IDs to run (e.g. A1 B2 E7)')
    p.add_argument('--group', '-g', type=str, default=None,
                   help='Run all scenarios in one group (A|B|C|D|E|F)')
    p.add_argument('--all', '-a', action='store_true',
                   help='Run all 27 scenarios')
    p.add_argument('--output', '-o', type=str, default='study_results',
                   help='Output directory (default: study_results)')
    p.add_argument('--tol', type=float, default=1e-4,
                   help='LS convergence tolerance (default: 1e-4)')
    p.add_argument('--N', type=int, default=None,
                   help='Override grid resolution N for all scenarios')
    p.add_argument('--verbose', '-v', action='store_true',
                   help='Print iteration-level progress')
    p.add_argument('--list', '-l', action='store_true',
                   help='List all available scenarios and exit')
    return p.parse_args(argv)


def resolve_scenario_ids(args) -> list:
    """Return the ordered list of scenario IDs to run."""
    ids = []

    if args.all:
        ids = sorted(SCENARIOS.keys())
    elif args.group:
        g = args.group.upper()
        ids = sorted(sid for sid in SCENARIOS if sid.startswith(g))
        if not ids:
            raise ValueError(f"No scenarios found for group '{g}'")
    elif args.scenarios:
        ids = list(args.scenarios)
    else:
        raise ValueError(
            "Specify at least one scenario ID, --group G, or --all.\n"
            "Use --list to see available scenarios."
        )

    # Validate
    unknown = [sid for sid in ids if sid not in SCENARIOS]
    if unknown:
        raise ValueError(
            f"Unknown scenario(s): {unknown}\n"
            f"Use --list to see available scenarios."
        )

    return ids


def main(argv=None):
    args = parse_args(argv)

    if args.list:
        list_scenarios()
        print(f"\nTotal: {len(SCENARIOS)} scenarios")
        return 0

    try:
        scenario_ids = resolve_scenario_ids(args)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    override = {}
    if args.N is not None:
        override['N'] = args.N

    print(f"Running {len(scenario_ids)} scenario(s): {scenario_ids}")
    print(f"Output directory: {args.output}")
    print(f"Tolerance: {args.tol}")
    if override:
        print(f"Overrides: {override}")

    failed = []
    results = {}

    for sid in scenario_ids:
        try:
            r = run_scenario(
                scenario_id    = sid,
                output_dir     = args.output,
                verbose        = args.verbose,
                tol            = args.tol,
                override_params = override if override else None,
            )
            results[sid] = r
        except Exception as exc:
            import traceback
            print(f"\n[ERROR] Scenario {sid} failed: {exc}", file=sys.stderr)
            traceback.print_exc()
            failed.append(sid)

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Study complete: {len(results)}/{len(scenario_ids)} scenarios succeeded")
    if failed:
        print(f"  Failed: {failed}")
    print(f"Outputs written to: {args.output}/")

    if results:
        print("\nFinal-step L2 errors summary (err_sigma):")
        print(f"  {'Scenario':<10} {'err_ε':>10} {'err_σ':>10} {'err_p':>10}  "
              f"{'Δiters':>8}  {'converged':>10}")
        for sid, r in results.items():
            m = r['metrics']
            conv_fft = 'Y' if m['converged_fft'].all() else 'N'
            conv_kan = 'Y' if m['converged_kan'].all() else 'N'
            delta_i  = m['total_iters_kan'] - m['total_iters_fft']
            print(f"  {sid:<10} {m['err_eps_final']:>10.3e} "
                  f"{m['err_sigma_final']:>10.3e} "
                  f"{m['err_p_final']:>10.3e}  "
                  f"{delta_i:>+8d}  "
                  f"FFT:{conv_fft} KAN:{conv_kan}")

    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
