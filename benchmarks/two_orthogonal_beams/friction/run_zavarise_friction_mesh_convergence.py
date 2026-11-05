#!/usr/bin/env python3
"""
Friction-aware mesh-convergence orchestrator for the Zavarise & Wriggers
(2000) Example 2 benchmark.

Runs `contact_bending_soft_soft.py` for every (mu, nx) combination in the
requested grid, SEQUENTIALLY (avoiding CPU contention on shared machines),
collects the per-step compare files, and produces the two-panel
mesh-convergence figures via `plot_friction_mesh_convergence.py`.

Designed for cluster execution:
  - matplotlib backend is forced to "Agg" (no popups, no DISPLAY required).
  - Each child process is launched with `python -u` so stdout/stderr stream
    incrementally into the job log.
  - Any run whose compare file already exists is SKIPPED (idempotent
    restart on cluster pre-emption).

Usage
-----
    # Default grid: mu in {0.1, 0.6}, nx in {21, 51, 101}, 6 load steps:
    python3 run_zavarise_friction_mesh_convergence.py

    # Restrict to a subset:
    python3 run_zavarise_friction_mesh_convergence.py \
        --mu-list 0.1 --nx-list 21 51

    # Force re-run of existing results:
    python3 run_zavarise_friction_mesh_convergence.py --force

    # Only regenerate plots from existing compare files (no re-runs):
    python3 run_zavarise_friction_mesh_convergence.py --plot-only

Outputs
-------
contact_output_files_soft_soft/
    zavarise_compare_steps6_nx{NX}_mu{MU_TAG}.txt    one per (mu, nx)
    zavarise_coords_steps6_nx{NX}_mu{MU_TAG}.txt
    zavarise_forces_steps6_nx{NX}_mu{MU_TAG}.txt
    zavarise_curvR_steps6_nx{NX}_mu{MU_TAG}.txt
    zavarise_summary_steps6_nx{NX}_mu{MU_TAG}.txt
    zavarise_friction_mesh_convergence_mu{MU_TAG}.png   one per mu
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIM_SCRIPT   = PROJECT_ROOT / "contact_bending_soft_soft.py"
PLOT_SCRIPT  = Path(__file__).resolve().parent / "plot_friction_mesh_convergence.py"
OUT_DIR      = PROJECT_ROOT / "contact_output_files_soft_soft"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def derive_n_z(n_x: int, L_X: float = 10.0, L_Z: float = 14.0) -> int:
    """Choose n_z so Beam Z elements are the same length as Beam X elements."""
    elem_len = L_X / (n_x - 1)
    return int(round(L_Z / elem_len)) + 1


def mu_tag(mu: float) -> str:
    """Encode mu = 0.1 → '01', 0.6 → '06', 0.8 → '08', 1.5 → '15'."""
    return f"{int(round(mu * 10)):02d}"


def comparison_filename(out_dir: Path, n_steps: int, nx: int, mu: float) -> Path:
    return out_dir / f"zavarise_compare_steps{n_steps}_nx{nx}_mu{mu_tag(mu)}.txt"


def is_comparison_complete(path: Path, n_steps: int) -> bool:
    """A compare file is 'complete' if it contains a row for every step
    from 1 to n_steps."""
    if not path.exists():
        return False
    seen = set()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("step"):
                continue
            try:
                seen.add(int(line.split("\t", 1)[0]))
            except ValueError:
                continue
    return all(s in seen for s in range(1, n_steps + 1))


def run_single(mu: float, nx: int, n_steps: int) -> int:
    """Launch ONE simulation as a subprocess and stream its stdout/stderr
    to the orchestrator's job log.  Returns the child's exit code."""
    nz  = derive_n_z(nx)
    tag = f"_nx{nx}_mu{mu_tag(mu)}"
    cmd = [
        sys.executable, "-u",                          # unbuffered output
        str(SIM_SCRIPT),
        str(n_steps),
        "--nx", str(nx),
        "--nz", str(nz),
        "--mu", str(mu),
        "--tag", tag,
    ]
    print("=" * 78)
    print(f"[mesh] mu = {mu}   nx = {nx}   nz = {nz}   tag = '{tag}'")
    print(f"[mesh] cmd: {' '.join(cmd)}")
    print("=" * 78, flush=True)

    t0 = time.time()
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")           # belt + suspenders
    env.setdefault("MPLBACKEND", "Agg")                # no popups, no DISPLAY
    proc = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT), check=False)
    dt   = time.time() - t0
    print(f"[mesh] mu={mu} nx={nx} finished exit={proc.returncode} "
          f"in {dt/60:.1f} min", flush=True)
    return proc.returncode


def run_plots() -> None:
    """Regenerate every per-mu PNG by invoking the plot script."""
    cmd = [sys.executable, "-u", str(PLOT_SCRIPT)]
    print("=" * 78)
    print(f"[mesh] launching plot script: {' '.join(cmd)}")
    print("=" * 78, flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("MPLBACKEND", "Agg")
    proc = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT), check=False)
    print(f"[mesh] plot script exit={proc.returncode}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Friction-aware mesh-convergence orchestrator for the "
                    "Zavarise & Wriggers (2000) Example 2 benchmark.")
    parser.add_argument("--mu-list", type=float, nargs="+",
                        default=[0.1, 0.6],
                        help="Friction coefficients to sweep "
                             "(default: %(default)s).")
    parser.add_argument("--nx-list", type=int, nargs="+",
                        default=[21, 51, 101],
                        help="Beam-X node counts to sweep "
                             "(default: %(default)s).")
    parser.add_argument("--n-steps", type=int, default=6,
                        help="Number of load steps per run "
                             "(default: %(default)s).")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even when the compare file already "
                             "exists and looks complete.")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip all simulations; only regenerate the "
                             "PNG plots from existing compare files.")
    parser.add_argument("--no-plot", action="store_true",
                        help="Run the simulations but do NOT invoke the "
                             "plot script.  Useful on the cluster where "
                             "matplotlib has occasionally hit a "
                             "Path.__deepcopy__ infinite-recursion bug; "
                             "regenerate the PNGs locally afterwards with "
                             "--plot-only.")
    parser.add_argument("--order", choices=("fast-first", "as-given"),
                        default="fast-first",
                        help="Order in which to run the (mu, nx) grid.  "
                             "'fast-first' sorts by nx ascending so the "
                             "smaller meshes finish (and plot) early; "
                             "'as-given' uses the CLI order.  "
                             "Default: %(default)s.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build the (mu, nx) grid in execution order
    grid = [(mu, nx) for mu in args.mu_list for nx in args.nx_list]
    if args.order == "fast-first":
        grid.sort(key=lambda p: (p[1], p[0]))      # nx ascending, then mu

    # ---- Simulation phase ------------------------------------------------
    if not args.plot_only:
        print(f"[mesh] grid = {grid}")
        print(f"[mesh] n_steps = {args.n_steps}, force = {args.force}, "
              f"no_plot = {args.no_plot}", flush=True)
        for mu, nx in grid:
            compare = comparison_filename(OUT_DIR, args.n_steps, nx, mu)
            if (not args.force) and is_comparison_complete(compare, args.n_steps):
                print(f"[mesh] SKIP mu={mu} nx={nx}  ({compare.name} already "
                      f"complete; pass --force to re-run)", flush=True)
                continue
            rc = run_single(mu, nx, args.n_steps)
            if rc != 0:
                print(f"[mesh] WARNING: mu={mu} nx={nx} exited with code "
                      f"{rc}; continuing.", flush=True)
            # Update plots incrementally (skip on the cluster via --no-plot)
            if not args.no_plot:
                try:
                    run_plots()
                except Exception as e:
                    print(f"[mesh] WARNING: incremental plot failed: {e}",
                          flush=True)

    # ---- Final plot pass -------------------------------------------------
    if not args.no_plot:
        run_plots()
    else:
        print("[mesh] --no-plot: skipping final plot pass.  Regenerate "
              "PNGs locally with `python3 run_zavarise_friction_mesh_"
              "convergence.py --plot-only`.", flush=True)
    print("[mesh] done.", flush=True)


if __name__ == "__main__":
    main()
