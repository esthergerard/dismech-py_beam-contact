#!/usr/bin/env python3
"""
Mesh-convergence study for the Zavarise & Wriggers (2000) Example 2 benchmark.

Runs `contact_bending_soft_soft.py` for several Beam-X discretizations,
collects the per-step contact-force tables progressively, and finally
overlays the F_N(step) curves on top of the paper's Table IV reference.

The runs are launched **sequentially** as subprocesses so that:
- each finishes (and its txt files are flushed) before the next one starts,
- if the wall-clock limit on the cluster expires mid-study, the txt files
  from completed runs are still on disk, and the final plot can be
  regenerated from them with `python3 run_zavarise_mesh_convergence.py --plot-only`.

Default mesh sequence (Beam X nodes):   11  →  21  →  51  →  101
The Beam Z mesh is derived to keep the element length equal on both beams
(Beam Z length is 14, Beam X length is 10 → n_z = round(14/(10/(n_x-1))) + 1).

Usage
-----
    # Full study (run all four meshes and produce the plot):
    python3 run_zavarise_mesh_convergence.py

    # Limit to a subset (for quick local tests):
    python3 run_zavarise_mesh_convergence.py --nx-list 11 21

    # Only regenerate the plot from existing txt files (no re-run):
    python3 run_zavarise_mesh_convergence.py --plot-only

    # Change the number of load steps (default 6 = paper's Case 1):
    python3 run_zavarise_mesh_convergence.py --n-steps 6

Outputs
-------
contact_output_files_soft_soft/
    zavarise_compare_steps{N}_nx{nx}.txt    one per mesh
    zavarise_mesh_convergence_steps{N}.png  final overlay figure
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Paper reference (Zavarise & Wriggers 2000, Table IV, μ=0)
# ---------------------------------------------------------------------------
ZAVARISE_TABLE_IV_FN = {0: 0.0, 1: 0.0, 2: 1e-9,
                        3: 17.2, 4: 34.3, 5: 51.3, 6: 68.4}


def zavarise_FN_reference(step_id):
    if step_id in ZAVARISE_TABLE_IV_FN:
        return ZAVARISE_TABLE_IV_FN[step_id]
    if step_id >= 3:
        return 17.1 * (step_id - 2)
    return 0.0


def analytical_EB_reference(step_id, L1=10.0):
    """Analytical rigid-contact F_N (Euler-Bernoulli, paper's Figure 9 layout)."""
    EI = 1.0e8 * 2.0e-4
    L_x = 10.0
    Kx_inv = L_x**3 / (192.0 * EI)
    Kz_inv = L1**3 / (3.0 * EI)
    Y_A = -0.5 + 0.3 * step_id
    rhs = Y_A - 0.1
    if rhs <= 0:
        return 0.0
    return rhs / (Kx_inv + Kz_inv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def derive_n_z(n_x, L_X=10.0, L_Z=14.0):
    """Choose n_z so Beam Z elements are the same length as Beam X elements."""
    elem_len = L_X / (n_x - 1)
    return int(round(L_Z / elem_len)) + 1


def comparison_filename(out_dir, n_steps, n_x):
    return out_dir / f"zavarise_compare_steps{n_steps}_nx{n_x}.txt"


def load_compare_file(path):
    """Parse a compare file written by contact_bending_soft_soft.py.

    Returns a dict {step_id: FN_local}.
    Handles both the new format (FN_local in column 2) and the old format
    (F_N_dismech in column 2 — which was the global ΣFy magnitude).
    """
    steps = {}
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            try:
                step = int(parts[0])
                FN_local = float(parts[1])
            except ValueError:
                continue
            steps[step] = FN_local
    return steps


def run_single_mesh(n_x, n_steps, python_exe=None, extra_args=None):
    """Run the simulation once for a given Beam-X mesh.

    Streams stdout to the console as it arrives, so cluster logs stay live.
    Returns the subprocess exit code.
    """
    if python_exe is None:
        python_exe = sys.executable
    n_z = derive_n_z(n_x)
    tag = f"_nx{n_x}"
    cmd = [python_exe, "-u", str(Path(__file__).resolve().parent / "contact_bending_soft_soft.py"),
           str(n_steps), "--nx", str(n_x), "--nz", str(n_z), "--tag", tag]
    if extra_args:
        cmd.extend(extra_args)

    print("=" * 78)
    print(f"[mesh convergence] launching run with Beam-X nodes = {n_x} "
          f"(Beam-Z nodes = {n_z})")
    print(f"  cmd: {' '.join(cmd)}")
    print("=" * 78, flush=True)

    t0 = time.time()
    # Forward stdout/stderr in real time
    proc = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    print(f"[mesh convergence] mesh nx={n_x} finished with exit code "
          f"{proc.returncode} in {elapsed/60:.1f} min", flush=True)
    return proc.returncode


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _write_overlay_csv(out_dir, n_steps, nx_list, csv_path):
    """Always write a plain-text "overlay table" alongside (or instead of)
    the PNG, so the user can regenerate the plot locally even if matplotlib
    blew up on the cluster."""
    rows = []
    header = ["step", "FN_paper[N]", "FN_EulerBernoulli[N]"]
    for nx in nx_list:
        header.append(f"FN_dismech_nx{nx}[N]")
    rows.append(header)

    # Collect per-mesh data
    mesh_data = {}
    for nx in nx_list:
        path = comparison_filename(out_dir, n_steps, nx)
        mesh_data[nx] = load_compare_file(path) if path.exists() else {}

    for s in range(0, n_steps + 1):
        row = [s, f"{zavarise_FN_reference(s):.6e}",
               f"{analytical_EB_reference(s):.6e}"]
        for nx in nx_list:
            val = mesh_data[nx].get(s, float("nan"))
            row.append(f"{val:.6e}")
        rows.append(row)

    with open(csv_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    print(f"  [plot] wrote overlay table → {csv_path}", flush=True)


def plot_overlay(out_dir, n_steps, nx_list, save_path):
    """Generate the overlay PNG comparing all available meshes to the paper.

    Defensive: wraps the matplotlib calls in try/except and also writes a
    plain-text overlay table next to the PNG so the data is never lost
    even if matplotlib crashes (we ran into matplotlib's known
    `Path.__deepcopy__` recursion bug on the cluster).
    """
    # Always write the CSV first — guaranteed to succeed.
    csv_path = save_path.with_suffix(".tsv")
    _write_overlay_csv(out_dir, n_steps, nx_list, csv_path)

    # Now try to render the PNG.  If matplotlib fails (recursion bug or
    # otherwise) we just log and return — the CSV is enough to regenerate
    # the plot offline.
    import sys as _sys
    try:
        _sys.setrecursionlimit(50000)  # safety net for matplotlib's deepcopy
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(11, 7))
        ax = fig.add_subplot(1, 1, 1)

        cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(nx_list)))
        plotted_any = False
        for k, n_x in enumerate(nx_list):
            path = comparison_filename(out_dir, n_steps, n_x)
            if not path.exists():
                continue
            data = load_compare_file(path)
            if not data:
                continue
            steps = sorted(data.keys())
            FN = [data[s] for s in steps]
            ax.plot(steps, FN, marker="o", linestyle="-", color=cmap[k],
                    linewidth=2, markersize=6, label=f"dismech, n_x = {n_x}")
            plotted_any = True

        # Paper reference (μ=0).  Use only simple markers to avoid the
        # matplotlib Path deepcopy bug — 'x' and 's' have triggered it.
        ref_steps = sorted(ZAVARISE_TABLE_IV_FN.keys())
        ax.plot(ref_steps, [zavarise_FN_reference(s) for s in ref_steps],
                marker="o", linestyle="--", color="black",
                linewidth=2.5, markersize=8, markerfacecolor="none",
                label="Zavarise & Wriggers (2000), Table IV, mu=0")

        # Analytical Euler-Bernoulli reference
        eb_steps = list(range(0, n_steps + 1))
        eb_vals = [analytical_EB_reference(s) for s in eb_steps]
        ax.plot(eb_steps, eb_vals, marker="o", linestyle=":", color="green",
                linewidth=2.5, markersize=7, markerfacecolor="none",
                label="Analytical Euler-Bernoulli (rigid contact, L1=10)")

        ax.set_xlabel("Load step", fontsize=12)
        ax.set_ylabel("Normal contact force |F_N| (local frame)  [N]",
                      fontsize=12)
        ax.set_title(
            f"Zavarise Example 2 - mesh-convergence study  "
            f"({n_steps} load steps)\n"
            "Off-center crossing: contact 10 m from clamped end A, "
            "4 m from free end B", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
        ax.set_xticks(ref_steps)

        # NOTE: explicitly avoid bbox_inches='tight' — that codepath
        # deepcopies the figure inside matplotlib and is what triggered the
        # Path.__deepcopy__ infinite recursion in the original cluster run.
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  [plot] saved overlay PNG → {save_path}", flush=True)
        if not plotted_any:
            print("  [plot] note: no dismech mesh data yet — figure only "
                  "shows reference curves.", flush=True)
    except RecursionError as e:
        print(f"  [plot] WARNING: matplotlib RecursionError "
              f"(Path.__deepcopy__ bug): {e}\n"
              f"         The overlay TSV at {csv_path} contains the same data; "
              f"plot it locally with --plot-only.", flush=True)
    except Exception as e:
        print(f"  [plot] WARNING: plotting failed with {type(e).__name__}: {e}\n"
              f"         The overlay TSV at {csv_path} contains the same data.",
              flush=True)
        import traceback
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Mesh-convergence orchestrator for the Zavarise benchmark.")
    parser.add_argument("--nx-list", type=int, nargs="+",
                        default=[11, 21, 51, 101],
                        help="Sequence of Beam-X node counts to run "
                             "(default: %(default)s).")
    parser.add_argument("--n-steps", type=int, default=6,
                        help="Number of load steps per run (default: %(default)s).")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip the simulations and only regenerate the "
                             "overlay PNG from existing txt files.")
    parser.add_argument("--no-plot", action="store_true",
                        help="Run the simulations but do not attempt to "
                             "generate the matplotlib overlay PNG.  The plain-"
                             "text overlay table is still written and can be "
                             "plotted later locally with --plot-only.")
    parser.add_argument("--extra-arg", action="append", default=[],
                        metavar="ARG",
                        help="Forward an extra CLI argument to each simulation "
                             "(repeatable).  Example: --extra-arg --fixed-kc "
                             "--extra-arg 1.0e4")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent / "contact_output_files_soft_soft"
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / f"zavarise_mesh_convergence_steps{args.n_steps}.png"

    def _safe_plot():
        """Wrapper around plot_overlay that NEVER lets a plot failure kill
        the orchestrator (the simulations are the expensive part)."""
        if args.no_plot:
            # Still write the plain-text overlay table — that is cheap and
            # cannot crash matplotlib.
            csv = png_path.with_suffix(".tsv")
            try:
                _write_overlay_csv(out_dir, args.n_steps, args.nx_list, csv)
            except Exception as e:
                print(f"  [plot] WARNING: CSV write failed: {e}", flush=True)
            return
        try:
            plot_overlay(out_dir, args.n_steps, args.nx_list, png_path)
        except Exception as e:
            print(f"[mesh convergence] WARNING: plot step crashed "
                  f"({type(e).__name__}: {e}); continuing with next mesh.",
                  flush=True)

    # ---- Run sequence ------------------------------------------------------
    if not args.plot_only:
        for n_x in args.nx_list:
            rc = run_single_mesh(n_x, args.n_steps, extra_args=args.extra_arg)
            if rc != 0:
                print(f"[mesh convergence] WARNING: nx={n_x} run failed "
                      f"(exit {rc}).  Continuing with the next mesh.",
                      flush=True)
            # Update plot progressively (failures are swallowed)
            _safe_plot()

    # ---- Final plot --------------------------------------------------------
    _safe_plot()
    print("\n[mesh convergence] done.", flush=True)


if __name__ == "__main__":
    main()
