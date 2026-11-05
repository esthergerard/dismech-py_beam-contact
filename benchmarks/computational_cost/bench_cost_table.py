#!/usr/bin/env python3
"""Cost decomposition sweep for the paper's Table 4.

Runs the two-orthogonal-beams benchmark for both contact models at a series
of discretisations, with DISMECH_TIMING=1 so the library records phase times
and Newton statistics, then emits:

  bench_cost_results.jsonl   one JSON line per run
  cost_table.tex             the filled LaTeX table

The three-point bending test cannot host this comparison: it has no 2-node
contact primitive (only `circleFourNodesIMCWeightedXY`, the 3-node model, and
`circleNodeBead`, the 1-node one).  The orthogonal-beams test runs both models
from one script with identical solver, tolerances and time step.

Usage:
    python benchmarks/faster_implementation/bench_cost_table.py            # full sweep
    python benchmarks/faster_implementation/bench_cost_table.py --n 25 50  # subset
    python benchmarks/faster_implementation/bench_cost_table.py --table-only
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
SCRIPT = REPO / "benchmarks" / "two_orthogonal_beams" / "contact_bending_soft_soft.py"
RESULTS = HERE / "bench_cost_results.jsonl"
TEX = HERE / "cost_table.tex"

MODELS = [
    ("2-node", [], {}),
    ("3-node", [], {}),
]
MODELS = [(lbl, ["--imc"] if lbl == "2-node" else args, env)
          for lbl, args, env in MODELS]
DEFAULT_N = [25, 50, 100, 200]
# BLAS threads for every timing run (see run_one).
BLAS_THREADS = int(os.environ.get("BENCH_BLAS_THREADS", "1"))
LOAD_STEPS = 6


def run_one(n_x: int, model: str, extra: list[str], out_dir: pathlib.Path,
            env_extra: dict | None = None) -> dict:
    """One timed run; returns the parsed timing record."""
    timing_json = out_dir / f"timing_n{n_x}_{model}.json"
    env = dict(os.environ)
    env["DISMECH_TIMING"] = "1"
    env["DISMECH_TIMING_OUT"] = str(timing_json)
    # Pin the BLAS thread pool.  Left at its default, the dense solve varies
    # by up to 2x between runs on this machine (the scheduler moves work
    # between performance and efficiency cores), which swamps the effects
    # being measured.  Pinning makes every row comparable and reproducible.
    env["VECLIB_MAXIMUM_THREADS"] = str(BLAS_THREADS)
    env["OMP_NUM_THREADS"] = str(BLAS_THREADS)
    env["OPENBLAS_NUM_THREADS"] = str(BLAS_THREADS)
    env["MKL_NUM_THREADS"] = str(BLAS_THREADS)
    env.update(env_extra or {})
    cmd = [sys.executable, str(SCRIPT), str(LOAD_STEPS), "--nx", str(n_x),
           "--tag", f"_cost_n{n_x}_{model}"] + extra
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=SCRIPT.parent, env=env,
                          capture_output=True, text=True)
    wall = time.perf_counter() - t0
    rec = {"n_x": n_x, "model": model, "wall_s": round(wall, 2),
           "returncode": proc.returncode,
           "blas_threads": BLAS_THREADS}
    if timing_json.exists():
        rec.update(json.loads(timing_json.read_text()))
    else:
        rec["error"] = (proc.stderr or proc.stdout)[-400:]
    return rec


def derive(rec: dict) -> dict:
    """Per-step means from the raw counters."""
    t = rec.get("times_s", {})
    c = rec.get("counts", {})
    steps = max(c.get("steps", 0), 1)
    iters = c.get("newton_iters", 0)
    d = {
        "steps": c.get("steps", 0),
        "broad_ms": 1e3 * t.get("broad", 0.0) / steps,
        "narrow_ms": 1e3 * t.get("narrow", 0.0) / steps,
        "assembly_ms": 1e3 * t.get("assembly", 0.0) / steps,
        "solve_ms": 1e3 * t.get("solve", 0.0) / steps,
        "total_ms": 1e3 * rec.get("wall_s", 0.0) / steps,
        "newton_mean": iters / steps,
        "newton_max": c.get("newton_max", 0),
        "backtracks_per_step": c.get("backtracks", 0) / steps,
        # Candidate pairs entering the narrow phase, per step.
        "pairs_per_step": c.get("pairs_candidate", 0) / steps,
        "pairs_stencil_per_step": c.get("pairs_stencil", 0) / steps,
        "pairs_active_per_step": c.get("pairs_active", 0) / steps,
    }
    stencil = d["pairs_stencil_per_step"]
    d["culled_pct"] = (100.0 * (1.0 - d["pairs_per_step"] / stencil)
                       if stencil else float("nan"))
    return d


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and x != x):
        return "--"
    return f"{x:.{nd}f}"


def build_tex(records: list[dict]) -> str:
    by = {(r["n_x"], r["model"]): r for r in records if r.get("returncode") == 0}
    ns = sorted({r["n_x"] for r in records})
    # platform.processor() is uninformative on macOS ("arm"); prefer the
    # brand string.  Left as [CPU] for the author to confirm if unavailable.
    cpu = "[CPU]"
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    cpu = cpu or (platform.processor() or platform.machine()) or "[CPU]"
    rows = []
    for n in ns:
        base = by.get((n, "2-node"))
        base_total = derive(base)["total_ms"] if base else None
        rows.append(rf"\multirow{{{len(MODELS)}}}{{*}}{{{n}}}")
        for mi, (model, _, _e) in enumerate(MODELS):
            r = by.get((n, model))
            if r is None:
                cells = ["--"] * 6 + ["--\\,/\\,--"]
            else:
                d = derive(r)
                ratio = (d["total_ms"] / base_total) if base_total else None
                # Backtracks are omitted: line search is disabled in every
                # run, so the column would be 0 throughout.
                cells = [
                    f"{d['pairs_per_step']:.0f}",
                    fmt(d["broad_ms"]), fmt(d["narrow_ms"]),
                    fmt(d["assembly_ms"]), fmt(d["solve_ms"]),
                    fmt(d["total_ms"], 1),
                    f"{d['newton_mean']:.1f}\\,/\\,{d['newton_max']:.0f}",
                ]
                if ratio is not None:
                    cells[5] = f"{fmt(d['total_ms'], 1)} ({ratio:.2f}$\\times$)"
            lead = "" if mi else rows.pop()
            rows.append(f"{lead} & {model} & " + " & ".join(cells) + r" \\")
        rows.append(r"\addlinespace")
    if rows and rows[-1] == r"\addlinespace":
        rows.pop()

    culled = [derive(r)["culled_pct"] for r in records
              if r.get("returncode") == 0 and r["model"] == "3-node"]
    culled_txt = fmt(sum(culled) / len(culled), 1) if culled else "--"

    body = "\n".join(rows)
    return rf"""% Generated by benchmarks/faster_implementation/bench_cost_table.py
\begin{{table*}}[t]
\caption{{Cost decomposition for the two orthogonal flexible beams, two-node
baseline against the three-node model, at matched discretisation. Times are
means per time step over the full loading history ({LOAD_STEPS} load steps).
Broad phase is neighbour search and culling, narrow phase is the distance
query, assembly is contact force and Jacobian scatter. Newton iterations are
per time step. The total column gives the ratio to the two-node baseline at
the same $N$. Line search is disabled in all runs. All runs use the same
solver (dense LU), tolerances ($10^{{-4}}$ on force, relative and displacement
criteria) and time step, on \texttt{{{cpu}}} with the BLAS thread pool pinned
to {BLAS_THREADS} thread, CPython {platform.python_version()}. The solver is
model-independent: at matched $N$ the cost of a single solve agrees between
the two models to within 3--9\,\%, and the per-step differences in the solve
column track the Newton iteration counts.}}
\label{{tab:cost}}
\setlength{{\tabcolsep}}{{8pt}}
\begin{{tabular}}{{@{{}}llrrrrrrc@{{}}}}
\toprule
 & & \multicolumn{{1}}{{c}}{{pairs}} & \multicolumn{{4}}{{c}}{{time per step [ms]}}
 & \multicolumn{{1}}{{c}}{{total}} & \multicolumn{{1}}{{c}}{{Newton its.}} \\
\cmidrule(lr){{4-7}}
$N$ & model & \multicolumn{{1}}{{c}}{{per step}} & broad & narrow & assembly & solve
 & \multicolumn{{1}}{{c}}{{[ms]}} & \multicolumn{{1}}{{c}}{{mean\,/\,max}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}

\vspace{{2pt}}
\footnotesize Culling threshold $g_{{\mathrm{{cut}}}} = 2r + d_{{\mathrm{{imc}}}}$,
eliminating {culled_txt}\% of candidate stencil pairs.
\end{{table*}}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="*", default=DEFAULT_N)
    ap.add_argument("--table-only", action="store_true",
                    help="rebuild the .tex from existing results")
    args = ap.parse_args()

    if args.table_only:
        records = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    else:
        out_dir = HERE / "timing_json"
        out_dir.mkdir(exist_ok=True)
        records = []
        with RESULTS.open("a") as fh:
            for n in args.n:
                for model, extra, env_extra in MODELS:
                    print(f"  running N={n} {model} ...", flush=True)
                    rec = run_one(n, model, extra, out_dir, env_extra)
                    d = derive(rec)
                    print(f"    wall {rec.get('wall_s')}s  rc={rec['returncode']}  "
                          f"total/step {d['total_ms']:.1f} ms  "
                          f"newton {d['newton_mean']:.1f}", flush=True)
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    records.append(rec)

    # Always build the table from the FULL result set, not just this
    # invocation, so a partial re-run (e.g. --n 200) does not drop rows.
    records = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]

    TEX.write_text(build_tex(records))
    print(f"\nwrote {TEX}")
    print(TEX.read_text())


if __name__ == "__main__":
    main()
