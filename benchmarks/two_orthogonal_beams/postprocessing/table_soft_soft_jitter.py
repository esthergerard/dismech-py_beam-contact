#!/usr/bin/env python3
"""
Jitter metrics for the soft-soft test (two orthogonal flexible beams,
Zavarise & Wriggers 2000 Example 2), 2-node IMC vs 3-node Eberly, at
mu = 0 and mu = 0.1.

This is the soft-soft counterpart of ``fig6_jitter_metrics.py`` (three-point
bend).  It reuses the SAME ``analyze_jitter`` implementation so the numbers
are directly comparable between the two tables in the manuscript.

Signal analysed
---------------
The per-dt-step trace files record, at every time step, the contact and
friction force resultants on Beam X:

    t  Fx_contact Fy_contact Fz_contact  Fx_fric Fy_fric Fz_fric

The three-point-bend table uses "the normed sum of the vertical contact
forces".  Here the contact normal is the y-axis (Beam X is offset in y),
so the equivalent scalar signal is |F_y_contact|.  Set SIGNAL = "norm" to
use ||F_contact|| instead, or "friction" for ||F_fric||.

Only the portion of the trace where contact is actually engaged is used,
so the leading zero-force ramp does not dilute the metrics.

Output
------
  - a plain-text table (printed + figures/table_soft_soft_jitter.txt)
  - a ready-to-paste LaTeX table (figures/table_soft_soft_jitter.tex)

Usage
-----
    python table_soft_soft_jitter.py
    python table_soft_soft_jitter.py --signal norm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
# analyze_jitter lives with the three-point-bend postprocessing (shared so
# the two jitter tables of the manuscript use the same implementation)
sys.path.insert(0, str(Path(__file__).parents[2]
                       / 'three-pts_bend_test' / 'postprocessing'))
from fig6_jitter_metrics import (  # noqa: E402
    FORCE_START_THRESHOLD,
    analyze_jitter,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR  = Path(__file__).resolve().parents[1] / "contact_output_files_soft_soft"

# Smoothing window in SECONDS.  NOT 2 s as in the three-point-bend table:
# that record is several hundred seconds long, whereas the soft-soft trace is
# 6 s total.  A 2 s window would span half the analysed record and the metric
# would measure the load ramp instead of the jitter.  0.2 s is short compared
# with the 1 s load step but long compared with the ripple.
TIME_SMOOTH_WINDOW_S = 0.2

# Contact engages at t ~ 2 s (one full load step before that the beams are
# still approaching).  The first ~0.5 s after engagement is a genuine impact
# transient -- large, physical, and present in every model -- so metrics are
# computed only once contact is established: one full load step after the
# first non-zero force.  For the extended 29-step record the impact
# ring-down visibly persists to t ~ 3.5 s, so use ``--settling 2`` there
# (the analysed window is ~26 s long; losing one more second is free).
SETTLING_TIME_S = 1.0

# Which trace column group to analyse.
#   "normal"   -> |F_y_contact|          (contact normal is +y here)
#   "norm"     -> ||F_contact||
#   "friction" -> ||F_fric||
SIGNAL_DEFAULT = "normal"

# Default number of load steps (6 = paper's Case 1; 29 = paper's Case 2 —
# the longer run makes the contact point slide across element boundaries,
# which is where the 2-node node-passing jitter shows up).
N_STEPS_DEFAULT = 6


def runs_for_steps(n_steps: int, variant: str = ""):
    """One entry per (friction case, contact model).  ``label`` is what shows
    up in the table; ``group`` is the mu block it belongs to.
    All four traces are regenerated from ONE library state: the h_scale fix in
    eberly_contact_3d.py (consistent tangent, 1/h amplification off by default),
    every run with the adaptive contact stiffness and no per-run overrides.
    Regenerate with (replace 6 by the desired step count):
      python contact_bending_soft_soft.py 6 --nx 21 --per-dt-trace --tag _fix_eberly_mu0
      python contact_bending_soft_soft.py 6 --nx 21 --per-dt-trace --tag _fix_eberly_mu01 --mu 0.1
      python contact_bending_soft_soft.py 6 --nx 21 --per-dt-trace --tag _fix_imc_mu0   --imc
      python contact_bending_soft_soft.py 6 --nx 21 --per-dt-trace --tag _fix_imc_mu01  --imc --mu 0.1
    """
    s = f"steps{n_steps}"
    v = f"{variant}_" if variant else ""
    return [
        {
            "group": r"$\mu = 0$",
            "label": "2-node",
            "path": DATA_DIR / f"zavarise_per_dt_trace_{s}_fix_{v}imc_mu0.txt",
        },
        {
            "group": r"$\mu = 0$",
            "label": "3-node (ours)",
            "path": DATA_DIR / f"zavarise_per_dt_trace_{s}_fix_{v}eberly_mu0.txt",
        },
        {
            "group": r"$\mu = 0.1$",
            "label": "2-node",
            "path": DATA_DIR / f"zavarise_per_dt_trace_{s}_fix_{v}imc_mu01.txt",
        },
        {
            "group": r"$\mu = 0.1$",
            "label": "3-node (ours)",
            "path": DATA_DIR / f"zavarise_per_dt_trace_{s}_fix_{v}eberly_mu01.txt",
        },
    ]

# Metrics reported, in table order: (dict key, plain header, LaTeX header).
METRICS = [
    ("max_p2p",          "Max P2P",       "Max P2P"),
    ("mean_p2p",         "Mean P2P",      "Mean P2P"),
    ("three_point_max",  "Max 3-pt",      "Max 3-pt"),
    ("three_point_mean", "Mean 3-pt",     "Mean 3-pt"),
    ("rms_vs_smooth",    "RMS vs Smooth", "RMS vs Smooth"),
]

OUT_DIR = Path(__file__).parent / "figures"


def out_paths(n_steps: int, variant: str = ""):
    """Keep the historical names for the 6-step table (already referenced by
    the manuscript build); suffix longer runs with the step count and
    non-default variants (e.g. planar) with the variant name."""
    suffix = "" if n_steps == N_STEPS_DEFAULT else f"_steps{n_steps}"
    if variant:
        suffix += f"_{variant}"
    return (OUT_DIR / f"table_soft_soft_jitter{suffix}.txt",
            OUT_DIR / f"table_soft_soft_jitter{suffix}.tex")


# ============================================================================
# LOADING
# ============================================================================

def load_trace(path: Path):
    """Return (t, F_contact[n,3], F_fric[n,3]) from a per-dt trace file."""
    arr = np.loadtxt(path, comments="#")
    if arr.ndim != 2 or arr.shape[1] < 7:
        raise ValueError(f"{path.name}: expected 7 columns, got {arr.shape}")
    return arr[:, 0], arr[:, 1:4], arr[:, 4:7]


def build_signal(F_contact, F_fric, which: str):
    if which == "normal":
        return np.abs(F_contact[:, 1])
    if which == "norm":
        return np.linalg.norm(F_contact, axis=1)
    if which == "friction":
        return np.linalg.norm(F_fric, axis=1)
    raise ValueError(f"unknown signal {which!r}")


def metrics_for_trace(path: Path, which: str):
    """Jitter metrics on the established-contact portion of one trace.

    The window is [t_engage + SETTLING, t_release - SETTLING]: the leading
    impact transient is excluded (as before), and if the beams DISENGAGE
    before the end of the record (which happens in the extended 29-step
    runs at mu = 0.1 -- Beam Z snaps past Beam X at t ~ 18 s) the trailing
    release transient and the dead zero tail are excluded symmetrically.
    Traces that stay in contact to the end keep the historical window.
    """
    t, F_contact, F_fric = load_trace(path)
    y = build_signal(F_contact, F_fric, which)

    engaged = np.abs(y) > FORCE_START_THRESHOLD
    if not engaged.any():
        return None, 0, (np.nan, np.nan)
    t_engage = t[int(np.argmax(engaged))]

    i_last = len(engaged) - 1 - int(np.argmax(engaged[::-1]))
    if i_last < len(engaged) - 1:          # disengages before end of record
        t_release = t[i_last]
        keep = (t >= t_engage + SETTLING_TIME_S) & \
               (t <= t_release - SETTLING_TIME_S)
    else:
        keep = t >= t_engage + SETTLING_TIME_S
    t_e, y_e = t[keep], y[keep]
    if y_e.size < 3:
        return None, y_e.size, (np.nan, np.nan)

    dt = float(np.median(np.diff(t_e)))
    window = max(3, int(round(TIME_SMOOTH_WINDOW_S / max(dt, 1e-12))))
    return analyze_jitter(y_e, t_e, window), y_e.size, (t_e[0], t_e[-1])


# ============================================================================
# FORMATTING
# ============================================================================

def _fmt_plain(x):
    if x is None or not np.isfinite(x):
        return "NaN"
    return f"{x:.1e}"


def _fmt_tex(x, bold=False):
    """Match the three-point-bend table's '7.8e-6' style."""
    if x is None or not np.isfinite(x):
        return "---"
    s = f"{x:.1e}"
    return rf"\textbf{{{s}}}" if bold else s


def print_plain_table(rows, which, out_lines):
    header = ["mu", "Model"] + [h for _, h, _ in METRICS]
    widths = [max(len(header[0]), 8), max(len(header[1]), 14)] + \
             [max(len(h), 9) for _, h, _ in METRICS]

    def line(cells):
        return "  ".join(f"{c:>{w}}" for c, w in zip(cells, widths))

    title = (f"Soft-soft jitter metrics  --  signal = {which}, "
             f"smoothing window = {TIME_SMOOTH_WINDOW_S} s")
    head = line(header)
    for s in (title, "=" * len(head), head, "-" * len(head)):
        print(s); out_lines.append(s + "\n")

    for r in rows:
        cells = [r["group_plain"], r["label"]] + \
                [_fmt_plain(r["metrics"].get(k) if r["metrics"] else None)
                 for k, _, _ in METRICS]
        s = line(cells)
        print(s); out_lines.append(s + "\n")


def build_latex_table(rows, which, n_steps=N_STEPS_DEFAULT,
                      peak_force=71.0, variant="") -> str:
    """LaTeX table in the same shape as the three-point-bend one."""
    # Bold the better of the two models within each mu block, per metric.
    best = {}
    for group in dict.fromkeys(r["group"] for r in rows):
        block = [r for r in rows if r["group"] == group and r["metrics"]]
        for key, _, _ in METRICS:
            vals = [(r["label"], r["metrics"][key]) for r in block
                    if np.isfinite(r["metrics"].get(key, np.nan))]
            if vals:
                best[(group, key)] = min(vals, key=lambda kv: kv[1])[0]

    col_headers = " & ".join(h for _, _, h in METRICS)
    signal_desc = {
        "normal":   r"the normal contact force $|F_{N,y}|$ on Beam~X",
        "norm":     r"the normed contact force $\|\mathbf{F}_N\|$ on Beam~X",
        "friction": r"the normed friction force $\|\mathbf{F}_T\|$ on Beam~X",
    }[which]

    lines = [
        r"\begin{table*}[ht]",
        r"    \centering",
        r"    \small",
        r"    \caption{Jitter metrics of the soft-soft test (two orthogonal "
        r"flexible beams, $N_x = 21$, "
        f"{n_steps}" + r"~load steps"
        + (r", planar: Beam~Z $x$-DOF fixed" if variant == "planar"
           else (f", {variant}" if variant else ""))
        + r") for the different contact models, "
        r"computed on the time signal of " + signal_desc + r" over the "
        r"established-contact phase (from "
        f"{SETTLING_TIME_S:.0f}" + r"~s after first contact to the end of "
        r"loading, with a "
        f"{TIME_SMOOTH_WINDOW_S:.1f}" + r"~s smoothing window for the "
        r"RMS-vs-smooth metric), at $\mu = 0$ and $\mu = 0.1$. Forces are "
        r"in newtons; the peak normal force is $\approx "
        f"{peak_force:.0f}" + r"$~N in every case.}",
        r"    \label{tab:metrics_jitter_soft_soft}",
        r"    \begin{tabular}{llccccc}",
        r"        \toprule",
        r"        Friction & Simulation & " + col_headers + r" \\",
        r"        \midrule",
    ]

    prev_group = None
    for i, r in enumerate(rows):
        if prev_group is not None and r["group"] != prev_group:
            lines.append(r"        \midrule")
        group_cell = r["group"] if r["group"] != prev_group else ""
        cells = [
            _fmt_tex(r["metrics"].get(key) if r["metrics"] else None,
                     bold=best.get((r["group"], key)) == r["label"])
            for key, _, _ in METRICS
        ]
        label = r["label"] + (r"$^{\dagger}$" if r.get("note") else "")
        lines.append(f"        {group_cell} & {label} & "
                     + " & ".join(cells) + r" \\")
        prev_group = r["group"]

    notes = [r["note"] for r in rows if r.get("note")]
    if notes:
        lines += [
            r"        \midrule",
            r"        \multicolumn{7}{l}{\footnotesize $^{\dagger}$ "
            + notes[0]
            + r"; all other rows use the adaptive contact stiffness. The "
              r"frictionless 3-node case does not converge past first "
              r"contact with the adaptive stiffness.} \\",
        ]

    lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main(which: str = SIGNAL_DEFAULT, n_steps: int = N_STEPS_DEFAULT,
         settling: float = SETTLING_TIME_S, variant: str = "") -> None:
    global SETTLING_TIME_S
    SETTLING_TIME_S = float(settling)
    rows, missing = [], []
    peak_force = float("nan")
    for run in runs_for_steps(n_steps, variant):
        path = Path(run["path"])
        if not path.is_file():
            missing.append(path)
            rows.append({**run,
                         "group_plain": run["group"].replace("$", ""),
                         "metrics": None})
            continue
        m, n, span = metrics_for_trace(path, which)
        print(f"  {path.name}: {n} engaged samples, "
              f"t = {span[0]:.2f}..{span[1]:.2f} s")
        t, F_contact, F_fric = load_trace(path)
        y = build_signal(F_contact, F_fric, which)
        pk = float(np.nanmax(np.abs(y)))
        peak_force = pk if np.isnan(peak_force) else max(peak_force, pk)
        rows.append({**run,
                     "group_plain": run["group"].replace("$", ""),
                     "metrics": m})

    if missing:
        print("\n  [warn] missing trace files -- rows left blank:")
        for p in missing:
            print(f"    {p}")
        print("  Generate them with, e.g.:")
        print(f"    python contact_bending_soft_soft.py {n_steps} --nx 21 "
              "--per-dt-trace --tag _fix_eberly_mu0")
        print(f"    python contact_bending_soft_soft.py {n_steps} --nx 21 "
              "--imc --per-dt-trace --tag _fix_imc_mu0\n")

    out_lines = []
    print_plain_table(rows, which, out_lines)

    tex = build_latex_table(rows, which, n_steps=n_steps,
                            peak_force=peak_force, variant=variant)
    print("\n" + tex)

    out_txt, out_tex = out_paths(n_steps, variant)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("".join(out_lines))
    out_tex.write_text(tex)
    print(f"Wrote {out_txt}")
    print(f"Wrote {out_tex}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--signal", default=SIGNAL_DEFAULT,
                    choices=["normal", "norm", "friction"],
                    help="which force signal to analyse "
                         "(default: %(default)s)")
    ap.add_argument("--steps", type=int, default=N_STEPS_DEFAULT,
                    help="number of load steps of the trace files to "
                         "analyse (default: %(default)s; 29 = paper Case 2)")
    ap.add_argument("--settling", type=float, default=SETTLING_TIME_S,
                    help="settling time [s] excluded after first contact "
                         "(and before a disengagement, if any) "
                         "(default: %(default)s; use 2 for the 29-step "
                         "record, whose impact ring-down outlasts 1 s)")
    ap.add_argument("--variant", type=str, default="",
                    help="trace-file variant infix, e.g. 'planar' for the "
                         "zavarise_per_dt_trace_stepsN_fix_planar_* runs "
                         "(Beam Z x-DOF fixed).  Output files get the "
                         "variant appended.")
    args = ap.parse_args()
    main(args.signal, args.steps, args.settling, args.variant)
