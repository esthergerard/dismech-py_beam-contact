#!/usr/bin/env python3
"""
FINAL 2-node (IMC) vs 3-node (Eberly) PLANAR comparison (Beam Z x-DOF
fixed -- no lateral/out-of-plane sliding), 29 load steps, both mu = 0
and mu = 0.1.  Both models now run the COMPLETE benchmark cleanly
(`--eberly-straight-R-threshold 2e2`, see NOTES.md "VALIDATED FIX,
threshold=2e2" -- fixes the Eberly planar livelock without any
src/dismech change).  Supersedes fig_planar_mu01_comparison.py, whose
3-node run was truncated by the (now-fixed) livelock.

  (a) mu = 0,   full 29 s record, both models overlaid
  (b) mu = 0.1, full 29 s record, both models overlaid
  (c) mu = 0,   zoom on the tail (t = 20..29 s) -- where element
      crossings concentrate node-passing jitter
  (d) mu = 0.1, same zoom

Jitter metrics (established-contact phase, computed by
table_soft_soft_jitter.py --steps 29 --settling 2 --variant planarRth2):
the 3-node model wins ALL 10 metrics across both friction cases; at
mu = 0 its 3-point metric (5.8e-05) is 3 orders of magnitude below the
2-node's (5.5e-02) -- i.e. essentially free of the non-monotonic
node-passing jitter that metric targets.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import LAB_COLORS, apply_style, save_figure, setup_axes  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "contact_output_files_soft_soft"
TRACES = {
    ("mu0",  "2-node"): DATA_DIR / "zavarise_per_dt_trace_steps29_fix_planarRth2_imc_mu0.txt",
    ("mu0",  "3-node"): DATA_DIR / "zavarise_per_dt_trace_steps29_fix_planarRth2_eberly_mu0.txt",
    ("mu01", "2-node"): DATA_DIR / "zavarise_per_dt_trace_steps29_fix_planarRth2_imc_mu01.txt",
    ("mu01", "3-node"): DATA_DIR / "zavarise_per_dt_trace_steps29_fix_planarRth2_eberly_mu01.txt",
}

ZOOM_T = (20.0, 29.0)
COLOR_2N = tuple(LAB_COLORS[0])
COLOR_3N = tuple(LAB_COLORS[1])

OUT_DIR      = Path(__file__).parent / "figures"
OUT_BASENAME = "fig_planar_final_comparison_steps29"


def load_Fy(path: Path):
    a = np.loadtxt(path, comments="#")
    return a[:, 0], np.abs(a[:, 2])


def _draw_pair(ax, mu_key, t_range=None, lw=1.0):
    handles, labels = [], []
    for label, color in (("2-nodes", COLOR_2N), ("3-nodes", COLOR_3N)):
        key = (mu_key, "2-node" if label == "2-nodes" else "3-node")
        t, fy = load_Fy(TRACES[key])
        if t_range is not None:
            m = (t >= t_range[0]) & (t <= t_range[1])
            t, fy = t[m], fy[m]
        h, = ax.plot(t, fy, color=color, linewidth=lw)
        handles.append(h); labels.append(label)
    setup_axes(ax)
    ax.set_ylabel(r"$|F_{N,y}|$ on Beam X  [N]")
    return handles, labels


def _legend(ax, handles, labels, loc="lower right"):
    leg = ax.legend(handles, labels, loc=loc, frameon=True, fontsize=8,
                    handlelength=2.0)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.6)


def main() -> None:
    apply_style()
    fig = plt.figure(figsize=(9.0, 6.4))
    gs = fig.add_gridspec(2, 2, left=0.08, right=0.98, top=0.95, bottom=0.08,
                          hspace=0.32, wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    h, l = _draw_pair(ax_a, "mu0")
    _legend(ax_a, h, l)
    ax_a.set_xlabel("time [s]")
    ax_a.axvspan(*ZOOM_T, color="0.88", zorder=0)
    ax_a.set_title(r"(a) planar, $\mu=0$: full record", fontsize=10)

    h, l = _draw_pair(ax_b, "mu01")
    _legend(ax_b, h, l)
    ax_b.set_xlabel("time [s]")
    ax_b.axvspan(*ZOOM_T, color="0.88", zorder=0)
    ax_b.set_title(r"(b) planar, $\mu=0.1$: full record", fontsize=10)

    h, l = _draw_pair(ax_c, "mu0", t_range=ZOOM_T, lw=1.2)
    _legend(ax_c, h, l)
    ax_c.set_xlim(*ZOOM_T)
    ax_c.set_xlabel("time [s]")
    ax_c.set_title(f"(c) $\\mu=0$: zoom t={ZOOM_T[0]:.0f}-{ZOOM_T[1]:.0f} s", fontsize=10)

    h, l = _draw_pair(ax_d, "mu01", t_range=ZOOM_T, lw=1.2)
    _legend(ax_d, h, l)
    ax_d.set_xlim(*ZOOM_T)
    ax_d.set_xlabel("time [s]")
    ax_d.set_title(f"(d) $\\mu=0.1$: zoom t={ZOOM_T[0]:.0f}-{ZOOM_T[1]:.0f} s", fontsize=10)

    for ax, lab in ((ax_a, "(a)"), (ax_b, "(b)"), (ax_c, "(c)"), (ax_d, "(d)")):
        pass  # titles already carry the panel labels

    save_figure(fig, OUT_DIR, OUT_BASENAME)
    plt.close(fig)


if __name__ == "__main__":
    main()
