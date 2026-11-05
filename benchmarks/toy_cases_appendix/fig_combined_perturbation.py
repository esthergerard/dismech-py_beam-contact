#!/usr/bin/env python3
"""
Combined two-panel manuscript figure for the 2-node perturbation study:

  (a) baseline sliding trace |F_c| vs s/h with active-set change events
      (the trace_R1 plot),
  (b) log-log residual peak-to-peak A vs edge length h with power-law fit
      (the A_vs_h plot).

Reuses the metrics pipeline of analyze_perturbation.py and the lab figure
style (plot_style.py): CM fonts via mathtext (no local LaTeX), lab colour
palette, hollow markers, black-edged legend, PDF + SVG output.

Writes figures/fig_perturbation_combined.{pdf,svg} (+ .png preview).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_perturbation import metrics
from plot_style import LAB_COLORS, apply_style, panel_legend, save_figure

FIG = HERE / "figures"

C_DATA = LAB_COLORS[3]   # cyan/blue -- data curve / measured points
C_ACC = LAB_COLORS[2]    # red       -- events / fit line

S_MAX = 4.0              # panel (a) window, in units of s/h


def main() -> None:
    apply_style()

    m1 = metrics("R1")
    M = {r: metrics(r) for r in ("R1", "R5", "R6")}
    hs = np.array([M[r]["p"]["h"] for r in ("R1", "R5", "R6")])
    As = np.array([M[r]["A"] for r in ("R1", "R5", "R6")])
    cf, cov = np.polyfit(np.log(hs), np.log(As), 1, cov=True)
    slope, serr = cf[0], np.sqrt(cov[0, 0])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.8, 2.8))

    # ---- (a) baseline trace with active-set events -------------------------
    sel = m1["s_h"] <= S_MAX
    ax_a.plot(m1["s_h"][sel], m1["F"][sel], "-", color=C_DATA, lw=0.8,
              label=r"$|F_c|$")
    evs = [i for i in m1["events"] if m1["s_h"][i] <= S_MAX]
    ax_a.plot(m1["s_h"][evs], m1["F"][evs], "x", color=C_ACC, ms=5,
              label="active-set change")
    ax_a.set_xlabel(r"sliding distance $s/h$")
    ax_a.set_ylabel(r"contact force $|F_c|$  ($k_c = 1$)")
    # headroom so neither the opaque legend nor the (a) tag covers the peaks
    ax_a.set_ylim(0.0, 1.35 * m1["F"][sel].max())
    panel_legend(ax_a, *ax_a.get_legend_handles_labels(), "upper right")

    # ---- (b) A vs h, log-log, with power-law fit ----------------------------
    ax_b.loglog(hs * 1e3, As, "o", mfc="none", mec=C_DATA, label="measured")
    hf = np.linspace(hs.min(), hs.max(), 50)
    ax_b.loglog(hf * 1e3, np.exp(cf[1]) * hf ** slope, "-", color=C_ACC,
                lw=1.0, label=rf"fit: slope ${slope:.2f} \pm {serr:.2f}$")
    ax_b.set_xlabel(r"edge length $h$ [mm]")
    ax_b.set_ylabel(r"residual peak-to-peak $A$")
    ax_b.margins(x=0.1, y=0.15)   # keep the extreme markers off the spines
    # Plain-number y ticks (100, 120, ...) instead of the log axis's default
    # "1 x 10^2" style, so this panel reads consistently with (a)'s linear
    # axis -- both cover a similar numeric range.  The data span under one
    # decade, so matplotlib's default log locator only places a single
    # MAJOR tick (100) and relies on minor ticks for 120..200; promote
    # those same values to major (plain-formatted) ticks instead of just
    # killing their (sci-notation) labels.
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    ylo, yhi = ax_b.get_ylim()
    yt = [v for v in range(100, 241, 20) if ylo <= v <= yhi]
    ax_b.yaxis.set_major_locator(FixedLocator(yt))
    yfmt = ScalarFormatter()
    yfmt.set_scientific(False)
    ax_b.yaxis.set_major_formatter(yfmt)
    ax_b.yaxis.set_minor_formatter(NullFormatter())
    # legend in the free lower-right corner so the top-left hosts the (b) tag
    panel_legend(ax_b, *ax_b.get_legend_handles_labels(), "lower right")

    fig.tight_layout()
    for ax, letter in ((ax_a, "(a)"), (ax_b, "(b)")):
        ax.text(0.035, 0.955, letter, transform=ax.transAxes,
                ha="left", va="top")

    save_figure(fig, FIG, "fig_perturbation_combined")
    fig.savefig(FIG / "fig_perturbation_combined.png", dpi=300)
    print(f"  wrote {FIG / 'fig_perturbation_combined.png'}")


if __name__ == "__main__":
    main()
