#!/usr/bin/env python3
"""
Manuscript figure for Appendix B.5: numerical verification that an Eberly
contact stencil pair carries zero blend-weight at every active-set
transition (two-orthogonal-beams benchmark, mu = 0, 21 nodes per beam).

Reuses pair_event_verification.py's data loading and the lab figure style
(plot_style.py).  Two panels:
  (a) weight product w = w1*w2 at each of the 164 transitions, in call
      order -- shows every transition sitting near machine-zero on a log
      scale, with the median and worst-case values marked.
  (b) w vs. 1-|s*| (log-log) -- shows the blend's quartic falloff: even the
      worst-case transitions, where the neighbour's arc parameter is only
      ~1e-4 away from an arc end, already carry a weight ~1e-8, i.e. the
      weight vanishes much faster than the arc parameter approaches the
      arc end.

Writes figures/fig_pair_event_verification.{pdf,svg}.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pair_event_verification import load_events, split_center_vs_transitions, LOG_FILE  # noqa: E402
from plot_style import LAB_COLORS, apply_style, panel_legend, save_figure, setup_axes  # noqa: E402
from dismech.contact.circle_four_nodes_imc_weighted import compute_weights_from_s  # noqa: E402

OUT_DIR = Path(__file__).parent / "figures"
OUT_BASENAME = "fig_pair_event_verification"

C_W = LAB_COLORS[1]     # green -- weight product
C_REF = LAB_COLORS[4]   # black -- reference lines / guide

LABEL_FONTSIZE = 10
SMALL_FONTSIZE = 8
PANEL_LABEL_KW = dict(fontsize=LABEL_FONTSIZE, fontweight="bold",
                      va="top", ha="left")
PANEL_LABEL_OFFSET_PT = (5.0, -5.0)


def main() -> None:
    events = load_events(LOG_FILE)
    _, transitions = split_center_vs_transitions(events)

    calls = np.array([e["call"] for e in transitions])
    s1 = np.array([e["s_i1"] for e in transitions])
    s2 = np.array([e["s_i2"] for e in transitions])
    w1 = compute_weights_from_s(s1)
    w2 = compute_weights_from_s(s2)
    w = w1 * w2
    dev1 = 1.0 - np.abs(s1)

    order = np.argsort(calls)
    calls, w, dev1 = calls[order], w[order], dev1[order]

    apply_style()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # (a) weight product per transition, in call order.
    idx = np.arange(1, len(w) + 1)
    ax_a.scatter(idx, w, s=14, facecolor="none", edgecolor=C_W,
                linewidth=1.0, zorder=3)
    ax_a.axhline(np.median(w), color=C_REF, linestyle="--", linewidth=1.0,
                label=f"median = {np.median(w):.1e}")
    ax_a.axhline(w.max(), color=C_REF, linestyle=":", linewidth=1.0,
                label=f"max = {w.max():.1e}")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("transition index (call order)")
    ax_a.set_ylabel(r"weight product $w = w_1 w_2$")
    setup_axes(ax_a)
    panel_legend(ax_a, *ax_a.get_legend_handles_labels(), loc="upper right",
                fontsize=SMALL_FONTSIZE)

    # (b) w vs. 1-|s*|: the quartic blend falloff near an arc end.
    ax_b.scatter(dev1, w, s=14, facecolor="none", edgecolor=C_W,
                linewidth=1.0, zorder=3)
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlabel(r"$1-|s^\ast|$  (neighbour's distance from an arc end)")
    ax_b.set_ylabel(r"weight product $w = w_1 w_2$")
    setup_axes(ax_b)

    for ax, lab in ((ax_a, "(a)"), (ax_b, "(b)")):
        ax.annotate(lab, xy=(0, 1), xycoords="axes fraction",
                    xytext=PANEL_LABEL_OFFSET_PT, textcoords="offset points",
                    **PANEL_LABEL_KW)

    fig.tight_layout()
    save_figure(fig, OUT_DIR, OUT_BASENAME)
    plt.close(fig)


if __name__ == "__main__":
    main()
