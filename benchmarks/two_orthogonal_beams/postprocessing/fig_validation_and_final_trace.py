#!/usr/bin/env python3
"""
Composite figure: the original 6-step mesh-convergence validation against
the Zavarise & Wriggers reference (top row, reusing fig_soft_soft.py's
drawing code verbatim, just relabelled "reference" instead of "paper"),
paired with the FINAL, complete 29-step planar mu=0.1 comparison (bottom
row) that the whole --planar / --eberly-straight-R-threshold investigation
produced (see NOTES.md).

  (a) mu = 0,   F_N vs load step, mesh convergence vs reference (6 steps)
  (b) mu = 0.1, F_N (top) + F_T (bottom) vs load step, mesh convergence
      vs reference (6 steps)
  (c) mu = 0.1, |F_N,y| vs time, full 29-step planar record, both models
  (d) mu = 0.1, zoom on the tail -- the x/y box shown here is outlined
      (not shaded) on panel (c)

Does not modify fig_soft_soft.py or its output; reuses its drawing
functions with different label text.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import LAB_COLORS, apply_style, save_figure, setup_axes  # noqa: E402
from fig_soft_soft import (  # noqa: E402
    MESH_MU0, MESH_MU01,
    _draw_mesh_FN_per_step, _draw_friction_mesh,
    PANEL_LABEL_OFFSET_PT, PANEL_LABEL_KW, _legend_kwargs,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "contact_output_files_soft_soft"
TRACE_2N = DATA_DIR / "zavarise_per_dt_trace_steps29_fix_planarRth2_imc_mu01.txt"
TRACE_3N = DATA_DIR / "zavarise_per_dt_trace_steps29_fix_planarRth2_eberly_mu01.txt"

ZOOM_T = (20.0, 29.0)
COLOR_2N = tuple(LAB_COLORS[0])
COLOR_3N = tuple(LAB_COLORS[1])

OUT_DIR      = Path(__file__).parent / "figures"
OUT_BASENAME = "fig_validation_and_final_trace"


def load_Fy(path: Path):
    a = np.loadtxt(path, comments="#")
    return a[:, 0], np.abs(a[:, 2])


def _draw_trace_pair(ax, t_range=None, lw=1.0):
    handles, labels = [], []
    for label, path, color in (("2-nodes", TRACE_2N, COLOR_2N),
                              ("3-nodes", TRACE_3N, COLOR_3N)):
        t, fy = load_Fy(path)
        if t_range is not None:
            m = (t >= t_range[0]) & (t <= t_range[1])
            t, fy = t[m], fy[m]
        h, = ax.plot(t, fy, color=color, linewidth=lw)
        handles.append(h); labels.append(label)
    setup_axes(ax)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"$|F_{N,y}|$ on Beam X  [N]")
    return handles, labels


def _legend(ax, handles, labels):
    """Anchored below the panel-label corner (fig_soft_soft convention) so
    it never collides with the (c)/(d) annotation placed in the same
    corner."""
    leg = ax.legend(handles, labels,
                    **_legend_kwargs(ax, fontsize=8, handlelength=2.0))
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.6)


def main() -> None:
    apply_style()
    fig = plt.figure(figsize=(8.0, 7.2))
    outer = fig.add_gridspec(2, 2, left=0.085, right=0.97, top=0.96,
                             bottom=0.07, height_ratios=[1.0, 1.0],
                             hspace=0.40, wspace=0.30)

    # ---- top row: 6-step validation vs reference (reuse fig_soft_soft) ----
    ax_a = fig.add_subplot(outer[0, 0])
    gs_b = outer[0, 1].subgridspec(2, 1, hspace=0.12)
    ax_b_fn = fig.add_subplot(gs_b[0])
    ax_b_ft = fig.add_subplot(gs_b[1], sharex=ax_b_fn)

    _draw_mesh_FN_per_step(ax_a, MESH_MU0, paper_curve_label=r"reference, $\mu = 0$")
    _draw_friction_mesh(ax_b_fn, ax_b_ft, MESH_MU01,
                        paper_label=r"reference, $\mu = 0.1$")

    # ---- bottom row: final 29-step planar mu=0.1 trace + zoom ----
    ax_c = fig.add_subplot(outer[1, 0])
    ax_d = fig.add_subplot(outer[1, 1])

    h, l = _draw_trace_pair(ax_c)
    _legend(ax_c, h, l)

    h, l = _draw_trace_pair(ax_d, t_range=ZOOM_T, lw=1.2)
    _legend(ax_d, h, l)
    ax_d.set_xlim(*ZOOM_T)
    ax_d.text(0.95, 0.95, "ZOOM", transform=ax_d.transAxes,
              fontsize=10, ha="right", va="top")

    # Frame (outline, not shading) on panel (c) marking the exact x/y box
    # shown in panel (d).
    y_lo, y_hi = ax_d.get_ylim()
    box_color = "0.25"
    rect = Rectangle((ZOOM_T[0], y_lo), ZOOM_T[1] - ZOOM_T[0], y_hi - y_lo,
                     facecolor="none", edgecolor=box_color, linewidth=1.0,
                     zorder=5)
    ax_c.add_patch(rect)

    # Connector lines from the box's near (right-hand) corners in (c) to the
    # facing (left-hand) corners of the zoom panel (d) -- standard
    # "zoom inset" visual idiom, so the box is unambiguously tied to (d).
    for y_frac, y_data in ((1.0, y_hi), (0.0, y_lo)):
        con = ConnectionPatch(
            xyA=(ZOOM_T[1], y_data), coordsA=ax_c.transData,
            xyB=(0.0, y_frac), coordsB=ax_d.transAxes,
            color=box_color, linewidth=0.8, linestyle="--", zorder=5)
        fig.add_artist(con)

    # Panel labels, all four consistent with fig_soft_soft's own style/offset.
    dx, dy = PANEL_LABEL_OFFSET_PT
    for ax, lab in ((ax_a, "(a)"), (ax_b_fn, "(b)"), (ax_c, "(c)"), (ax_d, "(d)")):
        ax.annotate(lab, xy=(0, 1), xycoords="axes fraction",
                    xytext=(dx, dy), textcoords="offset points",
                    **PANEL_LABEL_KW)

    save_figure(fig, OUT_DIR, OUT_BASENAME)
    plt.close(fig)


if __name__ == "__main__":
    main()
