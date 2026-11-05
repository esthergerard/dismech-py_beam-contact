#!/usr/bin/env python3
"""
Schematic for Appendix A.2 (static multiplicity probe): side-view geometry
of the three rigid configurations swept by apex_multiplicity_probe.py, at
the critical alignment (xs = 0) where the active-pair count peaks.

  (a) straight rod A + sphere obstacle, sphere directly under a node
  (b) rod A with a 3-node apex (parabola) + nearly straight rod B,
      obstacle apex directly over the node
  (c) rod A and rod B both with a 3-node apex, apexes facing

In every panel the node shared by two segments (the node realizing the
contact at xs = 0) is marked, and its two flanking segments are drawn bold:
both push on the same obstacle at once, which is the double-counting the
appendix text describes.

Geometry constants mirror apex_multiplicity_probe.py -- keep in sync; not
imported directly so this script stays free of the dismech/JAX dependency
chain needed for the actual force sweep.

Writes figures/fig_multiplicity_schematic.{pdf,svg} (+ .png preview).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "net_pullout" / "postprocessing"))
from plot_style import LAB_COLORS, apply_style, save_figure

FIG = HERE / "figures"

# ---- geometry (mirrors apex_multiplicity_probe.py) -------------------------
A = 1e-3                 # rod / sphere radius [m]
H = 1e-2                 # edge length [m]
NROD = 13                # nodes per rod (center node at x = 0)
PEN = 0.05 * A           # centerline approach 1.95a
D0 = 2 * A - PEN         # rod-A-to-rod-B / sphere centerline offset
C2 = 2.5                 # apex parabola y = -C2 x^2  (kappa = 5 /m)
C2_STRAIGHT = 0.01       # "straight" rod B bow in (b)
XGRID = (np.arange(NROD) - (NROD - 1) / 2) * H
CENTER = (NROD - 1) // 2  # index of x = 0

C_A = LAB_COLORS[3]       # blue   -- rod A
C_B = LAB_COLORS[0]       # orange -- rod B / sphere
C_HL = LAB_COLORS[2]      # red    -- shared / apex node + its two segments

MM = 1e3

# Cross-section radius and centerline offset are drawn at EXAG x true scale
# so the (otherwise sub-millimetre) contact gap reads at this print size;
# node spacing and apex curvature (set by h) are kept to true scale.
EXAG = 3.0
A_D = EXAG * A
D0_D = EXAG * D0


def draw_rod(ax, x, y, color, *, highlight=None):
    """Filled band of radius A_D + centerline + node ticks."""
    ax.fill_between(x * MM, (y - A_D) * MM, (y + A_D) * MM,
                    color=color, alpha=0.35, linewidth=0, zorder=1)
    ax.plot(x * MM, y * MM, "-", color=color, lw=0.7, zorder=2)
    ax.plot(x * MM, y * MM, "o", mfc="none", mec=color, ms=2.5, zorder=3)
    if highlight is not None:
        lo, hi = max(highlight - 1, 0), min(highlight + 1, len(x) - 1)
        ax.plot(x[lo:highlight + 1] * MM, y[lo:highlight + 1] * MM,
                "-", color=C_HL, lw=2.2, zorder=4)
        ax.plot(x[highlight:hi + 1] * MM, y[highlight:hi + 1] * MM,
                "-", color=C_HL, lw=2.2, zorder=4)
        ax.plot(x[highlight] * MM, y[highlight] * MM, "o",
                mfc=C_HL, mec=C_HL, ms=4.5, zorder=5)


def draw_sphere(ax, cx, cy, color):
    ax.add_patch(plt.Circle((cx * MM, cy * MM), A_D * MM,
                            facecolor=color, alpha=0.35,
                            edgecolor=color, lw=0.7, zorder=1))


def draw_slide_arrow(ax, y_mm, half_span_mm, *, label=None):
    """Double-headed arrow marking the translation (not rotation) the swept
    body undergoes; drawn behind the rod/sphere (zorder=0)."""
    ax.annotate("", xy=(half_span_mm, y_mm), xytext=(-half_span_mm, y_mm),
                arrowprops=dict(arrowstyle="<->", color="0.35", lw=0.8,
                                mutation_scale=9),
                zorder=0)
    if label:
        ax.text(half_span_mm + 1.5, y_mm, label, color="0.35",
               fontsize=6.5, ha="left", va="center")


def panel_common(ax, label, xspan, ylim, *, xlabel=False):
    ax.set_xlim(-xspan * MM, xspan * MM)
    ax.set_ylim(ylim[0] * MM, ylim[1] * MM)
    ax.set_yticks([])
    if xlabel:
        ax.set_xlabel(r"$x$ [mm]")
    else:
        ax.tick_params(axis="x", labelbottom=False)
    ax.text(0.015, 0.95, label, transform=ax.transAxes,
            ha="left", va="top")


def place_row(fig, bottom_frac, height_frac, *, left=0.03, width=0.94):
    return fig.add_axes((left, bottom_frac, width, height_frac))


def main() -> None:
    apply_style()

    # zoom to the +/-4h window around the junction -- the far field of the
    # rod carries no additional information about the double-counting
    xspan = 4.0 * H + 0.5 * H

    ylim_a = (-D0_D - 1.3 * A_D, 1.3 * A_D)
    yA_full = -C2 * XGRID ** 2
    ylim_b = (yA_full.min() - 1.3 * A_D,
             (D0_D + C2_STRAIGHT * XGRID ** 2).max() + 1.3 * A_D)
    ylim_c = (yA_full.min() - 1.3 * A_D,
             (D0_D + C2 * XGRID ** 2).max() + 1.3 * A_D)

    # ---- lay out 3 rows of IDENTICAL width, height set by true data aspect
    FIG_W = 3.5                      # advisor spec: single-column width [in]
    LEFT, WIDTH = 0.03, 0.94
    axw_in = FIG_W * WIDTH
    aspects = [(yl[1] - yl[0]) / (2 * xspan) for yl in (ylim_a, ylim_b, ylim_c)]
    heights_in = [axw_in * asp for asp in aspects]
    top_in, bottom_in, gap_in, legend_in = 0.06, 0.38, 0.10, 0.52
    fig_h = top_in + bottom_in + legend_in + 2 * gap_in + sum(heights_in)

    fig = plt.figure(figsize=(FIG_W, fig_h))
    y_cursor = fig_h - top_in
    axes = []
    for h_in in heights_in:
        bottom = (y_cursor - h_in) / fig_h
        axes.append(place_row(fig, bottom, h_in / fig_h, left=LEFT, width=WIDTH))
        y_cursor -= h_in + gap_in
    ax_a, ax_b, ax_c = axes

    # ---- (a) straight rod + sphere -----------------------------------------
    yA = np.zeros(NROD)
    draw_slide_arrow(ax_a, -D0_D * MM, 2.0 * H * MM)
    draw_rod(ax_a, XGRID, yA, C_A, highlight=CENTER)
    draw_sphere(ax_a, 0.0, -D0_D, C_B)
    panel_common(ax_a, "(a)", xspan, ylim_a)

    # ---- (b) bent rod (apex) + nearly straight rod -------------------------
    yA = -C2 * XGRID ** 2
    yB = D0_D + C2_STRAIGHT * XGRID ** 2
    draw_slide_arrow(ax_b, float(yB[CENTER] * MM), 2.0 * H * MM)
    draw_rod(ax_b, XGRID, yA, C_A, highlight=CENTER)
    draw_rod(ax_b, XGRID, yB, C_B)
    panel_common(ax_b, "(b)", xspan, ylim_b)

    # ---- (c) two bent rods, apexes facing ----------------------------------
    yA = -C2 * XGRID ** 2
    yB = D0_D + C2 * XGRID ** 2
    draw_slide_arrow(ax_c, float(yB[CENTER] * MM), 2.0 * H * MM)
    draw_rod(ax_c, XGRID, yA, C_A, highlight=CENTER)
    draw_rod(ax_c, XGRID, yB, C_B, highlight=CENTER)
    panel_common(ax_c, "(c)", xspan, ylim_c, xlabel=True)

    handles = [
        plt.Line2D([], [], color=C_A, lw=2, label="rod A"),
        plt.Line2D([], [], color=C_B, lw=2, label="rod B / obstacle"),
        plt.Line2D([], [], color=C_HL, lw=2.2, marker="o", ms=4.5,
                   label="node shared by 2 segments"),
        plt.Line2D([], [], color="0.35", lw=0.8, marker="$\\leftrightarrow$",
                   ms=7, label="slide direction"),
    ]
    leg = fig.legend(handles, [h.get_label() for h in handles],
                     loc="lower center", bbox_to_anchor=(0.5, 0.0),
                     ncol=2, frameon=True, fontsize=6.5,
                     columnspacing=1.2, handletextpad=0.5)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.6)

    save_figure(fig, FIG, "fig_multiplicity_schematic")
    fig.savefig(FIG / "fig_multiplicity_schematic.png", dpi=300)
    print(f"  wrote {FIG / 'fig_multiplicity_schematic.png'}")


if __name__ == "__main__":
    main()
