#!/usr/bin/env python3
"""
Paper-ready re-implementation of ``matlab_postprocessing/extended_arcs.png``.

(a)  A small 5-node beam (black markers + dashed centreline) overlaid with
     the THREE extended-arc fits that the regularized contact model uses
     (each arc is fitted to a sliding triplet of nodes and stretched by
     ±50 % beyond the inner-edge midpoints).  A red dashed circle
     represents the obstacle.  For every arc the minimum-distance point
     to the obstacle is marked, with the (s*, gap) pair printed next to it.

(b)  The weight function used by the model, taken VERBATIM from the
     implementation in ``src/dismech/contact/circle_four_nodes_imc_weighted.py``:

         w(s) = 1 + arctan(k (s^4 - 2 s^2)) / arctan(k),   k = 0.8204.

     The curve is drawn in black; the three (s, w(s)) pairs corresponding
     to the arcs in (a) are marked with the same colour as their arc.

Lab style: serif / cm mathtext, 10 pt labels, 8 pt legends/annotations,
inward ticks, all four spines visible, vector PDF + SVG output.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import (  # noqa: E402
    LAB_COLORS,
    apply_style,
    save_figure,
    setup_axes,
)

# Pull the exact weight function from the regularized contact source.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from dismech.contact.circle_four_nodes_imc_weighted import (  # noqa: E402
    compute_weights_from_s,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Beam: 5 nodes forming a strongly asymmetric upward arch wrapping over
# the obstacle.  Curvature is shallow on the left, very tight near the
# peak, then steep on the way down: the three sliding-triplet arcs end
# up with markedly different radii and centres, so their extended
# portions clearly fan out instead of overlaying.
NODES = np.array([
    [0.040, 0.06000],
    [0.050, 0.07100],
    [0.060, 0.07600],
    [0.070, 0.07120],
    [0.080, 0.06150],
])

# Obstacle: red disk just below the arch's peak.
OBST_CENTER = np.array([0.0625, 0.06600])
OBST_RADIUS = 0.0014
BEAM_RADIUS = 0.0010

# Per-arc colours (must keep aligned with panel-b markers).
ARC_COLORS = [
    tuple(LAB_COLORS[3]),    # arc 1 -- lab cyan
    tuple(LAB_COLORS[0]),    # arc 2 -- lab orange
    tuple(LAB_COLORS[1]),    # arc 3 -- lab green
]

LABEL_FONTSIZE  = 10
SMALL_FONTSIZE  = 8

PANEL_LABEL_KW  = dict(fontsize=LABEL_FONTSIZE, fontweight="bold",
                        va="top", ha="left")
PANEL_LABEL_OFFSET_PT = (5.0, -5.0)

FIG_SIZE = (7.0, 3.2)

OUT_DIR      = Path(__file__).parent / "figures"
OUT_BASENAME = "fig_extended_arcs"


# ============================================================================
# EXTENDED-ARC GEOMETRY  (re-implementation of the contact-module function,
# instrumented so we can read out C_beam, R, angles and the closest point.)
# ============================================================================

def _curvature_signed(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p2
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    m1 = np.linalg.norm(v1); m2 = np.linalg.norm(v2)
    if m1 < 1e-12 or m2 < 1e-12:
        return 0.0
    return 2.0 * cross / (m1 * m2 * (m1 + m2))


def _fix_arc_angles(a1, a2, kappa):
    a1 %= 2 * np.pi
    a2 %= 2 * np.pi
    if kappa > 0 and a2 < a1:
        a2 += 2 * np.pi
    if kappa < 0 and a1 < a2:
        a1 += 2 * np.pi
    return a1, a2


def extended_arc(p1, p2, p3, circle_center, circle_radius, beam_radius):
    """
    Full extended-arc info for a triplet.

    Returns a dict with:
        C, R, kappa
        angle_mid1, angle_mid2                 (inner midpoint angles)
        angle_mid1_ext, angle_mid2_ext         (extended endpoints, ± 50 %)
        closest_angle, closest_point           (point realising the min gap)
        s, gap                                 (normalised arc parameter and gap)
    """
    kappa = _curvature_signed(p1, p2, p3)
    if abs(kappa) < 1e-8:
        kappa = 1e-8 if kappa >= 0 else -1e-8
    R_beam = 1.0 / kappa
    R_curv = abs(R_beam)

    mid1 = (p1 + p2) / 2.0
    mid2 = (p2 + p3) / 2.0
    chord_midpoint = (mid1 + mid2) / 2.0
    chord_vector = mid2 - mid1
    chord_length = float(np.linalg.norm(chord_vector))
    perp = np.array([-chord_vector[1], chord_vector[0]]) / chord_length
    L_half = chord_length / 2.0
    d = float(np.sqrt(R_beam ** 2 - L_half ** 2))
    C = chord_midpoint + (d if kappa > 0 else -d) * perp

    a1 = math.atan2(mid1[1] - C[1], mid1[0] - C[0])
    a2 = math.atan2(mid2[1] - C[1], mid2[0] - C[0])
    a1, a2 = _fix_arc_angles(a1, a2, kappa)

    if kappa > 0:
        span = a2 - a1
    else:
        span = a1 - a2
    if span < 0:
        span += 2 * np.pi
    ext = span / 2.0
    if kappa > 0:
        a1_ext, a2_ext = a1 - ext, a2 + ext
    else:
        a1_ext, a2_ext = a1 + ext, a2 - ext

    # Closest point of the (extended) arc to the obstacle centre.
    dir_to = circle_center - C
    angle_to = math.atan2(dir_to[1], dir_to[0])

    def in_range(a, lo, hi):
        a = a % (2 * np.pi); lo = lo % (2 * np.pi); hi = hi % (2 * np.pi)
        if lo <= hi:
            return lo <= a <= hi
        return a >= lo or a <= hi

    if kappa > 0:
        is_in = in_range(angle_to, a1_ext, a2_ext)
    else:
        is_in = in_range(angle_to, a2_ext, a1_ext)
    if is_in:
        closest_angle = angle_to
    else:
        def ang_dist(x, y):
            diff = (y - x) % (2 * np.pi)
            return min(diff, 2 * np.pi - diff)
        d1 = ang_dist(angle_to, a1_ext)
        d2 = ang_dist(angle_to, a2_ext)
        closest_angle = a1_ext if d1 < d2 else a2_ext

    # s in [-1, 1]
    if kappa > 0:
        total = a2_ext - a1_ext
        if total < 0:
            total += 2 * np.pi
        diff = closest_angle - a1_ext
    else:
        total = a1_ext - a2_ext
        if total < 0:
            total += 2 * np.pi
        diff = a1_ext - closest_angle
    if diff < 0:
        diff += 2 * np.pi
    s_norm = diff / total if total > 0 else 0.0
    s = 2.0 * s_norm - 1.0

    closest_point = C + R_curv * np.array(
        [math.cos(closest_angle), math.sin(closest_angle)])
    gap = float(np.linalg.norm(closest_point - circle_center)
                - circle_radius - beam_radius)

    return dict(
        C=C, R=R_curv, kappa=kappa,
        angle_mid1=a1, angle_mid2=a2,
        angle_mid1_ext=a1_ext, angle_mid2_ext=a2_ext,
        closest_angle=closest_angle, closest_point=closest_point,
        s=s, gap=gap,
    )


def _arc_polyline(C, R, a_start, a_end, n=128):
    """Return (x, y) tracing the arc from a_start to a_end on the
    short side (sign-aware)."""
    # Sweep from a_start to a_end in the natural direction.
    theta = np.linspace(a_start, a_end, n)
    return C[0] + R * np.cos(theta), C[1] + R * np.sin(theta)


# ============================================================================
# PANEL DRAWING
# ============================================================================

def _draw_panel_a(ax):
    # Beam centreline + nodes (black, dashed).
    centre_line, = ax.plot(
        NODES[:, 0], NODES[:, 1],
        color="black", linestyle=":", linewidth=0.9,
        marker="o", markersize=5.0,
        markerfacecolor="black", markeredgecolor="black",
    )

    # Obstacle (red dashed circle).
    th = np.linspace(0, 2 * np.pi, 256)
    obst_line, = ax.plot(
        OBST_CENTER[0] + OBST_RADIUS * np.cos(th),
        OBST_CENTER[1] + OBST_RADIUS * np.sin(th),
        color=tuple(LAB_COLORS[2]),
        linestyle="--", linewidth=1.0,
    )
    ax.plot(OBST_CENTER[0], OBST_CENTER[1],
            marker="x", color=tuple(LAB_COLORS[2]), markersize=4.0,
            markeredgewidth=1.0)

    arc_records = []
    for k in range(3):
        p1, p2, p3 = NODES[k], NODES[k + 1], NODES[k + 2]
        info = extended_arc(p1, p2, p3,
                            OBST_CENTER, OBST_RADIUS, BEAM_RADIUS)
        color = ARC_COLORS[k]

        # Plot the EXTENDED arc as the primary curve.
        x_arc, y_arc = _arc_polyline(info["C"], info["R"],
                                      info["angle_mid1_ext"],
                                      info["angle_mid2_ext"])
        ax.plot(x_arc, y_arc, color=color, linewidth=1.4)

        # Closest-point marker.
        cp = info["closest_point"]
        ax.plot(cp[0], cp[1], marker="o", color=color, markersize=4.8,
                markerfacecolor=color, markeredgecolor="black",
                markeredgewidth=0.6, zorder=5)

        # Thin grey leader from the closest point to the obstacle centre
        # so the "distance" read off each (s, gap) entry is visualised.
        ax.plot([cp[0], OBST_CENTER[0]], [cp[1], OBST_CENTER[1]],
                color="0.55", linestyle=":", linewidth=0.6, zorder=2)

        # Boxed annotation with (s, gap), with an arrow pointing to the
        # closest-point marker.  Per-arc offsets so the three boxes don't
        # collide with the geometry or run off the right frame.
        offsets_pt = {
            0: (-55,  15),   # blue arc 1  -- left, lower (kept inside frame)
            1: (-40,  52),   # orange arc 2 -- pushed left
            2: ( 35,  38),   # green arc 3  -- compact right
        }
        ox, oy = offsets_pt[k]
        ax.annotate(
            rf"$s^\ast_{{{k+1}}} = {info['s']:+.2f}$"
            "\n"
            rf"$g_{{{k+1}}} = {info['gap']*1e3:+.2f}$ mm",
            xy=cp, xytext=(ox, oy), textcoords="offset points",
            color=color, fontsize=SMALL_FONTSIZE,
            ha="center", va="center",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.6,
                             shrinkA=0, shrinkB=2),
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                       ec=color, lw=0.6),
            zorder=6,
        )

        arc_records.append(info)

    setup_axes(ax)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")

    # Legend lists the curves only -- the (s, gap) values are shown next
    # to each closest-point marker.
    handles = [centre_line, obst_line]
    labels  = ["beam centreline", "obstacle"]
    for k in range(3):
        handles.append(
            Line2D([0], [0], color=ARC_COLORS[k], linewidth=1.4)
        )
        labels.append(f"arc {k+1}")
    leg = ax.legend(handles, labels,
                    loc="lower center", frameon=True, fontsize=SMALL_FONTSIZE,
                    handlelength=1.4)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.6)

    return arc_records


def _draw_panel_b(ax, arc_records):
    s = np.linspace(-1.0, 1.0, 401)
    w = compute_weights_from_s(s)

    ax.plot(s, w, color="black", linewidth=1.4)

    handles, labels = [], []
    for k, info in enumerate(arc_records):
        w_k = float(compute_weights_from_s(np.array([info["s"]])))
        color = ARC_COLORS[k]
        h, = ax.plot(info["s"], w_k,
                     marker="o", markersize=5.5,
                     markerfacecolor=color, markeredgecolor="black",
                     markeredgewidth=0.6,
                     linestyle="None", color=color)
        handles.append(h); labels.append(f"arc {k+1}")

    setup_axes(ax)
    ax.set_xlabel("arc parameter, $s$")
    ax.set_ylabel("weight, $w(s)$")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.05, 1.10)

    leg = ax.legend(handles, labels,
                    loc="lower center", frameon=True, fontsize=SMALL_FONTSIZE,
                    ncol=3, columnspacing=0.8, handlelength=1.2)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.6)


# ============================================================================
# COMPOSE
# ============================================================================

def plot_extended_arcs(out_dir: Path = OUT_DIR,
                        out_basename: str = OUT_BASENAME) -> None:
    apply_style()
    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(
        1, 2,
        left=0.075, right=0.985, top=0.95, bottom=0.16,
        wspace=0.32,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    arc_records = _draw_panel_a(ax_a)
    _draw_panel_b(ax_b, arc_records)

    dx, dy = PANEL_LABEL_OFFSET_PT
    for ax, lab in ((ax_a, "(a)"), (ax_b, "(b)")):
        ax.annotate(lab,
                    xy=(0, 1), xycoords="axes fraction",
                    xytext=(dx, dy), textcoords="offset points",
                    **PANEL_LABEL_KW)

    save_figure(fig, out_dir, out_basename)
    plt.close(fig)


if __name__ == "__main__":
    plot_extended_arcs()
