#!/usr/bin/env python3
"""
Multi-panel summary figure:

  +-----------------------+-----------------------+
  | (a)  fig 2  F vs δ    | (b)  fig 2b  zoom     |
  |      full range       |      δ/L ∈ [0.3, 0.4] |
  +-----------------------+-----------------------+
  | (c)  fig 4  jitter +  | (d)  fig 3  mesh      |
  |      active nodes     |      convergence      |
  +-----------------------+-----------------------+

Output: figures/fig_combined.{pdf,svg}
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import (  # noqa: E402
    LAB_COLORS,
    add_zoom_indicator,
    apply_style,
    load_npz_run,
    panel_legend,
    save_figure,
    setup_axes,
)
from fig2_force_displacement import (  # noqa: E402
    ANALYTICAL_CSV,
    ANALYTICAL_FACTOR,
    ANALYTICAL_STYLE,
    SIM_RUNS,
    extract_from_npz,
    extract_from_txt,
    read_analytical_csv,
)
from fig3_mesh_convergence import (  # noqa: E402
    REFERENCE_LABEL,
    REFERENCE_VALUE,
    THREE_NODE,
    TWO_NODE,
    _unpack,
)
from fig4_jitter_correlation import (  # noqa: E402
    ACTIVE_THRESHOLD,
    NODE_INDEX_LIMITS,
    NPZ_FILE,
    TIME_WINDOW,
    _active_segments,
    _slice_window,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

OUT_DIR      = Path(__file__).parent / "figures"
OUT_BASENAME = "fig_combined"

# Sized so each panel is ~3.8 x 3.0 inches with room for axis labels.
FIG_SIZE     = (8.0, 6.8)

PANEL_LABEL_KW = dict(
    fontsize=11, fontweight="bold",
    va="top", ha="left",
)

FORCE_COLOR    = tuple(LAB_COLORS[3])
SEGMENT_COLOR  = tuple(LAB_COLORS[0])
FORCE_LW       = 1.0
SEGMENT_LW     = 1.0

# Window that panel (b) magnifies out of panel (a).
ZOOM_XLIM      = (0.30, 0.40)


# ============================================================================
# PER-PANEL DRAW HELPERS
# ============================================================================

def _draw_force_displacement(ax, *, xlim: tuple | None = None):
    """Panel (a) or (b)."""
    handles, labels = [], []
    y_in_view = []

    # Analytical first (back).
    if Path(ANALYTICAL_CSV).is_file():
        eta, fL2 = read_analytical_csv(ANALYTICAL_CSV)
        if eta.size:
            line, = ax.plot(eta, fL2 * ANALYTICAL_FACTOR, **ANALYTICAL_STYLE)
            handles.append(line); labels.append("Analytical")
            if xlim is not None:
                m = (eta >= xlim[0]) & (eta <= xlim[1])
                if m.any():
                    y_in_view.append(fL2[m] * ANALYTICAL_FACTOR)

    # Sim runs in declared order (1-node first → back, 3-nodes last → top).
    for run in SIM_RUNS:
        try:
            if run["kind"] == "txt":
                dL, FL2 = extract_from_txt(Path(run["coords"]),
                                            Path(run["forces"]))
            else:
                dL, FL2 = extract_from_npz(Path(run["path"]))
        except FileNotFoundError:
            continue
        if dL.size == 0:
            continue
        dl_max = run.get("delta_l_max")
        if dl_max is not None:
            keep = dL <= dl_max
            dL, FL2 = dL[keep], FL2[keep]
        line, = ax.plot(dL, FL2, **run["style"])
        handles.append(line); labels.append(run["label"])
        if xlim is not None:
            m = (dL >= xlim[0]) & (dL <= xlim[1])
            if m.any():
                y_in_view.append(FL2[m])

    setup_axes(ax)
    if xlim is not None:
        ax.set_xlim(*xlim)
        if y_in_view:
            yc = np.concatenate(y_in_view)
            lo, hi = float(yc.min()), float(yc.max())
            pad = 0.05 * (hi - lo) if hi > lo else 0.05 * abs(hi)
            ax.set_ylim(lo - pad, hi + pad)

    ax.set_xlabel(r"Normalized displacement, $\delta / L$")
    ax.set_ylabel(r"Normalized force, $F L^{2} / (E I)$")
    return handles, labels


def _draw_jitter(ax_left):
    """Panel (c).  Returns (handles, labels) -- empty for this panel."""
    ax_right = ax_left.twinx()

    run = load_npz_run(NPZ_FILE)
    t, beam_forces = _slice_window(run, TIME_WINDOW)
    F_total_mag = np.linalg.norm(beam_forces.sum(axis=1), axis=1)
    F_mag = np.linalg.norm(beam_forces, axis=2)

    # Left axis: total contact-force magnitude.
    ax_left.plot(t, F_total_mag, color=FORCE_COLOR, linewidth=FORCE_LW)
    setup_axes(ax_left)
    ax_left.set_xlim(t[0], t[-1])
    pad = 0.05 * (F_total_mag.max() - F_total_mag.min())
    ax_left.set_ylim(F_total_mag.min() - pad, F_total_mag.max() + pad)

    # Right axis: per-node active segments.
    for i in range(F_mag.shape[1]):
        active = F_mag[:, i] > ACTIVE_THRESHOLD
        if not np.any(active):
            continue
        for ts, te in _active_segments(t, active):
            ax_right.hlines(i, ts, te,
                            colors=SEGMENT_COLOR, linewidth=SEGMENT_LW)

    ax_right.set_ylim(*NODE_INDEX_LIMITS)
    ax_right.spines["top"].set_visible(True)
    ax_right.spines["right"].set_visible(True)

    # Colour-coordinate.
    ax_left.tick_params(axis="y", direction="in", colors=FORCE_COLOR)
    ax_left.spines["left"].set_color(FORCE_COLOR)
    ax_left.yaxis.label.set_color(FORCE_COLOR)
    ax_right.spines["right"].set_color(SEGMENT_COLOR)
    ax_right.tick_params(axis="y", direction="in", colors=SEGMENT_COLOR)
    ax_right.yaxis.label.set_color(SEGMENT_COLOR)
    ax_left.spines["right"].set_color(SEGMENT_COLOR)

    ax_left.set_xlabel("time, $t$ [s]")
    ax_left.set_ylabel(
        r"Contact force magnitude, $\|\sum_i F_i\|$, [N]")
    ax_right.set_ylabel("Active beam nodes index")
    # Shift the rotated ylabels down a touch so the long math doesn't
    # overhang the top of the axes.
    ax_left.yaxis.set_label_coords(-0.16, 0.45)
    return [], []


def _draw_mesh_convergence(ax):
    """Panel (d)."""
    N2, val2 = _unpack(TWO_NODE)
    N3, val3 = _unpack(THREE_NODE)

    line2, = ax.plot(
        N2, val2,
        linestyle="-", linewidth=1.0,
        marker="o", markersize=4.5,
        markerfacecolor="none",
        markeredgecolor=LAB_COLORS[0], markeredgewidth=1.0,
        color=LAB_COLORS[0],
    )
    line3, = ax.plot(
        N3, val3,
        linestyle="-", linewidth=1.0,
        marker="s", markersize=4.5,
        markerfacecolor="none",
        markeredgecolor=LAB_COLORS[1], markeredgewidth=1.0,
        color=LAB_COLORS[1],
    )
    ref_line = ax.axhline(REFERENCE_VALUE,
                          linestyle="--", linewidth=0.8,
                          color=LAB_COLORS[4])

    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    setup_axes(ax)

    # Pin the y-axis to start at 0 so the convergence asymptote can be read
    # against the absolute force scale.
    y_top = max(val2.max(), val3.max(), REFERENCE_VALUE)
    ax.set_ylim(0.0, y_top * 1.08)

    ax.set_xlabel("Number of nodes, $N$")
    ax.set_ylabel(r"Force maximum, $\max\,F L^{2} / (E I)$")
    return ([line2, line3, ref_line],
            ["2-node", "3-node (ours)", REFERENCE_LABEL])


# ============================================================================
# COMPOSE
# ============================================================================

def plot_combined(out_dir: Path = OUT_DIR,
                  out_basename: str = OUT_BASENAME) -> None:
    apply_style()

    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(
        2, 2,
        left=0.085, right=0.93, top=0.97, bottom=0.08,
        hspace=0.32, wspace=0.30,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # (a) full range force-displacement
    handles_a, labels_a = _draw_force_displacement(ax_a)
    panel_legend(ax_a, handles_a, labels_a, "lower right")

    # (b) zoom on ZOOM_XLIM -- same legend as (a), plus a ZOOM tag and the
    # matching outline drawn back onto (a).
    handles_b, labels_b = _draw_force_displacement(ax_b, xlim=ZOOM_XLIM)
    panel_legend(ax_b, handles_b, labels_b, "lower left")
    add_zoom_indicator(fig, ax_a, ax_b, ZOOM_XLIM)

    # (c) jitter
    _draw_jitter(ax_c)

    # (d) mesh convergence
    handles_d, labels_d = _draw_mesh_convergence(ax_d)
    panel_legend(ax_d, handles_d, labels_d, "upper right")

    # Panel labels.
    for ax, lab in zip((ax_a, ax_b, ax_c, ax_d), ("(a)", "(b)", "(c)", "(d)")):
        ax.text(0.03, 0.97, lab, transform=ax.transAxes, **PANEL_LABEL_KW)

    save_figure(fig, out_dir, out_basename)
    plt.close(fig)


if __name__ == "__main__":
    plot_combined()
