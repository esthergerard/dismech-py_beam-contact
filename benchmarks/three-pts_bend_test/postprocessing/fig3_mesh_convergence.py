#!/usr/bin/env python3
"""
Figure 3 -- mesh convergence: peak normalised force vs N (number of beam nodes).

Two curves at L = 1.6 (matches MATLAB postprocessing), tabulated by hand from
prior runs of both contact models:

  - 3-node Eberly (regularised) summary block from Reg_Mesh_Conv_Post_Process.m
  - 2-node IMC                  summary block printed by the Python sim

Both lists are hardcoded below so the figure can be regenerated without
re-running the full N-sweep simulations.  Drop in new (N, peak) pairs as
they become available.

A horizontal reference line at 9.54/2 = 4.77 marks the analytical
asymptote (r/L = 0.125).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import (  # noqa: E402
    DEFAULT_FIG_SIZE,
    LAB_COLORS,
    apply_style,
    save_figure,
    setup_axes,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

# 3-node (Eberly / regularised) -- from Reg_Mesh_Conv_Post_Process.m summary
THREE_NODE = [
    (  5, 7.71560),
    ( 10, 5.74146),
    ( 20, 5.18449),
    ( 50, 4.92255),
    (100, 4.84225),
    (200, 4.80021),
    (300, 4.78339),
]

# 2-node (IMC) -- from the Python NPZ-sweep summary
TWO_NODE = [
    (  5, 8.00065),
    ( 10, 5.90813),
    ( 30, 5.12527),
    ( 50, 4.94322),
    (100, 4.88119),
    (200, 4.84927),
    (300, 4.83666),
]

REFERENCE_VALUE = 9.54 / 2.0
REFERENCE_LABEL = "Analytical $9.54/2$"

OUT_DIR = Path(__file__).parent / "figures"
OUT_BASENAME = "fig3_mesh_convergence"
INCLUDE_LABELS = True


# ============================================================================
# PLOT
# ============================================================================

def _unpack(pairs):
    N = np.asarray([p[0] for p in pairs], dtype=float)
    val = np.asarray([p[1] for p in pairs], dtype=float)
    order = np.argsort(N)
    return N[order], val[order]


def plot_mesh_convergence(
    two_node=TWO_NODE,
    three_node=THREE_NODE,
    out_dir: Path = OUT_DIR,
    out_basename: str = OUT_BASENAME,
    include_labels: bool = INCLUDE_LABELS,
) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)

    N2, val2 = _unpack(two_node)
    N3, val3 = _unpack(three_node)

    # Few data points -> use symbols connected by a thin line.
    line2, = ax.plot(
        N2, val2,
        linestyle="-", linewidth=1.0,
        marker="o", markersize=4.5,
        markerfacecolor="none",
        markeredgecolor=LAB_COLORS[0],
        markeredgewidth=1.0,
        color=LAB_COLORS[0],
    )
    line3, = ax.plot(
        N3, val3,
        linestyle="-", linewidth=1.0,
        marker="s", markersize=4.5,
        markerfacecolor="none",
        markeredgecolor=LAB_COLORS[1],
        markeredgewidth=1.0,
        color=LAB_COLORS[1],
    )
    ref_line = ax.axhline(REFERENCE_VALUE,
                          linestyle="--", linewidth=0.8,
                          color=LAB_COLORS[4])

    # Linear x; let matplotlib auto-tick (manual ticks at 5/10/20/30 collide).
    setup_axes(ax)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)

    if include_labels:
        ax.set_xlabel("Number of nodes, $N$")
        ax.set_ylabel(r"Force maximum, $\max\,F L^{2} / (E I)$")
        leg = ax.legend([line2, line3, ref_line],
                        ["2-nodes", "3-nodes", REFERENCE_LABEL],
                        loc="upper right", frameon=True)
        leg.get_frame().set_edgecolor("black")
        leg.get_frame().set_linewidth(0.6)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    fig.tight_layout()
    save_figure(fig, out_dir, out_basename)
    plt.close(fig)


if __name__ == "__main__":
    plot_mesh_convergence()
