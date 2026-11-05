#!/usr/bin/env python3
"""
``fig_combined`` with the finite-difference stiffness inserted as a middle row:

  +-----------------------+-----------------------+
  | (a)  F vs delta       | (b)  ZOOM of (a)      |
  |      full range       |      delta/L in       |
  |                       |      [0.30, 0.40]     |
  +-----------------------+-----------------------+
  | (c)  dF/d(delta)      | (d)  ZOOM of (c)      |
  |      full range       |      same window      |
  +-----------------------+-----------------------+
  | (e)  jitter + active  | (f)  mesh convergence |
  |      nodes            |                       |
  +-----------------------+-----------------------+

Panels (e) and (f) are ``fig_combined``'s (c) and (d), pushed down one row.
The new middle row is where the jitter is impossible to miss: a few-percent
force wobble becomes a stiffness swing of several hundred percent.

Output: figures/fig_combined_withd.{pdf,svg}
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import (  # noqa: E402
    add_zoom_indicator,
    apply_style,
    panel_legend,
    save_figure,
)
from fig_combined import (  # noqa: E402
    PANEL_LABEL_KW,
    ZOOM_XLIM,
    _draw_force_displacement,
    _draw_jitter,
    _draw_mesh_convergence,
)
from fig_derivative_spectrum import (  # noqa: E402
    _analytical_derivative,
    _draw_derivative,
    _load_curves,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

OUT_DIR      = Path(__file__).parent / "figures"
OUT_BASENAME = "fig_combined_withd"

# Three rows of the same ~3.0 in panel height used by fig_combined.
FIG_SIZE     = (8.0, 10.2)


# ============================================================================
# COMPOSE
# ============================================================================

def plot_combined_withd(out_dir: Path = OUT_DIR,
                        out_basename: str = OUT_BASENAME) -> None:
    apply_style()

    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(
        3, 2,
        left=0.095, right=0.93, top=0.98, bottom=0.055,
        hspace=0.32, wspace=0.30,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, 0])
    ax_f = fig.add_subplot(gs[2, 1])

    # --- top row: force-displacement ----------------------------------------
    handles_a, labels_a = _draw_force_displacement(ax_a)
    panel_legend(ax_a, handles_a, labels_a, "lower right")

    handles_b, labels_b = _draw_force_displacement(ax_b, xlim=ZOOM_XLIM)
    panel_legend(ax_b, handles_b, labels_b, "lower left")
    add_zoom_indicator(fig, ax_a, ax_b, ZOOM_XLIM)

    # --- bottom row: finite-difference stiffness ----------------------------
    curves = _load_curves()
    analytical = _analytical_derivative()

    # Legend corners differ from the top row: the derivative fills the right
    # half of (c) and almost all of (d), so the boxes go where the traces
    # leave room -- low-delta in (c), bottom-right in (d).
    handles_c, labels_c = _draw_derivative(ax_c, curves, analytical=analytical)
    panel_legend(ax_c, handles_c, labels_c, "lower left")

    handles_d, labels_d = _draw_derivative(ax_d, curves, xlim=ZOOM_XLIM,
                                           analytical=analytical)
    panel_legend(ax_d, handles_d, labels_d, "lower right")
    add_zoom_indicator(fig, ax_c, ax_d, ZOOM_XLIM)

    # --- bottom row: fig_combined's former (c) and (d) ----------------------
    _draw_jitter(ax_e)

    handles_f, labels_f = _draw_mesh_convergence(ax_f)
    panel_legend(ax_f, handles_f, labels_f, "upper right")

    # Panel labels.
    for ax, lab in zip((ax_a, ax_b, ax_c, ax_d, ax_e, ax_f),
                       ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")):
        ax.text(0.03, 0.97, lab, transform=ax.transAxes, **PANEL_LABEL_KW)

    save_figure(fig, out_dir, out_basename)
    plt.close(fig)


if __name__ == "__main__":
    plot_combined_withd()
