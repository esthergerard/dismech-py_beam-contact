#!/usr/bin/env python3
"""
Figure 1 -- beam geometry over time with the rigid circular obstacle.

Layout
------
Plots N snapshots of the beam centerline in the (x, y) plane.  Beam nodes are
drawn as filled circles connected by thin lines.  Each snapshot is coloured
by simulation time (turbo colormap, dark = early / bright = late).  The
rigid circle obstacle is drawn as a black outline.

Configuration block at the top -- edit the file paths / circle parameters
to match the simulation you want to visualise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import (  # noqa: E402
    DEFAULT_FIG_SIZE,
    apply_style,
    load_coords_file,
    save_figure,
    setup_axes,
)


# ============================================================================
# CONFIGURATION  -- edit these to match the run you want to visualise.
# ============================================================================

# Coords file: rows = time, node_id, x, y, z
COORDS_FILE = Path(__file__).resolve().parents[1] \
    / "external_reference_data" / "reg_beam_coords_N50.txt"

# Rigid circle obstacle drawn on top of the beam.
# NOTE: defaults match the contact_bending_test_3D.py geometry.  Change if
# this data set used a different obstacle position / radius.
CIRCLE_CENTER = (0.8, 0.0)   # (x, y) in metres
CIRCLE_RADIUS = 0.2          # m

# Number of beam snapshots to overlay.  'all' to draw every saved frame.
N_FRAMES = 12

# Output directory + basename (the script writes <basename>.pdf and .svg).
OUT_DIR = Path(__file__).parent / "figures"
OUT_BASENAME = "fig1_geometry_evolution"

# --- Companion 1-node bead variant (fig 1b) --------------------------------
COORDS_FILE_BEAD = Path(__file__).resolve().parents[1] \
    / "contact_output_files_3D_bead" / "reg_beam_coords_N50_bead.txt"
OUT_BASENAME_BEAD = "fig1b_geometry_evolution_1node"

# Final-publication switch: when True, axis labels and tick labels are kept
# (useful while iterating); when False they are stripped so labels can be
# added back in Inkscape on top of the saved SVG.
INCLUDE_LABELS = True


# ============================================================================
# PLOTTING
# ============================================================================

def plot_geometry_evolution(
    coords_file: Path = COORDS_FILE,
    circle_center: tuple = CIRCLE_CENTER,
    circle_radius: float = CIRCLE_RADIUS,
    n_frames: int | str = N_FRAMES,
    out_dir: Path = OUT_DIR,
    out_basename: str = OUT_BASENAME,
    include_labels: bool = INCLUDE_LABELS,
) -> None:
    apply_style()

    coords = load_coords_file(coords_file)
    times_all = np.unique(coords[:, 0])

    if n_frames == "all" or (isinstance(n_frames, int) and n_frames >= times_all.size):
        sample_times = times_all
    else:
        idx = np.round(np.linspace(0, times_all.size - 1, int(n_frames))).astype(int)
        sample_times = times_all[idx]

    cmap = mpl.colormaps["turbo"]
    colors = cmap(np.linspace(0.05, 0.95, sample_times.size))

    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)

    # ---- beam snapshots ----------------------------------------------------
    for color, t in zip(colors, sample_times):
        mask = coords[:, 0] == t
        snap = coords[mask]
        snap = snap[np.argsort(snap[:, 1])]  # sort by node_id
        ax.plot(
            snap[:, 2], snap[:, 3],
            linestyle="-",
            linewidth=0.6,
            marker="o",
            markersize=2.0,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.0,
            color=color,
        )

    # ---- rigid circle obstacle --------------------------------------------
    theta = np.linspace(0.0, 2.0 * np.pi, 256)
    cx, cy = circle_center
    circle_line, = ax.plot(
        cx + circle_radius * np.cos(theta),
        cy + circle_radius * np.sin(theta),
        color="black",
        linewidth=1.0,
    )

    # ---- legend proxies ---------------------------------------------------
    # Pick a colour from the middle of the time colormap for the "flexible
    # beam" legend entry so the line + markers visually match what is shown
    # in the plot (with the actual line getting its colour from the time
    # colorbar).
    from matplotlib.lines import Line2D
    mid_color = cmap(0.5)
    beam_proxy = Line2D(
        [0], [0],
        linestyle="-", linewidth=0.6,
        marker="o", markersize=3.0,
        markerfacecolor=mid_color,
        markeredgecolor=mid_color,
        color=mid_color,
    )

    ax.set_aspect("equal", adjustable="datalim")
    setup_axes(ax)

    # ---- time colour bar ---------------------------------------------------
    norm = mpl.colors.Normalize(vmin=float(sample_times[0]),
                                vmax=float(sample_times[-1]))
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.outline.set_linewidth(0.6)
    if include_labels:
        cbar.set_label("$t$ [s]")

    if include_labels:
        ax.set_xlabel("$x$ [m]")
        ax.set_ylabel("$y$ [m]")
        leg = ax.legend(
            [beam_proxy, circle_line],
            ["flexible beam", "rigid fixed circle"],
            loc="upper right", frameon=True, fontsize=8,
        )
        leg.get_frame().set_edgecolor("black")
        leg.get_frame().set_linewidth(0.6)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        cbar.set_ticks([])

    fig.tight_layout()
    save_figure(fig, out_dir, out_basename)
    plt.close(fig)


if __name__ == "__main__":
    plot_geometry_evolution()
    # fig 1b: same plot, but using the 1-node bead output file.
    plot_geometry_evolution(
        coords_file=COORDS_FILE_BEAD,
        out_basename=OUT_BASENAME_BEAD,
    )
