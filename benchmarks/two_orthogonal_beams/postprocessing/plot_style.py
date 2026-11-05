"""
Shared style for paper-ready figures.

Lab requirements (summarised):
  - figure width ~ 3.5 in (single column) or 3-4 in plots inside a multi-panel figure
  - font size 8 or 10 (do NOT use more than two distinct sizes)
  - font family CMR10 for text, CMMI10 for math (Computer Modern)
  - axes / legends / labels are placeholders -- final formatting happens in
    Inkscape, so the script saves both PDF and SVG for editability
  - omit legend when only one curve is on the plot
  - simulation data -> lines or symbols (lines when many points)
  - experimental data -> symbols only, with std-dev error bars

Implementation notes
  - We default to mathtext + ``font.family = serif`` with the Computer Modern
    fontset.  This gives the CMR look without requiring a system LaTeX
    install.  Flip ``USE_TEX = True`` if you have LaTeX and want true CMR10.
  - The 5-colour palette mirrors the lab sample plot.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------------------------
# Configuration knobs
# ----------------------------------------------------------------------------

USE_TEX = False         # set True if LaTeX (cmr10 / cmmi10) is installed
FONT_SIZE = 10          # 8 or 10 per lab spec; pick one and keep it consistent
COLUMN_WIDTH_IN = 3.5   # single-column figure width
DEFAULT_FIG_SIZE = (3.5, 2.6)  # ~3.5 x 2.6 in -> tidy single-column plot


# Lab colour palette (RGB / 255) -- order matches the sample script.
LAB_COLORS = (np.array([
    [247, 148,  30],   # orange
    [  0, 166,  81],   # green
    [237,  28,  36],   # red
    [  0, 174, 239],   # cyan/blue
    [  0,   0,   0],   # black
], dtype=float) / 255.0)


def apply_style(*, use_tex: bool = USE_TEX, font_size: int = FONT_SIZE) -> None:
    """Set rcParams once.  Call this at the top of every figure script."""
    if use_tex:
        mpl.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["cmr10", "Computer Modern Roman"],
            # Math is rendered via LaTeX, but keep cm fontset as a fallback.
            "mathtext.fontset": "cm",
        })
    else:
        mpl.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": [
                "cmr10",
                "Computer Modern Roman",
                "STIXGeneral",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "cm",
            # cmr10 + Unicode minus print a warning; use the basic hyphen.
            "axes.unicode_minus": False,
        })

    mpl.rcParams.update({
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # cmr10 + tick labels need this to silence the warning.
        "axes.formatter.use_mathtext": True,
    })


def setup_axes(ax, *, box_aspect: float | None = None) -> None:
    """Apply the standard 'box on' / inward ticks look to a single axes."""
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
    if box_aspect is not None:
        ax.set_box_aspect(box_aspect)


# Grey used for the zoom outline / leader lines so they never compete with
# the data curves.
ZOOM_EDGE    = "0.30"
ZOOM_BOX_KW  = dict(facecolor="none", edgecolor=ZOOM_EDGE,
                    linewidth=0.7, zorder=5)
ZOOM_LINK_KW = dict(color=ZOOM_EDGE, linewidth=0.6,
                    linestyle=(0, (3, 3)), zorder=4)


def panel_legend(ax, handles, labels, loc, *, fontsize: int = 8):
    """Legend with the black frame the lab style requires."""
    leg = ax.legend(handles, labels, loc=loc, frameon=True, fontsize=fontsize)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.6)
    return leg


def add_zoom_indicator(fig, ax_src, ax_zoom, xlim, *, tag: str = "ZOOM"):
    """
    Outline on ``ax_src`` the window that ``ax_zoom`` magnifies, link the two
    with dashed leader lines, and tag the zoom panel.

    ``ax_zoom`` must already be drawn: its current y-limits set the box height.
    """
    x0, x1 = xlim
    y0, y1 = ax_zoom.get_ylim()
    ax_src.add_patch(
        mpl.patches.Rectangle((x0, y0), x1 - x0, y1 - y0, **ZOOM_BOX_KW))

    # Make sure the outline is fully inside the source axes -- a clipped box
    # reads as a plotting bug rather than as a zoom marker.
    s0, s1 = ax_src.get_ylim()
    ax_src.set_ylim(min(s0, y0 - 0.02 * (s1 - s0)),
                    max(s1, y1 + 0.02 * (s1 - s0)))

    # Leader lines: right-hand corners of the box -> left corners of the zoom.
    for src_corner, dst_corner in (((x1, y1), (0.0, 1.0)),
                                   ((x1, y0), (0.0, 0.0))):
        fig.add_artist(mpl.patches.ConnectionPatch(
            xyA=src_corner, coordsA=ax_src.transData,
            xyB=dst_corner, coordsB=ax_zoom.transAxes,
            **ZOOM_LINK_KW))

    if tag:
        ax_zoom.text(0.97, 0.97, tag, transform=ax_zoom.transAxes,
                     fontsize=9, fontweight="bold", color=ZOOM_EDGE,
                     va="top", ha="right")


def save_figure(fig, out_dir: str | Path, basename: str) -> None:
    """Save as PDF and SVG into ``out_dir/<basename>.{pdf,svg}``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{basename}.pdf"
    svg_path = out_dir / f"{basename}.svg"
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(svg_path, format="svg")
    print(f"  wrote {pdf_path}")
    print(f"  wrote {svg_path}")


# ----------------------------------------------------------------------------
# Common parsers for the simulation text files
# ----------------------------------------------------------------------------

def load_coords_file(path: str | Path) -> np.ndarray:
    """
    Load a reg_beam_coords_*.txt file.

    Returns
    -------
    arr : (n_rows, 5) array with columns [time, node_id, x, y, z].
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.loadtxt(path, comments="#")


def load_forces_file(path: str | Path) -> np.ndarray:
    """
    Load a reg_contact_forces_*.txt file.

    Returns
    -------
    arr : (n_rows, 5) array with columns [time, node_id, fx, fy, fz].
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.loadtxt(path, comments="#")


def load_npz_run(path: str | Path):
    """
    Load a Python-style .npz file produced by the contact-bending sims
    (impV_noD_contact_bending_N*.npz / contact_bending_N*.npz).

    Returns a dict with:
      - time              : (n_steps,) seconds
      - beam_idx          : (n_beam,)  beam-A node indices into the
                            (n_steps, n_total, 3) coord / force arrays
      - beam_coords       : (n_steps, n_beam, 3)
      - beam_forces       : (n_steps, n_beam, 3)
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=False)
    t = data["time"]
    coords_all = data["node_coordinates"]
    forces_all = data["contact_forces"]
    beam_idx = data["beam_node_indices"]
    return {
        "time": t,
        "beam_idx": beam_idx,
        "beam_coords": coords_all[:, beam_idx, :],
        "beam_forces": forces_all[:, beam_idx, :],
    }


def pivot_per_time(arr: np.ndarray, value_col: int, n_nodes: int):
    """
    Reshape a long-format array (time, node_id, value...) into
    (times, n_nodes) plus the sorted unique time vector.

    Rows with duplicate (time, node_id) keep the first occurrence.
    """
    t_all = arr[:, 0]
    n_all = arr[:, 1].astype(int)
    v_all = arr[:, value_col]

    # First-occurrence deduplication on (time, node_id)
    _, uniq_rows = np.unique(np.stack([t_all, n_all.astype(float)], axis=1),
                             axis=0, return_index=True)
    uniq_rows.sort()
    t_all, n_all, v_all = t_all[uniq_rows], n_all[uniq_rows], v_all[uniq_rows]

    unique_times, inverse = np.unique(t_all, return_inverse=True)
    grid = np.full((unique_times.size, n_nodes), np.nan)
    valid = (n_all >= 0) & (n_all < n_nodes)
    grid[inverse[valid], n_all[valid]] = v_all[valid]
    return unique_times, grid
