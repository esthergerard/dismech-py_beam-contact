#!/usr/bin/env python3
"""
Soft-soft orthogonal-beams test  (Zavarise & Wriggers 2000, Example 2):
unified, paper-ready post-processing of the existing output files.

Layout (outer GridSpec 3 × 2; bottom row spans both cols):

  +----------------------+----------------------+
  | (a) top view (x-z)   | (b) side view of     |
  |     initial config   |     Beam Z centreline|
  |                      |     over 6 steps     |
  +----------------------+----------------------+
  | (c) F_N vs step,     | (d) F_N (top) and    |
  |     mu = 0,          |     F_T (bottom) vs  |
  |     multiple N_X     |     step, mu = 0.1,  |
  |                      |     multiple N_X     |
  +----------------------+----------------------+
  | (e) per-dt force trace IMC vs Eberly,       |
  |     mu = 0.1, N_X = 21                      |
  |     three stacked sub-panels: F_N, F_T,     |
  |     F_T / F_N + Coulomb limit               |
  +---------------------------------------------+

Reference PNGs reproduced (in style and substance):
  contact_output_files_soft_soft/
    zavarise_mesh_convergence_steps6.png            -> panel (c)
    zavarise_friction_mesh_convergence_mu01.png     -> panel (d)
    zavarise_per_dt_trace_eberly_vs_imc_mu01.png    -> panel (e)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import (  # noqa: E402
    LAB_COLORS,
    apply_style,
    save_figure,
    setup_axes,
)


# ============================================================================
# GEOMETRY PARAMETERS (must match contact_bending_soft_soft.py)
# ============================================================================

X_BEAM_LEN   = 10.0
Z_BEAM_LEN   = 14.0
BEAM_RADIUS  = 0.2
Y_X          = +0.5
Z_X_CONTACT  = +3.0


# ============================================================================
# CONFIGURATION
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR  = Path(__file__).resolve().parents[1] / "contact_output_files_soft_soft"

# Coords for the geometry panels (mu=0.1, n_x=21).
COORDS_FILE = DATA_DIR / "zavarise_coords_steps6_nx21_mu01.txt"

# Mesh convergence (mu = 0): one curve per N_X, plotted vs load step.
MESH_MU0 = [
    (11,  DATA_DIR / "zavarise_compare_steps6_nx11.txt"),
    (21,  DATA_DIR / "zavarise_compare_steps6_nx21.txt"),
    (51,  DATA_DIR / "zavarise_compare_steps6_nx51.txt"),
    (101, DATA_DIR / "zavarise_compare_steps6_nx101.txt"),
]

# Mesh convergence (mu = 0.1): two stacked sub-panels (F_N, F_T).
MESH_MU01 = [
    (21,  DATA_DIR / "zavarise_compare_steps6_nx21_mu01.txt"),
    (51,  DATA_DIR / "zavarise_compare_steps6_nx51_mu01.txt"),
    (101, DATA_DIR / "zavarise_compare_steps6_nx101_mu01.txt"),
]

# Per-dt force traces (IMC edge-edge "per-pair fix" vs Eberly+smooth).
TRACE_IMC    = DATA_DIR / "zavarise_per_dt_trace_steps6_trace_imc_fixed_mu01.txt"
TRACE_EBERLY = DATA_DIR / "zavarise_per_dt_trace_steps6_trace_eberly_mu01.txt"

MU_COULOMB   = 0.1

# Lab spec: at most two font sizes per figure.  We use 10 pt for axes
# titles / labels / panel tags and 8 pt for legends / colorbars / small
# annotations.  Anything else should be flagged.
LABEL_FONTSIZE   = 10
SMALL_FONTSIZE   = 8

PANEL_LABEL_KW = dict(fontsize=LABEL_FONTSIZE, fontweight="bold",
                       va="top", ha="left")

# Single-figure sizes (now split: geometry and metrics live in their own
# figures).  Geometry is ~ half page wide, metrics is full page wide.
FIG_SIZE_GEOM    = (7.0, 3.2)
FIG_SIZE_METRICS = (8.0, 6.8)

# Absolute (in points) offsets from each axes' top-left corner.  The panel
# label and the legend share the same horizontal offset, so the *gap*
# between them is the (LEGEND_Y - PANEL_LABEL_Y) constant across all
# panels regardless of their axes size.
PANEL_LABEL_OFFSET_PT  = (5.0,  -5.0)    # (right, down)
LEGEND_OFFSET_PT       = (5.0, -19.0)    # (right, down) -- smaller vertical gap

OUT_DIR             = Path(__file__).parent / "figures"
OUT_BASENAME_GEOM   = "fig_soft_soft_geometry"
OUT_BASENAME_METRIC = "fig_soft_soft_metrics"


# ============================================================================
# READERS
# ============================================================================

def _n_x_from_total(n_total: int) -> int:
    ratio = Z_BEAM_LEN / X_BEAM_LEN
    return int(round((n_total - 1 + ratio) / (1 + ratio)))


def load_coords_per_step(path: Path):
    snapshots = {}
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            try:
                step = int(parts[1])
                nid  = int(parts[2])
                xyz  = (float(parts[3]), float(parts[4]), float(parts[5]))
            except ValueError:
                continue
            snapshots.setdefault(step, {})[nid] = xyz
    out = {}
    for step, nodes in snapshots.items():
        n_total = max(nodes) + 1
        arr = np.zeros((n_total, 3))
        for nid, xyz in nodes.items():
            arr[nid] = xyz
        out[step] = arr
    return out


def split_beams(coords: np.ndarray):
    n_x = _n_x_from_total(coords.shape[0])
    return coords[:n_x], coords[n_x:]


def load_compare(path: Path):
    """
    Load a `zavarise_compare_steps*.txt` file.

    Returns
    -------
    steps    : (n,) int
    FN_sim   : (n,) float   (column 2 in the file; index 1 in 0-based)
    FN_paper : (n,) float   (column 3; index 2)
    FT_sim   : (n,) float or None.  None if file has the mu = 0 layout
                                     (no friction columns).
    FT_paper : (n,) float or None.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                step = int(parts[0])
                vals = [float(x) if x.lower() != "nan" else np.nan
                        for x in parts[1:]]
            except ValueError:
                continue
            rows.append((step, vals))
    if not rows:
        raise RuntimeError(f"no rows in {path}")
    rows.sort(key=lambda r: r[0])
    steps  = np.array([r[0] for r in rows], dtype=int)
    FN_sim = np.array([r[1][0] for r in rows])
    FN_pap = np.array([r[1][1] for r in rows])
    # mu = 0 file has 7 numeric cols after step (8 incl step); the mu = 0.1
    # file has at least 8 (FN_sim, FN_paper, FN_ratio, FT_sim, FT_paper, ...).
    if len(rows[0][1]) >= 5:
        FT_sim = np.array([r[1][3] for r in rows])
        FT_pap = np.array([r[1][4] for r in rows])
    else:
        FT_sim = None
        FT_pap = None
    return steps, FN_sim, FN_pap, FT_sim, FT_pap


def load_per_dt_trace(path: Path):
    """
    Load a per-dt-step contact-force trace.

    Columns: t  Fx_contact  Fy_contact  Fz_contact  Fx_fric  Fy_fric  Fz_fric

    Returns t, F_N (vector magnitude of contact), F_T (vector magnitude of
    friction).
    """
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.loadtxt(path, comments="#")
    if arr.ndim == 1:
        arr = arr[None, :]
    t = arr[:, 0]
    F_contact = arr[:, 1:4]
    F_fric    = arr[:, 4:7]
    FN = np.linalg.norm(F_contact, axis=1)
    FT = np.linalg.norm(F_fric,    axis=1)
    return t, FN, FT


# ============================================================================
# PER-PANEL DRAWING
# ============================================================================

def _legend_transform(ax):
    """Transform that anchors at the axes top-left + LEGEND_OFFSET_PT."""
    fig = ax.figure
    dx, dy = LEGEND_OFFSET_PT
    return ax.transAxes + ScaledTranslation(
        dx / 72.0, dy / 72.0, fig.dpi_scale_trans)


def _legend_kwargs(ax, **extra):
    """Common kwargs to put a legend at a fixed absolute position below
    the panel-(c)/(d)/(e) tag."""
    return dict(
        loc="upper left",
        bbox_to_anchor=(0, 1),
        bbox_transform=_legend_transform(ax),
        borderaxespad=0.0,
        frameon=True,
        **extra,
    )



def _draw_top_view(ax):
    """(a) top view (x-z) of the initial configuration."""
    # Match the per-beam colour scheme used in panel (b): grey for the
    # rigid Beam X, turbo at the FIRST step value for Beam Z (= same as
    # the colour used for step 1 in panel (b), so panel (a) and the
    # earliest configuration in (b) share a consistent Beam Z colour).
    beam_x_color = "0.45"
    beam_z_color = mpl.colormaps["turbo"](0.05)

    snap = load_coords_per_step(COORDS_FILE)
    coords0 = snap[min(snap.keys())]
    X, Z = split_beams(coords0)

    ax.plot(X[:, 0], X[:, 2],
            linestyle="-", linewidth=1.0,
            marker="o", markersize=3.0,
            color=beam_x_color)
    ax.plot(Z[:, 0], Z[:, 2],
            linestyle="-", linewidth=1.0,
            marker="s", markersize=3.0,
            color=beam_z_color)

    ax.scatter([0.0], [Z_X_CONTACT],
               facecolor="none", edgecolor=LAB_COLORS[2],
               s=70, marker="o", linewidth=1.4, zorder=5)
    ax.scatter([X[0, 0], X[-1, 0]], [X[0, 2], X[-1, 2]],
               color=LAB_COLORS[4], s=35, marker="s", zorder=6)
    ax.scatter([Z[0, 0]], [Z[0, 2]],
               color=LAB_COLORS[4], s=35, marker="s", zorder=6)

    setup_axes(ax)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$z$ [m]")

    handles = [
        Line2D([0], [0], color=beam_x_color,
               marker="o", markersize=4.0, linewidth=1.0,
               label="Beam X"),
        Line2D([0], [0], color=beam_z_color,
               marker="s", markersize=4.0, linewidth=1.0,
               label="Beam Z"),
        Line2D([0], [0], color=LAB_COLORS[4],
               marker="s", linestyle="None", markersize=5.5,
               label="clamp"),
        Line2D([0], [0], color=LAB_COLORS[2],
               marker="o", linestyle="None", markersize=6.5,
               markerfacecolor="none", markeredgewidth=1.2,
               label="contact"),
    ]
    leg = ax.legend(handles=handles, loc="lower right",
                    frameon=True, fontsize=8)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.6)


def _draw_side_view(ax):
    """(b) Beam Z + Beam X centreline nodes (z-y) over the 6 loaded steps."""
    snap = load_coords_per_step(COORDS_FILE)
    steps = [s for s in sorted(snap.keys()) if s >= 1]

    cmap = mpl.colormaps["turbo"]
    colors = cmap(np.linspace(0.05, 0.95, len(steps)))
    beam_x_color = "0.45"                 # medium grey

    for color, step in zip(colors, steps):
        X, Z = split_beams(snap[step])
        # Beam Z deformed centreline (nodes + lines), time-coloured.
        ax.plot(Z[:, 2], Z[:, 1],
                linestyle="-", linewidth=0.8,
                marker="s", markersize=2.4,
                markeredgewidth=0.0, color=color)
        # Beam X centreline nodes, single fixed colour for all steps so the
        # contact partner is clearly distinguishable from Beam Z's evolving
        # deformation.
        ax.plot(X[:, 2], X[:, 1],
                linestyle="None",
                marker="o", markersize=2.4,
                markeredgewidth=0.0,
                color=beam_x_color)

    setup_axes(ax)
    ax.set_xlabel("$z$ [m]")
    ax.set_ylabel("$y$ [m]")

    # Legend identifying the two beams (Beam X listed first).  Beam Z's
    # legend swatch uses the colormap value at the FIRST plotted step so
    # the (a) and (b) Beam Z colours match.
    handles = [
        Line2D([0], [0], color=beam_x_color,
               marker="o", markersize=4.0, linestyle="None",
               label="Beam X"),
        Line2D([0], [0], color=colors[0],
               marker="s", markersize=4.0, linewidth=0.8,
               label="Beam Z"),
    ]
    leg = ax.legend(handles=handles,
                    loc="lower right", frameon=True, fontsize=8)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.6)

    norm = mpl.colors.Normalize(vmin=float(steps[0]),
                                vmax=float(steps[-1]))
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, fraction=0.045, pad=0.025)
    cbar.outline.set_linewidth(0.6)
    cbar.set_label("step", fontsize=8)
    cbar.ax.tick_params(labelsize=8)


def _mesh_colormap(n_curves: int):
    """Discrete viridis-like palette ordered from N_X small -> large."""
    cmap = mpl.colormaps["viridis"]
    return cmap(np.linspace(0.0, 0.9, n_curves))


def _draw_mesh_FN_per_step(ax, mesh_list, paper_curve_label):
    """(c) F_N vs load step -- one curve per N_X, mu=0 layout."""
    colors = _mesh_colormap(len(mesh_list))
    paper_steps = None
    paper_FN    = None
    handles, labels = [], []

    for color, (n_x, path) in zip(colors, mesh_list):
        try:
            steps, FN_sim, FN_pap, _, _ = load_compare(path)
        except FileNotFoundError:
            print(f"  [warn] missing {path.name}")
            continue
        h, = ax.plot(steps, FN_sim,
                     linestyle="-", linewidth=1.0,
                     marker="o", markersize=4.0,
                     markerfacecolor=color,
                     markeredgecolor=color,
                     color=color)
        handles.append(h); labels.append(f"$N_X = {n_x}$")
        if paper_steps is None:
            paper_steps = steps
            paper_FN    = FN_pap

    if paper_steps is not None:
        h, = ax.plot(paper_steps, paper_FN,
                     linestyle="--", linewidth=1.0,
                     marker="x", markersize=5.0,
                     color="black")
        handles.append(h); labels.append(paper_curve_label)

    setup_axes(ax)
    ax.set_xlabel("Load step")
    ax.set_ylabel(r"$|F_N|$ [N]")
    ax.set_xticks(range(0, 7))
    leg = ax.legend(handles, labels,
                    **_legend_kwargs(ax, fontsize=8, handlelength=1.6))
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.6)


def _draw_friction_mesh(ax_fn, ax_ft, mesh_list, paper_label):
    """(d) F_N (top) + F_T (bottom) vs step, mu=0.1."""
    colors = _mesh_colormap(len(mesh_list))
    paper_steps = None
    paper_FN    = None
    paper_FT    = None
    handles, labels = [], []

    for color, (n_x, path) in zip(colors, mesh_list):
        try:
            steps, FN_sim, FN_pap, FT_sim, FT_pap = load_compare(path)
        except FileNotFoundError:
            print(f"  [warn] missing {path.name}")
            continue
        h_fn, = ax_fn.plot(steps, FN_sim,
                           linestyle="-", linewidth=1.0,
                           marker="o", markersize=4.0,
                           markerfacecolor=color,
                           markeredgecolor=color, color=color)
        if FT_sim is not None:
            ax_ft.plot(steps, np.abs(FT_sim),
                       linestyle="-", linewidth=1.0,
                       marker="o", markersize=4.0,
                       markerfacecolor=color,
                       markeredgecolor=color, color=color)
        handles.append(h_fn); labels.append(f"$N_X = {n_x}$")
        if paper_steps is None:
            paper_steps = steps
            paper_FN    = FN_pap
            paper_FT    = (np.abs(FT_pap) if FT_pap is not None else None)

    if paper_steps is not None:
        h_p, = ax_fn.plot(paper_steps, paper_FN,
                          linestyle="--", linewidth=1.0,
                          marker="x", markersize=5.0,
                          color="black")
        handles.append(h_p); labels.append(paper_label)
        if paper_FT is not None:
            ax_ft.plot(paper_steps, paper_FT,
                       linestyle="--", linewidth=1.0,
                       marker="x", markersize=5.0,
                       color="black")

    setup_axes(ax_fn)
    setup_axes(ax_ft)
    ax_fn.set_xticks(range(0, 7))
    ax_ft.set_xticks(range(0, 7))
    ax_fn.tick_params(labelbottom=False)
    ax_fn.set_ylabel(r"$|F_N|$ [N]")
    ax_ft.set_ylabel(r"$|F_T|$ [N]")
    ax_ft.set_xlabel("Load step")
    leg = ax_fn.legend(handles, labels,
                       **_legend_kwargs(ax_fn,
                                         fontsize=8, handlelength=1.6))
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.6)


def _draw_per_dt_trace(ax_fn, ax_ft):
    """(e) per-dt contact-force trace, 2-nodes vs 3-nodes, mu=0.1."""
    handles, labels = [], []
    color_2n = tuple(LAB_COLORS[0])   # lab orange  -- matches fig 2 "2-nodes"
    color_3n = tuple(LAB_COLORS[1])   # lab green   -- matches fig 2 "3-nodes"
    lw = 1.4                          # thicker than panel default

    if TRACE_IMC.exists():
        t_i, FN_i, FT_i = load_per_dt_trace(TRACE_IMC)
        h_i, = ax_fn.plot(t_i, FN_i, color=color_2n, linewidth=lw)
        ax_ft.plot(t_i, FT_i, color=color_2n, linewidth=lw)
        handles.append(h_i); labels.append("2-nodes")
    else:
        print(f"  [warn] missing {TRACE_IMC.name}")

    if TRACE_EBERLY.exists():
        t_e, FN_e, FT_e = load_per_dt_trace(TRACE_EBERLY)
        h_e, = ax_fn.plot(t_e, FN_e, color=color_3n, linewidth=lw)
        ax_ft.plot(t_e, FT_e, color=color_3n, linewidth=lw)
        handles.append(h_e); labels.append("3-nodes")
    else:
        print(f"  [warn] missing {TRACE_EBERLY.name}")

    for ax in (ax_fn, ax_ft):
        setup_axes(ax)
    ax_fn.tick_params(labelbottom=False)
    ax_fn.set_ylabel(r"$|F_N|$ on Beam X  [N]")
    ax_ft.set_ylabel(r"$|F_T|$ on Beam X  [N]")
    ax_ft.set_xlabel("time [s]")

    leg = ax_fn.legend(handles, labels,
                       **_legend_kwargs(ax_fn,
                                         fontsize=8, handlelength=2.0))
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.6)


# ============================================================================
# COMPOSE
# ============================================================================

def _add_panel_labels(pairs):
    """Place (a), (b), ... tags at a uniform absolute offset on every axes."""
    dx, dy = PANEL_LABEL_OFFSET_PT
    for ax, lab in pairs:
        ax.annotate(lab,
                    xy=(0, 1), xycoords="axes fraction",
                    xytext=(dx, dy), textcoords="offset points",
                    **PANEL_LABEL_KW)


def plot_geometry(out_dir: Path = OUT_DIR,
                  out_basename: str = OUT_BASENAME_GEOM) -> None:
    """Figure with panels (a) and (b) only."""
    apply_style()
    fig = plt.figure(figsize=FIG_SIZE_GEOM)
    gs = fig.add_gridspec(
        1, 2,
        left=0.085, right=0.95, top=0.95, bottom=0.14,
        wspace=0.30,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    _draw_top_view(ax_a)
    _draw_side_view(ax_b)
    _add_panel_labels(((ax_a, "(a)"), (ax_b, "(b)")))
    save_figure(fig, out_dir, out_basename)
    plt.close(fig)


def plot_metrics(out_dir: Path = OUT_DIR,
                 out_basename: str = OUT_BASENAME_METRIC) -> None:
    """Figure with panels (c), (d), (e)."""
    apply_style()
    fig = plt.figure(figsize=FIG_SIZE_METRICS)
    outer = fig.add_gridspec(
        2, 2,
        left=0.085, right=0.95, top=0.97, bottom=0.07,
        height_ratios=[1.0, 1.4],
        hspace=0.42, wspace=0.30,
    )
    ax_c = fig.add_subplot(outer[0, 0])
    # Panel (d): nested 2x1 in the (0, 1) cell.
    gs_d = outer[0, 1].subgridspec(2, 1, hspace=0.12)
    ax_d_fn = fig.add_subplot(gs_d[0])
    ax_d_ft = fig.add_subplot(gs_d[1], sharex=ax_d_fn)
    # Panel (e): nested 2x1 spanning both cols of row 1.
    gs_e = outer[1, :].subgridspec(2, 1, hspace=0.12)
    ax_e_fn = fig.add_subplot(gs_e[0])
    ax_e_ft = fig.add_subplot(gs_e[1], sharex=ax_e_fn)

    _draw_mesh_FN_per_step(ax_c, MESH_MU0,
                            paper_curve_label=r"paper, $\mu = 0$")
    _draw_friction_mesh(ax_d_fn, ax_d_ft, MESH_MU01,
                         paper_label=r"paper, $\mu = 0.1$")
    _draw_per_dt_trace(ax_e_fn, ax_e_ft)

    # Since the geometry and metrics figures are split, each gets its own
    # (a), (b), ... numbering.
    _add_panel_labels((
        (ax_c, "(a)"), (ax_d_fn, "(b)"), (ax_e_fn, "(c)"),
    ))
    save_figure(fig, out_dir, out_basename)
    plt.close(fig)


def plot_soft_soft() -> None:
    """Generate both companion figures."""
    plot_geometry()
    plot_metrics()


if __name__ == "__main__":
    plot_soft_soft()
