#!/usr/bin/env python3
"""
Figure 5c -- jitter on the total contact-force magnitude, correlated with
which beam nodes are currently in contact.  Window: [600, 800] s.

  - Left y-axis (blue):   Total contact-force magnitude  ||sum_i F_i(t)||
  - Right y-axis (black): Beam node index.  For each node i we draw a thin
                          horizontal segment at y = i across the time
                          intervals where |F_i(t)| > THRESHOLD * max|F|.
                          Segments are NOT connected across nodes, so the
                          plot reads as a constellation of short horizontal
                          ticks showing which nodes carry contact load.

Data source: the 2-node IMC run in
../external_reference_data/impV_noD_contact_bending_N50.npz (same file
Fig. 2 uses for its 2-node curve). It runs to t=900s at dt=0.01s, which
covers the [600, 800]s analysis window at the same resolution the figure
was produced at.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import (  # noqa: E402
    COLUMN_WIDTH_IN,
    LAB_COLORS,
    apply_style,
    load_npz_run,
    save_figure,
    setup_axes,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

NPZ_FILE = Path(__file__).resolve().parents[1] \
    / "external_reference_data" / "impV_noD_contact_bending_N50.npz"

TIME_WINDOW = (600.0, 800.0)   # seconds, inclusive

# Per-node "active" threshold: absolute floor on |F_i(t)|, in Newtons.
# Anything above this is treated as a genuinely-engaged contact (a
# floating-point floor at 1e-6 N excludes only exact numerical zeros).
# Set to 0.0 to mark every non-zero sample.
ACTIVE_THRESHOLD = 1.0e-6

# Drawing style for the left-axis force trace.
FORCE_COLOR    = tuple(LAB_COLORS[3])   # lab cyan/blue
FORCE_LW       = 1.0

# Drawing style for the right-axis "active" segments.
# Same colour is used for the right axis spine / ticks / label so it is
# visually keyed to the segments.
SEGMENT_COLOR  = tuple(LAB_COLORS[0])   # lab orange
SEGMENT_LW     = 1.0

# Fixed right-axis range (beam node index).
NODE_INDEX_LIMITS = (35, 50)

OUT_DIR = Path(__file__).parent / "figures"
OUT_BASENAME = "fig4_jitter_stacked"
INCLUDE_LABELS = True


# ============================================================================
# DATA
# ============================================================================

def _slice_window(run: dict, time_window: tuple | None):
    t = run["time"]
    if time_window is None:
        mask = np.ones_like(t, dtype=bool)
    else:
        lo, hi = time_window
        mask = (t >= lo) & (t <= hi)
        if not np.any(mask):
            raise RuntimeError(f"no samples in time window {time_window}")
    return t[mask], run["beam_forces"][mask, :, :]


def _active_segments(t: np.ndarray, active_mask: np.ndarray):
    """
    Given (T,) time and (T,) bool mask (active or not), yield (t_start, t_end)
    pairs for each contiguous run of True.  We use the timestamps themselves
    as segment edges so the segments align with sample boundaries.
    """
    if active_mask.dtype != bool:
        active_mask = active_mask.astype(bool)
    # Find rising / falling edges.
    diff = np.diff(active_mask.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    ends   = np.where(diff == -1)[0]
    if active_mask[0]:
        starts = np.r_[0, starts]
    if active_mask[-1]:
        ends = np.r_[ends, active_mask.size - 1]
    for s, e in zip(starts, ends):
        yield t[s], t[e]


# ============================================================================
# PLOT
# ============================================================================

def plot_jitter(
    npz_file: Path = NPZ_FILE,
    time_window: tuple | None = TIME_WINDOW,
    threshold: float = ACTIVE_THRESHOLD,
    out_dir: Path = OUT_DIR,
    out_basename: str = OUT_BASENAME,
    include_labels: bool = INCLUDE_LABELS,
) -> None:
    apply_style()
    run = load_npz_run(npz_file)
    t, beam_forces = _slice_window(run, time_window)

    # ||F_total(t)|| = magnitude of the sum-vector across all beam nodes.
    F_total_mag = np.linalg.norm(beam_forces.sum(axis=1), axis=1)

    # Per-node magnitude  |F_i(t)|.
    F_mag = np.linalg.norm(beam_forces, axis=2)
    if F_mag.size == 0:
        raise RuntimeError("empty force window")

    n_beam = F_mag.shape[1]

    # ---- figure layout ----------------------------------------------------
    # Square-ish, slightly larger than the column-width default so the
    # rotated left-axis label has room without overhanging the top spine.
    fig, ax_left = plt.subplots(figsize=(4.2, 3.4))
    ax_right = ax_left.twinx()

    # ---- left: total force magnitude --------------------------------------
    ax_left.plot(t, F_total_mag,
                 color=FORCE_COLOR, linewidth=FORCE_LW,
                 rasterized=False)
    setup_axes(ax_left)
    ax_left.set_xlim(t[0], t[-1])
    pad = 0.05 * (F_total_mag.max() - F_total_mag.min())
    ax_left.set_ylim(F_total_mag.min() - pad, F_total_mag.max() + pad)

    # ---- right: per-node active segments (NOT connected across nodes) ----
    active_nodes_seen = []
    for i in range(n_beam):
        active = F_mag[:, i] > threshold
        if not np.any(active):
            continue
        active_nodes_seen.append(i)
        for t_s, t_e in _active_segments(t, active):
            ax_right.hlines(i, t_s, t_e,
                            colors=SEGMENT_COLOR,
                            linewidth=SEGMENT_LW)

    # Fixed right-axis range so the comparison is consistent across runs.
    ax_right.set_ylim(*NODE_INDEX_LIMITS)

    # ---- colour-coordinate each y-axis with the data it carries ----------
    ax_left.tick_params(axis="y", direction="in", colors=FORCE_COLOR)
    ax_left.spines["left"].set_color(FORCE_COLOR)
    ax_left.yaxis.label.set_color(FORCE_COLOR)

    ax_right.spines["top"].set_visible(True)
    ax_right.spines["right"].set_visible(True)
    ax_right.spines["right"].set_color(SEGMENT_COLOR)
    ax_right.tick_params(axis="y", direction="in", colors=SEGMENT_COLOR)
    ax_right.yaxis.label.set_color(SEGMENT_COLOR)
    # The shared right spine on the LEFT axes also needs the segment colour
    # (matplotlib twins share spines).
    ax_left.spines["right"].set_color(SEGMENT_COLOR)

    if include_labels:
        ax_left.set_xlabel("time, $t$ [s]")
        # y < 0.5 shifts the rotated label's center DOWN along the axis, so
        # the "[N]" at the top of the rotated string stays inside the frame.
        ax_left.set_ylabel(
            r"Contact force magnitude, $\|\sum_i F_i\|$, [N]",
            y=0.5)
        ax_right.set_ylabel("Active beam nodes index", y=0.5)
    else:
        ax_left.set_xticklabels([])
        ax_left.set_yticklabels([])
        ax_right.set_yticklabels([])

    fig.tight_layout()
    save_figure(fig, out_dir, out_basename)
    plt.close(fig)


if __name__ == "__main__":
    plot_jitter()
