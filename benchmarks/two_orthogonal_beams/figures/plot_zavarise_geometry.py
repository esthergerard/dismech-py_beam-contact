#!/usr/bin/env python3
"""
Two-panel rendering of the Zavarise & Wriggers Example 2 geometry:

  (1) TOP view of the initial undeformed configuration  (x-z plane),
      showing the off-center crossing (10 m from clamped end A,
      4 m from free end B).

  (2) SIDE view of the deformed configuration at the final load step
      (z-y plane), showing Beam Z bending around the contact with
      Beam X (which stays essentially straight because it is
      clamped at both ends and very stiff in this projection).

The script reads node coordinates from one of the simulation output files
(zavarise_coords_steps{N}{tag}.txt).  Usage:

  python3 plot_zavarise_geometry.py
        # auto-pick the highest-resolution / newest coords file available
  python3 plot_zavarise_geometry.py --coords path/to/zavarise_coords_steps6_nx101.txt
        # explicit file
  python3 plot_zavarise_geometry.py --step 6
        # plot a specific load step (default = last step in the file)
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Parameters (must match contact_bending_soft_soft.py)
X_BEAM_LEN  = 10.0
Z_BEAM_LEN  = 14.0
BEAM_RADIUS = 0.2
Y_X = +0.5           # Beam X centerline y
Y_Z = -0.5           # Beam Z centerline y (undeformed)
Z_X_CONTACT = +3.0   # Beam X z-position = contact z


# ----------------------------------------------------------------------------
# Reading the simulation coords file
# ----------------------------------------------------------------------------
def load_coords_at_step(path, step_id=None, n_x_nodes=None, n_z_nodes=None):
    """Return (x_beam_xyz, z_beam_xyz, step_used).

    The file format produced by contact_bending_soft_soft.py is:
        # t_s  step_id  node_id  x_m  y_m  z_m

    `n_x_nodes` and `n_z_nodes` only need to be passed if the file does not
    cover them implicitly (we detect them from the max node_id).
    """
    snapshots = {}  # step -> {node_id: (x, y, z)}
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
    if not snapshots:
        raise RuntimeError(f"No snapshots found in {path}")

    if step_id is None:
        step_id = max(snapshots.keys())
    if step_id not in snapshots:
        raise RuntimeError(f"Step {step_id} not found in {path} "
                           f"(available: {sorted(snapshots)})")
    pts = snapshots[step_id]
    n_total = max(pts.keys()) + 1
    coords = np.zeros((n_total, 3))
    for nid, xyz in pts.items():
        coords[nid] = xyz

    if n_x_nodes is None:
        # Detect from y-coordinate: Beam X nodes have y ~ +0.5, Beam Z nodes
        # have y ~ -0.5 (initially).  Use the count of nodes with y > 0 at
        # step 0 as a robust fallback, but here just use the simplest rule:
        # the first contiguous block where x varies and z is constant is Beam X.
        # Easier: try the conventional values that match contact_bending_soft_soft.py.
        # Beam X has length 10 with 1 m elements => 11 nodes; Beam Z has 14 m
        # with 1 m elements => 15 nodes. For finer meshes we infer from n_total.
        # The element length on both beams is kept equal in the test, so
        # n_z = (Z_BEAM_LEN / X_BEAM_LEN) * (n_x - 1) + 1.
        # Solve for n_x given n_total = n_x + n_z:
        #   n_total = n_x + (Z_BEAM_LEN/X_BEAM_LEN)*(n_x - 1) + 1
        # => n_x = (n_total - 1 + Z_BEAM_LEN/X_BEAM_LEN) / (1 + Z_BEAM_LEN/X_BEAM_LEN)
        ratio = Z_BEAM_LEN / X_BEAM_LEN
        n_x_nodes = int(round((n_total - 1 + ratio) / (1 + ratio)))
        n_z_nodes = n_total - n_x_nodes

    return coords[:n_x_nodes], coords[n_x_nodes:n_x_nodes + n_z_nodes], step_id


def auto_pick_coords_file(out_dir):
    """Return the highest-mesh coords file, or fall back to the old midspan
    geometry coords if that's all we have locally."""
    # Try new-geometry coords first, in order of decreasing mesh size
    for nx in (101, 51, 21, 11):
        p = out_dir / f"zavarise_coords_steps6_nx{nx}.txt"
        if p.exists():
            return p, f"_nx{nx}"
    # Plain (no _nx tag) — single-mesh run with the new geometry
    p = out_dir / "zavarise_coords_steps6.txt"
    if p.exists():
        return p, ""
    # Old midspan geometry, as a last resort
    p = out_dir / "_old_midspan_geometry" / "zavarise_coords_steps6.txt"
    if p.exists():
        return p, "_old_midspan_geometry"
    raise FileNotFoundError(
        "No suitable coords file found under contact_output_files_soft_soft/. "
        "Run the simulation first or copy a coords file from the cluster.")


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def make_figure(coords_path, step_id, save_path):
    # ---- Initial-geometry node positions (no data needed) ----------------
    n_x0, n_z0 = 11, 15
    x0 = np.zeros((n_x0, 3))
    x0[:, 0] = np.linspace(-X_BEAM_LEN / 2, X_BEAM_LEN / 2, n_x0)
    x0[:, 1] = Y_X
    x0[:, 2] = Z_X_CONTACT
    z0 = np.zeros((n_z0, 3))
    z0[:, 0] = 0.0
    z0[:, 1] = Y_Z
    z0[:, 2] = np.linspace(-Z_BEAM_LEN / 2, Z_BEAM_LEN / 2, n_z0)

    # ---- Deformed coordinates from the simulation ------------------------
    x_def, z_def, step_used = load_coords_at_step(coords_path, step_id)

    fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(15, 7))

    # =====================================================================
    # (1) Top view (x-z plane) — INITIAL geometry
    # =====================================================================
    ax_top.plot(x0[:, 0], x0[:, 2], color="C0", linewidth=2.5,
                marker="o", markersize=5, label="Beam X (L = 10 m)")
    ax_top.plot(z0[:, 0], z0[:, 2], color="C1", linewidth=2.5,
                marker="s", markersize=5, label="Beam Z (L = 14 m)")
    # Clamps + contact + endpoints
    ax_top.scatter([x0[0, 0], x0[-1, 0]], [x0[0, 2], x0[-1, 2]],
                   color="red", s=120, marker="s", zorder=6)
    ax_top.scatter([z0[0, 0]], [z0[0, 2]], color="darkgreen", s=130,
                   marker="s", zorder=6)
    ax_top.scatter([z0[-1, 0]], [z0[-1, 2]], color="black", s=80,
                   marker="o", zorder=6)
    ax_top.scatter([0], [Z_X_CONTACT], facecolor="none", edgecolor="magenta",
                   s=180, marker="o", linewidth=2, zorder=7,
                   label=f"contact (z = +{Z_X_CONTACT:g})")

    ax_top.annotate("C", (x0[0, 0],  x0[0, 2]),
                    xytext=(-14, -12), textcoords="offset points",
                    color="red", fontsize=12)
    ax_top.annotate("D", (x0[-1, 0], x0[-1, 2]),
                    xytext=(6, -12), textcoords="offset points",
                    color="red", fontsize=12)
    ax_top.annotate("A (clamped + driven)", (z0[0, 0], z0[0, 2]),
                    xytext=(10, -6), textcoords="offset points",
                    color="darkgreen", fontsize=11)
    ax_top.annotate("B (free)", (z0[-1, 0], z0[-1, 2]),
                    xytext=(10, -3), textcoords="offset points",
                    color="black", fontsize=11)

    # L1, L2 brackets
    ax_top.annotate("", xy=(0, Z_X_CONTACT), xytext=(0, z0[0, 2]),
                    arrowprops=dict(arrowstyle="<->", color="darkgreen", lw=2))
    ax_top.text(0.3, (Z_X_CONTACT + z0[0, 2]) / 2,
                "L₁ = 10 m\n(A → contact)",
                color="darkgreen", fontsize=11, fontweight="bold",
                verticalalignment="center")
    ax_top.annotate("", xy=(0, z0[-1, 2]), xytext=(0, Z_X_CONTACT),
                    arrowprops=dict(arrowstyle="<->", color="black", lw=2))
    ax_top.text(0.3, (Z_X_CONTACT + z0[-1, 2]) / 2,
                "L₂ = 4 m\n(contact → B)",
                color="black", fontsize=11, fontweight="bold",
                verticalalignment="center")

    ax_top.set_xlabel("x [m]", fontsize=12)
    ax_top.set_ylabel("z [m]", fontsize=12)
    ax_top.set_title("Initial geometry — top view (x-z plane)\n"
                     "Off-center crossing", fontsize=12)
    ax_top.set_aspect("equal")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="upper right", fontsize=10)

    # =====================================================================
    # (2) Side view (z-y plane) — DEFORMED shape at the last step
    # =====================================================================
    # Beam Z (in the simulation's deformed state)
    ax_side.plot(z_def[:, 2], z_def[:, 1], color="C1", linewidth=2.5,
                 marker="s", markersize=5,
                 label=f"Beam Z deformed (step {step_used})")
    # Beam Z initial (reference)
    ax_side.plot(z0[:, 2], z0[:, 1], color="C1", linewidth=1.0,
                 linestyle=":", alpha=0.6,
                 label="Beam Z initial")
    # Beam X (a single point in this projection, since it's at z=+3, y=+0.5
    # in the undeformed configuration).  Show its actual deformed average y.
    x_y_mean = float(np.mean(x_def[:, 1]))
    x_z_mean = float(np.mean(x_def[:, 2]))
    ax_side.scatter([x_z_mean], [x_y_mean], color="C0", s=160, marker="o",
                    zorder=6,
                    label=f"Beam X midspan (≈ point at z=+{Z_X_CONTACT:g})")
    # Initial Beam X marker (open circle)
    ax_side.scatter([Z_X_CONTACT], [Y_X], facecolor="none",
                    edgecolor="C0", s=160, marker="o", linewidth=1.5,
                    label="Beam X initial position")

    # Mark A, B
    ax_side.scatter([z_def[0, 2]], [z_def[0, 1]], color="darkgreen", s=130,
                    marker="s", zorder=6)
    ax_side.scatter([z_def[-1, 2]], [z_def[-1, 1]], color="black", s=80,
                    marker="o", zorder=6)
    ax_side.annotate(f"A  (Δuy = {z_def[0, 1] - Y_Z:+.2f} m,\n"
                     f"     Δuz = {z_def[0, 2] - z0[0, 2]:+.2f} m)",
                     (z_def[0, 2], z_def[0, 1]),
                     xytext=(8, -8), textcoords="offset points",
                     color="darkgreen", fontsize=10)
    ax_side.annotate(f"B", (z_def[-1, 2], z_def[-1, 1]),
                     xytext=(8, -5), textcoords="offset points",
                     color="black", fontsize=12)

    # Mark contact point (midpoint of the closest pair)
    # Find the Beam Z node closest in z to the Beam X position
    iz = int(np.argmin(np.abs(z_def[:, 2] - x_z_mean)))
    ax_side.scatter([z_def[iz, 2]], [z_def[iz, 1]], facecolor="none",
                    edgecolor="magenta", s=200, marker="o", linewidth=2,
                    zorder=7)
    ax_side.annotate("contact", (z_def[iz, 2], z_def[iz, 1]),
                     xytext=(-50, -18), textcoords="offset points",
                     color="magenta", fontsize=10)

    # Reference horizontal lines
    ax_side.axhline(Y_X, color="grey", lw=0.7, ls="--", alpha=0.5)
    ax_side.text(z_def[-1, 2] + 0.2, Y_X, "Beam X initial y",
                 fontsize=8, color="grey", verticalalignment="center")

    ax_side.set_xlabel("z [m]", fontsize=12)
    ax_side.set_ylabel("y [m]", fontsize=12)
    ax_side.set_title(f"Deformed configuration — side view (z-y plane)\n"
                      f"Beam Z bent around the contact (step {step_used})",
                      fontsize=12)
    ax_side.grid(True, alpha=0.3)
    ax_side.legend(loc="lower left", fontsize=9)

    fig.suptitle(
        f"Zavarise & Wriggers (2000) Example 2  —  "
        f"coords source: {coords_path.name}",
        fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Render initial top-view + deformed side-view geometry.")
    parser.add_argument("--coords", type=Path, default=None,
                        help="Path to a zavarise_coords_steps*.txt file. "
                             "If omitted, auto-picks the highest-mesh new "
                             "geometry available, or falls back to the "
                             "midspan archive.")
    parser.add_argument("--step", type=int, default=None,
                        help="Load step to plot (default: last in the file).")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent.parent / "contact_output_files_soft_soft"
                                  / "zavarise_geometry.png",
                        help="Output PNG path.")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "contact_output_files_soft_soft"

    if args.coords is None:
        coords_path, tag = auto_pick_coords_file(out_dir)
        print(f"Using {coords_path}  (tag = '{tag}')")
    else:
        coords_path = args.coords

    args.out.parent.mkdir(parents=True, exist_ok=True)
    make_figure(coords_path, args.step, args.out)


if __name__ == "__main__":
    main()
