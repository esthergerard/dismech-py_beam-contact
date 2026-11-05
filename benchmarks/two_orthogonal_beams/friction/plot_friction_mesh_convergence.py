#!/usr/bin/env python3
"""
Plot mesh-convergence overlays of F_N and F_T vs load step for the
Zavarise & Wriggers (2000) Example 2 benchmark, friction cases.

For each value of mu, draws a 2-panel figure with:
  - top   panel: |F_N|(step)   for nx in {21, 51, 101} + paper Table IV (if known)
  - bottom panel: |F_T|(step)  same overlays

Reads `zavarise_compare_steps6_nx{nx}_mu{mu_tag}.txt` files written by
contact_bending_soft_soft.py.  Each row has columns:
  step, FN_sim, FN_paper, FN_ratio, FT_sim, FT_paper, FT_ratio,
  Ffric_z, tilt_deg
"""
import os
import sys
from pathlib import Path

# Force the headless backend BEFORE pyplot is imported (cluster-safe; never
# attempts to open a display window).
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = Path(__file__).resolve().parent.parent / "contact_output_files_soft_soft"


# Zavarise & Wriggers Table IV (F_N, F_T) per (step, mu)
ZAVARISE_TABLE_IV = {
    0.0: {1: (0.00,  0.00),  2: (1e-9, 0.00),  3: (17.2, 0.00),
          4: (34.3, 0.00),  5: (51.3, 0.00),   6: (68.4, 0.00)},
    0.1: {1: (0.00,  0.00),  2: (1e-9, 0.00),  3: (17.0, 1.70),
          4: (33.8, 3.38),  5: (50.4, 5.04),   6: (67.0, 6.70)},
    0.8: {1: (0.00,  0.00),  2: (1e-9, 0.00),  3: (16.0, 11.5),
          4: (30.8, 24.5),  5: (45.0, 36.0),   6: (58.4, 46.7)},
    1.5: {1: (0.00,  0.00),  2: (1e-9, 0.00),  3: (16.0, 11.5),
          4: (30.8, 24.7),  5: (44.3, 39.4),   6: (56.4, 55.8)},
}


def mu_tag(mu):
    """Encode mu = 0.1 → '01', 0.6 → '06', etc."""
    return f"{int(round(mu*10)):02d}"


def load_compare(path):
    """Return dict step -> (FN, FT).  Skip rows where FN/FT are 'nan'."""
    out = {}
    if not path.exists():
        return out
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            try:
                step = int(parts[0])
                FN = float(parts[1])
                FT = float(parts[4])
            except (ValueError, IndexError):
                continue
            out[step] = (FN, FT)
    return out


def plot_one_mu(mu, nx_list, out_path, n_steps=6):
    """Two-panel mesh-convergence figure for a single mu.

    If no curves can be drawn at all (no compare files yet AND no paper
    reference for this mu), the figure is NOT written — keeps the disk
    free of empty PNGs during incremental cluster runs.
    """
    tag = mu_tag(mu)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(nx_list)))

    fig, (ax_N, ax_T) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    n_curves = 0

    # Paper reference if available
    if mu in ZAVARISE_TABLE_IV:
        ref = ZAVARISE_TABLE_IV[mu]
        ref_steps = sorted(ref.keys())
        ax_N.plot(ref_steps, [ref[s][0] for s in ref_steps],
                  marker="o", linestyle="--", color="black",
                  linewidth=2.5, markersize=9, markerfacecolor="none",
                  label=f"Zavarise & Wriggers (2000), $\\mu$={mu}")
        ax_T.plot(ref_steps, [ref[s][1] for s in ref_steps],
                  marker="o", linestyle="--", color="black",
                  linewidth=2.5, markersize=9, markerfacecolor="none",
                  label=f"Zavarise & Wriggers (2000), $\\mu$={mu}")
        n_curves += 1

    for k, nx in enumerate(nx_list):
        path = OUT_DIR / f"zavarise_compare_steps{n_steps}_nx{nx}_mu{tag}.txt"
        data = load_compare(path)
        if not data:
            print(f"  [plot] missing: {path.name} — skipping nx={nx}")
            continue
        steps = sorted(data.keys())
        FN = [data[s][0] for s in steps]
        FT = [data[s][1] for s in steps]
        ax_N.plot(steps, FN, marker="o", linestyle="-",
                  color=cmap[k], linewidth=2, markersize=6,
                  label=f"dismech, $n_x = {nx}$")
        ax_T.plot(steps, FT, marker="o", linestyle="-",
                  color=cmap[k], linewidth=2, markersize=6,
                  label=f"dismech, $n_x = {nx}$")
        n_curves += 1

    if n_curves == 0:
        plt.close(fig)
        print(f"  [plot] mu={mu}: no data yet AND no paper reference; "
              f"skipping {out_path.name}")
        return

    for ax, label in [(ax_N, "$|F_N|$  [N]"),
                       (ax_T, "$|F_T|$  [N]")]:
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(label, fontsize=12)
        ax.legend(loc="upper left", fontsize=9)
    ax_T.set_xlabel("Load step", fontsize=12)
    ax_T.set_xticks(range(1, n_steps + 1))

    fig.suptitle(
        f"Zavarise & Wriggers (2000) Example 2 — mesh convergence  "
        f"($\\mu = {mu}$, {n_steps} load steps)",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nx_list = [21, 51, 101]
    # mu=0.1 is the frictional coefficient used in the paper (Sec. 4.3); an
    # earlier mu=0.8 trial was explored but never carried through to final
    # data (see NOTES.md: "Two friction coefficients retained for the paper").
    for mu in (0.1,):
        out_png = OUT_DIR / f"zavarise_friction_mesh_convergence_mu{mu_tag(mu)}.png"
        plot_one_mu(mu, nx_list, out_png)


if __name__ == "__main__":
    main()
