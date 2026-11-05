#!/usr/bin/env python3
"""
Manuscript figure for Appendix A.1: numerical verification that the jump of
the distance-gradient at a concave two-segment junction equals 2 cos(alpha)
(Eq. A3).  Overlays the analytic prediction with the discrete geometric
check of jump_2cosalpha_check.py and annotates the worst relative error.

Reuses the geometry of jump_2cosalpha_check.py and the lab figure style
(plot_style.py): CM fonts via mathtext (no local LaTeX), lab colour palette,
hollow markers, black-edged legend, PDF + SVG output.

Writes figures/fig_jump_2cosalpha.{pdf,svg} (+ .png preview).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from jump_2cosalpha_check import grad_delta
from plot_style import LAB_COLORS, apply_style, panel_legend, save_figure

FIG = HERE / "figures"

C_THEORY = LAB_COLORS[4]   # black -- analytic prediction
C_NUM = LAB_COLORS[3]      # blue  -- numerically measured jump

D, EPS = 0.3, 1e-9
ALPHA_DEG_SAMPLES = np.arange(10.0, 80.0 + 1e-9, 5.0)


def measured_jump(alpha_deg: float) -> float:
    al = np.radians(alpha_deg)
    p_left = np.array([-np.sin(al), np.cos(al)])
    p_right = np.array([np.sin(al), np.cos(al)])
    verts = [p_left, np.zeros(2), p_right]
    g_l = grad_delta(np.array([-EPS, D]), verts)
    g_r = grad_delta(np.array([+EPS, D]), verts)
    return np.linalg.norm(g_r - g_l)


def main() -> None:
    apply_style()

    alpha_fine = np.linspace(1.0, 89.0, 400)
    theory = 2 * np.cos(np.radians(alpha_fine))

    measured = np.array([measured_jump(a) for a in ALPHA_DEG_SAMPLES])
    predicted = 2 * np.cos(np.radians(ALPHA_DEG_SAMPLES))
    worst = np.max(np.abs(measured - predicted) / predicted)

    fig, ax = plt.subplots(figsize=(3.5, 2.625))
    ax.plot(alpha_fine, theory, "-", color=C_THEORY, lw=1.0,
            label="analytic")
    ax.plot(ALPHA_DEG_SAMPLES, measured, "o", mfc="none", mec=C_NUM, ms=5,
            label="numerical (2-node model)")
    ax.set_xlabel(r"half-angle $\alpha$ [deg]")
    ax.set_ylabel(r"gradient jump $\Vert[\![\nabla\Delta]\!]\Vert$")
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 2.1)
    ax.set_box_aspect(3.0 / 4.0)
    panel_legend(ax, *ax.get_legend_handles_labels(), "upper right")

    fig.tight_layout()
    save_figure(fig, FIG, "fig_jump_2cosalpha")
    fig.savefig(FIG / "fig_jump_2cosalpha.png", dpi=300)
    print(f"  wrote {FIG / 'fig_jump_2cosalpha.png'}")
    print(f"  worst relative error over the sweep: {worst:.2e}")


if __name__ == "__main__":
    main()
