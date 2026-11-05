#!/usr/bin/env python3
"""
Two exploratory companions to panels (a)/(b) of ``fig_combined``.

1. ``fig_force_derivative`` -- contact stiffness dF*/d(delta/L), obtained by
   central finite differences on the force-displacement curves:

     +-----------------------+-----------------------+
     | (a) full range        | (b) ZOOM on the same  |
     |                       |     window as fig_comb|
     +-----------------------+-----------------------+

   The derivative is the cleanest way to see the jitter: a force wobble of a
   few percent becomes a stiffness swing of several hundred percent.

2. ``fig_force_spectrum`` -- frequency content of the same curves:

     +-----------------------+-----------------------+
     | (a) detrended residual| (b) amplitude spectrum|
     +-----------------------+-----------------------+

   The curves are sampled on a uniform delta/L grid, so the "frequency" axis
   is a spatial frequency in cycles per unit delta/L.  A smooth trend
   (Savitzky-Golay) is removed first, leaving only the node-crossing ripple.

Output: figures/fig_force_derivative.{pdf,svg}
        figures/fig_force_spectrum.{pdf,svg}
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import (  # noqa: E402
    LAB_COLORS,
    add_zoom_indicator,
    apply_style,
    panel_legend,
    save_figure,
    setup_axes,
)
from fig2_force_displacement import (  # noqa: E402
    ANALYTICAL_CSV,
    ANALYTICAL_FACTOR,
    ANALYTICAL_STYLE,
    L,
    SIM_RUNS,
    extract_from_npz,
    extract_from_txt,
    read_analytical_csv,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

OUT_DIR = Path(__file__).parent / "figures"

FIG_SIZE   = (8.0, 3.4)          # two panels side by side
ZOOM_XLIM  = (0.30, 0.40)        # same window as fig_combined panel (b)

# Central-difference half-width, in delta/L.  The raw sampling step is
# 6.25e-6, so differencing neighbouring samples would only show integrator
# noise.  ~5e-4 still resolves the shortest ripple (period ~2.8e-3) while
# averaging the per-step noise away.
FD_HALF_WIDTH = 5.0e-4

# Analysis window for the spectrum: away from the loading onset and short of
# where the shortest (1-node) run ends.
SPEC_WINDOW  = (0.15, 0.50)
SG_WINDOW    = 4001              # Savitzky-Golay detrend length, in samples
SG_ORDER     = 3
N_BEAM_NODES = 50                # nodes used in these runs (see fig2 header)

PANEL_LABEL_KW = dict(fontsize=11, fontweight="bold", va="top", ha="left")


# ============================================================================
# DATA
# ============================================================================

def _load_curves():
    """Return [(label, style, delta/L, F*)] for every available sim run."""
    curves = []
    for run in SIM_RUNS:
        try:
            if run["kind"] == "txt":
                dL, FL2 = extract_from_txt(Path(run["coords"]),
                                           Path(run["forces"]))
            else:
                dL, FL2 = extract_from_npz(Path(run["path"]))
        except FileNotFoundError:
            print(f"  [warn] {run['label']!r}: data missing; skipping")
            continue
        if dL.size == 0:
            continue
        dl_max = run.get("delta_l_max")
        if dl_max is not None:
            keep = dL <= dl_max
            dL, FL2 = dL[keep], FL2[keep]
        curves.append((run["label"], run["style"], dL, FL2))
    return curves


def _analytical_derivative():
    """(eta, dF*/d(delta/L)) from the analytical CSV, or None if absent."""
    if not Path(ANALYTICAL_CSV).is_file():
        return None
    eta, fL2 = read_analytical_csv(ANALYTICAL_CSV)
    if eta.size <= 2:
        return None
    order = np.argsort(eta)
    eta, fL2 = eta[order], fL2[order] * ANALYTICAL_FACTOR
    return eta, np.gradient(fL2, eta)


def _central_difference(x, y, half_width=FD_HALF_WIDTH):
    """
    dy/dx by central differences over +/- ``half_width`` in x.

    ``x`` is uniformly sampled here, so the half-width maps onto a fixed
    sample stride ``k`` and the estimate is (y[i+k] - y[i-k]) / (2 k dx).
    """
    dx = float(np.median(np.diff(x)))
    k = max(1, int(round(half_width / dx)))
    if 2 * k >= x.size:
        k = max(1, x.size // 4)
    dy = (y[2 * k:] - y[:-2 * k]) / (x[2 * k:] - x[:-2 * k])
    return x[k:-k], dy


def _residual(x, y, window=SG_WINDOW, order=SG_ORDER):
    """Curve minus a smooth Savitzky-Golay trend of the same curve."""
    n = y.size
    w = min(window, n if n % 2 else n - 1)
    if w <= order:
        return np.zeros_like(y)
    return y - savgol_filter(y, w, order)


def _amplitude_spectrum(x, r):
    """Single-sided amplitude spectrum of ``r`` vs spatial frequency in x."""
    dx = float(np.median(np.diff(x)))
    n = r.size
    win = np.hanning(n)
    amp = 2.0 * np.abs(np.fft.rfft(r * win)) / win.sum()
    freq = np.fft.rfftfreq(n, dx)
    return freq, amp


def _log_bin(freq, amp, n_bins=220):
    """Geometric-mean binning: keeps the peaks, kills the FFT grass."""
    pos = freq > 0
    freq, amp = freq[pos], amp[pos]
    edges = np.geomspace(freq[0], freq[-1], n_bins + 1)
    idx = np.digitize(freq, edges) - 1
    out_f, out_a = [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out_f.append(freq[m].mean())
            out_a.append(amp[m].max())
    return np.asarray(out_f), np.asarray(out_a)


# ============================================================================
# FIGURE 1 -- DERIVATIVE
# ============================================================================

def _draw_derivative(ax, curves, *, xlim=None, analytical=None):
    handles, labels = [], []
    y_in_view = []
    y_all = []

    if analytical is not None:
        eta, dfd = analytical
        line, = ax.plot(eta, dfd, **ANALYTICAL_STYLE)
        handles.append(line); labels.append("Analytical")
        if xlim is not None:
            m = (eta >= xlim[0]) & (eta <= xlim[1])
            if m.any():
                y_in_view.append(dfd[m])

    for label, style, dL, FL2 in curves:
        x, dy = _central_difference(dL, FL2)
        line, = ax.plot(x, dy, **{**style, "linewidth": 0.9})
        handles.append(line); labels.append(label)
        y_all.append(dy)
        if xlim is not None:
            m = (x >= xlim[0]) & (x <= xlim[1])
            if m.any():
                y_in_view.append(dy[m])

    ax.axhline(0.0, color="0.6", linewidth=0.5, zorder=0)
    setup_axes(ax)
    if xlim is not None:
        ax.set_xlim(*xlim)
        if y_in_view:
            yc = np.concatenate(y_in_view)
            lo, hi = float(yc.min()), float(yc.max())
            pad = 0.05 * (hi - lo) if hi > lo else 0.05 * abs(hi)
            ax.set_ylim(lo - pad, hi + pad)
    elif y_all:
        # Symmetric, percentile-clipped range: the end-of-run transients
        # (the 1-node curve collapsing near delta/L = 0.57) would otherwise
        # squash everything else into a flat band.
        span = float(np.percentile(np.abs(np.concatenate(y_all)), 99.5))
        ax.set_ylim(-1.15 * span, 1.15 * span)

    ax.set_xlabel(r"Normalized displacement, $\delta / L$")
    ax.set_ylabel(r"Normalized stiffness, "
                  r"$\mathrm{d}(F L^{2}/EI) \, / \, \mathrm{d}(\delta/L)$")
    return handles, labels


def plot_derivative(out_dir: Path = OUT_DIR,
                    out_basename: str = "fig_force_derivative") -> None:
    apply_style()
    curves = _load_curves()

    analytical = _analytical_derivative()

    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(1, 2, left=0.09, right=0.985, top=0.95, bottom=0.16,
                          wspace=0.30)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    h_a, l_a = _draw_derivative(ax_a, curves, analytical=analytical)
    panel_legend(ax_a, h_a, l_a, "lower right")

    h_b, l_b = _draw_derivative(ax_b, curves, xlim=ZOOM_XLIM,
                                analytical=analytical)
    panel_legend(ax_b, h_b, l_b, "lower left")
    add_zoom_indicator(fig, ax_a, ax_b, ZOOM_XLIM)

    for ax, lab in zip((ax_a, ax_b), ("(a)", "(b)")):
        ax.text(0.03, 0.97, lab, transform=ax.transAxes, **PANEL_LABEL_KW)

    save_figure(fig, out_dir, out_basename)
    plt.close(fig)


# ============================================================================
# FIGURE 2 -- SPECTRUM
# ============================================================================

def plot_spectrum(out_dir: Path = OUT_DIR,
                  out_basename: str = "fig_force_spectrum") -> None:
    apply_style()
    curves = _load_curves()

    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(1, 2, left=0.09, right=0.985, top=0.95, bottom=0.16,
                          wspace=0.28)
    ax_r, ax_s = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    handles, labels, rms_labels = [], [], []
    for label, style, dL, FL2 in curves:
        m = (dL >= SPEC_WINDOW[0]) & (dL <= SPEC_WINDOW[1])
        if m.sum() < SG_WINDOW:
            print(f"  [warn] {label!r}: window too short for the spectrum")
            continue
        x, y = dL[m], FL2[m]
        r = _residual(x, y)

        line, = ax_r.plot(x, r, **{**style, "linewidth": 0.8})
        handles.append(line)
        labels.append(label)
        # RMS in the residual-panel label: the 3-node trace is invisible at
        # this scale precisely because it is ~1000x smaller -- state the number.
        exponent = int(np.floor(np.log10(r.std())))
        mantissa = r.std() / 10.0**exponent
        rms_labels.append(rf"{label}, rms $= {mantissa:.1f}"
                          rf"\times 10^{{{exponent}}}$")

        freq, amp = _amplitude_spectrum(x, r)
        fb, ab = _log_bin(freq, amp)
        ax_s.plot(fb, ab, **{**style, "linewidth": 1.0})
        print(f"  {label:<16s} residual RMS = {r.std():.2e}"
              f"   peak at {freq[np.argmax(amp)]:.1f} cycles per delta/L")

    # Node-crossing reference: one beam-node spacing is L/(N-1), i.e.
    # 1/(N-1) in delta/L units, so the contact point crosses (N-1) nodes per
    # unit delta/L if it swept the full rod.
    f_node = float(N_BEAM_NODES - 1)
    ref = ax_s.axvline(f_node, color=LAB_COLORS[4], linestyle=":",
                       linewidth=0.8)
    handles.append(ref)
    labels.append(rf"node spacing, $N-1 = {N_BEAM_NODES - 1}$")

    setup_axes(ax_r)
    ax_r.set_xlim(*SPEC_WINDOW)
    ax_r.set_xlabel(r"Normalized displacement, $\delta / L$")
    ax_r.set_ylabel(r"Force residual, $\Delta (F L^{2}/EI)$")
    panel_legend(ax_r, handles[:-1], rms_labels, "lower left")

    setup_axes(ax_s)
    ax_s.set_xscale("log")
    ax_s.set_yscale("log")
    ax_s.set_xlim(5.0, 5.0e3)
    ax_s.set_xlabel(r"Spatial frequency [cycles per unit $\delta/L$]")
    ax_s.set_ylabel(r"Amplitude, $|\widehat{\Delta F^{*}}|$")
    panel_legend(ax_s, handles, labels, "lower left")

    for ax, lab in zip((ax_r, ax_s), ("(a)", "(b)")):
        ax.text(0.03, 0.97, lab, transform=ax.transAxes, **PANEL_LABEL_KW)

    save_figure(fig, out_dir, out_basename)
    plt.close(fig)


if __name__ == "__main__":
    plot_derivative()
    plot_spectrum()
