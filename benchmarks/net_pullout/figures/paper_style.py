"""Paper-figure style helper (advisor spec).

Applies Khalid's group's manuscript-figure rules: serif Computer-Modern
text at 10 pt, black-edged spines and legend frames, hollow markers,
PDF+SVG output.  Every paper-figure script in this project should call
`apply_paper_style()` and use the palette in `COLORS`, so the manuscript
stays visually consistent.

The advisor's sample script uses `text.usetex=True`; that requires a
LaTeX toolchain on PATH.  On this Mac there is none, so `apply_paper_style`
falls back to matplotlib mathtext with the CM font set, which renders
visually close to CMR10 without any external dependency.  Flip
`use_tex=True` (or set env `USETEX=1`) when running where LaTeX is
installed to get the final CMR10/CMMI10 rendering.
"""
from __future__ import annotations
import os
import shutil
from pathlib import Path
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# --- Advisor palette (RGB / 255) --------------------------------------
_PALETTE_RGB = np.array([
    [247, 148,  30],  # orange
    [  0, 166,  81],  # green   -- central strand
    [237,  28,  36],  # red     -- y-beams
    [  0, 174, 239],  # blue    -- outer x-beams
    [  0,   0,   0],  # black
], dtype=float) / 255.0

COLORS = {
    "orange": tuple(_PALETTE_RGB[0]),
    "green":  tuple(_PALETTE_RGB[1]),
    "red":    tuple(_PALETTE_RGB[2]),
    "blue":   tuple(_PALETTE_RGB[3]),
    "black":  tuple(_PALETTE_RGB[4]),
}

# --- Standard sizes (inches) ------------------------------------------
COLUMN_WIDTH = 3.5        # single-column journal figure
PAGE_WIDTH   = 6.5        # full-page figure
PWIDTH       = 4.0        # sample-script default single plot
PHEIGHT      = 3.0

FONTSIZE = 10             # advisor: 8 or 10, at most 2 sizes per figure


def _latex_available() -> bool:
    return shutil.which("latex") is not None and \
           shutil.which("dvipng") is not None


def apply_paper_style(use_tex: bool | None = None) -> None:
    """Set matplotlib rcParams to the advisor's paper spec.

    Parameters
    ----------
    use_tex : bool or None
        If None, auto-detect: use text.usetex when LaTeX+dvipng are on
        PATH (or when USETEX=1 in env), else fall back to mathtext-cm.
    """
    if use_tex is None:
        use_tex = os.environ.get("USETEX", "").strip() == "1" or _latex_available()

    if use_tex:
        plt.rcParams["text.usetex"] = True
        plt.rcParams["font.family"] = "serif"
    else:
        # Register matplotlib's bundled Computer Modern TTF files so we can
        # use the TRUE cmr10 / cmmi10 fonts (visually identical to LaTeX
        # CMR10) without needing a LaTeX toolchain.
        cm_dir = Path(mpl.get_data_path()) / "fonts" / "ttf"
        for ttf in cm_dir.glob("cm*.ttf"):
            try:
                fm.fontManager.addfont(str(ttf))
            except Exception:
                pass
        plt.rcParams["text.usetex"] = False
        plt.rcParams["font.family"] = "cmr10"
        # Math is rendered via mathtext with the "cm" font set (CMMI10 for
        # italic, CMSY10 for symbols), matching LaTeX output.
        plt.rcParams["mathtext.fontset"] = "cm"
        # cmr10 lacks a minus glyph; use mathtext for the axis tick minus
        # sign to avoid the "Glyph missing" warning.
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["axes.formatter.use_mathtext"] = True

    plt.rcParams["font.size"]        = FONTSIZE
    plt.rcParams["axes.labelsize"]   = FONTSIZE
    plt.rcParams["axes.titlesize"]   = FONTSIZE
    plt.rcParams["xtick.labelsize"]  = FONTSIZE
    plt.rcParams["ytick.labelsize"]  = FONTSIZE
    plt.rcParams["legend.fontsize"]  = FONTSIZE
    plt.rcParams["figure.titlesize"] = FONTSIZE

    plt.rcParams["axes.linewidth"]      = 0.8
    plt.rcParams["xtick.major.width"]   = 0.8
    plt.rcParams["ytick.major.width"]   = 0.8
    plt.rcParams["xtick.direction"]     = "in"
    plt.rcParams["ytick.direction"]     = "in"
    plt.rcParams["xtick.top"]           = True
    plt.rcParams["ytick.right"]         = True

    plt.rcParams["legend.frameon"]      = True
    plt.rcParams["legend.edgecolor"]    = "black"
    plt.rcParams["legend.framealpha"]   = 1.0
    plt.rcParams["legend.fancybox"]     = False

    plt.rcParams["lines.linewidth"]     = 1.0
    plt.rcParams["patch.linewidth"]     = 0.8

    plt.rcParams["savefig.bbox"]        = "tight"
    plt.rcParams["savefig.pad_inches"]  = 0.02


def save_paper(fig, path_no_ext: str, also_svg: bool = True,
               also_png: bool = True, png_dpi: int = 200) -> None:
    """Save a figure to PDF (canonical) + optionally SVG (Inkscape) and
    PNG (previewable in a terminal / GH review).  `path_no_ext` is the
    destination path WITHOUT extension."""
    fig.savefig(f"{path_no_ext}.pdf", format="pdf")
    if also_svg:
        fig.savefig(f"{path_no_ext}.svg", format="svg")
    if also_png:
        fig.savefig(f"{path_no_ext}.png", format="png", dpi=png_dpi)
