#!/usr/bin/env python3
"""
Figure 2 -- normalised force vs normalised displacement.

Three curves at N = 50 beam nodes:

  - 2-node IMC          -> external_reference_data/impV_noD_contact_bending_N50.npz
  - 3-node Eberly (ours)-> external_reference_data/reg_*N50.txt
  - Analytical          -> external_reference_data/exact_rL_0p125.csv (divided
                           by 2, matching the symmetry factor used to derive it)

See external_reference_data/README.md for the provenance of each file and
how to regenerate the regenerable ones.

Normalisation uses L = 1.6 to match the reference solution (r/L = 0.125, which
is the reference value the analytical curve is computed for; the simulation
data lives on the same r/L curve once L = 1.6 is used).  Change L below if
you re-run the simulation at a different beam length.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import (  # noqa: E402
    DEFAULT_FIG_SIZE,
    LAB_COLORS,
    apply_style,
    load_coords_file,
    load_forces_file,
    load_npz_run,
    save_figure,
    setup_axes,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

BENCH_DIR  = Path(__file__).resolve().parents[1]
DATA_DIR   = BENCH_DIR / "external_reference_data"
BEAD_DIR   = BENCH_DIR / "contact_output_files_3D_bead"

# Beam length used to non-dimensionalise the simulation.
L = 1.6           # [m]   (matches MATLAB post_process_txtfile.m)
R = 0.001         # [m]   rod radius (used only for E I -- not for L)
E_YOUNG = 210e8   # [Pa]
I_AREA = np.pi * R**4 / 4.0

# Sim runs are drawn in the order listed below: the FIRST entry is at the
# back, the LAST entry on top.  Order them 1-node -> 2-nodes -> 3-nodes
# so that the 3-node curve sits on top, the 2-node curve below it, and
# the 1-node curve at the bottom of the stack.
#
# kind: 'txt' expects (coords_file, forces_file).
#       'npz' expects (path,).
# Optional per-run "delta_l_max": truncate the curve at the given δ/L
# (used for the 1-node bead model whose force collapses near δ/L = 0.57).
SIM_RUNS = [
    {
        "label": "1-node",
        "kind": "txt",
        "coords": BEAD_DIR / "reg_beam_coords_N50_bead.txt",
        "forces": BEAD_DIR / "reg_contact_forces_N50_bead.txt",
        "style": {"color": LAB_COLORS[2], "linestyle": "-",
                  "linewidth": 1.2},
        "delta_l_max": 0.57,
    },
    {
        "label": "2-node",
        "kind": "npz",
        "path": DATA_DIR / "impV_noD_contact_bending_N50.npz",
        "style": {"color": LAB_COLORS[0], "linestyle": "-",
                  "linewidth": 1.2},
    },
    {
        "label": "3-node (ours)",
        "kind": "txt",
        "coords": DATA_DIR / "reg_beam_coords_N50.txt",
        "forces": DATA_DIR / "reg_contact_forces_N50.txt",
        "style": {"color": LAB_COLORS[1], "linestyle": "-",
                  "linewidth": 1.2},
    },
]

ANALYTICAL_CSV    = DATA_DIR / "exact_rL_0p125.csv"
ANALYTICAL_FACTOR = 0.5    # reference solution plots T.FL2_EI/2
ANALYTICAL_STYLE  = {"color": LAB_COLORS[4], "linestyle": "--",
                     "linewidth": 1.0}

OUT_DIR = Path(__file__).parent / "figures"
OUT_BASENAME = "fig2_force_displacement"
INCLUDE_LABELS = True


# ============================================================================
# CURVE EXTRACTION
# ============================================================================

def _normalise(delta: np.ndarray, Fy_sum: np.ndarray):
    """Shift to loading onset, return (delta/L, F L^2 / (E I))."""
    if Fy_sum.size == 0 or Fy_sum.max() <= 0:
        return np.array([]), np.array([])
    tol = 1e-6 * Fy_sum.max()
    onset = int(np.argmax(Fy_sum > tol)) if np.any(Fy_sum > tol) else 0
    Fy_sum = Fy_sum[onset:] - Fy_sum[onset]
    delta = delta[onset:] - delta[onset]
    return delta / L, Fy_sum * L**2 / (E_YOUNG * I_AREA)


def extract_from_txt(coords_file: Path, forces_file: Path):
    """Return (delta/L, F L^2 / (E I)) from the .txt output pair."""
    coords = load_coords_file(coords_file)
    forces = load_forces_file(forces_file)

    # leftmost beam-node deflection
    ref_node = int(coords[:, 1].min())
    mask = coords[:, 1].astype(int) == ref_node
    t_coords = coords[mask, 0]
    y_coords = coords[mask, 3]
    order = np.argsort(t_coords)
    t_coords, y_coords = t_coords[order], y_coords[order]
    _, uniq = np.unique(t_coords, return_index=True)
    uniq.sort()
    t_coords, y_coords = t_coords[uniq], y_coords[uniq]
    delta = y_coords - y_coords[0]

    # total |F_y| over beam nodes (filtering zero-force rows)
    keep = ~((forces[:, 2] == 0.0) & (forces[:, 3] == 0.0)
             & (forces[:, 4] == 0.0))
    t_force = forces[keep, 0]
    fy = forces[keep, 3]
    unique_times, inv = np.unique(t_force, return_inverse=True)
    Fy_sum = np.abs(np.bincount(inv, weights=fy,
                                 minlength=unique_times.size))

    delta_interp = np.interp(unique_times, t_coords, delta,
                             left=np.nan, right=np.nan)
    ok = ~np.isnan(delta_interp)
    return _normalise(delta_interp[ok], Fy_sum[ok])


def extract_from_npz(npz_file: Path):
    """Return (delta/L, F L^2 / (E I)) from a contact-bending .npz dump."""
    run = load_npz_run(npz_file)
    # leftmost beam-node deflection (y)
    y0 = run["beam_coords"][:, 0, 1]
    delta = y0 - y0[0]
    Fy_sum = np.abs(np.sum(run["beam_forces"][:, :, 1], axis=1))
    return _normalise(delta, Fy_sum)


def read_analytical_csv(path: Path):
    """Return (eta, FL2_EI) from a 2-column header CSV."""
    with open(path, "r", newline="") as fh:
        sniff = fh.read(1024)
        fh.seek(0)
        dialect = csv.Sniffer().sniff(sniff, delimiters=",;\t ")
        reader = csv.DictReader(fh, dialect=dialect)
        eta, fL2 = [], []
        for row in reader:
            keys_lower = {k.lower(): k for k in row}
            eta_key = next((keys_lower[k] for k in ("eta", "delta_l",
                            "delta/l", "dl", "x") if k in keys_lower), None)
            fl2_key = next((keys_lower[k] for k in ("fl2_ei", "fl^2/ei",
                            "fl2", "y") if k in keys_lower), None)
            if eta_key is None or fl2_key is None:
                continue
            try:
                eta.append(float(row[eta_key]))
                fL2.append(float(row[fl2_key]))
            except (TypeError, ValueError):
                continue
    return np.asarray(eta), np.asarray(fL2)


# ============================================================================
# PLOT
# ============================================================================

def plot_force_displacement(
    sim_runs=SIM_RUNS,
    analytical_csv: Path | None = ANALYTICAL_CSV,
    analytical_factor: float = ANALYTICAL_FACTOR,
    out_dir: Path = OUT_DIR,
    out_basename: str = OUT_BASENAME,
    include_labels: bool = INCLUDE_LABELS,
    xlim: tuple | None = None,
) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    handles, labels = [], []
    # Track in-view (xlim) min/max across all curves so we can autoscale y
    # tightly to the visible window when xlim is set.
    y_in_view = []

    # --- analytical reference (drawn first, sim curves sit on top) ---------
    if analytical_csv is not None and Path(analytical_csv).is_file():
        eta, fL2 = read_analytical_csv(analytical_csv)
        if eta.size:
            line, = ax.plot(eta, fL2 * analytical_factor, **ANALYTICAL_STYLE)
            handles.append(line)
            labels.append("Analytical")
            if xlim is not None:
                m = (eta >= xlim[0]) & (eta <= xlim[1])
                if m.any():
                    y_in_view.append(fL2[m] * analytical_factor)
    elif analytical_csv is not None:
        print(f"  [warn] analytical CSV not found: {analytical_csv}")

    # --- simulation curves -------------------------------------------------
    for run in sim_runs:
        label = run["label"]
        kind = run["kind"]
        try:
            if kind == "txt":
                dL, FL2 = extract_from_txt(Path(run["coords"]),
                                            Path(run["forces"]))
            elif kind == "npz":
                dL, FL2 = extract_from_npz(Path(run["path"]))
            else:
                raise ValueError(f"unknown kind {kind!r}")
        except FileNotFoundError as e:
            print(f"  [warn] {label!r}: missing file {e}; skipping")
            continue
        if dL.size == 0:
            print(f"  [warn] {label!r}: no loading detected; skipping")
            continue
        # Optional per-run truncation at a maximum delta/L (e.g. the
        # 1-node bead curve collapses after delta/L ~= 0.57).
        dl_max = run.get("delta_l_max")
        if dl_max is not None:
            keep = dL <= dl_max
            dL, FL2 = dL[keep], FL2[keep]
        line, = ax.plot(dL, FL2, **run["style"])
        handles.append(line)
        labels.append(label)
        print(f"  {label:<20s} peak FL^2/(EI) = {FL2.max():.4f}")
        if xlim is not None:
            m = (dL >= xlim[0]) & (dL <= xlim[1])
            if m.any():
                y_in_view.append(FL2[m])

    setup_axes(ax)
    if xlim is not None:
        ax.set_xlim(*xlim)
        if y_in_view:
            y_concat = np.concatenate(y_in_view)
            lo, hi = float(y_concat.min()), float(y_concat.max())
            pad = 0.05 * (hi - lo) if hi > lo else 0.05 * abs(hi)
            ax.set_ylim(lo - pad, hi + pad)

    if include_labels:
        ax.set_xlabel(r"Normalized displacement, $\delta / L$")
        ax.set_ylabel(r"Normalized force, $F L^{2} / (E I)$")
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    if include_labels and len(handles) > 1:
        leg = ax.legend(handles, labels, loc="lower right", frameon=True)
        leg.get_frame().set_edgecolor("black")
        leg.get_frame().set_linewidth(0.6)

    fig.tight_layout()
    save_figure(fig, out_dir, out_basename)
    plt.close(fig)


if __name__ == "__main__":
    plot_force_displacement()
    # Companion zoom on the peak region.
    plot_force_displacement(out_basename="fig2b_force_displacement_zoom",
                            xlim=(0.3, 0.4))
