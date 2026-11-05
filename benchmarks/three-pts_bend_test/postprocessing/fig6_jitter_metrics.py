#!/usr/bin/env python3
"""
Jitter metrics for the 1-node / 2-node / 3-node N=50 datasets.

Python port of ``Post_Process_July24_Jitter_old.m`` (the ``analyze_jitter``
inner function) applied to the same N=50 files used by fig 2.

Two tables are computed for each model:

  Table 1 -- jitter on the FORCE-vs-TIME signal (Σ F_y vs t).
  Table 2 -- jitter on the NORMALIZED FORCE vs NORMALIZED DISPLACEMENT
             curve  (F L^2/(E I)  vs  δ/L), after resampling onto a
             uniform δ/L grid.

For each table the metrics are:

  - Max P2P        max |dF| between consecutive samples
  - Mean P2P       mean |dF| between consecutive samples
  - 3-Pt Max       largest "non-monotonic 3-point" perpendicular error
  - 3-Pt Mean      average of the same
  - RMS vs Smooth  RMS of (F - movmean(F, SMOOTHING_WINDOW))
  - STD vs Smooth  std of (F - movmean(F, SMOOTHING_WINDOW))    [time only]
  - RMS vs Truth   RMS of (F - linear ground-truth)             [disp only]

The smoothing window is given in SECONDS in the time-domain table and in
δ/L UNITS in the displacement-domain table, so the metric is dt-/sample-
independent.

Output: ``figures/fig6_jitter_metrics.txt``  +  printed to stdout.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import (  # noqa: E402
    load_coords_file,
    load_forces_file,
    load_npz_run,
)
from fig2_force_displacement import (  # noqa: E402
    SIM_RUNS,
    L,
    E_YOUNG,
    I_AREA,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Smoothing window expressed in PHYSICAL units (not in number of samples)
# so dt does not affect the metric.  Matches MATLAB's 20000 samples at
# dt=0.0001 s -> 2 seconds.
TIME_SMOOTH_WINDOW_S      = 2.0       # seconds (table 1)
DISP_SMOOTH_WINDOW_DL     = 0.02      # δ/L  units (table 2)

# Forces below this are treated as "contact not yet engaged".
FORCE_START_THRESHOLD     = 1e-6      # N

# Linear ground-truth slope for the post-contact F-δ curve, in our
# normalisation (slope = 1 in MATLAB's "F L^2 / (48 E I)" units becomes
# slope = 48 in our "F L^2 / (E I)" units).
POST_CONTACT_SLOPE_OURS   = 48.0

# Number of points retained on the resampled uniform δ/L grid for table 2.
DISP_RESAMPLE_N           = 20000

OUT_DIR  = Path(__file__).parent / "figures"
OUT_TXT  = OUT_DIR / "fig6_jitter_metrics.txt"


# ============================================================================
# CORE: analyze_jitter
# ============================================================================

def _movmean_centered(y: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average matching MATLAB's `movmean` (truncated at edges)."""
    if window <= 1:
        return y.copy()
    n = y.size
    csum = np.concatenate([[0.0], np.cumsum(y)])
    half = window // 2
    out = np.empty(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + (window - half))
        out[i] = (csum[hi] - csum[lo]) / (hi - lo)
    return out


def analyze_jitter(signal_y: np.ndarray, signal_x: np.ndarray,
                   smoothing_window: int) -> Dict[str, float]:
    """
    Direct port of the MATLAB ``analyze_jitter`` local function.

    Parameters
    ----------
    signal_y : (n,) the signal to analyse.
    signal_x : (n,) the abscissa (time or δ/L).
    smoothing_window : window size in SAMPLES (centered movmean).
    """
    if signal_y.size < 3:
        return {k: float("nan") for k in (
            "max_p2p", "mean_p2p", "three_point_max", "three_point_mean",
            "rms_vs_smooth", "std_vs_smooth",
            "max_p2p_x", "three_point_max_x",
        )}

    # --- Max / Mean point-to-point ---
    diffs = np.abs(np.diff(signal_y))
    p2p_idx = int(np.argmax(diffs))
    max_p2p = float(diffs[p2p_idx])
    max_p2p_x = float(signal_x[p2p_idx + 1])
    mean_p2p = float(diffs.mean())

    # --- 3-point non-monotonic perpendicular error ---
    y0 = signal_y[:-2]
    y1 = signal_y[1:-1]
    y2 = signal_y[2:]
    x0 = signal_x[:-2]
    x1 = signal_x[1:-1]
    x2 = signal_x[2:]
    non_mono = (y1 - y0) * (y2 - y1) < 0          # MATLAB uses >= 0 = monotonic
    valid_dx = np.abs(x2 - x0) > np.finfo(float).eps
    mask = non_mono & valid_dx
    if mask.any():
        y_on_line = y0[mask] + (y2[mask] - y0[mask]) * \
            (x1[mask] - x0[mask]) / (x2[mask] - x0[mask])
        errs = np.abs(y1[mask] - y_on_line)
        three_point_max = float(errs.max())
        three_point_mean = float(errs.mean())
        three_point_max_x = float(x1[mask][int(np.argmax(errs))])
    else:
        three_point_max = 0.0
        three_point_mean = 0.0
        three_point_max_x = float("nan")

    # --- Smoothing-based metrics ---
    if signal_y.size < smoothing_window:
        rms_vs_smooth = float("nan")
        std_vs_smooth = float("nan")
    else:
        smoothed = _movmean_centered(signal_y, smoothing_window)
        err = signal_y - smoothed
        rms_vs_smooth = float(np.sqrt(np.mean(err**2)))
        std_vs_smooth = float(np.std(err, ddof=0))

    return dict(
        max_p2p=max_p2p, mean_p2p=mean_p2p,
        three_point_max=three_point_max,
        three_point_mean=three_point_mean,
        rms_vs_smooth=rms_vs_smooth,
        std_vs_smooth=std_vs_smooth,
        max_p2p_x=max_p2p_x,
        three_point_max_x=three_point_max_x,
    )


# ============================================================================
# DATA LOADERS  (return t, y0_leftmost, Fysum)
# ============================================================================

def _from_txt(coords_file: Path, forces_file: Path):
    coords = load_coords_file(coords_file)
    forces = load_forces_file(forces_file)
    ref = int(coords[:, 1].min())
    m_c = coords[:, 1].astype(int) == ref
    t_c = coords[m_c, 0]
    y_c = coords[m_c, 3]
    order = np.argsort(t_c)
    t_c = t_c[order]
    y_c = y_c[order]
    t_c, idx = np.unique(t_c, return_index=True)
    idx.sort()
    y_c = y_c[idx]

    t_f = forces[:, 0]
    fy = forces[:, 3]
    t_u, inv = np.unique(t_f, return_inverse=True)
    Fy_sum = np.bincount(inv, weights=fy, minlength=t_u.size)

    # Align on the force time grid (which is also the saved grid).
    y0_on_force = np.interp(t_u, t_c, y_c, left=np.nan, right=np.nan)
    return t_u, y0_on_force, Fy_sum


def _from_npz(npz_file: Path):
    run = load_npz_run(npz_file)
    t = run["time"]
    y0 = run["beam_coords"][:, 0, 1]
    Fy_sum = np.sum(run["beam_forces"][:, :, 1], axis=1)
    return t, y0, Fy_sum


def _load_runs():
    out = []
    for run in SIM_RUNS:
        try:
            if run["kind"] == "txt":
                t, y0, Fy = _from_txt(Path(run["coords"]), Path(run["forces"]))
            else:
                t, y0, Fy = _from_npz(Path(run["path"]))
        except FileNotFoundError as e:
            print(f"  [warn] missing file for {run['label']!r}: {e}")
            continue
        out.append({"label": run["label"], "t": t, "y0": y0,
                    "Fysum": Fy,
                    "delta_l_max": run.get("delta_l_max")})
    return out


# ============================================================================
# PER-RUN METRIC COMPUTATIONS
# ============================================================================

def _metrics_force_vs_time(t: np.ndarray, Fy: np.ndarray):
    mask = ~np.isnan(Fy)
    t_clean = t[mask]
    Fy_clean = Fy[mask]
    start = np.argmax(np.abs(Fy_clean) > FORCE_START_THRESHOLD) \
        if np.any(np.abs(Fy_clean) > FORCE_START_THRESHOLD) else None
    if start is None or (Fy_clean.size - start) < 3:
        return None, None
    F = np.abs(Fy_clean[start:])
    T = t_clean[start:]
    if T.size < 2:
        return None, None
    dt = float(np.median(np.diff(T)))
    window = max(3, int(round(TIME_SMOOTH_WINDOW_S / max(dt, 1e-12))))
    return analyze_jitter(F, T, window), (T, F)


def _metrics_force_vs_disp(t: np.ndarray, y0: np.ndarray, Fy: np.ndarray,
                            delta_l_max: float | None):
    valid = ~np.isnan(y0) & ~np.isnan(Fy)
    y0_clean = y0[valid]
    Fy_clean = Fy[valid]
    if y0_clean.size < 3:
        return None, None
    y0_init = y0_clean[0]
    norm_disp  = (y0_clean - y0_init) / L
    norm_force = np.abs(Fy_clean) * L**2 / (E_YOUNG * I_AREA)

    # Resample onto a uniform δ/L grid that spans the data.
    d_uniform = np.linspace(norm_disp.min(), norm_disp.max(), DISP_RESAMPLE_N)
    d_unique, idx = np.unique(norm_disp, return_index=True)
    idx.sort()
    F_unique = norm_force[idx]
    F_on_d = np.interp(d_uniform, d_unique, F_unique,
                       left=np.nan, right=np.nan)
    ok = ~np.isnan(F_on_d)
    F_on_d = F_on_d[ok]; d_uniform = d_uniform[ok]

    start = np.argmax(F_on_d > FORCE_START_THRESHOLD) \
        if np.any(F_on_d > FORCE_START_THRESHOLD) else None
    if start is None or (F_on_d.size - start) < 3:
        return None, None
    F = F_on_d[start:]
    D = d_uniform[start:]
    if delta_l_max is not None:
        keep = D <= delta_l_max
        F, D = F[keep], D[keep]
        if F.size < 3:
            return None, None
    dd = float(np.median(np.diff(D)))
    window = max(3, int(round(DISP_SMOOTH_WINDOW_DL / max(dd, 1e-12))))

    metrics = analyze_jitter(F, D, window)

    # RMS vs linear ground truth over the first GROUND_TRUTH window in δ/L
    # (= MATLAB's 0.1 fixed window).  We use δ/L = 0.1 like MATLAB.
    GT_DISP_WINDOW = 0.1
    end_target = D[0] + GT_DISP_WINDOW
    j = int(np.searchsorted(D, end_target))
    if j < 3:
        rms_vs_truth = float("nan")
    else:
        D_crop = D[:j]
        F_crop = F[:j]
        gt = F_crop[0] + POST_CONTACT_SLOPE_OURS * (D_crop - D_crop[0])
        err = F_crop - gt
        rms_vs_truth = float(np.sqrt(np.mean(err**2)))
    metrics["rms_vs_truth"] = rms_vs_truth
    return metrics, (D, F)


# ============================================================================
# TABLE FORMATTING
# ============================================================================

def _fmt(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "      NaN"
    if abs(x) >= 1e3 or (0 < abs(x) < 1e-3):
        return f"{x: .3e}"
    return f"{x: .4f}"


def _print_table(title, header, rows, out_lines):
    line = f"\n{title}\n"
    print(line, end="")
    out_lines.append(line)

    col_widths = [max(len(str(h)), 11) for h in header]
    # Make row labels (first column) wider.
    col_widths[0] = max(col_widths[0], max(len(str(r[0])) for r in rows))

    def fmt_row(values):
        return "  ".join(
            f"{str(v):>{w}}" for v, w in zip(values, col_widths)
        )

    head = fmt_row(header)
    sep  = "-" * len(head)
    for s in (head, sep):
        print(s); out_lines.append(s + "\n")

    for row in rows:
        cells = [row[0]] + [_fmt(v) for v in row[1:]]
        s = fmt_row(cells)
        print(s); out_lines.append(s + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    runs = _load_runs()
    if not runs:
        raise SystemExit("no data files found")

    header_time = (
        "Model", "Max P2P", "Mean P2P", "3-Pt Max", "3-Pt Mean",
        "RMS Smooth", "STD Smooth", "Max P2P t [s]", "3-Pt Max t [s]",
    )
    header_disp = (
        "Model", "Max P2P", "Mean P2P", "3-Pt Max", "3-Pt Mean",
        "RMS Smooth", "RMS Truth", "Max P2P δ/L", "3-Pt Max δ/L",
    )

    rows_time, rows_disp = [], []
    out_lines = []

    for r in runs:
        mt, _ = _metrics_force_vs_time(r["t"], r["Fysum"])
        if mt is None:
            rows_time.append((r["label"],) + ("NaN",) * (len(header_time) - 1))
        else:
            rows_time.append((
                r["label"], mt["max_p2p"], mt["mean_p2p"],
                mt["three_point_max"], mt["three_point_mean"],
                mt["rms_vs_smooth"], mt["std_vs_smooth"],
                mt["max_p2p_x"], mt["three_point_max_x"],
            ))

        md, _ = _metrics_force_vs_disp(r["t"], r["y0"], r["Fysum"],
                                        r.get("delta_l_max"))
        if md is None:
            rows_disp.append((r["label"],) + ("NaN",) * (len(header_disp) - 1))
        else:
            rows_disp.append((
                r["label"], md["max_p2p"], md["mean_p2p"],
                md["three_point_max"], md["three_point_mean"],
                md["rms_vs_smooth"], md["rms_vs_truth"],
                md["max_p2p_x"], md["three_point_max_x"],
            ))

    _print_table(
        f"Table 1: Jitter on  |Σ F_y| vs t   "
        f"(smoothing window = {TIME_SMOOTH_WINDOW_S} s)",
        header_time, rows_time, out_lines)
    _print_table(
        f"Table 2: Jitter on  F L²/(E I) vs δ/L  "
        f"(smoothing window = {DISP_SMOOTH_WINDOW_DL} δ/L, "
        f"L = {L} m)",
        header_disp, rows_disp, out_lines)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("".join(out_lines))
    print(f"\nWrote {OUT_TXT}")


if __name__ == "__main__":
    main()
