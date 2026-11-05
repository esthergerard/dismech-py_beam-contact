#!/usr/bin/env python3
"""
The convergence figure used in the paper: dismech (N=21..131, massdamp,
3-segment stitched for N=71/101/131 via resumeE/resumeE2, full A-E)
against a single Abaqus trace -- the quadratic C3D10M, alpha=1000
damping trial, the configuration settled on after the {1,10,50,100}
sweep (no visible effect) and the energy-based (ALLVD/ALLIE) derivation
that pointed at alpha~1000 as the value where damping starts to be
non-negligible (see ../quadratic_C3D10M_alpha1000/README.md for the
full derivation).

Legend intentionally just says "Abaqus" -- no element-type/damping
qualifiers -- now that this is the one Abaqus configuration being
carried forward rather than one of several exploratory variants.

Full A-E range. Dismech coordinates must come from the "_massdamp"
tagged files (current mass-proportional damping law), not the older
"_fast" tag (pre-damping-fix) -- see
make_paper_figure_phase_grid_alpha_variants.py's DISMECH_COORDS comment
for why those two are easy to confuse.

Writes FigureConvergence_3node_jitter_alpha1000_final.{pdf,svg,png} to
paper_figures/.
"""
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "figures"))
from paper_style import apply_paper_style, COLORS, PAGE_WIDTH  # noqa: E402

apply_paper_style()
import matplotlib.pyplot as plt  # noqa: E402

OUT = HERE / "paper_figures"
OUT.mkdir(exist_ok=True)

EA = 1.0e7 * np.pi * 1.5e-3 ** 2   # soft campaign: E=1e7 Pa, r=1.5mm
I_CENTRAL = 2
ALT_DIR = HERE.parent / "contact_output_files_net_pullout_alternated"
N_LIST = [131, 101, 71, 51, 31, 21]
LINESTYLES = {131: '-', 101: '--', 71: ':', 51: '-.', 31: (0, (3, 1, 1, 1)),
              21: (0, (5, 1))}
PHASE_BOUNDARIES = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 3.5, "E": 15.5}
ABQ_TO_DISMECH = 10.0
T_D = 3.5

# Per-N list of (tag, t_lo_exclusive_or_None, t_hi_inclusive_or_None)
# segments to stitch -- identical to make_paper_figure_convergence_jitter_
# massdamp.py's SEGMENTS dict.
SEGMENTS = {
    21: [("massdamp", None, None)],
    31: [("massdamp", None, None)],
    51: [("massdamp", None, None)],
    71: [("massdamp", None, T_D), ("massdamp_resumeE", T_D, None)],
    101: [("massdamp", None, T_D), ("massdamp_resumeE", T_D, 14.5),
          ("massdamp_resumeE2", 14.5, None)],
    131: [("massdamp", None, T_D), ("massdamp_resumeE", T_D, 12.5),
          ("massdamp_resumeE2", 12.5, None)],
}
RESUMED_N = {n for n, segs in SEGMENTS.items() if len(segs) > 1}


def _load_coords(path):
    if not path.is_file():
        return None
    try:
        return np.loadtxt(path, comments="#")
    except ValueError:
        text = path.read_text().rstrip().rsplit("\n", 1)[0]
        from io import StringIO
        return np.loadtxt(StringIO(text), comments="#")


def _clamp_force_series(arr, xyz0, n_per_beam):
    sa = I_CENTRAL * n_per_beam
    t_all = arr[:, 0]
    times = np.unique(t_all)
    t_out, f_out = [], []
    for t in times:
        rows = arr[t_all == t]
        rows = rows[np.argsort(rows[:, 1])]
        xyz = rows[:, 2:5]
        n0 = sa + 1
        e0, e = xyz0[n0 + 1] - xyz0[n0], xyz[n0 + 1] - xyz[n0]
        fl = abs(EA * (np.linalg.norm(e) / np.linalg.norm(e0) - 1.0))
        n1 = sa + n_per_beam - 3
        e0, e = xyz0[n1 + 1] - xyz0[n1], xyz[n1 + 1] - xyz[n1]
        fr = abs(EA * (np.linalg.norm(e) / np.linalg.norm(e0) - 1.0))
        t_out.append(t)
        f_out.append(0.5 * (fl + fr))
    return np.array(t_out), np.array(f_out)


def dismech_central_clamp_force(n_per_beam):
    segs = SEGMENTS[n_per_beam]
    tag0, _, hi0 = segs[0]
    base = ALT_DIR / (f"net_pullout_coords_eberly_N{n_per_beam}"
                       f"_alt_soft_eberly_N{n_per_beam}_{tag0}.txt")
    arr0 = _load_coords(base)
    if arr0 is None:
        print(f"skipping N={n_per_beam}: no original _{tag0} coords file")
        return None
    xyz0 = arr0[arr0[:, 0] == 0.0]
    xyz0 = xyz0[np.argsort(xyz0[:, 1])][:, 2:5]

    mask0 = arr0[:, 0] <= hi0 + 1e-9 if hi0 is not None else slice(None)
    t_all, f_all = _clamp_force_series(arr0[mask0], xyz0, n_per_beam)

    for tag, lo, hi in segs[1:]:
        path = ALT_DIR / (f"net_pullout_coords_eberly_N{n_per_beam}"
                           f"_alt_soft_eberly_N{n_per_beam}_{tag}.txt")
        arr = _load_coords(path)
        if arr is None:
            print(f"N={n_per_beam}: no _{tag} file yet -- stopping stitch here")
            break
        mask = arr[:, 0] > lo + 1e-9
        if hi is not None:
            mask &= arr[:, 0] <= hi + 1e-9
        t_seg, f_seg = _clamp_force_series(arr[mask], xyz0, n_per_beam)
        t_all = np.concatenate([t_all, t_seg])
        f_all = np.concatenate([f_all, f_seg])
    return t_all, f_all


# ---------------- single Abaqus trace: quadratic, alpha=1000 -------------
abq_csv = HERE / "quadratic_C3D10M_alpha1000" / "net_pullout_phaseE_pullout_central_clamp.csv"
cc = np.genfromtxt(abq_csv, delimiter=",", names=True, dtype=None, encoding="utf-8")
t_abq = cc["t_total"].astype(float) * ABQ_TO_DISMECH
f_abq = 0.5 * (np.abs(cc["F_left"].astype(float))
               + np.abs(cc["F_right"].astype(float)))
print(f"Abaqus (quadratic C3D10M, alpha=1000): {len(t_abq)} samples, "
      f"t up to {t_abq[-1]:.3f}s, peak |F|={f_abq.max():.3f}N")

# ---------------- compute all dismech series once -------------------------
dismech_series = {}
for n in N_LIST:
    res = dismech_central_clamp_force(n)
    if res is None:
        continue
    t, f = res
    note = ""
    if n in RESUMED_N and t[-1] < 15.5 - 1e-6:
        note = "  (resume still in progress -- plotted to current data end)"
    print(f"N={n}: last snapshot t={t[-1]:.3f}s{note}")
    dismech_series[n] = (t, f)

# ---------------- figure -----------------------------------------------
fig, ax = plt.subplots(figsize=(PAGE_WIDTH, PAGE_WIDTH * 0.42))

ax.plot(t_abq, f_abq, '-', color=COLORS["black"], lw=0.4, alpha=0.7,
        label='Abaqus', zorder=1)

for n in N_LIST:
    if n not in dismech_series:
        continue
    t, f = dismech_series[n]
    ax.plot(t, f, color=COLORS["green"], ls=LINESTYLES[n], lw=1.1,
            label=f"3-node (ours), $N={n}$", zorder=3)

for tb in PHASE_BOUNDARIES.values():
    ax.axvline(tb, color="0.75", lw=0.6, ls=":", zorder=0)
phase_starts = [0.0] + list(PHASE_BOUNDARIES.values())[:-1]
for (name, tb), t0 in zip(PHASE_BOUNDARIES.items(), phase_starts):
    ax.annotate(name, (0.5 * (t0 + tb), 1.0), xycoords=("data", "axes fraction"),
                xytext=(0, -10), textcoords="offset points", fontsize=8,
                color="0.3", ha="center")

ax.set_xlabel(r"time $t$ [s]")
ax.set_ylabel(r"central-beam clamp force $|F|$ [N]")
ax.set_xlim(0, 15.5)
ax.set_ylim(bottom=0)
ax.legend(loc='upper right', fontsize=7, handlelength=2.2, ncol=1)

fig.tight_layout()
out_base = OUT / "FigureConvergence_3node_jitter_alpha1000_final"
fig.savefig(f"{out_base}.pdf", format="pdf", bbox_inches="tight")
fig.savefig(f"{out_base}.svg", format="svg", bbox_inches="tight")
fig.savefig(f"{out_base}.png", format="png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\nwrote FigureConvergence_3node_jitter_alpha1000_final "
      f"(pdf+svg+png) to {OUT}/")
