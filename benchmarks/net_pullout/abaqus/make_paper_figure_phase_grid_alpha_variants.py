#!/usr/bin/env python3
"""
Six-panel gallery/topview layout -- (a) Initial, (b) End of engagement,
(c) Central strand released, (d) Left end freed, (e) Central strand
pulled, (f) End of pull-out -- this is Figure 8 of the paper: DisMech
(N=51, three-node model) vs. the quadratic C3D10M, alpha=1000 Abaqus
reference, at the six time instants quoted in the figure caption.

Reads the QUADRATIC mesh instead of the linear one:
  - MESH points at net_pullout_alt_coarse_quad.inp (via whichever alpha
    folder you've copied it into -- same mesh file, shared across all 3
    variants, identical regardless of damping). Node LABELS differ
    completely from the linear mesh (different generation entirely), so
    beam_nodes (node-label -> beam mapping) MUST come from this mesh,
    not the linear one -- reusing the linear mesh's beam_nodes against
    quadratic shape data would silently mismap every node.
  - Panel (b) uses the standard 5-job-chain naming convention
    (net_pullout_phaseA_engage_final_shape.csv).

REQUIRED EXTRACTIONS before this script can run (per alpha folder, on
citran -- all five .odb files are closed/complete for all 3 variants,
so this is safe regardless of job-name restrictions):
    abaqus python extract_final_shape.py net_pullout_phaseA_engage
    abaqus python extract_final_shape.py net_pullout_phaseC_release
    abaqus python extract_shape_near_time.py net_pullout_phaseD_free 0.04
    abaqus python extract_shape_near_time.py net_pullout_phaseE_pullout 0.2
(net_pullout_phaseE_pullout_final_shape.csv for panel (f) already exists
from extract_phaseA_to_E_chained_odb.py's regular output in each folder
-- no new extraction needed for that one, just pull it back if not
already local.)

Writes FigurePhaseGrid_topview_alpha1000.{pdf,svg,png} to paper_figures/.
"""
import pathlib
import re
import sys

import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "figures"))
from paper_style import apply_paper_style, save_paper, COLORS, PAGE_WIDTH  # noqa: E402

apply_paper_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OUT = HERE / "paper_figures"
OUT.mkdir(exist_ok=True)

N = 51
R_BEAM = 1.5e-3
SUB = 3  # Abaqus marker subsampling along each centerline
# Shared quadratic mesh -- same for every alpha variant (damping doesn't
# change the mesh). Pull a copy into any ONE of the alpha folders and
# point this at it (they're byte-identical, just copied around on citran).
MESH = HERE / "quadratic_C3D10M_nodamping" / "net_pullout_alt_coarse_quad.inp"

DISMECH_T_ACTUAL = {
    "a": 0.00,
    "b": 1.00,
    "c": 3.00,
    "d": 3.40,
    "e": 5.50,
    "f": 15.5,
}
# Panel (f) is pinned to t=15.5, the literal end of the simulation, even
# though it shows a visible out-of-plane transient -- scattered Abaqus
# markers off the main centerline, present across all three alpha
# variants (consistent with the noisy/spiky Phase-E signal seen
# elsewhere). An earlier snapshot (t=14.5, already extracted as
# net_pullout_phaseE_pullout_shape_t1.10.csv in each variant folder) is
# visibly cleaner and is the fallback if this panel is revisited.

VARIANTS = [
    ("quadratic_C3D10M_alpha1000", "alpha1000"),
]
# Originally looped over alpha in {0, 100, 1000} as a damping-sensitivity
# check; alpha=1000 is the only variant that appears in the paper (Fig. 8),
# so this repo ships only that data. See
# ../abaqus/quadratic_C3D10M_alpha1000/README.md for why alpha=1000 was
# chosen.

DISMECH_COORDS = (
    HERE.parent / "contact_output_files_net_pullout_alternated" /
    "net_pullout_coords_eberly_N51_alt_soft_eberly_N51_massdamp.txt"
)
# Must be the "_massdamp" tagged file, not "_fast" -- "_fast" predates the
# mass-proportional damping rewrite and is a different run, not a renamed
# copy. "_massdamp" is the current-damping dataset and covers the full
# A-E range for N=51 in one shot (no resume needed).

XLIM = (-12, 85)
YLIM = (-8, 48)


def phases_for(data_dir):
    return [
        ("a", "(a) Initial", None),
        ("b", "(b) End of engagement", data_dir / "net_pullout_phaseA_engage_final_shape.csv"),
        ("c", "(c) Central strand released", data_dir / "net_pullout_phaseC_release_final_shape.csv"),
        ("d", "(d) Left end freed", data_dir / "net_pullout_phaseD_free_shape_t0.04.csv"),
        ("e", "(e) Central strand pulled", data_dir / "net_pullout_phaseE_pullout_shape_t0.2.csv"),
        ("f", "(f) End of pull-out", data_dir / "net_pullout_phaseE_pullout_final_shape.csv"),
    ]


def xcol(i):
    return COLORS["green"] if i == 2 else COLORS["blue"]


def load_mesh_nodes_and_beams():
    text = MESH.read_text()
    elem_nodes = {}
    for blk in re.finditer(r'\*ELEMENT[^\n]*ELSET=Volume\d+\n((?:[^*][^\n]*\n)+)', text):
        for line in blk.group(1).strip().split('\n'):
            p = [int(v) for v in line.rstrip(',').split(',')]
            elem_nodes[p[0]] = p[1:5]
    beam_nodes = {}
    for m in re.finditer(r'\*ELSET,ELSET=([XY]_BEAM_\d)\n((?:[^*][^\n]*\n)+)', text):
        eids = [int(v) for v in re.split(r'[,\s]+', m.group(2).strip()) if v]
        labs = set()
        for e in eids:
            labs.update(elem_nodes[e])
        beam_nodes[m.group(1)] = labs
    node_xyz = {}
    nmatch = re.search(r'\*NODE\n((?:[^*][^\n]*\n)+)', text)
    for line in nmatch.group(1).strip().split('\n'):
        p = line.split(',')
        node_xyz[int(p[0])] = [float(v) for v in p[1:4]]
    return beam_nodes, node_xyz


def centerline(beam_nodes, name, axis, fin_xyz, bw=1.0e-3):
    P = np.array([fin_xyz[l] for l in beam_nodes[name] if l in fin_xyz])
    s = P[:, axis]
    bins = np.arange(s.min(), s.max() + 1e-9, bw)
    idx = np.digitize(s, bins)
    pts = np.array([P[idx == k].mean(axis=0)
                    for k in range(1, len(bins) + 1) if (idx == k).sum() >= 6])
    pts = pts[np.argsort(pts[:, axis])]
    return pts[2:-2]


def beam(q, k):
    return q[k * N:(k + 1) * N]


def draw_panel(ax, q1, fin_xyz, beam_nodes):
    for i in range(5):
        Pd = beam(q1, i)
        Pa = centerline(beam_nodes, f"X_BEAM_{i}", 0, fin_xyz)
        ax.fill_between(Pd[:, 0] * 1e3, (Pd[:, 1] - R_BEAM) * 1e3,
                        (Pd[:, 1] + R_BEAM) * 1e3, color=xcol(i),
                        alpha=0.25, lw=0)
        ax.plot(Pd[:, 0] * 1e3, Pd[:, 1] * 1e3, '-', color=xcol(i), lw=0.9)
        ax.plot(Pa[::SUB, 0] * 1e3, Pa[::SUB, 1] * 1e3, 'o', ms=2.0,
                markerfacecolor='none', markeredgecolor=xcol(i),
                markeredgewidth=0.6, linestyle='none')
    for j in range(5):
        Pd = beam(q1, 5 + j)
        Pa = centerline(beam_nodes, f"Y_BEAM_{j}", 1, fin_xyz)
        ax.fill_betweenx(Pd[:, 1] * 1e3, (Pd[:, 0] - R_BEAM) * 1e3,
                         (Pd[:, 0] + R_BEAM) * 1e3, color=COLORS["red"],
                         alpha=0.25, lw=0)
        ax.plot(Pd[:, 0] * 1e3, Pd[:, 1] * 1e3, '-', color=COLORS["red"],
                lw=0.9)
        ax.plot(Pa[::SUB, 0] * 1e3, Pa[::SUB, 1] * 1e3, 'o', ms=2.0,
                markerfacecolor='none', markeredgecolor=COLORS["red"],
                markeredgewidth=0.6, linestyle='none')


def make_figure(data_dir, out_suffix, beam_nodes, mesh_xyz, snap_at):
    phases = phases_for(data_dir)
    ncols = 3
    nrows = -(-len(phases) // ncols)
    panel_w = PAGE_WIDTH / ncols
    aspect = (YLIM[1] - YLIM[0]) / (XLIM[1] - XLIM[0])
    panel_h = panel_w * aspect
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(PAGE_WIDTH, nrows * panel_h + 0.55),
                              sharex=True, sharey=True)
    axes_flat = axes.flat

    missing = []
    for ax, (key, label, shape_csv) in zip(axes_flat, phases):
        t_dismech = DISMECH_T_ACTUAL[key]
        if shape_csv is None:
            draw_panel(ax, snap_at(t_dismech), mesh_xyz, beam_nodes)
        else:
            if not shape_csv.is_file():
                ax.text(0.5, 0.5, f"{shape_csv.name}\nmissing",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=8, color="0.55", style="italic")
                missing.append(str(shape_csv))
            else:
                d = np.loadtxt(shape_csv, delimiter=',', skiprows=1)
                fin_xyz = {int(r[0]): r[1:4] for r in d}
                draw_panel(ax, snap_at(t_dismech), fin_xyz, beam_nodes)
        ax.text(0.02, 0.95, label, transform=ax.transAxes, va="top",
                ha="left", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8,
                          pad=1.2))
        ax.set_aspect("equal")
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)

    for ax in axes_flat[len(phases):]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$x$ [mm]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$y$ [mm]")

    fig.legend(handles=[
        Line2D([], [], color=COLORS['black'], lw=1.0, label='DisMech'),
        Line2D([], [], marker='o', markerfacecolor='none',
               markeredgecolor=COLORS['black'], markeredgewidth=0.8,
               linestyle='none', ms=4, label='Abaqus')],
        loc='lower right', bbox_to_anchor=(0.98, 0.02), handlelength=1.6,
        frameon=True, edgecolor='black')

    fig.tight_layout()
    out_name = f"FigurePhaseGrid_topview_{out_suffix}"
    save_paper(fig, str(OUT / out_name))
    plt.close(fig)
    print(f"wrote {out_name} (pdf+svg+png) to {OUT}/")
    if missing:
        print(f"  NOTE: panels left blank, missing shape data for: {missing}")


def main():
    if not MESH.is_file():
        raise SystemExit(f"Quadratic mesh not found at {MESH} -- pull it "
                          f"back from citran first (see module docstring).")
    beam_nodes, mesh_xyz = load_mesh_nodes_and_beams()
    raw = np.loadtxt(DISMECH_COORDS, comments="#")
    t_all = raw[:, 0]
    times = np.unique(t_all)

    def snap_at(t):
        s = raw[np.abs(t_all - times[np.argmin(np.abs(times - t))]) < 1e-9]
        return s[np.argsort(s[:, 1])][:, 2:5]

    for folder, suffix in VARIANTS:
        data_dir = HERE / folder
        make_figure(data_dir, suffix, beam_nodes, mesh_xyz, snap_at)


if __name__ == "__main__":
    main()
