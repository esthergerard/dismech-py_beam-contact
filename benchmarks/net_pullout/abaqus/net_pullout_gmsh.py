#!/usr/bin/env python3
"""
Gmsh generator for the net pull-out INITIAL GEOMETRY, exported as an
Abaqus .inp with tetrahedral elements.

Reproduces the geometry of `benchmarks/net_pullout/contact_net_pullout.py`
(same constants, SI units / metres):

- 5 "x-beams": solid circular rods of radius r = 1.5 mm swept along
      z(s) = A * sin(4*pi*(s + L_pad)/L),   s in [-L_pad, L + L_pad]
  with L = 40 mm, L_pad = L/8 = 5 mm, A = 3.1 mm, one beam per
  y in {0, 10, 20, 30, 40} mm.  Endpoints at z = 0; the five crossing
  stations x = 0, 10, 20, 30, 40 mm sit at z = +A, -A, +A, -A, +A.
- 5 "y-beams": straight solid rods at z = 0, x in {0, 10, 20, 30, 40} mm,
  y in [-5, 45] mm.  (The 31-um parabolic sagitta of the dismech script
  is a numerical artifact for its arc-fit contact and is dropped here.)
- Initial surface gap at each of the 25 crossings: A - 2r = 0.1 mm.

Output (flat .inp, no parts/assembly):
- *ELEMENT blocks with ELSET=X_BEAM_0..4 / Y_BEAM_0..4  (one per beam)
- *NSET blocks appended for the two end caps of every beam
  (X_BEAM_i_END_NEG / _END_POS, same for Y), for later clamp BCs.

Usage:
    python net_pullout_gmsh.py [--lc 7.5e-4] [--order 2]
                               [--alternating-weave] [--out FILE.inp]
                               [--gui]

order=2 -> C3D10 (recommended for later contact runs), order=1 -> C3D4.
"""

import argparse
import math

import numpy as np
import gmsh

# ---------------------------------------------------------------------------
# Parameters (identical to contact_net_pullout.py, SI units)
# ---------------------------------------------------------------------------
L = 0.04                  # m, crossing-grid span
L_PAD = L / 8.0           # m, overhang past the grid on each side (= 5 mm)
R_BEAM = 1.5e-3           # m, beam radius
WEAVE_AMPLITUDE = 3.1e-3  # m, sine amplitude (gap = A - 2r = 0.1 mm)
N_BEAMS_PER_DIR = 5

N_SPLINE_PTS = 201        # sampling of the sine centerline for the spline

parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
parser.add_argument("--lc", type=float, default=7.5e-4,
                    help="target mesh size in metres (default 0.75 mm, "
                         "~4 elements across the 3 mm diameter)")
parser.add_argument("--order", type=int, default=2, choices=[1, 2],
                    help="element order: 2 -> C3D10 (default), 1 -> C3D4")
parser.add_argument("--alternating-weave", action="store_true",
                    help="phase-shift successive x-beams by pi (plain-weave "
                         "checkerboard); default is the in-phase historical "
                         "pattern")
parser.add_argument("--weave-amplitude", type=float, default=WEAVE_AMPLITUDE,
                    help="sine amplitude of the x-beams in metres (default "
                         "3.1e-3 -> 0.1 mm gap; the alternated cluster runs "
                         "use 3.0e-3 -> zero gap)")
parser.add_argument("--out", type=str, default="net_pullout_initial.inp",
                    help="output .inp path")
parser.add_argument("--gui", action="store_true",
                    help="open the gmsh GUI after meshing")
args = parser.parse_args()
WEAVE_AMPLITUDE = args.weave_amplitude

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
gmsh.initialize()
gmsh.model.add("net_pullout_initial")
occ = gmsh.model.occ

levels = np.linspace(0.0, L, N_BEAMS_PER_DIR)   # crossing stations, 0..40 mm
s0, s1 = -L_PAD, L + L_PAD                      # beam runs over 50 mm total


def make_x_beam(y_pos, amplitude):
    """Solid rod swept along the sine centerline at constant y."""
    s = np.linspace(s0, s1, N_SPLINE_PTS)
    z = amplitude * np.sin(4.0 * np.pi * (s + L_PAD) / L)
    pts = [occ.addPoint(si, y_pos, zi) for si, zi in zip(s, z)]
    spline = occ.addSpline(pts)
    wire = occ.addWire([spline])

    # Profile disk at the start point, normal rotated from +z onto the
    # local tangent t = (1, 0, dz/ds) / |.|
    dzds = amplitude * 4.0 * np.pi / L * math.cos(4.0 * np.pi * (s[0] + L_PAD) / L)
    t = np.array([1.0, 0.0, dzds])
    t /= np.linalg.norm(t)
    disk = occ.addDisk(s[0], y_pos, z[0], R_BEAM, R_BEAM)
    axis = np.cross([0.0, 0.0, 1.0], t)
    angle = math.acos(np.clip(t[2], -1.0, 1.0))
    if np.linalg.norm(axis) > 1e-12:
        occ.rotate([(2, disk)], s[0], y_pos, z[0], *axis, angle)

    out = occ.addPipe([(2, disk)], wire)
    vol = [tag for dim, tag in out if dim == 3][0]
    # The profile disk and the spine are construction helpers; drop them.
    occ.remove([(2, disk)], recursive=True)
    occ.remove([(1, wire)], recursive=True)
    return vol


def make_y_beam(x_pos):
    """Straight solid rod along y at z = 0."""
    return occ.addCylinder(x_pos, s0, 0.0, 0.0, s1 - s0, 0.0, R_BEAM)


x_vols, y_vols = [], []
for i in range(N_BEAMS_PER_DIR):
    amp = WEAVE_AMPLITUDE * ((-1) ** i if args.alternating_weave else 1.0)
    x_vols.append(make_x_beam(levels[i], amp))
for j in range(N_BEAMS_PER_DIR):
    y_vols.append(make_y_beam(levels[j]))

occ.synchronize()

# Drop orphan spline-sample points left over from the sweep construction
# (no upward adjacency = not a vertex of any curve/surface/volume).
orphans = [(0, tag) for _, tag in gmsh.model.getEntities(0)
           if len(gmsh.model.getAdjacencies(0, tag)[0]) == 0]
if orphans:
    gmsh.model.occ.remove(orphans)
    occ.synchronize()

# One physical volume per beam -> one ELSET per beam in the .inp
for i, v in enumerate(x_vols):
    gmsh.model.addPhysicalGroup(3, [v], name=f"X_BEAM_{i}")
for j, v in enumerate(y_vols):
    gmsh.model.addPhysicalGroup(3, [v], name=f"Y_BEAM_{j}")


def end_cap_faces(vol, axis_idx):
    """Return (face_at_min_end, face_at_max_end) of a beam volume, found by
    the along-axis coordinate of each boundary face's center of mass."""
    faces = gmsh.model.getBoundary([(3, vol)], oriented=False)
    lo, hi = None, None
    for dim, tag in faces:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        c = com[axis_idx]
        if abs(c - s0) < 2.0 * R_BEAM:
            lo = tag
        elif abs(c - s1) < 2.0 * R_BEAM:
            hi = tag
    assert lo is not None and hi is not None, f"end caps not found for vol {vol}"
    return lo, hi


cap_faces = {}   # nset name -> face tag
for i, v in enumerate(x_vols):
    lo, hi = end_cap_faces(v, axis_idx=0)
    cap_faces[f"X_BEAM_{i}_END_NEG"] = lo
    cap_faces[f"X_BEAM_{i}_END_POS"] = hi
for j, v in enumerate(y_vols):
    lo, hi = end_cap_faces(v, axis_idx=1)
    cap_faces[f"Y_BEAM_{j}_END_NEG"] = lo
    cap_faces[f"Y_BEAM_{j}_END_POS"] = hi

# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------
gmsh.option.setNumber("Mesh.MeshSizeMax", args.lc)
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.25 * args.lc)
# Resolve the circular cross-section regardless of lc. Set to 24 rather
# than a coarser value: at 12 segments/circle, order=2 curving bulges each
# tube's midside nodes enough that two tubes' curved surfaces can get too
# close at a crossing, producing a negative-Jacobian element the
# HighOrder optimizer cannot untangle (order-1 quality at the same
# geometry is excellent everywhere, min minSICN ~0.26 -- so this is
# purely a curving artifact, not a pre-existing bad linear element).
# Finer circumferential resolution means a smaller per-segment bulge.
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 24)
gmsh.option.setNumber("Mesh.Optimize", 1)
if args.order == 2:
    gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)

gmsh.model.mesh.generate(3)

# Report linear-mesh element quality per beam, before any high-order
# curving is attempted. Runs for --order 1 too (cheap, instant, cannot
# crash) so a marginal/degenerate element can be localized without ever
# reaching the order=2 HighOrder optimizer, which is crash-prone on
# near-tangent crossings (see the MeshSizeFromCurvature comment above).
print("\n--- order-1 element quality report (minSICN), per beam ---")
_all_q = []
for _i, _v in enumerate(x_vols):
    _etypes, _etags, _ = gmsh.model.mesh.getElements(3, _v)
    _tags = np.concatenate(_etags) if _etags else np.array([], dtype=int)
    if len(_tags) == 0:
        continue
    _q = np.array(gmsh.model.mesh.getElementQualities(_tags, "minSICN"))
    _all_q.append(_q)
    print(f"  X_BEAM_{_i}: n={len(_q)}  min={_q.min():.4f}  "
          f"n<0.1={int((_q < 0.1).sum())}  n<=0={int((_q <= 0.0).sum())}")
for _j, _v in enumerate(y_vols):
    _etypes, _etags, _ = gmsh.model.mesh.getElements(3, _v)
    _tags = np.concatenate(_etags) if _etags else np.array([], dtype=int)
    if len(_tags) == 0:
        continue
    _q = np.array(gmsh.model.mesh.getElementQualities(_tags, "minSICN"))
    _all_q.append(_q)
    print(f"  Y_BEAM_{_j}: n={len(_q)}  min={_q.min():.4f}  "
          f"n<0.1={int((_q < 0.1).sum())}  n<=0={int((_q <= 0.0).sum())}")
_all_q = np.concatenate(_all_q)
print(f"  OVERALL: n={len(_all_q)}  min={_all_q.min():.4f}  "
      f"n<0.1={int((_all_q < 0.1).sum())}  n<=0={int((_all_q <= 0.0).sum())}")
print("--- end order-1 quality report ---\n")
# ---- END DEBUG --------------------------------------------------------------


def n_inverted_tets():
    """Count 3D elements with non-positive signed inverse condition number
    (minSICN <= 0 means the element is inverted; Abaqus rejects those)."""
    tags = np.concatenate(gmsh.model.mesh.getElements(3)[1])
    q = np.array(gmsh.model.mesh.getElementQualities(tags, "minSICN"))
    return int((q <= 0.0).sum())


if args.order == 2:
    gmsh.model.mesh.setOrder(2)
    # setOrder alone can leave a few curved tets with negative Jacobians;
    # untangle the high-order nodes until every element is valid.
    for method in ("HighOrderFastCurving", "HighOrderElastic", "HighOrder"):
        nbad = n_inverted_tets()
        if nbad == 0:
            break
        print(f"{nbad} inverted tets remain -> optimizing ({method})")
        gmsh.model.mesh.optimize(method)
    nbad = n_inverted_tets()
    if nbad != 0:
        raise SystemExit(f"ERROR: {nbad} inverted C3D10 elements remain; "
                         f"try a smaller --lc or --order 1.")
    print("Element validity check: all tets have positive Jacobian (minSICN > 0).")

# ---------------------------------------------------------------------------
# Export: gmsh writes *NODE + per-physical-group *ELEMENT (C3D4/C3D10);
# then append the end-cap *NSETs (gmsh's own surface export would emit
# bogus 2D elements into the .inp, so we write the node sets ourselves).
# ---------------------------------------------------------------------------
gmsh.option.setNumber("Mesh.SaveAll", 0)   # only physical groups
gmsh.write(args.out)

# Strip gmsh's *Heading block so this file can be *INCLUDE'd from a model
# .inp (Abaqus allows a single *Heading, as the very first card).
with open(args.out) as f:
    content = f.read()
with open(args.out, "w") as f:
    f.write(content[content.index("*NODE"):])

with open(args.out, "a") as f:
    for name, face in sorted(cap_faces.items()):
        tags, _, _ = gmsh.model.mesh.getNodes(2, face, includeBoundary=True)
        tags = sorted(set(int(t) for t in tags))
        f.write(f"*NSET, NSET={name}\n")
        for k in range(0, len(tags), 16):
            f.write(", ".join(str(t) for t in tags[k:k + 16]) + "\n")

n_nodes = len(gmsh.model.mesh.getNodes()[0])
n_tets = sum(len(t) for ety, t in
             zip(*gmsh.model.mesh.getElements(3)[:2]))
etype = "C3D10" if args.order == 2 else "C3D4"
print(f"\nWrote {args.out}: {n_nodes} nodes, {n_tets} {etype} elements, "
      f"10 beam ELSETs, {len(cap_faces)} end-cap NSETs")
print(f"Units: metres (use SI: E in Pa, density in kg/m^3).")
print(f"Weave: {'ALTERNATING (checkerboard)' if args.alternating_weave else 'in-phase (historical default)'}, "
      f"gap at crossings = {(WEAVE_AMPLITUDE - 2 * R_BEAM) * 1e3:.3f} mm")

if args.gui:
    gmsh.fltk.run()
gmsh.finalize()
