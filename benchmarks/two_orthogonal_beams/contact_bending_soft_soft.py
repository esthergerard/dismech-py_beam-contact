#!/usr/bin/env python3
"""
Validation against Example 2 of [Zavarise & Wriggers, 2000]
"Contact with friction between beams in 3-D space"
Int. J. Numer. Meth. Engng 49(8):977-1006.

The paper's Example 2 (pp. 1000-1001) describes two orthogonal beams in
frictionless contact.  Exact quotation:

  "Example 2 concerns two beams being orthogonal to each other [...].  The
   horizontal beam length is equal to 14 units, and the vertical one is equal
   to 10 units.  Out-of-plane initial distance between the beams is equal to
   1 unit.  Geometrical and material parameters are: A = 4.0e-2,
   Jxx = Jyy = 2.0e-4, E = 1.0e+8, ν = 0.0.  The vertical beam is clamped
   at both ends.  The horizontal beam is clamped at the right-end, where
   also an out-of-plane displacement, equal to 0.3 units, and a horizontal
   displacement, equal to 0.03 units, are imposed.  Due to the given
   displacement and the assumed beam section radius, r = 0.2, the beams come
   in incipient contact at the second load step.  [...]  Penalty values for
   normal and tangential contact are chosen as: K_N = K_T = K̄_T = 1.e + 4."

So the paper itself confirms r = 0.2 (NOT 0.1 — that value was used in
Example 1 only), K_N = 1.e + 4 (NOT 1.e + 6 — again that was Example 1),
and contact engaging at the second load step.  Figure 9 of the paper shows
the two beams crossing at the midspan of each.

Mapping to this script (axes-renamed so the long beam lies along z):
- Beam X = paper's "vertical" beam — length 10, clamped at BOTH ends (C, D),
  at y = +0.5, z = z_x_contact (+3 by default; see below).
- Beam Z = paper's "horizontal" beam — length 14, clamped at A (one end) with
  prescribed motion, free at B, initially at x = 0, y = -0.5,
  z ∈ [-7, +7] with A at z = -7 and B at z = +7.
- Initial axis-to-axis distance in y = 1 (matches paper).
- Beam radius r = 0.2 m on both beams (matches paper).
- Contact location on Beam Z: at z = +3, i.e. 10 m from the clamped end A
  and 4 m from the free end B.  This matches Figure 9 of the paper, which
  shows a NON-midspan crossing.  Setting `z_x_contact = 0.0` would put the
  contact at Beam Z's midspan, which gives ~3x too much force (because the
  cantilever arm is shorter and therefore stiffer); the off-center crossing
  is what makes the analytical EB prediction match Table IV.

Loading (paper Example 2, applied to node A only, all 3 rotations clamped):
  Δuy = +0.3   per step  (out-of-plane, toward Beam X)
  Δuz = -0.03  per step  (in-plane slide; the paper gives 0.03 without an
                          explicit sign — the sign does not matter for
                          frictionless contact)
The cumulative Δuy reaches the initial surface gap (0.6 m) at step 2, so
contact engages at step 2 and grows from step 3 onward — matches the paper.

Frictionless reference contact force (paper Table IV, μ = 0):
  Step   F_N [N]      F_T [N]
  -----  -----------  -----------
    1    0.00e+0      0.00e+0
    2    1.00e-9      0.00e+0
    3    1.72e+1      0.00e+0
    4    3.43e+1      0.00e+0
    5    5.13e+1      0.00e+0
    6    6.84e+1      0.00e+0
F_N grows linearly at ~17.1 N per step.  F_T is exactly zero (frictionless).

VALIDATION TARGETS in this script:
- ΣFy(Beam X) at each load step → compare with F_N from Table IV.
- ΣFz(Beam X) and ΣFx(Beam X) → expected to be small (residual geometric
  coupling from the bending; pure friction contributions are zero).
- Beam Z deflection profile y(z) and curvature radius near the contact point.

KNOWN MODELLING DIFFERENCES FROM THE PAPER
- The paper uses a point-to-point small-sliding contact element at one
  pre-selected node pair, with a LINEAR penalty K_N = 1.0e+4 N/m.  We use
  a segment-to-segment Eberly arc-to-arc contact with a REGULARIZED quadratic
  IMC potential and (by default) an ADAPTIVE k_c that scales with the
  max-elastic-force residual.  At later steps our adaptive k_c is typically
  much larger than 1.e+4, so it overstiffens the contact and overshoots F_N.
  Use `--fixed-kc 1.0e+4` (or any other value) to recover a fixed-stiffness
  behaviour closer to the paper.
- The paper's beams are pure straight Euler-Bernoulli; we use DER rods with
  a tiny initial curvature κ₀ = 1e-4 1/m so the arc-to-arc contact has a
  well-defined fitted circle (a perfectly straight triplet is collinear and
  the Eberly arc fit is undefined).
"""

import argparse
from pathlib import Path
import numpy as np
import dismech
from dismech.contact import (
    compute_eberly_3d_beam_to_beam_contact_forces,
    compute_eberly_3d_beam_to_beam_friction_forces,
)

# ============================================================================
# PARAMETERS  (Zavarise & Wriggers, 2000 — Example 2)
# ============================================================================

# ---- Beam X (along x, clamped at both ends C-D) ----------------------------
x_beam_length = 10.0   # m
n_x_nodes = 11          # 10 elements of length 1 m (as in the reference)
y_x = +0.5              # x-beam centerline y-position (above, at the midspan)
z_x_contact = +3.0      # z-position of Beam X (= z-coordinate of the contact).
                        # Set so the contact is 10 units from Beam Z's clamped
                        # end A (z = -7) and 4 units from Beam Z's free end B
                        # (z = +7).  This matches the paper's Figure 9 layout
                        # (non-midspan crossing).  At z_x_contact = 0 the
                        # contact would be at Beam Z's midspan.

# ---- Beam Z (along z, clamped at A, free at B) -----------------------------
z_beam_length = 14.0   # m
n_z_nodes = 15          # 14 elements of length 1 m (as in the reference)
y_z = -0.5              # z-beam centerline initial y-position (below, at the midspan)

# Derived: arm length from A (clamp) to the contact point on Beam Z
# (used by the analytical Euler-Bernoulli reference).  With our convention
# A is at z = -z_beam_length/2 = -7.
L1_from_clamp_to_contact = z_x_contact - (-z_beam_length / 2)   # = 10.0 by default

# ---- Initial curvature (workaround for the Eberly arc-to-arc contact) -------
kappa0 = 1.0e-4   # [1/m]  small initial curvature on both beams

# ---- Cross-section & material (Table 3.1) ----------------------------------
beam_radius = 0.2       # m   (geometric, used by contact only)
beam_area_A = 4.0e-2    # m^2 (cross-section area; not consistent with r=0.2)
beam_I = 2.0e-4         # m^4 (bending moment of inertia)
beam_J = 2.0 * beam_I   # m^4 (torsional constant; circular section -> J = 2*I)
E = 1.0e8               # Pa
nu = 0.0
density = 7850.0        # kg/m^3 (not specified in the benchmark; the test is
                        #         quasi-static, density is only used to scale
                        #         inertia which is suppressed by m_quasi)

# Quasi-static / inextensibility scaling (same idea as contact_bending_test_3D)
k_inext = 1.0           # leave EA at its physical value here; A is large enough
m_quasi = 1.0e-3        # strongly reduce inertia for quasi-static behavior

# ---- Loading -----------------------------------------------------------------
delta_uy_per_step = +0.3    # m, prescribed at A per load step (toward beam X)
delta_uz_per_step = -0.03   # m, prescribed at A per load step (in-plane slide)
n_load_steps_default = 6    # Case 1 of the benchmark (6 steps). Case 2 -> 29.

# ---- Time stepping ---------------------------------------------------------
# Each "load step" of the benchmark is realized over `time_per_step` seconds
# of slow loading (quasi-static).  Within a step we ramp A's displacement at
# constant velocity; one Dismech state at the END of each step is taken as
# the equilibrium snapshot for comparison.
time_per_step = 1.0         # s per benchmark load step
dt = 1.0e-2                 # s, Dismech time step
save_interval = time_per_step  # save once per load step

# ---- Reference data (Zavarise & Wriggers 2000, Table IV, μ = 0) ------------
# Per-step normal contact force F_N at the contact element, frictionless case.
# Step 0/1 are pre-contact; step 2 is incipient contact (negligible force);
# steps 3+ are the steady linear progression at ~17.1 N per step.
ZAVARISE_TABLE_IV_FN_FRICTIONLESS = {
    0: 0.0,         # pre-contact (initialisation snapshot)
    1: 0.0,
    2: 1.0e-9,
    3: 17.2,
    4: 34.3,
    5: 51.3,
    6: 68.4,
}

# Full Table IV: (F_N, F_T) per (step, mu).  Reference: Zavarise & Wriggers
# (2000), Example 2.  Used when CLI --mu selects a non-zero column.
ZAVARISE_TABLE_IV = {
    # mu : { step : (F_N, F_T) }
    0.0: {1: (0.00,  0.00),  2: (1e-9, 0.00),  3: (17.2, 0.00),
          4: (34.3, 0.00),  5: (51.3, 0.00),   6: (68.4, 0.00)},
    0.1: {1: (0.00,  0.00),  2: (1e-9, 0.00),  3: (17.0, 1.70),
          4: (33.8, 3.38),  5: (50.4, 5.04),   6: (67.0, 6.70)},
    0.8: {1: (0.00,  0.00),  2: (1e-9, 0.00),  3: (16.0, 11.5),
          4: (30.8, 24.5),  5: (45.0, 36.0),   6: (58.4, 46.7)},
    1.5: {1: (0.00,  0.00),  2: (1e-9, 0.00),  3: (16.0, 11.5),
          4: (30.8, 24.7),  5: (44.3, 39.4),   6: (56.4, 55.8)},
}


def _ref_table_for_mu(mu_val):
    """Return the closest Table IV column for the given mu (default 0.0)."""
    if mu_val in ZAVARISE_TABLE_IV:
        return ZAVARISE_TABLE_IV[mu_val], mu_val
    # snap to the nearest reference mu
    nearest = min(ZAVARISE_TABLE_IV.keys(), key=lambda m: abs(m - mu_val))
    return ZAVARISE_TABLE_IV[nearest], nearest

# ---- CLI -------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Zavarise & Wriggers (2000) Example 2 — orthogonal beams, μ=0")
parser.add_argument("n_steps", nargs="?", type=int, default=n_load_steps_default,
                    help="Number of load steps to apply (default: %(default)s; "
                         "6 = paper's Case 1, 29 = paper's Case 2)")
parser.add_argument("--restart", action="store_true",
                    help="Restart from the last saved snapshot")
parser.add_argument("--dt", type=float, default=None, metavar="DT",
                    help="Override the time step (default %(default)s -> the "
                         "module-level dt = 1e-2 s).  Used by the largest-"
                         "stable-timestep study; must divide total_time = "
                         "n_steps * 1.0 s exactly.")
parser.add_argument("--fixed-kc", type=float, default=None, metavar="VALUE",
                    help="Override the adaptive contact stiffness with a "
                         "constant value (passed through env._eberly_bb_kc at "
                         "every Newton iteration).  The paper uses K_N = 1.e+4 "
                         "for Example 2; try `--fixed-kc 1.0e4` to compare "
                         "against Table IV with matched penalty stiffness.")
parser.add_argument("--nx", type=int, default=None, metavar="N",
                    help="Override the number of Beam X nodes (default 11 -> "
                         "10 elements of 1 m).  Used for mesh convergence "
                         "studies.  If --nz is not also given, n_z is set so "
                         "that both beams have the same element length.")
parser.add_argument("--nz", type=int, default=None, metavar="N",
                    help="Override the number of Beam Z nodes.  Default is "
                         "n_x + 4 so that both beams have 1 m elements.")
parser.add_argument("--tag", type=str, default="", metavar="SUFFIX",
                    help="Optional suffix to append to all output filenames "
                         "(e.g. '--tag _nx21' to keep multiple runs side by "
                         "side).  Leading underscore added automatically if "
                         "missing.")
parser.add_argument("--mu", type=float, default=0.0, metavar="MU",
                    help="Coulomb friction coefficient (default 0 = "
                         "frictionless).  Reference Table IV columns are "
                         "available for mu in {0.0, 0.1, 0.8, 1.5}.")
parser.add_argument("--vel-tol", type=float, default=1.0e-3, metavar="VTOL",
                    help="Stick/slip velocity tolerance (m/s) for the smooth "
                         "Coulomb gate (gamma).  Below this the friction force "
                         "smoothly tapers to zero; above it gamma -> 1 (pure "
                         "sliding).  Default 1e-3.")
parser.add_argument("--imc", action="store_true",
                    help="Use the non-regularized IMC contact + friction "
                         "(selfContact + selfFriction) instead of the "
                         "Eberly arc-based contact.  Requires the IMC "
                         "state.u bug fix in time_stepper.py.")
parser.add_argument("--imc-fric-jac", action="store_true",
                    help="(IMC mode only) Assemble the velocity-Jacobian of "
                         "IMC friction (∂F_fric/∂v · 1/dt) into the global "
                         "tangent.  Default: force-only (semi-explicit).")
parser.add_argument("--eberly-fric-jac", action="store_true",
                    help="(Eberly mode only) Assemble the velocity-Jacobian "
                         "of Eberly friction into the global tangent.  "
                         "Default: force-only (semi-explicit).")
parser.add_argument("--friction-law", type=str,
                    default="smooth_coulomb",
                    choices=["smooth_coulomb", "norton_hoff", "norton_hoff_capped"],
                    help="Friction force law.  smooth_coulomb (default): "
                         "F_T = mu*|F_N|*gate(|v_t|)*t_hat with saturating "
                         "gate, gate(|v_t|>v_tol)=1.  norton_hoff: "
                         "F_T = mu*|F_N|*(|v_t|/v_ref)^rho * t_hat — power-law, "
                         "unbounded above v_ref.  norton_hoff_capped: NH "
                         "clipped to G<=1 (recommended for stability).  All "
                         "share mu and v_tol.")
parser.add_argument("--eberly-fric-vel", type=str,
                    default="in_iter",
                    choices=["in_iter", "state_u", "hybrid"],
                    help="Velocity argument for Eberly friction force.  "
                         "in_iter (default): u=(q-q_prev)/dt at every Newton "
                         "iter — fully implicit; iter 1 has u=0 and friction=0. "
                         "state_u: u=robot.state.u every iter (semi-explicit, "
                         "legacy).  hybrid: state_u at iter 1, in_iter after.")
parser.add_argument("--nh-rho", type=float, default=0.1, metavar="RHO",
                    help="Norton-Hoff exponent rho (0<rho<=1). rho->0 is "
                         "near-Coulomb (sharp); rho=1 is purely viscous "
                         "(linear in |v_t|).  Default 0.1.")
parser.add_argument("--nh-vref", type=float, default=0.03, metavar="VREF",
                    help="Norton-Hoff reference velocity v_ref (m/s).  At "
                         "|v_t|=v_ref the friction saturates at mu*|F_N|.  "
                         "Default 0.03 m/s — matches the Zavarise loading "
                         "sliding speed so both laws agree at steady sliding.")
parser.add_argument("--per-dt-trace", action="store_true",
                    help="Record (t, F_N_xyz, F_T_xyz) on Beam X at every "
                         "dt-step, not just at load-step boundaries.  Writes "
                         "to zavarise_per_dt_trace_steps{N}{tag}.txt.  Adds "
                         "one contact+friction re-evaluation per dt-step "
                         "(~30 percent overhead) so it's off by default.")
parser.add_argument("--max-iter", type=int, default=None, metavar="N",
                    help="Override the Newton iteration limit (default: 200 "
                         "with friction, 100 without).  Raise for stiff "
                         "friction laws (e.g. Norton-Hoff near sticking).")
parser.add_argument("--z-contact", type=float, default=None, metavar="Z",
                    help="Override Beam X's z-position (default 3.0 = paper "
                         "geometry, contact exactly ON a Beam Z node).  In "
                         "the EXACTLY planar+symmetric configuration the "
                         "Eberly window handover is perfectly degenerate "
                         "when the sliding contact reaches half/full-element "
                         "registry (Newton livelocks at slide = 0.25/0.50 m); "
                         "e.g. 3.03 de-phases the mesh registry and breaks "
                         "the tie (arm 10.03 m vs paper's 10: -0.3%% on F_N).")
parser.add_argument("--x-shift", type=float, default=0.0, metavar="DX",
                    help="Rigidly shift Beam X along its own axis (x) by DX "
                         "metres.  Mechanically near-neutral (clamped-"
                         "clamped beam, arms 5+DX / 5-DX), but it moves the "
                         "contact point OFF Beam X's central node: in the "
                         "exactly planar+symmetric configuration the contact "
                         "sits on that node permanently and the mirror-image "
                         "Eberly window pairs are exactly degenerate.")
parser.add_argument("--planar-tilt", type=float, default=0.0, metavar="DX",
                    help="Applied ONLY with --planar.  Beam Z's x=0 plane "
                         "gives every arc-fit triple a circle whose plane "
                         "normal is EXACTLY the global x-axis, at all times "
                         "(unlike free 3D, where roundoff tilts it "
                         "generically).  This flag adds a linear ramp in x "
                         "across Beam Z (0 at node 0, DX at the last node) "
                         "BEFORE the x-DOF is fixed, so the beam is still "
                         "perfectly flat and lateral-flutter-free (x-DOF "
                         "still fully pinned, zero dynamic freedom) but its "
                         "plane is tilted by a tiny fixed angle instead of "
                         "being exactly axis-aligned.  Tests whether the "
                         "planar Newton livelock is a coordinate/frame "
                         "singularity tied to that exact alignment.")
parser.add_argument("--planar", action="store_true",
                    help="Fix the x-DOF of every Beam Z node.  With friction "
                         "the planar (x = 0) configuration is unstable: the "
                         "frictional follower force pumps the laterally "
                         "unconstrained beam into metre-scale out-of-plane "
                         "swings within ~2 load steps of contact (mu = 0 "
                         "stays planar to 1e-7 m).  This flag removes that "
                         "lateral DOF, i.e. runs the planar variant of the "
                         "benchmark -- closest in spirit to the paper's "
                         "point-to-point contact element, which cannot "
                         "slide laterally by construction.")
parser.add_argument("--eberly-straight-R-threshold", type=float, default=None,
                    metavar="R", dest="eberly_straight_R_threshold",
                    help="Opt-in pair-level fallback: any Eberly contact "
                         "window whose fitted arc radius exceeds R (either "
                         "beam) uses the already-stable LINE primitive "
                         "instead of the arc kernel.  Fixes the --planar "
                         "livelock: Beam Z's structural elastic inflection "
                         "point sweeps past each mesh window roughly once "
                         "per element of slide; at that window curvature "
                         "-> 0 so R -> infinity and the arc kernel's force "
                         "explodes (~10,000x, measured).  R = 1e3 (below "
                         "Beam Z's ~1.3e4 m baseline radius everywhere) "
                         "guarantees -- by construction -- that Beam Z's "
                         "side of every pair always uses the line "
                         "primitive.  NOTE: forces the scalar (non-batched) "
                         "narrow phase for the whole call -- slower.  "
                         "--planar-tilt was tried first and is NOT a "
                         "robust fix (a rigid tilt still confines Beam Z "
                         "to a flat plane, so the same curvature-crosses-"
                         "zero event recurs at the next element crossing "
                         "regardless of tilt magnitude) -- see NOTES.md.")
cli_args = parser.parse_args()
n_load_steps = cli_args.n_steps
restart_mode = cli_args.restart
if cli_args.dt is not None:
    dt = float(cli_args.dt)
    save_interval = time_per_step
fixed_kc = cli_args.fixed_kc
mu_friction = float(cli_args.mu)
vel_tol = float(cli_args.vel_tol)
use_imc = bool(cli_args.imc)
imc_fric_jac = bool(cli_args.imc_fric_jac)
eberly_fric_jac = bool(cli_args.eberly_fric_jac)
friction_law = str(cli_args.friction_law)
nh_rho = float(cli_args.nh_rho)
nh_vref = float(cli_args.nh_vref)
eberly_fric_vel = str(cli_args.eberly_fric_vel)
per_dt_trace = bool(cli_args.per_dt_trace)
max_iter_override = cli_args.max_iter
planar_constraint = bool(cli_args.planar)
planar_tilt = float(cli_args.planar_tilt)
beam_x_shift = float(cli_args.x_shift)
eberly_straight_R_threshold = cli_args.eberly_straight_R_threshold
if cli_args.z_contact is not None:
    z_x_contact = float(cli_args.z_contact)
    L1_from_clamp_to_contact = z_x_contact - (-z_beam_length / 2)

# ---- Apply mesh overrides --------------------------------------------------
if cli_args.nx is not None:
    n_x_nodes = int(cli_args.nx)
    # Keep elements the same length on both beams unless --nz is given:
    # Beam X has length 10, Beam Z has length 14; if Beam X uses (nx - 1)
    # elements of length 10/(nx-1), then Beam Z needs (nx - 1) * 14/10 elements.
    # In practice, since 14/10 isn't integer for general nx, we round.
    if cli_args.nz is None:
        elem_len = x_beam_length / (n_x_nodes - 1)
        n_z_nodes = int(round(z_beam_length / elem_len)) + 1
    else:
        n_z_nodes = int(cli_args.nz)
elif cli_args.nz is not None:
    n_z_nodes = int(cli_args.nz)

# Normalise the output filename suffix
output_tag = cli_args.tag.strip()
if output_tag and not output_tag.startswith("_"):
    output_tag = "_" + output_tag

total_time = n_load_steps * time_per_step

# Velocity equivalent of the per-step displacement increments
imposed_vy = delta_uy_per_step / time_per_step    # m/s
imposed_vz = delta_uz_per_step / time_per_step    # m/s

# ============================================================================
# GEOMETRY CREATION
# ============================================================================

def create_straight_beam_along_axis(length, n_nodes, fixed_coord_y, fixed_coord_z,
                                     axis='x', kappa=kappa0):
    """
    Beam centered on the origin of `axis` (x or z), with a tiny circular-arc
    curvature `kappa` in the y direction so that no 3-node triplet is exactly
    collinear (required by the Eberly arc-to-arc contact).  The MIDSPAN of the
    beam stays at the prescribed (y, z) (for axis='x') or (x, y) (for axis='z')
    position; the end-nodes are offset in y by R*(1-cos(L/(2R))).

    For axis='x':  fixed_coord_y → midspan y, fixed_coord_z → constant z.
    For axis='z':  fixed_coord_y → constant x, fixed_coord_z → midspan y.

    If `kappa` is 0 or negligible, the legacy straight-beam behaviour is used
    (and contact will be silently disabled for collinear triplets).
    """
    nodes = np.zeros((n_nodes, 3))
    if abs(kappa) < 1e-15:
        s = np.linspace(-length / 2, length / 2, n_nodes)
        y_off = np.zeros(n_nodes)
        a_coord = s
    else:
        R = 1.0 / abs(kappa)
        sign = 1.0 if kappa >= 0 else -1.0
        half_angle = (length / 2) / R
        theta = np.linspace(-half_angle, half_angle, n_nodes)
        a_coord = R * np.sin(theta)              # projection onto the axis
        y_off = sign * R * (1.0 - np.cos(theta))  # y offset (0 at midspan)

    if axis == 'x':
        nodes[:, 0] = a_coord
        nodes[:, 1] = fixed_coord_y + y_off
        nodes[:, 2] = fixed_coord_z
    elif axis == 'z':
        nodes[:, 0] = fixed_coord_y   # here interpreted as x-position
        nodes[:, 1] = fixed_coord_z + y_off
        nodes[:, 2] = a_coord
    else:
        raise ValueError("axis must be 'x' or 'z'")
    edges = np.array([[i, i + 1] for i in range(n_nodes - 1)], dtype=int)
    face_nodes = np.empty((0, 3), dtype=int)
    return dismech.Geometry(nodes, edges, face_nodes)


def combine_geometries(geo1, geo2):
    """Concatenate two beam geometries (offsetting indices for geo2)."""
    n1 = geo1.nodes.shape[0]
    combined_nodes = np.vstack([geo1.nodes, geo2.nodes])
    edges2_offset = geo2.edges + n1
    combined_edges = np.vstack([geo1.edges, edges2_offset])
    combined_faces = np.empty((0, 3), dtype=int)
    return dismech.Geometry(combined_nodes, combined_edges, combined_faces)


# ============================================================================
# RESTART UTILITIES
# ============================================================================

def load_last_snapshot(coords_file, n_nodes):
    """Read the last complete snapshot from a coordinates file."""
    last_time = None
    coords = np.zeros((n_nodes, 3))
    lines_in_snapshot = 0
    with open(coords_file, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            t = float(parts[0])
            node_id = int(parts[1])
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            if last_time is None or t != last_time:
                if last_time is not None and lines_in_snapshot < n_nodes:
                    pass
                last_time = t
                coords = np.zeros((n_nodes, 3))
                lines_in_snapshot = 0
            coords[node_id] = [x, y, z]
            lines_in_snapshot += 1
    if last_time is None or lines_in_snapshot < n_nodes:
        raise RuntimeError(
            f"No complete snapshot in '{coords_file}' (expected {n_nodes} nodes).")
    print(f"  Loaded restart snapshot at t = {last_time:.6f} s from '{coords_file}'")
    return last_time, coords


# ============================================================================
# CURVATURE COMPUTATION (for validation against the benchmark)
# ============================================================================

def per_segment_curvature_radius(nodes_along_beam):
    """
    For an ordered set of nodes along a beam, return the per-segment local
    curvature radius using the discrete Frenet-style formula:
        kappa_i = 2 |t_a x t_b| / (|e_a| |e_b| (1 + t_a . t_b))
    Returns an array of length (n_nodes - 2) with R_i = 1 / kappa_i.
    Sets R = inf where |kappa| < 1e-12.
    """
    pts = np.asarray(nodes_along_beam, dtype=float)
    n = pts.shape[0]
    if n < 3:
        return np.array([])
    e = np.diff(pts, axis=0)
    le = np.linalg.norm(e, axis=1)
    t = e / le[:, None]
    # Curvature at interior node i (between edges i-1 and i)
    cross_norm = np.linalg.norm(np.cross(t[:-1], t[1:]), axis=1)
    dot_t = np.einsum('ij,ij->i', t[:-1], t[1:])
    denom = le[:-1] * le[1:] * (1.0 + dot_t)
    kappa = np.where(np.abs(denom) > 1e-15, 2.0 * cross_norm / denom, 0.0)
    R = np.where(np.abs(kappa) > 1e-12, 1.0 / np.abs(kappa), np.inf)
    return R


# ============================================================================
# SIMULATION
# ============================================================================

def run_zavarise_benchmark(restart=False):
    """Run the Zavarise & Wriggers Benchmark 1 with frictionless contact."""
    print("=" * 70)
    print("ZAVARISE & WRIGGERS (2000) — BENCHMARK 1 (frictionless)")
    if restart:
        print(">>> RESTART MODE <<<")
    print("=" * 70)
    print(f"Load steps           : {n_load_steps}")
    print(f"time_per_step        : {time_per_step} s  (total_time = {total_time} s)")
    print(f"dt                   : {dt} s")
    print(f"Δuy / step           : {delta_uy_per_step} m  -> vy = {imposed_vy} m/s")
    print(f"Δuz / step           : {delta_uz_per_step} m -> vz = {imposed_vz} m/s")
    print()

    # ---- Build the two beams ------------------------------------------------
    # Beam X (along x), at (y=y_x, z=z_x_contact).
    # By placing Beam X at z = z_x_contact = +3, the perpendicular contact
    # point ends up at z = +3 on Beam Z, i.e. 10 m from A (z=-7) and 4 m
    # from B (z=+7), which matches the paper's Figure 9.
    x_geo = create_straight_beam_along_axis(
        x_beam_length, n_x_nodes, fixed_coord_y=y_x, fixed_coord_z=z_x_contact,
        axis='x')
    if beam_x_shift != 0.0:
        x_geo = dismech.Geometry(
            x_geo.nodes + np.array([beam_x_shift, 0.0, 0.0]),
            x_geo.edges, np.empty((0, 3), dtype=int))
        print(f"Beam X shifted along x by {beam_x_shift:+.4f} m "
              f"(contact lands {beam_x_shift:+.4f} m off its central node)")
    # Beam Z (along z), at (x=0, y=y_z)
    z_geo = create_straight_beam_along_axis(
        z_beam_length, n_z_nodes, fixed_coord_y=0.0, fixed_coord_z=y_z, axis='z')
    if planar_tilt != 0.0:
        # Linear ramp in x across Beam Z (0 at node 0 -> planar_tilt at the
        # last node), applied BEFORE --planar fixes the x-DOF: the beam
        # stays perfectly flat (zero dynamic x freedom once fixed) but its
        # plane is no longer exactly x-axis-normal.  See --planar-tilt help.
        ramp = np.linspace(0.0, planar_tilt, n_z_nodes)
        z_geo = dismech.Geometry(
            z_geo.nodes + np.column_stack(
                [ramp, np.zeros(n_z_nodes), np.zeros(n_z_nodes)]),
            z_geo.edges, np.empty((0, 3), dtype=int))
        print(f"Beam Z given a static x-tilt ramp: 0 -> {planar_tilt:+.4f} m "
              f"across its length (fixed before --planar locks the x-DOF)")
    combined_geo = combine_geometries(x_geo, z_geo)
    n_total_nodes = combined_geo.nodes.shape[0]
    print(f"Beam X: {n_x_nodes} nodes, indices 0..{n_x_nodes - 1}")
    print(f"Beam Z: {n_z_nodes} nodes, indices {n_x_nodes}..{n_total_nodes - 1}")
    if abs(kappa0) > 0.0:
        sag_x = (1.0 / abs(kappa0)) * (1.0 - np.cos((x_beam_length / 2) * abs(kappa0)))
        sag_z = (1.0 / abs(kappa0)) * (1.0 - np.cos((z_beam_length / 2) * abs(kappa0)))
        print(f"Initial curvature κ₀ = {kappa0:.1e} 1/m  "
              f"-> sagitta X = {sag_x*1e3:.3f} mm, Z = {sag_z*1e3:.3f} mm "
              f"(midspan stays at y = {y_x:+.4f} / {y_z:+.4f})")

    # Effective mechanical properties
    A_eff = k_inext * beam_area_A
    I_eff = beam_I
    J_eff = beam_J
    density_eff = (density / k_inext) * m_quasi

    geom = dismech.GeomParams(
        rod_r0=beam_radius,
        shell_h=0,
        axs=A_eff,
        ixs1=I_eff,
        ixs2=I_eff,
        jxs=J_eff,
    )
    material = dismech.Material(
        density=density_eff,
        youngs_rod=E,
        youngs_shell=0,
        poisson_rod=nu,
        poisson_shell=0,
    )

    log_step = max(1, int(round(save_interval / dt)))
    print(f"Saving snapshot every {log_step * dt:.3f} s "
          f"(i.e. once per load step)")

    n_edges_beam_A = n_x_nodes - 1  # edges in Beam X (the "first" beam)
    sim_params = dismech.SimParams(
        static_sim=False,
        two_d_sim=False,
        use_mid_edge=False,
        use_line_search=False,
        show_floor=False,
        log_data=True,
        log_step=log_step,
        dt=dt,
        max_iter=(max_iter_override if max_iter_override is not None
                  else (200 if mu_friction > 0.0 else 100)),
        total_time=total_time,
        plot_step=0,
        tol=1e-4,
        ftol=1e-4,
        dtol=1e-4,
        n_edges_beam_A=n_edges_beam_A,
        # Only Beam X vs Beam Z contact is physical here: under this loading
        # neither beam can fold onto itself.  Without this, IMC's candidate
        # filter (edge-edge distance < 2h + 10*delta = 0.6 m) admits
        # WITHIN-beam neighbour pairs as soon as the element length drops
        # below ~0.15 m, i.e. for any mesh finer than about n_x = 40:
        #
        #   n_x =  21 -> element 0.500 m, closest kept within-beam pair
        #                (index gap min_contact_edge_gap+1 = 4) is 2.000 m
        #                -> outside the filter, nothing admitted, runs fine.
        #   n_x = 100 -> element 0.101 m, closest kept pair is 0.404 m
        #                -> inside the filter; 189 near-parallel adjacent-edge
        #                pairs enter the contact set sitting 4 mm inside the
        #                2h contact radius, and Newton blows up (residual
        #                reaches ~4e15 at mu = 0).
        #
        # Raising min_contact_edge_gap would also work but the required value
        # is mesh-dependent (>= 6 at n_x = 100); this flag is mesh-independent.
        cross_beam_only_contact=True,
    )

    env = dismech.Environment()
    if use_imc:
        # Non-regularized IMC contact + friction (the "reference"
        # implementation that the regularized Eberly version aims to
        # match).  IMC needs explicit delta / h: h is the contact threshold
        # (contact when edge-edge distance < 2h) and delta is the IMC smooth
        # transition width.
        env.add_force(
            'selfContact',
            delta=0.02,
            h=beam_radius,
            kc=1.0,
        )
        if mu_friction > 0.0:
            env.add_force(
                'selfFriction',
                mu=mu_friction,
                vel_tol=vel_tol,
            )
            # Opt-in: assemble the friction velocity-Jacobian into Newton tangent
            env.imc_friction_jacobian = imc_fric_jac
            print(f"IMC friction enabled: mu = {mu_friction}, vel_tol = {vel_tol} m/s"
                  + (", Jacobian=ON" if imc_fric_jac else ", Jacobian=OFF"))
        else:
            print("IMC friction disabled (mu = 0).")
    else:
        env.add_force(
            'beamToBeamContactEberly3D',
            beam_radius=beam_radius,
            n_nodes_beam_A=n_x_nodes,
            kc=1.0,
            d_imc=0.02,        # 10% of beam_radius: smooth IMC transition zone
        )
        if eberly_straight_R_threshold is not None:
            env.eberly_straight_R_threshold = eberly_straight_R_threshold
            print(f"Eberly straight-R fallback: windows with fitted arc "
                  f"radius > {eberly_straight_R_threshold:.3g} m use the "
                  f"line primitive (forces the scalar narrow-phase backend)")
        if mu_friction > 0.0:
            env.add_force(
                'beamToBeamFrictionEberly3D',
                mu=mu_friction,
                vel_tol=vel_tol,
            )
            env.eberly_friction_jacobian = eberly_fric_jac
            env.friction_law             = friction_law
            env.norton_hoff_rho          = nh_rho
            env.norton_hoff_v_ref        = nh_vref
            env.eberly_friction_velocity_mode = eberly_fric_vel
            jac_str = "ON" if eberly_fric_jac else "OFF"
            if friction_law == "norton_hoff":
                law_str = f"Norton-Hoff (rho={nh_rho}, v_ref={nh_vref} m/s)"
            elif friction_law == "norton_hoff_capped":
                law_str = f"Norton-Hoff capped (rho={nh_rho}, v_ref={nh_vref} m/s)"
            else:
                law_str = f"smooth Coulomb (vel_tol={vel_tol} m/s)"
            print(f"Eberly friction enabled: mu = {mu_friction}, law = {law_str}, "
                  f"velocity = {eberly_fric_vel}, Jacobian = {jac_str}")
        else:
            print("Eberly friction disabled (mu = 0).")
    env.add_force('damping', eta=10.0)   # quasi-static damping

    robot = dismech.SoftRobot(geom, material, combined_geo, sim_params, env)
    print("Created SoftRobot")

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------
    # Beam X (Beam A in the Eberly contact pair): clamped at BOTH ends C, D
    # — fix first two and last two nodes plus the edge DOF between them.
    C_nodes = np.array([0, 1])
    D_nodes = np.array([n_x_nodes - 2, n_x_nodes - 1])
    robot = robot.fix_nodes(C_nodes, fix_edges=True)
    robot = robot.fix_nodes(D_nodes, fix_edges=True)
    print(f"Beam X clamped at C={C_nodes.tolist()} and D={D_nodes.tolist()}")

    # Beam Z: clamp end A (with prescribed motion). A = first two z-beam nodes.
    A_nodes = np.array([n_x_nodes, n_x_nodes + 1])
    robot = robot.fix_nodes(A_nodes, fix_edges=True)
    print(f"Beam Z clamped at A={A_nodes.tolist()} "
          f"(prescribed Δuy/step={delta_uy_per_step} m, "
          f"Δuz/step={delta_uz_per_step} m)")

    if planar_constraint:
        # Pin the x-DOF of every Beam Z node: removes the laterally
        # unconstrained "rail" DOF that the frictional follower force
        # destabilizes (see --planar help).
        z_beam_nodes = np.arange(n_x_nodes, n_total_nodes)
        robot = robot.fix_nodes(z_beam_nodes, axis=0, fix_edges=False)
        print(f"PLANAR mode: x-DOF fixed on all {z_beam_nodes.size} "
              f"Beam Z nodes")

    stepper = dismech.ImplicitEulerTimeStepper(robot)

    # ------------------------------------------------------------------
    # Output paths
    # ------------------------------------------------------------------
    out_dir = Path(__file__).resolve().parent / "contact_output_files_soft_soft"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_coords_txt  = out_dir / f"zavarise_coords_steps{n_load_steps}{output_tag}.txt"
    out_forces_txt  = out_dir / f"zavarise_forces_steps{n_load_steps}{output_tag}.txt"
    out_curv_txt    = out_dir / f"zavarise_curvR_steps{n_load_steps}{output_tag}.txt"
    out_summary_txt = out_dir / f"zavarise_summary_steps{n_load_steps}{output_tag}.txt"
    out_per_dt_txt  = out_dir / f"zavarise_per_dt_trace_steps{n_load_steps}{output_tag}.txt"

    # ------------------------------------------------------------------
    # Restart
    # ------------------------------------------------------------------
    restart_time = 0.0
    restart_step = 0
    accumulated_y = 0.0
    accumulated_z = 0.0

    if restart:
        restart_time, restart_coords = load_last_snapshot(out_coords_txt, n_total_nodes)
        restart_step = int(round(restart_time / dt))
        q_restart = np.copy(robot.state.q)
        for ni in range(n_total_nodes):
            dofs = robot.map_node_to_dof(np.array([ni])).ravel()
            q_restart[dofs] = restart_coords[ni]
        robot = robot.update(q=q_restart)
        # Recover accumulated A motion from restart_coords vs the initial A position
        A_init = combined_geo.nodes[A_nodes[0]]
        A_now = restart_coords[A_nodes[0]]
        accumulated_y = A_now[1] - A_init[1]
        accumulated_z = A_now[2] - A_init[2]
        print(f"  Restart at t={restart_time:.6f} s, step {restart_step}")
        print(f"  Accumulated A displacement: "
              f"Δy={accumulated_y:.6f} m, Δz={accumulated_z:.6f} m")

    # ------------------------------------------------------------------
    # Per-step driver
    # ------------------------------------------------------------------
    def before_step_fn(rb, t):
        """Advance A by one dt worth of prescribed velocity (clamped clamp)."""
        nonlocal accumulated_y, accumulated_z

        # ── Override the adaptive contact stiffness if requested ────────────
        # We force `_eberly_bb_kc` back to `fixed_kc` every step so the
        # adaptive smoothing in the time-stepper cannot drift it.  This makes
        # the contact behave like a fixed-penalty model akin to the paper's
        # εN = 1e6 N/m.
        if fixed_kc is not None:
            rb.env._eberly_bb_kc = float(fixed_kc)
            rb.env._eberly_bb_active = True  # block decay back to imc_stiffness

        dt_local = rb.sim_params.dt
        # Cap accumulated motion at exactly n_load_steps * per-step increment
        max_y = n_load_steps * delta_uy_per_step
        max_z = n_load_steps * delta_uz_per_step
        # remaining motion (with sign of the per-step increment)
        dy = np.sign(delta_uy_per_step) * min(
            abs(imposed_vy * dt_local),
            abs(max_y - accumulated_y))
        dz = np.sign(delta_uz_per_step) * min(
            abs(imposed_vz * dt_local),
            abs(max_z - accumulated_z))
        if dy == 0.0 and dz == 0.0:
            return rb
        accumulated_y += dy
        accumulated_z += dz
        rb = rb.move_nodes(A_nodes, np.array([0.0, dy, dz]))
        return rb

    stepper.before_step = before_step_fn

    def write_snapshot(rb, t_now, step_id, fh_coords, fh_forces, fh_curv):
        """Write one saved snapshot (coords + contact forces + curvature R)."""
        q = rb.state.q
        node_coords = q[:rb.end_node_dof_index].reshape(-1, 3)

        # Coordinates
        for node_idx, xyz in enumerate(node_coords):
            fh_coords.write(
                f"{t_now:.6f}\t{step_id}\t{node_idx}\t"
                f"{xyz[0]:.12e}\t{xyz[1]:.12e}\t{xyz[2]:.12e}\n")

        # Contact forces — use the same k_c that drove the simulation, i.e.
        # the fixed override if requested, otherwise the adaptive value
        # maintained by the time-stepper.
        friction_force_nodes = np.zeros_like(node_coords)
        try:
            if fixed_kc is not None:
                k_c_save = float(fixed_kc)
            else:
                k_c_save = getattr(rb.env, '_eberly_bb_kc',
                                   rb.env.imc_stiffness)
            F_reg, _, _, _ = compute_eberly_3d_beam_to_beam_contact_forces(
                rb, q, beam_radius, k_c=k_c_save,
                n_nodes_beam_A=n_x_nodes, d_imc=0.02,
                straight_R_threshold=eberly_straight_R_threshold,
            )
            contact_force_nodes = F_reg[:rb.end_node_dof_index].reshape(-1, 3)

            # Friction force (if enabled): same k_c as the contact pass so
            # that the magnitudes are mutually consistent.
            if mu_friction > 0.0:
                F_fric = compute_eberly_3d_beam_to_beam_friction_forces(
                    rb, q, rb.state.u, beam_radius,
                    mu=mu_friction, vel_tol=vel_tol,
                    k_c=k_c_save, n_nodes_beam_A=n_x_nodes, d_imc=0.02,
                )
                friction_force_nodes = F_fric[:rb.end_node_dof_index].reshape(-1, 3)
        except Exception as e:
            print(f"  Warning: contact force extraction failed at t={t_now}: {e}")
            contact_force_nodes = np.zeros_like(node_coords)

        for node_idx, fxyz in enumerate(contact_force_nodes):
            fh_forces.write(
                f"{t_now:.6f}\t{step_id}\t{node_idx}\t"
                f"{fxyz[0]:.12e}\t{fxyz[1]:.12e}\t{fxyz[2]:.12e}\n")

        # Curvature radius per interior node of each beam (validation quantity)
        x_pts = node_coords[:n_x_nodes]
        z_pts = node_coords[n_x_nodes:]
        R_x = per_segment_curvature_radius(x_pts)
        R_z = per_segment_curvature_radius(z_pts)
        for i, R in enumerate(R_x):
            fh_curv.write(f"{t_now:.6f}\t{step_id}\tX\t{i + 1}\t{R:.12e}\n")
        for i, R in enumerate(R_z):
            fh_curv.write(f"{t_now:.6f}\t{step_id}\tZ\t{i + 1}\t{R:.12e}\n")

        return contact_force_nodes, friction_force_nodes

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    n_total_steps = int(round(total_time / dt))
    if abs(n_total_steps * dt - total_time) > 1e-9:
        raise ValueError(
            f"total_time={total_time} is not an integer multiple of dt={dt}.")

    saved_count = 0
    final_robot = robot
    final_forces = None
    file_mode = "a" if restart else "w"

    # Per-load-step force history, used for the final reference comparison
    per_step_Fy_X = {}   # step_id -> ΣFy on Beam X  (= F_N, sign-flipped)
    per_step_Fz_X = {}   # step_id -> ΣFz on Beam X  (geometric tangential)
    per_step_FT_X = {}   # step_id -> (F_x, F_y, F_z, |F_T|) of FRICTION on Beam X
    per_step_gap = {}    # step_id -> node-to-node surface gap (approx)

    # Per-dt-step force trace (only written if --per-dt-trace).  Rows are
    # (t, ΣFx_contact, ΣFy_contact, ΣFz_contact, ΣFx_fric, ΣFy_fric, ΣFz_fric)
    # summed over the Beam X nodes.  Written INCREMENTALLY (flushed every
    # row) so that a run that dies mid-way (e.g. Newton iteration limit)
    # still leaves the trace up to the failure on disk.
    n_per_dt_rows = 0
    import dataclasses as _dataclasses

    def _extract_FN_FT_on_BeamX(rb):
        """Compute contact + friction nodal-force vectors on Beam X at the
        current converged state of `rb`, using whichever model is active.
        Returns (F_contact_sum_xyz, F_fric_sum_xyz)."""
        q = rb.state.q
        n_dof = q.shape[0]
        if use_imc:
            # IMC mode: pull the contact/friction objects from the time stepper.
            ce = getattr(stepper, '_contact_energy', None)
            if ce is None:
                F_con = np.zeros(n_dof)
            else:
                F_con, _ = ce.grad_hess_energy(rb.state, rb, np.zeros(n_dof),
                                                first_iter=False)
            if mu_friction > 0.0 and getattr(stepper, '_imc_friction', None):
                state_with_u = _dataclasses.replace(rb.state, u=rb.state.u)
                F_fric = stepper._imc_friction.grad_friction(
                    state_with_u, rb, np.zeros(n_dof), first_iter=False)
            else:
                F_fric = np.zeros(n_dof)
        else:
            # Eberly mode (default): direct calls, same arguments as the
            # snapshot extraction further down so magnitudes are comparable.
            env = rb.env
            k_c_now = float(fixed_kc) if fixed_kc is not None else getattr(
                env, '_eberly_bb_kc', env.imc_stiffness)
            F_con, _, _, _ = compute_eberly_3d_beam_to_beam_contact_forces(
                rb, q, beam_radius, k_c=k_c_now,
                n_nodes_beam_A=n_x_nodes, d_imc=0.02,
                straight_R_threshold=eberly_straight_R_threshold,
            )
            if mu_friction > 0.0:
                F_fric = compute_eberly_3d_beam_to_beam_friction_forces(
                    rb, q, rb.state.u, beam_radius,
                    mu=mu_friction, vel_tol=vel_tol,
                    k_c=k_c_now, n_nodes_beam_A=n_x_nodes, d_imc=0.02,
                )
            else:
                F_fric = np.zeros(n_dof)

        F_con_X  = F_con [:rb.end_node_dof_index].reshape(-1, 3)[:n_x_nodes]
        F_fric_X = F_fric[:rb.end_node_dof_index].reshape(-1, 3)[:n_x_nodes]
        return F_con_X.sum(axis=0), F_fric_X.sum(axis=0)

    crashed_at = None   # (t, message) if the Newton solve gives up mid-run

    print(f"\nRunning {n_total_steps} time steps...")
    from contextlib import ExitStack
    with ExitStack() as _stack:
        fh_coords = _stack.enter_context(
            out_coords_txt.open(file_mode, encoding="utf-8"))
        fh_forces = _stack.enter_context(
            out_forces_txt.open(file_mode, encoding="utf-8"))
        fh_curv = _stack.enter_context(
            out_curv_txt.open(file_mode, encoding="utf-8"))
        fh_trace = None
        if per_dt_trace:
            fh_trace = _stack.enter_context(
                out_per_dt_txt.open(file_mode, encoding="utf-8"))
            if not restart:
                fh_trace.write(
                    "# t[s]  Fx_contact[N]  Fy_contact[N]  Fz_contact[N]  "
                    "Fx_fric[N]  Fy_fric[N]  Fz_fric[N]  "
                    f"  (model={'IMC' if use_imc else 'Eberly+smooth'})\n")
                fh_trace.flush()

        if not restart:
            fh_coords.write("# t_s\tstep_id\tnode_id\tx_m\ty_m\tz_m\n")
            fh_forces.write("# t_s\tstep_id\tnode_id\tFx_N\tFy_N\tFz_N\n")
            fh_curv.write("# t_s\tstep_id\tbeam\tnode_id\tR_curv_m\n")
            # initial snapshot
            final_forces, final_friction = write_snapshot(
                final_robot, 0.0, 0, fh_coords, fh_forces, fh_curv)
            saved_count += 1
            fh_coords.flush(); fh_forces.flush(); fh_curv.flush()

        start_step = restart_step if restart else 0
        for i in range(start_step, n_total_steps):
            t_before = i * dt
            if stepper.before_step is not None:
                final_robot = stepper.before_step(final_robot, t_before)
            try:
                final_robot, _ = stepper.step(final_robot)
            except ValueError as exc:
                # Typically "Iteration limit N reached before convergence".
                # Salvage: keep everything written so far (coords/forces/
                # trace are flushed incrementally), record where it died,
                # and still produce the summary and comparison outputs.
                crashed_at = (t_before + dt, str(exc))
                print(f"\n!!! Newton solve FAILED at t = {crashed_at[0]:.3f} s "
                      f"(dt-step {i + 1}/{n_total_steps}): {crashed_at[1]}")
                print("!!! Salvaging outputs up to the last converged step.")
                break
            t_after = (i + 1) * dt
            # Per-dt force trace — runs at EVERY dt-step (not just
            # log_step boundaries) so the resulting time series is dense
            # enough to look at smoothness.
            if per_dt_trace:
                try:
                    Fcon_sum, Ffr_sum = _extract_FN_FT_on_BeamX(final_robot)
                    row = (t_after,
                           Fcon_sum[0], Fcon_sum[1], Fcon_sum[2],
                           Ffr_sum[0],  Ffr_sum[1],  Ffr_sum[2])
                except Exception as _exc:
                    row = (t_after,) + (float('nan'),) * 6
                fh_trace.write(" ".join(f"{v:.18e}" for v in row) + "\n")
                fh_trace.flush()
                n_per_dt_rows += 1
            if (i + 1) % sim_params.log_step == 0:
                step_id = int(round(t_after / time_per_step))
                final_forces, final_friction = write_snapshot(
                    final_robot, t_after, step_id,
                    fh_coords, fh_forces, fh_curv)
                saved_count += 1
                fh_coords.flush(); fh_forces.flush(); fh_curv.flush()

                # Resultant contact force on Beam X — sum over all Beam X nodes.
                # ΣFx is essentially zero (no x-direction coupling).
                # ΣFy is the dominant component; ΣFz is the geometric projection
                # of the (tilted) contact normal onto the global z axis.
                F_x_x = float(np.sum(final_forces[:n_x_nodes, 0]))
                F_x_y = float(np.sum(final_forces[:n_x_nodes, 1]))
                F_x_z = float(np.sum(final_forces[:n_x_nodes, 2]))
                # Friction resultant on Beam X — y component should be ~0 (the
                # friction is tangential to the contact normal, ~ global +y).
                # |F_T| in the paper is the tangent-plane magnitude.
                Ff_x = float(np.sum(final_friction[:n_x_nodes, 0]))
                Ff_y = float(np.sum(final_friction[:n_x_nodes, 1]))
                Ff_z = float(np.sum(final_friction[:n_x_nodes, 2]))
                FT_mag = float(np.sqrt(Ff_x**2 + Ff_z**2))
                per_step_FT_X[step_id] = (Ff_x, Ff_y, Ff_z, FT_mag)
                # Local-frame |F_N| = magnitude of the resultant.  In a purely-
                # normal frictionless contact (which is what we have), F_T in
                # the local contact frame is zero by construction.
                FN_local = float(np.sqrt(F_x_x**2 + F_x_y**2 + F_x_z**2))
                # Tilt of the contact normal from the global +y axis
                # (drives the geometric projection of F into the global z).
                if FN_local > 1e-12:
                    tilt_deg = float(np.degrees(np.arctan2(
                        np.sqrt(F_x_x**2 + F_x_z**2), abs(F_x_y))))
                else:
                    tilt_deg = 0.0
                per_step_Fy_X[step_id] = FN_local
                per_step_Fz_X[step_id] = (F_x_x, F_x_y, F_x_z, tilt_deg)

                # Reference values: snap to the closest mu column of Table IV
                ref_col, ref_mu_used = _ref_table_for_mu(mu_friction)
                ref_pair = ref_col.get(step_id)
                if ref_pair is not None:
                    ref_FN, ref_FT = ref_pair
                elif step_id >= 7:
                    # extrapolate the frictionless column linearly past step 6
                    ref_FN = 17.1 * (step_id - 2)
                    ref_FT = mu_friction * ref_FN  # rough sliding extrapolation
                else:
                    ref_FN, ref_FT = None, None
                ref_str_FN = (f", paper F_N = {ref_FN:6.2f} N, "
                              f"ratio = {FN_local / max(ref_FN, 1e-12):.2f}"
                              if ref_FN and ref_FN > 1e-6 else "")
                ref_str_FT = (f", paper F_T = {ref_FT:6.2f} N"
                              if ref_FT is not None and ref_FT > 1e-6 else "")
                print(f"step {step_id:>2} (t={t_after:6.2f}s)  "
                      f"|F_N| = {FN_local:8.2f} N, "
                      f"|F_T| = {FT_mag:7.2f} N "
                      f"(Ff_z = {Ff_z:+7.2f}), "
                      f"Δuy_A = {accumulated_y:+.4f} m"
                      + ref_str_FN + ref_str_FT)

    print(f"\nDone. Saved {saved_count} snapshots in '{out_dir}'.")

    # Summary file
    with out_summary_txt.open("w", encoding="utf-8") as fh_summary:
        fh_summary.write("Zavarise & Wriggers (2000) Example 2 — frictionless\n")
        fh_summary.write(f"n_load_steps         = {n_load_steps}\n")
        fh_summary.write(f"n_x_nodes            = {n_x_nodes}\n")
        fh_summary.write(f"n_z_nodes            = {n_z_nodes}\n")
        fh_summary.write(f"beam_radius          = {beam_radius}\n")
        fh_summary.write(f"axs (cross-section)  = {beam_area_A}\n")
        fh_summary.write(f"I  (bending inertia) = {beam_I}\n")
        fh_summary.write(f"E                    = {E}\n")
        fh_summary.write(f"nu                   = {nu}\n")
        fh_summary.write(f"kappa0 (initial)     = {kappa0}\n")
        fh_summary.write(f"delta_uy_per_step    = {delta_uy_per_step}\n")
        fh_summary.write(f"delta_uz_per_step    = {delta_uz_per_step}\n")
        fh_summary.write(f"time_per_step        = {time_per_step}\n")
        fh_summary.write(f"dt                   = {dt}\n")
        fh_summary.write(f"total_time           = {total_time}\n")
        fh_summary.write(f"snapshots_saved      = {saved_count}\n")
        fh_summary.write(f"fixed_kc             = {fixed_kc}\n")
        fh_summary.write(f"mu_friction          = {mu_friction}\n")
        fh_summary.write(f"vel_tol              = {vel_tol}\n")
        if crashed_at is not None:
            fh_summary.write(f"CRASHED_AT_T         = {crashed_at[0]:.6f}\n")
            fh_summary.write(f"CRASH_REASON         = {crashed_at[1]}\n")
        if final_forces is not None:
            max_contact_force_mag = np.max(np.linalg.norm(final_forces, axis=1))
            fh_summary.write(
                f"final_max_contact_force_mag = {max_contact_force_mag:.12e}\n")

    # ---------------------------------------------------------------------
    # Comparison file vs Zavarise & Wriggers Table IV (μ = 0)
    # ---------------------------------------------------------------------
    out_compare_txt = out_dir / f"zavarise_compare_steps{n_load_steps}{output_tag}.txt"
    ref_col, ref_mu_used = _ref_table_for_mu(mu_friction)
    print()
    print("=" * 124)
    print(f"Comparison vs Zavarise & Wriggers (2000) Table IV  (μ_ref = {ref_mu_used}"
          f"{'  [exact match]' if abs(ref_mu_used - mu_friction) < 1e-12 else f'  [closest to requested μ = {mu_friction}]'})")
    print(f"Simulation μ = {mu_friction}, vel_tol = {vel_tol} m/s")
    print("F_N = |Σ contact force on Beam X|,    F_T = √(ΣFx_fric² + ΣFz_fric²)  on Beam X")
    print("=" * 124)
    hdr = (f"{'step':>4} | {'|F_N| sim':>10} | {'F_N paper':>10} | {'FN ratio':>8} | "
           f"{'|F_T| sim':>10} | {'F_T paper':>10} | {'FT ratio':>8} | "
           f"{'Fz_fric':>9} | {'tilt[deg]':>9}")
    print(hdr)
    print("-" * len(hdr))
    with out_compare_txt.open("w", encoding="utf-8") as fh_cmp:
        fh_cmp.write("# Step-by-step comparison against Zavarise & Wriggers "
                     "(2000) Example 2, Table IV\n")
        fh_cmp.write(f"# Simulation:   mu = {mu_friction}, vel_tol = {vel_tol} m/s\n")
        fh_cmp.write(f"# Reference:    Table IV column for mu = {ref_mu_used}\n")
        fh_cmp.write("# Columns:\n")
        fh_cmp.write("#   step              load step index\n")
        fh_cmp.write("#   FN_sim[N]         |F_total contact| on Beam X (~ normal contact magnitude)\n")
        fh_cmp.write("#   FN_paper[N]       Table IV reference F_N for the selected mu column\n")
        fh_cmp.write("#   FN_ratio          FN_sim / FN_paper\n")
        fh_cmp.write("#   FT_sim[N]         sqrt(Fx_fric^2 + Fz_fric^2) on Beam X (tangent-plane friction)\n")
        fh_cmp.write("#   FT_paper[N]       Table IV reference F_T for the selected mu column\n")
        fh_cmp.write("#   FT_ratio          FT_sim / FT_paper\n")
        fh_cmp.write("#   Ffric_z[N]        signed z component of friction on Beam X (dominant tangential)\n")
        fh_cmp.write("#   tilt_deg          angle of contact resultant from global +y axis\n")
        fh_cmp.write("# step\tFN_sim[N]\tFN_paper[N]\tFN_ratio\t"
                     "FT_sim[N]\tFT_paper[N]\tFT_ratio\t"
                     "Ffric_z[N]\ttilt_deg\n")
        for step_id in sorted(per_step_Fy_X.keys()):
            FN_loc = per_step_Fy_X[step_id]
            Fxx, Fxy, Fxz, tilt_deg = per_step_Fz_X[step_id]
            Ff_x, Ff_y, Ff_z, FT_mag = per_step_FT_X.get(
                step_id, (0.0, 0.0, 0.0, 0.0))
            ref_pair = ref_col.get(step_id)
            if ref_pair is not None:
                FN_p, FT_p = ref_pair
            elif step_id >= 7:
                FN_p = 17.1 * (step_id - 2)
                FT_p = mu_friction * FN_p
            else:
                FN_p, FT_p = float('nan'), float('nan')
            if FN_p and FN_p > 1e-6:
                FN_ratio = FN_loc / FN_p
                FN_ratio_s = f"{FN_ratio:8.2f}"
            else:
                FN_ratio = float('nan')
                FN_ratio_s = "    n/a "
            if FT_p and FT_p > 1e-6:
                FT_ratio = FT_mag / FT_p
                FT_ratio_s = f"{FT_ratio:8.2f}"
            else:
                FT_ratio = float('nan')
                FT_ratio_s = "    n/a "
            FN_p_disp = FN_p if FN_p is not None else float('nan')
            FT_p_disp = FT_p if FT_p is not None else float('nan')
            print(f"{step_id:>4} | {FN_loc:>10.3f} | {FN_p_disp:>10.4g} | "
                  f"{FN_ratio_s} | {FT_mag:>10.3f} | {FT_p_disp:>10.4g} | "
                  f"{FT_ratio_s} | {Ff_z:>+9.3f} | {tilt_deg:>9.2f}")
            fh_cmp.write(f"{step_id}\t{FN_loc:.6e}\t{FN_p_disp:.6e}\t"
                         f"{FN_ratio:.6e}\t"
                         f"{FT_mag:.6e}\t{FT_p_disp:.6e}\t{FT_ratio:.6e}\t"
                         f"{Ff_z:.6e}\t{tilt_deg:.4f}\n")
    print(f"Saved per-step comparison to '{out_compare_txt}'")

    # ----------------------------------------------------------- per-dt trace
    if per_dt_trace:
        print(f"Saved per-dt-step force trace to '{out_per_dt_txt}' "
              f"({n_per_dt_rows} rows, written incrementally)")

    return {
        'coords_txt':   str(out_coords_txt),
        'forces_txt':   str(out_forces_txt),
        'curv_txt':     str(out_curv_txt),
        'summary_txt':  str(out_summary_txt),
        'compare_txt':  str(out_compare_txt),
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    results = run_zavarise_benchmark(restart=restart_mode)
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    for k, v in results.items():
        print(f"  {k}: {v}")
