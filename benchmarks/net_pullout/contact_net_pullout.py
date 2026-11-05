#!/usr/bin/env python3
"""
Net strand pull-out — same 5+5 woven net as `contact_net_tensile.py`,
but the simulation starts WITHOUT any initial interpenetration. Tension
is then applied to engage contact, the system is held to relax, and
finally one end of the central x-beam is released and the opposite end
is pulled to drag that strand out of the net.

Contact (selectable with --model):
- 'eberly' (DEFAULT, recommended): regularized 3D Eberly arc-to-arc
  contact (`beamToBeamContactEberly3D`) + smooth Coulomb friction
  (`beamToBeamFrictionEberly3D`).  Stable Newton convergence.
- 'imc':  non-regularized IMC penalty contact (`selfContact`) +
  IMC Coulomb friction (`selfFriction`).  Reference implementation.
  May need d_imc / kc tuning to converge for this geometry — see
  the per-model setup block below.
Default mu = 0.1 for both.

Geometry (initial):
- 5 x-beams of total length L + 2*L_pad = 5L/4 (= 50 mm), shaped as
  z = A * sin(4*pi*(s + L_pad)/L) with L_pad = L/8.  Endpoints at z = 0,
  intersections (at s = 0, L/4, L/2, 3L/4, L) at z = +A, -A, +A, -A, +A.
- 5 straight y-beams of the same extended length, essentially at z = 0
  (with sub-micron parabolic sagitta for the Eberly arc fit).
- All 40 clamp end-nodes sit at z = 0 (single plane): freeing the central
  x-beam's left clamp at Phase D does not snap the strand vertically
  because z = 0 IS its rest-state endpoint position.
- A = 3.1 mm > 2*r = 3 mm  -> initial surface gap = 0.1 mm at every
  intersection.  Small enough that Phase A engagement is short.

Phases (cumulative end-times set by the CLI):
  A (engage)  : pull every x-beam clamp outward at constant rate so the
                woven amplitude drops below 2*r and contact engages at
                all 25 intersections.  The y-beam clamps stay put: their
                strands stay essentially strain-free, only bending a
                little under the contact loads.
  B (relax)   : hold every clamp fixed, let the contact penalty and the
                bending energy redistribute.
  C (release) : on the central x-beam (i = 2), move the LEFT clamp BACK
                to its initial x-position.  This unwinds the local
                tension on the left half before we let it go free, so
                the free release does not snap.
  D (free)    : unfix the two left-most nodes of the central x-beam and
                let the strand settle.
  E (pullout) : pull the RIGHT clamp of the central x-beam further in
                the +x direction.  Contact and friction at each of the
                5 intersections along the strand resist the motion; the
                strand slides through the mesh.

Outputs (same conventions as `contact_net_tensile.py`):
  contact_output_files_net_pullout/
      net_pullout_coords_*.txt        per-snapshot coords
      net_pullout_forces_*.txt        per-snapshot contact forces
      net_pullout_shape_phase{A..E}_*.png   deformed shape at end of each phase
      net_pullout_force_history_*.png       per-intersection F_N(t), F_T(t)
      net_pullout_pull_traction_*.png       pull-force vs. pull-displacement
                                            on the central x-beam right clamp
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

import dismech
from dismech.contact.eberly_contact_3d import (
    circle_from_three_nodes,
    arc_interval_from_nodes,
    extend_arc,
    _broad_phase_pairs,
    _compute_pair_force_and_hessian,
)
from dismech.contact.eberly_batched import batched_pair_forces
import dismech.time_steppers.time_stepper as _ts_mod


def _proc_mem_mb():
    """(VmRSS, VmSize) in MiB from /proc/self/status, or (None, None) if
    unavailable (non-Linux, or the file layout differs)."""
    try:
        rss = vsz = None
        with open("/proc/self/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) / 1024.0
                elif line.startswith("VmSize:"):
                    vsz = int(line.split()[1]) / 1024.0
        return rss, vsz
    except Exception:
        return None, None


# ============================================================================
# PARAMETERS
# ============================================================================

# Beam geometry
L = 0.04                # m, beam length
r_beam = 1.5e-3         # m, beam radius
n_beams_per_dir = 5
# 20 elements between each pair of adjacent intersections (4 spans per beam)
# -> 80 elements per beam -> 81 nodes per beam. SLOW with the default dense
# solver (ndof ~ 3000). Use the CLI to drop to 21 (5/span) or 41 (10/span)
# for quick runs.
n_nodes_per_beam_default = 31

# Initial geometry.  Both x-beams and y-beams are EXTENDED past the
# intersection grid by `L_pad` on each side, and the x-beam wave-shape is
# chosen so its endpoints sit at z = 0 in the SAME PLANE as the y-beam
# endpoints.  This way the clamps for all 10 beams lie in z = 0, and
# freeing the central x-beam's left clamp at Phase D does not snap the
# strand vertically (the free end's rest position IS z = 0).
#
# Shape:
#   x-beam:  z(s) = A * sin(4*pi*(s + L_pad)/L)  for s in [-L_pad, L+L_pad]
#     -> z(-L_pad)        = 0     (clamped endpoint, coplanar with y-beams)
#     -> z(s = j*L/4)     = +-A   (intersections, alternating)
#     -> z(L + L_pad)     = 0     (clamped endpoint)
#   y-beam:  parabolic, z(s) = 0.5 * kappa * (s+L_pad) * (L+L_pad-s)
#     -> z(endpoints)     = 0
#     -> z(s in interior) ≈ kappa * (L/2 + L_pad)^2 / 2   (tiny sagitta)
#
# For sin to vanish at both endpoints with the cosine-like ±A pattern
# at the 5 intersections x = j*L/4, we need L_pad = L/8 (gives 5 half
# wavelengths over the total beam length L + 2*L_pad = 5L/4).
L_pad = 0.04 / 8.0              # m  (= L/8 = 5 mm)  -- DO NOT change naively
# Smaller amplitude than before: keep initial gap small enough that engaging
# contact does not require pulling the clamps too far.  Must satisfy
# weave_amplitude > 2*r_beam = 3 mm so the strands start un-penetrated.
weave_amplitude = 3.1e-3        # m -> initial gap = 0.1 mm
# y-beam initial curvature.  Two competing constraints:
#   (a) κ * edge_length > ~4e-6 rad so the Eberly arc fit
#       `circle_from_three_nodes` passes its collinearity guard
#       (|e_a × e_b|² > 1e-24).  At N=81 the edge is ~0.6 mm long
#       so κ > 6e-3 1/m is needed.
#   (b) The midspan sagitta κ·(L/2 + L_pad)² / 2 must stay well BELOW the
#       weave-amplitude minus 2*r margin, otherwise the y-beam bumps UP
#       into the x-beam's +A peak and creates initial penetration.
#       sagitta_max < A - 2*r = 0.1 mm => κ < ~0.3 1/m.
# Pick κ = 0.1 1/m: sagitta ≈ 31 μm (well below the 100 μm margin) and
# every Eberly arc fit succeeds up to ~N=200.
y_beam_kappa0 = 0.1             # 1/m

# Material defaults (STEEL-like).  CLI flags below override these for the
# "nylon"/textile-strand runs.
#   Steel default:   E = 210 GPa, nu = 0.30, rho = 7850 kg/m^3
#   Nylon override:  E =   1 GPa, nu = 0.40, rho = 1300 kg/m^3
E_young_default = 210e9
nu_default = 0.3
density_default = 7850.0
E_young = E_young_default
nu = nu_default
density = density_default

# Cross-section
A_beam = np.pi * r_beam ** 2
I_beam = np.pi * r_beam ** 4 / 4.0
J_beam = 2.0 * I_beam

# Quasi-static scaling
k_inext = 1.0
m_quasi = 1.0e-3

# Contact model selection
#   'eberly' -> regularized Eberly arc-to-arc + smooth Coulomb friction
#               (`beamToBeamContactEberly3D` + `beamToBeamFrictionEberly3D`)
#   'imc'    -> non-regularized IMC penalty contact + Coulomb friction
#               (`selfContact` + `selfFriction`)
contact_model_default = 'eberly'

# Contact regularization (used by BOTH models, with different roles):
#   Eberly: d_imc is the smooth IMC transition half-width on the surface gap.
#   IMC   : `delta_imc` plays the same role; we keep one knob for both so the
#           CLI controls the same physical regularization width.
d_imc_contact = 5.0e-5          # m = 50 um (initial 500-um clearance => no contact)
# Contact stiffness mode.  None (default) -> use the time-stepper's adaptive
# k_c update (`env._eberly_bb_kc` is left untouched in before_step_fn so the
# Eberly path in time_stepper.py auto-scales k_c by the per-step max nodal
# residual force).  Pass a positive number on the CLI to LOCK k_c at that
# value (used previously; produced 100-260 kN contact-force peaks at end of
# Phase A and Newton divergence at N=31 and N=81).
fixed_kc_default = None
# Phase A timing.  Slower engagement = gentler contact ramp = easier Newton.
t_engage_default = 2.0          # s
# Line search inside the implicit Euler Newton solve.  Big robustness gain
# at high contact penalty, small cost per iter (one extra residual eval).
use_line_search_default = True

# Friction
mu_friction_default = 0.1
vel_tol_default = 1.0e-3        # m/s

# Loading sequence
# Phase A: engage contact by pulling every x-beam clamp outward by
# `end_disp_engage` per side over `t_engage` seconds.  With A_init = 3.5 mm
# and L = 40 mm, end_disp = 5 mm per side reduces the woven amplitude to
# ~1.3 mm (gap ~ -1.7 mm = clear contact engagement at every intersection).
end_disp_engage_default = 5.0e-3        # m per side, x-beams only
t_engage = t_engage_default     # s (CLI: --t-engage)

# Phase B: hold for relaxation.
t_relax_after_engage = 1.0      # s

# Phase C: release tension on the LEFT end of the central x-beam by moving
# it back inward by `end_disp_engage` (i.e. back to its initial position).
t_release = 1.0                 # s

# Phase D: free that end and let the strand settle.
t_free_settle = 0.5             # s

# Phase E: pull the RIGHT clamp of the central x-beam further outward by
# `end_disp_pullout` per side over `t_pullout`.  30 mm slides the strand by
# about 75% of L -- enough to slip past 3-4 intersections.
end_disp_pullout_default = 30.0e-3      # m
# t_pullout and dt are CLI flags below.

# Time stepping
save_interval = 5.0e-2

# Damping.  As of the mass-proportional fix in
# src/dismech/external_forces/damping.py, `eta` is a RATE in 1/s
# (F = -eta * m_i * v_i, mass-proportional/Rayleigh damping), NOT a
# N.s/m-per-node coefficient -- the per-DOF coefficient c_i = eta * m_i
# now shrinks with finer meshes (m_i ~ 1/(n_nodes_per_beam - 1)), unlike
# the old flat-per-node scheme where c_i was the SAME at every N.  That
# meant the old scheme accidentally over-damped fine meshes relative to
# coarse ones -- which also happened to help Newton converge at exactly
# the N values that are hardest to converge.  To not lose that at high N,
# this default is anchored to the HARDEST mesh in the campaign (N=131,
# steel recipe: old c_i = 0.005*50 = 0.25 N.s/m/node there) plus ~20%
# margin, so every N now gets AT LEAST as much per-node damping as the
# old scheme gave its worst case (coarser N get substantially more --
# intentional, a uniformly more-damped campaign).  For the soft (E=1e7,
# rho=1100) campaign recipe, use --damping-eta 1.0e9 instead (see
# run_net_pullout_mesh_campaign.sh), anchored the same way against its
# N=131 case (old c_i = 0.005*500 = 2.5 N.s/m/node).
damping_eta = 1.4e7

# Solver tolerances
solver_tol = 1.0e-4
solver_max_iter = 200


# ============================================================================
# CLI
# ============================================================================

parser = argparse.ArgumentParser(
    description="Net strand pull-out: engage contact via tension, then "
                "release one end of the central x-beam and pull it out.")
parser.add_argument("n_nodes_per_beam", nargs="?", type=int,
                    default=n_nodes_per_beam_default,
                    help="Number of nodes per beam (default: %(default)s).")
parser.add_argument("--fixed-kc", type=float, default=fixed_kc_default,
                    metavar="VALUE",
                    help="Lock the contact penalty stiffness k_c at VALUE. "
                         "Default: unset = ADAPTIVE k_c "
                         "(time_stepper.py auto-scales k_c by the per-step "
                         "max nodal residual force; recommended).")
parser.add_argument("--end-disp-engage", type=float,
                    default=end_disp_engage_default, metavar="METRES",
                    help="Phase A outward end displacement per side on every "
                         "x-beam clamp (default: %(default)s m).")
parser.add_argument("--t-engage", type=float, default=t_engage_default,
                    metavar="SECONDS",
                    help="Phase A duration. Slower engagement = gentler "
                         "contact ramp = easier Newton convergence "
                         "(default: %(default)s s).")
parser.add_argument("--end-disp-pullout", type=float,
                    default=end_disp_pullout_default, metavar="METRES",
                    help="Phase E pull distance applied to the RIGHT clamp "
                         "of the central x-beam (default: %(default)s m).")
parser.add_argument("--use-line-search", dest="use_line_search",
                    action="store_true", default=use_line_search_default,
                    help="Enable Newton line search inside the implicit "
                         "Euler step (default: %(default)s).")
parser.add_argument("--no-line-search", dest="use_line_search",
                    action="store_false",
                    help="Disable Newton line search.")
parser.add_argument("--line-search-iters", type=int, default=20,
                    metavar="N",
                    help="Number of Newton iterations of unshielded full "
                         "steps BEFORE line search activates (dismech "
                         "default: 20).  For h-scale + tight contact the "
                         "default lets the residual explode by 3-4 orders "
                         "of magnitude before line search catches it; set "
                         "to 0 or 2 to activate line search from the first "
                         "iteration.  Requires --use-line-search.")
parser.add_argument("--no-plot", dest="make_plots",
                    action="store_false", default=True,
                    help="Skip all matplotlib figure generation; only write "
                         "txt outputs.  Use on Ruche to avoid the "
                         "matplotlib Path.__deepcopy__ infinite recursion.")
parser.add_argument("--mu", type=float, default=mu_friction_default,
                    metavar="MU",
                    help="Coulomb friction coefficient (default: %(default)s).")
parser.add_argument("--vel-tol", type=float, default=vel_tol_default,
                    metavar="VTOL",
                    help="Smooth Coulomb gate velocity tolerance, m/s "
                         "(default: %(default)s).")
parser.add_argument("--newton-tol", type=float, default=solver_tol,
                    metavar="TOL",
                    help="Newton solver residual tolerance (default: "
                         "%(default).0e).  Applied as tol/ftol/dtol.  "
                         "Tightening (e.g. 1e-7) reduces the Phase-B "
                         "kc-adaptation jitter that comes from per-step "
                         "Newton residual noise; the intrinsic arc-fit "
                         "asymmetry that drives the drift is unaffected.")
parser.add_argument("--max-newton-iter", type=int, default=solver_max_iter,
                    metavar="N",
                    help="Newton iteration cap per step before raising "
                         "'Iteration limit reached' (default: %(default)s). "
                         "Raise this for fine meshes at high pull-out "
                         "displacement, where a step can sit just above "
                         "newton_tol for many extra iterations before "
                         "clearing it.")
parser.add_argument("--tag", type=str, default="", metavar="SUFFIX",
                    help="Optional suffix to append to output filenames.")
parser.add_argument("--model", type=str, default=contact_model_default,
                    choices=['eberly', 'imc'],
                    help="Contact model.  'eberly' (default): regularized "
                         "Eberly arc-to-arc 3D contact + smooth Coulomb "
                         "friction.  'imc': non-regularized IMC penalty "
                         "self-contact (also used for cross-beam pairs "
                         "via n_edges_beam_A) + IMC friction.")
parser.add_argument("--t-pullout", type=float, default=4.0, metavar="SECONDS",
                    help="Phase E duration. Slower pullout = smaller per-step "
                         "displacement = less penalty penetration "
                         "(default: %(default)s s).")
parser.add_argument("--dt-pullout", type=float, default=None, metavar="SECONDS",
                    help="Override dt during Phase E (pullout). The loop "
                         "switches to this dt automatically at t = T_D.")
parser.add_argument("--dt-release", type=float, default=None, metavar="SECONDS",
                    help="Override dt during Phases C and D (release-left + "
                         "free-left).  Finer dt here helps the Newton solver "
                         "absorb the discontinuous BC change when the left "
                         "clamp is released.  Default: same as --dt.")
parser.add_argument("--dt", type=float, default=1.0e-2, metavar="SECONDS",
                    help="Global time step (default: %(default)s s).")
parser.add_argument("--resume-pullout", type=str, default=None, metavar="PATH",
                    help="Skip phases A-D and resume from a checkpoint "
                         "saved at end of Phase D. The path is an .npz file "
                         "produced automatically at end of Phase D by an "
                         "earlier run with the same mesh.")
parser.add_argument("--resume-settle-time", type=float, default=0.0,
                    metavar="SECONDS",
                    help="On --resume-pullout, hold ALL boundary conditions "
                         "fixed for this many sim-seconds before starting "
                         "Phase E, so Newton can relax to a self-consistent "
                         "quasi-static equilibrium.  Useful when the "
                         "checkpoint lacks the full DER material state (a1, "
                         "ref_twist etc.) and the elastic force is "
                         "inconsistent with the loaded q.  The settled "
                         "equilibrium is NOT identical to the main run's "
                         "end-of-D state -- it's a valid neighbouring "
                         "equilibrium.  Default 0 = no settling.")
parser.add_argument("--reconstruct-checkpoint", type=str, default=None,
                    metavar="COORDS_PATH",
                    help="Recover the missing DER material frames (a1, a2, "
                         "m1, m2, ref_twist) for a legacy checkpoint by "
                         "REPLAYING parallel-transport through the A->D "
                         "coords trajectory saved to COORDS_PATH (usually "
                         "'net_pullout_coords_<stem>.txt'), then write an "
                         "extended checkpoint next to --resume-pullout with "
                         "the reconstructed DER state + the legacy "
                         "checkpoint's q, u, kc and counters.  The script "
                         "exits after writing the extended checkpoint, does "
                         "NOT run the simulation.  Requires --resume-pullout.")
parser.add_argument("--synthesize-at-time", type=float, default=None,
                    metavar="T_SEC",
                    help="Instead of reading q/u from the legacy checkpoint, "
                         "SYNTHESIZE a fresh checkpoint from the coords file "
                         "at time T_SEC.  Used when the legacy checkpoint's "
                         "state at T_D=3.5 is known to be corrupted (e.g. "
                         "strand-through-strand jumps during the free-left "
                         "transient) and the last verifiably-good snapshot "
                         "is at t=T_SEC.  Positions come from the coords "
                         "file at T_SEC; velocities are estimated by "
                         "finite-differencing consecutive snapshots; theta "
                         "is set to 0 (circular isotropic cross-section, "
                         "twist stays near 0); kc is reset to the imc "
                         "baseline and will re-adapt over the first few "
                         "Newton steps.  Requires --reconstruct-checkpoint.")
parser.add_argument("--monotone-kc", action="store_true", default=False,
                    help="LOCAL patch (script-only, no dismech source "
                         "change): make Eberly's adaptive k_c monotone -- "
                         "cache the maximum k_c ever reached, and if "
                         "time_stepper.py's Eberly decay branch drops it "
                         "below that max, restore it before the next step. "
                         "Minor effect; the decay branch rarely fires "
                         "during real contact.  Reversible.")
parser.add_argument("--kc-under-relax", type=float, default=0.0, metavar="ALPHA",
                    help="LOCAL patch (script-only): after every stepper.step, "
                         "smooth env._eberly_bb_kc via kc <- ALPHA * kc_prev + "
                         "(1-ALPHA) * kc_new, on top of dismech's own 0.9/0.1 "
                         "adaptive update.  ALPHA=0 (default) disables this. "
                         "ALPHA=0.9 makes the effective kc adaptation ~10x "
                         "slower, ALPHA=0.99 ~100x slower.  Purpose: break "
                         "the discrete-time period-2 limit cycle that shows "
                         "up in static equilibrium (Phase B) where the "
                         "kc<->gap coupled map has two attractive fixed "
                         "points.  Reversible: drop the flag to restore "
                         "dismech's default.")
parser.add_argument("--eberly-h-scale", action="store_true", default=False,
                    help="LOCAL patch (script-only): amplify Eberly contact "
                         "force by 1/h = 1/beam_radius, mirroring IMC's "
                         "`grad_E *= scale * k_c` line where scale=1/h. "
                         "This is the ACTUAL fundamental reason IMC's "
                         "equilibrium overlap is ~700x smaller than "
                         "Eberly's for the same nominal k_c.  With this "
                         "flag, Eberly's effective force per pair matches "
                         "IMC's convention -- so the equilibrium overlap "
                         "should drop from mm-scale to um-scale.  "
                         "Reversible: drop the flag to restore Eberly's "
                         "default (dismech-unmodified) force scaling.")
parser.add_argument("--debug-fallback-counts", action="store_true", default=False,
                    help="DIAGNOSTIC (script-only): print n_candidates/"
                         "n_arc_fit_fail/n_fallback_used/gap_min on EVERY "
                         "Newton iteration (not one-shot like the existing "
                         "warnings), to check whether a near-collinear "
                         "triplet is flipping between the real arc-fit and "
                         "the straight-line fallback from one Newton "
                         "iterate to the next -- a plausible mechanism for "
                         "an exact-period error limit cycle, since the two "
                         "paths give different local force/Hessian "
                         "contributions for the same pair. Verbose; use "
                         "only for a short targeted run (e.g. --resume-"
                         "pullout near a known stuck point).")
parser.add_argument("--debug-fallback-after", type=float, default=0.0,
                    metavar="SECONDS",
                    help="Only start printing --debug-fallback-counts once "
                         "simulated time reaches SECONDS (default 0 = print "
                         "from the start). Use this to avoid a huge log "
                         "when running a long stretch to reach a known "
                         "trouble point -- e.g. --debug-fallback-after 11.8 "
                         "to stay quiet until just before a crash near 12.0.")
parser.add_argument("--checkpoint-interval", type=float, default=0.0,
                    metavar="SECONDS",
                    help="Periodically save a FULL-STATE checkpoint (same "
                         "contents as the one-shot end-of-Phase-D save: q, "
                         "u, a1/a2/m1/m2/ref_twist/tau, kc, counters) every "
                         "SECONDS of simulated time during Phase E, to "
                         "'<checkpoint_path>_t<T>.npz'. Default 0 = "
                         "disabled (only the single Phase-D checkpoint is "
                         "saved, as before). Lets a future --resume-pullout "
                         "restart EXACTLY (no synthesis approximation) near "
                         "any point reached by a run that used this flag, "
                         "instead of needing to re-run the whole phase.")
parser.add_argument("--stop-after-phase-d", action="store_true", default=False,
                    help="Exit the loop right after writing the end-of-Phase-D "
                         "checkpoint. Use as stage 1 of a two-stage cluster "
                         "pipeline where stage 2 resumes Phase E with a "
                         "refined dt via --resume-pullout.")
parser.add_argument("--free-y-pullout", action="store_true", default=False,
                    help="Do NOT impose y on the RIGHT clamp of the central "
                         "x-beam during Phase E: only x is prescribed "
                         "(pullout displacement) and z is held; y-DOF of "
                         "the two right-clamp nodes is freed so the strand "
                         "can self-align laterally as it slides. Applied "
                         "once at Phase E start (or on --resume-pullout).")
parser.add_argument("--free-y-from-start", action="store_true", default=False,
                    help="Do NOT impose y on the RIGHT clamp of the central "
                         "x-beam from t=0 (i.e. throughout Phases A-B-C-D). "
                         "Same BC change as --free-y-pullout but applied at "
                         "the initial BC setup rather than at Phase E start. "
                         "Use with --stop-after-phase-d to run a Phase-A-to-D "
                         "trial that shares the pullout's y-free BC.")
parser.add_argument("--youngs", type=float, default=E_young_default,
                    metavar="PA",
                    help="Young's modulus in Pa "
                         "(default: %(default).3e -- steel).  "
                         "Use 1e9 for a nylon/textile strand.")
parser.add_argument("--nu", type=float, default=nu_default, metavar="NU",
                    help="Poisson's ratio (default: %(default)s -- steel).  "
                         "Use 0.40 for a polymer strand.")
parser.add_argument("--density", type=float, default=density_default,
                    metavar="KG_M3",
                    help="Density in kg/m^3 (default: %(default).0f -- steel). "
                         " Use 1300 for a textile strand.")
parser.add_argument("--d-imc", type=float, default=None, metavar="METRES",
                    help="Override the IMC smooth-zone half-width d_imc "
                         "(=delta in the IMC energy formula). "
                         "Default: 5e-5 m (baked-in constant). For the IMC "
                         "path, 5e-5 makes k_1=15/delta too stiff and "
                         "Newton diverges past shallow contact; d_imc = 2e-4 "
                         "gives a sweet spot for the current geometry.")
parser.add_argument("--solver", type=str, default='np',
                    choices=['np', 'pardiso'],
                    help="Linear solver inside Newton.  'np' (default): "
                         "scipy/numpy dense solve, O(ndof^3) per iter. "
                         "'pardiso': sparse direct solver via pypardiso "
                         "(requires `pip install pypardiso`).  At N=51, "
                         "ndof ~ 1000 and dense solve already dominates "
                         "wall time; at N=81 (ndof ~ 1600) pardiso is "
                         "the only way to finish in a 3-day allocation.")
parser.add_argument("--weave-amplitude", type=float, default=None,
                    metavar="METRES",
                    help="Sinusoidal weave amplitude in metres. "
                         "Default (None) keeps the built-in 3.1e-3 m "
                         "(initial surface gap = 0.1 mm). Set to 3.0e-3 "
                         "for the flattest legal weave (gap = 0). Values "
                         "below 2*r_beam=3e-3 create initial interpenetration.")
parser.add_argument("--damping-eta", type=float, default=None,
                    metavar="ETA",
                    help="Mass-proportional damping RATE in 1/s "
                         "(F = -eta * m_i * v_i per DOF; see "
                         "src/dismech/external_forces/damping.py). The "
                         "per-DOF coefficient c_i = eta * m_i shrinks at "
                         "finer N (unlike the old flat per-node "
                         "coefficient), so this is anchored high enough "
                         "that every N gets AT LEAST as much per-node "
                         "damping as the old scheme gave its hardest-to-"
                         f"converge case (N=131). Default (None) keeps the "
                         f"built-in {damping_eta:g} for this script's "
                         "default (steel) recipe -- use --damping-eta 1.0e9 "
                         "for the soft (E=1e7, rho=1100) campaign recipe. "
                         "Larger values suppress dynamic transients that "
                         "push Newton to its iter limit.")
parser.add_argument("--alternating-weave", action="store_true", default=False,
                    help="Plain-weave variant: phase-shift successive "
                         "x-beams by pi so the crossings alternate "
                         "above/below ALONG each y-beam too (checkerboard "
                         "parity), as in the woven-net references. Outputs "
                         "go to contact_output_files_net_pullout_alternated/.")
cli_args = parser.parse_args()
n_nodes_per_beam = int(cli_args.n_nodes_per_beam)
fixed_kc = float(cli_args.fixed_kc) if cli_args.fixed_kc is not None else None
# Treat 0 / negative on --fixed-kc as "use adaptive" (lets a cluster sbatch
# pass FIXED_KC=0 to switch back from a previously-fixed-kc run).
if fixed_kc is not None and fixed_kc <= 0.0:
    fixed_kc = None
if cli_args.d_imc is not None:
    d_imc_contact = float(cli_args.d_imc)
# Material overrides from CLI (defaults are the module-level steel constants).
E_young = float(cli_args.youngs)
nu = float(cli_args.nu)
density = float(cli_args.density)
if cli_args.weave_amplitude is not None:
    weave_amplitude = float(cli_args.weave_amplitude)
if cli_args.damping_eta is not None:
    damping_eta = float(cli_args.damping_eta)
alternating_weave = bool(cli_args.alternating_weave)
end_disp_engage = float(cli_args.end_disp_engage)
t_engage = float(cli_args.t_engage)
end_disp_pullout = float(cli_args.end_disp_pullout)
t_pullout = float(cli_args.t_pullout)
dt = float(cli_args.dt)
resume_pullout_path = cli_args.resume_pullout
resume_settle_time = float(cli_args.resume_settle_time)
reconstruct_from_coords = cli_args.reconstruct_checkpoint
synthesize_at_time = cli_args.synthesize_at_time
monotone_kc = bool(cli_args.monotone_kc)
kc_under_relax = float(cli_args.kc_under_relax)
eberly_h_scale = bool(cli_args.eberly_h_scale)
debug_fallback_counts = bool(cli_args.debug_fallback_counts)
debug_fallback_after = float(cli_args.debug_fallback_after)
_debug_current_t = [0.0]   # updated by the main loop each step; read by
                           # _multi_pair to gate --debug-fallback-after
checkpoint_interval = float(cli_args.checkpoint_interval)
stop_after_phase_d = bool(cli_args.stop_after_phase_d)
free_y_pullout = bool(cli_args.free_y_pullout)
free_y_from_start = bool(cli_args.free_y_from_start)
solver_name = str(cli_args.solver)
dt_pullout = float(cli_args.dt_pullout) if cli_args.dt_pullout is not None else None
dt_release = float(cli_args.dt_release) if cli_args.dt_release is not None else None
if resume_pullout_path and dt_pullout is not None:
    dt = dt_pullout   # in resume mode, the global dt IS the Phase E dt
use_line_search = bool(cli_args.use_line_search)
line_search_iters = int(cli_args.line_search_iters)
solver_max_iter = int(cli_args.max_newton_iter)
make_plots = bool(cli_args.make_plots)
mu_friction = float(cli_args.mu)
vel_tol = float(cli_args.vel_tol)
solver_tol = float(cli_args.newton_tol)
contact_model = str(cli_args.model).lower()
output_tag = cli_args.tag.strip()
if output_tag and not output_tag.startswith("_"):
    output_tag = "_" + output_tag

if (n_nodes_per_beam - 1) % 5 != 0:
    print(
        f"WARNING: (n_nodes_per_beam - 1) = {n_nodes_per_beam - 1} is NOT a "
        f"multiple of 5 (the new extended-beam geometry needs intersections "
        f"at x_levels[j] = j*L/4 to coincide with mesh nodes on a beam that "
        f"spans 5L/4). Intersection nodes will be the NEAREST mesh nodes; "
        f"recommended N = 11, 21, 31, 41, 51, 81, 101.\n")

# Cumulative phase end-times.  With --resume-settle-time > 0 we extend
# Phase D so the resumed sim holds all BCs fixed for that many seconds
# before the pullout begins -- lets Newton re-settle after the checkpoint
# load (useful when the loaded state's DER material frames are missing
# or partially inconsistent with the loaded q).
T_A = t_engage
T_B = T_A + t_relax_after_engage
T_C = T_B + t_release
T_D = T_C + t_free_settle + (resume_settle_time if resume_pullout_path else 0.0)
T_E = T_D + t_pullout
total_time = T_E


# ============================================================================
# GEOMETRY  (identical to contact_net_tensile.py)
# ============================================================================

def make_woven_x_beam(y_pos, n_nodes, length, amplitude, pad):
    """X-beam from s = -pad to s = length + pad, sin-shaped so endpoints
    sit at z = 0 (coplanar with y-beam endpoints). Intersections at
    s = 0, L/4, L/2, 3L/4, L give z = +A, -A, +A, -A, +A when pad = L/8."""
    s = np.linspace(-pad, length + pad, n_nodes)
    z_vals = amplitude * np.sin(4.0 * np.pi * (s + pad) / length)
    nodes = np.zeros((n_nodes, 3))
    nodes[:, 0] = s
    nodes[:, 1] = y_pos
    nodes[:, 2] = z_vals
    edges = np.array([[i, i + 1] for i in range(n_nodes - 1)], dtype=int)
    face_nodes = np.empty((0, 3), dtype=int)
    return dismech.Geometry(nodes, edges, face_nodes)


def make_straight_y_beam(x_pos, n_nodes, length, kappa_tiny, pad):
    """Y-beam from s = -pad to s = length + pad with tiny parabolic
    sagitta that vanishes at both extended endpoints."""
    s = np.linspace(-pad, length + pad, n_nodes)
    z_vals = 0.5 * kappa_tiny * (s + pad) * (length + pad - s)
    nodes = np.zeros((n_nodes, 3))
    nodes[:, 0] = x_pos
    nodes[:, 1] = s
    nodes[:, 2] = z_vals
    edges = np.array([[i, i + 1] for i in range(n_nodes - 1)], dtype=int)
    face_nodes = np.empty((0, 3), dtype=int)
    return dismech.Geometry(nodes, edges, face_nodes)


def build_net_geometry(n_nodes_per_beam):
    y_levels = np.linspace(0.0, L, n_beams_per_dir)
    x_levels = np.linspace(0.0, L, n_beams_per_dir)

    x_geos = []
    x_beam_start = []
    cursor = 0
    for i in range(n_beams_per_dir):
        # Plain (alternated) weave: successive x-beams are phase-shifted
        # by pi (amplitude sign flip), so that ALONG EACH y-beam the
        # crossing strands alternate above/below (checkerboard parity),
        # as in the woven-net references.  The historical default keeps
        # all x-beams in phase (identical +-+-+ rows).
        amp_i = weave_amplitude * ((-1) ** i if alternating_weave else 1.0)
        geo = make_woven_x_beam(
            y_pos=y_levels[i], n_nodes=n_nodes_per_beam,
            length=L, amplitude=amp_i, pad=L_pad)
        x_geos.append(geo)
        x_beam_start.append(cursor)
        cursor += n_nodes_per_beam
    n_x_nodes_total = cursor

    y_geos = []
    y_beam_start = []
    for j in range(n_beams_per_dir):
        geo = make_straight_y_beam(
            x_pos=x_levels[j], n_nodes=n_nodes_per_beam,
            length=L, kappa_tiny=y_beam_kappa0, pad=L_pad)
        y_geos.append(geo)
        y_beam_start.append(cursor)
        cursor += n_nodes_per_beam
    n_y_nodes_total = cursor - n_x_nodes_total

    all_geos = x_geos + y_geos
    combined_nodes = np.vstack([g.nodes for g in all_geos])
    edges_list = []
    offset = 0
    for g in all_geos:
        edges_list.append(g.edges + offset)
        offset += g.nodes.shape[0]
    combined_edges = np.vstack(edges_list)
    combined_faces = np.empty((0, 3), dtype=int)
    combined_geo = dismech.Geometry(combined_nodes, combined_edges, combined_faces)

    intersection_nodes = np.zeros((n_beams_per_dir, n_beams_per_dir, 2), dtype=int)
    for i in range(n_beams_per_dir):
        x_along = combined_nodes[x_beam_start[i]:x_beam_start[i] + n_nodes_per_beam, 0]
        for j in range(n_beams_per_dir):
            k_x = int(np.argmin(np.abs(x_along - x_levels[j])))
            intersection_nodes[i, j, 0] = x_beam_start[i] + k_x
    for j in range(n_beams_per_dir):
        y_along = combined_nodes[y_beam_start[j]:y_beam_start[j] + n_nodes_per_beam, 1]
        for i in range(n_beams_per_dir):
            k_y = int(np.argmin(np.abs(y_along - y_levels[i])))
            intersection_nodes[i, j, 1] = y_beam_start[j] + k_y

    info = dict(
        x_beam_start=np.array(x_beam_start, dtype=int),
        y_beam_start=np.array(y_beam_start, dtype=int),
        x_beam_n=n_nodes_per_beam,
        y_beam_n=n_nodes_per_beam,
        n_x_nodes_total=n_x_nodes_total,
        n_y_nodes_total=n_y_nodes_total,
        x_levels=x_levels,
        y_levels=y_levels,
        intersection_nodes=intersection_nodes,
    )
    return combined_geo, info


# ============================================================================
# MULTI-PAIR EBERLY 3D CONTACT WRAPPER
# (kept self-contained so this script does not depend on the tensile case)
# ============================================================================

def _safe_circle_from_three_nodes(n0, n1, n2):
    """Wrap `circle_from_three_nodes` so that near-collinear triplets fall
    back to a SYNTHETIC very-large-radius arc (effectively a straight
    cylinder), instead of raising ValueError and dropping the contact pair.

    The arc lies in a plane spanned by the chord (n2 - n0) and an arbitrary
    perpendicular direction.  Radius R_large = 1e6 * |chord| makes the arc
    indistinguishable from a straight segment at any reasonable beam scale,
    but provides a well-defined (C, R, n_axis) tuple for downstream Eberly
    code that expects an arc.
    """
    try:
        return circle_from_three_nodes(n0, n1, n2)
    except (ValueError, np.linalg.LinAlgError):
        n0 = np.asarray(n0, float)
        n2 = np.asarray(n2, float)
        chord = n2 - n0
        chord_len = float(np.linalg.norm(chord))
        if chord_len < 1e-15:
            # Truly degenerate: nodes coincide.  Re-raise so the caller skips.
            raise ValueError("Triplet endpoints coincide; cannot synthesise arc.")
        chord_dir = chord / chord_len
        # Pick a perpendicular direction
        if abs(chord_dir[2]) < 0.9:
            seed = np.array([0., 0., 1.])
        else:
            seed = np.array([1., 0., 0.])
        perp = seed - np.dot(seed, chord_dir) * chord_dir
        perp /= np.linalg.norm(perp)
        R_large = 1.0e6 * chord_len
        midpoint = 0.5 * (n0 + n2)
        C = midpoint + R_large * perp
        n_axis = np.cross(chord_dir, perp)
        n_axis /= np.linalg.norm(n_axis)
        return C, R_large, n_axis


def make_multi_pair_eberly_contact(beam_pairs):
    beam_pairs = tuple(beam_pairs)
    # Diagnostic counters: track FALLBACK use so we can warn if every candidate
    # pair is going through the straight-line fallback (typical symptom:
    # initial y_beam_kappa0 too small for the mesh density).
    skip_warned = [False]
    fallback_warned = [False]

    def _multi_pair(robot, q, beam_radius, k_c=1.0,
                    n_nodes_beam_A=None, d_imc=0.0008,
                    straight_R_threshold=None, **_ignored):
        # `straight_R_threshold` and any other future kwargs are accepted for
        # forward-compatibility with dismech's core wrapper (see the arc-fit
        # fallback comment in time_stepper.py).  We already handle the
        # collinear-triplet fallback via _safe_circle_from_three_nodes, so
        # this hook is ignored here.
        node_dof = robot.node_dof_indices
        ndof = q.shape[0]
        node_pos = q[node_dof]

        F = np.zeros(ndof)
        J = np.zeros((ndof, ndof))
        all_gaps = []
        n_candidates = 0       # pairs that passed the bounding-sphere filter
        n_arc_fit_fail = 0     # pairs killed by truly-degenerate triplets
        n_fallback_used = 0    # pairs where the straight-line fallback fired

        K_imc = 15.0 / d_imc
        beam_R_eff = 2.0 * beam_radius
        interaction_range = beam_R_eff + d_imc

        # ---- gather broad-phase survivors across ALL beam pairs ----------
        # (order preserved: beam_pairs sequence, then row-major (i, j))
        cand = []          # rows: (sA, i, num_seg_A, sB, j, num_seg_B)
        for (sA, nA, sB, nB) in beam_pairs:
            num_seg_A = nA - 2
            num_seg_B = nB - 2
            if num_seg_A <= 0 or num_seg_B <= 0:
                continue
            bb_i, bb_j = _broad_phase_pairs(
                node_pos, sA + np.arange(num_seg_A), sB + np.arange(num_seg_B),
                interaction_range)
            for i, j in zip(bb_i.tolist(), bb_j.tolist()):
                cand.append((sA, i, num_seg_A, sB, j, num_seg_B))
        n_candidates = len(cand)

        # ---- batched narrow phase over the whole survivor list -----------
        bres = None
        if cand:
            arr = np.array(cand, dtype=int)
            bres = batched_pair_forces(
                node_pos, arr[:, 0] + arr[:, 1], arr[:, 3] + arr[:, 4],
                beam_radius, d_imc, K_imc,
                arr[:, 1] == 0, arr[:, 1] == arr[:, 2] - 1,
                arr[:, 4] == 0, arr[:, 4] == arr[:, 5] - 1)

        for k, (sA, i, num_seg_A, sB, j, num_seg_B) in enumerate(cand):
            ni0, ni1, ni2 = sA + i, sA + i + 1, sA + i + 2
            nj0, nj1, nj2 = sB + j, sB + j + 1, sB + j + 2

            if not bres['fallback'][k]:
                if not bres['active'][k]:
                    continue
                energy = bres['energy'][k]
                force_local = bres['force'][k]
                hess_local = bres['hessian'][k]
                gap = bres['gap'][k]
                s1, s2 = bres['s_i1'][k], bres['s_i2'][k]
            else:
                NODES1 = (node_pos[ni0], node_pos[ni1], node_pos[ni2])
                NODES2 = (node_pos[nj0], node_pos[nj1], node_pos[nj2])

                # Scalar path (special-case branches): real Eberly arc fit,
                # with the synthetic huge-radius fallback on near-collinear
                # triplets so the pair still contributes a force.
                try:
                    try:
                        C1, R1, n1_ax = circle_from_three_nodes(*NODES1)
                    except (ValueError, np.linalg.LinAlgError):
                        C1, R1, n1_ax = _safe_circle_from_three_nodes(*NODES1)
                        n_fallback_used += 1
                    try:
                        C2, R2, n2_ax = circle_from_three_nodes(*NODES2)
                    except (ValueError, np.linalg.LinAlgError):
                        C2, R2, n2_ax = _safe_circle_from_three_nodes(*NODES2)
                        n_fallback_used += 1
                except (ValueError, np.linalg.LinAlgError):
                    n_arc_fit_fail += 1
                    continue

                arc1 = arc_interval_from_nodes(*NODES1, C1, R1, n1_ax)
                arc2 = arc_interval_from_nodes(*NODES2, C2, R2, n2_ax)
                arc1_ext = extend_arc(arc1)
                arc2_ext = extend_arc(arc2)

                (energy, force_local, hess_local,
                 gap, s1, s2) = _compute_pair_force_and_hessian(
                    NODES1, NODES2, arc1_ext, arc2_ext,
                    beam_radius, d_imc, K_imc,
                    i == 0, i == num_seg_A - 1,
                    j == 0, j == num_seg_B - 1,
                )

                if abs(energy) < 1e-30:
                    continue

            node_indices = [ni0, ni1, ni2, nj0, nj1, nj2]
            global_dof = np.concatenate(
                [node_dof[n] for n in node_indices])
            F[global_dof] += force_local
            J[np.ix_(global_dof, global_dof)] += hess_local

            all_gaps.append({
                'segment_pair': f'(A{sA}+{i},B{sB}+{j})',
                'gap': gap, 's_i1': s1, 's_i2': s2,
            })

        # Optional --eberly-h-scale: multiply by 1/h = 1/beam_radius to
        # match IMC's `grad_E *= scale * k_c` line in contact_energy.py.
        # See analysis above; this is the ACTUAL cause of the 700x
        # equilibrium-overlap difference between the two models.
        if eberly_h_scale:
            h_amp = 1.0 / beam_radius
            F *= k_c * h_amp
            J *= k_c * h_amp
        else:
            F *= k_c
            J *= k_c
        gap_min = min((g['gap'] for g in all_gaps), default=float('inf'))
        # One-shot warning: truly-degenerate triplets (endpoints coincide)
        # still drop pairs entirely; flag it on first occurrence.
        if (not skip_warned[0]
                and n_candidates > 0
                and n_arc_fit_fail == n_candidates):
            print(f"\n  *** WARNING: Eberly arc fit AND straight fallback "
                  f"both failed for ALL {n_candidates} in-range candidate "
                  f"pairs (truly-degenerate endpoints).  Contact silently "
                  f"disabled.  Check geometry for coincident nodes.\n")
            skip_warned[0] = True
        # One-shot heads-up: heavy reliance on the straight-line fallback
        # means a beam is essentially straight.  Not an error -- the contact
        # is still active -- but bumping kappa0 gives cleaner arc fits.
        if (not fallback_warned[0]
                and n_candidates > 0
                and n_fallback_used >= n_candidates):
            print(f"\n  *** NOTE: Eberly straight-line fallback used for "
                  f">= {n_candidates} arc-fit attempts at this iter "
                  f"(one or both beams are nearly straight).  Contact is "
                  f"still applied via the synthetic large-radius arc.\n")
            fallback_warned[0] = True
        if debug_fallback_counts and _debug_current_t[0] >= debug_fallback_after:
            print(f"    [debug] n_candidates={n_candidates} "
                  f"n_arc_fit_fail={n_arc_fit_fail} "
                  f"n_fallback_used={n_fallback_used} "
                  f"gap_min={gap_min:.6e}")
        return F, J, gap_min, all_gaps

    return _multi_pair


# ============================================================================
# CONTACT FORCE EXTRACTION (per intersection)
# ============================================================================

def extract_intersection_forces(F_full, node_dof_indices, intersection_nodes):
    nI = intersection_nodes.shape[0]
    nJ = intersection_nodes.shape[1]
    FN_x = np.zeros((nI, nJ))
    FN_y = np.zeros((nI, nJ))
    FT_x = np.zeros((nI, nJ))
    FT_y = np.zeros((nI, nJ))
    for i in range(nI):
        for j in range(nJ):
            n_x = intersection_nodes[i, j, 0]
            n_y = intersection_nodes[i, j, 1]
            f_x = F_full[node_dof_indices[n_x]]
            f_y = F_full[node_dof_indices[n_y]]
            FN_x[i, j] = abs(f_x[2])
            FN_y[i, j] = abs(f_y[2])
            FT_x[i, j] = np.sqrt(f_x[0] ** 2 + f_x[1] ** 2)
            FT_y[i, j] = np.sqrt(f_y[0] ** 2 + f_y[1] ** 2)
    return FN_x, FN_y, FT_x, FT_y


# ============================================================================
# SIMULATION
# ============================================================================

def run_pullout_simulation():
    print("=" * 70)
    print(f"NET STRAND PULL-OUT (3D Eberly, mu = {mu_friction})")
    print("=" * 70)
    print(f"contact_model        = {contact_model}")
    print(f"n_nodes_per_beam     = {n_nodes_per_beam}")
    print(f"weave_amplitude A0   = {weave_amplitude * 1e3:.3f} mm "
          f"(initial gap = {(weave_amplitude - 2 * r_beam) * 1e3:.3f} mm)")
    print(f"weave pattern        = "
          f"{'ALTERNATED (plain weave, checkerboard)' if alternating_weave else 'in-phase (historical)'}")
    print(f"beam_radius r        = {r_beam} m")
    kc_mode = "ADAPTIVE (time-stepper auto-scales)" if fixed_kc is None \
              else f"FIXED at {fixed_kc:g}"
    print(f"k_c mode             = {kc_mode}")
    print(f"line search          = {'ON' if use_line_search else 'OFF'} "
          f"(activates after iter {line_search_iters})")
    print(f"d_imc                = {d_imc_contact} m")
    print(f"mu / vel_tol         = {mu_friction} / {vel_tol} m/s")
    print(f"newton_tol           = {solver_tol:g}")
    print(f"max_newton_iter      = {solver_max_iter}")
    print(f"t_engage             = {t_engage} s")
    print(f"end_disp_engage      = {end_disp_engage * 1e3:.3f} mm/side  (Phase A)")
    print(f"end_disp_pullout     = {end_disp_pullout * 1e3:.3f} mm     (Phase E)")
    print(f"t_pullout            = {t_pullout} s  "
          f"-> v_pullout = {end_disp_pullout / t_pullout * 1e3:.3f} mm/s, "
          f"per-step disp = {end_disp_pullout / t_pullout * dt * 1e6:.1f} um")
    print(f"dt                   = {dt} s")
    if resume_pullout_path:
        print(f"resume_from          = {resume_pullout_path}")
    print(f"Phase end-times [s]  : A={T_A:.2f}, B={T_B:.2f}, C={T_C:.2f}, "
          f"D={T_D:.2f}, E={T_E:.2f}")
    print()

    # ---- Geometry -----------------------------------------------------------
    combined_geo, info = build_net_geometry(n_nodes_per_beam)
    n_total_nodes = combined_geo.nodes.shape[0]
    n_x_nodes_total = info['n_x_nodes_total']
    intersection_nodes = info['intersection_nodes']

    # Sanity check on initial gaps
    gaps0 = np.zeros((n_beams_per_dir, n_beams_per_dir))
    for ii in range(n_beams_per_dir):
        for jj in range(n_beams_per_dir):
            zx = combined_geo.nodes[intersection_nodes[ii, jj, 0], 2]
            zy = combined_geo.nodes[intersection_nodes[ii, jj, 1], 2]
            gaps0[ii, jj] = abs(zx - zy) - 2 * r_beam
    print(f"Initial intersection surface gap [mm]: "
          f"min={gaps0.min() * 1e3:.3f}, max={gaps0.max() * 1e3:.3f} "
          f"(positive = no contact)")
    if gaps0.min() <= 0:
        print("WARNING: some intersections start in contact -- bump up "
              "weave_amplitude or check r_beam.")

    # ---- Contact pair list and monkey-patch ---------------------------------
    # Only Eberly mode uses the per-pair beam-to-beam wrapper; IMC mode reuses
    # the SoftRobot-level cross-beam pair construction (`n_edges_beam_A`).
    beam_pairs = []
    for i in range(n_beams_per_dir):
        sA = int(info['x_beam_start'][i])
        nA = info['x_beam_n']
        for j in range(n_beams_per_dir):
            sB = int(info['y_beam_start'][j])
            nB = info['y_beam_n']
            beam_pairs.append((sA, nA, sB, nB))
    if contact_model == 'eberly':
        _ts_mod.compute_eberly_3d_beam_to_beam_contact_forces = (
            make_multi_pair_eberly_contact(beam_pairs))
        print(f"Beam-pair list       : {len(beam_pairs)} (x,y) pairs "
              f"(Eberly multi-pair wrapper installed)")
    else:
        print(f"Beam-pair list       : {len(beam_pairs)} (x,y) pairs "
              f"(IMC uses cross-beam pairs via n_edges_beam_A)")

    # ---- Material -----------------------------------------------------------
    geom = dismech.GeomParams(
        rod_r0=r_beam, shell_h=0,
        axs=k_inext * A_beam, ixs1=I_beam, ixs2=I_beam, jxs=J_beam)
    material = dismech.Material(
        density=(density / k_inext) * m_quasi,
        youngs_rod=E_young, youngs_shell=0,
        poisson_rod=nu, poisson_shell=0)

    # ---- Sim params ---------------------------------------------------------
    log_step = max(1, int(round(save_interval / dt)))
    n_edges_beam_A = n_x_nodes_total - n_beams_per_dir
    sim_params = dismech.SimParams(
        static_sim=False, two_d_sim=False, use_mid_edge=False,
        use_line_search=use_line_search, show_floor=False,
        log_data=True, log_step=log_step,
        dt=dt, max_iter=solver_max_iter, total_time=total_time,
        plot_step=0,
        tol=solver_tol, ftol=solver_tol, dtol=solver_tol,
        n_edges_beam_A=n_edges_beam_A,
        cross_beam_only_contact=True,   # x-x and y-y separations >> 2r
        line_search_iters=line_search_iters,
        solver=solver_name)

    # ---- Environment --------------------------------------------------------
    env = dismech.Environment()
    if contact_model == 'eberly':
        env.add_force(
            'beamToBeamContactEberly3D',
            beam_radius=r_beam,
            n_nodes_beam_A=n_x_nodes_total,
            kc=1.0,
            d_imc=d_imc_contact,
        )
        if mu_friction > 0.0:
            env.add_force(
                'beamToBeamFrictionEberly3D',
                mu=mu_friction,
                vel_tol=vel_tol,
            )
            env.friction_law = 'smooth_coulomb'
            env.eberly_friction_velocity_mode = 'in_iter'
            env.eberly_friction_jacobian = False
            print(f"Eberly friction enabled: mu = {mu_friction}, "
                  f"vel_tol = {vel_tol} m/s (smooth Coulomb gate)")
        else:
            print("Friction disabled (mu = 0).")
    else:   # contact_model == 'imc'
        # selfContact uses the IMC penalty: F(gap) ramps up over a width
        # `delta` once the surface gap drops below `2*h`.  h = r_beam, so
        # contact engages when centerline-to-centerline distance < 2*r.
        #
        # KNOWN ISSUE: this script's geometry (sub-mm r = 1.5 mm, 3.5 mm
        # initial centerline distance at every intersection) puts the IMC
        # pair-pruning threshold `2*h + 10*delta` close to the initial
        # state, which combined with the very stiff IMC Hessian
        # (k1 = 15/delta in normalized units) destabilizes the dense
        # Newton solver — convergence may fail in Phase A or right after
        # the free_nodes() call at Phase C->D.  The Eberly path is the
        # recommended model for this script; the IMC path is provided
        # for parity but may need d_imc / kc tuning per run.
        env.add_force(
            'selfContact',
            delta=d_imc_contact,
            h=r_beam,
            kc=1.0,
        )
        if mu_friction > 0.0:
            env.add_force(
                'selfFriction',
                mu=mu_friction,
                vel_tol=vel_tol,
            )
            env.imc_friction_jacobian = False
            print(f"IMC friction enabled: mu = {mu_friction}, "
                  f"vel_tol = {vel_tol} m/s")
        else:
            print("Friction disabled (mu = 0).")
    env.add_force('damping', eta=damping_eta)

    robot = dismech.SoftRobot(geom, material, combined_geo, sim_params, env)

    # ---- Boundary conditions -----------------------------------------------
    # Clamp the first 2 and last 2 nodes of every beam (full clamp at both
    # ends).  We keep handles to the two ends of the central x-beam (i=2)
    # because we will move / free them during Phases C-E.
    i_central = 2
    sA_central = int(info['x_beam_start'][i_central])
    left_pair_central = np.array([sA_central, sA_central + 1], dtype=int)
    right_pair_central = np.array([sA_central + n_nodes_per_beam - 2,
                                   sA_central + n_nodes_per_beam - 1], dtype=int)

    # Central x-beam (i=2): keep the CLAMP on both ends (2 nodes + edge)
    # because we drive both clamps in Phases A, C and E.
    # Non-central x-beams (i != 2, the "blue" strands): PIN the single
    # end node -- position fixed, rotation free.  This makes them act
    # like articulated links rather than cantilevered clamps, which
    # reduces the stored elastic energy at engagement and softens the
    # Phase D dynamics after the free-left event.
    # Y-beams: keep the clamp on both ends (fixed lattice).
    clamp_groups_x = []
    for i in range(n_beams_per_dir):
        sA = int(info['x_beam_start'][i])
        if i == i_central:
            left_pair = np.array([sA, sA + 1], dtype=int)
            right_pair = np.array([sA + n_nodes_per_beam - 2,
                                   sA + n_nodes_per_beam - 1], dtype=int)
            robot = robot.fix_nodes(left_pair, fix_edges=True)
            robot = robot.fix_nodes(right_pair, fix_edges=True)
        else:
            # "Articulated" end: fix the outer 2 nodes' positions but do
            # NOT fix the intermediate edge twist DOF.  Position + tangent
            # are still constrained (2 fixed nodes define a direction),
            # but the beam's TWIST at the endpoint is free.  This is the
            # numerically stable interpretation of a pin joint -- a true
            # single-node pin leaves a near-singular Jacobian in the
            # end-rotational direction and Newton does not converge.
            left_pair = np.array([sA, sA + 1], dtype=int)
            right_pair = np.array([sA + n_nodes_per_beam - 2,
                                   sA + n_nodes_per_beam - 1], dtype=int)
            robot = robot.fix_nodes(left_pair, fix_edges=False)
            robot = robot.fix_nodes(right_pair, fix_edges=False)
        clamp_groups_x.append((i, left_pair, right_pair))

    for j in range(n_beams_per_dir):
        sB = int(info['y_beam_start'][j])
        left_pair = np.array([sB, sB + 1], dtype=int)
        right_pair = np.array([sB + n_nodes_per_beam - 2,
                               sB + n_nodes_per_beam - 1], dtype=int)
        robot = robot.fix_nodes(left_pair, fix_edges=True)
        robot = robot.fix_nodes(right_pair, fix_edges=True)

    # --free-y-from-start: unblock y-DOF on the central x-beam's RIGHT clamp
    # from t=0 (same BC change as --free-y-pullout, but active throughout
    # Phases A-B-C-D instead of only at Phase E).  x and z remain fixed;
    # the twist edge remains fixed (fix_edges=False leaves the intermediate
    # edge DOF in fixed_dof).  Motion drivers in before_step_fn still shift
    # the pair in x via `+= dd * axial_x`; the += 0 on the y-component is a
    # no-op on the now-free DOF.
    if free_y_from_start:
        robot = robot.free_nodes(right_pair_central, axis=1, fix_edges=False)
        print(f"BC: --free-y-from-start ON -- freed y-DOF of central x-beam "
              f"right pair (nodes {right_pair_central.tolist()}) at t=0")
    print(f"BCs: central x-beam i={i_central} CLAMPED (2 nodes + edge twist), "
          f"non-central x-beams ARTICULATED (2 nodes fixed, twist FREE), "
          f"y-beams CLAMPED both ends.")
    print(f"  central left_pair={left_pair_central.tolist()}, "
          f"right_pair={right_pair_central.tolist()}")

    stepper = dismech.ImplicitEulerTimeStepper(robot)

    # ---- Contact-force snapshot helper (model-specific) --------------------
    # Returns (F_contact, gap_min) where F_contact is the (ndof,) vector of
    # contact-only nodal forces in PHYSICAL convention (positive = pushing
    # nodes apart along the contact normal).  Used by snapshot writing and
    # the pull-trace.

    class _FakeState:
        __slots__ = ('q',)
        def __init__(self, q):
            self.q = q

    def compute_contact_force(rb, q):
        if contact_model == 'eberly':
            kc_save = getattr(rb.env, '_eberly_bb_kc',
                              float(fixed_kc) if fixed_kc is not None
                              else rb.env.imc_stiffness)
            F_reg, _, gap_min, _ = _ts_mod.compute_eberly_3d_beam_to_beam_contact_forces(
                rb, q, r_beam,
                k_c=kc_save,
                n_nodes_beam_A=n_x_nodes_total,
                d_imc=d_imc_contact,
            )
            # Eberly: forces += F_reg in residual; F_reg already has the
            # physical sign convention (pushes nodes apart on engagement).
            return F_reg, gap_min
        # IMC path: re-evaluate the IMCEnergy gradient at the snapshot q.
        # `forces -= Fcon` in the stepper, so PHYSICAL force = -Fcon.
        Fcon, _ = stepper._contact_energy.grad_hess_energy(
            _FakeState(q), rb, np.zeros_like(q), first_iter=True)
        # gap_min is not directly exposed by IMCEnergy here; return inf.
        return -Fcon, float('inf')

    # ---- Phase driver -------------------------------------------------------
    axial_x = np.array([1.0, 0.0, 0.0])

    # Velocities for each ramp
    v_engage = end_disp_engage / t_engage              # m/s outward (both ends)
    v_release = end_disp_engage / t_release            # m/s back inward (left end of i=2)
    v_pullout = end_disp_pullout / t_pullout           # m/s outward (right end of i=2)

    # Cumulative displacement counters (m).  Sign is in the SAME direction as
    # the axial vector; left clamp moves NEGATIVE, right clamp moves POSITIVE.
    state = {
        'engage_each_end': 0.0,     # how much each x-beam clamp has been moved (per side, positive)
        'left_central_back': 0.0,   # how much the LEFT clamp of i=2 has been moved back (toward +x)
        'right_central_back': 0.0,  # how much the RIGHT clamp of i=2 has been moved back (toward -x)
        'pullout_right': 0.0,       # how much the RIGHT clamp of i=2 has been pulled further (+x)
        'left_freed': False,
        'right_y_freed': False,     # set once when --free-y-pullout is enabled
    }

    # ---- Reconstruct DER frames from A->D coords trajectory --------------
    # For a legacy checkpoint that only stored (q, u, kc), we can recover the
    # missing Bishop-frame history (a1, ref_twist) by REPLAYING parallel
    # transport step-by-step through the coords file, which recorded all
    # intermediate node positions.  compute_time_parallel and
    # compute_reference_twist both depend only on positions (not theta), so
    # the reconstructed a1 and ref_twist at end of Phase D are identical to
    # what a re-run would produce, up to the discretisation of the coords
    # sampling interval.
    if reconstruct_from_coords:
        if not resume_pullout_path:
            raise SystemExit(
                "--reconstruct-checkpoint requires --resume-pullout "
                "(the legacy checkpoint to source q, u, kc, counters from).")
        from pathlib import Path as _Path
        # --reconstruct-checkpoint may be a single path or a comma-
        # separated list -- e.g. "A_to_D.txt,phaseE.txt" to stitch the
        # main-run trajectory with a Phase-E resume file so we can
        # synthesise a checkpoint at ANY time in either.  The replay
        # walks the concatenated trajectory in file order.
        coords_paths = [_Path(p.strip())
                        for p in reconstruct_from_coords.split(",")
                        if p.strip()]
        for _p in coords_paths:
            if not _p.exists():
                raise SystemExit(f"Coords file not found: {_p}")
        print(f"\n*** Reconstructing DER frames from coords trajectory")
        for _i, _p in enumerate(coords_paths):
            print(f"    coords[{_i}] = {_p}")
        print(f"    legacy ck   = {resume_pullout_path}")

        from io import StringIO as _StringIO
        pieces = []
        for coords_path in coords_paths:
            raw = coords_path.read_bytes()
            _end = raw.find(b"\x00")
            if _end != -1:
                raw = raw[:_end]
            _text = raw.decode("utf-8", errors="ignore")
            _lines = [ln for ln in _text.split("\n")
                      if ln.strip() and not ln.startswith("#")]
            arr = np.loadtxt(_StringIO("\n".join(_lines)))
            # Drop rows whose time is already covered by earlier pieces
            # (avoid replaying the boundary snapshot twice).
            if pieces:
                prev_max_t = float(np.unique(pieces[-1][:, 0])[-1])
                arr = arr[arr[:, 0] > prev_max_t + 1e-9]
            if arr.size > 0:
                pieces.append(arr)
        traj = np.concatenate(pieces, axis=0) if len(pieces) > 1 else pieces[0]
        traj_times = np.unique(traj[:, 0])
        n_nodes_total = int(np.max(traj[:, 1])) + 1
        print(f"    stitched: {len(traj_times)} snapshots, "
              f"t = {traj_times[0]:.3f} -> {traj_times[-1]:.3f} s, "
              f"n_nodes = {n_nodes_total}")

        # If --synthesize-at-time was given, cut the trajectory at that
        # time -- we replay parallel-transport only up to the last
        # verifiably-good snapshot, then use the coords file at
        # synthesize_at_time as the checkpoint q (skipping the legacy
        # checkpoint's potentially corrupted q entirely).
        if synthesize_at_time is not None:
            # Snap requested time to nearest snapshot present in the file
            t_syn = float(traj_times[
                np.argmin(np.abs(traj_times - float(synthesize_at_time)))])
            # Keep only snapshots up to and including t_syn
            traj_times = traj_times[traj_times <= t_syn + 1e-9]
            print(f"    --synthesize-at-time = {synthesize_at_time} s "
                  f"(snapped to {t_syn:.4f} s); replay ends there")
            print(f"    trimmed trajectory to {len(traj_times)} snapshots")

        # Sanity: robot's initial state has a1 from compute_space_parallel
        # on the initial mesh geometry.  Replay time-parallel through the
        # trajectory.
        a1 = robot.state.a1.copy()
        ref_twist = robot.state.ref_twist.copy()
        q_prev = robot.state.q.copy()

        # First snapshot in the coords file is the initial state (t = 0);
        # subsequent snapshots hold the trajectory.
        for k, t in enumerate(traj_times):
            _cm = np.isclose(traj[:, 0], t)
            _rows = traj[_cm]
            _order = np.argsort(_rows[:, 1].astype(int))
            xyz_t = _rows[_order, 2:5]
            q_t = q_prev.copy()
            q_t[:3 * n_nodes_total] = xyz_t.reshape(-1)
            if k == 0:
                # No transport needed; the initial q matches robot state
                q_prev = q_t
                continue
            a1_new, _ = robot.compute_time_parallel(a1, q_prev, q_t)
            ref_twist_new = robot.compute_reference_twist(
                robot.twist_springs, q_t, a1_new, ref_twist)
            a1 = a1_new
            ref_twist = ref_twist_new
            q_prev = q_t

        # Load the legacy checkpoint for counters and (optionally) q/u/kc
        ck = np.load(resume_pullout_path, allow_pickle=False)

        # Decide q_final, u_final, kc_final, t_final:
        if synthesize_at_time is None:
            # Standard reconstruction: use legacy checkpoint's q/u/kc as-is
            q_final = ck['q'].copy()
            u_final = ck['u'] if 'u' in ck.files else robot.state.u.copy()
            kc_final = float(ck['kc'])
            t_final = float(ck['t'])
        else:
            # Synthesis: build q, u, kc from the coords trajectory at t_syn
            #   - q positions: from coords at t_syn
            #   - q theta   : ZERO (isotropic circular cross-section)
            #   - u         : finite difference from t_syn-Δ to t_syn
            #                 (backward diff, or central if possible)
            #   - kc        : imc_stiffness baseline (will re-adapt)
            t_final = t_syn
            q_final = robot.state.q.copy()   # gets initial theta=0
            q_final[:3 * n_nodes_total] = q_prev[:3 * n_nodes_total]

            # Velocity via backward finite difference on positions
            if len(traj_times) >= 2:
                t_prev_time = float(traj_times[-2])
                _cm2 = np.isclose(traj[:, 0], t_prev_time)
                _rp = traj[_cm2]
                xyz_p = _rp[np.argsort(_rp[:, 1].astype(int)), 2:5]
                dt_fd = t_syn - t_prev_time
                u_positions = (q_prev[:3 * n_nodes_total]
                              - xyz_p.reshape(-1)) / dt_fd
                u_final = robot.state.u.copy()
                u_final[:3 * n_nodes_total] = u_positions
                print(f"    synthesized u via backward diff over "
                      f"Δt = {dt_fd:.4f} s")
            else:
                u_final = robot.state.u.copy()
                print(f"    WARNING: only 1 snapshot in trajectory, "
                      f"setting u = 0")

            kc_final = float(getattr(robot.env, 'imc_stiffness', 1.0))
            print(f"    synthesized kc = {kc_final:.3e} "
                  f"(imc_stiffness baseline; will re-adapt in Newton)")

        # Final one-step transport onto q_final (if it differs from last
        # replay position; for synthesis they match exactly).
        a1_final, a2_final = robot.compute_time_parallel(a1, q_prev, q_final)
        m1_final, m2_final = robot.compute_material_directors(
            q_final, a1_final, a2_final)
        ref_twist_final = robot.compute_reference_twist(
            robot.twist_springs, q_final, a1_final, ref_twist)

        # Write extended checkpoint next to the legacy one.  For synthesis,
        # tag it "_synthetic_tXXXX" to distinguish from reconstruction.
        if synthesize_at_time is not None:
            suffix = f"_synthetic_t{int(round(t_syn*1000)):04d}"
        else:
            suffix = "_reconstructed"
        out_ck = _Path(resume_pullout_path).with_name(
            _Path(resume_pullout_path).stem + f"{suffix}.npz")
        s = robot.state    # use initial `a` and `tau` shapes / zeros

        # If we're synthesising a mid-Phase-E state (t_syn > T_D), the
        # pullout_right counter must reflect how far the right clamp
        # has ALREADY been pulled -- otherwise the sim would try to
        # pull the FULL end_disp_pullout starting from the current q,
        # double-counting the displacement.  Compute it from the
        # geometry: initial right-clamp x = L_pad + L_MAIN = 45 mm
        # (nominal for our setup), plus the Phase-A engagement of
        # +end_disp_engage that returned to nominal in Phase C, so at
        # end of Phase D the central right clamp sits at 45 mm.  Any
        # displacement past 45 mm in Phase E IS pullout_right.
        _synth_pullout_right = float(ck['pullout_right']) \
            if 'pullout_right' in ck.files else 0.0
        if synthesize_at_time is not None and t_syn > 3.5 + 1e-9:
            # Central x-beam right node index in the flat q array
            _right_idx_central = 2 * n_nodes_per_beam + n_nodes_per_beam - 1
            _x_right_now = float(q_final[3 * _right_idx_central])
            _x_right_initial = 45.0e-3     # 45 mm nominal end-of-D position
            _synth_pullout_right = max(0.0, _x_right_now - _x_right_initial)
            print(f"    synthesised pullout_right = "
                  f"{_synth_pullout_right*1e3:.4f} mm  "
                  f"(right clamp x now = {_x_right_now*1e3:.4f} mm, "
                  f"initial = {_x_right_initial*1e3:.4f} mm)")

        np.savez(
            out_ck,
            t=t_final,
            q=q_final,
            u=u_final,
            a=s.a,
            a1=a1_final, a2=a2_final,
            m1=m1_final, m2=m2_final,
            ref_twist=ref_twist_final,
            tau=s.tau,
            kc=kc_final,
            engage_each_end=float(ck['engage_each_end']),
            left_central_back=float(ck['left_central_back']),
            right_central_back=float(ck['right_central_back'])
                if 'right_central_back' in ck.files else 0.0,
            pullout_right=_synth_pullout_right,
            left_freed=bool(ck['left_freed']),
            n_nodes_per_beam=int(ck['n_nodes_per_beam']),
        )
        print(f"    Wrote extended checkpoint: {out_ck}")
        print(f"    a1 (edge 0) initial -> reconstructed: "
              f"{robot.state.a1[0]} -> {a1_final[0]}")
        print(f"    ref_twist max |value| = "
              f"{float(np.max(np.abs(ref_twist_final))):.3e}")

        # ---- Verification ----------------------------------------------
        # We compare (a) contact forces at the reconstructed q and kc,
        # against the last snapshot of the sibling forces file -- this
        # verifies that q and kc were loaded faithfully (contact force
        # depends on q and kc only, not on DER frames); and (b) the total
        # force residual assembled with the reconstructed DER state,
        # which should sit near Newton's tolerance if the DER frames are
        # consistent with q (the sim is at quasi-static equilibrium at
        # end of Phase D, so total force at free DOFs should be small).
        forces_path = coords_path.with_name(
            coords_path.name.replace("coords", "forces", 1))
        print(f"\n    Verification against {forces_path.name}:")
        if not forces_path.exists():
            print(f"      SKIPPED -- forces file not found at {forces_path}")
        else:
            raw_f = forces_path.read_bytes()
            _e = raw_f.find(b"\x00")
            if _e != -1:
                raw_f = raw_f[:_e]
            _text_f = raw_f.decode("utf-8", errors="ignore")
            _lines_f = [ln for ln in _text_f.split("\n")
                        if ln.strip() and not ln.startswith("#")]
            F_traj = np.loadtxt(_StringIO("\n".join(_lines_f)))
            # Pick the row set closest to the checkpoint time t
            t_ck = float(ck['t'])
            f_times = np.unique(F_traj[:, 0])
            t_pick = float(f_times[np.argmin(np.abs(f_times - t_ck))])
            _fm = np.isclose(F_traj[:, 0], t_pick)
            _rows = F_traj[_fm]
            _order = np.argsort(_rows[:, 1].astype(int))
            F_saved = _rows[_order, 2:5]                  # (n_nodes, 3)

            # (a) Recompute contact force at (q_final, kc_final)
            robot.env._eberly_bb_kc = float(ck['kc'])
            robot.env._eberly_bb_active = True
            F_recomp, _, _, _ = _ts_mod.compute_eberly_3d_beam_to_beam_contact_forces(
                robot, q_final, r_beam,
                k_c=float(ck['kc']),
                n_nodes_beam_A=n_x_nodes_total,
                d_imc=d_imc_contact,
            )
            F_recomp_nodes = F_recomp[:robot.end_node_dof_index].reshape(-1, 3)
            n_min = min(F_recomp_nodes.shape[0], F_saved.shape[0])
            F_recomp_nodes = F_recomp_nodes[:n_min]
            F_saved = F_saved[:n_min]
            diff = np.linalg.norm(F_recomp_nodes - F_saved, axis=1)
            mag_saved = np.linalg.norm(F_saved, axis=1)
            active = mag_saved > 1e-6
            rel_err = (diff[active] / mag_saved[active]) if active.any() \
                      else np.array([0.0])
            print(f"      Contact-force check (verifies q, kc load):")
            print(f"        forces_file t             = {t_pick:.3f} s "
                  f"(checkpoint t = {t_ck:.3f})")
            print(f"        |F|_saved  max            = "
                  f"{float(mag_saved.max()):.4e} N")
            print(f"        |F|_recomp max            = "
                  f"{float(np.linalg.norm(F_recomp_nodes, axis=1).max()):.4e} N")
            print(f"        max abs diff              = "
                  f"{float(diff.max()):.4e} N")
            print(f"        max rel err (active nodes)= "
                  f"{float(rel_err.max()):.4e}")

            # (b) Self-consistency check on the reconstructed frames.
            # Bishop frame a1 must be perpendicular to the edge tangent
            # at the same edge, to machine precision.  This is a direct
            # test that our per-step parallel-transport is preserving
            # the orthonormality invariant.
            tangents = robot._compute_tangent(q_final)
            dots = np.einsum('ij,ij->i', a1_final, tangents)
            a1_norms = np.linalg.norm(a1_final, axis=1)
            print(f"      Frame orthonormality (self-consistency):")
            print(f"        max |a1 . t|              = "
                  f"{float(np.max(np.abs(dots))):.4e}  "
                  f"(should be ~ 1e-15)")
            print(f"        max ||a1| - 1|            = "
                  f"{float(np.max(np.abs(a1_norms - 1.0))):.4e}  "
                  f"(should be ~ 1e-15)")

            # (c) OPTIONAL: if a "ground-truth" new-format checkpoint
            # (with the true DER fields saved) exists at the same stem,
            # compare our reconstructed a1 / ref_twist against it.  This
            # is the definitive check.  Look for
            # `<stem>_truthref.npz` next to the legacy checkpoint.
            truth_path = _Path(resume_pullout_path).with_name(
                _Path(resume_pullout_path).stem + "_truthref.npz")
            if truth_path.exists():
                truth = np.load(truth_path)
                if 'a1' in truth.files:
                    a1_err = np.max(np.linalg.norm(
                        a1_final - truth['a1'], axis=1))
                    rt_err = np.max(np.abs(
                        ref_twist_final - truth['ref_twist']))
                    print(f"      Ground-truth comparison "
                          f"({truth_path.name}):")
                    print(f"        max |a1_recon - a1_truth|  "
                          f"= {a1_err:.4e}")
                    print(f"        max |rt_recon - rt_truth|  "
                          f"= {rt_err:.4e}")
        raise SystemExit(0)

    # ---- Resume from end-of-Phase-D checkpoint (if requested) -------------
    resume_t0 = 0.0
    if resume_pullout_path:
        ck = np.load(resume_pullout_path, allow_pickle=False)
        if int(ck['n_nodes_per_beam']) != n_nodes_per_beam:
            raise SystemExit(
                f"Checkpoint mesh N={int(ck['n_nodes_per_beam'])} does not "
                f"match argument N={n_nodes_per_beam}")
        if bool(ck['left_freed']):
            final_robot_pre = robot.free_nodes(left_pair_central, fix_edges=True)
        else:
            final_robot_pre = robot
        # In-place state restore.  q and u are always present; the
        # DER material frames (a1, a2, m1, m2, ref_twist) are only in
        # checkpoints written by the extended save path -- if missing
        # (legacy checkpoint), the resumed run will start with
        # initial-mesh frames and see a boundary discontinuity in the
        # elastic energy.  Warn in that case rather than silently
        # producing wrong forces.
        s = final_robot_pre.state
        s.q[:] = ck['q']
        if 'u' in ck.files:
            s.u[:] = ck['u']
        if 'a' in ck.files:
            s.a[:] = ck['a']
        der_keys = ('a1', 'a2', 'm1', 'm2', 'ref_twist')
        if all(k in ck.files for k in der_keys):
            s.a1[:] = ck['a1']
            s.a2[:] = ck['a2']
            s.m1[:] = ck['m1']
            s.m2[:] = ck['m2']
            s.ref_twist[:] = ck['ref_twist']
            if 'tau' in ck.files and s.tau.size == ck['tau'].size:
                s.tau[:] = ck['tau']
            der_status = "full DER state restored"
        else:
            der_status = ("WARNING: DER material frames missing from "
                          "checkpoint -- expect a discontinuity in the "
                          "elastic forces at the Phase-D->E boundary. "
                          "Re-run the A->D main campaign to regenerate "
                          "checkpoints with the full state.")
        # Preserve the adaptive k_c that was active at end of Phase D
        env._eberly_bb_kc = float(ck['kc'])
        env._eberly_bb_active = True
        state['engage_each_end'] = float(ck['engage_each_end'])
        state['left_central_back'] = float(ck['left_central_back'])
        # right_central_back added later; default to 0 if resuming an old
        # checkpoint that didn't record it.
        state['right_central_back'] = float(ck['right_central_back']) \
            if 'right_central_back' in ck.files else 0.0
        state['left_freed'] = True
        # For a mid-Phase-E resume (synthetic or normal), the checkpoint
        # carries the pullout accumulated so far.  Old checkpoints written
        # at end of Phase D have pullout_right = 0 which is correct there.
        state['pullout_right'] = float(ck['pullout_right']) \
            if 'pullout_right' in ck.files else 0.0
        resume_t0 = float(ck['t'])
        # Replace `robot` so the stepper sees the freed DOF mask + loaded state
        robot = final_robot_pre
        stepper = dismech.ImplicitEulerTimeStepper(robot)
        print(f"\n*** Resumed from '{resume_pullout_path}'")
        print(f"    Phase D end time = {resume_t0:.3f} s")
        print(f"    loaded k_c = {float(ck['kc']):.3e}")
        print(f"    left pair freed = {bool(ck['left_freed'])}")
        print(f"    {der_status}\n")

    kc_ramp_time = 0.2  # s, brief ramp of k_c at the start

    def before_step_fn(rb, t):
        # Adaptive k_c override (constant after a brief ramp).  Only the
        # Eberly path exposes a per-step k_c hook; IMC uses an adaptive
        # k_c internally and ignores this attribute.
        if fixed_kc is not None and contact_model == 'eberly':
            if t < kc_ramp_time:
                frac = max(0.0, t) / kc_ramp_time
                kc_now = 1.0 + (float(fixed_kc) - 1.0) * frac
            else:
                kc_now = float(fixed_kc)
            rb.env._eberly_bb_kc = kc_now
            rb.env._eberly_bb_active = True

        dt_local = rb.sim_params.dt
        t_end = t + dt_local

        # ---- Phase A: 0 <= t < T_A : engage ----
        if t_end <= T_A + 1e-12:
            remaining = end_disp_engage - state['engage_each_end']
            if remaining > 1e-15:
                dd = min(v_engage * dt_local, remaining)
                state['engage_each_end'] += dd
                for (i, left_pair, right_pair) in clamp_groups_x:
                    rb = rb.move_nodes(left_pair, -dd * axial_x)
                    rb = rb.move_nodes(right_pair, +dd * axial_x)
            return rb

        # ---- Phase B: T_A <= t < T_B : hold ----
        if t_end <= T_B + 1e-12:
            return rb

        # ---- Phase C: T_B <= t < T_C : SYMMETRIC release of central x-beam --
        # Move BOTH left and right clamps of i=2 back toward their initial
        # positions, so all axial tension in that strand is unloaded before
        # the free-left event.  This gives a much softer Phase D transition
        # than releasing only the left clamp (which left the right half
        # under tension and drove strong lateral springback).
        if t_end <= T_C + 1e-12:
            remaining_L = end_disp_engage - state['left_central_back']
            remaining_R = end_disp_engage - state['right_central_back']
            if remaining_L > 1e-15:
                dd_L = min(v_release * dt_local, remaining_L)
                state['left_central_back'] += dd_L
                rb = rb.move_nodes(left_pair_central, +dd_L * axial_x)
            if remaining_R > 1e-15:
                dd_R = min(v_release * dt_local, remaining_R)
                state['right_central_back'] += dd_R
                rb = rb.move_nodes(right_pair_central, -dd_R * axial_x)
            return rb

        # ---- Phase D: T_C <= t < T_D : free left clamp + relax ----
        if t_end <= T_D + 1e-12:
            if not state['left_freed']:
                rb = rb.free_nodes(left_pair_central, fix_edges=True)
                state['left_freed'] = True
                print(f"  t = {t_end:.4f} s : freed left pair of central "
                      f"x-beam (nodes {left_pair_central.tolist()})")
            return rb

        # ---- Phase E: T_D <= t < T_E : pull RIGHT clamp of central x-beam ----
        # If --free-y-pullout: at the very first Phase-E step, free the y-DOF
        # of the two right-clamp nodes so the strand can self-align laterally
        # under the pull. x and z stay fixed (x is prescribed by move_nodes,
        # z stays clamped); the twist edge stays fixed (fix_edges=False on
        # free_nodes leaves the intermediate edge DOF in fixed_dof).
        if free_y_pullout and not state['right_y_freed']:
            rb = rb.free_nodes(right_pair_central, axis=1, fix_edges=False)
            state['right_y_freed'] = True
            print(f"  t = {t_end:.4f} s : freed y-DOF of central x-beam "
                  f"right pair (nodes {right_pair_central.tolist()}) -- "
                  f"pullout will not impose y")
        remaining = end_disp_pullout - state['pullout_right']
        if remaining > 1e-15:
            dd = min(v_pullout * dt_local, remaining)
            state['pullout_right'] += dd
            rb = rb.move_nodes(right_pair_central, +dd * axial_x)
        return rb

    stepper.before_step = before_step_fn

    # ---- Output paths -------------------------------------------------------
    out_dir = Path(__file__).resolve().parent / (
        "contact_output_files_net_pullout_alternated" if alternating_weave
        else "contact_output_files_net_pullout")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag_str = f"_{contact_model}_N{n_nodes_per_beam}{output_tag}"
    out_coords_txt = out_dir / f"net_pullout_coords{tag_str}.txt"
    out_forces_txt = out_dir / f"net_pullout_forces{tag_str}.txt"
    out_phase_txt = out_dir / f"net_pullout_phase_forces{tag_str}.txt"
    out_pull_txt = out_dir / f"net_pullout_pull_traction{tag_str}.txt"
    out_summary_txt = out_dir / f"net_pullout_summary{tag_str}.txt"

    def write_snapshot(rb, t_now, fh_coords, fh_forces):
        q = rb.state.q
        node_coords = q[:rb.end_node_dof_index].reshape(-1, 3)
        for ni, xyz in enumerate(node_coords):
            fh_coords.write(
                f"{t_now:.6f}\t{ni}\t{xyz[0]:.12e}\t{xyz[1]:.12e}\t{xyz[2]:.12e}\n")
        try:
            F_con, _ = compute_contact_force(rb, q)
            f_nodes = F_con[:rb.end_node_dof_index].reshape(-1, 3)
        except Exception as e:
            print(f"  Warning: contact force extraction failed at t={t_now}: {e}")
            f_nodes = np.zeros_like(node_coords)
        for ni, fxyz in enumerate(f_nodes):
            fh_forces.write(
                f"{t_now:.6f}\t{ni}\t{fxyz[0]:.12e}\t{fxyz[1]:.12e}\t{fxyz[2]:.12e}\n")
        return node_coords, f_nodes

    # ---- Run ----------------------------------------------------------------
    # The loop uses an EXPLICIT TIME counter rather than a step index so the
    # per-step dt can switch at phase boundaries:
    #   Phases A, B:  dt
    #   Phases C, D:  dt_release  (if provided, else dt)
    #   Phase  E:     dt_pullout  (if provided, else dt)
    # We snap dt at each step so we always land exactly on a phase boundary,
    # which guarantees end-of-phase snapshots / checkpoints fire on time.
    phase_times = {'A': T_A, 'B': T_B, 'C': T_C, 'D': T_D, 'E': T_E}
    phase_boundaries = (T_A, T_B, T_C, T_D, T_E)
    checkpoint_path = out_dir / f"pullout_checkpoint{tag_str}.npz"

    def dt_for_time(t):
        if t >= T_D - 1e-12 and dt_pullout is not None:
            return dt_pullout
        if T_B - 1e-12 <= t < T_D - 1e-12 and dt_release is not None:
            return dt_release
        return dt

    if resume_pullout_path:
        print(f"\nRunning only Phase E: t = {resume_t0:.3f} -> {T_E:.3f} s, "
              f"dt = {dt} s")
    else:
        dt_summary = f"dt = {dt}"
        if dt_release is not None:
            dt_summary += f", dt_release (C,D) = {dt_release}"
        if dt_pullout is not None:
            dt_summary += f", dt_pullout (E) = {dt_pullout}"
        print(f"\nRunning t = 0 -> {total_time} s ({dt_summary})")

    saved_count = 0
    snapshots = {}    # 'A'..'E' -> (coords, forces) at end of each phase
    pull_trace = []   # (t, pull_x_disp, right_clamp_x_force, gap_min, ...)
    # Per-intersection contact-force time series, recorded at every dt-step
    # (not just at snapshot times) so the time signal is dense.
    per_intersection_t = []
    per_intersection_FN = []   # list of (nI, nJ) arrays = |F_N| per intersection
    per_intersection_FT = []   # list of (nI, nJ) arrays = |F_T| per intersection

    # Incremental per-step force-history file.  Historically this was
    # written once in the post-run block, which means a run killed by the
    # wall-time limit (or a Newton-divergence crash) lost its ENTIRE
    # per-step force signal -- exactly the runs whose jitter we most want
    # to inspect.  Stream it instead: one line per (step, ii, jj),
    # flushed every step, same format as before.
    out_fhist_txt = out_dir / f"net_pullout_force_history{tag_str}.txt"
    fh_fhist = out_fhist_txt.open("w", encoding="utf-8")
    fh_fhist.write("# Per-intersection contact-force time history.\n")
    fh_fhist.write("# Columns: t_s, ii, jj, |FN|_N, |FT|_N\n")

    final_robot = robot
    with out_coords_txt.open("w", encoding="utf-8") as fh_coords, \
         out_forces_txt.open("w", encoding="utf-8") as fh_forces:

        fh_coords.write("# t_s\tnode_id\tx_m\ty_m\tz_m\n")
        fh_forces.write("# t_s\tnode_id\tFx_N\tFy_N\tFz_N\n")

        coords0, forces0 = write_snapshot(
            final_robot, resume_t0, fh_coords, fh_forces)
        saved_count += 1
        fh_coords.flush(); fh_forces.flush()

        t_now = resume_t0
        last_save_t = resume_t0   # initial snapshot already written above
        last_periodic_checkpoint_t = resume_t0
        step_idx = 0
        while t_now < total_time - 1e-12:
            step_idx += 1
            if step_idx % 50 == 0:
                rss_mb, vsz_mb = _proc_mem_mb()
                if rss_mb is not None:
                    print(f"  [mem] step {step_idx}  t={t_now:.3f}s  "
                          f"RSS={rss_mb:.0f} MiB  VmSize={vsz_mb:.0f} MiB",
                          flush=True)
            # Choose the step size for this step
            dt_step = dt_for_time(t_now)
            # If the next boundary falls within this step, snap dt to land
            # exactly on it -- guarantees end-of-phase logic fires on time.
            for boundary in phase_boundaries:
                if t_now + 1e-12 < boundary <= t_now + dt_step + 1e-12:
                    dt_step = boundary - t_now
                    break
            # Mutate sim_params so the stepper picks up the new dt.  This
            # works for the Eberly friction path (dt is passed per-call).
            # IMC friction would NOT see the change -- it caches dt at
            # stepper construction -- but we're not running IMC here.
            final_robot.sim_params.dt = dt_step

            t_before = t_now
            _debug_current_t[0] = t_before
            final_robot = stepper.before_step(final_robot, t_before)
            # Monotone-k_c patch (Eberly only, opt-in via --monotone-kc):
            # before the step, restore k_c to the max we've seen so
            # time_stepper.py's decay branch has a big value to start
            # from rather than the previously-decayed one.
            if monotone_kc and contact_model == 'eberly':
                kc_max = state.get('_max_kc_ever', 1.0)
                cur_kc = getattr(final_robot.env, '_eberly_bb_kc', 1.0)
                if cur_kc < kc_max:
                    final_robot.env._eberly_bb_kc = kc_max
                    final_robot.env._eberly_bb_active = True
            final_robot, _ = stepper.step(final_robot)
            # After the step, k_c may have grown OR been decayed by the
            # dismech's `elif env._eberly_bb_active` branch.  Record the
            # max so the next iteration restores it before the step.
            if monotone_kc and contact_model == 'eberly':
                kc_after = getattr(final_robot.env, '_eberly_bb_kc', 1.0)
                state['_max_kc_ever'] = max(state.get('_max_kc_ever', 1.0),
                                             kc_after)
            # Extra script-side under-relaxation on top of dismech's 0.9/0.1
            # update, to kill the period-2 kc<->gap limit cycle in static
            # equilibrium.  With ALPHA=0.99, the effective per-step change
            # in kc is ~100x smaller than dismech alone.
            if kc_under_relax > 0.0 and contact_model == 'eberly':
                kc_new = getattr(final_robot.env, '_eberly_bb_kc', 1.0)
                kc_prev = state.get('_kc_prev_under_relax', kc_new)
                kc_smoothed = (kc_under_relax * kc_prev
                               + (1.0 - kc_under_relax) * kc_new)
                final_robot.env._eberly_bb_kc = kc_smoothed
                state['_kc_prev_under_relax'] = kc_smoothed
            t_now += dt_step
            t_after = t_now

            # Phase boundary detection
            boundary_label = None
            for k, T in phase_times.items():
                if abs(t_after - T) < 1e-9:
                    boundary_label = k
                    break

            save_now = (boundary_label is not None
                        or (t_after - last_save_t >= save_interval - 1e-12))
            if save_now:
                coords_i, forces_i = write_snapshot(
                    final_robot, t_after, fh_coords, fh_forces)
                saved_count += 1
                fh_coords.flush(); fh_forces.flush()
                last_save_t = t_after
                if boundary_label is not None:
                    snapshots[boundary_label] = (coords_i.copy(),
                                                  forces_i.copy())

            # Auto-save a checkpoint at end of Phase D so the user can
            # iterate on Phase E parameters without re-running A through D.
            if boundary_label == 'D' and not resume_pullout_path:
                kc_now = getattr(final_robot.env, '_eberly_bb_kc', 1.0)
                # Save the DER state IN FULL, not just q + u.  Bishop
                # frames (a1, a2), material directors (m1, m2) and the
                # accumulated reference twist are HISTORY-DEPENDENT via
                # parallel transport.  Omitting them makes the resume
                # start with initial-mesh frames but end-of-D positions:
                # the bending / twisting elastic energy is computed
                # against a mismatched reference, producing a spurious
                # force spike at t = T_D and driving Newton to a
                # different equilibrium than the main run had.
                s = final_robot.state
                np.savez(
                    checkpoint_path,
                    t=t_after,
                    q=s.q, u=s.u, a=s.a,
                    a1=s.a1, a2=s.a2, m1=s.m1, m2=s.m2,
                    ref_twist=s.ref_twist, tau=s.tau,
                    kc=kc_now,
                    engage_each_end=state['engage_each_end'],
                    left_central_back=state['left_central_back'],
                    right_central_back=state['right_central_back'],
                    pullout_right=state['pullout_right'],
                    left_freed=state['left_freed'],
                    n_nodes_per_beam=n_nodes_per_beam,
                )
                print(f"  *** Saved Phase-D checkpoint to "
                      f"'{checkpoint_path}' (kc = {kc_now:.3e}) "
                      f"[full DER state]")
                if stop_after_phase_d:
                    print(f"  *** Stop-after-Phase-D requested; exiting loop.")
                    break

            # Periodic FULL-STATE checkpoint during Phase E (--checkpoint-
            # interval), independent of the one-shot Phase-D save above --
            # fires regardless of resume_pullout_path, so a chain of
            # --resume-pullout runs each leave their own trail. Same
            # contents/format as the Phase-D checkpoint, just timestamped
            # in the filename instead of a single fixed name.
            if (checkpoint_interval > 0.0 and t_after >= T_D - 1e-12
                    and (t_after - last_periodic_checkpoint_t)
                        >= checkpoint_interval - 1e-12):
                kc_now_p = getattr(final_robot.env, '_eberly_bb_kc', 1.0)
                s = final_robot.state
                periodic_ckpt_path = checkpoint_path.with_name(
                    checkpoint_path.stem + f"_t{t_after:.3f}.npz")
                np.savez(
                    periodic_ckpt_path,
                    t=t_after,
                    q=s.q, u=s.u, a=s.a,
                    a1=s.a1, a2=s.a2, m1=s.m1, m2=s.m2,
                    ref_twist=s.ref_twist, tau=s.tau,
                    kc=kc_now_p,
                    engage_each_end=state['engage_each_end'],
                    left_central_back=state['left_central_back'],
                    right_central_back=state['right_central_back'],
                    pullout_right=state['pullout_right'],
                    left_freed=state['left_freed'],
                    n_nodes_per_beam=n_nodes_per_beam,
                )
                print(f"  *** Saved periodic checkpoint to "
                      f"'{periodic_ckpt_path}' (kc = {kc_now_p:.3e}) "
                      f"[full DER state]")
                last_periodic_checkpoint_t = t_after

            # Per-step contact-force diagnostic.  We record:
            #   - sum F_x on the right-pair of the central x-beam (pull load)
            #   - per-intersection |F_N| and |F_T| for the time-history plot
            #   - geometric min gap (surface gap at the closest intersection),
            #     independent of whether any pair is in the smooth zone --
            #     gives a meaningful "approaching contact" signal even when
            #     the contact-energy gap_min is +inf (no pair active yet).
            try:
                F_con, gap_min_now = compute_contact_force(
                    final_robot, final_robot.state.q)
                # geometric min surface gap, computed from intersection-node
                # z-coordinates (works regardless of contact model state)
                q_now = final_robot.state.q
                node_pos = q_now[final_robot.node_dof_indices]
                geom_gap_min = float('inf')
                for ii in range(n_beams_per_dir):
                    for jj in range(n_beams_per_dir):
                        nx = int(intersection_nodes[ii, jj, 0])
                        ny = int(intersection_nodes[ii, jj, 1])
                        d_centerline = float(np.linalg.norm(
                            node_pos[nx] - node_pos[ny]))
                        g = d_centerline - 2 * r_beam
                        if g < geom_gap_min:
                            geom_gap_min = g
                fnodes = F_con[:final_robot.end_node_dof_index].reshape(-1, 3)
                fpull = float(np.sum(fnodes[right_pair_central, 0]))
                FN_x_now, FN_y_now, FT_x_now, FT_y_now = extract_intersection_forces(
                    F_con[:final_robot.end_node_dof_index],
                    final_robot.node_dof_indices, intersection_nodes)
                FN_per_t = FN_x_now + FN_y_now      # (nI, nJ)
                FT_per_t = FT_x_now + FT_y_now
            except Exception:
                fpull = 0.0
                gap_min_now = float('inf')
                geom_gap_min = float('inf')
                FN_per_t = np.zeros((n_beams_per_dir, n_beams_per_dir))
                FT_per_t = np.zeros((n_beams_per_dir, n_beams_per_dir))

            pull_trace.append((t_after,
                               state['pullout_right'],
                               fpull,
                               gap_min_now,
                               state['engage_each_end'],
                               state['left_central_back'],
                               geom_gap_min))
            per_intersection_t.append(t_after)
            per_intersection_FN.append(FN_per_t.copy())
            per_intersection_FT.append(FT_per_t.copy())
            # Stream this step's per-intersection forces so a wall-time
            # kill cannot destroy the per-step signal.
            for ii in range(n_beams_per_dir):
                for jj in range(n_beams_per_dir):
                    fh_fhist.write(f"{t_after:.6f}\t{ii}\t{jj}\t"
                                   f"{FN_per_t[ii, jj]:.6e}\t"
                                   f"{FT_per_t[ii, jj]:.6e}\n")
            fh_fhist.flush()

            print(f"  t = {t_after:6.3f} s  | engage = {state['engage_each_end'] * 1e3:6.3f} mm "
                  f"| L-back = {state['left_central_back'] * 1e3:6.3f} mm "
                  f"| R-back = {state['right_central_back'] * 1e3:6.3f} mm "
                  f"| pullout = {state['pullout_right'] * 1e3:6.3f} mm "
                  f"| Fpull_x = {fpull:+.3e} N "
                  f"| geom_gap = {geom_gap_min * 1e3:+.3f} mm "
                  f"| smooth_gap_min = {gap_min_now:+.2e} m")

    print(f"\nDone. Saved {saved_count} snapshots in '{out_dir}'.")

    # ---- Phase-end summary file --------------------------------------------
    node_dof_indices = final_robot.node_dof_indices
    with out_phase_txt.open("w", encoding="utf-8") as fh:
        fh.write("# Per-intersection contact forces at the end of each phase.\n")
        fh.write("# Columns: phase, ii, jj, x_m, y_m, |FN_x|, |FN_y|, |FT_x|, |FT_y|\n")
        for phase_label in ('A', 'B', 'C', 'D', 'E'):
            if phase_label not in snapshots:
                continue
            coords_k, forces_k = snapshots[phase_label]
            FN_x, FN_y, FT_x, FT_y = extract_intersection_forces(
                forces_k.flatten(), node_dof_indices, intersection_nodes)
            for ii in range(n_beams_per_dir):
                for jj in range(n_beams_per_dir):
                    n_x = intersection_nodes[ii, jj, 0]
                    xpos = coords_k[n_x, 0]
                    ypos = coords_k[n_x, 1]
                    fh.write(
                        f"{phase_label}\t{ii}\t{jj}\t"
                        f"{xpos:.6e}\t{ypos:.6e}\t"
                        f"{FN_x[ii, jj]:.6e}\t{FN_y[ii, jj]:.6e}\t"
                        f"{FT_x[ii, jj]:.6e}\t{FT_y[ii, jj]:.6e}\n")
    print(f"Saved per-phase summary to '{out_phase_txt}'")

    # ---- Pull-trace text file ----------------------------------------------
    with out_pull_txt.open("w", encoding="utf-8") as fh:
        fh.write("# t_s\tpull_disp_m\tFpull_x_N\tsmooth_gap_min_m\t"
                 "engage_disp_m\tleft_back_m\tgeom_gap_min_m\n")
        for row in pull_trace:
            fh.write("\t".join(f"{v:.6e}" for v in row) + "\n")
    print(f"Saved pull-traction trace to '{out_pull_txt}'")

    # ---- Per-intersection contact-force time history text file -------------
    # Streamed incrementally during the run (see fh_fhist above); just close.
    fh_fhist.close()
    print(f"Saved per-intersection force history to '{out_fhist_txt}'")

    # ---- Summary file -------------------------------------------------------
    with out_summary_txt.open("w", encoding="utf-8") as fh:
        fh.write(f"NET STRAND PULL-OUT (model = {contact_model})\n")
        fh.write(f"contact_model        = {contact_model}\n")
        fh.write(f"n_nodes_per_beam     = {n_nodes_per_beam}\n")
        fh.write(f"n_beams_per_dir      = {n_beams_per_dir}\n")
        fh.write(f"L                    = {L}\n")
        fh.write(f"r_beam               = {r_beam}\n")
        fh.write(f"weave_amplitude      = {weave_amplitude}\n")
        fh.write(f"initial gap (mm)     = {(weave_amplitude - 2 * r_beam) * 1e3:.3f}\n")
        fh.write(f"d_imc                = {d_imc_contact}\n")
        fh.write(f"k_c mode             = {'adaptive' if fixed_kc is None else f'fixed {fixed_kc}'}\n")
        fh.write(f"use_line_search      = {use_line_search}\n")
        fh.write(f"mu                   = {mu_friction}\n")
        fh.write(f"vel_tol              = {vel_tol}\n")
        fh.write(f"end_disp_engage      = {end_disp_engage}\n")
        fh.write(f"end_disp_pullout     = {end_disp_pullout}\n")
        fh.write(f"t_engage T_A         = {t_engage}  -> {T_A}\n")
        fh.write(f"t_relax  T_B         = {t_relax_after_engage}  -> {T_B}\n")
        fh.write(f"t_release T_C        = {t_release}  -> {T_C}\n")
        fh.write(f"t_free   T_D         = {t_free_settle}  -> {T_D}\n")
        fh.write(f"t_pullout T_E        = {t_pullout}  -> {T_E}\n")
        fh.write(f"dt                   = {dt}\n")
        fh.write(f"snapshots_saved      = {saved_count}\n")
    print(f"Saved summary to '{out_summary_txt}'")

    # ---- Figures: deformed shape at end of each phase ----------------------
    figs_made = []
    if not make_plots:
        print("Skipping figure generation (--no-plot).")
        return {
            'coords_txt': str(out_coords_txt),
            'forces_txt': str(out_forces_txt),
            'phase_forces_txt': str(out_phase_txt),
            'pull_txt': str(out_pull_txt),
            'summary_txt': str(out_summary_txt),
            'figures': [],
        }
    for phase_label in ('A', 'B', 'C', 'D', 'E'):
        if phase_label not in snapshots:
            continue
        coords_k, _ = snapshots[phase_label]
        x_beams_pts = [coords_k[int(info['x_beam_start'][i]):
                                int(info['x_beam_start'][i]) + n_nodes_per_beam]
                       for i in range(n_beams_per_dir)]
        y_beams_pts = [coords_k[int(info['y_beam_start'][j]):
                                int(info['y_beam_start'][j]) + n_nodes_per_beam]
                       for j in range(n_beams_per_dir)]
        r_mm = r_beam * 1e3
        fig = plt.figure(figsize=(13, 9))

        # (a) Top view (x, y)
        ax_top = fig.add_subplot(2, 2, 1)
        for k, pts in enumerate(x_beams_pts):
            color = 'C2' if k == i_central else 'C0'
            ax_top.fill_between(pts[:, 0] * 1e3,
                                (pts[:, 1] - r_beam) * 1e3,
                                (pts[:, 1] + r_beam) * 1e3,
                                color=color, alpha=0.35)
            ax_top.plot(pts[:, 0] * 1e3, pts[:, 1] * 1e3, color=color, lw=0.8)
        for pts in y_beams_pts:
            ax_top.fill_betweenx(pts[:, 1] * 1e3,
                                 (pts[:, 0] - r_beam) * 1e3,
                                 (pts[:, 0] + r_beam) * 1e3,
                                 color='C3', alpha=0.35)
            ax_top.plot(pts[:, 0] * 1e3, pts[:, 1] * 1e3, color='C3', lw=0.8)
        ax_top.set_xlabel('x [mm]')
        ax_top.set_ylabel('y [mm]')
        ax_top.set_title(f"Top view (x, y) -- central x-beam in green")
        ax_top.set_aspect('equal')
        ax_top.grid(alpha=0.3)

        # (b) Side view (x, z)
        ax_xz = fig.add_subplot(2, 2, 2)
        for k, pts in enumerate(x_beams_pts):
            color = 'C2' if k == i_central else 'C0'
            ax_xz.fill_between(pts[:, 0] * 1e3,
                               (pts[:, 2] - r_beam) * 1e3,
                               (pts[:, 2] + r_beam) * 1e3,
                               color=color, alpha=0.35)
            ax_xz.plot(pts[:, 0] * 1e3, pts[:, 2] * 1e3, color=color, lw=0.8)
        # y-beams as circles in (x, z) projection
        from matplotlib.patches import Circle
        from matplotlib.collections import PatchCollection
        y_beam_circles = []
        for j in range(n_beams_per_dir):
            sB = int(info['y_beam_start'][j])
            pts = coords_k[sB:sB + n_nodes_per_beam]
            x_center = float(pts[:, 0].mean()) * 1e3
            z_center = float(pts[:, 2].mean()) * 1e3
            y_beam_circles.append(Circle((x_center, z_center), r_mm))
        ax_xz.add_collection(PatchCollection(y_beam_circles, facecolor='C3',
                                              edgecolor='C3', alpha=0.5))
        ax_xz.set_xlabel('x [mm]')
        ax_xz.set_ylabel('z [mm]')
        ax_xz.set_title(f"Side view (x, z) -- r = {r_mm:.2f} mm")
        ax_xz.set_aspect('equal')
        ax_xz.grid(alpha=0.3)

        # (c) 3D view (z exaggerated)
        ax_3d = fig.add_subplot(2, 2, 3, projection='3d')
        for k, pts in enumerate(x_beams_pts):
            color = 'C2' if k == i_central else 'C0'
            lw = 2.0 if k == i_central else 1.0
            ax_3d.plot(pts[:, 0] * 1e3, pts[:, 1] * 1e3, pts[:, 2] * 1e3,
                       color=color, lw=lw)
        for pts in y_beams_pts:
            ax_3d.plot(pts[:, 0] * 1e3, pts[:, 1] * 1e3, pts[:, 2] * 1e3,
                       color='C3', lw=1.0)
        ax_3d.set_xlabel('x [mm]')
        ax_3d.set_ylabel('y [mm]')
        ax_3d.set_zlabel('z [mm]')
        ax_3d.set_box_aspect((1.0, 1.0, 0.6))
        ax_3d.set_title('3D view (z exaggerated)')
        ax_3d.view_init(elev=20, azim=-60)

        # (d) Central x-beam standalone (x, z): shows the unweaving
        ax_central = fig.add_subplot(2, 2, 4)
        pts = x_beams_pts[i_central]
        ax_central.fill_between(pts[:, 0] * 1e3,
                                (pts[:, 2] - r_beam) * 1e3,
                                (pts[:, 2] + r_beam) * 1e3,
                                color='C2', alpha=0.35)
        ax_central.plot(pts[:, 0] * 1e3, pts[:, 2] * 1e3, color='C2', lw=1.0)
        # mark intersections
        for j in range(n_beams_per_dir):
            n_x = intersection_nodes[i_central, j, 0]
            ax_central.plot(coords_k[n_x, 0] * 1e3, coords_k[n_x, 2] * 1e3,
                            'ko', ms=4)
        ax_central.axhline(0, color='gray', lw=0.4)
        ax_central.set_xlabel('x [mm]')
        ax_central.set_ylabel('z [mm]')
        ax_central.set_title(f"Central x-beam (i={i_central})")
        ax_central.grid(alpha=0.3)
        ax_central.set_aspect('equal')

        phase_names = {'A': 'engage', 'B': 'relax', 'C': 'release-left',
                       'D': 'free-left', 'E': 'pull-out-right'}
        fig.suptitle(f"Phase {phase_label} ({phase_names[phase_label]})", fontsize=14)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        path = out_dir / f"net_pullout_shape_phase{phase_label}{tag_str}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figs_made.append(path)

    # ---- Pull traction figure ----------------------------------------------
    pt = np.asarray(pull_trace, dtype=float)
    if pt.size > 0:
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(pt[:, 0], pt[:, 2], 'b-', lw=1.0)
        axes[0].axvline(T_A, color='gray', ls='--', lw=0.6, label='A->B')
        axes[0].axvline(T_B, color='gray', ls='--', lw=0.6)
        axes[0].axvline(T_C, color='gray', ls='--', lw=0.6)
        axes[0].axvline(T_D, color='gray', ls='--', lw=0.6)
        axes[0].axhline(0, color='k', lw=0.4)
        axes[0].set_ylabel(r'$F_{contact,x}$ on right clamp [N]')
        axes[0].set_title('Contact force x-component on the central x-beam '
                          'right clamp (proxy for pull-out load)')
        axes[0].grid(alpha=0.3)
        axes[0].legend(['Fpull_x', 'A B C D phase ends'], loc='upper left',
                       fontsize=8)

        # Two gap signals:
        #   - geom_gap_min (column 6): geometric min surface gap across
        #     intersections, ALWAYS finite, falls smoothly as Phase A pulls
        #     the strands together.
        #   - smooth_gap_min (column 3): gap from contact-energy pairs;
        #     +inf until at least one pair enters the IMC smooth zone.
        axes[1].plot(pt[:, 0], pt[:, 6] * 1e3, 'g-', lw=1.2,
                     label='geometric min gap')
        sm = pt[:, 3].copy()
        sm[np.isinf(sm)] = np.nan
        axes[1].plot(pt[:, 0], sm * 1e3, 'r-', lw=1.2,
                     label='contact-energy gap (smooth-zone)')
        axes[1].axhline(0, color='k', lw=0.5)
        axes[1].axhline(d_imc_contact * 1e3, color='gray', lw=0.4, ls=':',
                        label='d_imc smooth-zone edge')
        axes[1].set_xlabel('time [s]')
        axes[1].set_ylabel('minimum surface gap [mm]')
        axes[1].set_title('Minimum surface gap across 25 intersections '
                          '(geometric = always finite)')
        axes[1].legend(loc='upper right', fontsize=8)
        axes[1].grid(alpha=0.3)

        fig.tight_layout()
        path = out_dir / f"net_pullout_pull_traction{tag_str}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figs_made.append(path)

    # ---- Per-intersection contact-force time history figure -----------------
    if per_intersection_t:
        nI = nJ = n_beams_per_dir
        t_arr = np.asarray(per_intersection_t)
        FN_arr = np.stack(per_intersection_FN, axis=0)   # (n_t, nI, nJ)
        FT_arr = np.stack(per_intersection_FT, axis=0)

        # Classify intersections so the 25 traces can be colored by role,
        # matching the convention used in contact_net_tensile.py.
        def role(i, j):
            corners = ((0, 0), (0, nJ - 1), (nI - 1, 0), (nI - 1, nJ - 1))
            if (i, j) in corners:
                return 'corner'
            if i == 0 or j == 0 or i == nI - 1 or j == nJ - 1:
                return 'edge'
            return 'interior'
        role_color = {'corner': 'C3', 'edge': 'C1', 'interior': 'C2'}

        # Highlight the row through the central x-beam i = i_central with a
        # bolder line so the user can see the pulled strand specifically.
        central_color = 'C0'

        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        for ii in range(nI):
            for jj in range(nJ):
                if ii == i_central:
                    color = central_color
                    lw = 1.6
                    alpha = 0.95
                else:
                    color = role_color[role(ii, jj)]
                    lw = 0.7
                    alpha = 0.6
                axes[0].plot(t_arr, FN_arr[:, ii, jj], color=color, lw=lw, alpha=alpha)
                axes[1].plot(t_arr, FT_arr[:, ii, jj], color=color, lw=lw, alpha=alpha)
        # Phase boundaries
        for ax in axes:
            for tT, lbl in zip((T_A, T_B, T_C, T_D), ('A|B', 'B|C', 'C|D', 'D|E')):
                ax.axvline(tT, color='gray', ls='--', lw=0.6)
                ax.text(tT, ax.get_ylim()[1] * 0.96 if ax.get_ylim()[1] > 0 else 0,
                        lbl, fontsize=7, color='gray', ha='center', va='top')
            ax.grid(alpha=0.3)
        axes[0].set_ylabel(r'$|F_N|$ at intersection  [N]')
        axes[0].set_title(f"Per-intersection contact-force time signal "
                          f"(model = {contact_model}, mu = {mu_friction})\n"
                          f"central x-beam (i={i_central}) row in blue; "
                          f"corner=red, edge=orange, interior=green")
        axes[1].set_ylabel(r'$|F_T|$ at intersection  [N]')
        axes[1].set_xlabel('time [s]')
        fig.tight_layout()
        path = out_dir / f"net_pullout_force_history{tag_str}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figs_made.append(path)

        # Companion figure: aggregated (sum across all pairs) and central-row
        # only — useful when 25 thin traces are too cluttered.
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        FN_central_row = FN_arr[:, i_central, :]   # (n_t, nJ)
        FT_central_row = FT_arr[:, i_central, :]
        for jj in range(nJ):
            axes[0].plot(t_arr, FN_central_row[:, jj], lw=1.2,
                         label=f'(i={i_central}, j={jj})')
            axes[1].plot(t_arr, FT_central_row[:, jj], lw=1.2)
        for ax in axes:
            for tT in (T_A, T_B, T_C, T_D):
                ax.axvline(tT, color='gray', ls='--', lw=0.6)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel(r'$|F_N|$  [N]')
        axes[0].set_title(f"Contact-force time signal on the central "
                          f"x-beam (i={i_central}), 5 intersections")
        axes[0].legend(loc='upper left', fontsize=8, ncol=2)
        axes[1].set_ylabel(r'$|F_T|$  [N]')
        axes[1].set_xlabel('time [s]')
        fig.tight_layout()
        path = out_dir / f"net_pullout_force_history_central{tag_str}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figs_made.append(path)

    print("Saved figures:")
    for p in figs_made:
        print(f"  {p}")

    return {
        'coords_txt': str(out_coords_txt),
        'forces_txt': str(out_forces_txt),
        'phase_forces_txt': str(out_phase_txt),
        'pull_txt': str(out_pull_txt),
        'summary_txt': str(out_summary_txt),
        'figures': [str(p) for p in figs_made],
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    results = run_pullout_simulation()
    print("\n" + "=" * 70)
    print("NET PULL-OUT COMPLETE")
    print("=" * 70)
    for k, v in results.items():
        print(f"  {k}: {v}")
