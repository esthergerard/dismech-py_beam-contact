#!/usr/bin/env python3
"""
Mechanism study: what dominates the contact-force perturbation of the
2-node (segment) IMC baseline while contact slides across the mesh?

Candidates (tested, not assumed):
  M1  surrogate-surface undulation (chord sagitta eta2 = kappa h^2 / 8)
  M2  active-set discretisation (candidate set frozen on the first Newton
      iteration of each step; contact_energy.py:98-104, 10*delta margin)
  M3  cutoff truncation (potential truncated at Dbar = 2 + delta)

Harness design — HONEST DEVIATIONS FROM THE SPEC
------------------------------------------------
* KINEMATIC sweep, not an elastic solve.  The rod (a circular arc of
  curvature kappa, or a straight rod for kappa=0) is swept rigidly along
  its own centerline past a fixed rigid cylinder, so the nominal gap is
  constant EXACTLY and the sweep can afford ~h/1000 sampling (metric B
  needs three decimation decades of the same log).  The contact force is
  evaluated through the real IMCEnergy pipeline (Lumelsky closest points,
  truncated smooth potential, broad-phase candidate logic) at every
  sample.  Elastic compliance would low-pass the amplitudes but does not
  generate any of the candidate perturbations; the discrimination ratios
  and scalings are properties of the force evaluation being sampled here.
* M2 emulation: the solver freezes the candidate set on the first Newton
  iteration of a step.  With no solve there are no iterations, so the
  baseline reproduces the freeze by rebuilding the candidate set only at
  step boundaries (every STEPS_PER_H-th sample) and evaluating in between
  with the frozen set (first_iter=False), exactly the solver's call
  pattern.  R4 rebuilds at every sample.
* The implemented potential is the SQUARED log form
  E = (log(1+exp(c(2-Dbar)))/c)^2  (contact_energy.py:11), not the
  unsquared U of the spec, so M3's closed-form step 2/(1+exp(c*delta))
  does not apply verbatim.  The truncation step in |dE/dD| at the cutoff
  for the implemented form is  2*(log(1+e^{-c d})/c)*(e^{-c d}/(1+e^{-c d}))
  with d = delta_bar; both forms are reported.
* Adaptive contact stiffness is disabled by a subclass whose
  get_contact_stiffness is a no-op (instrumentation only; solver files
  untouched).  k_c = 1 in every run.

Geometry
--------
Rod: N nodes over length L_ROD = 0.5 m; curvature kappa about the z-normal
(arc in the xy-plane, center at the origin) or straight along y for
kappa=0.  Rigid cylinder: one fixed edge of the same radius a, parallel to
z, whose axis passes at nominal centerline clearance DBAR0*a from the rod
centerline at the crossing.  Sliding = rotating the arc about its own
center (translating the straight rod along itself): every material point
retraces the same curve, so the swept surface, and hence the nominal gap,
is invariant; only the discretisation (node phase) moves.

Usage
-----
    python harness_slide_2node.py            # all runs (R6 gated on timing)
    python harness_slide_2node.py R1 R3 ...  # selected runs
Outputs one .npz per run in results/ with the full parameter set embedded.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from dismech.contact.imc_energy import IMCEnergy               # noqa: E402
from dismech.contact.contact_pairs import ContactPair          # noqa: E402

OUT = Path(__file__).resolve().parent / 'results'

# ---- fixed parameters --------------------------------------------------------
A_RAD = 1.0e-3          # rod (and cylinder) radius a [m]
C_DIMLESS = 50.0        # c: smoothing sharpness on Dbar (k_1 = c/a)
L_ROD = 0.5             # rod length [m]
DBAR0 = 1.95            # nominal centerline distance at NODE phase, /a
SLIDE_H = 12.0          # sliding span in units of edge length h
SAMPLES_PER_H = 1000    # finest log density (metric B decimates this)
STEPS_PER_H = 100       # emulated time steps per edge length (freeze grid)
L_CYL = 0.2             # fixed cylinder edge length [m] (axis along z)
KC = 1.0

RUNS = {
    #        kappa  delta_bar   N    refresh_every_sample
    'R1': dict(kappa=5.0, delta=0.15, N=50,  refresh=False),
    'R2': dict(kappa=0.0, delta=0.15, N=50,  refresh=False),
    'R3': dict(kappa=5.0, delta=0.05, N=50,  refresh=False),
    'R4': dict(kappa=5.0, delta=0.15, N=50,  refresh=True),
    'R5': dict(kappa=5.0, delta=0.15, N=100, refresh=False),
    'R6': dict(kappa=5.0, delta=0.15, N=200, refresh=False),
}


class FixedKcIMC(IMCEnergy):
    """IMCEnergy with the adaptive contact stiffness disabled (k_c frozen).
    Instrumentation-only subclass; the solver classes are untouched."""

    def get_contact_stiffness(self, robot, F):
        return


class _StateStub:
    def __init__(self, q):
        self.q = q


class _RobotStub:
    end_node_dof_index = 0   # only read by the (disabled) adaptive stiffness


def build_positions(kappa, N, s):
    """Node positions of the rod slid by arc length s, plus cylinder nodes.

    kappa > 0: arc of radius R=1/kappa centered at the origin in the
    xy-plane; sliding rotates the nodes by s/R.  The fixed cylinder axis
    (parallel to z) passes through (R + DBAR0*a, 0): closest approach of
    the CONTINUOUS centerline is exactly DBAR0*a, independent of s.

    kappa = 0: rod along y at x = 0, slid by s in +y; cylinder axis
    passes through (DBAR0*a, 0).
    """
    if kappa > 0:
        R = 1.0 / kappa
        theta_span = L_ROD / R
        th = np.linspace(-theta_span / 2, theta_span / 2, N) + s / R
        rod = np.stack([R * np.cos(th), R * np.sin(th), np.zeros(N)], axis=1)
        x_axis = R + DBAR0 * A_RAD
    else:
        y = np.linspace(-L_ROD / 2, L_ROD / 2, N) + s
        rod = np.stack([np.zeros(N), y, np.zeros(N)], axis=1)
        x_axis = DBAR0 * A_RAD
    cyl = np.array([[x_axis, 0.0, -L_CYL / 2],
                    [x_axis, 0.0, +L_CYL / 2]])
    return np.concatenate([rod, cyl])


def make_energy(N, delta_bar):
    """One ContactPair per rod edge against the single cylinder edge."""
    def map_node_to_dof(nodes):
        return (3 * np.asarray(nodes))[..., None] + np.array([0, 1, 2])
    pairs = [ContactPair(np.array([i, i + 1, N, N + 1]), map_node_to_dof)
             for i in range(N - 1)]
    return FixedKcIMC(pairs, delta=delta_bar * A_RAD, h=A_RAD, kc=KC,
                      k_1=C_DIMLESS / A_RAD, scale=True)


def run_one(name, kappa, delta, N, refresh):
    h = L_ROD / (N - 1)
    n_samp = int(SLIDE_H * SAMPLES_PER_H) + 1
    s_grid = np.linspace(0.0, SLIDE_H * h, n_samp)
    freeze_every = SAMPLES_PER_H // STEPS_PER_H       # samples per "step"

    energy = make_energy(N, delta)
    robot = _RobotStub()
    zero_F = np.zeros(3 * (N + 2))
    cyl_dofs = np.arange(3 * N, 3 * (N + 2))

    force = np.zeros((n_samp, 3))
    n_support = np.zeros(n_samp, dtype=int)
    pair_log = []          # per sample: active pairs (support), t, u, Dbar

    t0 = time.time()
    for k, s in enumerate(s_grid):
        q = build_positions(kappa, N, s).ravel()
        first = refresh or (k % freeze_every == 0)
        Fs, _ = energy.grad_hess_energy(_StateStub(q), robot, zero_F, first)
        force[k] = Fs[cyl_dofs].reshape(2, 3).sum(axis=0)

        # instrumentation: closest-point parameters and gap per candidate
        qn = q * energy.scale
        Delta = energy.get_Delta(qn)                   # normalized (Dbar*h)
        t_par, u_par = energy.get_lumelsky_coeff(qn)
        support = Delta < 2 * energy.norm_h + energy.norm_delta
        n_support[k] = int(np.count_nonzero(support))
        pair_ids = energy.pairs[support][:, 0] if support.any() else []
        pair_log.append({
            'pairs': [int(p) for p in np.atleast_1d(pair_ids)],
            't': [float(x) for x in t_par[support]],
            'u': [float(x) for x in u_par[support]],
            'Dbar': [float(x) for x in Delta[support]],
        })
    wall = time.time() - t0

    eta2 = kappa * h ** 2 / 8.0
    params = dict(run=name, kappa=kappa, delta_bar=delta, N=N,
                  refresh_every_sample=refresh, a=A_RAD, c=C_DIMLESS,
                  L_rod=L_ROD, h=h, Dbar0=DBAR0, slide_span_h=SLIDE_H,
                  samples_per_h=SAMPLES_PER_H, steps_per_h=STEPS_PER_H,
                  freeze_every_samples=freeze_every, k_c=KC,
                  eta2=eta2, eta2_over_delta_a=eta2 / (delta * A_RAD),
                  wall_s=wall)
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(
        OUT / f'{name.lower()}.npz',
        s=s_grid, s_over_h=s_grid / h, force=force, n_support=n_support,
        params=json.dumps(params),
        pair_log=json.dumps(pair_log))
    print(f'[{name}] N={N} kappa={kappa} delta={delta} refresh={refresh}: '
          f'{n_samp} samples in {wall:.1f}s -> results/{name.lower()}.npz')
    return wall


if __name__ == '__main__':
    wanted = [a.upper() for a in sys.argv[1:]] or list(RUNS)
    # Time the baseline first; R6 cost scales ~ N (pairs) x samples (same
    # count) — extrapolate before launching it, as per the study spec.
    if 'R1' in wanted:
        w1 = run_one('R1', **RUNS['R1'])
        est_r6 = w1 * (RUNS['R6']['N'] / RUNS['R1']['N'])
        print(f'    baseline wall {w1:.1f}s -> R6 estimated ~{est_r6:.0f}s')
        if 'R6' in wanted and est_r6 > 1800:
            print('    R6 estimate exceeds 30 min budget — dropping it '
                  '(R1+R5 still give a 2-point slope).')
            wanted.remove('R6')
    for name in wanted:
        if name == 'R1':
            continue
        run_one(name, **RUNS[name])
