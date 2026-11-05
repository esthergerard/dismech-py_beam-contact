#!/usr/bin/env python3
"""
Static multiplicity probe: does the 2-node routine double-count at shared
nodes, and does the 3-node partition of unity remove it?

Three rigid configurations, no solve:
  (i)   straight rod A + sphere obstacle at fixed gap, swept along x
  (ii)  rod A with a 3-node apex (parabola, apex AT a node), rod B nearly
        straight, swept past the apex
  (iii) rod A and rod B both with a 3-node apex, apexes facing, B swept

Per sample and per routine: active pair list, per-pair (Delta, t*, u*) or
(gap, s1, s2), total |F|, and multiplicity
    M = sum_pairs f_pair / max_pairs |dU/dDelta| ,
where f_pair is the pair's effective kernel magnitude (|dU/dDelta| for the
2-node routine; w1*w2*|dU/ddelta_c| for the 3-node routine) and the
denominator is the UNWEIGHTED kernel maximum.  For the 2-node routine the
two readings coincide; for the 3-node routine M == 1 is the partition-of-
unity statement.

2-node routine: IMCEnergy (adaptive k_c disabled), delta = 0.15a, default
k_1.  3-node routines: circle_four_nodes (rod-to-rigid, config i) and
Eberly beam-to-beam (configs ii/iii), default d_imc = 0.8a.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from dismech.contact.imc_energy import IMCEnergy                     # noqa: E402
from dismech.contact.contact_pairs import ContactPair                # noqa: E402
from dismech.contact.contact_energy import compute_energy_grad_hess  # noqa: E402
from dismech.contact.circle_four_nodes_imc_weighted import (         # noqa: E402
    compute_gap_with_si, compute_weights_from_s,
    compute_dE_IMC_ddelta_c, compute_contact_forces_wrt_q,
)
from dismech.contact.eberly_contact_3d import (                      # noqa: E402
    compute_eberly_3d_beam_to_beam_contact_forces,
)

A = 1e-3                 # rod radius [m]
H = 1e-2                 # edge length [m]
NROD = 13                # nodes per rod (apex at node 6, x=0)
PEN = 0.05 * A           # centerline approach 1.95a
D0 = 2 * A - PEN
C2 = 2.5                 # apex parabola y = -C2 x^2  (kappa = 5 /m)
C2_STRAIGHT = 0.01       # "straight" rod B bow in (ii)  (kappa = 0.02 /m)
DELTA_2N = 0.15 * A      # 2-node smooth-zone half width
D_IMC_3N = 0.0008        # 3-node default
SAMPLES = 601            # ~200 per edge over 3 edge lengths
XS = np.linspace(-1.5 * H, 1.5 * H, SAMPLES)
XGRID = (np.arange(NROD) - (NROD - 1) / 2) * H
OUT = Path(__file__).resolve().parent / 'results'


class FixedKcIMC(IMCEnergy):
    def get_contact_stiffness(self, robot, F):
        return


class _State:
    def __init__(self, q):
        self.q = q


class _Robot:
    end_node_dof_index = 0


def map_dof(nodes):
    return (3 * np.asarray(nodes))[..., None] + np.array([0, 1, 2])


def rod_apex_A():
    return np.stack([XGRID, -C2 * XGRID ** 2, np.zeros(NROD)], axis=1)


def rod_B(xs, c2):
    return np.stack([XGRID + xs, D0 + c2 * XGRID ** 2, np.zeros(NROD)],
                    axis=1)


def sweep_2node(name, nodesA_fn, nodesB_fn, nB_nodes, en=None):
    nA = NROD
    if en is None:
        pairs = [ContactPair(np.array([i, i + 1, nA + j, nA + j + 1]),
                             map_dof)
                 for i in range(nA - 1) for j in range(nB_nodes - 1)]
        en = FixedKcIMC(pairs, delta=DELTA_2N, h=A, kc=1.0, scale=True)
    rows = []
    for xs in XS:
        q = np.concatenate([nodesA_fn(xs), nodesB_fn(xs)]).ravel()
        Fs, _ = en.grad_hess_energy(_State(q), _Robot(),
                                    np.zeros(q.size), True)
        Fmag = float(np.linalg.norm(
            Fs.reshape(-1, 3)[nA:].sum(axis=0)))
        qn = q * en.scale
        Delta = en.get_Delta(qn)
        t, u = en.get_lumelsky_coeff(qn)
        _, gE, _ = compute_energy_grad_hess(
            Delta, en.norm_delta, en.norm_h, en.norm_k_1)
        act = np.abs(gE) > 0
        f = np.abs(gE[act])
        M = float(f.sum() / f.max()) if act.any() else 0.0
        rows.append(dict(
            xs=float(xs), F=Fmag, M=M,
            pairs=[[int(en.pairs[k][0]), int(en.pairs[k][2])]
                   for k in np.nonzero(act)[0]],
            Delta=[float(x) for x in Delta[act]],
            t=[float(x) for x in t[act]], u=[float(x) for x in u[act]]))
    return rows


def sweep_i_2node():
    eps = 5e-5
    rodA = np.stack([XGRID, np.zeros(NROD), np.zeros(NROD)], axis=1)
    obs = lambda xs: np.array([[xs, -D0, -eps], [xs, -D0, +eps]])
    return sweep_2node('i', lambda xs: rodA, obs, 2)


def sweep_i_3node():
    """Rod-to-rigid weighted 3-node path (2D).  The ROD is translated by
    -xs instead of moving the obstacle: identical relative motion, but the
    per-obstacle JAX JIT cache in compute_contact_forces_wrt_q stays warm
    (a moving center would recompile at every sample)."""
    center = np.array([0.0, -(A + A - PEN)])      # sphere radius = a
    K = 15.0 / D_IMC_3N
    rows = []
    for xs in XS:
        q2 = np.stack([XGRID - xs, np.zeros(NROD)], axis=1).ravel()
        _, F, _, _, weights = compute_contact_forces_wrt_q(
            q2, center, A, A, k_c=1.0)
        f_eff, f_raw, per = [], [], []
        for seg_id, w in weights.items():
            i = seg_id - 1
            gap, s_i = compute_gap_with_si(
                q2[2 * i:2 * i + 2], q2[2 * i + 2:2 * i + 4],
                q2[2 * i + 4:2 * i + 6], center, A, A)
            dU = abs(float(compute_dE_IMC_ddelta_c(gap + A, A, D_IMC_3N, K)))
            if dU > 0:
                f_eff.append(w * dU)
                f_raw.append(dU)
                per.append((i, float(gap), float(s_i), float(w)))
        M = float(np.sum(f_eff) / np.max(f_raw)) if f_raw else 0.0
        Fobs = float(np.linalg.norm(F.reshape(-1, 2).sum(axis=0)))
        rows.append(dict(xs=float(xs), F=Fobs, M=M,
                         pairs=[p[0] for p in per],
                         Delta=[p[1] for p in per],
                         t=[p[2] for p in per], u=[p[3] for p in per]))
    return rows


def sweep_eberly(nodesB_fn):
    class R:
        n_nodes = 2 * NROD
        node_dof_indices = np.arange(6 * NROD).reshape(2 * NROD, 3)
    K = 15.0 / D_IMC_3N
    rows = []
    for xs in XS:
        q = np.concatenate([rod_apex_A(), nodesB_fn(xs)]).ravel()
        F, _, _, gaps = compute_eberly_3d_beam_to_beam_contact_forces(
            R(), q, A, k_c=1.0, n_nodes_beam_A=NROD, d_imc=D_IMC_3N)
        f_eff, f_raw, per = [], [], []
        for g in gaps:
            w = (compute_weights_from_s(g['s_i1'])
                 * compute_weights_from_s(g['s_i2']))
            dU = abs(float(compute_dE_IMC_ddelta_c(
                g['gap'] + A, A, D_IMC_3N, K)))
            if dU > 0:
                f_eff.append(w * dU)
                f_raw.append(dU)
                per.append((g['segment_pair'], float(g['gap']),
                            float(g['s_i1']), float(g['s_i2'])))
        M = float(np.sum(f_eff) / np.max(f_raw)) if f_raw else 0.0
        Fmag = float(np.linalg.norm(F.reshape(-1, 3)[NROD:].sum(axis=0)))
        rows.append(dict(xs=float(xs), F=Fmag, M=M,
                         pairs=[p[0] for p in per],
                         Delta=[p[1] for p in per],
                         t=[p[2] for p in per], u=[p[3] for p in per]))
    return rows


def summarize(tag, rows):
    M = np.array([r['M'] for r in rows])
    F = np.array([r['F'] for r in rows])
    act = M > 0
    print(f'{tag:14s} M: min={M[act].min():.3f} median='
          f'{np.median(M[act]):.3f} max={M.max():.3f}   '
          f'|F|: [{F[act].min():.3g}, {F.max():.3g}]')
    k = int(np.argmax(M))
    r = rows[k]
    print(f'{"":14s}argmax M at xs/h={r["xs"] / H:+.3f}: pairs={r["pairs"]}')
    print(f'{"":14s}  Delta={["%.4g" % x for x in r["Delta"]]}')
    print(f'{"":14s}  t*={["%.3f" % x for x in r["t"]]} '
          f'u*={["%.3f" % x for x in r["u"]]}')
    return M


def main():
    OUT.mkdir(exist_ok=True)
    res = {}
    print('=== 2-node routine (IMCEnergy) ===')
    res['i_2node'] = sweep_i_2node()
    summarize('(i) sphere', res['i_2node'])
    pairs_bb = [ContactPair(np.array([i, i + 1, NROD + j, NROD + j + 1]),
                            map_dof)
                for i in range(NROD - 1) for j in range(NROD - 1)]
    en_bb = FixedKcIMC(pairs_bb, delta=DELTA_2N, h=A, kc=1.0, scale=True)
    res['ii_2node'] = sweep_2node(
        'ii', lambda xs: rod_apex_A(), lambda xs: rod_B(xs, C2_STRAIGHT),
        NROD, en=en_bb)
    summarize('(ii) apex-str', res['ii_2node'])
    res['iii_2node'] = sweep_2node(
        'iii', lambda xs: rod_apex_A(),
        lambda xs: rod_B(xs, C2), NROD, en=en_bb)
    summarize('(iii) apex2', res['iii_2node'])

    print('=== 3-node routines ===')
    res['i_3node'] = sweep_i_3node()
    M = summarize('(i) circle-w', res['i_3node'])
    print(f'{"":14s}max|M-1| = {np.max(np.abs(M[M > 0] - 1)):.3e}')
    res['ii_3node'] = sweep_eberly(lambda xs: rod_B(xs, C2_STRAIGHT))
    M = summarize('(ii) eberly', res['ii_3node'])
    print(f'{"":14s}max|M-1| = {np.max(np.abs(M[M > 0] - 1)):.3e}')
    res['iii_3node'] = sweep_eberly(lambda xs: rod_B(xs, C2))
    M = summarize('(iii) eberly', res['iii_3node'])
    print(f'{"":14s}max|M-1| = {np.max(np.abs(M[M > 0] - 1)):.3e}')

    with open(OUT / 'apex_multiplicity.json', 'w') as f:
        json.dump(res, f)
    print(f'\nlogs -> {OUT / "apex_multiplicity.json"}')


if __name__ == '__main__':
    main()
