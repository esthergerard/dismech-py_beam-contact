"""
Correctness and regression tests for ``dismech.contact.eberly_contact_3d``.

Two kinds of tests:

1. Implementation-independent checks
   - ``arc_min_dist`` against a dense brute-force scan of both arcs;
   - the envelope-theorem gradient ``compute_gradient_nodes`` against
     central finite differences (interior minima only, where the envelope
     theorem holds exactly).

2. Golden regression tests
   Outputs of every public entry point (and the per-pair kernel) on a fixed
   set of seeded configurations, compared against
   ``tests/resources/eberly_golden.npz``.  These pin the CURRENT behaviour
   so that refactors / performance work can be verified to be
   behaviour-preserving.

   Regenerate the golden file (only when a behaviour change is intended)
   with:

       python tests/test_eberly_contact.py --regen
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.append(str(pathlib.Path(__file__).parent.parent / 'src'))

from dismech.contact.eberly_contact_3d import (            # noqa: E402
    arc_interval_from_nodes,
    arc_min_dist,
    build_plane_frame,
    circle_from_three_nodes,
    compute_eberly_3d_beam_to_beam_contact_forces,
    compute_eberly_3d_beam_to_beam_friction_apply,
    compute_eberly_3d_beam_to_beam_friction_force,
    compute_eberly_3d_beam_to_beam_friction_force_with_jac,
    compute_eberly_3d_beam_to_beam_friction_forces,
    compute_eberly_3d_beam_to_beam_friction_pairs,
    compute_eberly_3d_beam_to_beam_friction_setup,
    compute_eberly_3d_self_contact_forces,
    compute_eberly_3d_self_friction_apply,
    compute_eberly_3d_self_friction_forces,
    compute_eberly_3d_self_friction_setup,
    compute_gradient_nodes,
    compute_hessian_nodes,
    extend_arc,
    _compute_pair_force_and_hessian,
    _compute_pair_force_and_hessian_line,
)

GOLDEN_PATH = pathlib.Path(__file__).parent / 'resources' / 'eberly_golden.npz'

# Geometry scale used throughout: matches the net-pullout benchmark.
R_BEAM = 1.5e-3          # beam cross-section radius [m]
D_IMC = 5e-5             # IMC smooth-zone half-width [m]
K_IMC = 15.0 / D_IMC
H_SEG = 0.01             # half-span of a 3-node segment [m]


# ============================================================================
# Configuration generators
# ============================================================================

def _triplet(center, u_dir, sag_dir, half_span, sag, jitter, rng):
    """3-node triplet: endpoints at center +- half_span*u_dir, middle sagging
    by `sag` along sag_dir, plus optional random jitter."""
    center = np.asarray(center, float)
    u_dir = np.asarray(u_dir, float) / np.linalg.norm(u_dir)
    sag_dir = np.asarray(sag_dir, float) / np.linalg.norm(sag_dir)
    n0 = center - half_span * u_dir + sag * sag_dir
    n1 = center.copy()
    n2 = center + half_span * u_dir + sag * sag_dir
    if jitter > 0:
        n0 = n0 + rng.normal(scale=jitter, size=3)
        n1 = n1 + rng.normal(scale=jitter, size=3)
        n2 = n2 + rng.normal(scale=jitter, size=3)
    return (n0, n1, n2)


def pair_configs():
    """Named (NODES1, NODES2) configurations for the per-pair kernel.

    Cover: interior minima (generic crossing), penetrating / separated gaps,
    endpoint-active (corner) minima, parallel-plane circles, near-straight
    triplets.  Gap values are relative to surface contact at 2*R_BEAM.
    """
    cfgs = []
    for k, gap in enumerate((-4e-5, -1e-5, +2e-5)):
        rng = np.random.default_rng(100 + k)
        z = 2 * R_BEAM + gap
        n1 = _triplet((0, 0, 0), (1, 0, 0), (0, 0, 1), H_SEG, 2e-3, 1e-4, rng)
        n2 = _triplet((0, 0, z), (0, 1, 0), (0, 0, 1), H_SEG, 2e-3, 1e-4, rng)
        cfgs.append((f'crossing_gap{gap:+.0e}', n1, n2))

    # Corner-active: second arc shifted along x beyond the first arc's span.
    rng = np.random.default_rng(200)
    z = 2 * R_BEAM - 1e-5
    n1 = _triplet((0, 0, 0), (1, 0, 0), (0, 0, 1), H_SEG, 2e-3, 0.0, rng)
    n2 = _triplet((1.6 * H_SEG, 0, z), (0, 1, 0), (0, 0, 1), H_SEG, 2e-3,
                  0.0, rng)
    cfgs.append(('corner_shifted', n1, n2))

    # Parallel planes: both triplets sag inside z = const planes.
    rng = np.random.default_rng(300)
    z = 2 * R_BEAM - 2e-5
    n1 = _triplet((0, 0, 0), (1, 0, 0), (0, 1, 0), H_SEG, 2e-3, 0.0, rng)
    n2 = _triplet((2e-3, 1e-3, z), (1, 0, 0), (0, 1, 0), H_SEG, 2e-3,
                  0.0, rng)
    cfgs.append(('parallel_planes', n1, n2))

    # Nearly straight (huge fitted radius) -- exercises ill-conditioned arcs.
    rng = np.random.default_rng(400)
    z = 2 * R_BEAM - 1e-5
    n1 = _triplet((0, 0, 0), (1, 0, 0), (0, 0, 1), H_SEG, 1e-7, 0.0, rng)
    n2 = _triplet((0, 0, z), (0, 1, 0), (0, 0, 1), H_SEG, 2e-3, 0.0, rng)
    cfgs.append(('near_straight', n1, n2))

    return cfgs


class DuckRobot:
    """Minimal robot stand-in: the Eberly entry points only read
    ``n_nodes`` and ``node_dof_indices``."""

    def __init__(self, n_nodes):
        self.n_nodes = n_nodes
        self.node_dof_indices = np.arange(3 * n_nodes).reshape(n_nodes, 3)


def two_beam_setup():
    """Two crossing 9-node beams (a node at the crossing on each), both
    genuinely curved, with slight surface penetration at the crossing so
    several segment pairs sit inside the activation zone.
    Returns (robot, q, u, n_nodes_A)."""
    n_a, n_b = 9, 9
    length = 0.08
    s = np.linspace(-length / 2, length / 2, n_a)
    amp_a, amp_b = 1.2e-3, 0.8e-3
    beam_a = np.stack([s, np.zeros(n_a),
                       amp_a * np.cos(2 * np.pi * s / length)], axis=1)
    # Note: the contact primitive is the arc through the EDGE MIDPOINTS of
    # each 3-node triplet (not through the nodes), which sits inside the
    # sampled cosine by its half-chord sagitta (~1.5e-4 here for both beams
    # combined).  The offset compensates so the arc-to-arc gap at the
    # crossing lands near -3e-5 (inside the activation zone).
    z_b = amp_a + 2 * R_BEAM - 1.8e-4 + amp_b
    beam_b = np.stack([np.zeros(n_b), s,
                       z_b - amp_b * np.cos(2 * np.pi * s / length)], axis=1)

    rng = np.random.default_rng(7)
    nodes = np.vstack([beam_a, beam_b])
    nodes += rng.normal(scale=5e-6, size=nodes.shape)

    robot = DuckRobot(n_a + n_b)
    q = nodes.ravel().astype(float)
    u = rng.normal(scale=1e-2, size=q.shape)      # synthetic velocities
    return robot, q, u, n_a


def hairpin_setup():
    """Single 16-node beam folded onto itself (two anti-parallel legs a
    surface-gap apart) so the SELF-contact path finds pairs with
    |i - j| >= min_segment_separation."""
    n_leg = 8
    length = 0.08
    sep = 2 * R_BEAM - 1e-5
    s = np.linspace(-length / 2, length / 2, n_leg)
    leg1 = np.stack([s, np.zeros(n_leg),
                     1e-3 * np.cos(2 * np.pi * s / length)], axis=1)
    leg2 = np.stack([-s, np.full(n_leg, 1e-4),
                     leg1[::-1, 2] + sep], axis=1)
    rng = np.random.default_rng(11)
    nodes = np.vstack([leg1, leg2])
    nodes += rng.normal(scale=2e-5, size=nodes.shape)

    robot = DuckRobot(2 * n_leg)
    q = nodes.ravel().astype(float)
    u = rng.normal(scale=1e-2, size=q.shape)
    return robot, q, u


# ============================================================================
# Brute-force reference for the arc-to-arc minimum distance
# ============================================================================

def brute_force_min_dist(C1, R1, ax1, arc1, C2, R2, ax2, arc2, n=2001):
    """Dense scan of both arcs.  Returns the grid minimum distance, an upper
    bound on the true minimum that is tight to O(grid step squared)."""
    e11, e12 = build_plane_frame(ax1)
    e21, e22 = build_plane_frame(ax2)

    def span(arc):
        a, b = float(arc[0]), float(arc[1])
        return np.linspace(a, a + ((b - a) % (2 * np.pi) or 2 * np.pi), n)

    t1, t2 = span(arc1), span(arc2)
    P1 = C1 + R1 * (np.cos(t1)[:, None] * e11 + np.sin(t1)[:, None] * e12)
    P2 = C2 + R2 * (np.cos(t2)[:, None] * e21 + np.sin(t2)[:, None] * e22)
    # (n, n) distance grid, blocked to keep memory modest
    dmin = np.inf
    block = 500
    for i0 in range(0, n, block):
        d2 = ((P1[i0:i0 + block, None, :] - P2[None, :, :]) ** 2).sum(-1)
        dmin = min(dmin, float(np.sqrt(d2.min())))
    return dmin


def _fit_pair(nodes1, nodes2):
    C1, R1, ax1 = circle_from_three_nodes(*nodes1)
    C2, R2, ax2 = circle_from_three_nodes(*nodes2)
    arc1 = extend_arc(arc_interval_from_nodes(*nodes1, C1, R1, ax1))
    arc2 = extend_arc(arc_interval_from_nodes(*nodes2, C2, R2, ax2))
    return C1, R1, ax1, arc1, C2, R2, ax2, arc2


# ============================================================================
# 1. Implementation-independent tests
# ============================================================================

@pytest.mark.parametrize('name,nodes1,nodes2', pair_configs())
def test_arc_min_dist_matches_bruteforce(name, nodes1, nodes2):
    try:
        C1, R1, ax1, arc1, C2, R2, ax2, arc2 = _fit_pair(nodes1, nodes2)
    except (ValueError, np.linalg.LinAlgError):
        pytest.skip('degenerate arc fit (covered by line-fallback tests)')

    res = arc_min_dist(C1, R1, ax1, arc1, C2, R2, ax2, arc2)
    d_grid = brute_force_min_dist(C1, R1, ax1, arc1, C2, R2, ax2, arc2)

    # The analytic minimum can only sit at or below the grid minimum
    # (grid resolution error), and never meaningfully above it.
    assert res['dist'] <= d_grid + 1e-9, \
        f'{name}: analytic dist above brute-force minimum'
    assert res['dist'] >= d_grid - 5e-6, \
        f'{name}: analytic dist far below brute-force minimum (wrong branch?)'

    # Reported closest points must reproduce the reported distance.
    d_pts = float(np.linalg.norm(res['M1'] - res['M2']))
    np.testing.assert_allclose(d_pts, res['dist'], rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize('name,nodes1,nodes2',
                         [c for c in pair_configs()
                          if c[0].startswith('crossing')])
def test_interior_gradient_matches_fd(name, nodes1, nodes2):
    """Envelope-theorem gradient of d* w.r.t. the 18 node coordinates,
    checked with central finite differences.  Interior minima only."""
    C1, R1, ax1, arc1, C2, R2, ax2, arc2 = _fit_pair(nodes1, nodes2)
    res = arc_min_dist(C1, R1, ax1, arc1, C2, R2, ax2, arc2)
    if res['kind'] != 'interior':
        pytest.skip(f'{name}: minimum is not interior ({res["kind"]})')

    grad = compute_gradient_nodes(res, C1, R1, ax1, nodes1,
                                  C2, R2, ax2, nodes2)
    dd_dN = np.concatenate([grad['dd_dN1'], grad['dd_dN2']])

    def dist_of(x):
        n1 = (x[0:3], x[3:6], x[6:9])
        n2 = (x[9:12], x[12:15], x[15:18])
        a, b, c, d, e, f, g, h = _fit_pair(n1, n2)
        return arc_min_dist(a, b, c, d, e, f, g, h)['dist']

    x0 = np.concatenate([np.concatenate(nodes1), np.concatenate(nodes2)])
    eps = 1e-7
    fd = np.zeros(18)
    for k in range(18):
        xp = x0.copy(); xp[k] += eps
        xm = x0.copy(); xm[k] -= eps
        fd[k] = (dist_of(xp) - dist_of(xm)) / (2 * eps)

    # atol sits at the central-difference noise floor for eps = 1e-7.
    np.testing.assert_allclose(dd_dN, fd, rtol=2e-4, atol=5e-7,
                               err_msg=f'{name}: envelope gradient vs FD')


def test_pair_hessian_symmetric_and_force_direction():
    """Gauss-Newton pair Hessian must be symmetric; the force must be
    parallel to dd_dN (single energy factor)."""
    _, nodes1, nodes2 = pair_configs()[0]
    C1, R1, ax1, arc1, C2, R2, ax2, arc2 = _fit_pair(nodes1, nodes2)
    energy, force, hess, gap, s1, s2 = _compute_pair_force_and_hessian(
        nodes1, nodes2, arc1, arc2, R_BEAM, D_IMC, K_IMC,
        False, False, False, False)

    assert energy > 0.0 and gap < D_IMC
    np.testing.assert_allclose(hess, hess.T, rtol=0, atol=1e-9 * np.abs(hess).max())
    # rank-1 Gauss-Newton: hess ~ c * outer(g, g) with force ~ g
    gnorm = force / np.linalg.norm(force)
    resid = hess - (gnorm @ hess @ gnorm) * np.outer(gnorm, gnorm)
    assert np.abs(resid).max() <= 1e-8 * np.abs(hess).max()


def test_beam_to_beam_no_force_when_separated():
    robot, q, _, n_a = two_beam_setup()
    q_far = q.copy()
    # lift beam B far above beam A
    q_far.reshape(-1, 3)[n_a:, 2] += 1.0
    F, J, gap_min, gaps = compute_eberly_3d_beam_to_beam_contact_forces(
        robot, q_far, R_BEAM, k_c=1.0, n_nodes_beam_A=n_a, d_imc=D_IMC)
    assert np.all(F == 0.0) and np.all(J == 0.0)
    assert gap_min == np.inf and gaps == []


# ============================================================================
# 2. Golden regression tests
# ============================================================================

def _compute_golden():
    """Evaluate every protected entry point on the fixed configurations.
    Returns a flat dict of arrays (the golden payload)."""
    out = {}

    # --- per-pair kernel on all configurations ---
    for name, nodes1, nodes2 in pair_configs():
        try:
            C1, R1, ax1, arc1, C2, R2, ax2, arc2 = _fit_pair(nodes1, nodes2)
        except (ValueError, np.linalg.LinAlgError):
            continue
        energy, force, hess, gap, s1, s2 = _compute_pair_force_and_hessian(
            nodes1, nodes2, arc1, arc2, R_BEAM, D_IMC, K_IMC,
            False, False, False, False)
        out[f'pair_{name}_scalars'] = np.array([energy, gap, s1, s2])
        out[f'pair_{name}_force'] = force
        out[f'pair_{name}_hess'] = hess

    # --- line-fallback kernel ---
    _, nodes1, nodes2 = pair_configs()[-1]        # near_straight
    energy, force, hess, gap, s1, s2 = _compute_pair_force_and_hessian_line(
        nodes1, nodes2, R_BEAM, D_IMC, K_IMC, False, False, False, False)
    out['line_scalars'] = np.array([energy, gap, s1, s2])
    out['line_force'] = force
    out['line_hess'] = hess

    # --- 18x18 full Hessian helper (off the hot path but public) ---
    _, nodes1, nodes2 = pair_configs()[0]
    C1, R1, ax1, arc1, C2, R2, ax2, arc2 = _fit_pair(nodes1, nodes2)
    out['hessian_nodes'] = compute_hessian_nodes(nodes1, nodes2, arc1, arc2)

    # --- beam-to-beam contact (arc path and line-fallback path) ---
    robot, q, u, n_a = two_beam_setup()
    F, J, gap_min, gaps = compute_eberly_3d_beam_to_beam_contact_forces(
        robot, q, R_BEAM, k_c=2.0, n_nodes_beam_A=n_a, d_imc=D_IMC,
        h_scale=True)
    out['b2b_F'] = F
    out['b2b_J'] = J
    out['b2b_scalars'] = np.array([gap_min, float(len(gaps))])
    out['b2b_gaps'] = np.array([[g['gap'], g['s_i1'], g['s_i2']]
                                for g in gaps])

    F2, J2, gap_min2, gaps2 = compute_eberly_3d_beam_to_beam_contact_forces(
        robot, q, R_BEAM, k_c=2.0, n_nodes_beam_A=n_a, d_imc=D_IMC,
        straight_R_threshold=0.05)
    out['b2b_line_F'] = F2
    out['b2b_line_J'] = J2
    out['b2b_line_scalars'] = np.array([gap_min2, float(len(gaps2))])

    # --- self contact ---
    robot_s, q_s, u_s = hairpin_setup()
    Fs, Js, gap_min_s, gaps_s = compute_eberly_3d_self_contact_forces(
        robot_s, q_s, R_BEAM, k_c=1.5, min_segment_separation=6, d_imc=D_IMC)
    out['self_F'] = Fs
    out['self_J'] = Js
    out['self_scalars'] = np.array([gap_min_s, float(len(gaps_s))])

    # --- friction: one-shot force paths ---
    out['b2b_fric_F'] = compute_eberly_3d_beam_to_beam_friction_forces(
        robot, q, u, R_BEAM, mu=0.3, vel_tol=1e-3, k_c=2.0,
        n_nodes_beam_A=n_a, d_imc=D_IMC)
    out['self_fric_F'] = compute_eberly_3d_self_friction_forces(
        robot_s, q_s, u_s, R_BEAM, mu=0.3, vel_tol=1e-3, k_c=1.5,
        min_segment_separation=6, d_imc=D_IMC)

    # --- friction: setup / apply (semi-implicit path) ---
    cache = compute_eberly_3d_beam_to_beam_friction_setup(
        robot, q, u, R_BEAM, mu=0.3, vel_tol=1e-3, k_c=2.0,
        n_nodes_beam_A=n_a, d_imc=D_IMC)
    Ff, Jf = compute_eberly_3d_beam_to_beam_friction_apply(
        robot, q, cache, R_BEAM, k_c=2.0, d_imc=D_IMC)
    out['b2b_fric_apply_F'] = Ff
    out['b2b_fric_apply_J'] = Jf

    # --- friction: self-contact setup / apply ---
    cache_s = compute_eberly_3d_self_friction_setup(
        robot_s, q_s, u_s, R_BEAM, mu=0.3, vel_tol=1e-3, k_c=1.5,
        min_segment_separation=6, d_imc=D_IMC)
    Ffs, Jfs = compute_eberly_3d_self_friction_apply(
        robot_s, q_s, cache_s, R_BEAM, k_c=1.5, d_imc=D_IMC)
    out['self_fric_apply_F'] = Ffs
    out['self_fric_apply_J'] = Jfs

    # --- friction: fixed-pair-list in-iter path (+ jacobian variant) ---
    pairs = compute_eberly_3d_beam_to_beam_friction_pairs(
        robot, q, R_BEAM, n_nodes_beam_A=n_a, d_imc=D_IMC)
    out['b2b_fric_pairs_n'] = np.array([float(len(pairs))])
    out['b2b_fric_force_F'] = compute_eberly_3d_beam_to_beam_friction_force(
        robot, q, u, pairs, R_BEAM, mu=0.3, vel_tol=1e-3, k_c=2.0,
        d_imc=D_IMC)
    Fwj, Jwj = compute_eberly_3d_beam_to_beam_friction_force_with_jac(
        robot, q, u, pairs, R_BEAM, mu=0.3, vel_tol=1e-3, dt=5e-3,
        k_c=2.0, d_imc=D_IMC)
    out['b2b_fric_wj_F'] = Fwj
    out['b2b_fric_wj_J'] = Jwj

    return out


@pytest.fixture(scope='module')
def golden():
    if not GOLDEN_PATH.exists():
        pytest.fail(
            f'Golden file missing: {GOLDEN_PATH}\n'
            'Generate it (from a KNOWN-GOOD state of the code) with:\n'
            '    python tests/test_eberly_contact.py --regen')
    with np.load(GOLDEN_PATH) as data:
        return {k: data[k] for k in data.files}


@pytest.fixture(scope='module')
def current():
    return _compute_golden()


def _keys():
    """Golden keys, resolved without evaluating the module fixtures."""
    if GOLDEN_PATH.exists():
        with np.load(GOLDEN_PATH) as data:
            return sorted(data.files)
    return ['<golden-missing>']


@pytest.mark.parametrize('key', _keys())
def test_golden(key, golden, current):
    assert key in current, f'entry point no longer produces output: {key}'
    ref = golden[key]
    got = current[key]
    assert got.shape == ref.shape, f'{key}: shape changed'
    scale = max(1.0, float(np.abs(ref).max()))
    np.testing.assert_allclose(
        got, ref, rtol=1e-7, atol=1e-9 * scale,
        err_msg=f'golden regression mismatch for {key}')


def test_golden_covers_all_current_outputs(golden, current):
    missing = sorted(set(current) - set(golden))
    assert not missing, (
        f'new outputs not covered by the golden file: {missing} '
        f'(regenerate with python tests/test_eberly_contact.py --regen)')


# ============================================================================
# Batched backend equivalence (scalar path is the oracle)
# ============================================================================

def _random_two_beam(seed):
    """Randomized crossing-beam variant of two_beam_setup."""
    rng = np.random.default_rng(seed)
    n_per = 9
    length = 0.08
    s = np.linspace(-length / 2, length / 2, n_per)
    amp_a = rng.uniform(0.5e-3, 2e-3)
    amp_b = rng.uniform(0.5e-3, 2e-3)
    pen = rng.uniform(-6e-5, 6e-5)          # gap at the crossing (arc level)
    beam_a = np.stack([s, np.zeros(n_per),
                       amp_a * np.cos(2 * np.pi * s / length)], axis=1)
    z_b = amp_a + 2 * R_BEAM - 1.5e-4 + pen + amp_b
    beam_b = np.stack([np.zeros(n_per), s,
                       z_b - amp_b * np.cos(2 * np.pi * s / length)], axis=1)
    nodes = np.vstack([beam_a, beam_b])
    nodes += rng.normal(scale=2e-5, size=nodes.shape)
    robot = DuckRobot(2 * n_per)
    return robot, nodes.ravel(), n_per


@pytest.mark.parametrize('seed', range(12))
def test_batched_matches_scalar_beam_to_beam(seed):
    robot, q, n_a = _random_two_beam(seed)
    out_s = compute_eberly_3d_beam_to_beam_contact_forces(
        robot, q, R_BEAM, k_c=1.7, n_nodes_beam_A=n_a, d_imc=D_IMC,
        backend='scalar')
    out_b = compute_eberly_3d_beam_to_beam_contact_forces(
        robot, q, R_BEAM, k_c=1.7, n_nodes_beam_A=n_a, d_imc=D_IMC,
        backend='batched')
    Fs, Js, gs, gaps_s = out_s
    Fb, Jb, gb, gaps_b = out_b
    scale_F = max(1.0, np.abs(Fs).max())
    scale_J = max(1.0, np.abs(Js).max())
    np.testing.assert_allclose(Fb, Fs, rtol=1e-9, atol=1e-12 * scale_F)
    np.testing.assert_allclose(Jb, Js, rtol=1e-9, atol=1e-12 * scale_J)
    assert len(gaps_b) == len(gaps_s)
    for a, b in zip(gaps_s, gaps_b):
        assert a['segment_pair'] == b['segment_pair']
        np.testing.assert_allclose(
            [b['gap'], b['s_i1'], b['s_i2']],
            [a['gap'], a['s_i1'], a['s_i2']], rtol=1e-9, atol=1e-12)
    if np.isfinite(gs):
        np.testing.assert_allclose(gb, gs, rtol=1e-9)


def test_batched_matches_scalar_self_contact():
    robot, q, u = hairpin_setup()
    out_s = compute_eberly_3d_self_contact_forces(
        robot, q, R_BEAM, k_c=1.5, min_segment_separation=6, d_imc=D_IMC,
        backend='scalar')
    out_b = compute_eberly_3d_self_contact_forces(
        robot, q, R_BEAM, k_c=1.5, min_segment_separation=6, d_imc=D_IMC,
        backend='batched')
    np.testing.assert_allclose(out_b[0], out_s[0], rtol=1e-9,
                               atol=1e-12 * max(1.0, np.abs(out_s[0]).max()))
    np.testing.assert_allclose(out_b[1], out_s[1], rtol=1e-9,
                               atol=1e-12 * max(1.0, np.abs(out_s[1]).max()))
    assert len(out_b[3]) == len(out_s[3])


@pytest.mark.parametrize('seed', range(6))
@pytest.mark.parametrize('law', ['smooth_coulomb', 'norton_hoff',
                                 'norton_hoff_capped'])
def test_batched_friction_matches_scalar(seed, law):
    robot, q, n_a = _random_two_beam(seed)
    rng = np.random.default_rng(1000 + seed)
    u = rng.normal(scale=1e-2, size=q.shape)
    kw = dict(mu=0.3, vel_tol=1e-3, k_c=1.7, n_nodes_beam_A=n_a,
              d_imc=D_IMC, friction_law=law)
    if law == 'norton_hoff_capped':
        # v_ref far below the test velocities (~1e-2) so the cap G = 1
        # actually engages; the default v_ref = 1.0 would never reach it.
        kw['nh_v_ref'] = 1e-3
    F_s = compute_eberly_3d_beam_to_beam_friction_forces(
        robot, q, u, R_BEAM, backend='scalar', **kw)
    F_b = compute_eberly_3d_beam_to_beam_friction_forces(
        robot, q, u, R_BEAM, backend='batched', **kw)
    scale = max(1.0, np.abs(F_s).max())
    np.testing.assert_allclose(F_b, F_s, rtol=1e-9, atol=1e-12 * scale)


def test_batched_narrow_phase_matches_arc_min_dist():
    """Direct kernel-level check incl. the parallel-plane fallback flag."""
    from dismech.contact.eberly_batched import batched_arc_min_dist

    cfgs = pair_configs()
    nodes1 = []
    nodes2 = []
    keep = []
    for name, na, nb in cfgs:
        try:
            _fit_pair(na, nb)
        except (ValueError, np.linalg.LinAlgError):
            continue
        nodes1.append(np.stack(na))
        nodes2.append(np.stack(nb))
        keep.append(name)
    res = batched_arc_min_dist(np.stack(nodes1), np.stack(nodes2))

    for k, name in enumerate(keep):
        na = tuple(nodes1[k])
        nb = tuple(nodes2[k])
        C1, R1, a1, arc1, C2, R2, a2, arc2 = _fit_pair(na, nb)
        ref = arc_min_dist(C1, R1, a1, arc1, C2, R2, a2, arc2)
        if res['fallback'][k]:
            # Special-case branches must be flagged, not silently wrong.
            # Symmetric test geometries legitimately land here (parallel
            # planes; P7(c) = 0 from exact symmetry) — but the generic
            # crossing configs must stay on the batched fast path.
            assert not name.startswith('crossing'), \
                f'{name}: generic config unexpectedly fell back'
            continue
        np.testing.assert_allclose(res['dist'][k], ref['dist'], rtol=1e-9,
                                   err_msg=f'{name}: batched dist')
        np.testing.assert_allclose(res['M1'][k], ref['M1'], rtol=0, atol=1e-9,
                                   err_msg=f'{name}: batched M1')
        np.testing.assert_allclose(res['M2'][k], ref['M2'], rtol=0, atol=1e-9,
                                   err_msg=f'{name}: batched M2')


# ============================================================================
# Multiplicative (tensor-product) blend: partition of unity, smooth
# active-set switching, and gradient consistency
# ============================================================================
# The pair energy is E_ij = w(s_i*) w(s_j*) U(delta_c).  Summed over the
# active pairs of one crossing, the weights must total ~1 regardless of where
# the contact sits relative to the meshes (the additive blend w_i U + w_j U
# would total n_i + n_j instead), and a pair must switch off exactly when
# either closest point reaches an arc end.

from dismech.contact.eberly_contact_3d import (             # noqa: E402
    compute_weights_from_s,
    pair_event_log,
)
from dismech.contact.circle_four_nodes_imc_weighted import computeE_IMC  # noqa: E402

_SWEEP_N = 21          # nodes per rod
_SWEEP_L = 10.0        # rod length; edge h = 0.5
_SWEEP_R = 0.2         # beam radius
_SWEEP_DIMC = 0.0008
_SWEEP_GAP = -0.001    # constant surface penetration (quadratic IMC zone)
_SWEEP_RC = 100.0      # in-plane curvature radius (arcs need non-collinear
                       # triplets; both rods bow inside their own plane)


def _crossing_q(x_c):
    """Rod A along x bowing in +y at z=0; rod B along y bowing in +x at
    z = 2*R + gap.  The rods live in parallel planes, so the centerline
    min distance is constant while the crossing slides with ``x_c``."""
    t = np.linspace(-_SWEEP_L / 2, _SWEEP_L / 2, _SWEEP_N)
    bow = _SWEEP_RC - np.sqrt(_SWEEP_RC**2 - t**2)
    z0 = 2 * _SWEEP_R + _SWEEP_GAP
    beam_a = np.stack([t, bow, np.zeros(_SWEEP_N)], axis=1)
    y_cross = _SWEEP_RC - np.sqrt(_SWEEP_RC**2 - x_c**2)
    beam_b = np.stack(
        [x_c + bow - (_SWEEP_RC - np.sqrt(_SWEEP_RC**2 - y_cross**2)),
         t + y_cross, np.full(_SWEEP_N, z0)], axis=1)
    return np.concatenate([beam_a, beam_b]).ravel()


def _crossing_eval(q):
    robot = DuckRobot(2 * _SWEEP_N)
    F, _, _, all_gaps = compute_eberly_3d_beam_to_beam_contact_forces(
        robot, q, _SWEEP_R, k_c=1.0, n_nodes_beam_A=_SWEEP_N,
        d_imc=_SWEEP_DIMC)
    k_imc = 15.0 / _SWEEP_DIMC
    energy, sum_mul, sum_add = 0.0, 0.0, 0.0
    for g in all_gaps:
        w1 = compute_weights_from_s(g['s_i1'])
        w2 = compute_weights_from_s(g['s_i2'])
        u = float(computeE_IMC(g['gap'] + _SWEEP_R, _SWEEP_R,
                               _SWEEP_DIMC, k_imc))
        energy += w1 * w2 * u
        sum_mul += w1 * w2
        sum_add += w1 + w2
    return F, energy, sum_mul, sum_add


def test_blend_partition_of_unity_and_smooth_force():
    """Slide the crossing over 3 edge lengths: sum(w_i*w_j) stays ~1 (the
    residual is the ~4e-3 partition-of-unity error of w itself), the additive
    sum oscillates around n_i + n_j, and the normal force has no steps at
    active-set changes."""
    h = _SWEEP_L / (_SWEEP_N - 1)
    xcs = np.linspace(-1.5 * h, 1.5 * h, 121)
    fz, mul, add = [], [], []
    for x_c in xcs:
        F, _, sum_mul, sum_add = _crossing_eval(_crossing_q(x_c))
        fz.append(F.reshape(-1, 3)[_SWEEP_N:, 2].sum())
        mul.append(sum_mul)
        add.append(sum_add)
    fz, mul, add = np.array(fz), np.array(mul), np.array(add)

    assert mul.min() > 0.99 and mul.max() < 1.01, \
        f'sum(w1*w2) not ~1: [{mul.min():.4f}, {mul.max():.4f}]'
    assert add.min() > 3.5, \
        'sum(w1+w2) should show the additive over-count (~n_i + n_j)'
    # force free of steps: adjacent-sample jump well below the smooth
    # sample-to-sample variation budget (0.5% of peak at this resolution)
    step = np.abs(np.diff(fz)).max() / np.abs(fz).max()
    assert step < 5e-3, f'force step {step:.2%} of peak'


def test_force_matches_fd_gradient_near_arc_end():
    """Central-difference check of the assembled force against the pair
    energy sum(w1*w2*U), in a configuration where one arc sits at s* ~ -1
    (largest weight-derivative terms).  The production force drops the
    w'(s) ds*/dq transport terms, whose relative size is O(penetration/h);
    at this penetration they must stay below 0.5%."""
    q0 = _crossing_q(0.67)          # s1 ~ {+0.34, -0.66}, s2 ~ {0, -1}
    F0, _, _, _ = _crossing_eval(q0)
    dofs = np.nonzero(np.abs(F0) > 0)[0]
    assert dofs.size, 'no active contact in FD configuration'
    eps = 1e-7
    g_fd = np.zeros(dofs.size)
    for m, d in enumerate(dofs):
        qp, qm = q0.copy(), q0.copy()
        qp[d] += eps
        qm[d] -= eps
        g_fd[m] = (_crossing_eval(qp)[1] - _crossing_eval(qm)[1]) / (2 * eps)
    rel = np.linalg.norm(F0[dofs] - g_fd) / np.linalg.norm(g_fd)
    assert rel < 5e-3, f'force vs FD gradient: rel error {rel:.2e}'


def test_pair_event_log_smooth_switch_off():
    """Every pair that enters or leaves the active set while the contact
    slides must do so with w(s_i*) * w(s_j*) ~ 0 (multiplicative switch-off).
    Events at step 0 are the initial engagement, not sliding transitions."""
    pair_event_log.reset()
    pair_event_log.enable()
    try:
        for step, x_c in enumerate(np.linspace(-1.0, 1.0, 161)):
            pair_event_log.mark_step(step)
            robot = DuckRobot(2 * _SWEEP_N)
            compute_eberly_3d_beam_to_beam_contact_forces(
                robot, _crossing_q(x_c), _SWEEP_R, k_c=1.0,
                n_nodes_beam_A=_SWEEP_N, d_imc=_SWEEP_DIMC)
    finally:
        pair_event_log.disable()

    sliding = [ev for ev in pair_event_log.events if ev['step'] > 0]
    assert sliding, 'sweep produced no active-set transitions'
    for ev in sliding:
        w = (compute_weights_from_s(ev['s_i1'])
             * compute_weights_from_s(ev['s_i2']))
        assert w < 5e-3, \
            f"{ev['event']} {ev['pair']} at step {ev['step']}: w1*w2={w:.2e}"
    pair_event_log.reset()


# ============================================================================
# Regeneration entry point
# ============================================================================

if __name__ == '__main__':
    if '--regen' in sys.argv:
        payload = _compute_golden()
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(GOLDEN_PATH, **payload)
        n_active = int(payload['b2b_scalars'][1])
        print(f'Wrote {GOLDEN_PATH} ({len(payload)} arrays).')
        print(f'  beam-to-beam active pairs : {n_active}')
        print(f'  beam-to-beam gap_min      : {payload["b2b_scalars"][0]:.3e}')
        print(f'  self-contact active pairs : {int(payload["self_scalars"][1])}')
        print(f'  friction pair-list length : {int(payload["b2b_fric_pairs_n"][0])}')
        if n_active == 0:
            print('WARNING: no active beam-to-beam pairs -- golden is weak!')
    else:
        print('Run with --regen to regenerate the golden file, '
              'or use pytest to run the tests.')
