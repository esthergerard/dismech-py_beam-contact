"""
3D regularized beam contact based on Eberly's exact circle-circle distance.

Contact model
-------------
Each overlapping 3-node window (n0, n1, n2) of a discrete rod is replaced by
a circular-arc proxy:

* the circle is tangent to edge (n0,n1) at its midpoint m_a and to edge
  (n1,n2) at its midpoint m_b (``circle_from_three_nodes``).  NOTE: the
  circle passes through the edge MIDPOINTS, not through the nodes -- the
  middle node lies slightly outside the circle;
* the arc runs from m_a to m_b (``arc_interval_from_nodes``) and is widened
  by 50% of its span on each side (``extend_arc``) so neighbouring windows
  overlap and the response stays smooth as the contact point travels.

For every candidate pair of arcs the exact minimum distance d* is found with
Eberly's method: in a canonical frame the stationarity conditions reduce to
the real roots in [-1, 1] of a degree-8 polynomial phi(cos theta)
(``build_phi`` / ``pairs_from_phi``), plus closed-form treatment of the
parallel-plane and point-circle special cases.  Arc-interval constraints are
handled by enumerating interior stationary points, edge candidates (one
angle pinned at an arc endpoint) and the four corner candidates, and taking
the global minimum (``arc_min_dist``).

The surface gap ``gap = d* - 2*beam_radius`` then drives the same smoothed
IMC penalty energy used by the 2D circle contact (imported from
``circle_four_nodes_imc_weighted``), weighted by a partition-of-unity bump
in the arc coordinate so overlapping windows share the load.  Forces are the
exact envelope-theorem gradient of d* w.r.t. the 18 node coordinates; the
assembled tangent keeps only the positive-semidefinite Gauss-Newton term
(matching the 2D implementation) for Newton stability.

Public entry points
-------------------
Contact:   ``compute_eberly_3d_self_contact_forces``
           ``compute_eberly_3d_beam_to_beam_contact_forces``
Friction:  three interchangeable schemes, selected by the time stepper:
           1. one-shot explicit  (``*_friction_forces``)
           2. semi-implicit frozen-direction setup/apply
              (``*_friction_setup`` / ``*_friction_apply``)
           3. IMC-convention fixed-pair-list recompute
              (``*_friction_pairs`` / ``*_friction_force`` /
              ``*_friction_force_with_jac``) -- the recommended scheme.

Sign conventions: contact forces are returned in the physical convention
(the caller ADDS them to the residual); friction follows IMCFrictionEnergy
(the caller SUBTRACTS it).  See the individual docstrings.

References
----------
D. Eberly, "Distance between two circles in 3D" (Geometric Tools, 2024/2025)
and "Distance from a point to a circle in 3D"; derivation notebook:
regularized_contact_Esther/arc_distance_eberly.ipynb (self-contained copy of
the geometry functions, including derivations dropped from this module).
Regression baseline: tests/test_eberly_contact.py (golden file
tests/resources/eberly_golden.npz).
"""

import math
import os

import numpy as np
from numpy.polynomial.polynomial import polyval, polyroots
from typing import Tuple, List, Dict, TYPE_CHECKING

from .. import profiling as _prof

if TYPE_CHECKING:
    from ..soft_robot import SoftRobot

from .circle_four_nodes_imc_weighted import (
    computeE_IMC,
    compute_dE_IMC_ddelta_c,
    compute_d2E_IMC_ddelta_c,
    compute_weights_from_s,
)


# =============================================================================
# Optional diagnostic: active-set entry/exit event log
# =============================================================================

class PairEventLog:
    """Optional log of (s_i1, s_i2, delta_c) for every segment pair at the
    driver call where it enters or leaves the active set.

    The distribution of w(s*) at these instants is the direct measurement of
    whether the multiplicative blend switches pairs off smoothly: a pair
    should only appear in / vanish from the active set with w1*w2 ~ 0, i.e.
    with at least one of |s_i1|, |s_i2| at 1.

    Disabled by default (one boolean check per driver call).  Enable either
    programmatically::

        from dismech.contact.eberly_contact_3d import pair_event_log
        pair_event_log.enable()
        ... run ...
        pair_event_log.save('pair_events.jsonl')   # or inspect .events

    or by setting ``DISMECH_PAIR_EVENT_LOG=<path>`` in the environment before
    importing dismech; the log is then written to <path> at interpreter exit.

    The contact drivers run once per Newton iteration, so each event carries
    the per-context driver-call counter ``call``; intra-time-step chatter can
    be filtered downstream.  The outer loop may call ``mark_step()`` once per
    accepted time step so events are also stamped with a step index (-1 when
    never called).  ``context`` distinguishes independent driver call sites
    (self-contact vs. beam-to-beam, and different beam splits).
    """

    def __init__(self):
        self.enabled = False
        self.events = []
        self.step = -1
        self._active = {}   # context -> {pair_tag: (s_i1, s_i2, delta_c)}
        self._calls = {}    # context -> number of driver calls seen
        self._path = None

    def enable(self, path=None):
        self.enabled = True
        if path is not None:
            self._path = path

    def disable(self):
        self.enabled = False

    def reset(self):
        self.events = []
        self.step = -1
        self._active = {}
        self._calls = {}

    def mark_step(self, step=None):
        self.step = (self.step + 1) if step is None else step

    def record(self, context, all_gaps, beam_radius):
        """Diff one driver call's active set against the previous call for
        the same context; append an event per pair entering (with its current
        s/delta_c) or leaving (with its last-active s/delta_c)."""
        call = self._calls.get(context, 0)
        self._calls[context] = call + 1
        prev = self._active.get(context, {})
        # delta_c = d* - beam_radius = gap + beam_radius (driver convention)
        cur = {g['segment_pair']: (g['s_i1'], g['s_i2'], g['gap'] + beam_radius)
               for g in all_gaps}
        for tag, (s1, s2, dc) in cur.items():
            if tag not in prev:
                self.events.append({
                    'event': 'enter', 'context': context, 'pair': tag,
                    's_i1': float(s1), 's_i2': float(s2),
                    'delta_c': float(dc), 'call': call, 'step': self.step})
        for tag, (s1, s2, dc) in prev.items():
            if tag not in cur:
                self.events.append({
                    'event': 'leave', 'context': context, 'pair': tag,
                    's_i1': float(s1), 's_i2': float(s2),
                    'delta_c': float(dc), 'call': call, 'step': self.step})
        self._active[context] = cur

    def save(self, path=None):
        import json
        path = path or self._path
        if path is None or not self.events:
            return
        with open(path, 'w') as f:
            for ev in self.events:
                f.write(json.dumps(ev) + '\n')


pair_event_log = PairEventLog()

if os.environ.get('DISMECH_PAIR_EVENT_LOG'):
    pair_event_log.enable(os.environ['DISMECH_PAIR_EVENT_LOG'])
    import atexit as _atexit
    _atexit.register(pair_event_log.save)


# =============================================================================
# Fast scalar 3-vector primitives
# =============================================================================
# The per-pair kernel is dominated by Python/NumPy dispatch overhead on
# length-3 arrays (np.cross alone costs ~20 us/call via moveaxis).  These
# inlined equivalents are ~20-50x faster and keep the hot path readable.

def _cross3(a, b):
    """np.cross for two 3-vectors."""
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


def _norm3(v):
    """Euclidean norm of a 3-vector (returns a Python float)."""
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _dot3(a, b):
    """Dot product of two 3-vectors (returns a Python float)."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _solve3(A, b):
    """Closed-form solve of a 3x3 system (Cramer / adjugate).

    LAPACK-free on purpose: inside a simulation the BLAS thread pool is kept
    warm by the large Newton solves, and every tiny per-pair
    ``np.linalg.solve``/``inv`` call then pays a thread-synchronisation cost
    of O(100 us) -- orders of magnitude above the arithmetic.  The geometry
    matrices here are well-conditioned by construction (the collinear case
    is rejected before the solve), so the unpivoted closed form is safe.
    """
    a, b_, c = A[0]
    d, e, f = A[1]
    g, h, i = A[2]
    co_a = e * i - f * h
    co_b = f * g - d * i
    co_c = d * h - e * g
    det = a * co_a + b_ * co_b + c * co_c
    x0, x1, x2 = b
    return np.array([
        (x0 * co_a + x1 * (c * h - b_ * i) + x2 * (b_ * f - c * e)) / det,
        (x0 * co_b + x1 * (a * i - c * g) + x2 * (c * d - a * f)) / det,
        (x0 * co_c + x1 * (b_ * g - a * h) + x2 * (a * e - b_ * d)) / det,
    ])


def _inv3(A):
    """Closed-form inverse of a 3x3 matrix (see ``_solve3`` for rationale)."""
    a, b, c = A[0]
    d, e, f = A[1]
    g, h, i = A[2]
    co_a = e * i - f * h
    co_b = f * g - d * i
    co_c = d * h - e * g
    det = a * co_a + b * co_b + c * co_c
    return np.array([
        [co_a / det, (c * h - b * i) / det, (b * f - c * e) / det],
        [co_b / det, (a * i - c * g) / det, (c * d - a * f) / det],
        [co_c / det, (b * g - a * h) / det, (a * e - b * d) / det],
    ])


def _pmul(a, b):
    """Product of two 1-D polynomial coefficient arrays (low-order first).
    Same contract as numpy.polynomial.polynomial.polymul without the
    as_series validation overhead."""
    return np.convolve(a, b)


def _padd(a, b):
    """Sum of two coefficient arrays of possibly different lengths."""
    if len(a) < len(b):
        a, b = b, a
    out = a.copy()
    out[:len(b)] += b
    return out


def _psub(a, b):
    """Difference of two coefficient arrays of possibly different lengths."""
    n = max(len(a), len(b))
    out = np.zeros(n)
    out[:len(a)] = a
    out[:len(b)] -= b
    return out


def _broad_phase_pairs(node_pos, starts_A, starts_B, interaction_range):
    """Vectorized bounding-sphere culling.

    ``starts_A`` / ``starts_B`` are the first-node indices of the candidate
    3-node windows on each side.  A window starting at ``s`` spans nodes
    (s, s+1, s+2); its bounding sphere is centred at the midpoint of nodes
    s and s+2 with radius half their distance.  Returns the index pairs
    (into starts_A / starts_B) whose spheres, inflated by
    ``interaction_range``, overlap — identical to the scalar test
    ``|c_j - c_i| > range + r_i + r_j  -> skip``.
    """
    pA = node_pos[starts_A]
    pA2 = node_pos[starts_A + 2]
    pB = node_pos[starts_B]
    pB2 = node_pos[starts_B + 2]
    cA = 0.5 * (pA + pA2)
    cB = 0.5 * (pB + pB2)
    rA = 0.5 * np.sqrt(((pA - pA2) ** 2).sum(axis=1))
    rB = 0.5 * np.sqrt(((pB - pB2) ** 2).sum(axis=1))
    diff = cA[:, None, :] - cB[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    keep = dist <= interaction_range + rA[:, None] + rB[None, :]
    return np.nonzero(keep)


# =============================================================================
# Circle construction from three nodes
# =============================================================================

def circle_from_three_nodes(n0, n1, n2):
    """
    Build a circle tangent to edge (n0,n1) at its midpoint m_a
    and tangent to edge (n1,n2) at its midpoint m_b.

    Tangency at the midpoint means the edge direction is perpendicular
    to the radius there, i.e. (C - m_a) orthogonal to t1 and (C - m_b) orthogonal to t2.
    The center also lies in the plane spanned by the two edges.
    """
    n0, n1, n2 = (np.asarray(p, float) for p in (n0, n1, n2))

    e_a = n1 - n0
    e_b = n2 - n1
    cross = _cross3(e_a, e_b)
    cross_n2 = _dot3(cross, cross)
    if cross_n2 < 1e-24:
        raise ValueError("Collinear edges; circle undefined.")

    n_axis = cross / math.sqrt(cross_n2)      # plane normal

    m_a = 0.5 * (n0 + n1)                     # tangency point on edge a
    m_b = 0.5 * (n1 + n2)                     # tangency point on edge b

    # Solve  [ e_a^T ; e_b^T ; n_axis^T ] C  =  [ e_a·m_a ; e_b·m_b ; n_axis·n1 ]
    A = np.vstack([e_a, e_b, n_axis])
    rhs = np.array([_dot3(e_a, m_a), _dot3(e_b, m_b), _dot3(n_axis, n1)])
    C = _solve3(A, rhs)

    R = _norm3(C - m_a)                       # = ||C - m_b|| by construction

    # Orient normal with the BendEnergy kb direction
    te = e_a / _norm3(e_a)
    tf = e_b / _norm3(e_b)
    kb = 2.0 * _cross3(te, tf) / (1.0 + _dot3(te, tf))
    if _dot3(n_axis, kb) < 0.0:
        n_axis = -n_axis

    return C, R, n_axis


def _triplet_fit_radii(node_pos):
    """Approximate fitted-circle radius of every 3-node window, vectorized.

    For the tangent-at-midpoint construction of ``circle_from_three_nodes``
    with near-equal edge lengths, R = (l_mean/2) / tan(theta/2) where theta
    is the turning angle between the two edges.  Degenerate (collinear)
    windows get R = inf, so a caller-side ``R > threshold`` test routes them
    to the straight-line primitive instead of dropping the pair.

    Returns an array of length ``n_nodes - 2``; entry ``i`` is the radius of
    the window over nodes (i, i+1, i+2).
    """
    seg = node_pos[1:] - node_pos[:-1]
    slen = np.linalg.norm(seg, axis=1)
    e_a, e_b = seg[:-1], seg[1:]
    cross = np.cross(e_a, e_b)
    sin_ab = np.linalg.norm(cross, axis=1)          # |e_a||e_b| sin(theta)
    cos_ab = np.einsum('ij,ij->i', e_a, e_b)        # |e_a||e_b| cos(theta)
    ab = slen[:-1] * slen[1:]
    l_mean = 0.5 * (slen[:-1] + slen[1:])
    with np.errstate(divide='ignore', invalid='ignore'):
        tan_half = sin_ab / (ab + cos_ab)           # tan(theta/2)
        R = 0.5 * l_mean / tan_half
    R[~np.isfinite(R)] = np.inf
    return R


def _pair_line_weights(ii, jj, node_pos, straight_R_threshold):
    """Smoothstep line-primitive weight per pair, or None when disabled.

    0 → pure arc kernel, 1 → pure line primitive, in between → blend.
    The zone spans [0.8, 1.25]·threshold in the pair's max fitted window
    radius, so a window drifting across the threshold changes the residual
    smoothly instead of jumping between primitives (which breaks Newton
    when many active windows hover near the threshold).
    """
    if straight_R_threshold is None or len(ii) == 0:
        return None
    R_trip = _triplet_fit_radii(node_pos)
    Rmax = np.maximum(R_trip[np.asarray(ii)], R_trip[np.asarray(jj)])
    R_lo = 0.8 * straight_R_threshold
    R_hi = 1.25 * straight_R_threshold
    x = np.clip((Rmax - R_lo) / (R_hi - R_lo), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# =============================================================================
# Frame and angle utilities
# =============================================================================

def rotation_to_z(n):
    """Rotation matrix R such that R n = z_normalized = (0,0,1)."""
    n = n / _norm3(n)
    # Same aligned / anti-aligned detection as np.allclose(n, +-z) with the
    # default tolerances (atol = 1e-8, rtol = 1e-5), without its overhead.
    if abs(n[0]) <= 1e-8 and abs(n[1]) <= 1e-8:
        if abs(n[2] - 1.0) <= 1e-8 + 1e-5:
            return np.eye(3)
        if abs(n[2] + 1.0) <= 1e-8 + 1e-5:
            return np.diag([1., -1., -1.])
    # v = n x z, s = |v|, c = n.z  (Rodrigues rotation onto z)
    v = np.array([n[1], -n[0], 0.0])
    s2 = v[0] * v[0] + v[1] * v[1]
    c = float(n[2])
    Vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + Vx + Vx @ Vx * (1 - c) / s2


def build_plane_frame(n):
    """Orthonormal basis (e1, e2) for the plane with unit normal n."""
    n = n / _norm3(n)
    t = np.array([1., 0., 0.]) if abs(n[0]) < 0.9 else np.array([0., 1., 0.])
    e1 = _cross3(n, t);  e1 /= _norm3(e1)
    e2 = _cross3(n, e1); e2 /= _norm3(e2)
    return e1, e2


def recover_angle(M, center, e1, e2):
    """Recover theta such that M = center + R*(cos(theta) e1 + sin(theta) e2)."""
    v = M - center
    return math.atan2(_dot3(v, e2), _dot3(v, e1))


def angle_in_arc(theta, arc, tol=1e-8):
    """True if theta (mod 2pi) lies in the arc from arc[0] to arc[1]."""
    alpha, beta = float(arc[0]), float(arc[1])
    span = (beta - alpha) % (2*np.pi)            # always in [0, 2π)
    if span < tol or span > 2*np.pi - tol:        # degenerate or full circle
        return True
    return ((theta - alpha) % (2*np.pi)) <= span + tol


def make_circle_funcs(C, R, n):
    """Return M(t), M'(t), M''(t) as callables in the original frame."""
    e1, e2 = build_plane_frame(n)
    M   = lambda t: C + R * (np.cos(t)*e1 + np.sin(t)*e2)
    dM  = lambda t: R * (-np.sin(t)*e1 + np.cos(t)*e2)
    d2M = lambda t: R * (-np.cos(t)*e1 - np.sin(t)*e2)
    return M, dM, d2M


# =============================================================================
# Eberly canonical transform and polynomial solver
# =============================================================================

def eberly_prepare(Ca, Ra, na, Cb, Rb, nb):
    """
    Eberly canonical transform matching the C++ PrepareCircles().
    Returns (C0_can, N0_can, r0_can, R_total, t, scale, swapped).
    Forward: Q = scale * R_total @ (P + t)
    Inverse: P = R_total.T @ Q / scale - t
    """
    swapped = (Ra > Rb)
    if swapped:
        C0, R0, n0 = Cb.copy(), Rb, nb.copy()
        Cl, Rl, nl = Ca.copy(), Ra, na.copy()
    else:
        C0, R0, n0 = Ca.copy(), Ra, na.copy()
        Cl, Rl, nl = Cb.copy(), Rb, nb.copy()

    if n0[2] < 0: n0 = -n0
    if nl[2] < 0: nl = -nl

    t = -Cl;  scale = 1.0 / Rl
    R_rot   = rotation_to_z(nl)
    C0_can  = scale * (R_rot @ (C0 + t))
    N0_can  = R_rot @ n0
    r0_can  = R0 * scale
    R_total = R_rot.copy()

    if abs(C0_can[0]) > 1e-14:
        L  = np.sqrt(C0_can[0]**2 + C0_can[1]**2)
        sr = C0_can[0] / L;  cr = C0_can[1] / L
        rot1 = np.array([[cr, -sr, 0.], [sr, cr, 0.], [0., 0., 1.]])
        C0_can    = rot1 @ C0_can;  C0_can[0] = 0.
        N0_can    = rot1 @ N0_can
        R_total   = rot1 @ R_total

    return C0_can, N0_can, r0_can, R_total, t, scale, swapped


def build_phi(C0, N0, r0):
    """
    Build phi(c) = P6^2(c) - (1-c^2)P7^2(c) following Eberly (2025).
    Polynomials are stored as 1D numpy arrays: coeff[i] = coefficient of c^i.
    """
    D     = -C0
    N0xD  = _cross3(N0, D)
    N0xU1 = np.array([ 0.,   N0[2], -N0[1]])   # N0 × ê_x
    N0xV1 = np.array([-N0[2], 0.,    N0[0]])   # N0 × ê_y

    A0 = float(D[0])                            # = 0 (C0[0] = 0 by construction)
    A1 = float(D[1])
    A2 = _dot3(N0xD,  N0xD)
    A3 = _dot3(N0xD,  N0xU1)
    A4 = _dot3(N0xD,  N0xV1)
    A5 = _dot3(N0xU1, N0xU1)
    A6 = _dot3(N0xU1, N0xV1)
    A7 = _dot3(N0xV1, N0xV1)

    P0 = np.array([A2+A7,   2*A3,    A5-A7])
    P1 = np.array([2*A4,    2*A6          ])
    P2 = np.array([0.,      A1            ])
    P3 = np.array([-A0                    ])   # = [0]
    P4 = np.array([-A6,     A4,     2*A6  ])
    P5 = np.array([-A3,     A7-A5         ])
    T0 = np.array([ 1.,     0.,    -1.    ])   # 1 − c²

    T1 = _padd(_pmul(P2, P2), _pmul(T0, _pmul(P3, P3)))
    T2 = 2. * _pmul(P2, P3)                    # = 0 (P3 = [0])
    T3 = _padd(_pmul(P4, P4), _pmul(T0, _pmul(P5, P5)))
    T4 = 2. * _pmul(P4, P5)

    r0sq = r0**2
    P6 = _psub(_padd(_pmul(P0, T1), _pmul(_pmul(T0, P1), T2)), r0sq * T3)
    P7 = _psub(_padd(_pmul(P0, T2), _pmul(P1, T1)),            r0sq * T4)
    phi = _psub(_pmul(P6, P6), _pmul(T0, _pmul(P7, P7)))
    return phi, P6, P7


def pairs_from_phi(phi, P6, P7, tol_imag=1e-6):
    """
    Extract all (cos(theta), sin(theta)) pairs from the real roots of phi in [-1, 1].
    sin(theta) = -P6(c)/P7(c) when P7(c) /= 0, else ±sqrt(1-c**2).
    """
    # Trailing-zero trim (np.trim_zeros without the overhead).
    n = len(phi)
    while n > 0 and phi[n - 1] == 0.0:
        n -= 1
    if n <= 1:
        return []
    roots = polyroots(phi[:n])

    def _horner(cs, coeffs):
        # numpy.polynomial.polynomial.polyval evaluation order, inlined.
        acc = coeffs[-1]
        for k in range(len(coeffs) - 2, -1, -1):
            acc = acc * cs + coeffs[k]
        return acc

    seen  = set();  pairs = []
    for r in roots:
        if abs(r.imag) > tol_imag: continue
        cs = r.real
        if cs < -1.0: cs = -1.0
        elif cs > 1.0: cs = 1.0
        cs = float(cs)
        key = round(cs, 8)
        if key in seen: continue
        seen.add(key)
        p7v = _horner(cs, P7)
        if abs(p7v) > 1e-10:
            pairs.append((cs, float(-_horner(cs, P6) / p7v)))
        else:
            sn = math.sqrt(max(0., 1. - cs**2))
            pairs.append((cs, sn))
            if sn > 1e-10: pairs.append((cs, -sn))
    return pairs


# =============================================================================
# Full-circle distance: parallel planes and general case
# =============================================================================

def full_circle_parallel_planes(Ca, Ra, na, Cb, Rb, nb):
    """
    Full-circle distance when the two circle planes are parallel (or coplanar).
    Follows Eberly (2024) Sections 7.1 and 7.2.

    Returns a list of (sqrDist, Ma, Mb) candidates, where Ma is on circle A
    and Mb is on circle B.
    """
    D  = Cb - Ca
    N0dD = _dot3(na, D)
    normProj = N0dD * na              # height component (along normal)
    compProj = D - normProj           # in-plane component
    d = _norm3(compProj)

    r0, r1 = Ra, Rb
    results = []

    if d > 1e-12:
        U = compProj / d
    else:
        U = None  # concentric

    dmr1 = d - r1

    if dmr1 >= r0:  # separated or externally tangent
        Ma = Ca + r0 * U
        Mb = Cb - r1 * U
        sqrD = float(np.dot(Ma - Mb, Ma - Mb))
        results.append((sqrD, Ma, Mb))

    elif d + r1 <= r0:  # circle B inside circle A
        if U is not None:
            Ma = Ca + r0 * U
            Mb = Cb + r1 * U
        else:  # concentric
            e1, _ = build_plane_frame(na)
            Ma = Ca + r0 * e1
            Mb = Cb + r1 * e1
        sqrD = float(np.dot(Ma - Mb, Ma - Mb))
        results.append((sqrD, Ma, Mb))

    elif dmr1 <= -r0:  # circle A inside circle B
        if U is not None:
            Ma = Ca - r0 * U
            Mb = Cb - r1 * U
        else:  # concentric
            e1, _ = build_plane_frame(na)
            Ma = Ca + r0 * e1
            Mb = Cb + r1 * e1
        sqrD = float(np.dot(Ma - Mb, Ma - Mb))
        results.append((sqrD, Ma, Mb))

    else:  # overlapping circles
        r0sqr, r1sqr, dsqr = r0**2, r1**2, d**2
        s = (1.0 + (r0sqr - r1sqr) / dsqr) / 2.0
        h = np.sqrt(max(r0sqr - dsqr * s * s, 0.0))
        midpoint = Ca + s * compProj
        hNxU = h * np.cross(na, U)
        # Two intersection points (both achieve minimum distance = height)
        Ma1 = midpoint + hNxU
        Mb1 = Ma1 + normProj
        Ma2 = midpoint - hNxU
        Mb2 = Ma2 + normProj
        sqrD1 = float(np.dot(Ma1 - Mb1, Ma1 - Mb1))
        sqrD2 = float(np.dot(Ma2 - Mb2, Ma2 - Mb2))
        results.append((sqrD1, Ma1, Mb1))
        results.append((sqrD2, Ma2, Mb2))

    return results


def closest_on_circle0(cs, sn, C0, N0, r0):
    """
    Given angle (cs, sn) = (cos(theta), sin(theta)) on the unit circle (circle 1 in canonical
    frame), find the closest point on circle 0 and return (sqrDist, M0, M1).
    M1 = (cs, sn, 0) is on the unit circle.
    M0 is on circle 0 (center C0, normal N0, radius r0).
    """
    M1c   = np.array([cs, sn, 0.])
    delta = M1c - C0
    N0xd  = _cross3(N0, delta)
    lnd   = _norm3(N0xd)
    if lnd > 1e-12:
        N0d  = _dot3(N0, delta)
        diff = lnd - r0
        sqrD = N0d**2 + diff**2
        proj = delta - N0d * N0;  proj /= _norm3(proj)
        M0c  = C0 + r0 * proj
    else:  # equidistant / on-axis case
        U0 = (np.array([-N0[2], 0., N0[0]]) if abs(N0[0]) > abs(N0[1])
              else np.array([0., N0[2], -N0[1]]))
        U0 /= _norm3(U0)
        r0U0 = r0 * U0
        d0 = delta - r0U0
        sqrD = _dot3(d0, d0)
        M0c  = C0 + r0U0
    return sqrD, M0c, M1c


def point_circle_eberly(P, C, R, N):
    """
    Closest point on a 3D circle to a point P (Eberly, DistanceToCircle3.pdf).
    Returns (sqrDist, Q, equidistant) where Q is the closest point on the
    *full* circle.  Uses an unnormalized orthogonal basis {U, V, N} and the
    scaled projection  Dot(N,N)*Dot(U,U)*(Q-C),  avoiding divisions before
    the final normalization.  Handles the equidistant case (P on the axis).
    """
    P = np.asarray(P, float); C = np.asarray(C, float); N = np.asarray(N, float)
    PmC = P - C

    # ComputeOrthogonalBasis(1, N, U, V)
    if abs(N[0]) > abs(N[1]):
        U = np.array([-N[2], 0.0, N[0]])
    else:
        U = np.array([0.0, N[2], -N[1]])
    V = _cross3(N, U)

    NdotN = _dot3(N, N)
    scaledQmC = (NdotN * _dot3(U, PmC)) * U + _dot3(V, PmC) * V
    L = _norm3(scaledQmC)

    if L > 0.0:
        Q = C + R * (scaledQmC / L)
        sqrtNdotN = math.sqrt(NdotN)
        height = _dot3(N, PmC) / sqrtNdotN
        radial = _norm3(_cross3(N, PmC)) / sqrtNdotN - R
        sqrDist = height*height + radial*radial
        return sqrDist, Q, False
    else:
        # P on the circle axis — entire circle is equidistant.
        if abs(N[0]) > abs(N[1]):
            T = np.array([-N[2], 0.0, N[0]])
        else:
            T = np.array([0.0, N[2], -N[1]])
        T = T / _norm3(T)
        Q = C + R * T
        sqrDist = _dot3(PmC, PmC) + R*R
        return sqrDist, Q, True


# =============================================================================
# Active-set classification and arc-constrained minimum distance
# =============================================================================

def active_set_label(t1, t2, arc1, arc2, tol=1e-6):
    """Classify the active set at (t1, t2)."""
    def on_bdry(t, arc):
        return any(min(abs((t-ep) % (2*np.pi)), abs((t-ep) % (2*np.pi) - 2*np.pi)) < tol
                   for ep in arc)
    b1, b2 = on_bdry(t1, arc1), on_bdry(t2, arc2)
    if not b1 and not b2: return 'interior'
    if not b1 and b2:     return 'edge_t1_free'
    if b1 and not b2:     return 'edge_t2_free'
    return 'corner'


def eberly_all_candidates(Ca, Ra, na, Cb, Rb, nb):
    """
    Run the Eberly polynomial solver and return all stationary-point candidates
    as a list of (sqrDist_original_frame, M_a_orig, M_b_orig).
    """
    C0c, N0c, r0c, Rt, t, scale, swapped = eberly_prepare(Ca, Ra, na, Cb, Rb, nb)
    def inv(Q): return Rt.T @ Q / scale - t
    if N0c[2] >= 1. - 1e-10:
        return full_circle_parallel_planes(Ca, Ra, na, Cb, Rb, nb)
    phi, P6, P7 = build_phi(C0c, N0c, r0c)
    results = []
    for cs, sn in pairs_from_phi(phi, P6, P7):
        sqD_can, M0c, M1c = closest_on_circle0(cs, sn, C0c, N0c, r0c)
        Ma, Mb = inv(M0c), inv(M1c)    # in original frame
        if swapped: Ma, Mb = Mb, Ma    # restore: Ma on Ca, Mb on Cb
        results.append((sqD_can / scale**2, Ma, Mb))
    return results


def arc_min_dist(C1, R1, n1, arc1, C2, R2, n2, arc2, n_grid=25):
    """
    Arc-constrained minimum distance between two circles in 3D.

    Edge sub-problem (one theta pinned at an arc endpoint, the other free) uses
    Eberly's closed-form point-to-circle distance: the closest point on the
    *full* free circle is found analytically; if its angle lies inside the
    free arc it is the edge candidate, otherwise the constrained minimum
    sits at a corner (already enumerated below).
    """
    ef1, ef2 = build_plane_frame(n1)
    eg1, eg2 = build_plane_frame(n2)

    # Point-on-circle evaluators sharing the frames above (identical to
    # make_circle_funcs, which would rebuild the same plane frames).
    def M1f(t):
        return C1 + R1 * (math.cos(t) * ef1 + math.sin(t) * ef2)

    def M2f(t):
        return C2 + R2 * (math.cos(t) * eg1 + math.sin(t) * eg2)

    def fsq(t1, t2):
        d = M1f(t1) - M2f(t2)
        return _dot3(d, d)

    candidates = []  # (sqrDist, t1, t2)

    # 1) Interior: Eberly polynomial, keep those inside both arcs
    try:
        for sqD, Ma, Mb in eberly_all_candidates(C1, R1, n1, C2, R2, n2):
            t1 = recover_angle(Ma, C1, ef1, ef2)
            t2 = recover_angle(Mb, C2, eg1, eg2)
            if angle_in_arc(t1, arc1) and angle_in_arc(t2, arc2):
                candidates.append((sqD, t1, t2))
    except NotImplementedError:
        pass

    # 2) Edge candidates via Eberly point-to-circle
    for t1_fix in arc1:                              # θ1 pinned, θ2 free
        P = M1f(t1_fix)
        sqD, Q, equidist = point_circle_eberly(P, C2, R2, n2)
        if not equidist:
            t2 = recover_angle(Q, C2, eg1, eg2)
            if angle_in_arc(t2, arc2):
                candidates.append((sqD, t1_fix, t2))

    for t2_fix in arc2:                              # θ2 pinned, θ1 free
        P = M2f(t2_fix)
        sqD, Q, equidist = point_circle_eberly(P, C1, R1, n1)
        if not equidist:
            t1 = recover_angle(Q, C1, ef1, ef2)
            if angle_in_arc(t1, arc1):
                candidates.append((sqD, t1, t2_fix))

    # 3) Corners: all 4 endpoint pairs
    for t1 in arc1:
        for t2 in arc2:
            candidates.append((fsq(t1, t2), t1, t2))

    # Global minimum
    sqD_b, t1_b, t2_b = min(candidates, key=lambda x: x[0])
    kind = active_set_label(t1_b, t2_b, arc1, arc2)
    return {'dist':     np.sqrt(sqD_b),
            'sqrDist':  sqD_b,
            'M1':       M1f(t1_b),
            'M2':       M2f(t2_b),
            'theta1':   t1_b,
            'theta2':   t2_b,
            'kind':     kind}


# =============================================================================
# Gradient and Hessian of arc distance w.r.t. node coordinates
# =============================================================================

def dCircle_dnodes(n0, n1, n2):
    """
    Analytical Jacobians of (C, R, n_axis) w.r.t. the 9 stacked node
    coordinates [n0; n1; n2] for the midpoint-tangent circle construction.

    Returns
    -------
    dC_dN : (3, 9)
    dR_dN : (9,)
    dn_dN : (3, 9)
    """
    n0 = np.asarray(n0, float); n1v = np.asarray(n1, float); n2 = np.asarray(n2, float)
    I3 = np.eye(3); Z3 = np.zeros((3, 3))

    e_a = n1v - n0
    e_b = n2 - n1v
    cross = _cross3(e_a, e_b)
    nrm = _norm3(cross)
    n_axis = cross / nrm

    m_a = 0.5*(n0 + n1v)
    m_b = 0.5*(n1v + n2)

    A = np.vstack([e_a, e_b, n_axis])
    rhs = np.array([_dot3(e_a, m_a), _dot3(e_b, m_b), _dot3(n_axis, n1v)])
    # Closed-form inverse, reused below for the (3, 9) right-hand side.
    # Avoids LAPACK entirely: both the multi-RHS gesv path and, inside a
    # running simulation, ANY tiny LAPACK call are pathologically slow with
    # a warm OpenBLAS thread pool (O(100 us) per call).
    A_inv = _inv3(A)
    C = A_inv @ rhs

    # Building-block Jacobians (each 3×9)
    de_a = np.hstack([-I3,  I3,  Z3])
    de_b = np.hstack([ Z3, -I3,  I3])
    dm_a = np.hstack([0.5*I3, 0.5*I3, Z3])
    dm_b = np.hstack([Z3, 0.5*I3, 0.5*I3])
    dn1_ = np.hstack([Z3, I3, Z3])

    def skew(v):
        return np.array([[ 0.0, -v[2],  v[1]],
                         [ v[2],  0.0, -v[0]],
                         [-v[1],  v[0],  0.0]])

    # d(cross)/dN = -[e_b]× de_a + [e_a]× de_b
    dcross = -skew(e_b) @ de_a + skew(e_a) @ de_b           # (3, 9)

    # d(n_axis) = (I - n n^T)/||cross|| · d(cross)
    P = (I3 - np.outer(n_axis, n_axis)) / nrm
    dn_axis = P @ dcross                                     # (3, 9)

    # dA · C : rows are (de_a·C, de_b·C, dn_axis·C)
    dA_C = np.vstack([C @ de_a, C @ de_b, C @ dn_axis])     # (3, 9)

    # drhs
    drhs0 = m_a @ de_a + e_a @ dm_a
    drhs1 = m_b @ de_b + e_b @ dm_b
    drhs2 = n1v @ dn_axis + n_axis @ dn1_
    drhs  = np.vstack([drhs0, drhs1, drhs2])                # (3, 9)

    dC_dN = A_inv @ (drhs - dA_C)                           # (3, 9)

    diff = C - m_a
    R = _norm3(diff)
    dR_dN = (diff @ (dC_dN - dm_a)) / R                     # (9,)

    # Sign-flip n_axis to match the BendEnergy kb convention used in
    # circle_from_three_nodes.  Flipping n does NOT change C or R, but
    # the sign of dn_dN must be flipped to match the returned n.
    te = e_a / _norm3(e_a)
    tf = e_b / _norm3(e_b)
    kb = 2.0 * _cross3(te, tf) / (1.0 + _dot3(te, tf))
    if _dot3(n_axis, kb) < 0.0:
        dn_axis = -dn_axis

    return dC_dN, dR_dN, dn_axis


def compute_gradient_nodes(res, C1, R1, n1, NODES1, C2, R2, n2, NODES2):
    """
    Gradient of f* = (d*)² and of d* w.r.t. the 9 node coordinates of
    each circle (two 9-vectors).  Uses the chain rule:
        df/dN = df/dC · dC/dN + df/dR · dR/dN + df/dn · dn/dN
    """
    Delta = res['M1'] - res['M2']
    d     = res['dist']
    rho1  = res['M1'] - C1
    rho2  = res['M2'] - C2

    # ∂f/∂(C, R, n) — envelope theorem on M_i = C_i + ρ_i
    df_dC1 =  2 * Delta
    df_dR1 =  2 * _dot3(Delta, rho1) / R1
    df_dn1 = -2 * _dot3(Delta, n1) * rho1

    df_dC2 = -2 * Delta
    df_dR2 = -2 * _dot3(Delta, rho2) / R2
    df_dn2 =  2 * _dot3(Delta, n2) * rho2

    dC1_dN, dR1_dN, dn1_dN = dCircle_dnodes(*NODES1)
    dC2_dN, dR2_dN, dn2_dN = dCircle_dnodes(*NODES2)

    df_dN1 = df_dC1 @ dC1_dN + df_dR1 * dR1_dN + df_dn1 @ dn1_dN   # (9,)
    df_dN2 = df_dC2 @ dC2_dN + df_dR2 * dR2_dN + df_dn2 @ dn2_dN   # (9,)

    return {'df_dN1': df_dN1, 'df_dN2': df_dN2,
            'dd_dN1': df_dN1 / (2*d),
            'dd_dN2': df_dN2 / (2*d)}


def compute_hessian_p(res, C1, R1, n1, C2, R2, n2):
    """
    Analytical 14×14 Hessian of f* = (d*)^2 w.r.t. parameters
    p = (C1, R1, n1, C2, R2, n2).  Active-set aware via IFT on θ*.

    Index layout in p:
        0..2  = C1
        3     = R1
        4..6  = n1
        7..9  = C2
        10    = R2
        11..13= n2
    """
    M1, M2 = res['M1'], res['M2']
    Delta = M1 - M2
    rho1  = M1 - C1
    rho2  = M2 - C2
    t1, t2 = res['theta1'], res['theta2']
    kind   = res['kind']

    _, dM1f, d2M1f = make_circle_funcs(C1, R1, n1)
    _, dM2f, d2M2f = make_circle_funcs(C2, R2, n2)
    tau1  = dM1f(t1);  tau2  = dM2f(t2)
    d2M1  = d2M1f(t1); d2M2  = d2M2f(t2)

    P  = 14
    I3 = np.eye(3)

    # ── Explicit-at-fixed-θ derivatives of (C_i, R_i, n_i, M_i, ρ_i, τ_i) ──
    dC1_p = np.zeros((3, P)); dC1_p[:, 0:3]  = I3
    dC2_p = np.zeros((3, P)); dC2_p[:, 7:10] = I3
    dR1_p = np.zeros(P);      dR1_p[3]  = 1.0
    dR2_p = np.zeros(P);      dR2_p[10] = 1.0
    dn1_p = np.zeros((3, P)); dn1_p[:, 4:7]   = I3
    dn2_p = np.zeros((3, P)); dn2_p[:, 11:14] = I3

    # ∂M_i/∂p at fixed θ : =∂C_i + (∂R_i/R)ρ_i − n_i (ρ_i·∂n_i)
    dM1_expl = np.zeros((3, P))
    dM1_expl[:, 0:3] = I3                          # C1
    dM1_expl[:, 3]   = rho1 / R1                   # R1
    dM1_expl[:, 4:7] = -np.outer(n1, rho1)         # n1
    dM2_expl = np.zeros((3, P))
    dM2_expl[:, 7:10]  = I3
    dM2_expl[:, 10]    = rho2 / R2
    dM2_expl[:, 11:14] = -np.outer(n2, rho2)

    # τ_i transforms like an in-plane vector → same structure as ρ_i
    dtau1_p = np.zeros((3, P))
    dtau1_p[:, 3]   = tau1 / R1
    dtau1_p[:, 4:7] = -np.outer(n1, tau1)
    dtau2_p = np.zeros((3, P))
    dtau2_p[:, 10]    = tau2 / R2
    dtau2_p[:, 11:14] = -np.outer(n2, tau2)

    dDelta_expl = dM1_expl - dM2_expl              # (3, P)

    # ── ∂²f/∂θ_i ∂p (at fixed θ) ──
    #   ∂f/∂θ1 =  2 Δ·τ1   ⇒  ∂²f/∂θ1∂p =  2 (∂Δ·τ1 + Δ·∂τ1)
    #   ∂f/∂θ2 = -2 Δ·τ2   ⇒  ∂²f/∂θ2∂p = -2 (∂Δ·τ2 + Δ·∂τ2)
    d2f_dt1_dp =  2.0 * (tau1 @ dDelta_expl + Delta @ dtau1_p)
    d2f_dt2_dp = -2.0 * (tau2 @ dDelta_expl + Delta @ dtau2_p)

    # ── H_θ (2×2) ──
    H11 =  2*np.dot(tau1, tau1) + 2*np.dot(Delta, d2M1)
    H22 =  2*np.dot(tau2, tau2) - 2*np.dot(Delta, d2M2)
    H12 = -2*np.dot(tau1, tau2)
    H_theta = np.array([[H11, H12], [H12, H22]])

    # ── dθ*/dp via IFT (active-set aware) ──
    dG_dp = np.vstack([d2f_dt1_dp, d2f_dt2_dp])    # (2, P)
    dTheta_dp = np.zeros((2, P))
    if kind == 'interior':
        dTheta_dp = np.linalg.solve(H_theta, -dG_dp)
    elif kind == 'edge_t1_free':
        dTheta_dp[0] = -dG_dp[0] / H11
    elif kind == 'edge_t2_free':
        dTheta_dp[1] = -dG_dp[1] / H22
    # corner: zeros

    # ── Total dM_i*/dp = explicit + τ_i ⊗ dθ_i*/dp ──
    dM1_tot = dM1_expl + np.outer(tau1, dTheta_dp[0])
    dM2_tot = dM2_expl + np.outer(tau2, dTheta_dp[1])
    dDelta  = dM1_tot - dM2_tot
    drho1   = dM1_tot - dC1_p
    drho2   = dM2_tot - dC2_p

    # ── Assemble H_p row by row by differentiating g_p ──
    H_p = np.zeros((P, P))

    # g_p[C1] = 2Δ
    H_p[0:3, :] = 2.0 * dDelta
    # g_p[R1] = 2 Δ·ρ1 / R1
    H_p[3, :] = ((2.0/R1) * (rho1 @ dDelta + Delta @ drho1)
                 - (2.0 * np.dot(Delta, rho1) / R1**2) * dR1_p)
    # g_p[n1] = -2 (Δ·n1) ρ1
    coeff1 = (n1 @ dDelta) + (Delta @ dn1_p)       # (P,)
    H_p[4:7, :] = -2.0 * (np.outer(rho1, coeff1)
                          + np.dot(Delta, n1) * drho1)
    # g_p[C2] = -2Δ
    H_p[7:10, :] = -2.0 * dDelta
    # g_p[R2] = -2 Δ·ρ2 / R2
    H_p[10, :] = (-(2.0/R2) * (rho2 @ dDelta + Delta @ drho2)
                  + (2.0 * np.dot(Delta, rho2) / R2**2) * dR2_p)
    # g_p[n2] = +2 (Δ·n2) ρ2
    coeff2 = (n2 @ dDelta) + (Delta @ dn2_p)
    H_p[11:14, :] = 2.0 * (np.outer(rho2, coeff2)
                           + np.dot(Delta, n2) * drho2)

    H_p = 0.5 * (H_p + H_p.T)
    return H_p, dTheta_dp


def compute_hessian_nodes(NODES1, NODES2, arc1, arc2, eps_geom=1e-5):
    """
    18x18 Hessian of f* = (d*)^2 w.r.t. all node coordinates [N1; N2].
    Chain rule:
        H_N = J^T H_p J + sum_rho (g_p)_rho (∂² p_rho / ∂N²)
    where g_p (14,) and H_p (14x14) are analytical, J (14x18) is analytical
    via dCircle_dnodes, and ∂²p/∂N² is central-FD on the analytical J.
    """
    C1, R1, n1 = circle_from_three_nodes(*NODES1)
    C2, R2, n2 = circle_from_three_nodes(*NODES2)
    res = arc_min_dist(C1, R1, n1, arc1, C2, R2, n2, arc2)

    # ── g_p (14,) analytical ──
    Delta = res['M1'] - res['M2']
    rho1  = res['M1'] - C1
    rho2  = res['M2'] - C2
    g_p = np.zeros(14)
    g_p[0:3]   =  2*Delta
    g_p[3]     =  2*np.dot(Delta, rho1)/R1
    g_p[4:7]   = -2*np.dot(Delta, n1)*rho1
    g_p[7:10]  = -2*Delta
    g_p[10]    = -2*np.dot(Delta, rho2)/R2
    g_p[11:14] =  2*np.dot(Delta, n2)*rho2

    # ── H_p (14×14) analytical ──
    H_p, _ = compute_hessian_p(res, C1, R1, n1, C2, R2, n2)

    # ── J (14×18) analytical ──
    dC1, dR1_, dn1_ = dCircle_dnodes(*NODES1)
    dC2, dR2_, dn2_ = dCircle_dnodes(*NODES2)
    J = np.zeros((14, 18))
    J[0:3,  0:9 ] = dC1
    J[3,    0:9 ] = dR1_
    J[4:7,  0:9 ] = dn1_
    J[7:10, 9:18] = dC2
    J[10,   9:18] = dR2_
    J[11:14,9:18] = dn2_

    # ── ∂² p / ∂N² via central FD on the analytical Jacobian ──
    def J_circle(N_flat):
        n0, n1v, n2v = N_flat[:3], N_flat[3:6], N_flat[6:9]
        dC, dR, dn = dCircle_dnodes(n0, n1v, n2v)
        return np.vstack([dC, dR[None, :], dn])    # (7, 9)

    def d2p_circle(N):
        H7 = np.zeros((7, 9, 9))
        for k in range(9):
            ep = N.copy(); ep[k] += eps_geom
            em = N.copy(); em[k] -= eps_geom
            H7[:, :, k] = (J_circle(ep) - J_circle(em)) / (2*eps_geom)
        return 0.5 * (H7 + H7.transpose(0, 2, 1))

    N1 = np.concatenate(NODES1)
    N2 = np.concatenate(NODES2)
    H7_1 = d2p_circle(N1)
    H7_2 = d2p_circle(N2)

    d2p = np.zeros((14, 18, 18))
    d2p[0:7,  0:9,  0:9 ] = H7_1
    d2p[7:14, 9:18, 9:18] = H7_2

    # ── Combine ──
    H_N = J.T @ H_p @ J + np.einsum('a,aij->ij', g_p, d2p)
    H_N = 0.5 * (H_N + H_N.T)
    return H_N


# =============================================================================
# Arc interval and partition-of-unity helpers
# =============================================================================

def arc_interval_from_nodes(n0, n1, n2, C, R, n_axis):
    """
    Compute the arc interval [alpha, beta] for a 3-node segment on its
    fitted circle.  The arc goes from m_a = (n0+n1)/2 to m_b = (n1+n2)/2
    in the direction consistent with the edge tangents.
    """
    e1, e2 = build_plane_frame(n_axis)
    m_a = 0.5 * (n0 + n1)
    m_b = 0.5 * (n1 + n2)
    alpha = recover_angle(m_a, C, e1, e2)
    beta = recover_angle(m_b, C, e1, e2)
    # The circle tangent at m_a should be aligned with the edge direction
    e_a = n1 - n0
    tau_alpha = R * (-math.sin(alpha) * e1 + math.cos(alpha) * e2)
    if _dot3(tau_alpha, e_a) < 0:
        alpha, beta = beta, alpha
    return np.array([alpha, beta])


def extend_arc(arc, extension_factor=0.5):
    """Extend an arc interval by a fraction of its span on each side."""
    alpha, beta = float(arc[0]), float(arc[1])
    span = (beta - alpha) % (2 * np.pi)
    ext = extension_factor * span
    return np.array([alpha - ext, beta + ext])


def compute_s_from_theta(theta, arc):
    """Compute normalized arc coordinate s in [-1, 1] from angle and arc interval."""
    alpha, beta = float(arc[0]), float(arc[1])
    span = (beta - alpha) % (2 * np.pi)
    if span < 1e-10:
        return 0.0
    angle_diff = (theta - alpha) % (2 * np.pi)
    s_normalized = min(angle_diff / span, 1.0)
    return float(2 * s_normalized - 1)


# =============================================================================
# Per-pair contact force and Hessian
# =============================================================================

def _compute_pair_force_and_hessian(NODES1, NODES2, arc1_ext, arc2_ext,
                                     beam_radius, d_imc, K_imc,
                                     is_first_seg_1, is_last_seg_1,
                                     is_first_seg_2, is_last_seg_2):
    """
    Compute contact force and Hessian for a single segment pair.

    Parameters
    ----------
    NODES1, NODES2 : tuple of 3 arrays, each (3,)
    arc1_ext, arc2_ext : (2,) extended arc intervals
    beam_radius : float
    d_imc, K_imc : IMC regularization parameters
    is_first/last_seg : boundary flags for weight correction

    Returns
    -------
    energy : float
    force : (18,) gradient of energy w.r.t. node coordinates
    hessian : (18, 18) Hessian of energy w.r.t. node coordinates
    gap : float, physical gap (negative = penetration)
    s_i1, s_i2 : float, arc coordinates for partition of unity
    """
    zero_result = (0.0, np.zeros(18), np.zeros((18, 18)), float('inf'), 0.0, 0.0)

    try:
        C1, R1, n1 = circle_from_three_nodes(*NODES1)
        C2, R2, n2 = circle_from_three_nodes(*NODES2)
    except (ValueError, np.linalg.LinAlgError):
        return zero_result

    try:
        res = arc_min_dist(C1, R1, n1, arc1_ext, C2, R2, n2, arc2_ext)
    except (ValueError, np.linalg.LinAlgError):
        return zero_result

    d_star = res['dist']
    if d_star < 1e-14:
        return zero_result

    # Partition-of-unity weights
    s_i1 = compute_s_from_theta(res['theta1'], arc1_ext)
    s_i2 = compute_s_from_theta(res['theta2'], arc2_ext)
    w1 = compute_weights_from_s(s_i1)
    w2 = compute_weights_from_s(s_i2)

    # Boundary corrections: beam endpoints get full weight
    if is_first_seg_1 and s_i1 < 0:
        w1 = 1.0
    if is_last_seg_1 and s_i1 > 0:
        w1 = 1.0
    if is_first_seg_2 and s_i2 < 0:
        w2 = 1.0
    if is_last_seg_2 and s_i2 > 0:
        w2 = 1.0

    # IMC energy — no scaling, same convention as the 2D circle contact
    # (compute_contact_forces_wrt_q in circle_four_nodes_imc_weighted.py):
    #   delta_c = gap + beam_radius,  where gap = d_star - 2*beam_radius
    beam_R_eff = 2 * beam_radius
    gap = d_star - beam_R_eff
    delta_c = gap + beam_radius   # = d_star - beam_radius

    E = computeE_IMC(delta_c, beam_radius, d_imc, K_imc)
    dE_dd = compute_dE_IMC_ddelta_c(delta_c, beam_radius, d_imc, K_imc)
    d2E_dd2 = compute_d2E_IMC_ddelta_c(delta_c, beam_radius, d_imc, K_imc)

    # Skip if no contact force
    if abs(E) < 1e-30 and abs(dE_dd) < 1e-30:
        return (0.0, np.zeros(18), np.zeros((18, 18)), gap, s_i1, s_i2)

    # Gradient of d* w.r.t. nodes (18-vector)
    grad_nodes = compute_gradient_nodes(res, C1, R1, n1, NODES1, C2, R2, n2, NODES2)
    dd_dN = np.concatenate([grad_nodes['dd_dN1'], grad_nodes['dd_dN2']])

    # Assemble weighted force and Hessian
    # Use Gauss-Newton (positive-definite) approximation only, matching the
    # 2D code in self_contact.py which drops the indefinite d2gap term for
    # Newton stability ("Hessian: positive-definite terms only").
    w = w1 * w2
    force = w * dE_dd * dd_dN
    hessian = w * d2E_dd2 * np.outer(dd_dN, dd_dN)

    energy = w * E

    return energy, force, hessian, gap, s_i1, s_i2


def _node_partition_from_s(s):
    """
    Per-node partition-of-unity for a 3-node arc segment, evaluated at the
    contact-point arc parameter s in [-1, +1].  Used both to interpolate
    nodal velocities to the contact point AND to distribute the friction
    force back to the nodes (consistent virtual work).

    Returns (w0, w1, w2) with:
        s = -1 (contact at node 0): (1, 0, 0)
        s =  0 (contact at middle node): (0, 1, 0)
        s = +1 (contact at node 2): (0, 0, 1)

    Smoothness: this is the C^infinity-smooth replacement for the original
    piecewise-linear tent w0 = max(0,-s), w1 = 1-|s|, w2 = max(0,+s).  The
    middle weight reuses `compute_weights_from_s` (already used by the
    Eberly contact-force pass for the per-pair partition along the rod),
    so the entire friction force becomes a smooth function of q with no
    C^1 kink when the contact point migrates across the middle node.

    Construction:
        W(s)      := compute_weights_from_s(s)   # bump centered at s=0
        w1(s)     := W(s)
        rest(s)   := 1 - W(s)        ( in [0, 1] )
        w0(s)     := rest(s) * (1 - s) / 2
        w2(s)     := rest(s) * (1 + s) / 2

    Verifies the four standard partition-of-unity properties:
      - partition of unity: w0 + w1 + w2 = W + (1-W)*((1-s)/2 + (1+s)/2) = 1
      - endpoint reproduction: W(+/-1) = 0, so weights at s=+/-1 are
        (1,0,0) or (0,0,1) as required
      - non-negativity: W in [0,1] and (1+/-s)/2 in [0,1] for s in [-1,1]
      - symmetry: W is even in s, so w0(s) = w2(-s) by construction

    Compared to the quadratic Bernstein basis (w0=(1-s)^2/4, w1=(1-s^2)/2,
    w2=(1+s)^2/4) this choice retains the original "midpoint contact lands
    on the middle node" property of the linear tent — at s = 0 the
    Bernstein basis spreads the contact (1/4, 1/2, 1/4) across three nodes,
    whereas W(0) = 1 here gives (0, 1, 0).
    """
    s = float(np.clip(s, -1.0, 1.0))
    w1 = float(compute_weights_from_s(s))    # smooth bump, W(0)=1, W(+/-1)=0
    rest = 1.0 - w1
    w0 = rest * (1.0 - s) / 2.0
    w2 = rest * (1.0 + s) / 2.0
    return (w0, w1, w2)


# Backwards-compatible alias — keeps any external code or tests that import
# the old name working.  Both names point at the smooth partition above.
_tent_weights_from_s = _node_partition_from_s


def _friction_gate(tv_n, vel_tol, friction_law='smooth_coulomb',
                   nh_rho=0.1, nh_v_ref=1.0):
    """Return (G, dG/d|t_v|) for the chosen friction law.

    All laws give friction force `F_T = mu * |F_n| * G(|v_t|) * t_hat`.

    Parameters
    ----------
    tv_n : float, |v_t| (>= 0)
    vel_tol : float, smooth-Coulomb stick/slip threshold (m/s)
    friction_law : 'smooth_coulomb' or 'norton_hoff'
    nh_rho : float, Norton-Hoff exponent (0 < rho <= 1; small -> Coulomb-like)
    nh_v_ref : float, Norton-Hoff reference velocity (m/s) — G(v_ref)=1
    """
    if friction_law == 'smooth_coulomb':
        K2 = 15.0 / vel_tol
        if tv_n > vel_tol:
            return 1.0, 0.0
        y = np.exp(-K2 * tv_n)
        G = 2.0 / (1.0 + y) - 1.0
        dG = 2.0 * K2 * y / (1.0 + y) ** 2
        return G, dG

    if friction_law == 'norton_hoff':
        # G = (|v_t| / v_ref)^rho. Below a tiny floor, return 0 to avoid 0^0.
        if tv_n < 1e-14:
            return 0.0, 0.0
        G = (tv_n / nh_v_ref) ** nh_rho
        dG = nh_rho * G / tv_n
        return G, dG

    if friction_law == 'norton_hoff_capped':
        # Capped Norton-Hoff: G = min((|v_t|/v_ref)^rho, 1).
        # Below the cap behaves like NH; above the cap saturates at the
        # Coulomb cone like smooth Coulomb does above v_tol.  Used when
        # transient velocity spikes in Newton would push pure NH into the
        # unbounded regime and destabilize the solver.
        if tv_n < 1e-14:
            return 0.0, 0.0
        G_unc = (tv_n / nh_v_ref) ** nh_rho
        if G_unc >= 1.0:
            return 1.0, 0.0
        return G_unc, nh_rho * G_unc / tv_n

    raise ValueError(f"Unknown friction_law: {friction_law!r}")


def _compute_pair_friction(NODES1, NODES2, U1, U2, arc1_ext, arc2_ext,
                            beam_radius, d_imc, K_imc, k_c, mu, vel_tol,
                            is_first_seg_1, is_last_seg_1,
                            is_first_seg_2, is_last_seg_2,
                            friction_law='smooth_coulomb',
                            nh_rho=0.1, nh_v_ref=1.0):
    """
    Coulomb-like friction force for a single Eberly contact pair.

    Recomputes the regularized normal contact magnitude (so this helper is
    self-contained, matching IMCFrictionEnergy's convention) and applies the
    same smooth sticking/sliding gate as the IMC friction model.

    The contact-point velocity is interpolated by the per-node partition
    (`_node_partition_from_s`) on the arc parameter s, NOT by dM/dN @ U.  The
    latter blows up when the circle-fit matrix is ill-conditioned (small
    initial curvature → nearly-parallel edges → near-singular A in
    dCircle_dnodes), producing v_M values orders of magnitude larger than
    the input nodal velocities.  Tent interpolation is bounded by
    max(|U|) and physically faithful: when the contact is at one of the
    3 nodes, only that node's velocity contributes.

    Parameters
    ----------
    NODES1, NODES2 : tuple of 3 arrays, each (3,)
        Node positions of the two arcs.
    U1, U2 : (9,) arrays
        Velocity DOFs of the two arcs (stacked node-wise, same order as NODES).
    arc1_ext, arc2_ext : (2,) extended arc intervals.
    beam_radius, d_imc, K_imc : IMC regularization parameters.
    k_c : float
        Contact stiffness multiplier (same scaling as the contact force pass).
    mu, vel_tol : floats
        Coulomb friction coefficient and sticking-to-sliding velocity tolerance.
    boundary flags : as in _compute_pair_force_and_hessian.

    Returns
    -------
    friction_force : (18,) array
        Equivalent nodal friction force, in the same sign convention as the
        IMC friction force (i.e. the caller should subtract this from the
        residual: `forces -= friction_force`).
    """
    zero_force = np.zeros(18)

    try:
        C1, R1, n1 = circle_from_three_nodes(*NODES1)
        C2, R2, n2 = circle_from_three_nodes(*NODES2)
    except (ValueError, np.linalg.LinAlgError):
        return zero_force

    try:
        res = arc_min_dist(C1, R1, n1, arc1_ext, C2, R2, n2, arc2_ext)
    except (ValueError, np.linalg.LinAlgError):
        return zero_force

    d_star = res['dist']
    if d_star < 1e-14:
        return zero_force

    s_i1 = compute_s_from_theta(res['theta1'], arc1_ext)
    s_i2 = compute_s_from_theta(res['theta2'], arc2_ext)
    return _pair_friction_from_geometry(
        d_star, s_i1, s_i2, res['M1'], res['M2'], U1, U2,
        beam_radius, d_imc, K_imc, k_c, mu, vel_tol,
        is_first_seg_1, is_last_seg_1, is_first_seg_2, is_last_seg_2,
        friction_law, nh_rho, nh_v_ref)


def _compute_pair_friction_line(NODES1, NODES2, U1, U2,
                                beam_radius, d_imc, K_imc, k_c, mu, vel_tol,
                                is_first_seg_1, is_last_seg_1,
                                is_first_seg_2, is_last_seg_2,
                                friction_law='smooth_coulomb',
                                nh_rho=0.1, nh_v_ref=1.0):
    """Line-primitive twin of `_compute_pair_friction` for near-straight
    3-node windows: contact geometry from the edge-midpoint segments (same
    primitive as `_compute_pair_force_and_hessian_line`), then the shared
    friction law."""
    zero_force = np.zeros(18)
    NA0, NA1, NA2 = NODES1
    NB0, NB1, NB2 = NODES2
    # Same EXTENDED chords as `_compute_pair_force_and_hessian_line`.
    mA1 = 0.5 * (NA0 + NA1); mA2 = 0.5 * (NA1 + NA2)
    mB1 = 0.5 * (NB0 + NB1); mB2 = 0.5 * (NB1 + NB2)
    eA1 = mA1 - 0.5 * (mA2 - mA1); eA2 = mA2 + 0.5 * (mA2 - mA1)
    eB1 = mB1 - 0.5 * (mB2 - mB1); eB2 = mB2 + 0.5 * (mB2 - mB1)
    s_A, s_B, cpA, cpB, d_star = _lumelsky_segment_segment_3d(eA1, eA2,
                                                              eB1, eB2)
    if d_star < 1e-14:
        return zero_force
    s_i1 = 2.0 * s_A - 1.0
    s_i2 = 2.0 * s_B - 1.0
    return _pair_friction_from_geometry(
        d_star, s_i1, s_i2, cpA, cpB, U1, U2,
        beam_radius, d_imc, K_imc, k_c, mu, vel_tol,
        is_first_seg_1, is_last_seg_1, is_first_seg_2, is_last_seg_2,
        friction_law, nh_rho, nh_v_ref)


def _pair_friction_from_geometry(d_star, s_i1, s_i2, M1, M2, U1, U2,
                                 beam_radius, d_imc, K_imc, k_c, mu, vel_tol,
                                 is_first_seg_1, is_last_seg_1,
                                 is_first_seg_2, is_last_seg_2,
                                 friction_law='smooth_coulomb',
                                 nh_rho=0.1, nh_v_ref=1.0):
    """Friction force from an already-computed contact geometry
    (d_star, arc coordinates s, closest points M1/M2) — shared by the arc
    and line primitives."""
    zero_force = np.zeros(18)

    # Partition-of-unity weights (identical to the contact-force pass)
    w1 = compute_weights_from_s(s_i1)
    w2 = compute_weights_from_s(s_i2)
    if is_first_seg_1 and s_i1 < 0: w1 = 1.0
    if is_last_seg_1  and s_i1 > 0: w1 = 1.0
    if is_first_seg_2 and s_i2 < 0: w2 = 1.0
    if is_last_seg_2  and s_i2 > 0: w2 = 1.0
    w = w1 * w2

    # Regularized normal-force magnitude at this pair, in the same units
    # the contact pass uses after `F *= k_c`.
    beam_R_eff = 2 * beam_radius
    gap = d_star - beam_R_eff
    delta_c = gap + beam_radius
    dE_dd = compute_dE_IMC_ddelta_c(delta_c, beam_radius, d_imc, K_imc)
    f_n_mag = abs(k_c * w * dE_dd)
    if f_n_mag < 1e-30:
        return zero_force

    # Per-node smooth partition-of-unity for velocity interpolation and
    # friction distribution.  See `_node_partition_from_s`.
    w1_0, w1_1, w1_2 = _node_partition_from_s(s_i1)
    w2_0, w2_1, w2_2 = _node_partition_from_s(s_i2)

    # Contact-point velocities: bounded linear interpolation of the
    # 3 nodal velocities of each arc (no dM/dN amplification).
    U1_nodes = U1.reshape(3, 3)
    U2_nodes = U2.reshape(3, 3)
    v_M1 = w1_0 * U1_nodes[0] + w1_1 * U1_nodes[1] + w1_2 * U1_nodes[2]
    v_M2 = w2_0 * U2_nodes[0] + w2_1 * U2_nodes[1] + w2_2 * U2_nodes[2]
    v_rel = v_M1 - v_M2

    # Contact normal directly from geometry.  Sign is irrelevant for the
    # tangent projection that follows.
    n_hat = (M1 - M2) / d_star
    v_rel_n = _dot3(v_rel, n_hat)
    tv_rel = v_rel - v_rel_n * n_hat
    tv_n = _norm3(tv_rel)

    eps = 1e-14
    if tv_n < eps:
        return zero_force

    # Friction gate G(|v_t|) — smooth Coulomb or Norton-Hoff
    gamma, _ = _friction_gate(tv_n, vel_tol, friction_law, nh_rho, nh_v_ref)

    # Sign convention matches IMCFrictionEnergy (verified with a standalone
    # test): with v_rel = v_M1 - v_M2, body 1 nodes receive +f_tan and body
    # 2 nodes receive -f_tan in F_fric.  After `forces -= F_fric` the
    # stationary body is dragged in the direction of the other body's
    # motion; the moving body is decelerated.  Momentum balance:
    # sum(F_arc1) + sum(F_arc2) = (w1_0+w1_1+w1_2 - w2_0-w2_1-w2_2)*f_tan = 0
    # since each tent partition sums to 1.
    f_tan = (mu * gamma * f_n_mag) * (tv_rel / tv_n)

    F_arc1 = np.empty(9)
    F_arc1[0:3] =  w1_0 * f_tan
    F_arc1[3:6] =  w1_1 * f_tan
    F_arc1[6:9] =  w1_2 * f_tan

    F_arc2 = np.empty(9)
    F_arc2[0:3] = -w2_0 * f_tan
    F_arc2[3:6] = -w2_1 * f_tan
    F_arc2[6:9] = -w2_2 * f_tan

    return np.concatenate([F_arc1, F_arc2])


def _compute_pair_friction_with_velocity_jacobian(
    NODES1, NODES2, U1, U2,
    arc1_ext, arc2_ext,
    beam_radius, d_imc, K_imc, k_c,
    mu, vel_tol,
    is_first_seg_1, is_last_seg_1,
    is_first_seg_2, is_last_seg_2,
    friction_law='smooth_coulomb', nh_rho=0.1, nh_v_ref=1.0,
):
    """
    Friction Term-1 helper: returns (f_local_18, jac_18x18) where jac is the
    PER-PAIR ∂F_fric/∂U block (velocity-Jacobian).  The caller multiplies by
    1/dt and assembles into the global tangent.

    Math
    ----
    f_tan(v_rel) = mu * gamma(|tv|) * |f_n| * tv / |tv|
        where  tv = (I - n n^T) v_rel  is the tangential rel. velocity.
    ∂f_tan/∂v_rel = mu*|f_n| * [ gamma/|tv| * (I - t t^T) + γ' * t t^T ] · (I - n n^T)
                  =: A_eff
    Distribution: F[3k:3k+3] = sw[k] * f_tan with
        sw = [w1_0, w1_1, w1_2, -w2_0, -w2_1, -w2_2].
    Hence per-pair (18,18) block:
        J[3k:3k+3, 3j:3j+3] = sw[k] * sw[j] * A_eff
    """
    zero_F = np.zeros(18)
    zero_J = np.zeros((18, 18))
    zero_aux = (0.0, np.zeros(6), (0.0, 0.0, np.zeros(3)))

    try:
        C1, R1, n1_ax = circle_from_three_nodes(*NODES1)
        C2, R2, n2_ax = circle_from_three_nodes(*NODES2)
    except (ValueError, np.linalg.LinAlgError):
        return zero_F, zero_J, *zero_aux

    try:
        res = arc_min_dist(C1, R1, n1_ax, arc1_ext, C2, R2, n2_ax, arc2_ext)
    except (ValueError, np.linalg.LinAlgError):
        return zero_F, zero_J, *zero_aux

    d_star = res['dist']
    if d_star < 1e-14:
        return zero_F, zero_J, *zero_aux

    s_i1 = compute_s_from_theta(res['theta1'], arc1_ext)
    s_i2 = compute_s_from_theta(res['theta2'], arc2_ext)
    w1 = compute_weights_from_s(s_i1)
    w2 = compute_weights_from_s(s_i2)
    if is_first_seg_1 and s_i1 < 0: w1 = 1.0
    if is_last_seg_1  and s_i1 > 0: w1 = 1.0
    if is_first_seg_2 and s_i2 < 0: w2 = 1.0
    if is_last_seg_2  and s_i2 > 0: w2 = 1.0
    w = w1 * w2

    beam_R_eff = 2 * beam_radius
    gap = d_star - beam_R_eff
    delta_c = gap + beam_radius
    dE_dd = compute_dE_IMC_ddelta_c(delta_c, beam_radius, d_imc, K_imc)
    f_n_mag = abs(k_c * w * dE_dd)
    if f_n_mag < 1e-30:
        return zero_F, zero_J, *zero_aux

    w1_0, w1_1, w1_2 = _node_partition_from_s(s_i1)
    w2_0, w2_1, w2_2 = _node_partition_from_s(s_i2)
    sw = np.array([w1_0, w1_1, w1_2, -w2_0, -w2_1, -w2_2])

    U1_nodes = U1.reshape(3, 3)
    U2_nodes = U2.reshape(3, 3)
    v_M1 = w1_0 * U1_nodes[0] + w1_1 * U1_nodes[1] + w1_2 * U1_nodes[2]
    v_M2 = w2_0 * U2_nodes[0] + w2_1 * U2_nodes[1] + w2_2 * U2_nodes[2]
    v_rel = v_M1 - v_M2

    n_hat = (res['M1'] - res['M2']) / d_star
    v_rel_n = _dot3(v_rel, n_hat)
    tv_rel = v_rel - v_rel_n * n_hat
    tv_n = _norm3(tv_rel)

    eps = 1e-14
    if tv_n < eps:
        return zero_F, zero_J, *zero_aux

    gamma, dgamma = _friction_gate(tv_n, vel_tol, friction_law, nh_rho, nh_v_ref)

    t_hat = tv_rel / tv_n
    f_tan = (mu * gamma * f_n_mag) * t_hat

    # Force (same as _compute_pair_friction)
    F_local = np.empty(18)
    for k, sk in enumerate(sw):
        F_local[3 * k:3 * k + 3] = sk * f_tan

    # Velocity-Jacobian: A_eff = ∂f_tan/∂v_rel  (3,3)
    I3 = np.eye(3)
    P_n = I3 - np.outer(n_hat, n_hat)
    P_t = I3 - np.outer(t_hat, t_hat)
    A_loc = mu * f_n_mag * (
        (gamma / tv_n) * P_t + dgamma * np.outer(t_hat, t_hat)
    )
    A_eff = A_loc @ P_n  # ∂f_tan/∂v_rel

    # Per-pair (18,18) block: outer(sw, sw) ⊗ A_eff
    J_local = np.kron(np.outer(sw, sw), A_eff)
    return F_local, J_local, f_n_mag, np.array([w1_0, w1_1, w1_2,
                                                w2_0, w2_1, w2_2]), \
           (gamma, mu, t_hat)


def _compute_pair_friction_unit_distribution(NODES1, NODES2, U1, U2,
                                              arc1_ext, arc2_ext,
                                              mu, vel_tol):
    """
    Semi-implicit friction setup: precompute the per-pair friction force
    PER UNIT OF NORMAL CONTACT FORCE, freezing the tangent direction, the
    smooth Coulomb gate, AND the contact-point distribution Jacobians at
    the start-of-timestep configuration.

    Returns
    -------
    (unit_F_arc1, unit_F_arc2) : tuple of two (9,) arrays
        Each is `±mu * gamma * (dM_i/dN_i)^T @ t_hat` so that at apply time
        F_arc_i = f_n_mag(q_iter) * unit_F_arc_i.
    or None
        If the pair is not in contact at first_iter, or has no tangential
        relative velocity.
    """
    try:
        C1, R1, n1 = circle_from_three_nodes(*NODES1)
        C2, R2, n2 = circle_from_three_nodes(*NODES2)
    except (ValueError, np.linalg.LinAlgError):
        return None

    try:
        res = arc_min_dist(C1, R1, n1, arc1_ext, C2, R2, n2, arc2_ext)
    except (ValueError, np.linalg.LinAlgError):
        return None

    d_star = res['dist']
    if d_star < 1e-14:
        return None

    # Explicit (fixed-theta) dM_i/dN_i — same construction as in
    # _compute_pair_friction.
    rho1 = res['M1'] - C1
    rho2 = res['M2'] - C2
    I3 = np.eye(3)
    dM1_expl = np.zeros((3, 7))
    dM1_expl[:, 0:3] = I3
    dM1_expl[:, 3]   = rho1 / R1
    dM1_expl[:, 4:7] = -np.outer(n1, rho1)
    dM2_expl = np.zeros((3, 7))
    dM2_expl[:, 0:3] = I3
    dM2_expl[:, 3]   = rho2 / R2
    dM2_expl[:, 4:7] = -np.outer(n2, rho2)

    dC1_dN, dR1_dN, dn1_dN = dCircle_dnodes(*NODES1)
    dC2_dN, dR2_dN, dn2_dN = dCircle_dnodes(*NODES2)
    J_circ1 = np.vstack([dC1_dN, dR1_dN[None, :], dn1_dN])
    J_circ2 = np.vstack([dC2_dN, dR2_dN[None, :], dn2_dN])

    dM1_dN1 = dM1_expl @ J_circ1
    dM2_dN2 = dM2_expl @ J_circ2

    v_M1 = dM1_dN1 @ U1
    v_M2 = dM2_dN2 @ U2
    v_rel = v_M1 - v_M2

    n_hat = (res['M1'] - res['M2']) / d_star
    v_rel_n = _dot3(v_rel, n_hat)
    tv_rel = v_rel - v_rel_n * n_hat
    tv_n = _norm3(tv_rel)

    eps = 1e-14
    if tv_n < eps:
        return None

    K2 = 15.0 / vel_tol
    if tv_n > vel_tol:
        gamma = 1.0
    else:
        gamma = 2.0 / (1.0 + np.exp(-K2 * tv_n)) - 1.0

    t_hat = tv_rel / tv_n

    # Sign convention matches IMCFrictionEnergy: body 1 gets +mu*gamma*|f_n|*t_hat
    # distributed via (dM1/dN1)^T; body 2 gets the opposite sign.  Split so
    # the per-iter |f_n(q)| is supplied at apply time:
    #     F_arc_i = |f_n(q_iter)| * unit_F_arc_i
    coeff = mu * gamma
    unit_F_arc1 =  coeff * (dM1_dN1.T @ t_hat)     # (9,)
    unit_F_arc2 = -coeff * (dM2_dN2.T @ t_hat)     # (9,)
    return (unit_F_arc1, unit_F_arc2)


def _compute_pair_fn_magnitude(NODES1, NODES2, arc1_ext, arc2_ext,
                                beam_radius, d_imc, K_imc, k_c,
                                is_first_seg_1, is_last_seg_1,
                                is_first_seg_2, is_last_seg_2):
    """
    Per-pair contact-force magnitude at current q.  Recomputes circles,
    arc_min_dist, partition-of-unity weight and the regularized dE_dd.
    Returns 0.0 if the pair is not in contact or geometry fails.
    """
    try:
        C1, R1, n1 = circle_from_three_nodes(*NODES1)
        C2, R2, n2 = circle_from_three_nodes(*NODES2)
    except (ValueError, np.linalg.LinAlgError):
        return 0.0

    try:
        res = arc_min_dist(C1, R1, n1, arc1_ext, C2, R2, n2, arc2_ext)
    except (ValueError, np.linalg.LinAlgError):
        return 0.0

    d_star = res['dist']
    if d_star < 1e-14:
        return 0.0

    s_i1 = compute_s_from_theta(res['theta1'], arc1_ext)
    s_i2 = compute_s_from_theta(res['theta2'], arc2_ext)
    w1 = compute_weights_from_s(s_i1)
    w2 = compute_weights_from_s(s_i2)
    if is_first_seg_1 and s_i1 < 0: w1 = 1.0
    if is_last_seg_1  and s_i1 > 0: w1 = 1.0
    if is_first_seg_2 and s_i2 < 0: w2 = 1.0
    if is_last_seg_2  and s_i2 > 0: w2 = 1.0
    w = w1 * w2

    beam_R_eff = 2 * beam_radius
    gap = d_star - beam_R_eff
    delta_c = gap + beam_radius
    dE_dd = compute_dE_IMC_ddelta_c(delta_c, beam_radius, d_imc, K_imc)
    return abs(k_c * w * dE_dd)


# =============================================================================
# Self-contact force computation (3D Eberly)
# =============================================================================

def _scalar_arc_pair(node_pos, i0, j0, flags, beam_radius, d_imc, K_imc):
    """Scalar per-pair body shared by the drivers: arc fit -> intervals ->
    per-pair kernel.  Returns the kernel 6-tuple, or None if the arc fit is
    degenerate (pair skipped)."""
    NODES1 = (node_pos[i0], node_pos[i0 + 1], node_pos[i0 + 2])
    NODES2 = (node_pos[j0], node_pos[j0 + 1], node_pos[j0 + 2])
    try:
        C1, R1, n1_ax = circle_from_three_nodes(*NODES1)
        C2, R2, n2_ax = circle_from_three_nodes(*NODES2)
    except (ValueError, np.linalg.LinAlgError):
        return None
    arc1_ext = extend_arc(arc_interval_from_nodes(*NODES1, C1, R1, n1_ax))
    arc2_ext = extend_arc(arc_interval_from_nodes(*NODES2, C2, R2, n2_ax))
    return _compute_pair_force_and_hessian(
        NODES1, NODES2, arc1_ext, arc2_ext, beam_radius, d_imc, K_imc, *flags)


def compute_eberly_3d_self_contact_forces(
    robot: "SoftRobot",
    q: np.ndarray,
    beam_radius: float,
    k_c: float = 1.0,
    min_segment_separation: int = 6,
    d_imc: float = 0.0008,
    h_scale: bool = False,
    backend: str = 'batched',
    straight_R_threshold: float = None,
    conforming_edge_blend: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float, List[Dict]]:
    """
    Compute self-contact forces within a single beam using Eberly's 3D arc
    distance.  Operates directly on 3D node coordinates — no 2D projection.

    ``conforming_edge_blend`` (opt-in, default OFF -> bit-identical
    behavior): crossfade to the two-node edge model where the smooth
    conformity field detects extended contact — see the section comment
    above ``_conformity_gammas``.

    Parameters
    ----------
    robot : SoftRobot
    q : full DOF vector
    beam_radius : cross-section radius
    k_c : contact stiffness multiplier
    min_segment_separation : minimum index gap between segment pairs
    d_imc : IMC regularization width
    straight_R_threshold : float or None
        When set, pairs whose 3-node windows are near-straight (fitted arc
        radius around/above this value, or a degenerate collinear fit) use
        the IMC edge-to-edge line primitive instead of the arc kernel,
        with a smooth blend over [0.8, 1.25]·threshold (see
        `_pair_line_weights`).  The arc kernel's distance gradient is
        ill-conditioned for R >> beam_radius (straightening tails,
        braid-exit transition zones), and a collinear fit previously
        dropped the pair's force entirely.

    Returns
    -------
    F : (ndof,) force vector (gradient of contact energy)
    J : (ndof, ndof) Jacobian (Hessian of contact energy)
    gap_min : minimum gap across all segment pairs
    all_gaps : list of gap info dicts
    """
    n_nodes = robot.n_nodes
    node_dof = robot.node_dof_indices  # (n_nodes, 3)
    ndof = q.shape[0]
    node_pos = q[node_dof]  # (n_nodes, 3)

    num_segments = n_nodes - 2
    F = np.zeros(ndof)
    J = np.zeros((ndof, ndof))
    all_gaps = []

    if num_segments <= 1:
        return F, J, float('inf'), all_gaps

    K_imc = 15.0 / d_imc
    beam_R_eff = 2 * beam_radius
    interaction_range = beam_R_eff + d_imc

    # Broad phase (vectorized): bounding-sphere culling + minimum index
    # separation along the rod.
    starts = np.arange(num_segments)
    ii, jj = _broad_phase_pairs(node_pos, starts, starts, interaction_range)
    sep_ok = jj >= ii + min_segment_separation
    ii, jj = ii[sep_ok], jj[sep_ok]

    # Near-straight routing with a SMOOTH arc→line blend: pure arc below
    # R_lo = 0.8·thr, pure line above R_hi = 1.25·thr, smoothstep mix in
    # between (weight treated as constant w.r.t. q, envelope-style).  A hard
    # switch at thr makes the residual discontinuous the instant a window's
    # fitted radius crosses it; in a tightening trefoil ~20 active windows
    # hover right at the threshold and the summed jumps broke Newton
    # (50-iter limit cycles, 20260917 smoke tests).
    blend_w = _pair_line_weights(ii, jj, node_pos, straight_R_threshold)

    # Conforming-contact crossfade weights (opt-in; 0 on transversal
    # contact -> those pairs are untouched).
    gammas = gamma_edge = edge_partners = None
    if conforming_edge_blend and ii.size:
        gammas, gamma_edge, edge_partners = _conformity_edge_field(
            node_pos, ii, jj, beam_radius, d_imc)

    bres = None
    arc_pos = None      # position of pair k inside the batched arrays
    if backend == 'batched' and ii.size:
        from .eberly_batched import batched_pair_forces
        if blend_w is not None:
            arc_sel = np.nonzero(blend_w < 1.0)[0]   # pairs needing the arc
        else:
            arc_sel = np.arange(ii.size)
        if arc_sel.size:
            iia, jja = ii[arc_sel], jj[arc_sel]
            bres = batched_pair_forces(
                node_pos, iia, jja, beam_radius, d_imc, K_imc,
                iia == 0, iia == num_segments - 1,
                jja == 0, jja == num_segments - 1)
            arc_pos = np.full(ii.size, -1)
            arc_pos[arc_sel] = np.arange(arc_sel.size)

    _zeros_pair = (0.0, None, None, np.inf, 0.0, 0.0)   # inactive sentinel

    # Edge-model pass: one midpoint sample per conforming EDGE against the
    # opposing polyline, weight gamma/2 per side (energy density = one
    # penalty per rod length per strand — registration-independent).
    if gamma_edge:
        for e, g_e in sorted(gamma_edge.items()):
            out_e = _edge_sample_contact(node_pos, e, edge_partners[e],
                                         beam_radius, d_imc, K_imc)
            if out_e is None:
                continue
            Ee, Fe, He, gap_e, nds = out_e
            w_e = 0.5 * g_e
            dof_e = np.concatenate([node_dof[nd] for nd in nds])
            F[dof_e] += w_e * Fe
            J[np.ix_(dof_e, dof_e)] += w_e * He
            all_gaps.append({
                'segment_pair': f'(e{e}->e{nds[2]})[edge]',
                'gap': gap_e, 's_i1': 0.0, 's_i2': 0.0,
            })

    for k in range(ii.size):
        i, j = int(ii[k]), int(jj[k])
        flags = (i == 0, i == num_segments - 1,
                 j == 0, j == num_segments - 1)
        w_line = float(blend_w[k]) if blend_w is not None else 0.0
        gamma_k = float(gammas[k]) if gammas is not None else 0.0

        line_res = _zeros_pair
        if w_line > 0.0:
            NODES1 = (node_pos[i], node_pos[i + 1], node_pos[i + 2])
            NODES2 = (node_pos[j], node_pos[j + 1], node_pos[j + 2])
            line_res = _compute_pair_force_and_hessian_line(
                NODES1, NODES2, beam_radius, d_imc, K_imc, *flags)

        arc_res = _zeros_pair
        if w_line < 1.0:
            kb = arc_pos[k] if arc_pos is not None else k
            if bres is not None and kb >= 0 and not bres['fallback'][kb]:
                if bres['active'][kb]:
                    arc_res = (bres['energy'][kb], bres['force'][kb],
                               bres['hessian'][kb], bres['gap'][kb],
                               bres['s_i1'][kb], bres['s_i2'][kb])
            else:
                out = _scalar_arc_pair(node_pos, i, j, flags,
                                       beam_radius, d_imc, K_imc)
                if out is not None:
                    arc_res = out

        win_scale = 1.0 - gamma_k
        energy = win_scale * ((1.0 - w_line) * arc_res[0]
                              + w_line * line_res[0])
        if abs(energy) < 1e-30:
            continue
        force_local = np.zeros(18)
        hess_local = np.zeros((18, 18))
        if arc_res[1] is not None:
            force_local += win_scale * (1.0 - w_line) * arc_res[1]
            hess_local += win_scale * (1.0 - w_line) * arc_res[2]
        if line_res[1] is not None:
            force_local += win_scale * w_line * line_res[1]
            hess_local += win_scale * w_line * line_res[2]
        gap = min(arc_res[3], line_res[3])
        s1, s2 = (line_res[4], line_res[5]) if w_line >= 0.5 \
            else (arc_res[4], arc_res[5])
        if w_line >= 1.0:
            pair_tag = f'({i},{j})[line]'
        elif w_line > 0.0:
            pair_tag = f'({i},{j})[blend]'
        else:
            pair_tag = f'({i},{j})'
        if gamma_k > 0.005:
            pair_tag += '[conf]'

        # Map local 18 DOFs to global DOFs
        node_indices = [i, i + 1, i + 2, j, j + 1, j + 2]
        global_dof = np.concatenate([node_dof[ni] for ni in node_indices])

        F[global_dof] += force_local
        J[np.ix_(global_dof, global_dof)] += hess_local

        all_gaps.append({
            'segment_pair': pair_tag,
            'gap': gap, 's_i1': s1, 's_i2': s2,
        })

    # Optional 1/h amplification -- see the note in
    # compute_eberly_3d_beam_to_beam_contact_forces.  Off by default.
    scale = (1.0 / beam_radius) if h_scale else 1.0
    F *= k_c * scale
    J *= k_c * scale

    if pair_event_log.enabled:
        pair_event_log.record(('self', n_nodes), all_gaps, beam_radius)

    gap_min = min((g['gap'] for g in all_gaps), default=float('inf'))
    return F, J, gap_min, all_gaps


# =============================================================================
# Near-straight-arc fallback: IMC edge-to-edge contact
# =============================================================================

def _lumelsky_segment_segment_3d(pa, pb, qa, qb):
    """Closest-point parameters (s in [0,1]) and points on two segments in 3D.

    Segment 1 goes from pa to pb; segment 2 from qa to qb. Returns
    (s1, s2, cp1, cp2, d) with cp1 = pa + s1*(pb-pa), cp2 similar, d = |cp1-cp2|.
    """
    d1 = pb - pa
    d2 = qb - qa
    r = pa - qa
    A = float(np.dot(d1, d1))
    B = float(np.dot(d1, d2))
    C = float(np.dot(d2, d2))
    D = float(np.dot(d1, r))
    E = float(np.dot(d2, r))
    det = A * C - B * B
    eps = 1e-15
    if det > eps and A > eps and C > eps:
        s = (B * E - C * D) / det
        t = (A * E - B * D) / det
    else:
        # Nearly parallel — every point of the projected overlap is equally
        # close, so use the overlap MIDPOINT.  An endpoint choice (the
        # classic "pick s=0" shortcut) lands on s_i = ±1 where the
        # partition-of-unity weight vanishes, silently dropping the pair's
        # entire contact force for parallel overlapping segments.
        if A > eps:
            u0 = -D / A                       # proj of qa on segment 1
            u1 = float(np.dot(d1, qb - pa)) / A   # proj of qb
            lo, hi = max(0.0, min(u0, u1)), min(1.0, max(u0, u1))
            if lo <= hi:
                s = 0.5 * (lo + hi)
            else:
                s = 0.0 if hi < 0.0 else 1.0
        else:
            s = 0.0
        t = (s * B + E) / C if C > eps else 0.0
    s = float(np.clip(s, 0.0, 1.0))
    t = float(np.clip(t, 0.0, 1.0))
    # Re-solve if clamped to keep the closest-point relationship consistent.
    # (Standard Lumelsky "clamp-then-re-optimize" pass.)
    if C > eps:
        t = float(np.clip((s * B + E) / C, 0.0, 1.0))
    if A > eps:
        s = float(np.clip((t * B - D) / A, 0.0, 1.0))
    cp1 = pa + s * d1
    cp2 = qa + t * d2
    d = float(np.linalg.norm(cp1 - cp2))
    return s, t, cp1, cp2, d


def _compute_pair_force_and_hessian_line(
    NODES1, NODES2, beam_radius, d_imc, K_imc,
    is_first_seg_1, is_last_seg_1,
    is_first_seg_2, is_last_seg_2,
):
    """IMC edge-to-edge contact for near-straight 3-node arc triplets.

    Fallback used when either arc's fitted circle radius exceeds a
    caller-set threshold. Uses the same IMC energy + partition-of-unity
    weights as the arc-to-arc path, but replaces the arc primitive with
    two straight segments (midpoint-of-edge to midpoint-of-edge). This
    avoids ill-conditioned `dd_dN` when the fitted arc has R >> beam_radius
    (long straight tails etc.).

    Returns
    -------
    energy, force (18-vec), hessian (18x18), gap, s_i1, s_i2   — same layout
    as `_compute_pair_force_and_hessian` so the caller can splice the two.
    """
    zero_result = (0.0, np.zeros(18), np.zeros((18, 18)), np.inf, 0.0, 0.0)

    NA0, NA1, NA2 = NODES1
    NB0, NB1, NB2 = NODES2

    # Midpoints define the two straight segments (matches the arc's
    # tangent-at-midpoint construction in circle_from_three_nodes), then
    # EXTENDED by 50 % per side exactly like `extend_arc` does for the arc
    # primitive.  Without the extension the chords have zero handoff
    # overlap: a transversal contact point crossing a chord junction saw
    # every covering pair at s = ±1 (PoU weight 0) and the force dipped to
    # zero over a ~1 mm zone (found by the 2026-09-18 transition tests;
    # the extension reduces the dip to ~6 %, matching arc behavior).
    mA1 = 0.5 * (NA0 + NA1); mA2 = 0.5 * (NA1 + NA2)
    mB1 = 0.5 * (NB0 + NB1); mB2 = 0.5 * (NB1 + NB2)
    eA1 = mA1 - 0.5 * (mA2 - mA1); eA2 = mA2 + 0.5 * (mA2 - mA1)
    eB1 = mB1 - 0.5 * (mB2 - mB1); eB2 = mB2 + 0.5 * (mB2 - mB1)

    s_A, s_B, cpA, cpB, d_star = _lumelsky_segment_segment_3d(eA1, eA2,
                                                              eB1, eB2)

    beam_R_eff = 2 * beam_radius
    gap = d_star - beam_R_eff
    if gap >= d_imc:      # outside activation zone — no force
        return zero_result

    # PoU weights: Lumelsky s in [0,1] over the EXTENDED chord maps to the
    # arc convention s in [-1,+1] over the extended interval.
    s_i1 = 2.0 * s_A - 1.0
    s_i2 = 2.0 * s_B - 1.0
    w1 = compute_weights_from_s(s_i1)
    w2 = compute_weights_from_s(s_i2)
    if is_first_seg_1 and s_i1 < 0: w1 = 1.0
    if is_last_seg_1  and s_i1 > 0: w1 = 1.0
    if is_first_seg_2 and s_i2 < 0: w2 = 1.0
    if is_last_seg_2  and s_i2 > 0: w2 = 1.0

    # IMC energy (same delta_c as arc path)
    delta_c = d_star - beam_radius     # = gap + beam_radius
    E = computeE_IMC(delta_c, beam_radius, d_imc, K_imc)
    dE_dd = compute_dE_IMC_ddelta_c(delta_c, beam_radius, d_imc, K_imc)
    d2E_dd2 = compute_d2E_IMC_ddelta_c(delta_c, beam_radius, d_imc, K_imc)
    if abs(E) < 1e-30 and abs(dE_dd) < 1e-30:
        return (0.0, np.zeros(18), np.zeros((18, 18)), gap, s_i1, s_i2)

    # Envelope gradient: treat (s_A, s_B) as constants w.r.t. node
    # positions.  Contact point on the extended chord:
    #   cpA = (1-s)·eA1 + s·eA2 with eA1 = 0.75·NA0 + 0.5·NA1 - 0.25·NA2
    #   and eA2 = -0.25·NA0 + 0.5·NA1 + 0.75·NA2
    #   -> node coefficients (0.75 - s_A, 0.5, -0.25 + s_A); sums to 1,
    #   reduces to (0.25, 0.5, 0.25) at the chord middle.
    if d_star < 1e-15:
        return zero_result
    u_hat = (cpA - cpB) / d_star           # unit vector from B->A
    coefs_A = [0.75 - s_A, 0.5, -0.25 + s_A]      # weights on NA0, NA1, NA2
    coefs_B = [0.75 - s_B, 0.5, -0.25 + s_B]      # weights on NB0, NB1, NB2

    dd_dN = np.zeros(18)
    for k, ck in enumerate(coefs_A):
        dd_dN[3*k : 3*k + 3]      =  ck * u_hat
    for k, ck in enumerate(coefs_B):
        dd_dN[3*(3 + k) : 3*(3 + k) + 3] = -ck * u_hat

    w = w1 * w2
    force = w * dE_dd * dd_dN
    # Gauss-Newton Hessian (drops d2gap/dN2, matches the arc-path convention).
    hessian = w * d2E_dd2 * np.outer(dd_dN, dd_dN)
    energy = w * E

    return energy, force, hessian, gap, s_i1, s_i2


# =============================================================================
# Conforming-contact crossfade to the two-node edge model (opt-in)
# =============================================================================
#
# The 3-node-window + partition-of-unity machinery assumes ISOLATED point
# contact.  Conforming (tangential / wrapped) contact violates that
# assumption in two documented ways (knots_catia_Esther/
# stall_diagnosis_20260918/): (1) along a flat contact ridge every window
# pair's minimizer sits at its patch edge, PoU weight 0, so the mid-ridge
# carries no force; (2) a single window pair can have two near-tied minima
# whose hard-min swap makes the force discontinuous.  Both broke Newton.
#
# Where a smooth CONFORMITY field detects extended contact, the pair force
# crossfades to the classic two-node edge model (per-edge Lumelsky penalty,
# no PoU): total = (1-gamma)·window_model + gamma·edge_model.  The edge
# model has no coverage holes and no location ambiguity; its known cost
# (adjacent-pair force redundancy at shared nodes) is acceptable exactly
# there, where the physics is line contact anyway.
#
# Conformity detection: window-chord gap activations a_ij (smoothstep of
# the chord-to-chord surface gap over the d_imc band — smooth in q) are
# summed along the two index diagonals (i+k, j±k), k in [-2, 2].  An
# isolated transversal crossing activates only a 2x2 block (diagonal sum
# <~ 2); a conforming ridge activates a whole diagonal (sum -> 4-5).
# gamma ramps smoothly over [_CONF_D_LO, _CONF_D_HI].  gamma == 0 exactly
# on transversal contact, so with the feature enabled the model is
# bit-identical there; the feature itself is opt-in (default OFF).

_CONF_D_LO = 2.0     # diagonal activation sum where gamma starts rising
_CONF_D_HI = 3.5     # ... and where it saturates at 1
_CONF_K = 2          # diagonal half-width (k in [-K, K])


def _smoothstep01(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def _window_chord_gap(node_pos, i, j, beam_radius):
    """Surface gap between the midpoint-chord proxies of windows i and j
    (same geometry as the line primitive)."""
    mA1 = 0.5 * (node_pos[i] + node_pos[i + 1])
    mA2 = 0.5 * (node_pos[i + 1] + node_pos[i + 2])
    mB1 = 0.5 * (node_pos[j] + node_pos[j + 1])
    mB2 = 0.5 * (node_pos[j + 1] + node_pos[j + 2])
    _, _, _, _, d = _lumelsky_segment_segment_3d(mA1, mA2, mB1, mB2)
    return d - 2.0 * beam_radius


def _conformity_gammas(node_pos, ii, jj, beam_radius, d_imc):
    """Smooth conformity weight gamma in [0, 1] for every pair in (ii, jj).

    gamma is a stateless, smooth function of q: chord-gap activations are
    C^1 in the node positions, the diagonal sums are finite sums of them,
    and the final ramp is a smoothstep.  No counts, no branches on state.
    """
    num_windows = len(node_pos) - 2
    cache = {}

    def act(i, j):
        if not (0 <= i < num_windows and 0 <= j < num_windows):
            return 0.0
        key = (i, j)
        if key not in cache:
            g = _window_chord_gap(node_pos, i, j, beam_radius)
            cache[key] = _smoothstep01((d_imc - g) / d_imc)
        return cache[key]

    gammas = np.zeros(len(ii))
    for k in range(len(ii)):
        i, j = int(ii[k]), int(jj[k])
        d_par = sum(act(i + m, j + m) for m in range(-_CONF_K, _CONF_K + 1))
        d_anti = sum(act(i + m, j - m) for m in range(-_CONF_K, _CONF_K + 1))
        D = max(d_par, d_anti)
        gammas[k] = _smoothstep01((D - _CONF_D_LO) / (_CONF_D_HI - _CONF_D_LO))
    return gammas


def _conformity_edge_field(node_pos, ii, jj, beam_radius, d_imc):
    """Per-pair gammas plus the per-EDGE sampling field for the edge model.

    Returns (gammas, gamma_edge, partners):
      gammas      — (len(ii),) smooth conformity weight per window pair;
      gamma_edge  — dict edge -> max gamma over the window pairs whose
                    windows contain that edge (the sample's weight gate);
      partners    — dict edge -> set of candidate OPPOSING edges, taken
                    from ALL broad-phase pairs (not gamma-gated): a
                    partner that enters the candidate set does so at
                    gap ~ d_imc where the penalty is zero, so the
                    min-over-partners stays continuous.
    """
    gammas = _conformity_gammas(node_pos, ii, jj, beam_radius, d_imc)
    gamma_edge = {}
    partners = {}
    for k in range(len(ii)):
        i, j = int(ii[k]), int(jj[k])
        g = float(gammas[k])
        for ei in (i, i + 1):
            partners.setdefault(ei, set()).update((j, j + 1))
            if g > 0.0:
                gamma_edge[ei] = max(gamma_edge.get(ei, 0.0), g)
        for ej in (j, j + 1):
            partners.setdefault(ej, set()).update((i, i + 1))
            if g > 0.0:
                gamma_edge[ej] = max(gamma_edge.get(ej, 0.0), g)
    return gammas, gamma_edge, partners


def _point_segment_closest(P, Q0, Q1):
    """Closest point on segment [Q0, Q1] to P: (t, Q, d)."""
    dvec = Q1 - Q0
    L2 = _dot3(dvec, dvec)
    t = 0.0 if L2 < 1e-30 else min(1.0, max(0.0, _dot3(P - Q0, dvec) / L2))
    Q = Q0 + t * dvec
    return t, Q, _norm3(P - Q)


# Two-point Gauss abscissae on [0, 1] and the softmin temperature for the
# point-to-polyline distance (removes the C1 kink when a sample's nearest
# partner segment switches — adjacent partner segments share endpoints, so
# near the switch their distances near-tie and the softmin blends them).
_EDGE_GAUSS = (0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0))
_EDGE_SOFTMIN_EPS = 0.01   # mm


def _sample_vs_polyline(P, node_pos, partner_edges):
    """Softmin-blended closest-approach of point P to the partner
    polyline: list of (weight, f, t, Q, d) plus d_min, or None."""
    cands = []
    for f in partner_edges:
        t, Q, d = _point_segment_closest(P, node_pos[f], node_pos[f + 1])
        cands.append((f, t, Q, d))
    if not cands:
        return None
    d_min = min(c[3] for c in cands)
    out = []
    wsum = 0.0
    for f, t, Q, d in cands:
        x = (d - d_min) / _EDGE_SOFTMIN_EPS
        if x > 4.0:
            continue
        w = np.exp(-x) * max(0.0, 1.0 - x / 4.0)
        out.append([w, f, t, Q, d])
        wsum += w
    for c in out:
        c[0] /= wsum
    return out, d_min


def _edge_sample_contact(node_pos, e, partner_edges, beam_radius, d_imc,
                         K_imc):
    """Two-node edge-model penalty, sampled per EDGE (not per edge pair).

    Two Gauss points per edge are penalized against the opposing POLYLINE
    (softmin over the candidate partner segments).  Total sample weight
    1/2 per side makes the contact energy density exactly one penalty per
    rod length l on each strand — registration-independent, unlike
    per-edge-pair Lumelsky penalties which count a conforming contact
    2-4x depending on how the discretizations happen to align.

    Returns (E, F, H, gap, dofs) on [e-nodes + involved partner-edge
    nodes], or None when inactive.  Envelope convention (sample
    locations, partner weights and t treated as constants w.r.t. q).
    """
    samples = []       # (gauss_w, a, d_min, cands)
    gap_min = np.inf
    for a in _EDGE_GAUSS:
        P = (1.0 - a) * node_pos[e] + a * node_pos[e + 1]
        sv = _sample_vs_polyline(P, node_pos, partner_edges)
        if sv is None:
            continue
        cands, d_min = sv
        gap_min = min(gap_min, d_min - 2.0 * beam_radius)
        if d_min - 2.0 * beam_radius >= d_imc or d_min < 1e-14:
            continue
        samples.append((0.5, a, d_min, [(w, f, t, (P - Q) / d)
                                        for w, f, t, Q, d in cands]))
    if not samples:
        return None

    dofs = [e, e + 1]
    for _, _, _, cands in samples:
        for _, f, _, _ in cands:
            for nd in (f, f + 1):
                if nd not in dofs:
                    dofs.append(nd)
    idx = {nd: 3 * k for k, nd in enumerate(dofs)}
    nd3 = 3 * len(dofs)

    E_tot = 0.0
    F = np.zeros(nd3)
    Hs = np.zeros((nd3, nd3))
    active = False
    for w_g, a, d_min, cands in samples:
        # Energy magnitude from the TRUE min distance (smooth along the
        # polyline); the softmin weights only blend the force GEOMETRY
        # (direction and attachment), which is what switches at the
        # partner-segment Voronoi boundaries.
        delta_c = d_min - beam_radius
        E = computeE_IMC(delta_c, beam_radius, d_imc, K_imc)
        dE_dd = compute_dE_IMC_ddelta_c(delta_c, beam_radius, d_imc, K_imc)
        d2E_dd2 = compute_d2E_IMC_ddelta_c(delta_c, beam_radius, d_imc,
                                           K_imc)
        if abs(E) < 1e-30 and abs(dE_dd) < 1e-30:
            continue
        active = True
        dd_dN = np.zeros(nd3)
        for w, f, t, u_hat in cands:
            dd_dN[idx[e]:idx[e] + 3] += w * (1.0 - a) * u_hat
            dd_dN[idx[e + 1]:idx[e + 1] + 3] += w * a * u_hat
            dd_dN[idx[f]:idx[f] + 3] += -w * (1.0 - t) * u_hat
            dd_dN[idx[f + 1]:idx[f + 1] + 3] += -w * t * u_hat
        E_tot += w_g * E
        F += w_g * dE_dd * dd_dN
        Hs += w_g * d2E_dd2 * np.outer(dd_dN, dd_dN)
    if not active:
        return None
    return E_tot, F, Hs, gap_min, tuple(dofs)


def _edge_sample_friction(node_pos, u_nodes, e, partner_edges, beam_radius,
                          d_imc, K_imc, k_c, mu, vel_tol,
                          friction_law='smooth_coulomb',
                          nh_rho=0.1, nh_v_ref=1.0):
    """Coulomb friction for the per-edge samples of `_edge_sample_contact`.

    Returns (F, dofs) or None.  Same Gauss points, softmin partner
    weights and normal-force magnitudes as the contact pass; contact-point
    velocities are barycentric interpolations of the nodal velocities."""
    samples = []
    for a in _EDGE_GAUSS:
        P = (1.0 - a) * node_pos[e] + a * node_pos[e + 1]
        sv = _sample_vs_polyline(P, node_pos, partner_edges)
        if sv is None:
            continue
        cands, d_min = sv
        if d_min - 2.0 * beam_radius >= d_imc or d_min < 1e-14:
            continue
        samples.append((0.5, a, d_min,
                        [(w, f, t, Q, P, d) for w, f, t, Q, d in cands]))
    if not samples:
        return None

    dofs = [e, e + 1]
    for _, _, _, cands in samples:
        for _, f, _, _, _, _ in cands:
            for nd in (f, f + 1):
                if nd not in dofs:
                    dofs.append(nd)
    idx = {nd: 3 * k for k, nd in enumerate(dofs)}
    out = np.zeros(3 * len(dofs))
    any_f = False
    for w_g, a, d_min, cands in samples:
        # f_n magnitude from the true min distance, geometry blended —
        # mirroring the contact pass.
        delta_c = d_min - beam_radius
        dE_dd = compute_dE_IMC_ddelta_c(delta_c, beam_radius, d_imc, K_imc)
        f_n_mag = abs(k_c * dE_dd)
        if f_n_mag < 1e-30:
            continue
        for w, f, t, Q, P, d in cands:
            v1 = (1.0 - a) * u_nodes[e] + a * u_nodes[e + 1]
            v2 = (1.0 - t) * u_nodes[f] + t * u_nodes[f + 1]
            v_rel = v1 - v2
            n_hat = (P - Q) / d
            tv = v_rel - _dot3(v_rel, n_hat) * n_hat
            tv_n = _norm3(tv)
            if tv_n < 1e-14:
                continue
            gate, _ = _friction_gate(tv_n, vel_tol, friction_law, nh_rho,
                                     nh_v_ref)
            f_tan = (w_g * w * mu * gate * f_n_mag) * (tv / tv_n)
            out[idx[e]:idx[e] + 3] += (1.0 - a) * f_tan
            out[idx[e + 1]:idx[e + 1] + 3] += a * f_tan
            out[idx[f]:idx[f] + 3] += -(1.0 - t) * f_tan
            out[idx[f + 1]:idx[f + 1] + 3] += -t * f_tan
            any_f = True
    if not any_f:
        return None
    return out, tuple(dofs)


# =============================================================================
# Beam-to-beam contact force computation (3D Eberly)
# =============================================================================

def compute_eberly_3d_beam_to_beam_contact_forces(
    robot: "SoftRobot",
    q: np.ndarray,
    beam_radius: float,
    k_c: float = 1.0,
    n_nodes_beam_A: int = None,
    d_imc: float = 0.0008,
    straight_R_threshold: float = None,
    h_scale: bool = False,
    backend: str = 'batched',
) -> Tuple[np.ndarray, np.ndarray, float, List[Dict]]:
    """
    Compute contact forces between two beams using Eberly's 3D arc distance.
    Beam A occupies nodes [0, n_nodes_beam_A), beam B occupies [n_nodes_beam_A, n_nodes).
    Operates directly on 3D node coordinates — no 2D projection.

    Parameters
    ----------
    robot : SoftRobot
    q : full DOF vector
    beam_radius : cross-section radius (same for both beams)
    k_c : contact stiffness multiplier
    n_nodes_beam_A : number of nodes in beam A
    d_imc : IMC regularization width

    Returns
    -------
    F : (ndof,) force vector
    J : (ndof, ndof) Jacobian
    gap_min : minimum gap
    all_gaps : list of gap info dicts
    """
    n_nodes = robot.n_nodes
    node_dof = robot.node_dof_indices
    ndof = q.shape[0]
    node_pos = q[node_dof]

    if n_nodes_beam_A is None:
        n_nodes_beam_A = n_nodes // 2

    n_nodes_B = n_nodes - n_nodes_beam_A
    num_seg_A = n_nodes_beam_A - 2
    num_seg_B = n_nodes_B - 2

    F = np.zeros(ndof)
    J = np.zeros((ndof, ndof))
    all_gaps = []

    if num_seg_A <= 0 or num_seg_B <= 0:
        return F, J, float('inf'), all_gaps

    K_imc = 15.0 / d_imc
    beam_R_eff = 2 * beam_radius
    interaction_range = beam_R_eff + d_imc

    # Broad phase (vectorized): bounding-sphere culling across the two beams.
    with _prof.phase('broad'):
        ii, jj = _broad_phase_pairs(node_pos,
                                    np.arange(num_seg_A),
                                    n_nodes_beam_A + np.arange(num_seg_B),
                                    interaction_range)
    _prof.count('contact_calls')
    _prof.count('pairs_candidate', ii.size)
    _prof.count('pairs_stencil', num_seg_A * num_seg_B)

    # The batched narrow phase covers the pure arc-to-arc path; the
    # straight-line fallback mode keeps the (already cheap) scalar path.
    bres = None
    if backend == 'batched' and straight_R_threshold is None and ii.size:
        from .eberly_batched import batched_pair_forces
        with _prof.phase('narrow'):
            bres = batched_pair_forces(
                node_pos, ii, n_nodes_beam_A + jj, beam_radius, d_imc, K_imc,
                ii == 0, ii == num_seg_A - 1,
                jj == 0, jj == num_seg_B - 1)

    for k in range(ii.size):
        i, j = int(ii[k]), int(jj[k])
        ni0, ni1, ni2 = i, i + 1, i + 2
        nj0 = n_nodes_beam_A + j
        nj1 = n_nodes_beam_A + j + 1
        nj2 = n_nodes_beam_A + j + 2
        flags = (i == 0, i == num_seg_A - 1,
                 j == 0, j == num_seg_B - 1)
        pair_tag = f'(A{i},B{j})'

        if bres is not None and not bres['fallback'][k]:
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

            R1 = R2 = None
            try:
                C1, R1, n1_ax = circle_from_three_nodes(*NODES1)
                C2, R2, n2_ax = circle_from_three_nodes(*NODES2)
                arc_fit_ok = True
            except (ValueError, np.linalg.LinAlgError):
                arc_fit_ok = False

            # Line fallback: if either arc is near-straight (R above the
            # caller-set threshold) OR the arc fit itself failed, use the
            # IMC edge-to-edge primitive instead of arc-to-arc. This is
            # activated only when `straight_R_threshold` is not None.
            use_line = (straight_R_threshold is not None) and (
                (not arc_fit_ok) or
                (R1 is not None and R1 > straight_R_threshold) or
                (R2 is not None and R2 > straight_R_threshold)
            )

            if use_line:
                with _prof.phase('narrow'):
                    energy, force_local, hess_local, gap, s1, s2 = \
                        _compute_pair_force_and_hessian_line(
                            NODES1, NODES2, beam_radius, d_imc, K_imc, *flags)
                pair_tag = f'(A{i},B{j})[line]'
            else:
                if not arc_fit_ok:
                    continue
                with _prof.phase('narrow'):
                    arc1 = arc_interval_from_nodes(*NODES1, C1, R1, n1_ax)
                    arc2 = arc_interval_from_nodes(*NODES2, C2, R2, n2_ax)
                    arc1_ext = extend_arc(arc1)
                    arc2_ext = extend_arc(arc2)
                    energy, force_local, hess_local, gap, s1, s2 = \
                        _compute_pair_force_and_hessian(
                            NODES1, NODES2, arc1_ext, arc2_ext,
                            beam_radius, d_imc, K_imc, *flags)

        if abs(energy) < 1e-30:
            continue

        with _prof.phase('assembly'):
            node_indices = [ni0, ni1, ni2, nj0, nj1, nj2]
            global_dof = np.concatenate([node_dof[ni] for ni in node_indices])

            F[global_dof] += force_local
            J[np.ix_(global_dof, global_dof)] += hess_local
        _prof.count('pairs_active')

        all_gaps.append({
            'segment_pair': pair_tag,
            'gap': gap, 's_i1': s1, 's_i2': s2,
        })

    # Optional 1/h amplification of the contact energy, mirroring IMCEnergy's
    # effective stiffness: IMCEnergy differentiates in coordinates normalised
    # by h (contact_energy.py:93), so for the same nominal k_c its PHYSICAL
    # force carries a 1/h factor and Eberly is ~1/h times softer without this.
    #
    # This path differentiates w.r.t. PHYSICAL coordinates, so multiplying the
    # contact energy by `scale` multiplies BOTH the force and the tangent by
    # `scale`.  IMCEnergy's `scale**2` on its Hessian is a change of variable,
    # NOT an extra stiffness factor -- copying it here leaves the tangent
    # inconsistent with the residual by 1/h, which stalls Newton (the
    # frictionless Zavarise soft-soft benchmark fails at first contact) and,
    # because `_converged` also accepts a small-displacement criterion, can
    # declare convergence away from equilibrium.
    #
    # Off by default so results stay comparable with runs predating this
    # option; enable per simulation with `env.eberly_h_scale = True`.
    scale = (1.0 / beam_radius) if h_scale else 1.0
    F *= k_c * scale
    J *= k_c * scale

    if pair_event_log.enabled:
        pair_event_log.record(('b2b', n_nodes_beam_A, n_nodes),
                              all_gaps, beam_radius)

    gap_min = min((g['gap'] for g in all_gaps), default=float('inf'))
    return F, J, gap_min, all_gaps


# =============================================================================
# Friction force computation (3D Eberly)
# =============================================================================

def compute_eberly_3d_self_friction_forces(
    robot: "SoftRobot",
    q: np.ndarray,
    u: np.ndarray,
    beam_radius: float,
    mu: float,
    vel_tol: float,
    k_c: float = 1.0,
    min_segment_separation: int = 6,
    d_imc: float = 0.0008,
    backend: str = 'batched',
) -> np.ndarray:
    """
    Coulomb friction forces for 3D Eberly self-contact.

    Mirrors the structure of `compute_eberly_3d_self_contact_forces` but
    returns a single (ndof,) friction force vector.  Sign convention matches
    `IMCFrictionEnergy.grad_friction`: the caller should subtract this from
    the Newton residual (`forces -= F_fric`).

    Parameters
    ----------
    u : (ndof,) velocity vector (state.u).
    mu, vel_tol : Coulomb friction parameters.
    k_c : same stiffness multiplier used by the contact-force pass — the
        normal-force magnitude inside this routine is scaled by k_c so the
        friction force is consistent with the regularized contact force.
    Other parameters as in `compute_eberly_3d_self_contact_forces`.
    """
    pairs = compute_eberly_3d_self_friction_pairs(
        robot, q, beam_radius,
        min_segment_separation=min_segment_separation, d_imc=d_imc)
    return compute_eberly_3d_self_friction_force(
        robot, q, u, pairs, beam_radius, mu, vel_tol, k_c=k_c, d_imc=d_imc,
        backend=backend)


def compute_eberly_3d_beam_to_beam_friction_forces(
    robot: "SoftRobot",
    q: np.ndarray,
    u: np.ndarray,
    beam_radius: float,
    mu: float,
    vel_tol: float,
    k_c: float = 1.0,
    n_nodes_beam_A: int = None,
    d_imc: float = 0.0008,
    friction_law: str = 'smooth_coulomb',
    nh_rho: float = 0.1,
    nh_v_ref: float = 1.0,
    backend: str = 'batched',
) -> np.ndarray:
    """
    Coulomb friction forces for 3D Eberly beam-to-beam contact.
    Same conventions as `compute_eberly_3d_self_friction_forces`.
    """
    pairs = compute_eberly_3d_beam_to_beam_friction_pairs(
        robot, q, beam_radius, n_nodes_beam_A=n_nodes_beam_A, d_imc=d_imc)
    return compute_eberly_3d_beam_to_beam_friction_force(
        robot, q, u, pairs, beam_radius, mu, vel_tol, k_c=k_c, d_imc=d_imc,
        friction_law=friction_law, nh_rho=nh_rho, nh_v_ref=nh_v_ref,
        backend=backend)


# =============================================================================
# Semi-implicit friction: setup (cache directions) + apply (rebuild force)
# =============================================================================

def _fd_dfn_dq(NODES1, NODES2, arc1_ext, arc2_ext,
               beam_radius, d_imc, K_imc, k_c, flags,
               eps: float = 1e-6) -> np.ndarray:
    """
    Central-difference Jacobian of the per-pair contact-force magnitude
    f_n_mag w.r.t. the 18 stacked local DOFs [N1.x, N1.y, N1.z,
    N1+1.x, ..., N2.z, N2+1.x, ..., N2+2.z].  Used by the semi-implicit
    friction setup to give Newton the dominant friction-Jacobian
    contribution and break chatter at the contact-engagement boundary.
    """
    df = np.zeros(18)
    NODES1_l = [np.asarray(n, float).copy() for n in NODES1]
    NODES2_l = [np.asarray(n, float).copy() for n in NODES2]
    for k in range(18):
        if k < 9:
            node_i, dim = k // 3, k % 3
            orig = NODES1_l[node_i][dim]
            NODES1_l[node_i][dim] = orig + eps
            f_p = _compute_pair_fn_magnitude(
                tuple(NODES1_l), NODES2, arc1_ext, arc2_ext,
                beam_radius, d_imc, K_imc, k_c, *flags)
            NODES1_l[node_i][dim] = orig - eps
            f_m = _compute_pair_fn_magnitude(
                tuple(NODES1_l), NODES2, arc1_ext, arc2_ext,
                beam_radius, d_imc, K_imc, k_c, *flags)
            NODES1_l[node_i][dim] = orig
        else:
            kk = k - 9
            node_i, dim = kk // 3, kk % 3
            orig = NODES2_l[node_i][dim]
            NODES2_l[node_i][dim] = orig + eps
            f_p = _compute_pair_fn_magnitude(
                NODES1, tuple(NODES2_l), arc1_ext, arc2_ext,
                beam_radius, d_imc, K_imc, k_c, *flags)
            NODES2_l[node_i][dim] = orig - eps
            f_m = _compute_pair_fn_magnitude(
                NODES1, tuple(NODES2_l), arc1_ext, arc2_ext,
                beam_radius, d_imc, K_imc, k_c, *flags)
            NODES2_l[node_i][dim] = orig
        df[k] = (f_p - f_m) / (2.0 * eps)
    return df


def compute_eberly_3d_beam_to_beam_friction_setup(
    robot: "SoftRobot",
    q: np.ndarray,
    u: np.ndarray,
    beam_radius: float,
    mu: float,
    vel_tol: float,
    k_c: float = 1.0,
    n_nodes_beam_A: int = None,
    d_imc: float = 0.0008,
) -> List[Dict]:
    """
    Semi-implicit friction SETUP for beam-to-beam contact.

    For each pair currently in (or near) contact, compute the tangent
    direction `t_hat` and the smooth Coulomb gate `gamma` from the
    start-of-timestep state (q, u).  Return a cache of per-pair entries
    that can be consumed by the matching apply() each Newton iteration.

    The pair list is FROZEN by this setup call: subsequent Newton iters
    only revisit these pairs.  Pairs that newly enter contact during the
    Newton solve are silently ignored (negligible friction since
    f_n -> 0 there).

    Returns
    -------
    cache : list of dicts
        Each entry has keys:
            global_dof  : (18,) int — DOF indices for assembly
            t_hat       : (3,) float — cached tangent direction
            gamma       : float — cached Coulomb gate value
            arc1_ext    : (2,) float — extended arc interval, arc 1
            arc2_ext    : (2,) float — extended arc interval, arc 2
            node_idxs   : tuple of 6 int — segment node indices in q
            flags       : 4-tuple of boundary flags
    """
    n_nodes = robot.n_nodes
    node_dof = robot.node_dof_indices
    node_pos = q[node_dof]

    if n_nodes_beam_A is None:
        n_nodes_beam_A = n_nodes // 2

    n_nodes_B = n_nodes - n_nodes_beam_A
    num_seg_A = n_nodes_beam_A - 2
    num_seg_B = n_nodes_B - 2

    cache: List[Dict] = []
    if num_seg_A <= 0 or num_seg_B <= 0:
        return cache

    beam_R_eff = 2 * beam_radius
    interaction_range = beam_R_eff + d_imc

    for i in range(num_seg_A):
        for j in range(num_seg_B):
            ni0, ni1, ni2 = i, i + 1, i + 2
            nj0 = n_nodes_beam_A + j
            nj1 = n_nodes_beam_A + j + 1
            nj2 = n_nodes_beam_A + j + 2

            center_i = (node_pos[ni0] + node_pos[ni2]) / 2
            center_j = (node_pos[nj0] + node_pos[nj2]) / 2
            r_i = _norm3(node_pos[ni0] - node_pos[ni2]) / 2
            r_j = _norm3(node_pos[nj0] - node_pos[nj2]) / 2
            if _norm3(center_j - center_i) > interaction_range + r_i + r_j:
                continue

            NODES1 = (node_pos[ni0], node_pos[ni1], node_pos[ni2])
            NODES2 = (node_pos[nj0], node_pos[nj1], node_pos[nj2])

            try:
                C1, R1, n1_ax = circle_from_three_nodes(*NODES1)
                C2, R2, n2_ax = circle_from_three_nodes(*NODES2)
            except (ValueError, np.linalg.LinAlgError):
                continue

            arc1 = arc_interval_from_nodes(*NODES1, C1, R1, n1_ax)
            arc2 = arc_interval_from_nodes(*NODES2, C2, R2, n2_ax)
            arc1_ext = extend_arc(arc1)
            arc2_ext = extend_arc(arc2)

            node_indices = [ni0, ni1, ni2, nj0, nj1, nj2]
            global_dof = np.concatenate([node_dof[ni] for ni in node_indices])
            U1 = u[global_dof[0:9]]
            U2 = u[global_dof[9:18]]

            result = _compute_pair_friction_unit_distribution(
                NODES1, NODES2, U1, U2, arc1_ext, arc2_ext,
                mu, vel_tol,
            )
            if result is None:
                continue
            unit_F_arc1, unit_F_arc2 = result

            flags = (i == 0, i == num_seg_A - 1,
                     j == 0, j == num_seg_B - 1)
            K_imc = 15.0 / d_imc
            df_n_dq = _fd_dfn_dq(
                NODES1, NODES2, arc1_ext, arc2_ext,
                beam_radius, d_imc, K_imc, k_c, flags,
            )
            unit_F_18 = np.concatenate([unit_F_arc1, unit_F_arc2])

            cache.append({
                'global_dof':  global_dof,
                'unit_F_arc1': unit_F_arc1,
                'unit_F_arc2': unit_F_arc2,
                'unit_F_18':   unit_F_18,
                'df_n_dq':     df_n_dq,
                'arc1_ext':    arc1_ext,
                'arc2_ext':    arc2_ext,
                'node_idxs':   (ni0, ni1, ni2, nj0, nj1, nj2),
                'flags':       flags,
            })

    return cache


def compute_eberly_3d_beam_to_beam_friction_apply(
    robot: "SoftRobot",
    q: np.ndarray,
    cache: List[Dict],
    beam_radius: float,
    k_c: float = 1.0,
    d_imc: float = 0.0008,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Semi-implicit friction APPLY for beam-to-beam contact.

    Returns (F_fric, J_fric) where both have ndof / (ndof, ndof) shape.
    J_fric is the dominant friction Jacobian contribution:
        J_fric_per_pair = outer(unit_F_18_cached, df_n_dq_cached)
    Caller does `forces -= F_fric; jacobian -= J_fric`.

    For each cached pair we recompute only the SCALAR contact-force
    magnitude |f_n(q_iter)| — the cached `unit_F_*` already bakes mu,
    gamma, t_hat, and dM/dN at first_iter geometry, and `df_n_dq` was
    FD'd once at first_iter.  Newton sees a smooth friction force whose
    Jacobian is exact-to-FD-precision under the cached-direction
    assumption.
    """
    ndof = q.shape[0]
    F_fric = np.zeros(ndof)
    J_fric = np.zeros((ndof, ndof))
    if not cache:
        return F_fric, J_fric

    node_dof = robot.node_dof_indices
    node_pos = q[node_dof]
    K_imc = 15.0 / d_imc

    for entry in cache:
        ni0, ni1, ni2, nj0, nj1, nj2 = entry['node_idxs']
        NODES1 = (node_pos[ni0], node_pos[ni1], node_pos[ni2])
        NODES2 = (node_pos[nj0], node_pos[nj1], node_pos[nj2])

        f_n_mag = _compute_pair_fn_magnitude(
            NODES1, NODES2, entry['arc1_ext'], entry['arc2_ext'],
            beam_radius, d_imc, K_imc, k_c, *entry['flags'],
        )
        if f_n_mag < 1e-30:
            continue

        gdof = entry['global_dof']
        F_fric[gdof[0:9]]  += f_n_mag * entry['unit_F_arc1']
        F_fric[gdof[9:18]] += f_n_mag * entry['unit_F_arc2']

        # Friction Jacobian contribution: rank-1 outer product.
        J_local = np.outer(entry['unit_F_18'], entry['df_n_dq'])     # (18, 18)
        J_fric[np.ix_(gdof, gdof)] += J_local

    return F_fric, J_fric


def compute_eberly_3d_self_friction_setup(
    robot: "SoftRobot",
    q: np.ndarray,
    u: np.ndarray,
    beam_radius: float,
    mu: float,
    vel_tol: float,
    k_c: float = 1.0,
    min_segment_separation: int = 6,
    d_imc: float = 0.0008,
) -> List[Dict]:
    """
    Semi-implicit friction SETUP for self-contact.  See
    `compute_eberly_3d_beam_to_beam_friction_setup` for semantics.
    """
    n_nodes = robot.n_nodes
    node_dof = robot.node_dof_indices
    node_pos = q[node_dof]

    num_segments = n_nodes - 2
    cache: List[Dict] = []
    if num_segments <= 1:
        return cache

    beam_R_eff = 2 * beam_radius
    interaction_range = beam_R_eff + d_imc

    for i in range(num_segments):
        for j in range(i + min_segment_separation, num_segments):
            center_i = (node_pos[i] + node_pos[i + 2]) / 2
            center_j = (node_pos[j] + node_pos[j + 2]) / 2
            r_i = _norm3(node_pos[i] - node_pos[i + 2]) / 2
            r_j = _norm3(node_pos[j] - node_pos[j + 2]) / 2
            if _norm3(center_j - center_i) > interaction_range + r_i + r_j:
                continue

            NODES1 = (node_pos[i], node_pos[i + 1], node_pos[i + 2])
            NODES2 = (node_pos[j], node_pos[j + 1], node_pos[j + 2])

            try:
                C1, R1, n1_ax = circle_from_three_nodes(*NODES1)
                C2, R2, n2_ax = circle_from_three_nodes(*NODES2)
            except (ValueError, np.linalg.LinAlgError):
                continue

            arc1 = arc_interval_from_nodes(*NODES1, C1, R1, n1_ax)
            arc2 = arc_interval_from_nodes(*NODES2, C2, R2, n2_ax)
            arc1_ext = extend_arc(arc1)
            arc2_ext = extend_arc(arc2)

            node_indices = [i, i + 1, i + 2, j, j + 1, j + 2]
            global_dof = np.concatenate([node_dof[ni] for ni in node_indices])
            U1 = u[global_dof[0:9]]
            U2 = u[global_dof[9:18]]

            result = _compute_pair_friction_unit_distribution(
                NODES1, NODES2, U1, U2, arc1_ext, arc2_ext,
                mu, vel_tol,
            )
            if result is None:
                continue
            unit_F_arc1, unit_F_arc2 = result

            flags = (i == 0, i == num_segments - 1,
                     j == 0, j == num_segments - 1)
            K_imc = 15.0 / d_imc
            df_n_dq = _fd_dfn_dq(
                NODES1, NODES2, arc1_ext, arc2_ext,
                beam_radius, d_imc, K_imc, k_c, flags,
            )
            unit_F_18 = np.concatenate([unit_F_arc1, unit_F_arc2])

            cache.append({
                'global_dof':  global_dof,
                'unit_F_arc1': unit_F_arc1,
                'unit_F_arc2': unit_F_arc2,
                'unit_F_18':   unit_F_18,
                'df_n_dq':     df_n_dq,
                'arc1_ext':    arc1_ext,
                'arc2_ext':    arc2_ext,
                'node_idxs':   (i, i + 1, i + 2, j, j + 1, j + 2),
                'flags':       flags,
            })

    return cache


def compute_eberly_3d_self_friction_apply(
    robot: "SoftRobot",
    q: np.ndarray,
    cache: List[Dict],
    beam_radius: float,
    k_c: float = 1.0,
    d_imc: float = 0.0008,
) -> Tuple[np.ndarray, np.ndarray]:
    """Semi-implicit friction APPLY for self-contact (twin of beam-to-beam).

    Returns (F_fric, J_fric).  Caller does
    `forces -= F_fric; jacobian -= J_fric`.
    """
    ndof = q.shape[0]
    F_fric = np.zeros(ndof)
    J_fric = np.zeros((ndof, ndof))
    if not cache:
        return F_fric, J_fric

    node_dof = robot.node_dof_indices
    node_pos = q[node_dof]
    K_imc = 15.0 / d_imc

    for entry in cache:
        ni0, ni1, ni2, nj0, nj1, nj2 = entry['node_idxs']
        NODES1 = (node_pos[ni0], node_pos[ni1], node_pos[ni2])
        NODES2 = (node_pos[nj0], node_pos[nj1], node_pos[nj2])

        f_n_mag = _compute_pair_fn_magnitude(
            NODES1, NODES2, entry['arc1_ext'], entry['arc2_ext'],
            beam_radius, d_imc, K_imc, k_c, *entry['flags'],
        )
        if f_n_mag < 1e-30:
            continue

        gdof = entry['global_dof']
        F_fric[gdof[0:9]]  += f_n_mag * entry['unit_F_arc1']
        F_fric[gdof[9:18]] += f_n_mag * entry['unit_F_arc2']

        J_local = np.outer(entry['unit_F_18'], entry['df_n_dq'])
        J_fric[np.ix_(gdof, gdof)] += J_local

    return F_fric, J_fric


# =============================================================================
# IMC-convention pair-cache friction (recommended scheme)
# =============================================================================
#
# Follows the IMCFrictionEnergy convention:
#   1. The pair list is frozen at the first Newton iteration of each dt-step
#      (matching IMC's `self.ind = self._ind[first_Delta < ...]` filter).
#   2. The friction force is fully recomputed at every Newton iteration from
#      the in-iter velocity `u`; no friction Jacobian by default (matching
#      IMC).  Because the tangent direction adapts smoothly with q as Newton
#      iterates, this scheme avoids the chattering seen with fully frozen
#      directions, at the price of an unsymmetric residual contribution.

def compute_eberly_3d_beam_to_beam_friction_pairs(
    robot: "SoftRobot",
    q: np.ndarray,
    beam_radius: float,
    n_nodes_beam_A: int = None,
    d_imc: float = 0.0008,
) -> List[Dict]:
    """
    Identify the active pair list at first_iter via broad-phase culling.
    The list is reused unchanged for the rest of the dt-step's Newton solve.
    """
    n_nodes = robot.n_nodes
    node_dof = robot.node_dof_indices
    node_pos = q[node_dof]

    if n_nodes_beam_A is None:
        n_nodes_beam_A = n_nodes // 2

    n_nodes_B = n_nodes - n_nodes_beam_A
    num_seg_A = n_nodes_beam_A - 2
    num_seg_B = n_nodes_B - 2

    pairs: List[Dict] = []
    if num_seg_A <= 0 or num_seg_B <= 0:
        return pairs

    beam_R_eff = 2 * beam_radius
    interaction_range = beam_R_eff + d_imc

    for i in range(num_seg_A):
        for j in range(num_seg_B):
            ni0, ni1, ni2 = i, i + 1, i + 2
            nj0 = n_nodes_beam_A + j
            nj1 = n_nodes_beam_A + j + 1
            nj2 = n_nodes_beam_A + j + 2

            center_i = (node_pos[ni0] + node_pos[ni2]) / 2
            center_j = (node_pos[nj0] + node_pos[nj2]) / 2
            r_i = _norm3(node_pos[ni0] - node_pos[ni2]) / 2
            r_j = _norm3(node_pos[nj0] - node_pos[nj2]) / 2
            if _norm3(center_j - center_i) > interaction_range + r_i + r_j:
                continue

            node_indices = [ni0, ni1, ni2, nj0, nj1, nj2]
            global_dof = np.concatenate([node_dof[ni] for ni in node_indices])
            pairs.append({
                'node_idxs':  (ni0, ni1, ni2, nj0, nj1, nj2),
                'global_dof': global_dof,
                'flags':      (i == 0, i == num_seg_A - 1,
                               j == 0, j == num_seg_B - 1),
            })

    return pairs


def compute_eberly_3d_beam_to_beam_friction_force(
    robot: "SoftRobot",
    q: np.ndarray,
    u: np.ndarray,
    pairs: List[Dict],
    beam_radius: float,
    mu: float,
    vel_tol: float,
    k_c: float = 1.0,
    d_imc: float = 0.0008,
    friction_law: str = 'smooth_coulomb',
    nh_rho: float = 0.1,
    nh_v_ref: float = 1.0,
    backend: str = 'batched',
) -> np.ndarray:
    """
    Compute friction force at the in-iter (q, u) over a FIXED pair list.
    Mimics IMCFrictionEnergy.grad_friction: every-Newton-iter recompute.
    Caller does `forces -= F`.
    """
    ndof = q.shape[0]
    F_fric = np.zeros(ndof)
    if not pairs:
        return F_fric

    node_dof = robot.node_dof_indices
    node_pos = q[node_dof]
    K_imc = 15.0 / d_imc

    bres = None
    if backend == 'batched':
        from .eberly_batched import batched_pair_friction_forces
        u_nodes = u[node_dof]
        arr = np.array([e['node_idxs'] for e in pairs], dtype=int)
        flg = np.array([e['flags'] for e in pairs], dtype=bool)
        bres = batched_pair_friction_forces(
            node_pos, u_nodes, arr[:, 0], arr[:, 3],
            beam_radius, d_imc, K_imc, k_c, mu, vel_tol,
            flg[:, 0], flg[:, 1], flg[:, 2], flg[:, 3],
            friction_law=friction_law, nh_rho=nh_rho, nh_v_ref=nh_v_ref)

    for k, entry in enumerate(pairs):
        gdof = entry['global_dof']
        if bres is not None and not bres['fallback'][k]:
            F_fric[gdof] += bres['force'][k]
            continue

        ni0, ni1, ni2, nj0, nj1, nj2 = entry['node_idxs']
        NODES1 = (node_pos[ni0], node_pos[ni1], node_pos[ni2])
        NODES2 = (node_pos[nj0], node_pos[nj1], node_pos[nj2])

        try:
            C1, R1, n1_ax = circle_from_three_nodes(*NODES1)
            C2, R2, n2_ax = circle_from_three_nodes(*NODES2)
        except (ValueError, np.linalg.LinAlgError):
            continue

        arc1 = arc_interval_from_nodes(*NODES1, C1, R1, n1_ax)
        arc2 = arc_interval_from_nodes(*NODES2, C2, R2, n2_ax)
        arc1_ext = extend_arc(arc1)
        arc2_ext = extend_arc(arc2)

        U1 = u[gdof[0:9]]
        U2 = u[gdof[9:18]]

        f_local = _compute_pair_friction(
            NODES1, NODES2, U1, U2, arc1_ext, arc2_ext,
            beam_radius, d_imc, K_imc, k_c, mu, vel_tol,
            *entry['flags'],
            friction_law=friction_law, nh_rho=nh_rho, nh_v_ref=nh_v_ref,
        )
        F_fric[gdof] += f_local

    return F_fric


def compute_eberly_3d_beam_to_beam_friction_force_with_jac(
    robot: "SoftRobot",
    q: np.ndarray,
    u: np.ndarray,
    pairs: List[Dict],
    beam_radius: float,
    mu: float,
    vel_tol: float,
    dt: float,
    k_c: float = 1.0,
    d_imc: float = 0.0008,
    include_term2: bool = True,
    friction_law: str = 'smooth_coulomb',
    nh_rho: float = 0.1,
    nh_v_ref: float = 1.0,
):
    """
    Beam-to-beam Eberly friction with full friction Jacobian:

        J_fric = J_term1 + J_term2
        J_term1 = (∂F_fric/∂U) · (1/dt)              (velocity dependence)
        J_term2 = outer(F_fric / f_n_mag, ∂f_n_mag/∂q)  (contact-force-magnitude)

    Term 2 uses ∂f_n_mag/∂q from _fd_dfn_dq (central differences per pair).
    Set `include_term2=False` to recover the Term-1-only path.

    Caller convention: `forces -= F; jacobian -= J`.
    """
    ndof = q.shape[0]
    F_fric = np.zeros(ndof)
    J_fric = np.zeros((ndof, ndof))
    if not pairs:
        return F_fric, J_fric

    node_dof = robot.node_dof_indices
    node_pos = q[node_dof]
    K_imc = 15.0 / d_imc

    inv_dt = 1.0 / dt

    for entry in pairs:
        ni0, ni1, ni2, nj0, nj1, nj2 = entry['node_idxs']
        NODES1 = (node_pos[ni0], node_pos[ni1], node_pos[ni2])
        NODES2 = (node_pos[nj0], node_pos[nj1], node_pos[nj2])

        try:
            C1, R1, n1_ax = circle_from_three_nodes(*NODES1)
            C2, R2, n2_ax = circle_from_three_nodes(*NODES2)
        except (ValueError, np.linalg.LinAlgError):
            continue

        arc1 = arc_interval_from_nodes(*NODES1, C1, R1, n1_ax)
        arc2 = arc_interval_from_nodes(*NODES2, C2, R2, n2_ax)
        arc1_ext = extend_arc(arc1)
        arc2_ext = extend_arc(arc2)

        gdof = entry['global_dof']
        U1 = u[gdof[0:9]]
        U2 = u[gdof[9:18]]

        f_local, j_term1, f_n_mag, _, _ = _compute_pair_friction_with_velocity_jacobian(
            NODES1, NODES2, U1, U2,
            arc1_ext, arc2_ext,
            beam_radius, d_imc, K_imc, k_c, mu, vel_tol,
            *entry['flags'],
            friction_law=friction_law, nh_rho=nh_rho, nh_v_ref=nh_v_ref,
        )

        F_fric[gdof] += f_local
        # Term 1 only: velocity-Jacobian × 1/dt.  Term 2
        # (∂F_fric/∂f_n_mag · ∂f_n_mag/∂q) is intentionally NOT added: FD
        # comparison showed the full ∂F_fric/∂q is dominated by geometric
        # terms (∂t_hat/∂q, ∂weights/∂q, ∂n_hat/∂q), and adding Term 2
        # alone -- without those -- degrades Newton convergence.  Term 2 is
        # only consistent under the frozen-direction approximation used by
        # the setup/apply scheme.
        j_pair = j_term1 * inv_dt

        J_fric[gdof[:, None], gdof[None, :]] += j_pair

    return F_fric, J_fric


def compute_eberly_3d_self_friction_pairs(
    robot: "SoftRobot",
    q: np.ndarray,
    beam_radius: float,
    min_segment_separation: int = 6,
    d_imc: float = 0.0008,
) -> List[Dict]:
    """Self-contact pair list at first_iter (twin of beam-to-beam version)."""
    n_nodes = robot.n_nodes
    node_dof = robot.node_dof_indices
    node_pos = q[node_dof]

    num_segments = n_nodes - 2
    pairs: List[Dict] = []
    if num_segments <= 1:
        return pairs

    beam_R_eff = 2 * beam_radius
    interaction_range = beam_R_eff + d_imc

    for i in range(num_segments):
        for j in range(i + min_segment_separation, num_segments):
            center_i = (node_pos[i] + node_pos[i + 2]) / 2
            center_j = (node_pos[j] + node_pos[j + 2]) / 2
            r_i = _norm3(node_pos[i] - node_pos[i + 2]) / 2
            r_j = _norm3(node_pos[j] - node_pos[j + 2]) / 2
            if _norm3(center_j - center_i) > interaction_range + r_i + r_j:
                continue

            node_indices = [i, i + 1, i + 2, j, j + 1, j + 2]
            global_dof = np.concatenate([node_dof[ni] for ni in node_indices])
            pairs.append({
                'node_idxs':  (i, i + 1, i + 2, j, j + 1, j + 2),
                'global_dof': global_dof,
                'flags':      (i == 0, i == num_segments - 1,
                               j == 0, j == num_segments - 1),
            })

    return pairs


def compute_eberly_3d_self_friction_force(
    robot: "SoftRobot",
    q: np.ndarray,
    u: np.ndarray,
    pairs: List[Dict],
    beam_radius: float,
    mu: float,
    vel_tol: float,
    k_c: float = 1.0,
    d_imc: float = 0.0008,
    friction_law: str = 'smooth_coulomb',
    nh_rho: float = 0.1,
    nh_v_ref: float = 1.0,
    backend: str = 'batched',
    straight_R_threshold: float = None,
    conforming_edge_blend: bool = False,
) -> np.ndarray:
    """Self-contact friction at the in-iter (q, u) over a FIXED pair list.

    `straight_R_threshold` mirrors the contact pass: pairs whose 3-node
    window is near-straight (fitted arc radius above the threshold, or a
    degenerate collinear fit) use the line primitive geometry
    (`_compute_pair_friction_line`) so friction stays consistent with the
    normal force computed on the same primitive.

    `conforming_edge_blend` mirrors the contact pass's crossfade to the
    two-node edge model; gamma is recomputed here from the same smooth
    chord-gap field (cheap), so friction needs no state from the contact
    pass."""
    ndof = q.shape[0]
    F_fric = np.zeros(ndof)
    if not pairs:
        return F_fric

    node_dof = robot.node_dof_indices
    node_pos = q[node_dof]
    K_imc = 15.0 / d_imc

    arr = np.array([e['node_idxs'] for e in pairs], dtype=int)

    # Same smooth arc→line blend as the contact pass, so friction stays
    # consistent with the normal force computed on the same primitives.
    blend_w = _pair_line_weights(arr[:, 0], arr[:, 3], node_pos,
                                 straight_R_threshold)

    gammas = gamma_edge = edge_partners = None
    if conforming_edge_blend and len(pairs):
        gammas, gamma_edge, edge_partners = _conformity_edge_field(
            node_pos, arr[:, 0], arr[:, 3], beam_radius, d_imc)
    if gamma_edge:
        u_nodes_c = u[node_dof]
        for e, g_e in sorted(gamma_edge.items()):
            out_e = _edge_sample_friction(
                node_pos, u_nodes_c, e, edge_partners[e],
                beam_radius, d_imc, K_imc, k_c, mu, vel_tol,
                friction_law=friction_law, nh_rho=nh_rho, nh_v_ref=nh_v_ref)
            if out_e is None:
                continue
            Fe, nds = out_e
            dof_e = np.concatenate([node_dof[nd] for nd in nds])
            F_fric[dof_e] += (0.5 * g_e) * Fe

    bres = None
    arc_pos = None
    if backend == 'batched':
        from .eberly_batched import batched_pair_friction_forces
        u_nodes = u[node_dof]
        flg = np.array([e['flags'] for e in pairs], dtype=bool)
        if blend_w is not None:
            arc_sel = np.nonzero(blend_w < 1.0)[0]
        else:
            arc_sel = np.arange(len(pairs))
        if arc_sel.size:
            bres = batched_pair_friction_forces(
                node_pos, u_nodes, arr[arc_sel, 0], arr[arc_sel, 3],
                beam_radius, d_imc, K_imc, k_c, mu, vel_tol,
                flg[arc_sel, 0], flg[arc_sel, 1],
                flg[arc_sel, 2], flg[arc_sel, 3],
                friction_law=friction_law, nh_rho=nh_rho, nh_v_ref=nh_v_ref)
            arc_pos = np.full(len(pairs), -1)
            arc_pos[arc_sel] = np.arange(arc_sel.size)

    for k, entry in enumerate(pairs):
        w_line = float(blend_w[k]) if blend_w is not None else 0.0
        gamma_k = float(gammas[k]) if gammas is not None else 0.0
        ni0, ni1, ni2, nj0, nj1, nj2 = entry['node_idxs']
        NODES1 = (node_pos[ni0], node_pos[ni1], node_pos[ni2])
        NODES2 = (node_pos[nj0], node_pos[nj1], node_pos[nj2])
        gdof = entry['global_dof']
        U1 = u[gdof[0:9]]
        U2 = u[gdof[9:18]]

        f_local = np.zeros(18)

        if w_line > 0.0:
            f_local += w_line * _compute_pair_friction_line(
                NODES1, NODES2, U1, U2,
                beam_radius, d_imc, K_imc, k_c, mu, vel_tol,
                *entry['flags'],
                friction_law=friction_law, nh_rho=nh_rho, nh_v_ref=nh_v_ref,
            )

        if w_line < 1.0:
            kb = arc_pos[k] if arc_pos is not None else k
            if bres is not None and kb >= 0 and not bres['fallback'][kb]:
                f_local += (1.0 - w_line) * bres['force'][kb]
            else:
                f_arc = None
                try:
                    C1, R1, n1_ax = circle_from_three_nodes(*NODES1)
                    C2, R2, n2_ax = circle_from_three_nodes(*NODES2)
                except (ValueError, np.linalg.LinAlgError):
                    pass
                else:
                    arc1_ext = extend_arc(
                        arc_interval_from_nodes(*NODES1, C1, R1, n1_ax))
                    arc2_ext = extend_arc(
                        arc_interval_from_nodes(*NODES2, C2, R2, n2_ax))
                    f_arc = _compute_pair_friction(
                        NODES1, NODES2, U1, U2, arc1_ext, arc2_ext,
                        beam_radius, d_imc, K_imc, k_c, mu, vel_tol,
                        *entry['flags'],
                        friction_law=friction_law, nh_rho=nh_rho,
                        nh_v_ref=nh_v_ref,
                    )
                if f_arc is not None:
                    f_local += (1.0 - w_line) * f_arc

        F_fric[gdof] += (1.0 - gamma_k) * f_local

    return F_fric
