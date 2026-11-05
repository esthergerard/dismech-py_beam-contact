"""
Batched (structure-of-arrays) narrow phase for the Eberly arc contact.

Evaluates the full per-pair pipeline of ``eberly_contact_3d`` — arc fit,
canonical transform, degree-8 polynomial stationary points, edge/corner
candidates, argmin, IMC energy, envelope gradient and Gauss-Newton Hessian —
for ALL broad-phase survivors at once as NumPy array operations, instead of
one Python call chain per pair.

Semantics contract
------------------
This module reproduces the scalar path step for step (same candidate
enumeration ORDER, same tolerances, same tie-breaking through first-minimum
argmin), so results match the scalar kernel to floating-point reordering
(~1e-12 relative).  Pairs that hit a special-case branch that is not
batched are flagged in the returned ``fallback`` mask and MUST be handled
by the caller with the original scalar code path:

* degenerate arc fit (collinear triplet),
* parallel-plane circles (``full_circle_parallel_planes`` branch),
* ``P7(cos theta) ~ 0`` root branch,
* on-axis (equidistant) interior candidate.

The scalar implementation in ``eberly_contact_3d`` remains the oracle; see
``tests/test_eberly_contact.py`` for the equivalence tests.
"""

import numpy as np

from .circle_four_nodes_imc_weighted import (
    computeE_IMC,
    compute_dE_IMC_ddelta_c,
    compute_d2E_IMC_ddelta_c,
    compute_weights_from_s,
)

TWO_PI = 2.0 * np.pi


# =============================================================================
# Small batched linear-algebra helpers
# =============================================================================

def _bdot(a, b):
    """Row-wise dot product of (..., 3) arrays -> (...)."""
    return (a * b).sum(axis=-1)


def _bcross(a, b):
    """Row-wise cross product of (..., 3) arrays."""
    return np.stack([
        a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
        a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
        a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
    ], axis=-1)


def _bnorm(a):
    return np.sqrt((a * a).sum(axis=-1))


def _binv3(A):
    """Batched closed-form inverse of (B, 3, 3) matrices (adjugate/det)."""
    a, b, c = A[:, 0, 0], A[:, 0, 1], A[:, 0, 2]
    d, e, f = A[:, 1, 0], A[:, 1, 1], A[:, 1, 2]
    g, h, i = A[:, 2, 0], A[:, 2, 1], A[:, 2, 2]
    co_a = e * i - f * h
    co_b = f * g - d * i
    co_c = d * h - e * g
    det = a * co_a + b * co_b + c * co_c
    out = np.empty_like(A)
    out[:, 0, 0] = co_a; out[:, 0, 1] = c * h - b * i; out[:, 0, 2] = b * f - c * e
    out[:, 1, 0] = co_b; out[:, 1, 1] = a * i - c * g; out[:, 1, 2] = c * d - a * f
    out[:, 2, 0] = co_c; out[:, 2, 1] = b * g - a * h; out[:, 2, 2] = a * e - b * d
    return out / det[:, None, None]


def _bskew(v):
    """Batched skew-symmetric matrices: (B, 3) -> (B, 3, 3)."""
    B = v.shape[0]
    S = np.zeros((B, 3, 3))
    S[:, 0, 1] = -v[:, 2]; S[:, 0, 2] = v[:, 1]
    S[:, 1, 0] = v[:, 2];  S[:, 1, 2] = -v[:, 0]
    S[:, 2, 0] = -v[:, 1]; S[:, 2, 1] = v[:, 0]
    return S


def _bconv(a, b):
    """Batched polynomial product: (B, na) x (B, nb) -> (B, na+nb-1),
    coefficients low-order first (same contract as np.convolve)."""
    Bn, na = a.shape
    nb = b.shape[1]
    out = np.zeros((Bn, na + nb - 1))
    for i in range(na):
        out[:, i:i + nb] += a[:, i:i + 1] * b
    return out


def _bhorner(cs, coeffs):
    """Batched Horner evaluation.  cs: (B, S); coeffs: (B, K) low-first.
    Evaluation order matches numpy.polynomial.polynomial.polyval."""
    acc = np.broadcast_to(coeffs[:, -1:], cs.shape).copy()
    for k in range(coeffs.shape[1] - 2, -1, -1):
        acc = acc * cs + coeffs[:, k:k + 1]
    return acc


# =============================================================================
# Batched geometry pipeline
# =============================================================================

def _bcircles(nodes):
    """Batched ``circle_from_three_nodes``.

    nodes : (B, 3, 3) stacked (n0, n1, n2) rows.
    Returns (C, R, n_axis, ok) — ``ok`` False where the fit is degenerate.
    """
    n0, n1, n2 = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    e_a = n1 - n0
    e_b = n2 - n1
    cross = _bcross(e_a, e_b)
    cross_n2 = _bdot(cross, cross)
    ok = cross_n2 >= 1e-24
    safe = np.where(ok, cross_n2, 1.0)
    n_axis = cross / np.sqrt(safe)[:, None]

    m_a = 0.5 * (n0 + n1)
    m_b = 0.5 * (n1 + n2)
    A = np.stack([e_a, e_b, n_axis], axis=1)              # (B, 3, 3)
    rhs = np.stack([_bdot(e_a, m_a), _bdot(e_b, m_b),
                    _bdot(n_axis, n1)], axis=1)           # (B, 3)
    with np.errstate(divide='ignore', invalid='ignore'):
        A_inv = _binv3(A)
    C = np.einsum('bij,bj->bi', A_inv, rhs)
    R = _bnorm(C - m_a)

    # Orient the normal with the BendEnergy kb convention.
    te = e_a / _bnorm(e_a)[:, None]
    tf = e_b / _bnorm(e_b)[:, None]
    kb = 2.0 * _bcross(te, tf) / (1.0 + _bdot(te, tf))[:, None]
    flip = _bdot(n_axis, kb) < 0.0
    n_axis = np.where(flip[:, None], -n_axis, n_axis)
    return C, R, n_axis, ok, A_inv, flip


def _bplane_frames(n):
    """Batched ``build_plane_frame``: (B, 3) unit normals -> (e1, e2)."""
    n = n / _bnorm(n)[:, None]
    ex = np.array([1.0, 0.0, 0.0])
    ey = np.array([0.0, 1.0, 0.0])
    t = np.where((np.abs(n[:, 0]) < 0.9)[:, None], ex, ey)
    e1 = _bcross(n, t)
    e1 = e1 / _bnorm(e1)[:, None]
    e2 = _bcross(n, e1)
    e2 = e2 / _bnorm(e2)[:, None]
    return e1, e2


def _bangle(M, C, e1, e2):
    """Batched ``recover_angle``.  M may be (B, 3) or (B, S, 3)."""
    v = M - (C if M.ndim == 2 else C[:, None, :])
    if M.ndim == 2:
        return np.arctan2(_bdot(v, e2), _bdot(v, e1))
    return np.arctan2(_bdot(v, e2[:, None, :]), _bdot(v, e1[:, None, :]))


def _bin_arc(theta, arc, tol=1e-8):
    """Batched ``angle_in_arc``.  theta: (B,) or (B, S); arc: (B, 2)."""
    alpha = arc[:, 0] if theta.ndim == 1 else arc[:, 0:1]
    beta = arc[:, 1] if theta.ndim == 1 else arc[:, 1:2]
    span = (beta - alpha) % TWO_PI
    degenerate = (span < tol) | (span > TWO_PI - tol)
    inside = ((theta - alpha) % TWO_PI) <= span + tol
    return degenerate | inside


def _barcs(nodes, C, R, n_axis, e1, e2):
    """Batched ``arc_interval_from_nodes`` + ``extend_arc`` (factor 0.5)."""
    n0, n1, n2 = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    m_a = 0.5 * (n0 + n1)
    m_b = 0.5 * (n1 + n2)
    alpha = _bangle(m_a, C, e1, e2)
    beta = _bangle(m_b, C, e1, e2)
    e_a = n1 - n0
    tau = R[:, None] * (-np.sin(alpha)[:, None] * e1 + np.cos(alpha)[:, None] * e2)
    swap = _bdot(tau, e_a) < 0
    a = np.where(swap, beta, alpha)
    b = np.where(swap, alpha, beta)
    span = (b - a) % TWO_PI
    ext = 0.5 * span
    return np.stack([a - ext, b + ext], axis=1)


def _brotation_to_z(n):
    """Batched ``rotation_to_z`` including the aligned special cases."""
    B = n.shape[0]
    n = n / _bnorm(n)[:, None]
    aligned = (np.abs(n[:, 0]) <= 1e-8) & (np.abs(n[:, 1]) <= 1e-8)
    near_pos = aligned & (np.abs(n[:, 2] - 1.0) <= 1e-8 + 1e-5)
    near_neg = aligned & (np.abs(n[:, 2] + 1.0) <= 1e-8 + 1e-5)

    v = np.stack([n[:, 1], -n[:, 0], np.zeros(B)], axis=1)
    s2 = v[:, 0] ** 2 + v[:, 1] ** 2
    c = n[:, 2]
    Vx = _bskew(v)
    s2_safe = np.where(s2 > 0, s2, 1.0)
    Rm = (np.eye(3)[None] + Vx
          + np.einsum('bij,bjk->bik', Vx, Vx)
          * ((1.0 - c) / s2_safe)[:, None, None])
    Rm[near_pos] = np.eye(3)
    Rm[near_neg] = np.diag([1.0, -1.0, -1.0])
    return Rm


def _bprepare(Ca, Ra, na, Cb, Rb, nb):
    """Batched ``eberly_prepare``.  Returns canonical quantities plus the
    parallel-plane fallback mask."""
    swapped = Ra > Rb
    sw = swapped[:, None]
    C0 = np.where(sw, Cb, Ca); R0 = np.where(swapped, Rb, Ra)
    n0 = np.where(sw, nb, na)
    Cl = np.where(sw, Ca, Cb); Rl = np.where(swapped, Ra, Rb)
    nl = np.where(sw, na, nb)

    n0 = np.where((n0[:, 2] < 0)[:, None], -n0, n0)
    nl = np.where((nl[:, 2] < 0)[:, None], -nl, nl)

    t = -Cl
    scale = 1.0 / Rl
    R_rot = _brotation_to_z(nl)
    C0_can = scale[:, None] * np.einsum('bij,bj->bi', R_rot, C0 + t)
    N0_can = np.einsum('bij,bj->bi', R_rot, n0)
    r0_can = R0 * scale
    R_total = R_rot.copy()

    m2 = np.abs(C0_can[:, 0]) > 1e-14
    L = np.sqrt(C0_can[:, 0] ** 2 + C0_can[:, 1] ** 2)
    L_safe = np.where(L > 0, L, 1.0)
    sr = C0_can[:, 0] / L_safe
    cr = C0_can[:, 1] / L_safe
    rot1 = np.zeros((C0_can.shape[0], 3, 3))
    rot1[:, 0, 0] = cr; rot1[:, 0, 1] = -sr
    rot1[:, 1, 0] = sr; rot1[:, 1, 1] = cr
    rot1[:, 2, 2] = 1.0
    C0_rot = np.einsum('bij,bj->bi', rot1, C0_can)
    C0_rot[:, 0] = 0.0
    N0_rot = np.einsum('bij,bj->bi', rot1, N0_can)
    Rt_rot = np.einsum('bij,bjk->bik', rot1, R_total)
    C0_can = np.where(m2[:, None], C0_rot, C0_can)
    N0_can = np.where(m2[:, None], N0_rot, N0_can)
    R_total = np.where(m2[:, None, None], Rt_rot, R_total)

    parallel = N0_can[:, 2] >= 1.0 - 1e-10
    return C0_can, N0_can, r0_can, R_total, t, scale, swapped, parallel


def _bphi(C0, N0, r0):
    """Batched ``build_phi``: phi (B, 9), P6 (B, 5), P7 (B, 4)."""
    B = C0.shape[0]
    D = -C0
    N0xD = _bcross(N0, D)
    N0xU1 = np.stack([np.zeros(B), N0[:, 2], -N0[:, 1]], axis=1)
    N0xV1 = np.stack([-N0[:, 2], np.zeros(B), N0[:, 0]], axis=1)

    A0 = D[:, 0]
    A1 = D[:, 1]
    A2 = _bdot(N0xD, N0xD)
    A3 = _bdot(N0xD, N0xU1)
    A4 = _bdot(N0xD, N0xV1)
    A5 = _bdot(N0xU1, N0xU1)
    A6 = _bdot(N0xU1, N0xV1)
    A7 = _bdot(N0xV1, N0xV1)
    z = np.zeros(B)

    P0 = np.stack([A2 + A7, 2 * A3, A5 - A7], axis=1)
    P1 = np.stack([2 * A4, 2 * A6], axis=1)
    P2 = np.stack([z, A1], axis=1)
    P3 = (-A0)[:, None]
    P4 = np.stack([-A6, A4, 2 * A6], axis=1)
    P5 = np.stack([-A3, A7 - A5], axis=1)
    T0 = np.broadcast_to(np.array([1.0, 0.0, -1.0]), (B, 3)).copy()

    T1 = _bconv(P2, P2)
    T1p = _bconv(T0, _bconv(P3, P3))
    T1 = _padd_b(T1, T1p)
    T2 = 2.0 * _bconv(P2, P3)
    T3 = _padd_b(_bconv(P4, P4), _bconv(T0, _bconv(P5, P5)))
    T4 = 2.0 * _bconv(P4, P5)

    r0sq = (r0 ** 2)[:, None]
    P6 = _psub_b(_padd_b(_bconv(P0, T1), _bconv(_bconv(T0, P1), T2)), r0sq * T3)
    P7 = _psub_b(_padd_b(_bconv(P0, T2), _bconv(P1, T1)), r0sq * T4)
    phi = _psub_b(_bconv(P6, P6), _bconv(T0, _bconv(P7, P7)))
    return phi, P6, P7


def _padd_b(a, b):
    if a.shape[1] < b.shape[1]:
        a, b = b, a
    out = a.copy()
    out[:, :b.shape[1]] += b
    return out


def _psub_b(a, b):
    n = max(a.shape[1], b.shape[1])
    out = np.zeros((a.shape[0], n))
    out[:, :a.shape[1]] = a
    out[:, :b.shape[1]] -= b
    return out


def _broots8(phi):
    """Batched degree-8 polyroots: rotated companion matrix + eigvals + sort,
    replicating numpy.polynomial.polynomial.polyroots.  Returns
    (roots (B, 8) complex, fallback mask for non-degree-8 polynomials)."""
    B = phi.shape[0]
    lead = phi[:, 8]
    # Non-degree-8 polynomials AND non-finite coefficient rows (degenerate
    # geometry upstream, e.g. collinear windows) go to the scalar fallback;
    # eigvals hard-rejects NaN/inf, so those rows get a placeholder matrix.
    finite = np.isfinite(phi).all(axis=1)
    fallback = (lead == 0.0) | ~finite
    lead_safe = np.where(fallback, 1.0, lead)
    comp = np.zeros((B, 8, 8))
    idx = np.arange(7)
    comp[:, idx + 1, idx] = 1.0                       # sub-diagonal ones
    comp[:, :, 7] -= np.where(fallback[:, None], 0.0,
                              phi[:, :8] / lead_safe[:, None])
    comp = comp[:, ::-1, ::-1]                        # rotated companion
    roots = np.linalg.eigvals(comp)
    roots = np.sort(roots, axis=1)
    return roots, fallback


# =============================================================================
# Batched narrow phase: minimum distance between the two extended arcs
# =============================================================================

def batched_arc_min_dist(nodes1, nodes2):
    """Batched ``arc_min_dist`` over (B, 3, 3) node windows.

    Returns dict with per-pair arrays:
        dist, theta1, theta2, M1, M2, arc1, arc2,
        C1, R1, n1, C2, R2, n2, A_inv1, A_inv2, flip1, flip2,
        fallback (bool)  — pairs the caller must recompute with the
                           scalar path (special-case branches).
    """
    B = nodes1.shape[0]
    C1, R1, n1, ok1, A_inv1, flip1 = _bcircles(nodes1)
    C2, R2, n2, ok2, A_inv2, flip2 = _bcircles(nodes2)
    fallback = ~(ok1 & ok2)

    ef1, ef2 = _bplane_frames(n1)
    eg1, eg2 = _bplane_frames(n2)
    with np.errstate(invalid='ignore'):
        arc1 = _barcs(nodes1, C1, R1, n1, ef1, ef2)
        arc2 = _barcs(nodes2, C2, R2, n2, eg1, eg2)

    # ---- canonical transform + phi + roots (interior candidates) ----
    with np.errstate(invalid='ignore', divide='ignore'):
        C0c, N0c, r0c, Rt, t, scale, swapped, parallel = _bprepare(
            C1, R1, n1, C2, R2, n2)
        fallback |= parallel
        phi, P6, P7 = _bphi(C0c, N0c, r0c)
        roots, deg_fb = _broots8(phi)
        fallback |= deg_fb

        # 8 interior slots
        real_ok = np.abs(roots.imag) <= 1e-6
        cs = np.clip(roots.real, -1.0, 1.0)                     # (B, 8)
        p7v = _bhorner(cs, P7)
        p7_small = real_ok & (np.abs(p7v) <= 1e-10)
        fallback |= p7_small.any(axis=1)
        p7_safe = np.where(np.abs(p7v) > 0, p7v, 1.0)
        sn = -_bhorner(cs, P6) / p7_safe

        # closest point on canonical circle 0 for each (cs, sn)
        M1c = np.stack([cs, sn, np.zeros_like(cs)], axis=2)     # (B, 8, 3)
        delta = M1c - C0c[:, None, :]
        N0xd = _bcross(np.broadcast_to(N0c[:, None, :], delta.shape), delta)
        lnd = _bnorm(N0xd)
        equid = real_ok & (lnd <= 1e-12)
        fallback |= equid.any(axis=1)
        N0d = _bdot(np.broadcast_to(N0c[:, None, :], delta.shape), delta)
        diff = lnd - r0c[:, None]
        sqD_can = N0d ** 2 + diff ** 2
        proj = delta - N0d[:, :, None] * N0c[:, None, :]
        pn = _bnorm(proj)
        proj = proj / np.where(pn > 0, pn, 1.0)[:, :, None]
        M0c = C0c[:, None, :] + r0c[:, None, None] * proj

        # back to the original frame: P = R_total^T Q / scale - t
        RtT = np.transpose(Rt, (0, 2, 1))
        inv_scale = (1.0 / scale)[:, None, None]
        Ma = np.einsum('bij,bsj->bsi', RtT, M0c) * inv_scale - t[:, None, :]
        Mb = np.einsum('bij,bsj->bsi', RtT, M1c) * inv_scale - t[:, None, :]
        swap3 = swapped[:, None, None]
        Ma, Mb = np.where(swap3, Mb, Ma), np.where(swap3, Ma, Mb)

        t1_int = _bangle(Ma, C1, ef1, ef2)                      # (B, 8)
        t2_int = _bangle(Mb, C2, eg1, eg2)
        valid_int = (real_ok & _bin_arc(t1_int, arc1) & _bin_arc(t2_int, arc2))
        sq_int = np.where(valid_int, sqD_can / (scale ** 2)[:, None], np.inf)

    # ---- 4 edge slots: one angle pinned at an arc endpoint ----
    def _point_on(Cc, Rr, u1, u2, th):
        return Cc + Rr[:, None] * (np.cos(th)[:, None] * u1
                                   + np.sin(th)[:, None] * u2)

    def _edge(P, Cc, Rr, Nn, u1, u2, arc):
        """Batched ``point_circle_eberly`` + angle recovery + in-arc test."""
        PmC = P - Cc
        big = (np.abs(Nn[:, 0]) > np.abs(Nn[:, 1]))[:, None]
        U = np.where(big,
                     np.stack([-Nn[:, 2], np.zeros(B), Nn[:, 0]], axis=1),
                     np.stack([np.zeros(B), Nn[:, 2], -Nn[:, 1]], axis=1))
        V = _bcross(Nn, U)
        NdotN = _bdot(Nn, Nn)
        scaled = (NdotN * _bdot(U, PmC))[:, None] * U + _bdot(V, PmC)[:, None] * V
        L = _bnorm(scaled)
        ok = L > 0.0
        L_safe = np.where(ok, L, 1.0)
        Q = Cc + Rr[:, None] * scaled / L_safe[:, None]
        sqrtN = np.sqrt(NdotN)
        height = _bdot(Nn, PmC) / sqrtN
        radial = _bnorm(_bcross(Nn, PmC)) / sqrtN - Rr
        sqD = height ** 2 + radial ** 2
        th = _bangle(Q, Cc, u1, u2)
        valid = ok & _bin_arc(th, arc)
        return np.where(valid, sqD, np.inf), th

    e_sq = []
    e_t1 = []
    e_t2 = []
    for k in range(2):                       # theta1 pinned at arc1[k]
        th1 = arc1[:, k]
        P = _point_on(C1, R1, ef1, ef2, th1)
        sqD, th2 = _edge(P, C2, R2, n2, eg1, eg2, arc2)
        e_sq.append(sqD); e_t1.append(th1); e_t2.append(th2)
    for k in range(2):                       # theta2 pinned at arc2[k]
        th2 = arc2[:, k]
        P = _point_on(C2, R2, eg1, eg2, th2)
        sqD, th1 = _edge(P, C1, R1, n1, ef1, ef2, arc1)
        e_sq.append(sqD); e_t1.append(th1); e_t2.append(th2)

    # ---- 4 corner slots ----
    c_sq = []
    c_t1 = []
    c_t2 = []
    for k1 in range(2):
        P1c = _point_on(C1, R1, ef1, ef2, arc1[:, k1])
        for k2 in range(2):
            P2c = _point_on(C2, R2, eg1, eg2, arc2[:, k2])
            d = P1c - P2c
            c_sq.append(_bdot(d, d))
            c_t1.append(arc1[:, k1]); c_t2.append(arc2[:, k2])

    # ---- global minimum over the 16 slots (scalar candidate order) ----
    sq_all = np.concatenate([sq_int,
                             np.stack(e_sq, axis=1),
                             np.stack(c_sq, axis=1)], axis=1)   # (B, 16)
    t1_all = np.concatenate([t1_int,
                             np.stack(e_t1, axis=1),
                             np.stack(c_t1, axis=1)], axis=1)
    t2_all = np.concatenate([t2_int,
                             np.stack(e_t2, axis=1),
                             np.stack(c_t2, axis=1)], axis=1)
    sq_all = np.where(np.isnan(sq_all), np.inf, sq_all)
    best = np.argmin(sq_all, axis=1)
    rows = np.arange(B)
    sq_b = sq_all[rows, best]
    t1_b = t1_all[rows, best]
    t2_b = t2_all[rows, best]

    M1 = _point_on(C1, R1, ef1, ef2, t1_b)
    M2 = _point_on(C2, R2, eg1, eg2, t2_b)

    return {
        'dist': np.sqrt(sq_b), 'theta1': t1_b, 'theta2': t2_b,
        'M1': M1, 'M2': M2, 'arc1': arc1, 'arc2': arc2,
        'C1': C1, 'R1': R1, 'n1': n1, 'C2': C2, 'R2': R2, 'n2': n2,
        'A_inv1': A_inv1, 'A_inv2': A_inv2, 'flip1': flip1, 'flip2': flip2,
        'fallback': fallback,
    }


# =============================================================================
# Batched envelope gradient and per-pair force / Gauss-Newton Hessian
# =============================================================================

_DE_A = np.zeros((3, 9)); _DE_A[:, 0:3] = -np.eye(3); _DE_A[:, 3:6] = np.eye(3)
_DE_B = np.zeros((3, 9)); _DE_B[:, 3:6] = -np.eye(3); _DE_B[:, 6:9] = np.eye(3)
_DM_A = np.zeros((3, 9)); _DM_A[:, 0:3] = 0.5 * np.eye(3); _DM_A[:, 3:6] = 0.5 * np.eye(3)
_DM_B = np.zeros((3, 9)); _DM_B[:, 3:6] = 0.5 * np.eye(3); _DM_B[:, 6:9] = 0.5 * np.eye(3)
_DN1 = np.zeros((3, 9)); _DN1[:, 3:6] = np.eye(3)


def _bdcircle_dnodes(nodes, A_inv, flip):
    """Batched ``dCircle_dnodes``: (dC_dN (B,3,9), dR_dN (B,9), dn_dN (B,3,9)).
    ``A_inv`` and ``flip`` are reused from the circle fit."""
    n0, n1v, n2 = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    e_a = n1v - n0
    e_b = n2 - n1v
    cross = _bcross(e_a, e_b)
    nrm = _bnorm(cross)
    n_axis = cross / nrm[:, None]
    m_a = 0.5 * (n0 + n1v)
    m_b = 0.5 * (n1v + n2)

    dcross = (-np.einsum('bij,jk->bik', _bskew(e_b), _DE_A)
              + np.einsum('bij,jk->bik', _bskew(e_a), _DE_B))    # (B, 3, 9)
    P = (np.eye(3)[None] - np.einsum('bi,bj->bij', n_axis, n_axis)) / nrm[:, None, None]
    dn_axis = np.einsum('bij,bjk->bik', P, dcross)

    rhs = np.stack([_bdot(e_a, m_a), _bdot(e_b, m_b), _bdot(n_axis, n1v)], axis=1)
    C = np.einsum('bij,bj->bi', A_inv, rhs)

    dA_C = np.stack([np.einsum('bi,ij->bj', C, _DE_A),
                     np.einsum('bi,ij->bj', C, _DE_B),
                     np.einsum('bi,bij->bj', C, dn_axis)], axis=1)  # (B, 3, 9)
    drhs = np.stack([
        np.einsum('bi,ij->bj', m_a, _DE_A) + np.einsum('bi,ij->bj', e_a, _DM_A),
        np.einsum('bi,ij->bj', m_b, _DE_B) + np.einsum('bi,ij->bj', e_b, _DM_B),
        np.einsum('bi,bij->bj', n1v, dn_axis) + np.einsum('bi,ij->bj', n_axis, _DN1),
    ], axis=1)
    dC_dN = np.einsum('bij,bjk->bik', A_inv, drhs - dA_C)

    diff = C - m_a
    R = _bnorm(diff)
    dR_dN = np.einsum('bi,bij->bj', diff, dC_dN - _DM_A[None]) / R[:, None]

    dn_axis = np.where(flip[:, None, None], -dn_axis, dn_axis)
    return dC_dN, dR_dN, dn_axis


def _bs_from_theta(theta, arc):
    """Batched ``compute_s_from_theta``."""
    alpha = arc[:, 0]
    beta = arc[:, 1]
    span = (beta - alpha) % TWO_PI
    small = span < 1e-10
    span_safe = np.where(small, 1.0, span)
    s_norm = np.minimum(((theta - alpha) % TWO_PI) / span_safe, 1.0)
    return np.where(small, 0.0, 2.0 * s_norm - 1.0)


def _bnode_partition(s):
    """Batched ``_node_partition_from_s``: smooth per-node partition of
    unity (w0, w1, w2) as a (B, 3) array."""
    s = np.clip(s, -1.0, 1.0)
    w1 = compute_weights_from_s(s)
    rest = 1.0 - w1
    return np.stack([rest * (1.0 - s) / 2.0, w1,
                     rest * (1.0 + s) / 2.0], axis=1)


def _bfriction_gate(tv_n, vel_tol, friction_law, nh_rho, nh_v_ref):
    """Batched ``_friction_gate`` (gate value only; no derivative)."""
    if friction_law == 'smooth_coulomb':
        K2 = 15.0 / vel_tol
        y = np.exp(-K2 * tv_n)
        return np.where(tv_n > vel_tol, 1.0, 2.0 / (1.0 + y) - 1.0)
    if friction_law == 'norton_hoff':
        safe = np.where(tv_n < 1e-14, 1.0, tv_n)
        return np.where(tv_n < 1e-14, 0.0, (safe / nh_v_ref) ** nh_rho)
    if friction_law == 'norton_hoff_capped':
        # G = min((|v_t|/v_ref)^rho, 1) -- mirrors the scalar _friction_gate.
        safe = np.where(tv_n < 1e-14, 1.0, tv_n)
        G = np.where(tv_n < 1e-14, 0.0, (safe / nh_v_ref) ** nh_rho)
        return np.minimum(G, 1.0)
    raise ValueError(f"unknown friction_law '{friction_law}'")


def batched_pair_friction_forces(node_pos, u_nodes, starts1, starts2,
                                 beam_radius, d_imc, K_imc, k_c, mu, vel_tol,
                                 first1, last1, first2, last2,
                                 friction_law='smooth_coulomb',
                                 nh_rho=0.1, nh_v_ref=1.0):
    """Batched equivalent of ``_compute_pair_friction`` over B window pairs.

    Parameters as in ``batched_pair_forces``, plus ``u_nodes``: (n_nodes, 3)
    nodal velocities and the friction parameters.  Returns dict with
    ``force`` (B, 18) — the caller's sign convention (subtract from the
    residual) — and ``fallback`` (B,) for pairs that must be recomputed
    with the scalar ``_compute_pair_friction``.
    """
    B = len(starts1)
    idx1 = np.asarray(starts1, int)[:, None] + np.arange(3)[None, :]
    idx2 = np.asarray(starts2, int)[:, None] + np.arange(3)[None, :]
    nodes1 = node_pos[idx1]
    nodes2 = node_pos[idx2]

    res = batched_arc_min_dist(nodes1, nodes2)
    fallback = res['fallback']
    d = res['dist']
    d_safe = np.where(d > 0, d, 1.0)
    ok = (~fallback) & (d >= 1e-14)

    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        s1 = _bs_from_theta(res['theta1'], res['arc1'])
        s2 = _bs_from_theta(res['theta2'], res['arc2'])
        w1 = compute_weights_from_s(s1)
        w2 = compute_weights_from_s(s2)
        w1 = np.where(np.asarray(first1) & (s1 < 0), 1.0, w1)
        w1 = np.where(np.asarray(last1) & (s1 > 0), 1.0, w1)
        w2 = np.where(np.asarray(first2) & (s2 < 0), 1.0, w2)
        w2 = np.where(np.asarray(last2) & (s2 > 0), 1.0, w2)
        w = w1 * w2

        delta_c = (d - 2.0 * beam_radius) + beam_radius
        dE = compute_dE_IMC_ddelta_c(delta_c, beam_radius, d_imc, K_imc)
        f_n_mag = np.abs(k_c * w * dE)
        ok &= f_n_mag >= 1e-30

        p1 = _bnode_partition(s1)                        # (B, 3)
        p2 = _bnode_partition(s2)
        U1 = u_nodes[idx1]                               # (B, 3, 3)
        U2 = u_nodes[idx2]
        v_M1 = (p1[:, :, None] * U1).sum(axis=1)         # (B, 3)
        v_M2 = (p2[:, :, None] * U2).sum(axis=1)
        v_rel = v_M1 - v_M2

        n_hat = (res['M1'] - res['M2']) / d_safe[:, None]
        v_rel_n = _bdot(v_rel, n_hat)
        tv_rel = v_rel - v_rel_n[:, None] * n_hat
        tv_n = _bnorm(tv_rel)
        ok &= tv_n >= 1e-14
        tv_safe = np.where(tv_n > 0, tv_n, 1.0)

        gamma = _bfriction_gate(tv_n, vel_tol, friction_law, nh_rho, nh_v_ref)
        f_tan = (mu * gamma * f_n_mag)[:, None] * (tv_rel / tv_safe[:, None])

    force = np.zeros((B, 18))
    for kn in range(3):
        force[:, 3 * kn:3 * kn + 3] = p1[:, kn:kn + 1] * f_tan
        force[:, 9 + 3 * kn:12 + 3 * kn] = -p2[:, kn:kn + 1] * f_tan
    force[~ok] = 0.0
    return {'force': force, 'fallback': fallback}


def batched_pair_forces(node_pos, starts1, starts2, beam_radius, d_imc, K_imc,
                        first1, last1, first2, last2):
    """Full batched equivalent of ``_compute_pair_force_and_hessian`` over
    B pairs of 3-node windows.

    Parameters
    ----------
    node_pos : (n_nodes, 3)
    starts1, starts2 : (B,) int — first-node index of each side's window
    first1/last1/first2/last2 : (B,) bool — boundary flags for the
        partition-of-unity weight override.

    Returns
    -------
    dict with per-pair arrays: energy (B,), force (B, 18),
    hessian (B, 18, 18), gap (B,), s_i1, s_i2, active (bool),
    fallback (bool).  ``fallback`` pairs carry no results and must be
    recomputed with the scalar path; ``active`` marks pairs whose energy
    passes the scalar driver's 1e-30 acceptance test.
    """
    B = len(starts1)
    idx1 = np.asarray(starts1, int)[:, None] + np.arange(3)[None, :]
    idx2 = np.asarray(starts2, int)[:, None] + np.arange(3)[None, :]
    nodes1 = node_pos[idx1]                                   # (B, 3, 3)
    nodes2 = node_pos[idx2]

    res = batched_arc_min_dist(nodes1, nodes2)
    fallback = res['fallback']
    d_star = res['dist']

    gap = d_star - 2.0 * beam_radius
    # The IMC law is compactly supported: E = dE = 0 exactly for
    # gap > d_imc, so those pairs are exactly the scalar driver's skips.
    near = (~fallback) & (gap <= d_imc) & (d_star >= 1e-14)

    energy = np.zeros(B)
    force = np.zeros((B, 18))
    hess = np.zeros((B, 18, 18))
    s_i1 = np.zeros(B)
    s_i2 = np.zeros(B)

    if near.any():
        nb = np.nonzero(near)[0]
        r = {k: (v[nb] if isinstance(v, np.ndarray) else v)
             for k, v in res.items()}
        s1 = _bs_from_theta(r['theta1'], r['arc1'])
        s2 = _bs_from_theta(r['theta2'], r['arc2'])
        w1 = compute_weights_from_s(s1)
        w2 = compute_weights_from_s(s2)
        w1 = np.where(first1[nb] & (s1 < 0), 1.0, w1)
        w1 = np.where(last1[nb] & (s1 > 0), 1.0, w1)
        w2 = np.where(first2[nb] & (s2 < 0), 1.0, w2)
        w2 = np.where(last2[nb] & (s2 > 0), 1.0, w2)
        w = w1 * w2

        delta_c = (d_star[nb] - 2.0 * beam_radius) + beam_radius
        with np.errstate(over='ignore'):
            E = computeE_IMC(delta_c, beam_radius, d_imc, K_imc)
            dE = compute_dE_IMC_ddelta_c(delta_c, beam_radius, d_imc, K_imc)
            d2E = compute_d2E_IMC_ddelta_c(delta_c, beam_radius, d_imc, K_imc)

        # Envelope gradient of d* w.r.t. the 18 node coordinates.
        Delta = r['M1'] - r['M2']
        rho1 = r['M1'] - r['C1']
        rho2 = r['M2'] - r['C2']
        df_dC1 = 2 * Delta
        df_dR1 = 2 * _bdot(Delta, rho1) / r['R1']
        df_dn1 = -2 * _bdot(Delta, r['n1'])[:, None] * rho1
        df_dC2 = -2 * Delta
        df_dR2 = -2 * _bdot(Delta, rho2) / r['R2']
        df_dn2 = 2 * _bdot(Delta, r['n2'])[:, None] * rho2

        dC1, dR1, dn1 = _bdcircle_dnodes(nodes1[nb], r['A_inv1'], r['flip1'])
        dC2, dR2, dn2 = _bdcircle_dnodes(nodes2[nb], r['A_inv2'], r['flip2'])

        df_dN1 = (np.einsum('bi,bij->bj', df_dC1, dC1)
                  + df_dR1[:, None] * dR1
                  + np.einsum('bi,bij->bj', df_dn1, dn1))
        df_dN2 = (np.einsum('bi,bij->bj', df_dC2, dC2)
                  + df_dR2[:, None] * dR2
                  + np.einsum('bi,bij->bj', df_dn2, dn2))
        dd_dN = np.concatenate([df_dN1, df_dN2], axis=1) / (2 * d_star[nb])[:, None]

        tiny = (np.abs(E) < 1e-30) & (np.abs(dE) < 1e-30)
        wE = np.where(tiny, 0.0, w * E)
        wdE = np.where(tiny, 0.0, w * dE)
        wd2E = np.where(tiny, 0.0, w * d2E)

        energy[nb] = wE
        force[nb] = wdE[:, None] * dd_dN
        hess[nb] = wd2E[:, None, None] * dd_dN[:, :, None] * dd_dN[:, None, :]
        s_i1[nb] = s1
        s_i2[nb] = s2

    active = (~fallback) & (np.abs(energy) >= 1e-30)
    return {
        'energy': energy, 'force': force, 'hessian': hess,
        'gap': gap, 's_i1': s_i1, 's_i2': s_i2,
        'active': active, 'fallback': fallback,
    }
