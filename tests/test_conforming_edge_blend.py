"""Toy-case tests for the opt-in conforming-contact edge-model crossfade.

The feature (``conforming_edge_blend=True`` on the Eberly self-contact and
self-friction drivers) crossfades window pairs to the classic two-node
edge model where a smooth conformity field detects extended contact.
These tests exercise the two conforming regimes that broke the window
model (straight-parallel and curved-conformal), the transversal regression
(feature must be a strict no-op), and — when the archived stall states are
present — the production death states.
"""
import pathlib
import types

import numpy as np
import pytest

from dismech.contact.eberly_contact_3d import (
    compute_eberly_3d_self_contact_forces,
    compute_eberly_3d_self_friction_pairs,
    compute_eberly_3d_self_friction_force,
    _conformity_gammas,
)

H = 1.0
D_IMC = 0.4
KC = 100.0
THR = 30.0

ARCHIVE = (pathlib.Path(__file__).resolve().parents[1]
           / "knots_catia_Esther" / "archive_partial_runs")


def _robot(n):
    return types.SimpleNamespace(
        n_nodes=n, node_dof_indices=np.arange(3 * n).reshape(n, 3))


def _forces(pts, on, min_sep=6):
    rob = _robot(len(pts))
    F, J, gmin, gaps = compute_eberly_3d_self_contact_forces(
        rob, pts.reshape(-1).astype(float), H, k_c=KC,
        min_segment_separation=min_sep, d_imc=D_IMC,
        straight_R_threshold=THR, conforming_edge_blend=on)
    return F, J, gmin, gaps


# ---------------------------------------------------------------- builders

def hairpin(delta=0.0, n_leg=14, l=2.0, sep=2.05):
    """Straight parallel legs, surface gap sep-2h; leg B slid by delta."""
    leg_A = np.array([[i * l, 0.0, 0.0] for i in range(n_leg)])
    th = np.linspace(0, np.pi, 6)[1:-1]
    cx = (n_leg - 1) * l
    bend = np.array([[cx + (sep / 2) * np.sin(t),
                      (sep / 2) * (1 - np.cos(t)), 0.0] for t in th])
    leg_B = np.array([[(n_leg - 1 - i) * l + delta, sep, 0.0]
                      for i in range(n_leg)])
    return np.vstack([leg_A, bend, leg_B])


def concentric(phi=0.0, Ra=20.0, gap=0.05, span=2.0 * np.pi / 3, l=2.0):
    """Two concentric arcs (surface gap `gap`) joined by an outward bridge.

    Arc A: radius Ra, angles [0, span].  Arc B: radius Ra + 2H + gap,
    angles [phi, span + phi], traversed backward.  The bridge detours to
    radius Ra + 12 so it stays out of contact range of both arcs.
    """
    Rb = Ra + 2 * H + gap
    a_ang = np.arange(0.0, span + 1e-9, l / Ra)
    A = np.stack([Ra * np.cos(a_ang), Ra * np.sin(a_ang),
                  np.zeros_like(a_ang)], axis=1)
    # bridge: spiral out at the span end, over, and back in to arc B's end
    br_out = []
    for r in np.arange(Ra + l, Ra + 12.0, l):
        th = span + 0.35
        br_out.append([r * np.cos(th), r * np.sin(th), 0.0])
    br_in = []
    for r in np.arange(Ra + 12.0, Rb + l - 1e-9, -l):
        th = span + 0.18
        br_in.append([r * np.cos(th), r * np.sin(th), 0.0])
    b_ang = np.arange(span + phi, phi - 1e-9, -l / Rb)
    B = np.stack([Rb * np.cos(b_ang), Rb * np.sin(b_ang),
                  np.zeros_like(b_ang)], axis=1)
    return np.vstack([A, np.array(br_out), np.array(br_in), B])


def crossing(angle_deg=60.0, gap=0.1, n_leg=12, l=2.0):
    """Two straight rods crossing at `angle_deg`, closest gap `gap`,
    joined far away (transversal point contact)."""
    a = np.radians(angle_deg)
    L = (n_leg - 1) * l
    t = np.linspace(-L / 2, L / 2, n_leg)
    A = np.stack([t, np.zeros_like(t), np.zeros_like(t)], axis=1)
    B = np.stack([t * np.cos(a), t * np.sin(a),
                  np.full_like(t, 2 * H + gap)], axis=1)[::-1]
    join = np.array([[L, -8.0, H + gap / 2],
                     [L + 4, -8.0, 2 * H + gap]])
    return np.vstack([A, join, B])


# ---------------------------------------------------------------- tests

def test_transversal_is_bit_identical():
    """gamma == 0 on point contact: feature ON must be a strict no-op."""
    for ang in (90.0, 60.0, 45.0):
        pts = crossing(ang)
        F0, J0, g0, gaps0 = _forces(pts, on=False)
        F1, J1, g1, gaps1 = _forces(pts, on=True)
        assert np.array_equal(F0, F1), f"force differs at {ang} deg"
        assert np.array_equal(J0, J1), f"jacobian differs at {ang} deg"
        assert len(gaps0) == len(gaps1)


def test_hairpin_conformity_detected():
    pts = hairpin()
    n = len(pts)
    num_seg = n - 2
    ii, jj = np.meshgrid(np.arange(num_seg), np.arange(num_seg),
                         indexing="ij")
    m = jj >= ii + 6
    g = _conformity_gammas(pts.astype(float), ii[m], jj[m], H, D_IMC)
    # the leg-on-leg pairs must be recognized as conforming
    assert g.max() > 0.99
    # and a decent number of pairs must be in the conforming regime
    assert (g > 0.9).sum() >= 6


def test_hairpin_slide_invariance_and_smoothness():
    """Normal force on the INTERIOR of leg A should be ~invariant as leg B
    slides by a full segment length, and vary smoothly, with the feature
    ON.  (Interior nodes only: the toy builder's bend junction stretches
    with the slide, so leg-end forces carry a geometry artifact that has
    nothing to do with the contact model.)"""
    deltas = np.linspace(0.0, 2.0, 21)
    tot = {False: [], True: []}
    for on in (False, True):
        for d in deltas:
            pts = hairpin(delta=d)
            F, _, _, _ = _forces(pts, on)
            Fy = F.reshape(-1, 3)[3:11, 1]    # interior leg-A nodes
            tot[on].append(Fy.sum())
    off = np.abs(np.array(tot[False]))
    on = np.abs(np.array(tot[True]))
    rip_off = (off.max() - off.min()) / max(off.mean(), 1e-30)
    rip_on = (on.max() - on.min()) / max(on.mean(), 1e-30)
    # Since the line-primitive chord extension (2026-09-18), the OFF
    # baseline handles STRAIGHT parallel contact well too, so both modes
    # must be near-invariant here.  (The case where OFF stays broken and
    # only ON works is curved conforming contact — see the concentric
    # test.)  ON's slightly larger ripple is a gamma-zone-boundary
    # artifact at the edges of the measurement window.
    assert on.min() > 0.0
    assert rip_off < 0.05, f"OFF ripple {rip_off:.4f}"
    assert rip_on < 0.08, f"ON ripple {rip_on:.4f}"
    jump_on = np.max(np.abs(np.diff(on))) / max(on.mean(), 1e-30)
    assert jump_on < 0.04, f"ON max jump {jump_on:.4f}"


def test_concentric_rotation_invariance_and_smoothness():
    """Curved conforming contact (the trefoil regime, R~20): force on the
    inner arc ~invariant and smooth under rotation of the outer arc."""
    l_over_R = 2.0 / 20.0
    phis = np.linspace(0.0, l_over_R, 17)
    tot = {False: [], True: []}
    n_arcA = len(np.arange(0.0, 2.0 * np.pi / 3 + 1e-9, l_over_R))
    for on in (False, True):
        for phi in phis:
            pts = concentric(phi=phi)
            F, _, _, _ = _forces(pts, on)
            Fn = F.reshape(-1, 3)[:n_arcA]
            # radial (outward-normal) component on inner-arc interior nodes
            pos = pts[:n_arcA]
            r_hat = pos / np.linalg.norm(pos, axis=1)[:, None]
            fr = np.einsum("ij,ij->i", Fn, r_hat)[2:-2]
            tot[on].append(fr.sum())
    off = np.abs(np.array(tot[False]))
    on = np.abs(np.array(tot[True]))
    rip_on = (on.max() - on.min()) / max(on.mean(), 1e-30)
    jump_on = np.max(np.abs(np.diff(on))) / max(on.mean(), 1e-30)
    assert on.min() > 0.0
    assert rip_on < 0.05, f"ON ripple {rip_on:.4f}"
    assert jump_on < 0.02, f"ON max jump {jump_on:.4f}"


def test_friction_momentum_balance_conforming():
    """Edge-model friction must conserve momentum on a conforming case."""
    pts = hairpin()
    n = len(pts)
    rob = _robot(n)
    q = pts.reshape(-1).astype(float)
    u = np.zeros(3 * n)
    u[0:3 * 14:3] = 5.0          # leg A slides axially
    pairs = compute_eberly_3d_self_friction_pairs(
        rob, q, H, min_segment_separation=6, d_imc=D_IMC)
    Ff = compute_eberly_3d_self_friction_force(
        rob, q, u, pairs, H, mu=0.1, vel_tol=0.1, k_c=KC, d_imc=D_IMC,
        straight_R_threshold=THR, conforming_edge_blend=True)
    assert np.abs(Ff).max() > 0.0
    assert np.abs(Ff.reshape(-1, 3).sum(axis=0)).max() < 1e-10


@pytest.mark.skipif(
    not (ARCHIVE / "trefoil_no_preload_eberly_final_nodes_20260918_103636.txt").exists(),
    reason="archived production stall state not present")
def test_death_state_ray_smoothness():
    """At the archived t=4.3757 stall state the contact force must be
    smooth along the last motion direction with the feature ON (the OFF
    model has a ~1.6e-2 N jump from a contact-minimum swap)."""
    final = np.loadtxt(
        ARCHIVE / "trefoil_no_preload_eberly_final_nodes_20260918_103636.txt")
    rows = []
    with open(ARCHIVE /
              "trefoil_no_preload_eberly_snapshots_0p1s_20260918_103636.txt") as f:
        for line in f:
            if not line.startswith("#"):
                rows.append(line)
    snap = np.fromstring(rows[-1], sep=" ")[1:].reshape(len(final), 3)
    direc = final - snap
    direc /= np.linalg.norm(direc)
    lams = np.linspace(-2e-3, 2e-3, 41)

    def scan(on):
        Fs = []
        for lam in lams:
            F, _, _, _ = _forces(final + lam * direc, on)
            Fs.append(F)
        Fs = np.array(Fs)
        d = np.linalg.norm(np.diff(Fs, axis=0), axis=1)
        return d.max(), np.median(d)

    jmax_off, jmed_off = scan(False)
    jmax_on, jmed_on = scan(True)
    # OFF exhibits the documented discontinuity (two orders above median)
    assert jmax_off > 50 * jmed_off
    # ON must be smooth: worst step comparable to the background
    assert jmax_on < 10 * jmed_on, (
        f"ON still jumps: max {jmax_on:.2e} vs median {jmed_on:.2e}")


# -------------------------------------------------- regime transitions

def crossing_generic(theta_deg, gap=0.1, n_leg=16, l=2.0, tau=0.73,
                     xc=0.37):
    """Two straight rods crossing at theta with the crossing point generic
    on BOTH strands (tau/xc offsets), far join."""
    a = np.radians(theta_deg)
    L = (n_leg - 1) * l
    t = np.linspace(-L / 2, L / 2, n_leg)
    A = np.stack([t, np.zeros_like(t), np.zeros_like(t)], axis=1)
    tb = t + tau
    B = np.stack([xc + tb * np.cos(a), tb * np.sin(a),
                  np.full_like(t, 2 * H + gap)], axis=1)[::-1]
    join = np.array([[L + 6, -10.0, H + gap / 2],
                     [L + 10, -10.0, 2 * H + gap]])
    return np.vstack([A, join, B])


def peel(alpha_deg, n_leg=16, l=2.0, sep=2.05):
    """Hairpin with leg B hinged open by alpha about the bend-side end:
    conforming near the bend, peeling to a point contact."""
    a = np.radians(alpha_deg)
    leg_A = np.array([[i * l, 0.0, 0.0] for i in range(n_leg)])
    th = np.linspace(0, np.pi, 6)[1:-1]
    cx = (n_leg - 1) * l
    bend = np.array([[cx + (sep / 2) * np.sin(t),
                      (sep / 2) * (1 - np.cos(t)), 0.0] for t in th])
    leg_B = np.array([[cx - k * l * np.cos(a), sep + k * l * np.sin(a), 0.0]
                      for k in range(n_leg)])
    return np.vstack([leg_A, bend, leg_B])


def _refinement_smooth(build, lo, hi, coarse, fine, jump_coarse_max):
    """Assert per-step jumps are bounded at coarse resolution AND shrink
    ~linearly under refinement (the signature of a smooth curve; a
    discontinuity's jump would not shrink)."""
    out = {}
    for npts in (coarse, fine):
        xs = np.linspace(hi, lo, npts)
        f = np.array([np.linalg.norm(_forces(build(x), True)[0])
                      for x in xs])
        rel = np.abs(np.diff(f)) / np.maximum(np.maximum(f[:-1], f[1:]),
                                              1e-12)
        out[npts] = rel.max()
    assert out[coarse] < jump_coarse_max, f"coarse jump {out[coarse]:.4f}"
    ratio = (fine - 1) / (coarse - 1)
    assert out[fine] < 2.5 * out[coarse] / ratio, (
        f"jump does not refine away: {out[coarse]:.4f} -> {out[fine]:.4f} "
        f"at {ratio:.0f}x refinement")


def test_transition_angle_sweep_smooth():
    """Transversal -> conforming via crossing angle: force continuous
    through the whole gamma ramp (21 deg -> 11 deg) and beyond."""
    _refinement_smooth(crossing_generic, 8.0, 24.0, 33, 129, 0.15)


def test_transition_deep_conforming_smooth():
    _refinement_smooth(crossing_generic, 0.8, 1.6, 33, 129, 0.02)


def test_transition_peel_smooth():
    """Spatial transition along one contact (conforming zone peeling to a
    point contact) — the braid-exit situation of the trefoil."""
    _refinement_smooth(peel, 0.0, 20.0, 41, 161, 0.15)
