#!/usr/bin/env python3
"""
Appendix A.1 numerical verification: for the minimum-based two-node model,
the jump of the distance gradient at a concave junction equals

    ||[[grad Delta]]|| = 2 cos(alpha) = 2 sin(phi/2),

with alpha the half-angle each segment makes with the bisector and
phi = pi - 2*alpha the turning angle (Eq. A3 of the paper).

Setup: a polyline of two unit segments meeting at the origin, opening
upward; the contacting point sits on the inward (concave-side) bisector,
i.e. on the +y axis, at fixed distance d.  The closest point is the
perpendicular foot on one segment on either side of the bisector, so the
gradient n = (x - foot)/|x - foot| jumps as x crosses it.  The two
one-sided gradients are evaluated at x = (+/-eps, d) and the jump
magnitude is compared with 2 cos(alpha) for alpha = 10..80 degrees.
"""
from pathlib import Path

import numpy as np


def closest_point_polyline(x, verts):
    best, bd = None, np.inf
    for a, b in zip(verts[:-1], verts[1:]):
        e = b - a
        t = np.clip(np.dot(x - a, e) / np.dot(e, e), 0.0, 1.0)
        p = a + t * e
        d = np.linalg.norm(x - p)
        if d < bd:
            bd, best = d, p
    return best, bd


def grad_delta(x, verts):
    p, d = closest_point_polyline(x, verts)
    return (x - p) / d


def main():
    d, eps = 0.3, 1e-9
    lines = [f'{"alpha[deg]":>10s} {"measured":>12s} {"2cos(alpha)":>12s} '
             f'{"rel.err":>10s}']
    worst = 0.0
    for alpha_deg in np.arange(10.0, 80.0 + 1e-9, 5.0):
        al = np.radians(alpha_deg)
        # segments from the origin, each at angle alpha from the +y bisector
        p_left = np.array([-np.sin(al), np.cos(al)])
        p_right = np.array([np.sin(al), np.cos(al)])
        verts = [p_left, np.zeros(2), p_right]
        g_l = grad_delta(np.array([-eps, d]), verts)
        g_r = grad_delta(np.array([+eps, d]), verts)
        jump = np.linalg.norm(g_r - g_l)
        pred = 2 * np.cos(al)
        rel = abs(jump - pred) / pred
        worst = max(worst, rel)
        lines.append(f'{alpha_deg:10.1f} {jump:12.8f} {pred:12.8f} {rel:10.2e}')
    lines.append(f'\nworst relative error over the sweep: {worst:.2e}')
    out = '\n'.join(lines)
    print(out)
    (Path(__file__).parent / 'results' /
     'jump_2cosalpha_check.txt').write_text(out + '\n')
    assert worst < 1e-6, 'measured jump does not match 2 cos(alpha)'


if __name__ == '__main__':
    main()
