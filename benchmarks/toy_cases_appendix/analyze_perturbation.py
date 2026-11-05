#!/usr/bin/env python3
"""
Metrics and deliverables for the 2-node perturbation mechanism study.

A: peak-to-peak of the force residual.  Detrending: moving average over a
   3h window (boxcar, reflected ends), i.e. three edge-crossing periods —
   removes the (constant) physical trend and any slow envelope while
   leaving the period-h ripple intact.
B: max|dF/ds| from the SAME log decimated to ds = h/10, h/100, h/1000.
   A true jump grows ~linearly with density; a continuous signal saturates.
C: dominant spatial frequency of the residual, cycles per edge length.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / 'results'
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)

DENSITIES = (10, 100, 1000)     # samples per h used for metric B


def moving_average(y, win):
    pad = win // 2
    ypad = np.concatenate([y[pad:0:-1], y, y[-2:-pad - 2:-1]])
    return np.convolve(ypad, np.ones(win) / win, mode='valid')[:len(y)]


def load(run):
    d = np.load(RES / f'{run.lower()}.npz')
    p = json.loads(str(d['params']))
    F = np.linalg.norm(d['force'], axis=1)
    return d, p, F


def metrics(run):
    d, p, F = load(run)
    s_h = d['s_over_h']
    sph = p['samples_per_h']

    win = 3 * sph + 1                       # 3 edge lengths
    resid = F - moving_average(F, win)
    A = float(resid.max() - resid.min())

    B = {}
    for dens in DENSITIES:
        step = sph // dens
        Fd, sd = F[::step], s_h[::step] * p['h']
        B[dens] = float(np.max(np.abs(np.diff(Fd) / np.diff(sd))))
    # largest single force increment at the finest density (jump amplitude
    # estimate if the signal is discontinuous)
    dF_fine = float(np.max(np.abs(np.diff(F))))

    # FFT of the residual against sliding distance (uniform grid)
    ham = np.hanning(len(resid))
    spec = np.abs(np.fft.rfft(resid * ham))
    freq = np.fft.rfftfreq(len(resid), d=1.0 / sph)   # cycles per h
    k0 = 1 + int(np.argmax(spec[1:]))
    C = float(freq[k0])

    ev = json.loads(str(d['pair_log']))
    sets = [tuple(e['pairs']) for e in ev]
    changes = [i for i in range(1, len(sets)) if sets[i] != sets[i - 1]]

    return dict(p=p, F=F, s_h=s_h, resid=resid, A=A, B=B, dF_fine=dF_fine,
                C=C, mean_F=float(F.mean()), events=changes)


def main():
    runs = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6']
    M = {r: metrics(r) for r in runs}

    hdr = (f"{'run':4s} {'swept':22s} {'A (p2p)':>10s} "
           + ''.join(f"B@h/{d:<5d}" for d in DENSITIES)
           + f" {'C [1/h]':>8s} {'eta2/(d a)':>11s} {'mean|F|':>9s}")
    lines = [hdr, '-' * len(hdr)]
    swept = {'R1': 'baseline', 'R2': 'kappa=0', 'R3': 'delta=0.05',
             'R4': 'refresh every iter', 'R5': 'N=100', 'R6': 'N=200'}
    for r in runs:
        m = M[r]
        lines.append(
            f"{r:4s} {swept[r]:22s} {m['A']:10.4g} "
            + ''.join(f"{m['B'][d]:9.3g}" for d in DENSITIES)
            + f" {m['C']:8.3f} {m['p']['eta2_over_delta_a']:11.3g}"
            + f" {m['mean_F']:9.4g}")
    table = '\n'.join(lines)
    print(table)
    (FIG / 'results_table.txt').write_text(table + '\n')

    print('\nDiscrimination:')
    print(f"  R1 vs R2 (kappa=0)  : A ratio = {M['R1']['A'] / max(M['R2']['A'], 1e-300):.3g}")
    print(f"  R3 / R1 (delta 0.05): A ratio = {M['R3']['A'] / M['R1']['A']:.3g}")
    print(f"  R4 vs R1 (refresh)  : max|F_R4 - F_R1| = "
          f"{np.max(np.abs(M['R4']['F'] - M['R1']['F'])):.3g}")
    hs = np.array([M[r]['p']['h'] for r in ('R1', 'R5', 'R6')])
    As = np.array([M[r]['A'] for r in ('R1', 'R5', 'R6')])
    cf, cov = np.polyfit(np.log(hs), np.log(As), 1, cov=True)
    slope, serr = cf[0], np.sqrt(cov[0, 0])
    print(f"  slope log A vs log h (R1,R5,R6): {slope:.3f} +/- {serr:.3f}")
    for r in runs:
        m = M[r]
        sat = m['B'][1000] / m['B'][100]
        print(f"  {r}: B(h/1000)/B(h/100) = {sat:.2f} "
              f"({'JUMP-like (~10 expected for a discontinuity)' if sat > 5 else 'saturating -> continuous'})"
              f", max single-sample |dF| = {m['dF_fine']:.3g}")

    # ---- plots (advisor style, mathtext fallback on this Mac) --------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'text.usetex': False, 'mathtext.fontset': 'cm',
        'font.family': 'serif', 'font.size': 9,
        'axes.linewidth': 0.8, 'legend.edgecolor': 'black',
        'legend.framealpha': 1.0, 'legend.fancybox': False,
    })

    # log-log A vs h
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.loglog(hs * 1e3, As, 'o', mfc='none', mec='#1f77b4', label='measured')
    hf = np.linspace(hs.min(), hs.max(), 50)
    ax.loglog(hf * 1e3, np.exp(cf[1]) * hf ** slope, '-', color='#d62728',
              lw=1, label=rf'fit: slope ${slope:.2f} \pm {serr:.2f}$')
    ax.set_xlabel(r'edge length $h$ [mm]')
    ax.set_ylabel(r'residual peak-to-peak $A$')
    ax.legend()
    fig.tight_layout()
    for ext in ('pdf', 'svg'):
        fig.savefig(FIG / f'A_vs_h.{ext}')
    plt.close(fig)

    # baseline trace with active-set events
    m = M['R1']
    sel = m['s_h'] <= 4.0
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(m['s_h'][sel], m['F'][sel], '-', color='#1f77b4', lw=0.8,
            label=r'$|F_c|$')
    evs = [i for i in m['events'] if m['s_h'][i] <= 4.0]
    ax.plot(m['s_h'][evs], m['F'][evs], 'x', color='#d62728', ms=5,
            label='active-set change')
    ax.set_xlabel(r'sliding distance $s/h$')
    ax.set_ylabel(r'contact force $|F_c|$  ($k_c = 1$)')
    ax.legend()
    fig.tight_layout()
    for ext in ('pdf', 'svg'):
        fig.savefig(FIG / f'trace_R1.{ext}')
    plt.close(fig)
    print(f'\nfigures -> {FIG}')


if __name__ == '__main__':
    main()
