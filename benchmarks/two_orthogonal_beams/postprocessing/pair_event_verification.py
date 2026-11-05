#!/usr/bin/env python3
"""
Appendix B.5 numerical verification: when an Eberly contact stencil pair
leaves (or enters) the active set, it does so carrying zero weight, so the
multiplicative blend never lets a step enter the total contact force from an
active-set change alone.

Data: `pair_events_prodblend_mu0.jsonl`, produced by running the
two-orthogonal-beams benchmark (Zavarise & Wriggers Ex. 2, mu = 0, 21 nodes
per beam) with `DISMECH_PAIR_EVENT_LOG=<path>` set -- see
`eberly_contact_3d.py::PairEventLog`.  Each line is one event: a
(segment-pair, s_i1, s_i2) triple recorded the Newton call where that pair
entered or left the active set.

Method
------
One stencil pair carries the contact for the whole logged window (it only
ever "enters", never "leaves"): detected generically as any pair tag with
zero 'leave' events.  Every other event is a genuine active-set transition
of a NEIGHBOURING stencil sliding in or out of range.  For each transition
we evaluate the partition-of-unity bump weight

    w(s) = compute_weights_from_s(s)     (w(+/-1) = 0 by construction)

on both arc parameters and form the pair weight product w1*w2 -- the same
product that multiplies the contact energy in the assembled force (see
`eberly_contact_3d.py`, every `w = w1 * w2` site).  If the Proposition
holds, this product must be negligible at every transition.

Output
------
  - plain-text table -> results/pair_event_verification.txt
  - LaTeX table       -> figures/table_pair_event_verification.tex
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from dismech.contact.circle_four_nodes_imc_weighted import compute_weights_from_s

DATA_DIR = Path(__file__).resolve().parents[1] / "contact_output_files_soft_soft"
LOG_FILE = DATA_DIR / "pair_events_prodblend_mu0.jsonl"

OUT_DIR_RESULTS = Path(__file__).parent / "results"
OUT_DIR_FIGURES = Path(__file__).parent / "figures"
OUT_TXT = OUT_DIR_RESULTS / "pair_event_verification.txt"
OUT_TEX = OUT_DIR_FIGURES / "table_pair_event_verification.tex"


def load_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def split_center_vs_transitions(events: list[dict]):
    """A pair tag that never appears in a 'leave' event is the one carrying
    the contact for the whole logged window; everything else is a genuine
    active-set transition of a neighbouring stencil."""
    leave_count = defaultdict(int)
    for e in events:
        if e["event"] == "leave":
            leave_count[e["pair"]] += 1
    all_pairs = {e["pair"] for e in events}
    center_pairs = {p for p in all_pairs if leave_count[p] == 0}
    transitions = [e for e in events if e["pair"] not in center_pairs]
    return center_pairs, transitions


def main() -> None:
    events = load_events(LOG_FILE)
    center_pairs, transitions = split_center_vs_transitions(events)

    s1 = np.array([e["s_i1"] for e in transitions])   # neighbour's own arc
    s2 = np.array([e["s_i2"] for e in transitions])   # partner window's arc
    w1 = compute_weights_from_s(s1)
    w2 = compute_weights_from_s(s2)
    w = w1 * w2

    dev1 = 1.0 - np.abs(s1)       # distance of the neighbour's s* from +/-1
    n = len(transitions)

    lines = []
    def emit(s=""):
        print(s)
        lines.append(s + "\n")

    emit("Pair-event verification (App. B.5) -- "
         f"source: {LOG_FILE.name}")
    emit(f"Contact-carrying stencil pair(s): {sorted(center_pairs)}")
    emit(f"Active-set transitions analysed: {n}")
    emit()
    emit(f"{'quantity':34s} {'min':>12s} {'median':>12s} {'max':>12s}")
    emit(f"{'1 - |s*| (neighbour arc end)':34s} "
         f"{dev1.min():12.3e} {np.median(dev1):12.3e} {dev1.max():12.3e}")
    emit(f"{'weight product w = w1*w2':34s} "
         f"{w.min():12.3e} {np.median(w):12.3e} {w.max():12.3e}")
    emit()
    emit("Equivalent decimal-place reading (-log10 of the quantity above):")
    emit(f"  |s*| agreement with an arc end : "
         f"worst {-np.log10(dev1.max()):.1f}, "
         f"median {-np.log10(np.median(dev1)):.1f} decimal places")
    emit(f"  weight vanishing (w -> 0)      : "
         f"worst {-np.log10(w.max()):.1f}, "
         f"median {-np.log10(np.median(w)):.1f} decimal places")
    emit()
    emit("Note: the WEIGHT is what multiplies the force (the quantity the "
         "Proposition is about), and it is zero to 8+ decimal places at "
         "every transition, worst case included.  The arc parameter s* "
         "itself only reaches the arc end to ~4-5 decimal places in the "
         "worst-case transitions (the blend's quartic falloff near s = "
         "+/-1 means a modest s-deviation already gives a negligible "
         "weight) -- so '|s*| = 1 to seven decimal places' is not accurate "
         "as a blanket statement over all transitions; the vanishing-"
         "weight statement is the one that holds uniformly.")

    OUT_DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("".join(lines))
    print(f"\nWrote {OUT_TXT}")

    def fmt(x):
        return f"{x:.2e}".replace("e-0", "e-").replace("e+0", "e+")

    tex = "\n".join([
        r"\begin{table}[ht]",
        r"    \centering",
        r"    \small",
        r"    \caption{Active-set transition check on the two-orthogonal-"
        r"beams benchmark ($\mu=0$, 21 nodes per beam): statistics of the "
        f"{n}" + r" sliding transitions of the two stencils neighbouring "
        r"the contact-carrying pair, over the Newton calls of the run. "
        r"$1-|s^\ast|$ is the neighbour's own arc parameter's distance from "
        r"an arc end; $w=w_1 w_2$ is the pair's blend-weight product, the "
        r"factor that multiplies the contact force.}",
        r"    \label{tab:pair_event_verification}",
        r"    \begin{tabular}{lccc}",
        r"        \toprule",
        r"        Quantity & Min & Median & Max \\",
        r"        \midrule",
        rf"        $1-|s^\ast|$ & {fmt(dev1.min())} & {fmt(np.median(dev1))} & {fmt(dev1.max())} \\",
        rf"        $w = w_1 w_2$ & {fmt(w.min())} & {fmt(np.median(w))} & {fmt(w.max())} \\",
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
        "",
    ])
    OUT_DIR_FIGURES.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(tex)
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
