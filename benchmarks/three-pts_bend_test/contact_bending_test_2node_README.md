# `contact_bending_test_2node.py` — independent, in-repo 2-node baseline

## Why this exists

The 2-node (segment) baseline used for the paper's Fig. 2 / Fig. 5 / Table 1
(`external_reference_data/impV_noD_contact_bending_N50.npz`) is not
regenerable from this repo: it was traced via git history to a separate,
non-`dismech` simulation implementation that only ever lived in the
`regularized_contact_Esther/` development folder (excluded from this repo).
See `external_reference_data/README.md` for that finding.

This script is an independent 2-node baseline that *is* fully regenerable
from code already in this repo, with no external dependency.

## How it works

It reuses dismech's existing 2-node (segment-to-segment) IMC contact
engine — the exact machinery already used for the 2-node comparisons in
`two_orthogonal_beams/` and `net_pullout/` (`env.add_force('selfContact',
...)`, backed by `IMCEnergy`/`ContactEnergy` in `src/dismech/contact/`).
**No new contact-energy code was added to `src/dismech`.**

The circular obstacle is discretized into 200 fixed nodes (matching the
paper's Sec. 3.1: "the support is itself discretized, with 200 nodes
around the circle"), combined with the beam into one geometry, and
contact is resolved as ordinary edge-to-edge IMC contact between beam
segments and circle segments — using the same cross-beam-only pair
construction (`n_edges_beam_A` + `cross_beam_only_contact=True`) already
used to pair two separate beams in `two_orthogonal_beams/`.

One physical correction was needed: `selfContact`'s gap formula assumes
both contacting bodies share the same edge radius `h` (gap = centerline
distance − 2h), which is correct for beam-to-beam contact but not for a
zero-thickness rigid curve. The discretized circle is therefore placed at
radius `circle_radius − beam_radius` (not `circle_radius`), which makes
the symmetric formula reproduce the true physical gap
`‖X − C‖ − circle_radius − beam_radius`. See the script's docstring for
the derivation.

## Status

Validated at small scale (N=10 beam nodes, 40 circle nodes): clean Newton
convergence (2-3 iterations/step), stable ~24-40 ms/step over a 5000-step
run with no slowdown or errors.

**Not yet run at paper scale.** At N=50 beam nodes / 200 circle nodes /
1,000,000 steps (dt=1e-3, total_time=1000s, matching
`contact_bending_test.py` exactly), the per-step cost scales with the
number of candidate contact pairs (~27x more than the validated small
case), putting a full run at an estimated **3.5-10 days of CPU time** —
consistent with how the other reference datasets in this repo were
produced (multi-day unattended runs; see e.g.
`net_pullout/NOTES.md`). This is not something to run interactively; run
it as a background/cluster job.

## Running it

```bash
cd benchmarks/three-pts_bend_test
python3 contact_bending_test_2node.py 50       # full paper-scale run (days)
python3 contact_bending_test_2node.py 10       # quick smoke test (minutes)
```

Output goes to `contact_output_files_2node/reg_{beam_coords,contact_forces,summary}_N{n}_2node.txt`,
in the same format as `contact_bending_test.py`'s output (so the existing
`load_coords_file`/`load_forces_file` helpers in `postprocessing/plot_style.py`
read it directly). It supports `--restart` for resuming an interrupted run,
and `--n-circle-nodes` to trade accuracy for speed during development.

This script's output is not currently wired into the Fig. 2/Fig. 5/Table 1
postprocessing scripts (which still read the original external npz) —
doing so is a reasonable follow-up once a full run has actually completed.
