# External reference data for the three-point bending figures (Fig. 2, Fig. 5, Table 1)

This folder is intentionally empty in the public repo. The data
`postprocessing/fig2_force_displacement.py`, `fig4_jitter_correlation.py`,
and `fig6_jitter_metrics.py` read from here (and from the sibling
`../contact_output_files_3D_bead/`) totals ~2.2GB across 5 files, which
exceeds GitHub's per-file size limit — so it isn't tracked in git. All
three scripts degrade gracefully (print a `[warn] ... skipping`, draw
whatever curves they do have data for) if a file below is missing,
rather than crashing.

| file | role | how to get it |
|---|---|---|
| `impV_noD_contact_bending_N50.npz` | 2-node (segment) model, N=50, full run to t=900s | **Not regenerable from this repo.** Traced via git history to a separate, non-`dismech` simulation implementation that was never part of this codebase. Contact the authors for this specific file if you need the exact curve shown in the paper. An independent, from-scratch 2-node baseline that *is* regenerable here exists as `../contact_bending_test_2node.py` (see `../contact_bending_test_2node_README.md`) — it reuses dismech's own 2-node contact engine, but has not been run at paper scale (estimated 3.5-10 days) and is not a byte-for-byte replacement for this file. |
| `exact_rL_0p125.csv` | Batista analytical solution, r/L=0.125 | Reference data, not a simulation output. Contact the authors if you need this specific file; it is a direct evaluation of the closed-form solution in Batista (2015), cited in the paper. |
| `reg_beam_coords_N50.txt` + `reg_contact_forces_N50.txt` | 3-node (ours) model, N=50 | **Regenerate:** `python ../contact_bending_test.py 50` (long run: dt=1e-3, t≈1000s, i.e. 1e6 steps — this is what the original reference data took to produce). |

The 1-node bead baseline (`../contact_output_files_3D_bead/reg_beam_coords_N50_bead.txt`
+ `reg_contact_forces_N50_bead.txt`) is likewise not shipped; regenerate
with `python ../contact_bending_test_3D_bead.py 50`.

If you just want to confirm the pipeline works without a multi-day
simulation, run any of the figure/table scripts at a much smaller N
(e.g. `python ../contact_bending_test.py 10`) — the curves won't match
the paper's N=50 reference, but the scripts, units, and plotting
pipeline are exercised identically.
