# Paper -> files map

This repo was curated from the development repo specifically to
reproduce the results of "Modeling contact in elastic rods: contact
geometry for jitter-free simulation." Every figure/table script listed
below was re-run against exactly the data shipped here while preparing
this repo, and reproduces the published result (spot-checked
numerically against Table 1 and the App. B.5 numbers, and visually
against Fig. 8).

## Fig. 2, Fig. 5, Table 1 — `three-pts_bend_test/`

- Runs: `contact_bending_test.py` (+ `run_discretization.sh` for the
  mesh sweep), `contact_bending_test_3D_bead.py 50` (1-node baseline).
  `contact_bending_test_2node.py` is an independent, in-repo-regenerable
  2-node baseline (not used by the figures below, which use the external
  npz reference instead) — see `contact_bending_test_2node_README.md`.
- Figures: `postprocessing/fig1_geometry_evolution.py` (Fig. 1-style
  geometry snapshots), `postprocessing/fig_extended_arcs.py` (Fig. 2,
  the arc-contact-model visualization), `postprocessing/fig2_force_displacement.py`
  + `fig3_mesh_convergence.py` + `fig4_jitter_correlation.py` composed
  by `fig_combined_withd.py` (Fig. 5 panels a/b/c/d).
  Table 1: `postprocessing/fig6_jitter_metrics.py` (prints two tables;
  the first, "Jitter on |Sum F_y| vs t", is Table 1 — reproduces every
  digit of the published values).
- Data: none of it is tracked in git (2.2GB, over GitHub's size limit).
  `contact_output_files_3D_bead/` (1-node curve) regenerates via
  `contact_bending_test_3D_bead.py 50`; `external_reference_data/`
  (3-node curve) regenerates via `contact_bending_test.py 50`; the
  2-node baseline and the analytical-solution CSV have no generating
  script in this repo — see `external_reference_data/README.md`.

## Fig. 6, Fig. 7, Table 2, App. B.5 — `two_orthogonal_beams/`

- Runs: `contact_bending_soft_soft.py`; mesh sweep via
  `run_zavarise_mesh_convergence.py`; frictional mesh sweep via
  `friction/run_zavarise_friction_mesh_convergence.py` (mu=0.1, the
  coefficient used in the paper); the validated 29-step extension via
  `run_zavarise_steps29_Rthresh2_cluster.sh` (see
  `NOTES.md`'s TL;DR for why `--planar --eberly-straight-R-threshold 2e2`
  is needed).
- Figures: `postprocessing/fig_soft_soft.py` (6-step geometry +
  mesh-convergence metrics), `postprocessing/fig_validation_and_final_trace.py`
  (reproduces Fig. 7's 4 panels in one composite), `postprocessing/fig_planar_final_comparison.py`
  (broader mu=0/mu=0.1 comparison), `friction/plot_friction_mesh_convergence.py`
  (frictional mesh-convergence overlay).
  Jitter table: `postprocessing/table_soft_soft_jitter.py` (default =
  6-step; `--steps 29 --settling 2 --variant planarRth2` = the 29-step
  table backing the "three orders of magnitude" claim in Sec. 4.4).
  App. B.5: `postprocessing/pair_event_verification.py` ->
  `results/pair_event_verification.txt` (reproduces the quoted
  decimal-place figures exactly), figure `postprocessing/fig_pair_event_verification.py`.
- Data: `contact_output_files_soft_soft/` — pruned to the canonical
  6-step baseline and the final validated `_fix_planarRth2` 29-step
  traces (every superseded intermediate from the `--planar` /
  `--eberly-straight-R-threshold` investigation was removed; see
  `NOTES.md` for that history if you need it), plus
  `pair_events_prodblend_mu0.jsonl` (the App. B.5 raw log).

## Fig. 8, Fig. 9 — `net_pullout/`

- Runs: `contact_net_pullout.py` via `run_net_pullout_mesh_campaign.sh`
  (N=21..131 under mass-proportional Rayleigh damping, tag
  `_massdamp`) and `run_resumeE*_N{71,101,131}.sh` (Phase-E
  continuations for the three largest meshes).
- Figures: `abaqus/make_paper_figure_phase_grid_alpha_variants.py`
  (**Fig. 8**, the six-panel top-view comparison — reproduces the
  published figure, including the asymmetric extraction in panel f),
  `abaqus/make_paper_figure_convergence_jitter_alpha1000_final.py`
  (**Fig. 9**, clamp-force history vs. mesh refinement — all six N
  reach t=15.5s).
- Data: `contact_output_files_net_pullout_alternated/` (the 11 coords
  files these two scripts actually read, out of the much larger
  historical campaign); `abaqus/quadratic_C3D10M_alpha1000/` (the
  canonical Abaqus run: 298,586 C3D10M elements, 480,770 nodes,
  alpha=1000 Rayleigh damping) + `abaqus/quadratic_C3D10M_nodamping/net_pullout_alt_coarse_quad.inp`
  (shared mesh, needed for node-to-strand mapping only).
- See `net_pullout/NOTES.md` and `net_pullout/abaqus/README.md` for
  what was left out (an alpha-damping sensitivity sweep and an earlier
  linear-element baseline, neither in the paper) and why.

## Sec. 4.5 — `computational_cost/`

- `bench_cost_table.py` -> `cost_table.tex`; measured records in
  `bench_cost_results.jsonl` ({2,3}-node x N in {25,50,100,200}); per-run
  phase timings in `timing_json/`.
  Regenerate table only: `python bench_cost_table.py --table-only`.
  Full re-measure shells out to `two_orthogonal_beams/contact_bending_soft_soft.py`.

## Appendix A — `toy_cases_appendix/`

- A.1 jump = 2cos(alpha) verification: `jump_2cosalpha_check.py` ->
  `results/jump_2cosalpha_check.txt` (machine precision); figure
  `fig_jump_2cosalpha.py`.
- A.2 static multiplicity probe (Table A1, Fig. A3 schematic):
  `apex_multiplicity_probe.py` -> `results/apex_multiplicity.json`;
  figure `fig_multiplicity_schematic.py`.
- A.2 sliding test (Fig. A4): `harness_slide_2node.py` ->
  `results/r1..r6.npz`; metrics `analyze_perturbation.py`; figure
  `fig_combined_perturbation.py`.

## Not part of this paper

`double_spiral/` and `shell_validation/` from the development repo are
separate work (a thesis-manuscript benchmark and shell-element
validation, respectively) and are not included here.
