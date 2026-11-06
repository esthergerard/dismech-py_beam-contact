# Woven-strand net pull-out

Plain-weave 5x5 net of soft elastomer strands (E=10 MPa, nu=0.45,
rho=1100 kg/m^3), loading phases A-E (paper Sec. 5).

## DisMech (rod model) side

`contact_net_pullout.py` is the simulation script. `run_net_pullout_mesh_campaign.sh`
runs the N=21..131 mesh sweep; `run_resumeE*_N{71,101,131}.sh` continue
the three largest meshes through Phase E. Output coordinates land in
`contact_output_files_net_pullout_alternated/`.

The clamp force used in the figures is computed directly from the
coordinate-derived axial strain in the first free segment adjacent to
the clamp (see paper Sec. 5.2) — no separate forces file is needed.

## Abaqus (solid FE reference) side

`abaqus/quadratic_C3D10M_alpha1000/` is the reference run (298,586
C3D10M elements, 480,770 nodes, alpha=1000 Rayleigh damping); see
`abaqus/README.md` for the extraction recipe.

## Figures

| output | script |
|---|---|
| Fig. 8 (six-panel top view, DisMech vs. Abaqus) | `abaqus/make_paper_figure_phase_grid_alpha_variants.py` |
| Fig. 9 (clamp-force history, mesh convergence) | `abaqus/make_paper_figure_convergence_jitter_alpha1000_final.py` |
