# Woven-strand net pull-out (paper Section 5) — reproduction notes

Plain-weave 5x5 net of soft elastomer strands (E=10 MPa, nu=0.45,
rho=1100 kg/m^3), loading phases A-E as described in the paper. This
file documents the canonical, current data/script set shipped in this
repo. (An earlier, much longer development history — a stiffer "nylon"
material campaign that could not complete Phase E, and a damping bug
that made the mesh-convergence sweep non-apples-to-apples — is not
reproduced here; it predates the setup actually used for the paper and
is not needed to regenerate the published figures.)

## DisMech (rod model) side

- `contact_net_pullout.py` is the simulation script. `run_net_pullout_mesh_campaign.sh`
  runs the full N=21..131 sweep under the current mass-proportional
  Rayleigh damping (tag `_massdamp`); `run_resumeE*_N{71,101,131}.sh`
  continue the three largest meshes through Phase E in stitched segments
  (tags `_massdamp_resumeE`, `_massdamp_resumeE2`).
- `contact_output_files_net_pullout_alternated/` ships exactly the coords
  files `make_paper_figure_convergence_jitter_alpha1000_final.py` (Fig. 9)
  and `make_paper_figure_phase_grid_alpha_variants.py` (Fig. 8) read:
  the `_massdamp` file for each of N=21/31/51/71/101/131, plus the two
  `_resumeE`/`_resumeE2` continuation files for N=71/101/131.
- The clamp force in both figures is computed directly from the
  coordinate-derived axial strain `EA*(stretch-1)` in the first free
  segment adjacent to the clamp (see Section 5.2 of the paper for why
  this equals the reaction force under quasi-static loading) — no
  separate forces file is needed.

## Abaqus (solid FE reference) side

See `abaqus/quadratic_C3D10M_alpha1000/README.md` for the canonical run
(298,586 C3D10M elements, 480,770 nodes, alpha=1000 Rayleigh damping —
the value used in the paper) and `abaqus/README.md` for the extraction
recipe. Only that run plus the shared mesh file (from
`quadratic_C3D10M_nodamping/`) are kept; the alpha-sensitivity sweep
(alpha in {0, 1, 10, 50, 100}) and the archived linear-element (C3D4)
baseline were exploratory and are not part of the paper.

## Figures

- **Fig. 8** (six-panel top view, DisMech vs. Abaqus): `abaqus/make_paper_figure_phase_grid_alpha_variants.py`
- **Fig. 9** (clamp-force history, mesh convergence vs. Abaqus): `abaqus/make_paper_figure_convergence_jitter_alpha1000_final.py`

Both were re-run against the exact file set shipped here to confirm
they reproduce the published figures before this repo was finalized.
