# Two orthogonal beams

Validation against Example 2 of Zavarise & Wriggers (2000), *Contact with
friction between beams in 3-D space*, IJNME 49(8):977-1006 (paper Sec. 4):
two orthogonal beams, off-centre crossing, loaded into contact.

Main script: `contact_bending_soft_soft.py` (`n_steps` positional argument;
6 = paper's Case 1, 29 = paper's Case 2, continuing the same loading to
look at sustained sliding). Mesh convergence: `run_zavarise_mesh_convergence.py`.
Frictional mesh convergence: `friction/run_zavarise_friction_mesh_convergence.py`
(mu=0.1).

Running the 29-step case needs `--planar --eberly-straight-R-threshold 2e2`
to converge cleanly through the full load history.

## Figures and tables

Scripts live in `postprocessing/` (and `friction/` for the frictional
mesh-convergence plot).

| output | script |
|---|---|
| Fig. 6 (geometry) | `fig_soft_soft.py` |
| Fig. 7 (6-step and 29-step comparison panels) | `fig_validation_and_final_trace.py` |
| Frictional mesh-convergence overlay | `friction/plot_friction_mesh_convergence.py` |
| Zavarise jitter table | `table_soft_soft_jitter.py` (`--steps 29 --settling 2 --variant planarRth2` for the 29-step table) |
| Table B2, App. B.5 active-set verification | `pair_event_verification.py` |
