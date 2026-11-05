# Three-point bending test with contact — history and conclusions

A beam is driven upward at its left end against a fixed circular
obstacle: contact + large bending in one controlled problem. Used as the
first validation of the regularized contact force and its Jacobian.

Two variants are kept (the paper's three-node vs. one-node comparison;
the 2-node baseline is a separate, external simulation — see "Paper
data provenance" below):

| script | contact primitive | outputs |
|---|---|---|
| `contact_bending_test.py` | three-node arc-to-arc, 3D model constrained to a plane (Sec. 3.1) | `contact_output_files/` |
| `contact_bending_test_3D_bead.py` | one-node bead (node-to-arc distance) | `contact_output_files_3D_bead/` |

## What was tried

- Velocity-controlled loading (no quasi-static criterion), run times
  extended to match the real experiment duration.
- Output refactored from npz to txt
  (`converged_version_contact_bending_test_results.npz` kept as the
  frozen reference solution from the converged 2D version).
- Contact-force computation fixed (3944ef3), then forces scaled w.r.t.
  the force residual; initial curvature increased to avoid
  curvature-sign switches in the arc fit.
- **Restart feature** for wall-time-interrupted runs
  (`--restart`, used by `run_discretization.sh`, array over mesh
  densities 300/500 nodes).
- 3D implementation: first attempt, then z-coordinate fix of the
  equivalent 3D configuration + Hessian fix in the 3D model (161ab39).
- Entrance into the contact zone softened + simulation sped up
  (e75b66e); bead model implemented last (b8be14b) as the cheap
  comparison primitive.

## Conclusions

- 2D and 3D-arc agree once the 3D z-geometry and Hessian were fixed —
  this validated the 3D arc-to-arc force/Jacobian implementation.
- The bead variant is the cheap fallback: same trend, less accurate
  near grazing contact, no arc-degeneracy issues.
- Mesh study run at 300/500 nodes via the restart-capable cluster
  script.

## Paper data provenance

Figure scripts live in `postprocessing/`; all data they read now lives
in `external_reference_data/` (see that folder's README for provenance
and which files are regenerable):

| paper figure | script | data file(s) |
|---|---|---|
| Fig 1 (geometry evolution, 3-node) | `fig1_geometry_evolution.py` | `external_reference_data/reg_beam_coords_N50.txt` |
| Fig 1b (1-node bead) | same script | `contact_output_files_3D_bead/reg_beam_coords_N50_bead.txt` |
| Fig 2 (force-displacement, 3 curves) | `fig2_force_displacement.py` | 1-node: `contact_output_files_3D_bead/reg_beam_coords_N50_bead.txt` + `reg_contact_forces_N50_bead.txt`; 2-node: `external_reference_data/impV_noD_contact_bending_N50.npz`; 3-node: `external_reference_data/reg_beam_coords_N50.txt` + `reg_contact_forces_N50.txt`; analytical: `external_reference_data/exact_rL_0p125.csv` |
| Fig 3 (mesh convergence) | `fig3_mesh_convergence.py` | values HARDCODED in the script (`THREE_NODE` / `TWO_NODE` lists), transcribed from the N=5..100 runs |
| Fig 5c / jitter | `fig4_jitter_correlation.py` | `external_reference_data/impV_noD_contact_bending_N50.npz` (same 2-node file Fig. 2 uses — verified dense enough at dt=0.01s to cover the [600,800]s analysis window) |
| Table 1 / jitter metrics | `fig6_jitter_metrics.py` | same files as Fig. 2 (prints and writes both a time-domain table that reproduces Table 1 exactly, and a displacement-domain variant) |
| extended-arcs illustration | `fig_extended_arcs.py` | no data (geometry construction via `src/dismech`) |

Re-verified while preparing this repo: `fig6_jitter_metrics.py`'s
time-domain table reproduces every digit of the paper's Table 1 (1-node:
1.4e-2/2.9e-5/7.4e-3/1.4e-3/4.9e-4; 2-node: 3.1e-3/3.1e-6/1.9e-3/8.2e-5/1.3e-4;
3-node: 7.8e-6/6.1e-7/1.8e-6/9.2e-7/1.8e-6).
