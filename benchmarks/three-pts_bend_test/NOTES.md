# Three-point bending test

A beam is driven upward at its left end against a fixed circular
obstacle (paper Sec. 3). Three contact-geometry variants are compared.

| script | contact primitive | output folder |
|---|---|---|
| `contact_bending_test.py` | three-node arc-to-arc (this paper's model) | `contact_output_files/` |
| `contact_bending_test_3D_bead.py` | one-node bead | `contact_output_files_3D_bead/` |
| `contact_bending_test_2node.py` | two-node segment-to-segment IMC | `contact_output_files_2node/` |

`contact_bending_test_2node.py` is an independent two-node baseline
built on dismech's own self-contact engine; it is not used to produce
the figures below.

## Figures and tables

Scripts live in `postprocessing/`. See `external_reference_data/README.md`
for how to regenerate the data they read.

| output | script |
|---|---|
| Fig. 2 (arc-contact-model illustration) | `fig_extended_arcs.py` |
| Fig. 4 (beam geometry over time) | `fig1_geometry_evolution.py` |
| Fig. 5 (force-displacement, zoom, jitter, mesh convergence) | `fig_combined.py` |
| Table 1 (jitter metrics) | `fig6_jitter_metrics.py` |

`fig_combined.py` imports `fig2_force_displacement.py`,
`fig3_mesh_convergence.py`, and `fig4_jitter_correlation.py` as building
blocks rather than being run standalone.
