# Shared quadratic mesh (mesh file only)

This folder intentionally holds only `net_pullout_alt_coarse_quad.inp`,
the quadratic C3D10M mesh (298,586 elements, 480,770 nodes) shared by
every alpha-damping variant. The full `quadratic_C3D10M_nodamping`
campaign (its own alpha=0 Phase A-E run) is not part of the paper and
was not carried into this repo -- only the mesh geometry is needed, by
`make_paper_figure_phase_grid_alpha_variants.py`, to map node labels to
beam/strand identity when drawing the Fig. 8 panels.

See `../quadratic_C3D10M_alpha1000/README.md` for the canonical run.
