# Abaqus solid finite-element reference (paper Section 5.2)

The reference model sweeps each of the 10 strands as a solid tube along
its centerline and meshes it with **298,586 quadratic tetrahedra
(C3D10M)**, totaling **480,770 nodes**. It is run in Abaqus/Explicit
with general contact over all exterior surfaces, default hard normal
behaviour, no contact stabilization, and mass-proportional (Rayleigh)
damping **alpha = 1000 s^-1** — see `quadratic_C3D10M_alpha1000/README.md`
for why that value was chosen. The same five loading phases (A-E) and
the same extraction as the rod model are imposed.

## What's in this folder

- `quadratic_C3D10M_alpha1000/` — the canonical run: the five `.inp`
  phase decks plus every extracted CSV (`*_final_shape.csv`,
  `*_shape_t*.csv`, `*_central_clamp.csv`, `*_energies.csv`) used by
  Fig. 8 and Fig. 9. The raw `.odb` database is not included (multi-GB
  binary, regenerates from the `.inp` decks with a licensed Abaqus
  install) — the CSVs are the reference data.
- `quadratic_C3D10M_nodamping/` — only the shared mesh file
  `net_pullout_alt_coarse_quad.inp` (298,586 elements / 480,770 nodes),
  needed by the Fig. 8 script to map node labels to strand identity.
- `net_pullout_gmsh.py` — generates the quadratic mesh from scratch.
- `extract_final_shape.py`, `extract_shape_near_time.py`,
  `extract_phaseA_to_E_chained_odb.py` — Abaqus-Python extraction
  scripts that produce the CSVs above from a `.odb` (require a licensed
  Abaqus install; not needed to reproduce the figures from the CSVs
  already here).
- `make_paper_figure_phase_grid_alpha_variants.py` — **Figure 8**.
- `make_paper_figure_convergence_jitter_alpha1000_final.py` — **Figure 9**.

## Reproducing the figures

Both figure scripts read CSV/text data already in this repo — no
Abaqus license is needed to regenerate Fig. 8 or Fig. 9:

```bash
python3 make_paper_figure_phase_grid_alpha_variants.py
python3 make_paper_figure_convergence_jitter_alpha1000_final.py
```
