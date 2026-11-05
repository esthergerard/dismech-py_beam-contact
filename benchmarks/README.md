# Benchmarks

One folder per benchmark in the paper. Each contains the simulation
script(s), their cluster launchers (`run_*.sh` — submit from inside the
folder), the data needed to reproduce the paper's figures/tables, a
`postprocessing/` or `figures/` folder with the figure-generating
scripts, and a `NOTES.md` with provenance/history.

See `PAPER_MAP.md` for the exact figure/table -> script -> data mapping,
and `paper_reproducibility_params.md` for every simulation parameter
(material, time step, contact stiffness, etc.) with its source location.

| folder | paper section | produces |
|---|---|---|
| `three-pts_bend_test/` | Sec. 3 | Fig. 2, Fig. 5, Table 1 |
| `two_orthogonal_beams/` (incl. `friction/`) | Sec. 4 | Fig. 6, Fig. 7, Table 2, App. B.5 |
| `net_pullout/` (incl. `abaqus/`) | Sec. 5 | Fig. 8, Fig. 9 |
| `computational_cost/` | Sec. 4.5 | cost-decomposition table |
| `toy_cases_appendix/` | App. A | Fig. A1, Fig. A2, Fig. A3, Fig. A4, Table A1 |

All data needed to run the figure/table scripts is included in the
repo (no external paths) except where a script says otherwise.
