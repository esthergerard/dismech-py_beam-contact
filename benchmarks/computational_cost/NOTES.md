# Computational cost

Cost decomposition for the two-node vs. three-node contact model (paper
Table 2), measured on the two orthogonal beams benchmark.

```bash
python3 bench_cost_table.py --table-only
```

Reads `bench_cost_results.jsonl` and `timing_json/` and writes
`cost_table.tex`. Full re-measurement re-runs
`two_orthogonal_beams/contact_bending_soft_soft.py` at each mesh size
(drop `--table-only`).
