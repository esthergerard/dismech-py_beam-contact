# External reference data

This folder is empty in the repo: the reference curves used by
`fig2_force_displacement.py`, `fig4_jitter_correlation.py`, and
`fig6_jitter_metrics.py` are too large to track in git. The scripts
skip any curve whose data file is missing rather than failing.

To regenerate the data files used here:

```bash
cd ..
python3 contact_bending_test.py 50
```

This writes `reg_beam_coords_N50.txt` and `reg_contact_forces_N50.txt`
into this folder (dt=1e-3, t≈1000s — a long run).

The one-node baseline (`../contact_output_files_3D_bead/reg_beam_coords_N50_bead.txt`
+ `reg_contact_forces_N50_bead.txt`) is regenerated the same way:

```bash
python3 contact_bending_test_3D_bead.py 50
```
