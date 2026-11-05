# External reference data for the three-point bending figures (Fig. 2, Fig. 5)

These four files feed `postprocessing/fig2_force_displacement.py` and
`postprocessing/fig4_jitter_correlation.py`. They are reference/baseline
runs rather than code, which is why they're kept as data instead of being
regenerated on every clone.

| file | role | regenerable? |
|---|---|---|
| `impV_noD_contact_bending_N50.npz` | 2-node (segment) model, N=50, full run to t=900s | yes, with the 2-node contact model (not included in this repo — this is the pre-existing baseline run) |
| `exact_rL_0p125.csv` | Batista analytical solution, r/L=0.125 | reference data, not a simulation output |
| `reg_beam_coords_N50.txt` | 3-node (ours) model, N=50, beam centerline vs time | **yes** — `python contact_bending_test.py 50` reproduces this file byte-for-byte in format (same output filename, same E/nu/dt/save_interval as currently in the script) |
| `reg_contact_forces_N50.txt` | 3-node (ours) model, N=50, per-node contact force vs time | **yes** — same command as above |

A full `contact_bending_test.py 50` run integrates to t≈1000s at dt=1e-3
with save_interval=0.01s (1e6 time steps), so it is a long run; the
precomputed files are shipped here to avoid requiring that rerun just to
reproduce the figures. `fig2_force_displacement.py` and
`fig4_jitter_correlation.py` both read from this folder by default.

Note on `fig4_jitter_correlation.py`: an earlier version of this script
read a separate 9.37GB `contact_bending_test_results.npz`. That file is
not needed — the 2-node run above is dense enough (dt=0.01s) to cover the
analysis window (t=[600,800]s) and uses the same on-disk schema, so it has
been pointed at `impV_noD_contact_bending_N50.npz` instead.
