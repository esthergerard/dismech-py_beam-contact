# `contact_bending_test_2node.py`

Independent two-node (segment-to-segment) contact baseline for the
three-point bending test, built on dismech's own self-contact IMC
engine (`env.add_force('selfContact', ...)`) — the same mechanism used
for two-node contact in `two_orthogonal_beams/` and `net_pullout/`.

The circular obstacle is discretized into fixed nodes (200 by default,
matching Sec. 3.1) and combined with the beam into one geometry; contact
is resolved as edge-to-edge IMC contact between beam and circle
segments.

## Running it

```bash
cd benchmarks/three-pts_bend_test
python3 contact_bending_test_2node.py 50       # N=50 beam nodes
python3 contact_bending_test_2node.py 10       # quicker, coarser run
```

Output goes to `contact_output_files_2node/reg_{beam_coords,contact_forces,summary}_N{n}_2node.txt`.
Supports `--restart` to resume an interrupted run, and
`--n-circle-nodes` to change the obstacle discretization.
