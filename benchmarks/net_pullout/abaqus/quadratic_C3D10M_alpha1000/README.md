# Quadratic C3D10M, alpha=1000 damping (beta=0) -- value used in the paper

The alpha value from the {1, 10, 50, 100, 1000} Rayleigh damping sweep
(../quadratic_C3D10M_alpha{1,10,50,100}/) that is carried into the
paper's comparison figures.

## Why alpha=1000

alpha = 1, 10, 50, 100 s^-1 produced no visible change in the Phase A-D
force response. Abaqus/Explicit's own built-in bulk viscosity
dissipates energy on every element regardless of the `*Damping` card,
and at these values the damping actually added
(`ALLVD(alpha) - ALLVD(alpha=0)`) stays below that baseline
dissipation's own run-to-run noise. Extrapolating from alpha=100 (the
strongest usable signal, ~3% of ALLIE at the end of Phase D) to where
the added damping becomes energetically non-negligible (~10% of ALLIE)
points to alpha ~ 300-1000 s^-1. alpha=1000 is the low end of that
range: the smallest value expected to produce a measurable effect,
chosen so the result reflects a genuine, if modest, damping
contribution rather than noise.

At this alpha, zeta = alpha/(2*omega) at the mesh's own
stability-limiting frequency (~1.33e6 rad/s) is ~3.8e-4 -- far below
the value that would require retuning the mass-scaling target.

## Difference from ../quadratic_C3D10M_nodamping/

Only `*Damping, alpha=1000` added to net_pullout_phaseA_engage.inp
(beta defaults to 0). Same quadratic mesh, same dt=1.5e-6 mass scaling.

## Mesh: reuse, do not regenerate

```bash
cp ../quadratic_C3D10M_nodamping/net_pullout_alt_coarse_quad.inp .
```

## Running

`run_phaseA_to_E_chained.sh` does not request `--exclusive`; confirm
node capacity before running alongside the other sweep trials.
