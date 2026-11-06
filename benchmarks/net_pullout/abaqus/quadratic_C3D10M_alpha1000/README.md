# Quadratic C3D10M, alpha=1000 damping

The Abaqus reference run used in the paper (Sec. 5.2): quadratic C3D10M
mesh, mass-proportional (Rayleigh) damping alpha=1000 s^-1, beta=0.

## Mesh

Reuses the shared mesh file rather than regenerating it:

```bash
cp ../quadratic_C3D10M_nodamping/net_pullout_alt_coarse_quad.inp .
```

## Running

```bash
sbatch run_phaseA_to_E_chained.sh
```
