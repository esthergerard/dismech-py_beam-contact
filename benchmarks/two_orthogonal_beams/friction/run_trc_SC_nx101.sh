#!/bin/bash
### Job name
#SBATCH --job-name=dismech_trc_SC_nx101
#SBATCH --partition=cpu_long
#SBATCH --output=%x.o%j
#SBATCH --error=%x.o%j
### Per-dt-step trace of Zavarise mu=0.8 nx=101 with smooth Coulomb friction
### + Term-1 velocity Jacobian.  At nx=101 the bare contact-bending sim
### already takes ~40 hr (Eberly contact pair loop scales ~ nx^2 worst case,
### broad-phase culling helps but is not perfect).  Per-dt-trace adds another
### contact+friction re-evaluation per dt-step on top — call it ~50% extra,
### so 60 hr ish.  Allocate 3 days.
#SBATCH --time=3-00:00:00
### Memory
#SBATCH --mem=20gb
### Resources
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1

### Load Python module
module load python/3.14.0/gcc-15.1.0

### Activate venv
source /gpfs/workdir/gerarde/Contact_Model/Reg_Contact/dismech-py_regularized-contact/.venv/bin/activate

echo "Running Zavarise per-dt-trace, SC + Term-1 Jacobian, mu=0.8 nx=101"
date

python3 -u ../contact_bending_soft_soft.py 6 \
    --eberly-fric-jac \
    --per-dt-trace \
    --nx 101 \
    --mu 0.8 \
    --vel-tol 1e-3 \
    --tag _trc_SC_mu08_nx101

echo "Done"
date
