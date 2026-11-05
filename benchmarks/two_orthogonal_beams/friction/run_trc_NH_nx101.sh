#!/bin/bash
### Job name
#SBATCH --job-name=dismech_trc_NH_nx101
#SBATCH --partition=cpu_long
#SBATCH --output=%x.o%j
#SBATCH --error=%x.o%j
### Per-dt-step trace of Zavarise mu=0.8 nx=101 with Norton-Hoff capped
### friction (rho=0.1, v_ref=0.03 m/s — chosen so both NH and SC saturate
### at mu*|F_N| at the Zavarise sliding speed v_slide=0.03 m/s) + Term-1
### velocity Jacobian.  Same sizing as the SC trace job (~60 hr).
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

echo "Running Zavarise per-dt-trace, NH-capped (rho=0.1, vref=0.03) + Term-1 Jacobian, mu=0.8 nx=101"
date

python3 -u ../contact_bending_soft_soft.py 6 \
    --eberly-fric-jac \
    --friction-law norton_hoff_capped \
    --nh-rho 0.1 \
    --nh-vref 0.03 \
    --per-dt-trace \
    --nx 101 \
    --mu 0.8 \
    --vel-tol 1e-3 \
    --tag _trc_NH_mu08_nx101

echo "Done"
date
