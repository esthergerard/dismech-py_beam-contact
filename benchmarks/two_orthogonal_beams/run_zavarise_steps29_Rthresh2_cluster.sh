#!/bin/bash
### Job name
#SBATCH --job-name=dismech_zav29_Rthresh2
#SBATCH --partition=cpu_long
#SBATCH --output=%x_%a.o%j
#SBATCH --error=%x_%a.o%j
### Second iteration of the `eberly_straight_R_threshold` planar fix
### (supersedes run_zavarise_steps29_Rthresh_cluster.sh's threshold=1e3,
### which got much further than any prior attempt but still crashed
### once more at t~10.7 s / slide~0.32 m -- see NOTES.md "CLUSTER RESULT
### (2026-10-05)").
###
### THRESHOLD = 2e2 m (was 1e3).  Chosen and validated as follows: the
### 2026-10-05 run's residual failure was initially (incorrectly)
### attributed to Beam X's own curvature -- re-examination showed Beam
### X's curvature AT the actual contact point (its midpoint, node 10)
### is a small, safe R~72 m; the large R~1036 m seen at nodes 5/15 is
### 2.5 m away from the contact point, outside the broad-phase
### interaction range, and irrelevant to the active contact pairs.  The
### real cause of the threshold=1e3 residual failure is still not
### precisely pinned down, but EMPIRICALLY a lower threshold resolves
### it: reconstructing from the REAL step-10 state saved by the
### 2026-10-05 threshold=1e3 run (i.e. a state already computed under
### that run's exact physics, not an approximation) and continuing with
### the actual ImplicitEulerTimeStepper:
###   threshold=1e3 (repeat of the cluster run): fails again at the same
###     point as the real run (confirms the local reconstruction is
###     faithful)
###   threshold=2e2: converges CLEANLY for the full remaining 1900
###     dt-steps (to the t=29s/slide=0.87m end of the benchmark) with
###     zero anomalies, steady 3 Newton iterations throughout.
### 2e2 m is still far above Beam Z's near-zero-curvature crossing
### values (which blow up toward infinity) and comfortably below its
### baseline (~1.3e4 m), so Beam Z remains fully protected; it is lower
### than the previous value mainly to leave a wider safety margin against
### whatever moderate-R numerical sensitivity caused the 1e3 residual
### failure, without a precise mechanistic explanation for it yet.
###
### 4 tasks = the four jitter-table rows, planar + thresholded (2e2).
### REQUIRES the 2026-10-03 contact_bending_soft_soft.py (--planar,
### --eberly-straight-R-threshold) on the cluster.
#SBATCH --time=2-00:00:00
#SBATCH --mem=8gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --array=0-3

### Load Python module
module load python/3.14.0/gcc-15.1.0

### Activate venv
source /gpfs/workdir/gerarde/Contact_Model/Reg_Contact/dismech-py_regularized-contact/.venv/bin/activate

N_STEPS=29

case "${SLURM_ARRAY_TASK_ID}" in
  0) EXTRA="";                 TAG="_fix_planarRth2_eberly_mu0"  ;;
  1) EXTRA="--mu 0.1";         TAG="_fix_planarRth2_eberly_mu01" ;;
  2) EXTRA="--imc";            TAG="_fix_planarRth2_imc_mu0"     ;;
  3) EXTRA="--imc --mu 0.1";   TAG="_fix_planarRth2_imc_mu01"    ;;
  *) echo "bad array id ${SLURM_ARRAY_TASK_ID}"; exit 1 ;;
esac

echo "Running Zavarise per-dt-trace PLANAR+Rthreshold(2e2), steps=${N_STEPS}, nx=21, tag=${TAG}, extra='${EXTRA}'"
date

python3 -u contact_bending_soft_soft.py ${N_STEPS} \
    --nx 21 \
    --per-dt-trace \
    --planar \
    --eberly-straight-R-threshold 2e2 \
    --max-iter 800 \
    ${EXTRA} \
    --tag "${TAG}"

echo "Done"
date

### Collect (from the repo root on your Mac):
###   scp "gerarde@ruche.mesocentre.universite-paris-saclay.fr:/gpfs/workdir/gerarde/Contact_Model/Reg_Contact/dismech-py_regularized-contact/benchmarks/two_orthogonal_beams/contact_output_files_soft_soft/zavarise_*_steps29_fix_planarRth2_*.txt" /Users/esthergerard/Documents/Recherche/Contact_Model/dismech-py_regularized-contact/benchmarks/two_orthogonal_beams/contact_output_files_soft_soft/
### then:
###   python postprocessing/table_soft_soft_jitter.py --steps 29 --settling 2 --variant planarRth2
