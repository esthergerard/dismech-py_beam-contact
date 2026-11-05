#!/bin/bash
#SBATCH --job-name=resumeE_N71
#SBATCH --output=%x.o%j
#SBATCH --error=%x.o%j
#SBATCH --partition=cpu_long
#SBATCH --time=4-00:00:00
#SBATCH --mem=64gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
### Resume N=71 (eberly, massdamp recipe) from its Phase-D checkpoint and
### retry Phase E with the MINIMAL intervention: finer dt_pullout and a
### higher Newton-iteration cap, keeping ADAPTIVE k_c (N=71 never needed
### k_c pinning before -- the original run failed for the first time
### only after the mass-proportional-damping fix, with "Iteration limit
### 200 reached" while the residual was declining very slowly (0.006 ->
### 0.002 over the last ~20 iters, not diverging). This tests whether
### that alone (not adaptive-k_c tunneling) was the bottleneck.
###
### If this ALSO fails, the next step is to add --fixed-kc 9.0355
### (this run's own Phase-D checkpoint kc) like N101/N131 below.
###
### Submit FROM THIS FOLDER (benchmarks/net_pullout/):
###   sbatch run_resumeE_N71.sh

module load python/3.14.0/gcc-15.1.0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

REPO=/gpfs/workdir/gerarde/Contact_Model/Reg_Contact/dismech-py_regularized-contact
source "$REPO/.venv/bin/activate"
cd "$REPO/benchmarks/net_pullout"

CKPT="contact_output_files_net_pullout_alternated/pullout_checkpoint_eberly_N71_alt_soft_eberly_N71_massdamp.npz"

echo "============================================================================"
echo "  N71 RESUME-E (adaptive k_c kept, finer dt_pullout, higher max_newton_iter)"
echo "  resuming from $CKPT"
echo "  host = $(hostname)   start = $(date -Iseconds)"
echo "============================================================================"

python3 -u contact_net_pullout.py 71 \
    --model eberly \
    --youngs 1e7 --nu 0.45 --density 1100 \
    --weave-amplitude 3.0e-3 \
    --damping-eta 1.0e9 --alternating-weave --mu 0.0 \
    --vel-tol 1e-3 --d-imc 5e-5 --newton-tol 1e-4 \
    --end-disp-engage 3.75e-3 --end-disp-pullout 30e-3 \
    --t-engage 1.0 --t-pullout 12.0 \
    --dt 1e-2 --dt-release 5e-3 --dt-pullout 2.5e-3 \
    --solver pardiso --use-line-search --line-search-iters 0 --no-plot \
    --eberly-h-scale \
    --max-newton-iter 500 \
    --checkpoint-interval 1.0 \
    --resume-pullout "$CKPT" \
    --tag "_alt_soft_eberly_N71_massdamp_resumeE"

echo "============================================================================"
echo "  end = $(date -Iseconds)"
echo "============================================================================"
