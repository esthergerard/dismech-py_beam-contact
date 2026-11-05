#!/bin/bash
#SBATCH --job-name=resumeE2_N101
#SBATCH --output=%x.o%j
#SBATCH --error=%x.o%j
#SBATCH --partition=cpu_long
#SBATCH --time=4-00:00:00
#SBATCH --mem=64gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
### CONTINUATION of run_resumeE_N101.sh (job 2176428), which was
### CANCELLED DUE TO TIME LIMIT at t=14.5-15.05s (96-97% through Phase
### E), not a crash. Resumes from its latest periodic checkpoint
### (t=14.500) with the SAME settings (fixed k_c, finer dt_pullout,
### higher max_newton_iter).
###
### NEW TAG (_massdamp_resumeE2, not _massdamp_resumeE): the script
### always opens coords/forces/force_history in OVERWRITE mode, even on
### --resume-pullout, so reusing the old tag would truncate the Phase-E
### data already collected (t=3.5 -> 14.5) from the first resumeE run.
### Analysis must stitch THREE segments: original _massdamp (phases
### A-D) + _massdamp_resumeE (phase E, t=3.5->14.5ish) +
### _massdamp_resumeE2 (phase E, t=14.5->15.5).
###
### Submit FROM THIS FOLDER (benchmarks/net_pullout/):
###   sbatch run_resumeE2_N101.sh

module load python/3.14.0/gcc-15.1.0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

REPO=/gpfs/workdir/gerarde/Contact_Model/Reg_Contact/dismech-py_regularized-contact
source "$REPO/.venv/bin/activate"
cd "$REPO/benchmarks/net_pullout"

CKPT="contact_output_files_net_pullout_alternated/pullout_checkpoint_eberly_N101_alt_soft_eberly_N101_massdamp_resumeE_t14.500.npz"

echo "============================================================================"
echo "  N101 RESUME-E2 (continuation after time-limit cancellation)"
echo "  resuming from $CKPT"
echo "  host = $(hostname)   start = $(date -Iseconds)"
echo "============================================================================"

python3 -u contact_net_pullout.py 101 \
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
    --fixed-kc 12.4447 --max-newton-iter 500 \
    --checkpoint-interval 1.0 \
    --resume-pullout "$CKPT" \
    --tag "_alt_soft_eberly_N101_massdamp_resumeE2"

echo "============================================================================"
echo "  end = $(date -Iseconds)"
echo "============================================================================"
