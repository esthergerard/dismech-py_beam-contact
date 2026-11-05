#!/bin/bash
#SBATCH --job-name=resumeE2_N131
#SBATCH --output=%x.o%j
#SBATCH --error=%x.o%j
#SBATCH --partition=cpu_long
#SBATCH --time=4-00:00:00
#SBATCH --mem=64gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
### CONTINUATION of run_resumeE_N131.sh (job 2176429), which was
### CANCELLED DUE TO TIME LIMIT at t=12.5-12.6s (~81% through Phase E),
### not a crash. Resumes from its latest periodic checkpoint (t=12.500)
### with the SAME settings (fixed k_c, finer dt_pullout, higher
### max_newton_iter).
###
### NEW TAG (_massdamp_resumeE2, not _massdamp_resumeE): the script
### always opens coords/forces/force_history in OVERWRITE mode, even on
### --resume-pullout, so reusing the old tag would truncate the Phase-E
### data already collected (t=3.5 -> 12.5ish) from the first resumeE
### run. Analysis must stitch THREE segments: original _massdamp
### (phases A-D) + _massdamp_resumeE (phase E, t=3.5->12.5ish) +
### _massdamp_resumeE2 (phase E, t=12.5->15.5).
###
### This phase still has ~3s of simulated time left (vs N101's ~1s) --
### may need a resumeE3 round too if it doesn't finish in 4 more days.
###
### Submit FROM THIS FOLDER (benchmarks/net_pullout/):
###   sbatch run_resumeE2_N131.sh

module load python/3.14.0/gcc-15.1.0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

REPO=/gpfs/workdir/gerarde/Contact_Model/Reg_Contact/dismech-py_regularized-contact
source "$REPO/.venv/bin/activate"
cd "$REPO/benchmarks/net_pullout"

CKPT="contact_output_files_net_pullout_alternated/pullout_checkpoint_eberly_N131_alt_soft_eberly_N131_massdamp_resumeE_t12.500.npz"

echo "============================================================================"
echo "  N131 RESUME-E2 (continuation after time-limit cancellation)"
echo "  resuming from $CKPT"
echo "  host = $(hostname)   start = $(date -Iseconds)"
echo "============================================================================"

python3 -u contact_net_pullout.py 131 \
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
    --fixed-kc 15.8703 --max-newton-iter 500 \
    --checkpoint-interval 1.0 \
    --resume-pullout "$CKPT" \
    --tag "_alt_soft_eberly_N131_massdamp_resumeE2"

echo "============================================================================"
echo "  end = $(date -Iseconds)"
echo "============================================================================"
