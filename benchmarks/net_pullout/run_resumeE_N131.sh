#!/bin/bash
#SBATCH --job-name=resumeE_N131
#SBATCH --output=%x.o%j
#SBATCH --error=%x.o%j
#SBATCH --partition=cpu_long
#SBATCH --time=4-00:00:00
#SBATCH --mem=64gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
### Resume N=131 (eberly, massdamp recipe) from its Phase-D checkpoint
### and retry the FULL Phase E (not truncated, unlike the earlier
### run_n131_debug_fallback.sh diagnostic) with the historical fix set:
### fixed k_c (adaptive k_c tunneling at N>=51 is a long-standing KNOWN
### ISSUE, not new), finer dt_pullout, higher max_newton_iter -- same
### recipe that got furthest before, now combined with the new
### mass-proportional damping (eta=1e9).
###
### --fixed-kc 15.8703 = this checkpoint's own adaptive k_c at end of
### Phase D (read directly from the .npz).
###
### N=131 has never cleanly completed Phase E even before this damping
### fix (see run_n131_debug_fallback.sh and NOTES) -- treat this as
### another attempt at an open problem, not a regression to "undo".
###
### Submit FROM THIS FOLDER (benchmarks/net_pullout/):
###   sbatch run_resumeE_N131.sh

module load python/3.14.0/gcc-15.1.0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

REPO=/gpfs/workdir/gerarde/Contact_Model/Reg_Contact/dismech-py_regularized-contact
source "$REPO/.venv/bin/activate"
cd "$REPO/benchmarks/net_pullout"

CKPT="contact_output_files_net_pullout_alternated/pullout_checkpoint_eberly_N131_alt_soft_eberly_N131_massdamp.npz"

echo "============================================================================"
echo "  N131 RESUME-E (fixed k_c, finer dt_pullout, higher max_newton_iter)"
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
    --tag "_alt_soft_eberly_N131_massdamp_resumeE"

echo "============================================================================"
echo "  end = $(date -Iseconds)"
echo "============================================================================"
