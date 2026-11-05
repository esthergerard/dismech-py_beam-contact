#!/bin/bash
#SBATCH --job-name=net_pullout_abq_chain_a1000
#SBATCH --output=%x.o%j
#SBATCH --error=%x.o%j
#SBATCH --partition=debug
#SBATCH --nodelist=citran
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
### ALPHA=1000 DAMPING TRIAL variant of the Phase A-E restart chain --
### follow-up to the {1,10,50,100} sweep in ../quadratic_C3D10M_alpha
### {1,10,50,100}/, which showed ZERO visible effect at any of those
### values (even alpha=100 was indistinguishable from undamped). Same
### mesh/dt as ../quadratic_C3D10M_nodamping/, now adding pure mass-
### proportional Rayleigh damping *Damping, alpha=1000 only (beta=0) --
### see the header comment in net_pullout_phaseA_engage.inp for the
### energy-based (ALLVD/ALLIE) derivation this value is testing, and
### ./README.md for the full rationale and how to compare the result
### against dismech's own Phase-C decay shape at the end.
###
### NOTE: no --exclusive here (same reasoning as the earlier sweep's
### scripts) -- confirm with whoever manages citran that this fits
### within the node's core/memory budget alongside whatever else is
### already running before submitting.
###
### BEFORE running this: copy the mesh in (same quadratic mesh as ../
### quadratic_C3D10M_nodamping/, do NOT regenerate):
###   cp ../quadratic_C3D10M_nodamping/net_pullout_alt_coarse_quad.inp .
### (or copy directly from wherever your other alpha-sweep folders keep
### it, if that's more convenient on citran.)
###
### Submit FROM THIS FOLDER:
###   sbatch run_phaseA_to_E_chained.sh
### Monitor:
###   squeue -u $USER
###   tail -f net_pullout_abq_chain_a1000.o<jobid>
###
### TO RESUME AFTER A CRASH: identical recipe to every other chain script
### in this project --
###  - job N failed outright: just resubmit; completed phases are
###    skipped (per-job "already completed" guard below).
###  - resume job N mid-step from its last intra-step checkpoint: delete
###    ONLY that job's .odb and resubmit with
###      sbatch --export=ALL,RESUME_INPLACE=net_pullout_phaseE_pullout \
###        run_phaseA_to_E_chained.sh
###
set -euo pipefail

### Load Abaqus
module load abaqus-2026

CPUS=${SLURM_CPUS_PER_TASK:-4}
RESUME_INPLACE=${RESUME_INPLACE:-}

# job_name  input_file                       oldjob (empty = fresh model)
JOBS=(
  "net_pullout_phaseA_engage   net_pullout_phaseA_engage.inp   "
  "net_pullout_phaseB_hold     net_pullout_phaseB_hold.inp     net_pullout_phaseA_engage"
  "net_pullout_phaseC_release  net_pullout_phaseC_release.inp  net_pullout_phaseB_hold"
  "net_pullout_phaseD_free     net_pullout_phaseD_free.inp     net_pullout_phaseC_release"
  "net_pullout_phaseE_pullout  net_pullout_phaseE_pullout.inp  net_pullout_phaseD_free"
)

for line in "${JOBS[@]}"; do
    read -r job input oldjob <<< "$line"

    job_completed() {
        [ -f "${1}.sta" ] && grep -q "THE ANALYSIS HAS COMPLETED SUCCESSFULLY" "${1}.sta"
    }

    if [ "$job" = "$RESUME_INPLACE" ]; then
        echo "=== Resuming $job IN PLACE from its last intra-step checkpoint ==="
        oldjob="$job"
    elif job_completed "$job"; then
        echo "=== $job already completed (per ${job}.sta) -- skipping ==="
        continue
    else
        echo "=== Running $job (oldjob='${oldjob:-<fresh model>}') ==="
    fi

    if [ -z "$oldjob" ]; then
        abaqus job="$job" input="$input" double=both cpus="$CPUS" interactive
    else
        abaqus job="$job" oldjob="$oldjob" input="$input" double=both cpus="$CPUS" interactive
    fi

    if ! job_completed "$job"; then
        echo "!!! $job did NOT complete (per ${job}.sta) -- treating as failed, stopping chain."
        echo "!!! Clean up its partial files before retrying: rm -f ${job}.*"
        exit 1
    fi
    echo "=== $job complete: ${job}.odb is final and independent of all later phases ==="
done

echo "All 5 phases complete."
