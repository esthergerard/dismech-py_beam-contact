#!/bin/bash
### Job name
#SBATCH --job-name=net_pullout_mesh
#SBATCH --output=%x.o%j_%a
#SBATCH --error=%x.o%j_%a
#SBATCH --partition=cpu_long
### MESH-CONVERGENCE CAMPAIGN, alternated weave, soft recipe.
###
### REDONE FOR THE MASS-PROPORTIONAL DAMPING FIX (2026-10-21): the old
### `compute_damping_force` applied a FLAT per-node coefficient
### (F = -eta * v, rod nodes effectively 0.005*eta), so the total damping
### force on a beam scaled with n_nodes_per_beam instead of its physical
### mass -- every N in this campaign was actually running with a
### DIFFERENT effective damping level. Fixed in
### src/dismech/external_forces/damping.py to mass-proportional Rayleigh
### damping (F = -eta * m_i * v_i); `eta` is now a RATE in 1/s and is
### mesh-independent by construction. ALL N below (including N=21/51,
### previously skipped as "already exist") are rerun from scratch under
### the new, consistent damping so the mesh-convergence comparison is
### apples-to-apples. Old output files from before this fix are stale
### for any eta-sensitive quantity (settling transients, Newton iter
### counts) and should not be mixed with these `_massdamp`-tagged runs.
###
### DAMPING_ETA=1.0e9 (1/s). The per-DOF coefficient c_i = eta * m_i now
### shrinks at finer N (m_i ~ 1/(N-1)), unlike the old flat per-node
### coefficient (always 0.005*500 = 2.5 N.s/m/node) -- so naively
### reusing the N=31-equivalent eta (2.0e8) would leave N=131 LESS
### locally damped per-node than it used to be, right where Newton
### already struggles most. Anchored instead against N=131 (c_i there
### would need eta ~ 8.4e8 to match the old 2.5 N.s/m/node) plus ~20%
### margin, so every N in this sweep gets AT LEAST as much per-node
### damping as the old scheme's hardest case -- coarser N get
### substantially more (intentional: a uniformly more-damped campaign,
### not just "no worse than before").
###
### Array covers eberly (batched pipeline) only by default -- IMC rows
### are kept in MODEL_LIST/N_LIST below for a future standalone rerun
### (--array=6-11) but are NOT part of the default submission. All N
### satisfy (N-1) % 5 == 0 (intersections on mesh nodes).
###
###   IDX :  0      1      2      3      4       5       6      7      8      9      10      11
###   task:  eb21   eb31   eb51   eb71   eb101   eb131   imc21  imc31  imc51  imc71  imc101  imc131
###                                                       (not run by default -- see above)
###
### KNOWN ISSUE this campaign is designed to measure: with the adaptive
### k_c seeding, the Eberly contact TUNNELS at N=51 (near-zero pull-out
### resistance, penetration >> d_imc).  Expect the same at N >= 51 here.
### To rerun a fine mesh with pinned contact stiffness instead:
###   sbatch --array=2 --export=ALL,FIXED_KC=<value> run_net_pullout_mesh_campaign.sh
### (grep 'k_c' in a completed N21 log to find a converged value).
###
### Submit FROM THIS FOLDER (benchmarks/net_pullout/):
###   sbatch run_net_pullout_mesh_campaign.sh
#SBATCH --time=4-00:00:00
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --array=0-5

### Load Python module
module load python/3.14.0/gcc-15.1.0

### Pin every threading library to the single allocated core.  OpenBLAS /
### MKL otherwise size their thread pools to the NODE's core count and the
### surplus threads thrash inside this job's 1-core cgroup (measured
### O(100 us) per tiny LAPACK call with a warm oversubscribed pool).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

### Activate venv
REPO=/gpfs/workdir/gerarde/Contact_Model/Reg_Contact/dismech-py_regularized-contact
source "$REPO/.venv/bin/activate"

IDX=$SLURM_ARRAY_TASK_ID
MODEL_LIST=(eberly eberly eberly eberly eberly eberly imc imc imc imc imc imc)
N_LIST=(21 31 51 71 101 131 21 31 51 71 101 131)
MODEL=${MODEL_LIST[$IDX]}
N=${N:-${N_LIST[$IDX]}}

### Soft material (identical to the validated alternated recipe)
YOUNGS=${YOUNGS:-1e7}
NU=${NU:-0.45}
DENSITY=${DENSITY:-1100}
WEAVE_AMPLITUDE=${WEAVE_AMPLITUDE:-3.0e-3}
### Mass-proportional damping RATE in 1/s (NOT the old N.s/m-per-node
### coefficient -- see header note above). Mesh-independent: do not scale
### this with N.
DAMPING_ETA=${DAMPING_ETA:-1.0e9}

### Contact params; mu = 0 for BOTH (friction-matched pair)
if [ "$MODEL" = "imc" ]; then
    D_IMC=${D_IMC:-2e-4}
else
    D_IMC=${D_IMC:-5e-5}
fi
MU=${MU:-0.0}
FIXED_KC=${FIXED_KC:-}
MAX_NEWTON_ITER=${MAX_NEWTON_ITER:-200}

### Time stepping (soft recipe)
VEL_TOL=${VEL_TOL:-1e-3}
END_DISP_ENGAGE=${END_DISP_ENGAGE:-3.75e-3}
END_DISP_PULLOUT=${END_DISP_PULLOUT:-30e-3}
T_ENGAGE=${T_ENGAGE:-1.0}
T_PULLOUT=${T_PULLOUT:-12.0}
DT=${DT:-1e-2}
DT_RELEASE=${DT_RELEASE:-5e-3}
DT_PULLOUT=${DT_PULLOUT:-5e-3}
NEWTON_TOL=${NEWTON_TOL:-1e-4}
LINE_SEARCH_ITERS=${LINE_SEARCH_ITERS:-0}
SOLVER=${SOLVER:-pardiso}

### Tag: _massdamp marks these as reruns under the mass-proportional
### damping fix -- do NOT merge with any pre-fix output files sharing the
### old _alt_soft_${MODEL}_N${N}[_fast] naming.
TAG="_alt_soft_${MODEL}_N${N}_massdamp"

echo "============================================================================"
echo "  MESH CAMPAIGN | task $IDX : model=$MODEL  N=$N"
echo "  E = $YOUNGS Pa  nu = $NU  rho = $DENSITY  A0 = $WEAVE_AMPLITUDE m"
echo "  damping = $DAMPING_ETA   d_imc = $D_IMC   mu = $MU   fixed_kc = '${FIXED_KC:-adaptive}'   max_newton_iter = $MAX_NEWTON_ITER"
echo "  dt = $DT (A,B) | dt_release = $DT_RELEASE (C,D) | dt_pullout = $DT_PULLOUT (E)"
echo "  tag = $TAG   solver = $SOLVER"
echo "  host = $(hostname)   start = $(date -Iseconds)"
echo "============================================================================"

### -------- PRE-FLIGHT (eberly tasks): full Eberly test suite --------------
if [ "$MODEL" = "eberly" ]; then
    echo ">>> pre-flight: Eberly regression + batched-equivalence test suite"
    python3 -m pytest "$REPO/tests/test_eberly_contact.py" -q
    PRE=$?
    if [ $PRE -ne 0 ]; then
        echo "!!! PRE-FLIGHT TESTS FAILED (exit $PRE). Not starting the simulation."
        exit $PRE
    fi
    echo ">>> pre-flight OK"
fi

PY_ARGS=(
    contact_net_pullout.py "$N"
    --model "$MODEL"
    --youngs "$YOUNGS" --nu "$NU" --density "$DENSITY"
    --weave-amplitude "$WEAVE_AMPLITUDE"
    --damping-eta "$DAMPING_ETA"
    --alternating-weave
    --mu "$MU"
    --vel-tol "$VEL_TOL"
    --d-imc "$D_IMC"
    --newton-tol "$NEWTON_TOL"
    --max-newton-iter "$MAX_NEWTON_ITER"
    --end-disp-engage "$END_DISP_ENGAGE"
    --end-disp-pullout "$END_DISP_PULLOUT"
    --t-engage "$T_ENGAGE"
    --t-pullout "$T_PULLOUT"
    --dt "$DT"
    --dt-release "$DT_RELEASE"
    --dt-pullout "$DT_PULLOUT"
    --solver "$SOLVER"
    --use-line-search
    --line-search-iters "$LINE_SEARCH_ITERS"
    --no-plot
    --tag "$TAG"
)
if [ "$MODEL" = "eberly" ]; then
    PY_ARGS+=(--eberly-h-scale)
fi
if [ -n "$FIXED_KC" ]; then
    PY_ARGS+=(--fixed-kc "$FIXED_KC")
fi

python3 -u "${PY_ARGS[@]}"
EXIT=$?

echo "============================================================================"
echo "  Mesh task $IDX ($MODEL/N=$N) finished with exit $EXIT"
echo "  end = $(date -Iseconds)"
echo "============================================================================"
exit $EXIT
