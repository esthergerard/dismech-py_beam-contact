#!/bin/bash
### Job name
#SBATCH --job-name=dismech_friction_mesh_conv
#SBATCH --partition=cpu_long
#SBATCH --output=%x.o%j
#SBATCH --error=%x.o%j
### Friction mesh convergence: mu in {0.1, 0.6}, nx in {21, 51, 101}.
### Same sizing logic as run_zavarise_mesh_convergence.sh — cost grows
### roughly with nx (broad-phase helps a lot but the Newton solve per
### dt-step still touches every contact pair).  Allocate generously.
#SBATCH --time=3-00:00:00
### Memory
#SBATCH --mem=20gb
### Resources
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1

### Load Python module (same as the other dismech jobs on Ruche)
module load python/3.14.0/gcc-15.1.0

### Activate venv
source /gpfs/workdir/gerarde/Contact_Model/Reg_Contact/dismech-py_regularized-contact/.venv/bin/activate

### Number of load steps (default 6 = paper's Case 1).
### Override at submission with `sbatch --export=N_STEPS=29 ...`.
N_STEPS=${N_STEPS:-6}

### Friction coefficients and Beam-X mesh sequence.  Override at
### submission with e.g.
###   sbatch --export=MU_LIST="0.1",NX_LIST="21 51" run_friction_mesh_cluster.sh
MU_LIST=${MU_LIST:-"0.8"}
NX_LIST=${NX_LIST:-"21 51 101"}

### Cluster matplotlib has historically hit a Path.__deepcopy__ infinite
### recursion (see the comment in run_zavarise_mesh_convergence.sh).  We
### skip plotting here and regenerate the PNGs locally after the run with:
###   python3 plot_friction_mesh_convergence.py
### or via the orchestrator's --plot-only mode:
###   python3 run_zavarise_friction_mesh_convergence.py --plot-only

echo "Running Zavarise friction mesh-convergence study"
echo "  N_STEPS = $N_STEPS"
echo "  MU_LIST = $MU_LIST"
echo "  NX_LIST = $NX_LIST"
date

python3 -u run_zavarise_friction_mesh_convergence.py \
    --n-steps $N_STEPS \
    --mu-list $MU_LIST \
    --nx-list $NX_LIST \
    --no-plot

echo "Done"
date
