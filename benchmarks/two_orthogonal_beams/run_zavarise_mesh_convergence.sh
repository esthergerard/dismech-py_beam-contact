#!/bin/bash
### Job name
#SBATCH --job-name=dismech_mesh_conv
#SBATCH --partition=cpu_long
#SBATCH --output=%x.o%j
#SBATCH --error=%x.o%j
### Mesh convergence runs 4 meshes (n_x = 11, 21, 51, 101) sequentially.
### Cost grows ~ (n_x)^2 due to the Eberly contact pair loop (broad-phase
### culling helps a lot but worst-case is quadratic).  Allocate generously:
### 11 nodes: ~30 min; 21: ~2 h; 51: ~12 h; 101: ~40 h.  → 3 days total.
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

### Number of load steps (default 6 = paper's Case 1).
### Override at submission with `sbatch --export=N_STEPS=29 ...`.
N_STEPS=${N_STEPS:-6}

### Mesh sequence (space-separated list).
### Override at submission with e.g. `sbatch --export=NX_LIST="11 21 51" ...`
NX_LIST=${NX_LIST:-"11 21 51 101"}

### Plotting on the cluster has triggered a matplotlib Path.__deepcopy__
### infinite-recursion bug.  We pass --no-plot here so the orchestrator
### only writes the per-mesh txt files + the plain-text overlay table.
### After all runs finish, regenerate the PNG locally with:
###   python3 run_zavarise_mesh_convergence.py --plot-only --n-steps $N_STEPS --nx-list $NX_LIST

echo "Running Zavarise mesh-convergence study"
echo "  N_STEPS = $N_STEPS"
echo "  NX_LIST = $NX_LIST"
date

python3 -u run_zavarise_mesh_convergence.py \
    --n-steps $N_STEPS \
    --nx-list $NX_LIST \
    --no-plot

echo "Done"
date
