#!/bin/bash
### Job name
#SBATCH --job-name=dismech_disc_restart
#SBATCH --partition=cpu_long
#SBATCH --output=%x.o%j_%a
#SBATCH --error=%x.o%j_%a
### 4 days wall time
#SBATCH --time=4-00:00:00
### Memory
#SBATCH --mem=20gb
### One task per array job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
### Run 3 jobs (indices 0-2), one per discretization to restart
#SBATCH --array=0-2

### Load Python module
module load python/3.14.0/gcc-15.1.0

### Activate venv
source /gpfs/workdir/gerarde/Contact_Model/Reg_Contact/dismech-py_regularized-contact/.venv/bin/activate

### Map array index to discretization value (flexible beam nodes)
DISC_VALUES=(300 500)
DISC=${DISC_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "Restarting job with flexible beam nodes: $DISC"
date

python3 contact_bending_test.py $DISC --restart

echo "Done"
date
