#!/bin/bash
### Job name
#SBATCH --job-name=dismech_soft_soft
#SBATCH --partition=cpu_long
#SBATCH --output=%x.o%j
#SBATCH --error=%x.o%j
### 3 days wall time (Case 2 with 29 load steps should fit comfortably)
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

### Number of load steps:
###  - 6  → Case 1 of Zavarise & Wriggers (2000)
###  - 29 → Case 2 (large-sliding study)
### Override at submission time:
###   sbatch --export=N_STEPS=29 run_contact_bending_soft_soft.sh
N_STEPS=${N_STEPS:-6}

echo "Running Zavarise & Wriggers (2000) benchmark with N_STEPS=$N_STEPS load steps"
date

python3 contact_bending_soft_soft.py $N_STEPS

echo "Done"
date
