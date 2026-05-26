#!/bin/bash
#SBATCH --job-name viraexplorer
#SBATCH --output output_%j.txt
#SBATCH --error errors_%j.txt
#SBATCH --mail-user simionricc@dei.unipd.it
#SBATCH --mail-type END
#SBATCH --time 04:00:00
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --partition allgroups
#SBATCH --mem 16G
#SBATCH --gres gpu:1          # request 1 GPU — check exact syntax with your sysadmin

cd $SLURM_SUBMIT_DIR

module load python/3.x        # adjust to what's available on your cluster
source myenv/bin/activate     # if you use a virtual environment

git clone https://github.com/NeuroCSUT/ViraMiner.git   # or skip if data already present

srun python viraexplorer.py