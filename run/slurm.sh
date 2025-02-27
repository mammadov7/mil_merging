#!/bin/sh
#SBATCH -A rnz@v100
#SBATCH --nodes=1
#SBATCH --job-name=soup3
#SBATCH --constraint=v100-32g
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --time=06:00:00
#SBATCH --hint=nomultithread 
#SBATCH --array=0-3

module load pytorch-gpu/py3/1.11.0
which python

cd ~/run
bash run$((${SLURM_ARRAY_TASK_ID}*5+1)).sh &
bash run$((${SLURM_ARRAY_TASK_ID}*5+2)).sh &
bash run$((${SLURM_ARRAY_TASK_ID}*5+3)).sh &
bash run$((${SLURM_ARRAY_TASK_ID}*5+4)).sh &
bash run$((${SLURM_ARRAY_TASK_ID}*5+5)).sh &
wait
