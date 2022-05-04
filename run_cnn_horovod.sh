#!/bin/bash
#SBATCH --out=%j.out
#SBATCH --err=%j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=2 ##### < change
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2 #### < change
#SBATCH --partition=m100_usr_prod
#SBATCH -A  # fill your account here <<<<
#SBATCH --time=00:30:00

module purge
module load profile/deeplrn
module load autoload cineca-ai/1.0.0
#conda init bash
source  /cineca/prod/opt/tools/anaconda/2020.11/none/etc/profile.d/conda.sh
conda activate $CINECA_AI_ENV

cd # directory where you cloned the repo <<<<<<

echo 'all modules loaded'

horovodrun -np 2 python python_code/horovod_dist_gen.py
#### change the -np _
