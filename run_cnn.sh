#!/bin/bash
#SBATCH --out=%j.out
#SBATCH --err=%j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=m100_usr_prod
#SBATCH -A ## fill your account here <<<<<
#SBATCH --time=08:00:00

module purge
module load profile/deeplrn
module load autoload cineca-ai/1.0.0
source  /cineca/prod/opt/tools/anaconda/2020.11/none/etc/profile.d/conda.sh
conda activate $CINECA_AI_ENV

cd # directory where you cloned the repo <<<<<<

echo 'all modules loaded'

python python_code/cnn.py
