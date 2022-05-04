#!/bin/bash
#SBATCH --out=%j.out
#SBATCH --err=%j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=m100_usr_prod
#SBATCH -A cin_staff
#SBATCH --time=08:00:00

export TMPDIR="/m100_scratch/userinternal/bagarwal/tmp" #### < change the folder here once
export TEMPDIR="/m100_scratch/userinternal/bagarwal/tmp" #### < change the folder here once 
export TMP="/m100_scratch/userinternal/bagarwal/tmp" #### < change the folder here once
export TEMP="/m100_scratch/userinternal/bagarwal/tmp" #### < change the folder here once
echo $TMPDIR

module purge
module load profile/deeplrn
module load autoload cineca-ai/1.0.0
source  /cineca/prod/opt/tools/anaconda/2020.11/none/etc/profile.d/conda.sh
conda activate $CINECA_AI_ENV

cd /m100_work/cin_staff/bagarwal/chips_git

echo 'all modules loaded'

python python_code/cnn.py
