#!/bin/bash
#SBATCH --out=%j.out
#SBATCH --err=%j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=2 ##### < change
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2 #### < change
#SBATCH --partition=m100_usr_prod
#SBATCH -A cin_staff
#SBATCH --time=00:30:00

export TMPDIR="/m100_scratch/userinternal/bagarwal/tmp" #### < change the folder here once
export TEMPDIR="/m100_scratch/userinternal/bagarwal/tmp" #### < change the folder here once 
export TMP="/m100_scratch/userinternal/bagarwal/tmp" #### < change the folder here once
export TEMP="/m100_scratch/userinternal/bagarwal/tmp" #### < change the folder here once
echo $TMPDIR

module purge
module load profile/deeplrn
module load autoload cineca-ai/1.0.0
#conda init bash
source  /cineca/prod/opt/tools/anaconda/2020.11/none/etc/profile.d/conda.sh
conda activate $CINECA_AI_ENV

cd /m100_work/cin_staff/bagarwal/chips_git
 #### < change the folder here once

echo 'all modules loaded'

horovodrun -np 2 python python_code/horovod_dist_gen.py
#### change the -np _
