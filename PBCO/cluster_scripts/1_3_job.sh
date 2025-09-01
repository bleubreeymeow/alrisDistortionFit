#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
###SBATCH --array=0-4
###SBATCH --gres=gpu:A100:1
#SBATCH --mem=64GB
#SBATCH --partition standard

###SBATCH --job-name=hello1 ## job name
#SBATCH --output=slurm_files/slurm-%A.out  ## standard out file

##vals=(0 1 2 3 4) ## seed
##valsArr=()

##for fs in ${vals[@]}; do
##    valsArr+=($fs)
##done

##value=${valsArr[$SLURM_ARRAY_TASK_ID]}

echo "Starting run"

##module load a100
module load anaconda3

##nvidia-smi

source activate tensorflow_env

python3 -u 1_3_PBCO_distortionfitpy.py

conda deactivate