#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=64
#SBATCH --ntasks=1
#SBATCH --array=0-4
###SBATCH --gres=gpu:A100:1
#SBATCH --mem=64GB
#SBATCH --partition standard
#SBATCH --mail-type=end
#SBATCH --mail-user=shiyangalris.dai@uzh.ch

###SBATCH --job-name=hello1 ## job name
#SBATCH --output=slurm_files/slurm-%A_%a.out  ## standard out file

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

python3 -u PBCO_distortion_incl_run.py $SLURM_ARRAY_TASK_ID
conda deactivate