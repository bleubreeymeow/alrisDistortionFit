#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --array=0-10
###SBATCH --gres=gpu:A100:1
#SBATCH --mem=64GB
#SBATCH --partition standard
#SBATCH --mail-type=end
#SBATCH --mail-user=shiyangalris.dai@uzh.ch

#SBATCH --job-name=multiple_runs ## job name
#SBATCH --output=slurm_files/slurm-%A_%a.out  ## standard out file

##vals=(0 1 2 3 4 5 6 7 8 9 10) ## seed
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

python3 -u P2_PBCO_incl.py $SLURM_ARRAY_TASK_ID

conda deactivate