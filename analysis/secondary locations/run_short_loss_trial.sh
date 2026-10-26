#!/bin/bash
#SBATCH --job-name=secondary_short_loss
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=11947M
#SBATCH --time=12:00:00
#SBATCH --constraint=AMD,EPYC_9654
#SBATCH --output=logs/secondary_short_loss_%j.log
#SBATCH --error=logs/secondary_short_loss_%j.log
set -eo pipefail
repository_directory="$1"
shift
source ~/.bashrc
set -u
module load stack/2024-06
module load gcc/12.2.0
module load python/3.10.13
module load eth_proxy
export MAVEN_USE_ENV_PROXY=true
export CC=gcc CXX=g++
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12
unset DISPLAY
cd "$repository_directory"
uv run python "$repository_directory/analysis/secondary locations/short_loss_trial.py" "$@"
