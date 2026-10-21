#!/bin/bash
#SBATCH --job-name=secondary_analysis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/secondary_analysis_%j.log
#SBATCH --error=logs/secondary_analysis_%j.log
set -eo pipefail
repository_directory="$1"
shift
source ~/.bashrc
set -u
module load stack/2024-06
module load gcc/12.2.0
module load python/3.10.13
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
unset DISPLAY
cd "$repository_directory"
uv run python "$repository_directory/analysis/secondary locations/run.py" "$@"
