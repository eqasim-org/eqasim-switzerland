#!/bin/bash
#SBATCH --job-name=matching_shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/matching_shared_%j.log
#SBATCH --error=logs/matching_shared_%j.log

source ~/.bashrc
set -eo pipefail
module load stack/2024-06 gcc/12.2.0 python/3.10.13 openjdk/25.0.2_10 maven eth_proxy
export MAVEN_USE_ENV_PROXY=true
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
unset DISPLAY
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
uv run -m analysis.matching.run_predictive --config "${1:-config_dib_10.yml}" --debug --shared-model
