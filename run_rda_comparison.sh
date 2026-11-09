#!/bin/bash
#SBATCH --job-name=rda_hours
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

source ~/.bashrc
set -eo pipefail
module load stack/2024-06
module load gcc/12.2.0
module load rust/1.85.0
module load python/3.10.13
module load openjdk/25.0.2_10
module load maven
module load eth_proxy
export MAVEN_USE_ENV_PROXY=true
export CC=gcc
export CXX=g++
# RDA uses 24 worker processes; avoid multiplying numerical-library threads.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
unset DISPLAY
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
uv run -m analysis.rda.run_cached --config "${1:-config_dib_10.yml}" --threads "${SLURM_CPUS_PER_TASK:-24}"
