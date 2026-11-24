#!/bin/bash
#SBATCH --job-name=mode_shares_spatial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mode_shares_spatial_%j.log
#SBATCH --error=logs/mode_shares_spatial_%j.log

source ~/.bashrc
set -eo pipefail
module load stack/2024-06 gcc/12.2.0 python/3.10.13 openjdk/25.0.2_10 maven eth_proxy
export MAVEN_USE_ENV_PROXY=true
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
unset DISPLAY
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
uv run --no-sync -m synpp "${1:-config_dib_10.yml}" \
  --run analysis.mode_shares.target --run analysis.mode_shares.simulated \
  --run analysis.mode_shares.trip_distributions --run analysis.mode_shares.run --debug true
