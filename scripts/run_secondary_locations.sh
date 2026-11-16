#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=5G
#SBATCH --time=24:00:00
#SBATCH --constraint=AMD,EPYC_9654
set -eo pipefail
source ~/.bashrc
set -u
module load stack/2024-06
module load gcc/12.2.0
module load rust/1.85.0
module load python/3.10.13
module load openjdk/25.0.2_10
module load maven
module load eth_proxy
export MAVEN_USE_ENV_PROXY=true
export CC=gcc CXX=g++
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
unset DISPLAY
repository_directory="$1"
config_path="$2"
cd "$repository_directory"
uv run -m synpp "$config_path"
