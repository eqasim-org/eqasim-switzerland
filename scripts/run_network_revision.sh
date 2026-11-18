#!/bin/bash
#SBATCH --job-name=network-pt2matsim
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=5G
#SBATCH --time=04:00:00
#SBATCH --constraint=AMD,EPYC_9654
#SBATCH --output=logs/network_revision_%j.log
#SBATCH --error=logs/network_revision_%j.log

set -eo pipefail
source ~/.bashrc
module load stack/2024-06 gcc/12.2.0 rust/1.85.0 python/3.10.13 openjdk/25.0.2_10 maven eth_proxy
export MAVEN_USE_ENV_PROXY=true
export CC=gcc CXX=g++
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
unset DISPLAY

config_path="${1:-config_dib_10.yml}"
report_dir="${2:?Provide the comparison report directory}"

uv run -m synpp "$config_path" \
  --run matsim.runtime.pt2matsim \
  --run matsim.scenario.network.convert_osm_pt2matsim \
  --run matsim.scenario.network.convert_osm

uv run -m scripts.compare_network_revision --config "$config_path" \
  --baseline "$report_dir/baseline.json" --output "$report_dir/comparison.json"
