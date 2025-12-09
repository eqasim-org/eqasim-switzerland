#!/bin/bash

module load stack/2024-06
module load gcc/12.2.0
module load python/3.10.13
module load eth_proxy
module load openjdk/21.0.3_9

source /cluster/project/cmdp/asallard/eqasim_venv/bin/activate

sbatch -n 1 --cpus-per-task=8 --time=06:30:00 --mem-per-cpu=12000 --wrap="python3 -m synpp config_aurore.yml"

source /cluster/home/anding/myenv/bin/activate

sbatch -n 1 --cpus-per-task=8 --time=8:00:00 --mem-per-cpu=16192 --wrap="python3 -m synpp config_andrew.yml"
