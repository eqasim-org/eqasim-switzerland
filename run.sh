#!/bin/bash

module load stack/2024-06
module load gcc/12.2.0
module load python/3.10.13
module load eth_proxy
module load openjdk/21.0.3_9

export PATH=/cluster/home/anding/apache-maven-3.9.9/bin:$PATH
source ~/.bashrc

source /cluster/home/anding/myenv/bin/activate

sbatch -n 1 --cpus-per-task=8 --time=8:00:00 --mem-per-cpu=16192 --wrap="python3 -m synpp config_andrew.yml"
