#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=8192
#SBATCH --output=/cluster/scratch/rsahleanu/synpp_%j.log
#SBATCH --error=/cluster/scratch/rsahleanu/synpp_%j.log

python3 -m synpp config.yml