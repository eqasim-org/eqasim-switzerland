#!/bin/bash
# Author: dabdelkader
# Date: 03.04.2025

#SBATCH --job-name=Eqasim # name f the job
#SBATCH -n 1                 # Number of tasks
#SBATCH --cpus-per-task=24    # CPUs per task
#SBATCH --time=12:00:00       # Maximum runtime (6 hours)
#SBATCH --mem-per-cpu=8G     # Memory per CPU (8GB)
#SBATCH -o /cluster/project/cmdp/dabdelkader/ch-zh-synpop/logs/synpp_autochoice_output_%j.txt  # Output file
#SBATCH -e /cluster/project/cmdp/dabdelkader/ch-zh-synpop/logs/synpp_autochoice_pyosmium_%j.log         # Error log file


# Source the interactive shell config (for osmosis)
source ~/.bashrc

# -----------------------------------------
# Step 1: Load Required Modules
# -----------------------------------------
module load stack/2024-06
module load gcc/12.2.0
module load python/3.10.13
module load openjdk/21.0.3_9
module load maven
module load eth_proxy

# -----------------------------------------
# Step 2: Activate Python Virtual Environment
# -----------------------------------------
source /cluster/home/dabdelkader/env/eqasim/bin/activate

# -----------------------------------------
# Step 3: Run the Job
# -----------------------------------------
python3 -m synpp config_dib.yml

# -----------------------------------------
# Step 4: Print Confirmation
# -----------------------------------------
echo "Job completed."
