#!/bin/bash
# Author: dabdelkader

#SBATCH --job-name=Eq_small # name f the job
#SBATCH -n 1                    # Number of tasks
#SBATCH --cpus-per-task=12      # CPUs per task
#SBATCH --time=12:00:00          # Maximum runtime 
#SBATCH --mem-per-cpu=8G        # Memory per CPU (8GB)
#SBATCH -o logs/synpp_small_%j.log   # Output file
#SBATCH -e logs/synpp_small_%j.log   # Error log file


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
# Step 3: Force it to unset the display
# -----------------------------------------
unset DISPLAY

# -----------------------------------------
# Step 4: Run the Job
# -----------------------------------------
python3 -m synpp config_dib_10_local.yml

# -----------------------------------------
# Step 5: Print Confirmation
# -----------------------------------------
echo "Job completed."
