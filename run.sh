module load stack/2024-06
module load gcc/12.2.0
module load python/3.10.13
module load openjdk/21.0.3_9
module load maven
module load eth_proxy

source /cluster/project/cmdp/asallard/eqasim_venv/bin/activate

sbatch -n 1 --cpus-per-task=12 --time=00:30:00 --mem-per-cpu=8192 --wrap="python3 -m synpp config_aurore.yml"

