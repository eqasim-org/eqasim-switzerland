echo $VIRTUAL_ENV
sbatch -n 1 --cpus-per-task=12 --time=4:00:00 --mem-per-cpu=12288 --wrap="/cluster/home/chaoch/csfm/myenv/bin/python3 -m synpp config_michelle.yml"