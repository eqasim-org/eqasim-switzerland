module load stack/2024-06
module load gcc/12.2.0
module load python/3.10.13
module load openjdk/21.0.3_9
module load maven
module load eth_proxy

cache_path="/cluster/scratch/asallard/eqasim/test_pt_routing/cache/matsim.simulation.run__f5503319108f4633e6c9386952ed3a3f.cache/simulation_output"
destination_path="/cluster/project/cmdp/asallard/Results/GTFS_routing/"

scp -r $cache_path/ITERS/it.60 $destination_path

scp $cache_path/logfile.log $destination_path
scp $cache_path/ph_modestats.csv $destination_path
scp $cache_path/pkm_modestats.csv $destination_path
scp $cache_path/stopwatch.csv $destination_path
scp $cache_path/traveldistancestats.csv $destination_path
scp $cache_path/modestats.csv $destination_path
scp $cache_path/output_config.xml $destination_path
scp $cache_path/output_trips.csv.gz $destination_path
scp $cache_path/output_transitSchedule.xml.gz $destination_path
scp $cache_path/output_transitVehicles.xml.gz $destination_path



