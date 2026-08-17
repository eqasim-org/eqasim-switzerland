from ..matching.counts import Counts
from ..matching.matcher import TrafficDataMatcher
from ..matching.plots import Plotter
from ..matching.results import save_count_results
import os
import geopandas as gpd

def configure(context):
    context.stage("analysis.counts.cantons.geneva")
    context.stage("analysis.counts.matching.network")
    context.stage("analysis.counts.matching.compare")
    context.stage("data.spatial.swiss_border")

    context.config("input_downsampling")
    context.config("output_path")
    context.config("output_id")
    context.config("simulation_directory", default = "simulation_output")
    context.config("only_weekday", default=False)

def execute(context):        
    if not context.config("only_weekday"):
        return None # these data only have weekday flow
     
    # paths and parameters  
    geneva_counts_data  = context.stage("analysis.counts.cantons.geneva")
    city = "geneva"
    sample_size = context.config("input_downsampling")
    path_to_images = os.path.join(context.config("output_path"), 
                                  context.config("output_id"), 
                                  context.config("simulation_directory"),
                                  "compare_counts_weekdays" if context.config("only_weekday") else "compare_counts_all_days")
    os.makedirs(path_to_images, exist_ok=True)

    # Load the network and the counts
    network = context.stage("analysis.counts.matching.network")

    # load counts
    counts = Counts(file_path=geneva_counts_data,  id_column="OBJECTID",
                columns_to_keep={'mean_flow_2025':"flow", "median_flow_2025":"median_flow",
                                 "osm_id":"osm_id", "angle":"angle"},
                context = context)
        
    # Match the manually identified OSM way and use its angle for direction.
    matcher = TrafficDataMatcher()
    matched = matcher.match(network=network, counts=counts)

    # Compare the with simulation
    cmp     = context.stage("analysis.counts.matching.compare")    
    flows   = cmp.compare_flow_total_efficient(counts, matched, network, 
                                            sample_size = sample_size, 
                                            get_average=False, 
                                            flow_col = 'flow')
    
    # Identify the stations that might be mismatched
    stations_to_drop = flows[(abs(flows.flow-flows.simulated_flow)>15000)|
                            (flows.simulated_flow<1000)|
                            (flows.flow<1000)|
                            (~flows.pdiff.between(-70,200))]["id"].unique()

    # Plot the network and highligh these links in green  
    plotter = Plotter()
    matched_links = plotter.plot_network_with_counts( 
                                    counts, matched, network,
                                    output=os.path.join(path_to_images, f"{city}_network.png"),
                                    lw = 0.7,
                                    markersize = 4, 
                                    figsize = (40,40),                                                    
                                    road_types = "all",
                                    cut = True,
                                    highlight_stations = stations_to_drop,
                                    return_matched_links = True)

    # filter out stations to drop and save results
    flows = flows[~flows['id'].isin(stations_to_drop)].reset_index(drop=True)
    matched = matched[~matched['id'].isin(stations_to_drop)].reset_index(drop=True)

    path_to_results = save_count_results(city, matched, flows, context.path())

    # Plot statistics
    plotter.plot_flow(flows = flows, 
                    counts = counts, 
                    distance_to_border = 2000, 
                    title = f"Observed vs Simulated Traffic Flows ({city})",
                    output_file = os.path.join(path_to_images, f"flow_comparaison_{city}.png"),
                    show_range = False)

    plotter.plot_flow_by_road_type(flows, network, matched, counts,
                                distance_to_border = 0, 
                                title = f"Average Observed vs Simulated Flow by Highway Type ({city})",
                                output_file = os.path.join(path_to_images, f"flow_by_road_type_{city}.png"))
    
    
    border = gpd.GeoDataFrame(context.stage("data.spatial.swiss_border").to_crs(epsg=4326))  
    roads_to_show = ['motorway', 'trunk', 'primary', 'motorway_link', 'trunk_link', 'primary_link', 'secondary','secondary_link','tertiary']      
    Plotter.create_map([network.get_ways(road_types = roads_to_show).to_crs(epsg=4326),
                        matched_links.to_crs(epsg=4326)], 
                        data_to_show=["link_id"], 
                        point_gdf=[counts.counts[['id','geometry']].merge(
                                   flows[["id","pdiff", "adiff"]], on="id", how="left").to_crs(epsg=4326)],
                        point_data_to_show=['id',"pdiff", "adiff"],
                        border = border,
                        cut_network = True,
                        path_to_save= os.path.join(path_to_images, f"counts_on_network_{city}.html"))
    
    return path_to_results
   







