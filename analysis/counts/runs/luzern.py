from ..matching.counts import Counts
from ..matching.matcher import TrafficDataMatcher
from ..matching.plots import Plotter
from ..matching.results import save_count_results
import os
from ..paths import configure_simulation_path, get_analysis_output_path, matches_found

def configure(context):
    context.stage("analysis.counts.cantons.luzern")
    context.stage("analysis.counts.matching.network")
    context.stage("analysis.counts.matching.compare")
    context.stage("data.spatial.swiss_border")
    
    context.config("input_downsampling")
    configure_simulation_path(context)
    context.config("only_weekday", default=False)

def execute(context):   
    if context.config("only_weekday"):
        return None # these data do not have weekday only counts   
    # paths and parameters  
    luzern_counts_data  = context.stage("analysis.counts.cantons.luzern")
    city = "luzern"
    sample_size = context.config("input_downsampling")
    path_to_images = get_analysis_output_path(context)
    os.makedirs(path_to_images, exist_ok=True)

    # Load the network and the counts
    network = context.stage("analysis.counts.matching.network")

    # load counts
    counts = Counts(file_path=luzern_counts_data, id_column="objectid",
                columns_to_keep={'flow':'flow'}, context = context)
            
    # Match the data with the network
    matcher = TrafficDataMatcher()
    matched = matcher.match(network = network, 
                            counts = counts, 
                            mode="bidirectional",
                            search_radius=10)

    if not matches_found(matched, city):
        return None

    # Compare the with simulation
    cmp     = context.stage("analysis.counts.matching.compare")    
    flows   = cmp.compare_flow_total_efficient(counts, matched, network, 
                                            sample_size = sample_size, 
                                            get_average=False, 
                                            flow_col = "flow")
    
    if not matches_found(flows, city, source="simulation flow results"):
        return None

    # Identify the stations that might be mismatched
    stations_to_drop = flows[(abs(flows.flow-flows.simulated_flow)>10000)|
                         (flows.simulated_flow<1000)|
                         (flows.flow<1000)|
                         (~flows.pdiff.between(-70,200))]["id"].unique()

    # Plot the network and highligh these links in green  
    plotter = Plotter()
    plotter.plot_network_with_counts( 
                                    counts, matched, network,
                                    output=os.path.join(path_to_images, f"{city}_network.png"),
                                    lw = 0.7,
                                    markersize = 4, 
                                    figsize = (40,40),                                                    
                                    road_types = "all",
                                    cut = True,
                                    highlight_stations = stations_to_drop)

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
    
    return path_to_results
   








