# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 10:17:26 2025

@author: dabdelkader
"""
from ..matching.counts import Counts
from ..matching.matcher import TrafficDataMatcher
from ..matching.plots import Plotter
from ..matching.utils.merge import Merge
import os
import geopandas as gpd

def configure(context):
    context.stage("analysis.counts.cantons.vaud")
    context.stage("analysis.counts.matching.network")
    context.stage("analysis.counts.matching.compare")
    context.stage("data.spatial.swiss_border")

    context.config("input_downsampling")
    context.config("output_path")
    context.config("output_id")
    context.config("simulation_directory", default = "simulation_output")
    context.config("only_weekday", default=False)

def execute(context):        
    # paths and parameters  
    vaud_counts_data  = context.stage("analysis.counts.cantons.vaud")
    city = "vaud"
    sample_size = context.config("input_downsampling")
    path_to_images = os.path.join(context.config("output_path"), 
                                  context.config("output_id"), 
                                  context.config("simulation_directory"),
                                  "compare_counts_weekdays" if context.config("only_weekday") else "compare_counts_all_days")
    os.makedirs(path_to_images, exist_ok=True)

    # Load the network and the counts
    network = context.stage("analysis.counts.matching.network")

    # load counts 
    counts = Counts(file_path=vaud_counts_data, id_column="id",
                columns_to_keep={'TJM':"flow", 'TJOM':"flow_w"},
                context = context)
            
    # Match the data with the network
    matcher = TrafficDataMatcher(city, cache = context.path())
    matched = matcher.match(network = network, 
                            counts = counts, 
                            search_radius=10, 
                            get_pairs= True,                        
                            by_highway_order=False, 
                            direction_from_osm=False,
                            only_two_link_ids=True)

    # Compare the with simulation
    cmp     = context.stage("analysis.counts.matching.compare")
    flows   = cmp.compare_flow_total_efficient(counts, matched, network, 
                                            sample_size = sample_size, 
                                            get_average=False, 
                                            flow_col = 'flow_w' if context.config("only_weekday") else 'flow')
    
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

    path_to_results = Merge(city = city, 
                            matched = matched, 
                            flows = flows, 
                            cache=context.path()
                            ).run(return_path = True)

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
    
    # Plot the map in html
    roads_to_show = ['motorway', 'trunk', 'primary', 'motorway_link', 'trunk_link', 'primary_link']
    border = gpd.GeoDataFrame(context.stage("data.spatial.swiss_border").to_crs(epsg=4326))        
    Plotter.create_map([network.get_ways(road_types = roads_to_show).to_crs(epsg=4326),
                        matched_links.to_crs(epsg=4326)], 
                        data_to_show=["link_id"], 
                        point_gdf=[counts.counts[['id','geometry']].merge(
                                   flows[["id","pdiff"]], on="id", how="left").to_crs(epsg=4326)],
                        point_data_to_show=['id',"pdiff"],
                        border = border,
                        path_to_save= os.path.join(path_to_images, f"counts_on_network_{city}.html"))
    
    return path_to_results
   











