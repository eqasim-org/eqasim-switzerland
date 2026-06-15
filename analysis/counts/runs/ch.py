# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 10:17:26 2025

@author: dabdelkader
"""
from ..matching.counts import Counts
from ..matching.matcher import TrafficDataMatcher
from ..matching.plots import Plotter, GEH
from ..matching.utils.merge import Merge
import os
import geopandas as gpd
import logging

logger = logging.getLogger("synpp")

def configure(context):
    context.stage("analysis.counts.cantons.ch")
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
    count_stations_file, counts_data_file, year  = context.stage("analysis.counts.cantons.ch")
    city = "ch"
    sample_size = context.config("input_downsampling")
    minimum_months = 8
    include_incomplete_data = True
    path_to_images = os.path.join(context.config("output_path"), 
                                  context.config("output_id"), 
                                  context.config("simulation_directory"),
                                  "compare_counts_weekdays" if context.config("only_weekday") else "compare_counts_all_days")
    os.makedirs(path_to_images, exist_ok=True)

    # Load the network and the counts
    network = context.stage("analysis.counts.matching.network")

    # load counts
    counts = Counts(counts_data_file, count_stations_file, 
                include_incomplete_data = include_incomplete_data, 
                minimum_months = minimum_months, context = context, year=year)
            
    # Match the data with the network
    matcher = TrafficDataMatcher(city, cache = context.path())
    matched = matcher.match(network=network, 
                            counts=counts, 
                            search_radius=80,
                            get_pairs= True,                        
                            by_highway_order=True, 
                            direction_from_osm=False,
                            only_two_link_ids=True)

    # Compare the with simulation
    cmp     = context.stage("analysis.counts.matching.compare")
    flows   = cmp.compare_flow_total_efficient(counts, matched, network, 
                                            sample_size = sample_size, 
                                            get_average=False, 
                                            flow_col = 'flow_w' if context.config("only_weekday") else 'flow')
    
    # Identify the stations that might be mismatched
    stations_to_drop = flows[(abs(flows.flow-flows.simulated_flow)>25000)|
                             (flows.simulated_flow< 200 * 24)|
                             (flows.flow< 200 * 24)|
                             (~flows.pdiff.between(-70,200))]["id"].unique()

    # geh = GEH(flows.flow.values, flows.simulated_flow.values, return_vector=True)
    # stations_to_drop = flows[(geh>50)]["id"].values
    # logger.info(f"Identified {len(stations_to_drop)} stations to be removed, out of {len(flows)} total stations. These stations will be dropped from the analysis.")

    # Plot the network and highligh these links in green  
    plotter = Plotter()
    roads_to_show = ['motorway', 'trunk', 'primary', 'motorway_link', 'trunk_link', 'primary_link']
    matched_links = plotter.plot_network_with_counts( 
                                    counts, matched, network,
                                    output=os.path.join(path_to_images, f"{city}_network.png"),
                                    lw = 0.7,
                                    markersize = 8, 
                                    figsize = (80,80),                                                    
                                    road_types = roads_to_show,
                                    cut = True,
                                    highlight_stations = stations_to_drop,
                                    return_matched_links = True)

    # filter out stations to drop and save results
    all_flows = flows.copy()
    all_flows["dropped"] = all_flows["id"].isin(stations_to_drop)
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
                    distance_to_border = 3000, 
                    title = f"Observed vs Simulated Traffic Flows ({city})",
                    output_file = os.path.join(path_to_images, f"flow_comparaison_{city}.png"),
                    show_range = False,
                    show_geh = True)

    plotter.plot_flow(flows = flows, 
                    counts = counts, 
                    distance_to_border = 3000, 
                    title = f"Observed vs Simulated Traffic Flows ({city})",
                    output_file = os.path.join(path_to_images, f"flow_comparaison_without_border_{city}.png"),
                    show_range = False,
                    show_geh = False,
                    remove_near_border = True)
    
    plotter.plot_flow_by_road_type(flows, network, matched, counts,
                                distance_to_border = 0, 
                                title = f"Average Observed vs Simulated Flow by Highway Type ({city})",
                                output_file = os.path.join(path_to_images, f"flow_by_road_type_{city}.png"))
    
    
    # Plot the map in html  
    border = gpd.GeoDataFrame(context.stage("data.spatial.swiss_border").to_crs(epsg=4326))        
    Plotter.create_map([network.get_ways(road_types = roads_to_show).to_crs(epsg=4326),
                        matched_links.to_crs(epsg=4326)], 
                        data_to_show=["link_id"], 
                        point_gdf=[counts.counts[['id','geometry']].merge(
                                   flows[["id","pdiff"]], on="id", how="left").to_crs(epsg=4326)],
                        point_data_to_show=['id',"pdiff"],
                        border = border,
                        path_to_save= os.path.join(path_to_images, f"counts_on_network_{city}.html"))
    
    Plotter.create_map([network.get_ways(road_types = roads_to_show).to_crs(epsg=4326),
                    matched_links.to_crs(epsg=4326)], 
                    data_to_show=["link_id"], 
                    point_gdf=[counts.counts[['id','geometry']].merge(
                               all_flows[["id","pdiff","adiff"]], on="id", how="left").to_crs(epsg=4326)],
                    point_data_to_show=['id',"pdiff","adiff"],
                    border = border,
                    path_to_save= os.path.join(path_to_images, f"counts_on_network_{city}_unfiltered.html"))
    
    return path_to_results
