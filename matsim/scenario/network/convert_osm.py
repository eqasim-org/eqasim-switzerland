import os
import re

import matsim.runtime.pt2matsim as pt2matsim
from matsim.scenario.network.utils.network_handler import NetworkHandler
import shutil

def configure(context):
    context.stage("matsim.scenario.network.convert_osm_pt2matsim")
    context.stage("data.osm.traffic_lights")
    context.stage("data.spatial.municipality_types") # used in speed correction
    context.stage("data.spatial.municipalities") # used in speed correction
    context.stage("data.spatial.swiss_border") # used in speed correction
    context.stage("calibration.road_regions.penalty_calibration")

    context.config("data_path")
    context.config("osm_file", "switzerland-latest.osm.gz")
    context.config("border_offset", 20000)
    context.config("correct_links_capacity", True)
    context.config("minimum_speed", 1.0) #in km/h
    context.config("input_downsampling")
    context.config("add_traffic_lights", True)
    context.config("assign_elevations", True)
    context.config("simplify_network_in_eqasim", True)
    # only if simplify network is true
    context.config("remove_network_loops", True)
    context.config("remove_replicate_links", False)
    context.config("remove_nodes_with_no_intersection", False)
    context.config("correct_speed", True)
    context.config("ensure_network_connectivity", True)
    # correct speed for uphill links only
    context.config("adjust_speed_uphill", True) # if true, it is triggered only if elevation is assigned
    context.config("adjust_speed_straightness", True) # if true, it is triggered only if elevation is assigned
    context.config("adjust_speed_mountain_links", True) # if true, it is triggered only if elevation is assigned
    context.config("max_gradient_threshold", 0.1) # in percentage (10% = 0.1)
    context.config("speed_factor_uphill", 0.9) 
    # reduce capacity outside border
    context.config("capacity_factor_outside_border", 0.5)
    # whether to route the bike in the network or not
    context.config("route_bike", True)
    # Tools (in France)
    context.stage("data.tolls.osm_links")
    context.config("include_tolls", True)
    context.config("average_tolls_prices_per_km", default = 0.12)
    context.config("only_french_tolls", True)

def execute(context):
    # move the osm network to this stage
    network_path, detailed_network_file = context.stage("matsim.scenario.network.convert_osm_pt2matsim")

    # move teh detailed network to this stage if it exists
    detailed_network_path = "%s/detailed_network.csv" % context.path()
    if os.path.exists(detailed_network_file):     
        shutil.copy(detailed_network_file, detailed_network_path)


    network_pickle = "%s/network.pkl" % context.path() # faster to read if we need it in the pipeline, it should be read from this file from now on

    return NetworkHandler(context, network_path, detailed_network_path).process_network(save_as_pickle = True, network_pickle = network_pickle)
