import logging
import os
from .network import RoadNetwork
logger = logging.getLogger("synpp")

# this is the same stage as .network. However, for dependencies issues, we build another stage here to get the network from 
# the prepare stage. The .network stage get the network from the output folder, thus not usable before running the simulation.

def configure(context):
    context.stage("data.spatial.swiss_border")
    # context.stage("matsim.simulation.prepare")
    context.stage("matsim.scenario.network.mapped")
    context.stage("matsim.scenario.network.convert_osm")

    context.config("output_prefix", "switzerland_")
    context.config("export_detailed_network", False)


def execute(context):  
    # network_file = os.path.join(context.path("matsim.simulation.prepare"), f"{context.config('output_prefix')}network.xml.gz")
    network_file = context.stage("matsim.scenario.network.mapped")["road_network"]
    network_geometry_file = None
    if context.config("export_detailed_network"):
        network_geometry_file = os.path.join(context.path("matsim.scenario.network.convert_osm"), "detailed_network.csv")
    
    logger.info("\t LOADING NETWORK FROM: %s" % network_file)
    network = RoadNetwork(network_file, network_geometry_file, overwrite=False, cache_dir= context.path())
    return network
