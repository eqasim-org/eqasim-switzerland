import logging
from .network import RoadNetwork
logger = logging.getLogger("synpp")

# TODO: this stage need to be renamed to network_from_mapped
# this is the same stage as .network. However, for dependencies issues, we build another stage here to get the network from
# the prepare stage. The .network stage get the network from the output folder, thus not usable before running the simulation.

def configure(context):
    context.stage("data.spatial.swiss_border")
    context.stage("matsim.scenario.network.mapped")

    context.config("output_prefix", "switzerland_")
    context.config("export_detailed_network", True)


def execute(context):
    mapped = context.stage("matsim.scenario.network.mapped")
    network_file = mapped["road_network"]
    network_geometry_file = mapped["geometry"]

    logger.info("\t LOADING NETWORK FROM: %s" % network_file)
    network = RoadNetwork(network_file, network_geometry_file, overwrite=True, cache_dir= context.path(), context=context)
    return network
