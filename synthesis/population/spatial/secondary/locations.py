import logging

logger = logging.getLogger("synpp")

def configure(context):
    context.config("locations_algorithm", default = "nn")
    if context.config("locations_algorithm").lower() in ["nn", "neural_network", "v2", "v.2", "locations_v2"]:
        context.stage("synthesis.population.spatial.secondary_nn.locations", alias="sec_locations")
    else:
        context.stage("synthesis.population.spatial.secondary.locations_rda", alias="sec_locations")


def execute(context):  
    return context.stage("sec_locations")
