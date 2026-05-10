import logging

logger = logging.getLogger("synpp")

def configure(context):
    context.config("locations_algorithm", default = "nn")
    if context.config("locations_algorithm").lower() in ["nn", "neural_network", "v2", "v.2", "locations_v2"]:
        context.stage("synthesis.population.spatial.secondary_nn.locations")
    else:
        context.stage("synthesis.population.spatial.secondary.locations_rda")


def execute(context):  
    if context.config("locations_algorithm").lower() in ["nn", "neural_network", "v2", "v.2", "locations_v2"]:
        return context.stage("synthesis.population.spatial.secondary_nn.locations")
    else:
        return context.stage("synthesis.population.spatial.secondary.locations_rda")