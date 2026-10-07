def configure(context):
    # Whether the PT routes mapped by pt2matsim are re-routed along the official line routes
    # (see matsim.scenario.network.corrected).
    if context.config("pt_route_correction", False):
        context.stage("matsim.scenario.network.corrected", alias = "pt_scenario")
    else:
        context.stage("matsim.scenario.network.mapped", alias = "pt_scenario")


def execute(context):
    """The network, schedule and vehicles used by the simulation."""
    return context.stage("pt_scenario")
