import logging

import matsim.scenario.network.utils.schedule_transfers as schedule_transfers

logger = logging.getLogger("synpp")


def configure(context):
    # Whether the PT routes mapped by pt2matsim are re-routed along the official line routes
    # (see matsim.scenario.network.corrected).
    if context.config("pt_route_correction", True):
        context.stage("matsim.scenario.network.corrected", alias = "pt_scenario")
    else:
        context.stage("matsim.scenario.network.mapped", alias = "pt_scenario")


def execute(context):
    """The network, schedule and vehicles used by the simulation.

    The schedule loses its minimal transfer times from or to stops that no route serves, on which SwissRailRaptor
    fails (see utils.schedule_transfers).
    """
    paths = dict(context.stage("pt_scenario"))
    schedule_path = "%s/transit_schedule.xml.gz" % context.path()

    removed, total = schedule_transfers.remove_unserved_transfers(paths["schedule"], schedule_path)
    logger.info("Removed %d of %d minimal transfer times from or to stops served by no route" % (removed, total))

    paths["schedule"] = schedule_path
    return paths
