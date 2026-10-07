import os

import matsim.scenario.network.utils.pt_route_check as pt_route_check


def configure(context):
    context.stage("matsim.scenario.network.mapped")
    context.config("data_path")
    context.config("pt_line_routes_path", "TPG_line_routes/TPG_LIGNES-SHP/TPG_LIGNES.shp")
    context.config("pt_route_modes", ["bus"])
    context.config("pt_route_buffer", 30.0)
    context.config("pt_route_min_share", 0.9)
    # GTFS agency_ids (e.g. 881 = TPG) whose lines are compared with the official routes. The
    # shapefile only holds the lines of one operator, but other operators have lines with the
    # same name in the area. Empty: all lines are compared.
    context.config("pt_route_agencies", ["881"])
    if context.config("pt_route_agencies"):
        context.stage("data.gtfs.cleaned")


def execute(context):
    """Reports which PT routes mapped by pt2matsim deviate from the official line routes."""
    paths = context.stage("matsim.scenario.network.mapped")
    report_path = "%s/route_check.csv" % context.path()

    pt_route_check.run(
        schedule_path = paths["schedule"], network_path = paths["network"],
        shapefile_path = os.path.join(context.config("data_path"), context.config("pt_line_routes_path")),
        report_path = report_path, corrected_schedule_path = None,
        modes = tuple(context.config("pt_route_modes")),
        buffer_m = context.config("pt_route_buffer"), min_share = context.config("pt_route_min_share"),
        agencies = [str(a) for a in context.config("pt_route_agencies")],
        gtfs_routes_path = "%s/output/routes.txt" % context.path("data.gtfs.cleaned") if context.config("pt_route_agencies") else None,
    )

    return report_path
