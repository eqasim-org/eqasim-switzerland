import os
import matsim.scenario.network.utils.pt_route_check as pt_route_check


def configure(context):
    context.stage("matsim.scenario.network.mapped")
    context.config("data_path")
    context.config("pt_line_routes_path", "TPG_line_routes/TPG_LIGNES-SHP/TPG_LIGNES.shp")
    context.config("pt_route_modes", ["bus"])
    # Links that do not allow the route mode but one of these modes can still be used by the
    # re-routing: buses run there in reality, so the route mode is added to them if they are used.
    # taxi: lets the routes use bus lanes that OSM opens to taxis but not to cars.
    context.config("pt_route_fallback_modes", ["car", "taxi"])
    context.config("pt_route_buffer", 30.0)
    context.config("pt_route_min_share", 0.9)
    # GTFS agency_ids (e.g. 881 = TPG) whose lines are compared with the official routes. The
    # shapefile only holds the lines of one operator, but other operators have lines with the
    # same name in the area. Empty: all lines are compared.
    context.config("pt_route_agencies", ["881"])
    if context.config("pt_route_agencies"):
        context.stage("data.gtfs.cleaned")
    context.config("pt_route_stop_radius", 80.0)
    context.config("pt_route_add_links", True)


def execute(context):
    """Re-routes the PT routes mapped by pt2matsim that deviate from the official line routes.

    The schedule changes (stop copies and route links). Where a corrected route uses a road on
    which the network does not allow buses (the OSM data lacks the bus mode there), the bus mode
    is added to that link in a patched copy of the network. Where the network has no sensible
    connection between two stops although the official route runs between them (e.g. a missing
    connection), bus-only links are added along the official route in that copy too. Vehicles are
    those of the mapped stage.
    """
    paths = context.stage("matsim.scenario.network.mapped")
    corrected_schedule_path = "%s/corrected_schedule.xml.gz" % context.path()
    patched_network_path = "%s/corrected_network.xml.gz" % context.path()

    _, n_patched = pt_route_check.run(
        schedule_path           = paths["schedule"], network_path = paths["network"],
        shapefile_path          = os.path.join(context.config("data_path"), context.config("pt_line_routes_path")),
        report_path             = "%s/route_correction.csv" % context.path(),
        corrected_schedule_path = corrected_schedule_path, patched_network_path = patched_network_path,
        modes                   = tuple(context.config("pt_route_modes")),
        fallback_modes          = tuple(context.config("pt_route_fallback_modes")),
        buffer_m                = context.config("pt_route_buffer"), min_share = context.config("pt_route_min_share"),
        stop_radius             = context.config("pt_route_stop_radius"),
        add_links               = context.config("pt_route_add_links"),
    )

    assert os.path.exists(corrected_schedule_path)

    return dict(
        network      = patched_network_path if n_patched > 0 else paths["network"],
        schedule     = corrected_schedule_path,
        road_network = paths["road_network"],
        vehicles     = paths["vehicles"],
        report       = "%s/route_correction.csv" % context.path(),
    )
