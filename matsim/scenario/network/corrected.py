import os
import matsim.scenario.network.utils.pt_route_check as pt_route_check


def configure(context):
    context.stage("matsim.scenario.network.mapped")
    context.config("data_path")
    # Official line routes, by GTFS agency_id: {agency_id: shapefile}, paths absolute or relative to data_path. The lines
    # of a shapefile are only compared with the lines of its agency (Geneva and Lausanne both have lines 1 to 9...).
    # Also accepted: one path (used for the lines of all the agencies).
    context.config("pt_line_routes_path", pt_route_check.DEFAULT_LINE_ROUTES)
    context.config("pt_route_modes", ["bus"])
    # Links that do not allow the route mode but one of these modes can still be used by the
    # re-routing: buses run there in reality, so the route mode is added to them if they are used.
    # taxi: lets the routes use bus lanes that OSM opens to taxis but not to cars.
    context.config("pt_route_fallback_modes", ["car", "taxi"])
    context.config("pt_route_buffer", 30.0)
    context.config("pt_route_min_share", 0.9)
    # GTFS agency_ids (881 = TPG, 151 = TL) whose lines are compared with their reference routes (GTFS shapes, else
    # the official routes of their agency). Empty: all lines are compared.
    context.config("pt_route_agencies", ["881", "151"])
    # The routes are compared with the GTFS shape of their trip; the official line routes of pt_line_routes_path
    # are only used for the routes without one.
    context.config("pt_route_gtfs_shapes", True)
    # the GTFS routes give the agency of the lines, and the GTFS shapes the reference routes
    if context.config("pt_route_agencies") or context.config("pt_route_gtfs_shapes") or isinstance(context.config("pt_line_routes_path"), dict):
        context.stage("data.gtfs.cleaned")
    context.config("pt_route_stop_radius", 80.0)
    context.config("pt_route_add_links", True)


def line_routes_paths(context):
    """pt_line_routes_path with absolute paths: {agency_id: path}, one path, or None"""
    return pt_route_check.resolve_line_routes(context.config("pt_line_routes_path"), context.config("data_path"))


def execute(context):
    """Re-routes the PT routes mapped by pt2matsim that deviate from their reference route: the GTFS shape of
    their trip, or the official line route if the trip has no shape.

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
    agencies = context.config("pt_route_agencies")
    shapefile_path = line_routes_paths(context)
    by_agency = isinstance(shapefile_path, dict)
    gtfs_path = "%s/output" % context.path("data.gtfs.cleaned") if agencies or by_agency or context.config("pt_route_gtfs_shapes") else None
    use_shapes = context.config("pt_route_gtfs_shapes") and os.path.exists("%s/shapes.txt" % gtfs_path)

    _, n_patched = pt_route_check.run(
        schedule_path           = paths["schedule"], network_path = paths["network"],
        shapefile_path          = shapefile_path,
        report_path             = "%s/route_correction.csv" % context.path(),
        corrected_schedule_path = corrected_schedule_path, patched_network_path = patched_network_path,
        modes                   = tuple(context.config("pt_route_modes")),
        fallback_modes          = tuple(context.config("pt_route_fallback_modes")),
        buffer_m                = context.config("pt_route_buffer"), min_share = context.config("pt_route_min_share"),
        stop_radius             = context.config("pt_route_stop_radius"),
        add_links               = context.config("pt_route_add_links"),
        gtfs_routes_path        = "%s/routes.txt" % gtfs_path if agencies or by_agency or use_shapes else None,
        agencies                = agencies,
        gtfs_trips_path         = "%s/trips.txt" % gtfs_path if use_shapes else None,
        gtfs_shapes_path        = "%s/shapes.txt" % gtfs_path if use_shapes else None,
    )

    assert os.path.exists(corrected_schedule_path)

    return dict(
        network      = patched_network_path if n_patched > 0 else paths["network"],
        schedule     = corrected_schedule_path,
        road_network = paths["road_network"],
        vehicles     = paths["vehicles"],
        report       = "%s/route_correction.csv" % context.path(),
    )
