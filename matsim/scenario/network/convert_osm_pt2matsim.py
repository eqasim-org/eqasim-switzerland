import matsim.runtime.pt2matsim as pt2matsim
from lxml import etree
from functools import partial
from matsim.config import tools as config_tools


def configure(context):
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.pt2matsim")
    context.stage("data.osm.clean")

    context.config("data_path")
    context.config("export_detailed_network", True)
    context.config("add_traffic_lights", True)
    context.config("parseTurnRestrictions", True)
    context.config("pt2matsim.respectVehicleAccess", True)
    context.config("pt2matsim.allowPtRoutesOnPrivateRoads", True)
    context.config("pt2matsim.maxLinkLength", 1000.0)
    context.config("pt2matsim.useOsmTurnArrows", True)
    context.config("java_memory", "50G", volatile=True)


def execute(context):
    network_file = context.stage("data.osm.clean")
    export_detailed_network = (
        context.config("export_detailed_network")
        if not context.config("add_traffic_lights")
        else True
    )
    detailed_network_file = "%s/detailed_network.csv" % context.path()

    # Generate template
    pt2matsim.run(
        context,
        "org.matsim.pt2matsim.run.CreateDefaultOsmConfig",
        ["convert_network_template.xml"],
        [],
    )

    template_path = "%s/convert_network_template.xml" % context.path()
    output_path = "%s/convert_network.xml" % context.path()

    # ------------------------------------------------------------------
    # Parse XML properly instead of hacking strings
    # ------------------------------------------------------------------
    tree = config_tools.read_config(template_path)
    root = tree.getroot()

    # Find the OsmConverter module unambiguously
    modules = root.xpath('.//module[@name="OsmConverter"]')
    if not modules:
        raise RuntimeError("OsmConverter module not found in template")
    osm_module = modules[0]

    # --- small helpers -------------------------------------------------
    get_param = partial(config_tools.get_param, parent=osm_module)
    set_param = partial(config_tools.set_param, parent=osm_module)

    # --- 1. Fix bus subnetwork ----------------------------------------
    for subnet in osm_module.xpath('parameterset[@type="routableSubnetwork"]'):
        allowed = get_param("allowedTransportModes", parent=subnet)
        if allowed is not None and allowed.get("value") == "bus, car":
            config_tools.set_param("allowedTransportModes", "bus", subnet)

    # --- 2. Add taxi & bus to all road wayDefaultParams ---------------
    for way in osm_module.xpath('parameterset[@type="wayDefaultParams"]'):
        allowed = get_param("allowedTransportModes", parent=way)
        if allowed is not None and allowed.get("value") == "car":
            config_tools.set_param("allowedTransportModes", "car,taxi,bus", way)
            highway = get_param("osmValue", parent=way)
            if highway is not None:
                freespeed = DEFAULT_SPEEDS.get(highway.get('value'))
                if freespeed is not None:
                    config_tools.set_param("freespeed", freespeed, way)

    # --- 3.1 Set scalar parameters -------------------------------------
    set_param("osmFile", network_file)
    set_param("outputCoordinateSystem", "epsg:2056")
    set_param("outputNetworkFile", "%s/converted_network.xml.gz" % context.path())
    set_param("maxLinkLength", str(round(context.config("pt2matsim.maxLinkLength"),1)))
    set_param("parseTurnRestrictions", str(context.config("parseTurnRestrictions")).lower())
    set_param("respectVehicleAccess", str(context.config("pt2matsim.respectVehicleAccess")).lower())
    set_param("allowPtRoutesOnPrivateRoads", str(context.config("pt2matsim.allowPtRoutesOnPrivateRoads")).lower())
    set_param("useOsmTurnArrows", str(context.config("pt2matsim.useOsmTurnArrows")).lower())

    if export_detailed_network:
        set_param("outputDetailedLinkGeometryFile", detailed_network_file)
    # --- 4. Add new routableSubnetworks safely ------------------------
    new_subnetworks = [
        {"allowedTransportModes": "car", "subnetworkMode": "car_passenger"},
        {"allowedTransportModes": "car", "subnetworkMode": "truck"},
        {"allowedTransportModes": "taxi", "subnetworkMode": "taxi"},
    ]

    for sn in new_subnetworks:
        ps = config_tools.get_parameter_set(osm_module, "routableSubnetwork",
                                             {"subnetworkMode": sn["subnetworkMode"]})
        config_tools.set_param("allowedTransportModes", sn["allowedTransportModes"], ps)

    # --- 5. Write back, preserving XML declaration + DOCTYPE ----------
    etree.indent(tree, space="    ")
    config_tools.write_config(tree, output_path)

    # Run conversion
    pt2matsim.run(
        context,
        "org.matsim.pt2matsim.run.Osm2MultimodalNetwork",
        [output_path],
        [],
    )

    network_path = "%s/converted_network.xml.gz" % context.path()
    return network_path, detailed_network_file









################ defaults and options ##################
DEFAULT_SPEEDS = {
    "motorway": 120,
    "motorway_link": 80,
    "trunk": 100,
    "trunk_link": 80,
    "primary": 80,
    "primary_link": 50,
    "secondary": 50,
    "secondary_link": 30,
    "tertiary": 50,
    "tertiary_link": 30,
    "unclassified": 30,
    "residential": 30,
    "living_street": 20,
}
DEFAULT_SPEEDS = {k: round(v / 3.6,1) for k, v in DEFAULT_SPEEDS.items()}  # convert km/h to m/s
