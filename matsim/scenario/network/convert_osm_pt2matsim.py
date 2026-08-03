import matsim.runtime.pt2matsim as pt2matsim
from lxml import etree

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

def configure(context):
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.pt2matsim")
    context.stage("data.osm.clean")

    context.config("data_path")
    context.config("export_detailed_network", True)
    context.config("add_traffic_lights", True)
    context.config("parseTurnRestrictions", True)


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
    tree = etree.parse(template_path)
    root = tree.getroot()

    # Find the OsmConverter module unambiguously
    modules = root.xpath('.//module[@name="OsmConverter"]')
    if not modules:
        raise RuntimeError("OsmConverter module not found in template")
    osm_module = modules[0]

    # --- small helpers -------------------------------------------------
    def get_param(name, parent=osm_module):
        """Return a <param> element or None."""
        return parent.find(f'param[@name="{name}"]')

    def set_param(name, value, parent=osm_module):
        """Set a <param> value, creating it if missing."""
        param = get_param(name, parent)
        if param is None:
            param = etree.SubElement(parent, "param", name=name, value=str(value))
        else:
            param.set("value", str(value))

    # --- 1. Fix bus subnetwork ----------------------------------------
    for subnet in osm_module.xpath('parameterset[@type="routableSubnetwork"]'):
        allowed = get_param("allowedTransportModes", subnet)
        if allowed is not None and allowed.get("value") == "bus, car":
            allowed.set("value", "bus")

    # --- 2. Add taxi & bus to all road wayDefaultParams ---------------
    for way in osm_module.xpath('parameterset[@type="wayDefaultParams"]'):
        allowed = get_param("allowedTransportModes", way)
        if allowed is not None and allowed.get("value") == "car":
            allowed.set("value", "car,taxi,bus")
            highway = get_param("osmValue", way)
            if highway is not None:
                freespeed = DEFAULT_SPEEDS.get(highway.get('value'))
                if freespeed is not None:
                    way_speed = get_param("freespeed", way)
                    way_speed.set("value", str(freespeed))

    # --- 3. Set scalar parameters -------------------------------------
    set_param("osmFile", network_file)
    set_param("outputCoordinateSystem", "epsg:2056")
    set_param("outputNetworkFile", "%s/converted_network.xml.gz" % context.path())
    set_param("maxLinkLength", "1000.0")
    set_param("parseTurnRestrictions", str(context.config("parseTurnRestrictions")).lower())

    if export_detailed_network:
        set_param("outputDetailedLinkGeometryFile", detailed_network_file)
        
    # --- 4. Add new routableSubnetworks safely ------------------------
    new_subnetworks = [
        {"allowedTransportModes": "car", "subnetworkMode": "car_passenger"},
        {"allowedTransportModes": "car", "subnetworkMode": "truck"},
        {"allowedTransportModes": "taxi", "subnetworkMode": "taxi"},
    ]

    for sn in new_subnetworks:
        ps = etree.SubElement(osm_module, "parameterset", type="routableSubnetwork")
        etree.SubElement(ps, "param", name="allowedTransportModes", value=sn["allowedTransportModes"])
        etree.SubElement(ps, "param", name="subnetworkMode", value=sn["subnetworkMode"])

    # --- 5. Write back, preserving XML declaration + DOCTYPE ----------
    etree.indent(tree, space="    ")
    tree.write(
        output_path,
        xml_declaration=True,
        encoding="UTF-8",
        pretty_print=True,
        doctype=tree.docinfo.doctype,  # keeps <!DOCTYPE config SYSTEM "...">
    )

    # Run conversion
    pt2matsim.run(
        context,
        "org.matsim.pt2matsim.run.Osm2MultimodalNetwork",
        [output_path],
        [],
    )

    network_path = "%s/converted_network.xml.gz" % context.path()
    return network_path, detailed_network_file