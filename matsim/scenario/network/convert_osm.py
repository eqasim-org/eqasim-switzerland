import re

import matsim.runtime.pt2matsim as pt2matsim
from matsim.scenario.network.utils.network_handler import NetworkHandler


def configure(context):
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.pt2matsim")
    context.stage("data.osm.clean")
    context.stage("data.osm.traffic_lights")
    context.stage("data.spatial.swiss_border")
    context.stage("data.spatial.municipality_types") # used in speed correction
    context.stage("data.spatial.municipalities") # used in speed correction
    context.stage("data.spatial.swiss_border") # used in speed correction
    context.stage("calibration.road_regions.penalty_calibration")

    context.config("data_path")
    context.config("osm_file", "switzerland-latest.osm.gz")
    context.config("border_offset", 20000)    
    context.config("export_detailed_network", True)    
    context.config("correct_links_capacity", True)
    context.config("minimum_speed", 1.0) #in km/h
    context.config("input_downsampling")
    context.config("add_traffic_lights", True)
    context.config("assign_elevations", True)
    context.config("parseTurnRestrictions", True)
    context.config("simplify_network_in_eqasim", True)
    # only if simplify network is true
    context.config("remove_network_loops", True)
    context.config("remove_replicate_links", True)
    context.config("remove_nodes_with_no_intersection", False)
    context.config("correct_speed", True)
    context.config("ensure_network_connectivity", True)
    # correct speed for uphill links only
    context.config("adjust_speed_uphill", True) # if true, it is triggered only if elevation is assigned
    context.config("adjust_speed_straightness", True) # if true, it is triggered only if elevation is assigned
    context.config("adjust_speed_mountain_links", True) # if true, it is triggered only if elevation is assigned
    context.config("max_gradient_threshold", 0.1) # in percentage (10% = 0.1)
    context.config("speed_factor_uphill", 0.9) 
    # reduce capacity outside border
    context.config("capacity_factor_outside_border", 0.5)
    # whether to route the bike in the network or not
    context.config("route_bike", True)


def execute(context):
    network_file = context.stage("data.osm.clean")
    # Export the detailed network if the traffic lights are added
    export_detailed_network = context.config("export_detailed_network") if not context.config("add_traffic_lights") else True

    pt2matsim.run(context, "org.matsim.pt2matsim.run.CreateDefaultOsmConfig", [
        "convert_network_template.xml"
    ],[])

    # Create MATSim network
    with open("%s/convert_network_template.xml" % context.path()) as f_read:
        content = f_read.read()

        content = content.replace(
            '<param name="allowedTransportModes" value="bus, car" />',
            '<param name="allowedTransportModes" value="bus" />'
        )

        content = re.sub(
            r'(<parameterset\s+type="wayDefaultParams"\s*>\s*<param\s+name="allowedTransportModes"\s+value=")car(")',
            r'\1car,taxi,bus\2',
            content,
            flags=re.DOTALL
        )

        content = content.replace(
            '<param name="osmFile" value="null" />',
            '<param name="osmFile" value="%s" />' % network_file
        )

        # Export detailed geometry of the links if needed
        if context.config("export_detailed_network"):
            content = content.replace(
                '<param name="outputDetailedLinkGeometryFile" value="null" />',
                '<param name="outputDetailedLinkGeometryFile" value="%s/detailed_network.csv" />' % context.path(),
            )
        
        content = content.replace(
            '<param name="outputCoordinateSystem" value="null" />',
            '<param name="outputCoordinateSystem" value="epsg:2056" />'
        )

        content = content.replace(
            '<param name="outputNetworkFile" value="null" />',
            '<param name="outputNetworkFile" value="%s/converted_network.xml.gz" />' % context.path()
        )
        
        # higher link length
        content = content.replace(
            '<param name="maxLinkLength" value="500.0" />',
            '<param name="maxLinkLength" value="1000.0" />'
        )
        # Export detailed geometry of the links if needed
        if export_detailed_network:
            content = content.replace(
                '<param name="outputDetailedLinkGeometryFile" value="null" />',
                '<param name="outputDetailedLinkGeometryFile" value="%s/detailed_network.csv" />' % context.path(),
            )

        if not context.config("parseTurnRestrictions"):
            content = content.replace(
                '<param name="parseTurnRestrictions" value="true" />',
                '<param name="parseTurnRestrictions" value="false" />'
            )

        content = content.replace(
            '</module>',
            """
                <parameterset type="routableSubnetwork">
                    <param name="allowedTransportModes" value="car" />
                    <param name="subnetworkMode" value="car_passenger" />
                </parameterset>
            </module>
            """
        )
        
        content = content.replace(
            '</module>',
            """
                <parameterset type="routableSubnetwork" >
                    <param name="allowedTransportModes" value="car" />
                    <param name="subnetworkMode" value="truck" />
                </parameterset>
            </module>
            """
        )

        content = content.replace(
            '</module>',
            """
                <parameterset type="routableSubnetwork" >
			        <param name="allowedTransportModes" value="taxi" />
			        <param name="subnetworkMode" value="taxi" />
		        </parameterset>
            </module>
            """
        )

        with open("%s/convert_network.xml" % context.path(), "w+") as f_write:
            f_write.write(content)
            
    pt2matsim.run(context, "org.matsim.pt2matsim.run.Osm2MultimodalNetwork", [
        "%s/convert_network.xml" % context.path()
    ],[])
    
    network_path = "%s/converted_network.xml.gz" % context.path()
    return NetworkHandler(context, network_path).process_network()
