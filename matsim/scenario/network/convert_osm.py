import os
import re

import matsim.runtime.pt2matsim as pt2matsim
import sys
import shutil
from matsim.readers import read_network
import json
import re
from matsim.scenario.network.utils.capacity_corrector import CapacityCorrector
from matsim.scenario.network.utils.traffic_light_matcher import TrafficLightsMatcher
from matsim.scenario.network.utils.elevation_estimator import ElevationEstimator
from matsim.scenario.network.utils.network_cleaner import networkCleaner
from matsim.scenario.network.utils.speed_corrector import SpeedCorrector


def configure(context):
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.pt2matsim")
    context.stage("data.osm.clean")
    context.stage("data.osm.traffic_lights")
    context.stage("data.spatial.swiss_border")
    context.stage("data.spatial.municipality_types") # used in speed correction
    context.stage("data.spatial.municipalities") # used in speed correction
    context.stage("data.spatial.swiss_border") # used in speed correction

    context.config("data_path")
    context.config("osm_file", "switzerland-latest.osm.gz")
    context.config("border_offset", 20000)    
    context.config("export_detailed_network", False)    
    context.config("correct_links_capacity", False)
    context.config("minimum_speed", 2) #in km/h
    context.config("input_downsampling")
    context.config("add_traffic_lights", False)
    context.config("assign_elevations", False)
    context.config("parseTurnRestrictions", True)
    context.config("simplify_network_in_eqasim", False)
    # only if simplify network is true
    context.config("remove_network_loops", True)
    context.config("remove_replicate_links", True)
    context.config("remove_nodes_with_no_intersection", False)
    context.config("correct_speed", True)
    context.config("ensure_network_connectivity", True)
    # this is for speed correction
    context.config("adjust_speed", True)
    context.config("speed_factor_urbancore", 0.85)
    context.config("speed_factor_urban", 0.93)
    context.config("speed_factor_suburban", 1.0)
    context.config("speed_factor_rural", 1.05)
    context.config("speed_factor_motorway", 1.15)
    context.config("speed_limit_for_correction", 75/3.6) # in m/s (speed limit below which we correct the speed)
    # correct speed for uphill links only
    context.config("adjust_speed_uphill", False) # if true, it is triggered only if elevation is assigned
    context.config("max_gradient_threshold", 0.1) # in percentage (10% = 0.1)
    context.config("speed_factor_uphill", 0.9) 
    # reduce capacity outside border
    context.config("capacity_factor_outside_border", 1)
    # reduce speed outside border
    context.config("speed_factor_outside_border", 1)

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
            '<param name="maxLinkLength" value="100.0" />'
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
    
   
    # Read the network
    network_path =  "%s/converted_network.xml.gz" % context.path()
    net = read_network(network_path)
    
    # Assign municipality type
    net.links = SpeedCorrector(context, net).assign_municipality_types_to_network()

    # Assign elevations to the nodes if requested
    if context.config("assign_elevations"):
        df_switzerland = context.stage("data.spatial.swiss_border")
        ch_polygon = df_switzerland.buffer(0).iloc[0] 
        net = ElevationEstimator(network=net,
                                    data_path=context.config("data_path"),
                                    polygone = ch_polygon
                                    ).run()

    # If traffic lights are requested, process them first (before simplifying the network)
    if context.config("add_traffic_lights"):
        traffic_lights_path = context.stage("data.osm.traffic_lights")
        detailed_network_path = "%s/detailed_network.csv" % context.path()
        net.links = TrafficLightsMatcher(net).run(traffic_lights_path, detailed_network_path)             

    # Simplify the network if requested, merge short links and remove loops and unconnected links
    if context.config("simplify_network_in_eqasim") :
        net, stats = networkCleaner(net).run(
            remove_network_loops=context.config("remove_network_loops"),
            remove_replicate_links=context.config("remove_replicate_links"),
            remove_nodes_with_no_intersection=context.config("remove_nodes_with_no_intersection"),
            correct_speeds=context.config("correct_speed"),
            ensure_network_connectivity=context.config("ensure_network_connectivity")
        )
        # Save stats
        with open("%s/statistics_of_cleaning_network.json" % context.path(), "w") as f:
            json.dump(stats, f, indent=4) 

    # correct link capacity for short links
    if context.config("correct_links_capacity"):
        net.links = CapacityCorrector(context, net).run()
    
    reduce_capacity_or_speed_outside_border  = (isinstance(context.config("osm_file"),list) & 
                                                (context.config("border_offset")>0) &
                                                ((context.config("capacity_factor_outside_border")<1) or 
                                                 (context.config("speed_factor_outside_border")<1))
                                                )
    if reduce_capacity_or_speed_outside_border:
        if context.config("capacity_factor_outside_border")<1:
            net.links = CapacityCorrector(context, net).reduce_capacity_outside_border()
        if context.config("speed_factor_outside_border")<1:
            net.links = SpeedCorrector(context, net).run("outside_border")
    
    if context.config("adjust_speed"):
        # adjust link speeds of car links based on municipality types and their speed limit
        net.links = SpeedCorrector(context, net).run("municipality_type")
        if abs(context.config("speed_factor_motorway")-1.0)>1e-3:
            net.links = SpeedCorrector(context, net).run("motorway")

    if context.config("adjust_speed_uphill"):
        if not context.config("assign_elevations"):
            raise ValueError("To correct speeds of uphill links, elevations must be assigned first.")
        # further correct link speeds of uphill links based on their gradient
        net.links = SpeedCorrector(context, net).run("uphill")

    
    if context.config("route_bike"):
        # the bike is added to car, bus, truck and taxi links. however, we check for network connectivity first
        net.links = networkCleaner(net).add_bike_to_network()
        
    # Do not remove the last version of the network, just rename it.
    shutil.move(network_path, network_path.replace("converted_network","converted_network_uncleaned"))
    net.save(network_path)
    
    assert (os.path.exists("%s/converted_network.xml.gz" % context.path()))
    return "%s/converted_network.xml.gz" % context.path()
