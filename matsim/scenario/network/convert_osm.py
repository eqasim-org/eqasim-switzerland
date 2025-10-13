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


def configure(context):
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.pt2matsim")
    context.stage("data.osm.clean")
    context.stage("data.osm.traffic_lights")
    context.stage("data.spatial.swiss_border")
    
    context.config("data_path")        
    context.config("export_detailed_network", False)
    context.config("simplify_network_in_eqasim", False)
    context.config("correct_links_capacity", False)
    context.config("minimum_speed", 3) #in km/h
    context.config("input_downsampling")
    context.config("add_trafic_lights", False)
    context.config("assign_elevations", False)
    context.config("parseTurnRestrictions", True)

def execute(context):
    network_file = context.stage("data.osm.clean")
    # Export the detailed network if the traffic lights are added
    export_detailed_network = context.config("export_detailed_network") if not context.config("add_trafic_lights") else True

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
            '<param name="maxLinkLength" value="99999.0" />'
        )
        # Export detailed geometry of the links if needed
        if export_detailed_network:
            content = content.replace(
                '<param name="outputDetailedLinkGeometryFile" value="null" />',
                '<param name="outputDetailedLinkGeometryFile" value="%s/detailed_network.csv" />' % context.path(),
            )

        if context.config("parseTurnRestrictions"):
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
    

    # Here we correct the capacity, simplify the network or include the traffic lights if required
    # The network is read once because reading it multiple times is slow

    if (context.config("simplify_network_in_eqasim") or 
        context.config("correct_links_capacity") or
        context.config("add_trafic_lights") or
        context.config("assign_elevations")):
       
        # Read the network
        network_path =  "%s/converted_network.xml.gz" % context.path()
        net = read_network(network_path)
        
        # Assign elevations to the nodes if requested
        if context.config("assign_elevations"):
            df_switzerland = context.stage("data.spatial.swiss_border")
            ch_polygon = df_switzerland.buffer(0).iloc[0] 
            net = ElevationEstimator(network=net,
                                     data_path=context.config("data_path")
                                     ).run()

        # If traffic lights are requested, process them first (before simplifying the network)
        if context.config("add_trafic_lights"):
            traffic_lights_path = context.stage("data.osm.traffic_lights")
            detailed_network_path = "%s/detailed_network.csv" % context.path()
            net.links = TrafficLightsMatcher(net).run(traffic_lights_path, detailed_network_path)             

        # Simplify the network if requested, merge short links and remove loops and unconnected links
        if context.config("simplify_network_in_eqasim") :
            net, stats = networkCleaner(net).run()
            # Save stats
            with open("%s/statistics_of_cleaning_network.json" % context.path(), "w") as f:
                json.dump(stats, f, indent=4) 

        # correct link capacity for short links
        if context.config("correct_links_capacity"):
            sampling_rate = context.config("input_downsampling")
            net.links = CapacityCorrector(net).run(  sampling_rate=sampling_rate,
                                                     minimum_speed=context.config("minimum_speed")/3.6)
            

            
        # Do not remove the last version of the network, just rename it.
        shutil.move(network_path, network_path.replace("converted_network","converted_network_uncleaned"))
        net.save(network_path)
        

    assert (os.path.exists("%s/converted_network.xml.gz" % context.path()))
    return "%s/converted_network.xml.gz" % context.path()
