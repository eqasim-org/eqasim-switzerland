import os
import matsim.runtime.pt2matsim as pt2matsim
import sys
import shutil
from matsim.readers import read_network
import json
import re

def configure(context):
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.pt2matsim")
    context.stage("data.osm.clean")
        
    context.config("export_detailed_network", False)
    context.config("simplify_network_in_eqasim", False)
    context.config("correct_links_capacity", False)
    context.config("minimum_speed", 3) #in km/h
    context.config("input_downsampling")

def execute(context):
    network_file = context.stage("data.osm.clean")

    pt2matsim.run(context, "org.matsim.pt2matsim.run.CreateDefaultOsmConfig", [
        "convert_network_template.xml"
    ],[])

    # Create MATSim network
    with open("%s/convert_network_template.xml" % context.path()) as f_read:
        content = f_read.read()

        content = content.replace(
            '<param name="osmFile" value="null" />',
            '<param name="osmFile" value="%s" />' % network_file
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
        if context.config("export_detailed_network"):
            content = content.replace(
                '<param name="outputDetailedLinkGeometryFile" value="null" />',
                '<param name="outputDetailedLinkGeometryFile" value="detailed_network.csv" />',
            )

        content = content.replace(
            '<param name="parseTurnRestrictions" value="false" />',
            '<param name="parseTurnRestrictions" value="true" />'
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


        with open("%s/convert_network.xml" % context.path(), "w+") as f_write:
            f_write.write(content)
            
    pt2matsim.run(context, "org.matsim.pt2matsim.run.Osm2MultimodalNetwork", [
        "%s/convert_network.xml" % context.path()
    ],[])
    
    if context.config("simplify_network_in_eqasim") or context.config("correct_links_capacity"):
        network_path =  "%s/converted_network.xml.gz" % context.path()
        net = read_network(network_path)

        if context.config("simplify_network_in_eqasim") :
            stats = net.clean_network()
            # Save stats
            with open("%s/statistics_of_cleaning_network.json" % context.path(), "w") as f:
                json.dump(stats, f, indent=4) 

        if context.config("correct_links_capacity"):
            sampling_rate = context.config("input_downsampling")
            net.correct_capacity( sampling_rate = sampling_rate,
                                  minimum_speed = context.config("minimum_speed")/3.6)
            
        # Do not remove the last version of the network, just rename it.
        shutil.move(network_path, network_path.replace("converted_network","converted_network_uncleaned"))
        net.save(network_path)
        

    assert (os.path.exists("%s/converted_network.xml.gz" % context.path()))
    return "%s/converted_network.xml.gz" % context.path()
