import os
import matsim.scenario.network.osmosis
import matsim.runtime.pt2matsim as pt2matsim
import sys
import shutil
from matsim.readers import read_network
import json
import re

def configure(context):
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.pt2matsim")
    context.stage("matsim.scenario.network.osmosis")
    
    context.config("data_path")
    context.config("osm_path", "switzerland-latest.osm.gz")
    context.config("export_detailed_network", False)
    context.config("simplify_network_in_eqasim", False)
    context.config("correct_links_capacity", False)
    context.config("input_downsampling")


###############################################################
# Functions used to convert the network file:

def convert_pbf_to_osm_pyosmium(input_file, output_file):
    import osmium # Import it only if it is used (maybe this function is never used!)
    
    class OSMHandler(osmium.SimpleHandler):
        def __init__(self, writer):
            super(OSMHandler, self).__init__()
            self.writer = writer

        def node(self, n):
            self.writer.add_node(n)

        def way(self, w):
            self.writer.add_way(w)

        def relation(self, r):
            self.writer.add_relation(r)

    print("using pyosmium to convert .pbf data")
    if os.path.exists(output_file):
        print("The file: %s already exists. It will be overridden." % output_file)
        os.remove(output_file)

    writer = osmium.SimpleWriter(output_file)
    handler = OSMHandler(writer)

    try:
        print(f"Processing {input_file} → {output_file} ...")
        handler.apply_file(input_file)
        print(f"Conversion successful: {output_file}")
    except Exception as e:
        print(f"Error during network conversion using pyosmium: {e}")
        sys.exit(1)
    finally:
        writer.close()

def convert_pbf_to_osm_osmosis(context, input_file, output_file):

    print("using osmosis to convert .pbf data")
    if os.path.exists(output_file):
        print("The file: %s already exists. It will be overridden." % output_file)
        os.remove(output_file)
    
    try:
        matsim.scenario.network.osmosis.run(context, [
                "--read-pbf", input_file,            
                "--tag-filter", "accept-ways", "highway=*", "railway=*",
                #"--tag-filter","reject-ways","highway=service",
                "completeWays=yes",     
                "--used-node", 
                "--write-xml", "compressionMethod=gzip", output_file
            ])
        
    except Exception as e:
        print(f"Error during network conversion using osmosis: {e}")
        sys.exit(1)

############################################################## 


def execute(context):
    osm_file = '%s/osm/%s' % (context.config("data_path"), context.config("osm_path"))

    if context.config("osm_path").endswith('.pbf'):
        osmosis_path = shutil.which("osmosis") 
        # If osmosis is installed, use it, else, use pyosmium
        if osmosis_path:
            new_file_name = osm_file[:-8]+"-osmosis.osm.gz"
            convert_pbf_to_osm_osmosis(context, osm_file, new_file_name)
        else:
            new_file_name = osm_file[:-8]+"-pyosmium.osm"
            convert_pbf_to_osm_pyosmium(osm_file, new_file_name)

        osm_file = new_file_name

    pt2matsim.run(context, "org.matsim.pt2matsim.run.CreateDefaultOsmConfig", [
        "convert_network_template.xml"
    ],[])

    # Create MATSim network
    with open("%s/convert_network_template.xml" % context.path()) as f_read:
        content = f_read.read()

        content = content.replace(
            '<param name="osmFile" value="null" />',
            '<param name="osmFile" value="%s" />' % osm_file
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
            net.correct_capacity(sampling_rate = sampling_rate,minimum_speed=2/3.6)
            
        # Do not remove the last version of the network, just rename it.
        shutil.move(network_path, network_path.replace("converted_network","converted_network_uncleaned"))
        net.save(network_path)
        

    assert (os.path.exists("%s/converted_network.xml.gz" % context.path()))
    return "%s/converted_network.xml.gz" % context.path()
