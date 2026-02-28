import os
import re

import matsim.runtime.pt2matsim as pt2matsim


def configure(context):
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.pt2matsim")
    context.stage("data.osm.clean")
    context.config("data_path")
    context.config("export_detailed_network")
    context.config("parseTurnRestrictions", True)
def execute(context):
    network_file = context.stage("data.osm.clean")

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

    assert (os.path.exists("%s/converted_network.xml.gz" % context.path()))
    return "%s/converted_network.xml.gz" % context.path()
