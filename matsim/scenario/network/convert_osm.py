import os

import matsim.runtime.pt2matsim as pt2matsim


def configure(context):
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.pt2matsim")
    context.config("data_path")


def execute(context):

    pt2matsim.run(context, "org.matsim.pt2matsim.run.CreateDefaultOsmConfig", [
        "convert_network_template.xml"
    ])

    # Create MATSim network
    with open("%s/convert_network_template.xml" % context.path()) as f_read:
        content = f_read.read()

        content = content.replace(
            '<param name="osmFile" value="null" />',
            '<param name="osmFile" value="%s/osm/switzerland-latest.osm.gz" />' % context.config("data_path")
        )
        
        content = content.replace(
            '<param name="outputCoordinateSystem" value="null" />',
            '<param name="outputCoordinateSystem" value="epsg:2056" />'
        )

        content = content.replace(
            '<param name="outputNetworkFile" value="null" />',
            '<param name="outputNetworkFile" value="%s/converted_network.xml.gz" />' % context.path()
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
    ])

    assert (os.path.exists("%s/converted_network.xml.gz" % context.path()))
    return "%s/converted_network.xml.gz" % context.path()
