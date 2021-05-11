import os


def configure(context):
    context.stage("matsim.java.pt2matsim")
    context.stage("utils.java")
    context.config("data_path")


def execute(context):
    jar, tmp_path = context.stage("matsim.java.pt2matsim")
    java = context.stage("utils.java")

    # Create MATSim network

    java(jar, "org.matsim.pt2matsim.run.CreateDefaultOsmConfig", [
        "convert_network_template.xml"
    ], cwd=context.cache_path, vm_arguments=["-Djava.io.tmpdir=%s" % tmp_path])

    content = open("%s/convert_network_template.xml" % context.cache_path).read()

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
        '<param name="outputNetworkFile" value="%s/converted_network.xml.gz" />' % context.cache_path
    )

    content = content.replace(
        '</module>',
        """
            <parameterset type="routableSubnetwork" >
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

    with open("%s/convert_network.xml" % context.cache_path, "w+") as f:
        f.write(content)

    java(jar, "org.matsim.pt2matsim.run.Osm2MultimodalNetwork", [
        "convert_network.xml"
    ], cwd=context.cache_path, vm_arguments=["-Djava.io.tmpdir=%s" % tmp_path])

    assert (os.path.exists("%s/converted_network.xml.gz" % context.cache_path))
    return "%s/converted_network.xml.gz" % context.cache_path
