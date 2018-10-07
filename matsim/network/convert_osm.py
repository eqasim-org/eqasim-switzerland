import subprocess as sp

def configure(context, require):
    require.stage("matsim.java.pt2matsim")
    require.config("raw_data_path")

def execute(context):
        jar = context.stage("matsim.java.pt2matsim")

        # Create MATSim network

        sp.check_call([
            "java", "-cp", jar, "org.matsim.pt2matsim.run.CreateDefaultOsmConfig", "convert_network_template.xml"
        ], cwd = context.cache_path, stdout = sp.PIPE, stderr = sp.PIPE)

        content = open("%s/convert_network_template.xml" % context.cache_path).read()

        content = content.replace(
            '<param name="osmFile" value="null" />',
            '<param name="osmFile" value="%s/osm/switzerland-latest.osm.gz" />' % context.config["raw_data_path"]
        )
        content = content.replace(
            '<param name="outputCoordinateSystem" value="null" />',
            '<param name="outputCoordinateSystem" value="EPSG:2056" />'
        )
        content = content.replace(
            '<param name="outputNetworkFile" value="null" />',
            '<param name="outputNetworkFile" value="%s/converted_network.xml.gz" />' % context.cache_path
        )

        with open("%s/convert_network.xml" % context.cache_path, "w+") as f:
            f.write(content)

        sp.check_call([
            "java", "-cp", jar, "org.matsim.pt2matsim.run.Osm2MultimodalNetwork", "convert_network.xml"
        ], cwd = context.cache_path)

        assert(os.path.exists("%s/converted_network.xml.gz" % context.cache_path))
        return "%s/converted_network.xml.gz" % context.cache_path
