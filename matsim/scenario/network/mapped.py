import os

import matsim.runtime.pt2matsim as pt2matsim
# import matsim.runtime.java as java


def configure(context):    
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.pt2matsim")
    
    context.stage("matsim.scenario.network.convert_osm")
    context.stage("matsim.scenario.network.convert_hafas")
    
    context.config("threads")


def execute(context):
    # jar, tmp_path = context.stage("matsim.java.pt2matsim")
    # java = context.stage("utils.java")

    unmapped_network_path = context.stage("matsim.scenario.network.convert_osm")
    unmapped_schedule_path = context.stage("matsim.scenario.network.convert_hafas")["schedule"]

    # Create and modify config file
    pt2matsim.run(context, "org.matsim.pt2matsim.run.CreateDefaultPTMapperConfig", [
        "map_network_template.xml"
    ])

    # java.run(context, "org.matsim.pt2matsim.run.CreateDefaultPTMapperConfig", [
    #     "map_network_template.xml"
    # ], cwd=context.path(), vm_arguments=["-Djava.io.tmpdir=%s" % tmp_path])

    # content = open("%s/map_network_template.xml" % context.path()).read()
    
    with open("%s/map_network_template.xml" % context.path()) as f_read:
        content = f_read.read()

        content = content.replace(
            '<param name="inputNetworkFile" value="" />',
            '<param name="inputNetworkFile" value="%s" />' % unmapped_network_path
        )
        content = content.replace(
            '<param name="inputScheduleFile" value="" />',
            '<param name="inputScheduleFile" value="%s" />' % unmapped_schedule_path
        )
        content = content.replace(
            '<param name="numOfThreads" value="2" />',
            '<param name="numOfThreads" value="%d" />' % context.config("threads")
        )
        content = content.replace(
            '<param name="outputNetworkFile" value="" />',
            '<param name="outputNetworkFile" value="%s/mapped_network.xml.gz" />' % context.path()
        )
        content = content.replace(
            '<param name="outputScheduleFile" value="" />',
            '<param name="outputScheduleFile" value="%s/mapped_schedule.xml.gz" />' % context.path()
        )
        content = content.replace(
            '<param name="outputStreetNetworkFile" value="" />',
            '<param name="outputStreetNetworkFile" value="%s/road_network.xml.gz" />' % context.path()
        )

        content = content.replace(
            '<param name="modesToKeepOnCleanUp" value="car" />',
            '<param name="modesToKeepOnCleanUp" value="car,car_passenger,truck" />'
        )

        with open("%s/map_network.xml" % context.path(), "w+") as f:
            f.write(content)

    # java(jar, "org.matsim.pt2matsim.run.PublicTransitMapper", [
    #     "map_network.xml"
    # ], cwd=context.path(), vm_arguments=["-Djava.io.tmpdir=%s" % tmp_path])
    
    # Run mapping process
    pt2matsim.run(context, "org.matsim.pt2matsim.run.PublicTransitMapper", [
        "map_network.xml"
    ])

    assert (os.path.exists("%s/mapped_network.xml.gz" % context.path()))
    assert (os.path.exists("%s/mapped_schedule.xml.gz" % context.path()))
    assert (os.path.exists("%s/road_network.xml.gz" % context.path()))
    assert (os.path.exists(context.stage("matsim.scenario.network.convert_hafas")["vehicles"]))

    return {
        "network": "%s/mapped_network.xml.gz" % context.path(),
        "schedule": "%s/mapped_schedule.xml.gz" % context.path(),
        "road_network": "%s/road_network.xml.gz" % context.path(),
        "vehicles": context.stage("matsim.scenario.network.convert_hafas")["vehicles"]
    }
