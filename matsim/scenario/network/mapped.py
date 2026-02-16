import os

import matsim.runtime.pt2matsim as pt2matsim
# import matsim.runtime.java as java


def configure(context):    
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.pt2matsim")
    
    context.stage("matsim.scenario.network.convert_osm")
    context.stage("matsim.scenario.network.convert_pt_schedule")
    
    context.config("threads")
    context.config("route_bike", True)


def execute(context):
    # jar, tmp_path = context.stage("matsim.java.pt2matsim")
    # java = context.stage("utils.java")

    unmapped_network_path = context.stage("matsim.scenario.network.convert_osm")
    unmapped_schedule_path = context.stage("matsim.scenario.network.convert_pt_schedule")["schedule"]

    # Create and modify config file
    pt2matsim.run(context, "org.matsim.pt2matsim.run.CreateDefaultPTMapperConfig", [
        "map_network_template.xml"
    ], [])

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

        modesToKeep = "car,car_passenger,truck,taxi"
        if context.config("route_bike"):
            modesToKeep += ",bike"

        content = content.replace(
            '<param name="modesToKeepOnCleanUp" value="car" />',
            '<param name="modesToKeepOnCleanUp" value="%s" />' % modesToKeep
        )

        content = content.replace(
            '<param name="networkRouter" value="SpeedyALT" />',
            '<param name="networkRouter" value="AStarLandmarks" />'
        )

        content = content.replace(
            '<param name="networkModes" value="car,bus" />',
            '<param name="networkModes" value="bus" />'
        )
        
        content = content.replace(
            '<param name="modeSpecificRules" value="false" />',
            '<param name="modeSpecificRules" value="true" />'
        )

        content = content.replace(
            '<param name="maxTravelCostFactor" value="5.0" />',
            '<param name="maxTravelCostFactor" value="10" />' # before 6.5
        )

        

        content = content.replace(
            '</module>',
            """
              <parameterset type="transportModeAssignment" >
			        <param name="maxLinkCandidateDistance" value="120.0" />
			        <param name="nLinkThreshold" value="1" />
			        <param name="networkModes" value="rail,light_rail,train" />
			        <param name="scheduleMode" value="rail" />
			        <param name="strictLinkRule" value="true" />
		        </parameterset>
            
               <parameterset type="transportModeAssignment" >
			        <param name="maxLinkCandidateDistance" value="180.0" />
			        <param name="nLinkThreshold" value="1" />
			        <param name="networkModes" value="tram" />
			        <param name="scheduleMode" value="tram" />
			        <param name="strictLinkRule" value="true" />
		        </parameterset>

				<parameterset type="transportModeAssignment" >
			        <param name="maxLinkCandidateDistance" value="180.0" />
			        <param name="nLinkThreshold" value="1" />
			        <param name="networkModes" value="light_rail,subway" />
			        <param name="scheduleMode" value="subway" />
			        <param name="strictLinkRule" value="true" />
		        </parameterset>

				<parameterset type="transportModeAssignment" >
			        <param name="maxLinkCandidateDistance" value="120.0" />
			        <param name="nLinkThreshold" value="1" />
			        <param name="networkModes" value="funicular" />
			        <param name="scheduleMode" value="funicular" />
			        <param name="strictLinkRule" value="true" />
		        </parameterset>
            </module>
            """
        )
        

        with open("%s/map_network.xml" % context.path(), "w+") as f:
            f.write(content)

    log4j_content = """<?xml version="1.0" encoding="UTF-8"?>
        <Configuration status="WARN">
        <Appenders>

            <!-- INFO goes to stdout -->
            <Console name="InfoOut" target="SYSTEM_OUT">
            <PatternLayout
                pattern="%d{yyyy-MM-dd HH:mm:ss.SSS} %-5level %msg%n"/>
            <LevelRangeFilter minLevel="info" maxLevel="info"
                                onMatch="ACCEPT" onMismatch="DENY"/>
            </Console>

            <!-- ERROR & FATAL go to stderr -->
            <Console name="ErrOut" target="SYSTEM_ERR">
            <PatternLayout
                pattern="%d{yyyy-MM-dd HH:mm:ss.SSS} %-5level %msg%n"/>
            <ThresholdFilter level="error"
                            onMatch="ACCEPT" onMismatch="DENY"/>
            </Console>

        </Appenders>

        <Loggers>
            <!-- keep root at INFO so INFO events are produced -->
            <Root level="info">
            <AppenderRef ref="InfoOut"/>
            <AppenderRef ref="ErrOut"/>
            </Root>
        </Loggers>
        </Configuration>
        """

    # Write to the file
    with open("%s/log4j.xml" % context.path(), 'w', encoding='utf-8') as f:
        f.write(log4j_content)

    # Run mapping process
    pt2matsim.run(context, "org.matsim.pt2matsim.run.PublicTransitMapper", [
        "map_network.xml"
    ], vm_arguments=["-Dlog4j.configurationFile=file:log4j.xml"])

    assert (os.path.exists("%s/mapped_network.xml.gz" % context.path()))
    assert (os.path.exists("%s/mapped_schedule.xml.gz" % context.path()))
    assert (os.path.exists("%s/road_network.xml.gz" % context.path()))
    assert (os.path.exists(context.stage("matsim.scenario.network.convert_pt_schedule")["vehicles"]))

    return {
        "network": "%s/mapped_network.xml.gz" % context.path(),
        "schedule": "%s/mapped_schedule.xml.gz" % context.path(),
        "road_network": "%s/road_network.xml.gz" % context.path(),
        "vehicles": context.stage("matsim.scenario.network.convert_pt_schedule")["vehicles"]
    }
