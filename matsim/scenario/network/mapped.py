import os
from functools import partial
from matsim.config import tools as config_tools
import matsim.runtime.pt2matsim as pt2matsim


def configure(context):
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.pt2matsim")

    context.stage("matsim.scenario.network.convert_osm")
    context.stage("matsim.scenario.network.convert_pt_schedule")

    context.config("threads", volatile=True)
    context.config("route_bike", True)
    context.config("export_detailed_network", True)
    context.config("java_memory", "50G", volatile=True)

def execute(context):
    unmapped_network_path = context.stage("matsim.scenario.network.convert_osm")
    unmapped_schedule_path = context.stage("matsim.scenario.network.convert_pt_schedule")["schedule"]

    # Create and modify config file
    pt2matsim.run(context, "org.matsim.pt2matsim.run.CreateDefaultPTMapperConfig", [
        "map_network_template.xml"
    ], [])

    config_path = write_mapping_config(context, unmapped_network_path, unmapped_schedule_path)

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
        config_path
    ], vm_arguments=["-Dlog4j.configurationFile=file:log4j.xml"])

    assert (os.path.exists("%s/mapped_network.xml.gz" % context.path()))
    assert (os.path.exists("%s/mapped_schedule.xml.gz" % context.path()))
    assert (os.path.exists("%s/road_network.xml.gz" % context.path()))
    if context.config("export_detailed_network"):
        assert os.path.exists(os.path.join(context.path(), "detailed_network.csv"))
    assert (os.path.exists(context.stage("matsim.scenario.network.convert_pt_schedule")["vehicles"]))

    return {
        "network": os.path.join(context.path(), "mapped_network.xml.gz"),
        "schedule": os.path.join(context.path(), "mapped_schedule.xml.gz"),
        "road_network": os.path.join(context.path(), "road_network.xml.gz"),
        "vehicles": context.stage("matsim.scenario.network.convert_pt_schedule")["vehicles"],
        "geometry": os.path.join(context.path(), "detailed_network.csv")
    }


####################### functions ######################

def write_mapping_config(context, network_path, schedule_path):
    """Set mapper parameters by name, independent of template defaults/formatting."""
    tree = config_tools.read_config(os.path.join(context.path(), "map_network_template.xml"))
    modules = tree.getroot().findall('module[@name="PublicTransitMapping"]')
    if len(modules) != 1:
        raise RuntimeError("Expected exactly one PublicTransitMapping module in template")
    module = modules[0]

    set_param = partial(config_tools.set_param, parent=module)

    set_param("inputNetworkFile", network_path)
    set_param("inputScheduleFile", schedule_path)
    set_param("numOfThreads", context.config("threads"))
    set_param("outputNetworkFile", os.path.join(context.path(), "mapped_network.xml.gz"))
    set_param("outputScheduleFile", os.path.join(context.path(), "mapped_schedule.xml.gz"))
    set_param("outputStreetNetworkFile", os.path.join(context.path(), "road_network.xml.gz"))
    modes_to_keep = "car,car_passenger,truck,taxi"
    if context.config("route_bike"):
        modes_to_keep += ",bike"
    set_param("modesToKeepOnCleanUp", modes_to_keep)
    set_param("modeSpecificRules", True)
    set_param("maxTravelCostFactor", 10)
    set_param("splitLinksAtStops", True)
    set_param("stopLinkSplittingModes", "bus")
    if context.config("export_detailed_network"):
        geometry_path = "%s/detailed_network.csv" % context.path("matsim.scenario.network.convert_osm")
        if not os.path.isfile(geometry_path):
            raise FileNotFoundError(f"Processed network geometry not found: {geometry_path}. "
                                    "Rebuild matsim.scenario.network.convert_osm with detailed export enabled.")
        set_param("inputNetworkGeometryFile", geometry_path)
        set_param("outputNetworkGeometryFile", os.path.join(context.path(), "detailed_network.csv"))
    else:
        set_param("inputNetworkGeometryFile", "")
        set_param("outputNetworkGeometryFile", "")

    assignment = config_tools.get_parameter_set(module, "transportModeAssignment", {"scheduleMode": "bus"})
    set_param("networkModes", "bus", parent=assignment)
    for mode, network_modes, distance in (
        ("rail", "rail,light_rail,train", 120.0),
        ("tram", "tram", 180.0),
        ("subway", "light_rail,subway", 180.0),
        ("funicular", "funicular", 120.0),
    ):
        assignment = config_tools.get_parameter_set(module, "transportModeAssignment", {"scheduleMode": mode})
        set_param("maxLinkCandidateDistance", distance, parent=assignment)
        set_param("nLinkThreshold", 1, parent=assignment)
        set_param("networkModes", network_modes, parent=assignment)
        set_param("strictLinkRule", True, parent=assignment)

    config_path = os.path.join(context.path(), "map_network.xml")
    config_tools.write_config(tree, config_path)

    return config_path
