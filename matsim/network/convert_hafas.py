import subprocess as sp
import os.path

def configure(context, require):
    require.stage("matsim.java.pt2matsim")
    require.stage("utils.java")
    require.config("raw_data_path")

def execute(context):
    jar, tmp_path = context.stage("matsim.java.pt2matsim")
    java = context.stage("utils.java")

    # Create MATSim schedule

    java(jar, "org.matsim.pt2matsim.run.Hafas2TransitSchedule", [
        "%s/hafas" % context.config["raw_data_path"], "EPSG:2056",
        "%s/transit_schedule.xml.gz" % context.cache_path,
        "%s/transit_vehicles.xml.gz" % context.cache_path,
        context.config["hafas_date"]
    ], cwd = context.cache_path, vm_arguments = ["-Djava.io.tmpdir=%s" % tmp_path])

    assert(os.path.exists("%s/transit_schedule.xml.gz" % context.cache_path))
    assert(os.path.exists("%s/transit_vehicles.xml.gz" % context.cache_path))

    return {
        "schedule" : "%s/transit_schedule.xml.gz" % context.cache_path,
        "vehicles" : "%s/transit_vehicles.xml.gz" % context.cache_path
    }
