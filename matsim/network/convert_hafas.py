import subprocess as sp
import os.path

def configure(context, require):
    require.stage("matsim.java.pt2matsim")
    require.config("raw_data_path")

def execute(context):
        jar = context.stage("matsim.java.pt2matsim")

        # Create MATSim schedule

        sp.check_call([
            "java", "-cp", jar, "org.matsim.pt2matsim.run.Hafas2TransitSchedule",
            "%s/hafas" % context.config["raw_data_path"], "EPSG:2056",
            "%s/transit_schedule.xml.gz" % context.cache_path,
            "%s/transit_vehicles.xml.gz" % context.cache_path
        ], cwd = context.cache_path)

        assert(os.path.exists("%s/transit_schedule.xml.gz" % context.cache_path))
        assert(os.path.exists("%s/transit_vehicles.xml.gz" % context.cache_path))

        return {
            "schedule" : "%s/transit_schedule.xml.gz" % context.cache_path,
            "vehicles" : "%s/transit_vehicles.xml.gz" % context.cache_path
        }
