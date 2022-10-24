import os.path

import matsim.runtime.pt2matsim as pt2matsim


def configure(context):
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.pt2matsim")
    
    context.config("data_path")
    context.config("hafas_date")


def execute(context):
    hafas_path = "%s/hafas" % context.config("data_path")

    # Create MATSim schedule
    pt2matsim.run(context, "org.matsim.pt2matsim.run.Hafas2TransitSchedule", [
        hafas_path, "epsg:2056",
        "%s/transit_schedule.xml.gz" % context.path(),
        "%s/transit_vehicles.xml.gz" % context.path(),
        context.config("hafas_date")
    ])

    assert (os.path.exists("%s/transit_schedule.xml.gz" % context.path()))
    assert (os.path.exists("%s/transit_vehicles.xml.gz" % context.path()))

    return dict(
        schedule = "%s/transit_schedule.xml.gz" % context.path(),
        vehicles = "%s/transit_vehicles.xml.gz" % context.path()
    )
