import shutil
import os.path

import matsim.runtime.eqasim as eqasim


def configure(context):
    context.stage("matsim.simulation.run")
    context.config("output_path")
    context.stage("matsim.runtime.eqasim")
    context.stage("matsim.runtime.java")

def execute(context):
    # path to Switzerland config path
    config_path = "%s/%s" % (
        context.path("matsim.simulation.run"),
        "simulation_output/output_config.xml"
    )

    # path to output folder
    output_folder = context.config("output_path")

    # path to input plans
    plans_path = "%s/%s" % (
        context.path("matsim.simulation.run"),
        "simulation_output/output_plans.xml.gz"
    )

    network_path = "%s/%s" % (
        context.path("matsim.simulation.run"),
        "simulation_output/output_network.xml.gz"
    )

    facilities_path = "%s/%s" % (
        context.path("matsim.simulation.run"),
        "simulation_output/output_facilities.xml.gz"
    )

    households_path = "%s/%s" % (
        context.path("matsim.simulation.run"),
        "simulation_output/output_households.xml.gz"
    )

    transit_schedule_path= "%s/%s" % (
        context.path("matsim.simulation.run"),
        "simulation_output/output_transitSchedule.xml.gz"
    )

    transit_vehicle_path= "%s/%s" % (
        context.path("matsim.simulation.run"),
        "simulation_output/output_transitVehicles.xml.gz"
    )

    # path to shp
    shp_path = "/nas/asallard/Switzerland/SNN_shapefile_zurich/zurich_5km.shp"

    # Cut scenario
    eqasim.run(context, "org.eqasim.core.scenario.cutter.RunScenarioCutter", [
        "--config-path", config_path,
        "--output-path", output_folder,
        "--extent-path", shp_path,
        "--config:plans.inputPlansFile", plans_path,
        "--config:network.inputNetworkFile", network_path,
        "--config:facilities.inputFacilitiesFile", facilities_path,
        "--config:households.inputFile", households_path,
        "--config:transit.transitScheduleFile", transit_schedule_path,
        "--config:transit.vehiclesFile", transit_vehicle_path,
        "--prefix", "Zurich5km",
        "--threads", str(24)
    ])


    #zurich_config_path = output_folder + "/Zurich5kmconfig.xml"

    #eqasim.run(context, "org.eqasim.switzerland.RunSimulation", [
    #    "--config-path", zurich_config_path,
    #    "--config:controler.lastIteration", str(60),
    #    "--config:controler.writeEventsInterval", str(10),
    #    "--config:controler.writePlansInterval", str(10),
    #])