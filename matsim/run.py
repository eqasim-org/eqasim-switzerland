import shutil
import os.path

def configure(context, require):
    require.stage("matsim.secondary_locations")
    require.stage("matsim.households")
    require.stage("matsim.facilities")
    require.stage("matsim.network.mapped")
    require.stage("matsim.java.matsim")
    require.stage("matsim.java.baseline")
    require.stage("utils.java")

def execute(context):
    network_path = context.stage("matsim.network.mapped")["network"]
    shutil.copyfile(network_path, "%s/switzerland_network.xml.gz" % context.cache_path)

    transit_schedule_path = context.stage("matsim.network.mapped")["schedule"]
    shutil.copyfile(transit_schedule_path, "%s/switzerland_transit_schedule.xml.gz" % context.cache_path)

    transit_vehicles_path = context.stage("matsim.network.mapped")["vehicles"]
    shutil.copyfile(transit_vehicles_path, "%s/switzerland_transit_vehicles.xml.gz" % context.cache_path)

    households_path = context.stage("matsim.households")
    shutil.copyfile(households_path, "%s/switzerland_households.xml.gz" % context.cache_path)

    facilities_path = context.stage("matsim.facilities")
    shutil.copyfile(facilities_path, "%s/switzerland_facilities.xml.gz" % context.cache_path)

    this_path = os.path.dirname(os.path.abspath(__file__))
    shutil.copyfile("%s/config_template.xml" % this_path, "%s/switzerland_config.xml" % context.cache_path)

    java = context.stage("utils.java")
    input_population_path = context.stage("matsim.secondary_locations")

    #java(
    #    context.stage("matsim.java.baseline"), "ch.ethz.matsim.baseline_scenario.preparation.Downsample", [
    #        input_population_path, "0.01", "%s/unrouted_switzerland_population.xml.gz" % context.cache_path
    #    ], cwd = context.cache_path)

    java(
        context.stage("matsim.java.baseline"), "ch.ethz.matsim.baseline_scenario.preparation.Routing", [
            "%s/switzerland_config.xml" % context.cache_path,
            "%s/switzerland_network.xml.gz" % context.cache_path,
            input_population_path,
            "%s/switzerland_transit_schedule.xml.gz" % context.cache_path,
            "%s/switzerland_population.xml.gz" % context.cache_path
        ], cwd = context.cache_path)

    #java(
    #    context.stage("matsim.java.matsim"), "org.matsim.run.Controler",
    #    ["switzerland_config.xml"], cwd = context.cache_path)

    java(
        context.stage("matsim.java.baseline"), "ch.ethz.matsim.baseline_scenario.RunIleDeFranceScenario", [
            "%s/switzerland_config.xml" % context.cache_path
        ], cwd = context.cache_path)

    return context.cache_path
