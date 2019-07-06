import shutil
import os.path

def configure(context, require):
    require.stage("matsim.secondary_locations")
    require.stage("matsim.households")
    require.stage("matsim.facilities")
    require.stage("matsim.network.mapped")
    require.stage("matsim.java.eqasim")
    require.stage("utils.java")

def execute(context):
    # Some files we just copy
    transit_schedule_path = context.stage("matsim.network.mapped")["schedule"]
    shutil.copyfile(transit_schedule_path, "%s/switzerland_transit_schedule.xml.gz" % context.cache_path)

    transit_vehicles_path = context.stage("matsim.network.mapped")["vehicles"]
    shutil.copyfile(transit_vehicles_path, "%s/switzerland_transit_vehicles.xml.gz" % context.cache_path)

    households_path = context.stage("matsim.households")
    shutil.copyfile(households_path, "%s/switzerland_households.xml.gz" % context.cache_path)

    # Some files we send through the preparation script
    network_input_path = context.stage("matsim.network.mapped")["network"]
    network_output_path = "%s/switzerland_network.xml.gz" % context.cache_path

    facilities_input_path = context.stage("matsim.facilities")
    facilities_output_path = "%s/switzerland_facilities.xml.gz" % context.cache_path

    population_input_path = context.stage("matsim.secondary_locations")
    population_prepared_path = "%s/prepared_population.xml.gz" % context.cache_path
    population_output_path = "%s/switzerland_population.xml.gz" % context.cache_path

    config_output_path = "%s/switzerland_config.xml" % context.cache_path

    # Call preparation script
    java = context.stage("utils.java")

    java(
        context.stage("matsim.java.eqasim"),
        "org.eqasim.scenario.preparation.RunPreparation", [
        "--input-facilities-path", facilities_input_path,
        "--output-facilities-path", facilities_output_path,
        "--input-population-path", population_input_path,
        "--output-population-path", population_prepared_path,
        "--input-network-path", network_input_path,
        "--output-network-path", network_output_path,
        "--threads", str(context.config["threads"])
    ], cwd = context.cache_path)

    java(
        context.stage("matsim.java.eqasim"),
        "org.eqasim.scenario.preparation.CreateDefaultConfig", [
        "--output-path", config_output_path,
        "--prefix", "switzerland_",
        "--sample-size", str(context.config["input_downsampling"]),
        "--config:global.randomSeed", str(1000),
        "--config:global.numberOfThreads", str(context.config["threads"]),
        "--config:qsim.numberOfThreads", str(min(12, context.config["threads"]))
    ], cwd = context.cache_path)

    java(
        context.stage("matsim.java.eqasim"),
        "org.eqasim.scenario.routing.RunPopulationRouting", [
        "--config-path", config_output_path,
        "--output-path", population_output_path,
        "--threads", str(context.config["threads"]),
        "--config:plans.inputPlansFile", population_prepared_path
    ], cwd = context.cache_path)

    java(
        context.stage("matsim.java.eqasim"),
        "org.eqasim.scenario.validation.RunScenarioValidator", [
        "--config-path", config_output_path
    ], cwd = context.cache_path)

    java(
        context.stage("matsim.java.eqasim"),
        "org.eqasim.simulation.RunSwitzerlandSimulation", [
        "--config-path", config_output_path,
        "--config:controler.lastIteration", str(1),
        "--config:controler.writeEventsInterval", str(1),
        "--config:controler.writePlansInterval", str(1),
    ], cwd = context.cache_path)

    return context.cache_path
