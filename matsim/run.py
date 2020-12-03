import shutil


def configure(context):
    context.stage("matsim.population")
    context.stage("matsim.households")
    context.stage("matsim.facilities")
    context.stage("matsim.network.mapped")
    context.stage("matsim.java.eqasim")
    context.stage("utils.java")


def execute(context):
    # Some files we just copy
    transit_vehicles_path = context.stage("matsim.network.mapped")["vehicles"]
    shutil.copyfile(transit_vehicles_path, "%s/switzerland_transit_vehicles.xml.gz" % context.cache_path)

    households_path = context.stage("matsim.households")
    shutil.copyfile(households_path, "%s/switzerland_households.xml.gz" % context.cache_path)

    # Some files we send through the preparation script
    transit_schedule_input_path = context.stage("matsim.network.mapped")["schedule"]
    transit_schedule_output_path = "%s/switzerland_transit_schedule.xml.gz" % context.cache_path

    network_input_path = context.stage("matsim.network.mapped")["network"]
    network_output_path = "%s/switzerland_network.xml.gz" % context.cache_path

    facilities_input_path = context.stage("matsim.facilities")
    facilities_output_path = "%s/switzerland_facilities.xml.gz" % context.cache_path

    population_input_path = context.stage("matsim.population")
    population_prepared_path = "%s/prepared_population.xml.gz" % context.cache_path
    population_output_path = "%s/switzerland_population.xml.gz" % context.cache_path

    config_output_path = "%s/switzerland_config.xml" % context.cache_path

    # Call preparation script
    java = context.stage("utils.java")

    java(
        context.stage("matsim.java.eqasim"),
        "org.eqasim.core.scenario.preparation.RunPreparation", [
            "--input-facilities-path", facilities_input_path,
            "--output-facilities-path", facilities_output_path,
            "--input-population-path", population_input_path,
            "--output-population-path", population_prepared_path,
            "--input-network-path", network_input_path,
            "--output-network-path", network_output_path,
            "--threads", str(context.config("threads"))
        ], cwd=context.cache_path)

    java(
        context.stage("matsim.java.eqasim"),
        "org.eqasim.core.scenario.config.RunGenerateConfig", [
            "--output-path", config_output_path,
            "--prefix", "switzerland_",
            "--sample-size", str(context.config("input_downsampling")),
            "--random-seed", str(1000),
            "--threads", str(context.config("threads"))
        ], cwd=context.cache_path)

    java(
        context.stage("matsim.java.eqasim"),
        "org.eqasim.switzerland.scenario.RunCalculateStopCategories", [
            "--input-path", transit_schedule_input_path,
            "--output-path", transit_schedule_output_path
        ], cwd=context.cache_path)

    java(
        context.stage("matsim.java.eqasim"),
        "org.eqasim.core.scenario.routing.RunPopulationRouting", [
            "--config-path", config_output_path,
            "--output-path", population_output_path,
            "--threads", str(context.config("threads")),
            "--config:plans.inputPlansFile", population_prepared_path
        ], cwd=context.cache_path)

    java(
        context.stage("matsim.java.eqasim"),
        "org.eqasim.core.scenario.validation.RunScenarioValidator", [
            "--config-path", config_output_path
        ], cwd=context.cache_path)

    java(
        context.stage("matsim.java.eqasim"),
        "org.eqasim.switzerland.scenario.RunAdaptConfig", [
            "--input-path", config_output_path,
            "--output-path", config_output_path
        ], cwd=context.cache_path)

    java(
        context.stage("matsim.java.eqasim"),
        "org.eqasim.switzerland.RunSimulation", [
            "--config-path", config_output_path,
            "--config:controler.lastIteration", str(1),
            "--config:controler.writeEventsInterval", str(1),
            "--config:controler.writePlansInterval", str(1),
        ], cwd=context.cache_path)

    return context.cache_path
