import shutil
import os.path

import matsim.runtime.eqasim as eqasim


def configure(context):
    context.stage("matsim.scenario.population")
    context.stage("matsim.scenario.households")

    context.stage("matsim.scenario.facilities")
    context.stage("matsim.scenario.supply.processed")
    context.stage("matsim.scenario.supply.hafas")

    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.eqasim")

    #context.stage("matsim.simulation.activity_list")

    context.config("input_downsampling")
    context.config("threads")
    context.config("random_seed")

    context.config("output_prefix", "switzerland_")
    #context.config("use_detailed_activities")


def execute(context):
    # Prepare input files
    facilities_path = "%s/%s" % (
        context.path("matsim.scenario.facilities"),
        context.stage("matsim.scenario.facilities")
    )

    population_path = "%s/%s" % (
        context.path("matsim.scenario.population"),
        context.stage("matsim.scenario.population")
    )

    network_path = "%s/%s" % (
        context.path("matsim.scenario.supply.processed"),
        context.stage("matsim.scenario.supply.processed")["network_path"]
    )

    eqasim.run(context, "org.eqasim.core.scenario.preparation.RunPreparation", [
        "--input-facilities-path", facilities_path,
        "--output-facilities-path", "%sfacilities.xml.gz" % context.config("output_prefix"),
        "--input-population-path", population_path,
        "--output-population-path", "prepared_population.xml.gz",
        "--input-network-path", network_path,
        "--output-network-path", "%snetwork.xml.gz" % context.config("output_prefix"),
        "--threads", context.config("threads")
    ])

    assert os.path.exists("%s/%sfacilities.xml.gz" % (context.path(), context.config("output_prefix")))
    assert os.path.exists("%s/prepared_population.xml.gz" % context.path())
    assert os.path.exists("%s/%snetwork.xml.gz" % (context.path(), context.config("output_prefix")))
    # Copy remaining input files
    households_path = "%s/%s" % (
        context.path("matsim.scenario.households"),
        context.stage("matsim.scenario.households")
    )
    shutil.copy(households_path, "%s/%shouseholds.xml.gz" % (context.cache_path, context.config("output_prefix")))

    transit_schedule_path = "%s/%s" % (
        context.path("matsim.scenario.supply.processed"),
        context.stage("matsim.scenario.supply.processed")["schedule_path"]
    )
    shutil.copy(transit_schedule_path,
                "%s/%stransit_schedule.xml.gz" % (context.cache_path, context.config("output_prefix")))

    transit_vehicles_path = "%s/%s" % (
        context.path("matsim.scenario.supply.hafas"),
        context.stage("matsim.scenario.supply.hafas")["vehicles_path"]
    )
    shutil.copy(transit_vehicles_path,
                "%s/%stransit_vehicles.xml.gz" % (context.cache_path, context.config("output_prefix")))

    # Generate base configuration
    eqasim.run(context, "org.eqasim.core.scenario.config.RunGenerateConfig", [
        "--sample-size", context.config("input_downsampling"),
        "--threads", context.config("threads"),
        "--prefix", context.config("output_prefix"),
        "--random-seed", context.config("random_seed"),
        "--output-path", "generic_config.xml"
    ])
    assert os.path.exists("%s/generic_config.xml" % context.path())

    # Adapt config for switzerland
    eqasim.run(context, "org.eqasim.switzerland.scenario.RunCalculateStopCategories", [
        "--input-path", transit_schedule_path,
        "--output-path", "%stransit_schedule.xml" % context.config("output_prefix")
    ])
    assert os.path.exists("%s/%stransit_schedule.xml" % (context.path(), context.config("output_prefix")))

    # to enable detailed activities
    #activities = context.stage("matsim.simulation.activity_list")

    #if (context.config("use_detailed_activities")):
    #    eqasim.run(context, "org.eqasim.switzerland.scenario.RunAdaptConfig", [
    #        "--input-path", "generic_config.xml",
    #        "--output-path", "%sconfig.xml" % context.config("output_prefix"),
    #        "--activity-list", activities
    #    ])
    if True:
        eqasim.run(context, "org.eqasim.switzerland.scenario.RunAdaptConfig", [
            "--input-path", "generic_config.xml",
            "--output-path", "%sconfig.xml" % context.config("output_prefix")
        ])
    assert os.path.exists("%s/%sconfig.xml" % (context.path(), context.config("output_prefix")))

    # Route population
    eqasim.run(context, "org.eqasim.core.scenario.routing.RunPopulationRouting", [
        "--config-path", "%sconfig.xml" % context.config("output_prefix"),
        "--output-path", "%spopulation.xml.gz" % context.config("output_prefix"),
        "--threads", context.config("threads"),
        "--config:plans.inputPlansFile", "prepared_population.xml.gz"
    ])
    assert os.path.exists("%s/%spopulation.xml.gz" % (context.path(), context.config("output_prefix")))

    # Validate scenario
    eqasim.run(context, "org.eqasim.core.scenario.validation.RunScenarioValidator", [
        "--config-path", "%sconfig.xml" % context.config("output_prefix")
    ])

    # Cleanup
    os.remove("%s/prepared_population.xml.gz" % context.path())

    return "%sconfig.xml" % context.config("output_prefix")
