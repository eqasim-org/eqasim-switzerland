import shutil
import os.path
import matsim.runtime.eqasim as eqasim
from dmc.constants import constants as dmc_constants

import matsim.simulation.config_utils as config_utils

def configure(context):
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.eqasim")
        
    context.stage("matsim.scenario.population")
    context.stage("matsim.scenario.households")
    context.stage("matsim.scenario.vehicles")

    context.stage("matsim.scenario.facilities")
    context.stage("matsim.scenario.network.mapped")

    context.stage("data.pt_pricing.pt_pricing")
    context.stage("calibration.pt_routing.pt_routing_parameters")
        
    context.stage("data.microcensus.shares")
    context.stage("dmc.params")
    context.stage("calibration.car_routing_vot.optimal_value")

    context.config("input_downsampling")
    context.config("threads")
    context.config("random_seed")
    
    context.config("output_prefix", "switzerland_")
    context.config("useScheduleBasedTransport", default = True)
    context.config("car_cost_model", dmc_constants.CAR_COST_MODEL)
    context.config("route_bike", True)


def execute(context):
    # Some files we just copy
    transit_vehicles_input_path  = context.stage("matsim.scenario.network.mapped")["vehicles"]
    transit_vehicles_output_path = "%s/%stransit_vehicles.xml.gz" % (context.path(), context.config("output_prefix"))

    sample_size     = context.config("input_downsampling")

    if context.config("input_downsampling") < 1.0 and not context.config("useScheduleBasedTransport"):
        config_utils.modify_PCEs(transit_vehicles_input_path, transit_vehicles_output_path, sample_size)

    else:
        shutil.copyfile(transit_vehicles_input_path, transit_vehicles_output_path)


    vehicles_input_path = context.stage("matsim.scenario.vehicles")
    vehicles_output_path = "%s/%svehicles.xml.gz" % (context.path(), context.config("output_prefix"))
    shutil.copyfile(vehicles_input_path, vehicles_output_path)

    households_input_path = context.stage("matsim.scenario.households")
    households_output_path = "%s/%shouseholds.xml.gz" % (context.path(), context.config("output_prefix"))
    shutil.copyfile(households_input_path, households_output_path)

    # PT pricing
    sbb_path, zones_path, pricing_path = context.stage("data.pt_pricing.pt_pricing")

    sbb_output_path =  f"{context.path()}/SBB_all_distances.csv" 
    shutil.copy(sbb_path, sbb_output_path)

    zones_output_path =  f"{context.path()}/gtfs_zones.csv" 
    shutil.copy(zones_path, zones_output_path)

    pricing_output_path =  f"{context.path()}/pricingDescription.xml" 
    shutil.copy(pricing_path, pricing_output_path)

    # copy the mode shares        
    global_shares_path, cantonal_shares_path = context.stage("data.microcensus.shares")
    shutil.copyfile(global_shares_path, 
                    "%s/%sglobal_mode_shares.csv" % (context.path(), context.config("output_prefix")))
    shutil.copyfile(cantonal_shares_path, 
                    "%s/%scantonal_mode_shares.csv" % (context.path(), context.config("output_prefix")))
    
    # copy the mode choice params
    mode_params_path, cost_params_path = context.stage("dmc.params")
    mode_parameters_path = "%s/dmc_parameters.yml" % context.path()
    cost_parameters_path = "%s/cost_parameters.yml" % context.path()
    shutil.copy(mode_params_path, mode_parameters_path)
    shutil.copy(cost_params_path, cost_parameters_path)

    # Some files we send through several preparation scripts
    
    # Run preparation
    facilities_input_path = context.stage("matsim.scenario.facilities")
    facilities_output_path = "%sfacilities.xml.gz" % context.config("output_prefix")
    
    population_input_path = context.stage("matsim.scenario.population")
    population_prepared_path = "prepared_population.xml.gz"
    
    network_input_path = context.stage("matsim.scenario.network.mapped")["network"]
    network_output_path = "%snetwork.xml.gz" % context.config("output_prefix")

    # Call the basic preparation script
    eqasim.run(context, "org.eqasim.core.scenario.preparation.RunPreparation", [
        "--input-facilities-path", facilities_input_path,
        "--output-facilities-path", facilities_output_path,
        "--input-population-path", population_input_path,
        "--output-population-path", population_prepared_path,
        "--input-network-path", network_input_path,
        "--output-network-path", network_output_path,
        "--threads", context.config("threads")
    ])
    
    assert os.path.exists("%s/%sfacilities.xml.gz" % (context.path(), context.config("output_prefix")))
    assert os.path.exists("%s/prepared_population.xml.gz" % context.path())
    assert os.path.exists("%s/%snetwork.xml.gz" % (context.path(), context.config("output_prefix")))

    # Generate the config file
    config_path = "%sconfig.xml" % context.config("output_prefix")
    eqasim.run(context, "org.eqasim.core.scenario.config.RunGenerateConfig", [
        "--output-path", config_path,
        "--prefix", context.config("output_prefix"),
        "--sample-size", context.config("input_downsampling"),
        "--random-seed", context.config("random_seed"),
        "--threads", context.config("threads"),
        "--eqasim-configurator", "org.eqasim.switzerland.ch_cmdp.SwitzerlandConfigurator"
    ])
    
    assert os.path.exists("%s/%sconfig.xml" % (context.path(), context.config("output_prefix")))
    
    # Calculate the stop categories
    transit_schedule_input_path = context.stage("matsim.scenario.network.mapped")["schedule"]
    transit_schedule_output_path = "%stransit_schedule.xml.gz" % context.config("output_prefix")

    eqasim.run(context, "org.eqasim.switzerland.ch.scenario.RunCalculateStopCategories", [
        "--input-path", transit_schedule_input_path,
        "--output-path", transit_schedule_output_path,
    ])
    
    assert os.path.exists("%s/%stransit_schedule.xml.gz" % (context.path(), context.config("output_prefix")))

    # Adapt the config
    eqasim.run(context, "org.eqasim.switzerland.ch_cmdp.scenario.RunAdaptConfig", [
    # eqasim.run(context, "org.eqasim.switzerland.ch.scenario.RunAdaptConfig", [
        "--input-path", config_path,
        "--output-path", config_path,
        "--downsamplingRate", context.config("input_downsampling"),
        "--replanningRate", "0.05",
        "--hasFreight", context.config("use_freight"),
        "--prefix", context.config("output_prefix"),
        "--carCostModel", context.config("car_cost_model").lower(),
        "--routingDistanceUtility", context.stage("calibration.car_routing_vot.optimal_value"),
        "--routeBikeInNetwork", str(context.config("route_bike")).lower()
        ])
    
    assert os.path.exists("%s/%sconfig.xml" % (context.path(), context.config("output_prefix")))

    # If we want to simulate buses, some lines have to be added to the config.
    if not context.config("useScheduleBasedTransport"):
        config_utils.add_SBBPT_module(context)

    # Create the module for PT pricing
    config_utils.add_ptZones_module(context)

    # Write PT routing parameters within the config
    pt_parameters = context.stage("calibration.pt_routing.pt_routing_parameters")
    config_utils.adjust_pt_routing_parameters(context, pt_parameters)

    # Route the population
    population_output_path = "%s/%spopulation.xml.gz" % (context.path(), context.config("output_prefix"))

    # routing class does not work with turn restrictions; it seems something to do with multi-threading as
    # it works when we use 1 thread; this ius however not an issue as MATSim routes teverything in iteration 0
    eqasim.run(context, "org.eqasim.core.scenario.routing.RunPopulationRouting", [
        "--config-path", config_path,
        "--output-path", population_output_path,
        "--threads", context.config("threads"),
        "--config:plans.inputPlansFile", population_prepared_path,
        "--eqasim-configurator", "org.eqasim.switzerland.ch_cmdp.SwitzerlandConfigurator"
    ])
    
    assert os.path.exists("%s/%spopulation.xml.gz" % (context.path(), context.config("output_prefix")))

    # Validate the scenario
    eqasim.run(context, "org.eqasim.core.scenario.validation.RunScenarioValidator", [
        "--config-path", config_path
    ])
    
    # Cleanup
    os.remove("%s/prepared_population.xml.gz" % context.path())

    return "%sconfig.xml" % context.config("output_prefix")
