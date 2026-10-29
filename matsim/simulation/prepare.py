import shutil
import os.path
import matsim.runtime.eqasim as eqasim
from matsim.config import config as shared_config
from matsim.simulation.cross_border_links import patch_directional_border_links

import matsim.config.tools as config_tools
from mode_choice.routing import options as routing_options, exports as routing_exports

def configure(context):
    context.stage("matsim.config.config")
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.eqasim")
        
    context.stage("matsim.scenario.population")
    context.stage("matsim.scenario.households")
    context.stage("matsim.scenario.vehicles")

    context.stage("matsim.scenario.facilities")
    context.stage("matsim.scenario.network.pt")

    context.stage("data.pt_pricing.pt_pricing")
    context.stage("calibration.pt_routing.pt_routing_parameters")
        
    context.stage("data.microcensus.shares")
    context.stage("dmc.parameter_files")

    context.config("input_downsampling")
    context.config("threads", volatile=True)
    context.config("java_memory", "50G", volatile=True)
    context.config("random_seed")
    context.config("use_vdf", False)
    context.config("export_detailed_network", True)
    context.config("last_iteration", 140)
    context.config("calibrate_alphas_in_matsim", True)
    context.config("alphaCalibration.level", 'canton')
    context.config("calibrate_betas_in_matsim", False)
    context.config("activate_traffic_light_delays", True)
    context.config("activate_unsignalized_intersections_delays", True)
    context.config("correct_links_capacity", True)
    context.config("minimum_speed", 1.0)
    context.config("preventwaitingtoentertraffic", False)
    context.config("writeexperiencedplans", False)
    
    context.config("output_prefix", "switzerland_")
    context.config("useScheduleBasedTransport", default = False)
    context.config("route_bike", True)
    context.config("enable_mode_choice", False)
    if context.config("enable_mode_choice"):
        routing_options.configure(context)
        if context.config("route_population_once"):
            routing_exports.configure(context)
    context.config("include_cross_border", default=False)
    context.stage("data.cross_border.swiss_residents_od")
    context.stage("data.spatial.swiss_border")

    if context.config("include_cross_border"):
        context.stage("data.cross_border.destinations")

    # network calibration
    context.config("network_calibration.activate", default=True)
    context.config("network_calibration.calibrate_disutilities", default=True)
    context.config("network_calibration.calibrate_freespeed", default=True)
    context.config("network_calibration.calibrate_agents_ascs", default=True)
    context.config("network_calibration.calibrate_subpopulations", default=True)

    need_counts = config_tools.need_counts_file(context)
    if context.config("network_calibration.activate") and need_counts:
        context.stage("analysis.counts.target")
        if context.config("network_calibration.calibrate_disutilities"):
            context.stage("calibration.road_regions.penalty_calibration")
        
    
    if context.config("network_calibration.activate") and context.config("network_calibration.calibrate_freespeed"):
        context.stage("analysis.travel_times.APIs.target")
        context.stage("calibration.road_regions.freespeed_calibration")


def execute(context):

    # Some files we just copy

    transit_vehicles_input_path  = context.stage("matsim.scenario.network.pt")["vehicles"]
    transit_vehicles_output_path = "%s/%stransit_vehicles.xml.gz" % (context.path(), context.config("output_prefix"))

    sample_size     = context.config("input_downsampling")

    if context.config("input_downsampling") < 1.0 and not context.config("useScheduleBasedTransport"):
        config_tools.modify_PCEs(transit_vehicles_input_path, transit_vehicles_output_path, sample_size)

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

    mode_params_path, cost_params_path = context.stage("dmc.parameter_files")
    mode_parameters_path = "%s/dmc_parameters.yml" % context.path()
    cost_parameters_path = "%s/cost_parameters.yml" % context.path()
    shutil.copy(mode_params_path, mode_parameters_path)
    shutil.copy(cost_params_path, cost_parameters_path)

    if context.config("export_detailed_network"):
        shutil.copyfile(
            context.stage("matsim.scenario.network.pt")["geometry"],
            "%s/%sdetailed_network.csv" % (context.path(), context.config("output_prefix")),
        )

    # Prepare facilities, population and network.

    facilities_input_path = context.stage("matsim.scenario.facilities")
    facilities_output_path = "%sfacilities.xml.gz" % context.config("output_prefix")
    
    population_input_path = context.stage("matsim.scenario.population")
    population_prepared_path = "prepared_population.xml.gz"
    
    network_input_path = context.stage("matsim.scenario.network.pt")["network"]
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

    # Java assigns facility links by nearest coordinate only. For directional
    # cross-border facilities we patch the prepared files in Python so entry and
    # exit activities use opposite one-way links before the population is routed.

    cross_border_destinations = None
    if context.config("include_cross_border"):
        cross_border_destinations = context.stage("data.cross_border.destinations")

    patch_directional_border_links(
        cross_border_destinations,
        context.stage("data.cross_border.swiss_residents_od"),
        context.stage("data.spatial.swiss_border"),
        "%s/%snetwork.xml.gz" % (context.path(), context.config("output_prefix")),
        "%s/%sfacilities.xml.gz" % (context.path(), context.config("output_prefix")),
        "%s/prepared_population.xml.gz" % context.path(),
        max_workers=context.config("threads"),
    )

    # Copy and modify the shared config.

    config_path = "%sconfig.xml" % context.config("output_prefix")
    local_config_path = os.path.join(context.path(), config_path)
    config_tools.copy_config(context.stage("matsim.config.config"), local_config_path)

    shared_config.adapt_for_routing(context, local_config_path, local=True)

    # Calculate the stop categories

    transit_schedule_input_path = context.stage("matsim.scenario.network.pt")["schedule"]
    transit_schedule_output_path = "%stransit_schedule.xml.gz" % context.config("output_prefix")

    eqasim.run(context, "org.eqasim.switzerland.ch.scenario.RunCalculateStopCategories", [
        "--input-path", transit_schedule_input_path,
        "--output-path", transit_schedule_output_path,
    ])
    
    assert os.path.exists("%s/%stransit_schedule.xml.gz" % (context.path(), context.config("output_prefix")))

    # Route the population

    population_output_path = "%s/%spopulation.xml.gz" % (context.path(), context.config("output_prefix"))

    routing_arguments = [
        "--config-path", config_path,
        "--output-path", population_output_path,
        "--threads", context.config("threads"),
        "--config:plans.inputPlansFile", population_prepared_path,
        "--eqasim-configurator", "org.eqasim.switzerland.ch_cmdp.SwitzerlandConfigurator"
    ]
    if context.config("enable_mode_choice") and context.config("route_population_once"):
        routing_arguments.extend(["--routes-path", routing_exports.population_routes(context)])
    eqasim.run(context, "org.eqasim.core.scenario.routing.RunPopulationRouting", routing_arguments)
    
    assert os.path.exists("%s/%spopulation.xml.gz" % (context.path(), context.config("output_prefix")))

    # Validate the scenario

    eqasim.run(context, "org.eqasim.core.scenario.validation.RunScenarioValidator", [
        "--config-path", config_path
    ])
    
    # Finish the portable simulation config after routing and validation.

    config_path = shared_config.adapt_for_simulation(context, local_config_path, local=True)

    # Cleanup

    os.remove("%s/prepared_population.xml.gz" % context.path())

    return config_path
