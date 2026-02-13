import shutil
import matsim.runtime.eqasim as eqasim
import xml.etree.ElementTree as ET
import gzip
import os
import matsim.simulation.config_utils as config_utils


def configure(context):
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.eqasim")

    context.stage("matsim.scenario.network.mapped")
    context.stage("data.pt_pricing.pt_pricing")
    
    context.config("input_downsampling")
    context.config("output_prefix", "switzerland_")
    context.config("random_seed")
    context.config("threads")
    context.config("useScheduleBasedTransport")


def execute(context):
    transit_vehicles_input_path = context.stage("matsim.scenario.network.mapped")["vehicles"]
    transit_vehicles_output_path = "%s/%stransit_vehicles.xml.gz" % (context.path(), context.config("output_prefix"))
    sample_size     = context.config("input_downsampling")
    
    if context.config("input_downsampling") < 1.0 and not context.config("useScheduleBasedTransport"):

        pce =  sample_size
        print(f"INFO setting PCE to {round(pce,2)} for PT vehicles.")

        # Register the namespace
        namespace = {'m': 'http://www.matsim.org/files/dtd'}

        # Read and parse the gzipped XML file
        with gzip.open(transit_vehicles_input_path, 'rt', encoding='utf-8') as f:
            tree = ET.parse(f)
            root = tree.getroot()

        # Find and update all <flowEfficiencyFactor> elements
        for fe in root.findall('.//m:passengerCarEquivalents', namespace):
            fe.set('pce', str(pce))

        ET.register_namespace('', 'http://www.matsim.org/files/dtd')

        # Write the modified XML back to a gzipped file
        with gzip.open(transit_vehicles_output_path, 'wt', encoding='utf-8') as f:
            tree.write(f, encoding='unicode', xml_declaration=True)

    else:
        shutil.copyfile(transit_vehicles_input_path, transit_vehicles_output_path)

    # PT pricing
    sbb_path, zones_path, pricing_path = context.stage("data.pt_pricing.pt_pricing")

    sbb_output_path =  f"{context.path()}/SBB_all_distances.csv" 
    shutil.copy(sbb_path, sbb_output_path)

    zones_output_path =  f"{context.path()}/gtfs_zones.csv" 
    shutil.copy(zones_path, zones_output_path)

    pricing_output_path =  f"{context.path()}/pricingDescription.xml" 
    shutil.copy(pricing_path, pricing_output_path)

    network_input_path = context.stage("matsim.scenario.network.mapped")["network"]
    network_output_path = "%s/%snetwork.xml.gz" % (context.path(), context.config("output_prefix"))

    shutil.copyfile(network_input_path, network_output_path)

    config_path = "%sconfig.xml" % context.config("output_prefix")
    eqasim.run(context, "org.eqasim.core.scenario.config.RunGenerateConfig", [
        "--output-path", config_path,
        "--prefix", context.config("output_prefix"),
        "--sample-size", context.config("input_downsampling"),
        "--random-seed", context.config("random_seed"),
        "--threads", context.config("threads"),
        "--eqasim-configurator", "org.eqasim.switzerland.ch.SwitzerlandConfigurator"
    ])

    assert os.path.exists("%s/%sconfig.xml" % (context.path(), context.config("output_prefix")))

    transit_schedule_input_path = context.stage("matsim.scenario.network.mapped")["schedule"]
    transit_schedule_output_path = "%stransit_schedule.xml.gz" % (context.config("output_prefix"))

    eqasim.run(context, "org.eqasim.switzerland.ch.scenario.RunCalculateStopCategories", [
        "--input-path", transit_schedule_input_path,
        "--output-path", transit_schedule_output_path,
    ])

    assert os.path.exists("%s/%stransit_schedule.xml.gz" % (context.path(), context.config("output_prefix")))

    eqasim.run(context, "org.eqasim.switzerland.ch.scenario.RunAdaptConfig", [
        "--input-path", config_path,
        "--output-path", config_path,
        "--downsamplingRate", context.config("input_downsampling"),
        "--replanningRate", "0.05",
        "--hasFreight", False,
        "--prefix", context.config("output_prefix")    ])

    assert os.path.exists("%s/%sconfig.xml" % (context.path(), context.config("output_prefix")))

    config_utils.add_ptZones_module(context)
    
    if not context.config("useScheduleBasedTransport"):
        config_utils.add_SBBPT_module(context)

    config_utils.change_param(context, "facilities", "inputFacilitiesFile", "null")
    config_utils.change_param(context, "households", "inputFile", "null")
    config_utils.change_param(context, "plans", "inputPlansFile", "null")
    config_utils.change_param(context, "DiscreteModeChoice", "modeAvailability", "SwissModeAvailability")
    config_utils.change_param(context, "vehicles", "vehiclesFile", "null")

    return "%s/%sconfig.xml" % (context.path(), context.config("output_prefix"))



