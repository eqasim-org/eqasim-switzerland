import shutil
import os.path
import xml.etree.ElementTree as ET
from lxml import etree
import matsim.runtime.eqasim as eqasim
import gzip

def configure(context):
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.eqasim")
        
    context.stage("matsim.scenario.population")
    context.stage("matsim.scenario.households")
    context.stage("matsim.scenario.vehicles")

    context.stage("matsim.scenario.facilities")
    context.stage("matsim.scenario.network.mapped")
        
    context.config("input_downsampling")
    context.config("threads")
    context.config("random_seed")
    
    context.config("output_prefix", "switzerland_")
    context.config("useScheduleBasedTransport", default = True)


def execute(context):
    # Some files we just copy
    transit_vehicles_input_path = context.stage("matsim.scenario.network.mapped")["vehicles"]
    transit_vehicles_output_path = "%s/%stransit_vehicles.xml.gz" % (context.path(), context.config("output_prefix"))

    sample_size     = context.config("input_downsampling")
    flow_eff_factor = 1.0 / sample_size
    print(f"INFO setting flow efficiency factors to {round(flow_eff_factor,2)} for PT vehicles.")
    
    if context.config("input_downsampling") < 1.0 and not context.config("useScheduleBasedTransport"):
        # Register the namespace
        namespace = {'m': 'http://www.matsim.org/files/dtd'}

        # Read and parse the gzipped XML file
        with gzip.open(transit_vehicles_input_path, 'rt', encoding='utf-8') as f:
            tree = ET.parse(f)
            root = tree.getroot()

        # Find and update all <flowEfficiencyFactor> elements
        for fe in root.findall('.//m:flowEfficiencyFactor', namespace):
            fe.set('factor', str(flow_eff_factor))

        ET.register_namespace('', 'http://www.matsim.org/files/dtd')

        # Write the modified XML back to a gzipped file
        with gzip.open(transit_vehicles_output_path, 'wt', encoding='utf-8') as f:
            tree.write(f, encoding='unicode', xml_declaration=True)

    else:
        shutil.copyfile(transit_vehicles_input_path, transit_vehicles_output_path)

    vehicles_input_path = context.stage("matsim.scenario.vehicles")
    vehicles_output_path = "%s/%svehicles.xml.gz" % (context.path(), context.config("output_prefix"))
    shutil.copyfile(vehicles_input_path, vehicles_output_path)

    households_input_path = context.stage("matsim.scenario.households")
    households_output_path = "%s/%shouseholds.xml.gz" % (context.path(), context.config("output_prefix"))
    shutil.copyfile(households_input_path, households_output_path)
        
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
        "--eqasim-configurator", "org.eqasim.switzerland.ch.SwitzerlandConfigurator"
    ])
    
    assert os.path.exists("%s/%sconfig.xml" % (context.path(), context.config("output_prefix")))
    
    # Calculate the stop categories
    transit_schedule_input_path = context.stage("matsim.scenario.network.mapped")["schedule"]
    transit_schedule_output_path = "%stransit_schedule.xml.gz" % context.config("output_prefix")

    eqasim.run(context, "org.eqasim.switzerland.ch.scenario.RunCalculateStopCategories", [
        "--input-path", transit_schedule_input_path,
        "--output-path", transit_schedule_output_path
    ])
    
    assert os.path.exists("%s/%stransit_schedule.xml.gz" % (context.path(), context.config("output_prefix")))

    # Adapt the config
    eqasim.run(context, "org.eqasim.switzerland.ch.scenario.RunAdaptConfig", [
        "--input-path", config_path,
        "--output-path", config_path,
        "--downsamplingRate", context.config("input_downsampling"),
        "--replanningRate", "0.05",
        "--hasFreight", context.config("use_freight"),
        "--prefix", context.config("output_prefix")    ])
    
    assert os.path.exists("%s/%sconfig.xml" % (context.path(), context.config("output_prefix")))

    # If we want to simulate buses, some lines have to be added to the config.
    if not context.config("useScheduleBasedTransport"):
        config_path = f"{context.path()}/{context.config('output_prefix')}config.xml"
        assert os.path.exists(config_path)

        # Parse XML and preserve DOCTYPE and comments
        parser = etree.XMLParser(remove_blank_text=True)
        tree = etree.parse(config_path, parser)
        root = tree.getroot()

        # Create new module
        module = etree.Element("module", name="SBBPt")
        # everything else will be simulated
        etree.SubElement(module, "param", name="deterministicServiceModes",
                        value="rail,subway,ferry,tram,funicular,cable-car,gondola,other")
        # The deterministic PT engine (rail/tram/subway/ferry/...) only writes
        # linkEnter/linkLeave events when iterationNumber % interval == 0. The
        # webmap's pt_link_volumes are built from those events, so the LAST
        # iteration must satisfy that or those modes get no link volumes (only
        # bus, which runs on the QSim, does). value=1 => every iteration writes
        # them, so the output iteration always has rail/tram/etc. link volumes.
        etree.SubElement(module, "param", name="createLinkEventsInterval", value="1")

        # Append to root
        root.append(module)

        # Write back to file with DOCTYPE
        doctype_str = '<!DOCTYPE config SYSTEM "http://www.matsim.org/files/dtd/config_v2.dtd">'
        tree.write(config_path, pretty_print=True, xml_declaration=True, encoding="UTF-8", doctype=doctype_str)


    # Route the population
    population_output_path = "%s/%spopulation.xml.gz" % (context.path(), context.config("output_prefix"))

    # routing class does not work with turn restrictions; it seems something to do with multi-threading as
    # it works when we use 1 thread; this ius however not an issue as MATSim routes teverything in iteration 0
    eqasim.run(context, "org.eqasim.core.scenario.routing.RunPopulationRouting", [
        "--config-path", config_path,
        "--output-path", population_output_path,
        "--threads", context.config("threads"),
        "--config:plans.inputPlansFile", population_prepared_path,
        "--eqasim-configurator", "org.eqasim.switzerland.ch.SwitzerlandConfigurator"
    ])
    
    assert os.path.exists("%s/%spopulation.xml.gz" % (context.path(), context.config("output_prefix")))

    # Validate the scenario
    eqasim.run(context, "org.eqasim.core.scenario.validation.RunScenarioValidator", [
        "--config-path", config_path
    ])
    
    # Cleanup
    os.remove("%s/prepared_population.xml.gz" % context.path())

    return "%sconfig.xml" % context.config("output_prefix")
