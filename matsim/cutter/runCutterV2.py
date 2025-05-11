import os.path

import matsim.runtime.eqasim as eqasim

# this cutter can currently be used only if Volume Delay Function (VDF) is
# used to model traffic; it can be activated through the config file and has to be used
# for the entire Swiss scenario


def configure(context):
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.eqasim")
    context.stage("matsim.simulation.run")
    context.config("extent_path")
    context.config("shape_name")
    context.config("extent_prefix")
    context.config("use_vdf", default=False)

def execute(context):

    assert context.config("use_vdf"), "use of vdf needs to be enabled to use cutterV2"

    # get the output config from the ssimulation run in matsim.simulation.run stage
    config_path = "%s/%s" % (
        context.path("matsim.simulation.run"),
        "simulation_output/output_config.xml"
    )
    assert os.path.exists(config_path)
    
    # change the path of all input files to point to the output files
    # population, households, facitilies, network, transit_schedule, transit_vehicles, vehicles

    with open(config_path) as f_read:
        content = f_read.read()

        content = content.replace(
            'switzerland_population.xml.gz',
            '%s/%s/output_plans.xml.gz' % (context.path("matsim.simulation.run"), "simulation_output")
        )

        content = content.replace(
            'switzerland_households.xml.gz',
            '%s/%s/output_households.xml.gz' % (context.path("matsim.simulation.run"), "simulation_output")
        )

        content = content.replace(
            'switzerland_facilities.xml.gz',
            '%s/%s/output_facilities.xml.gz' % (context.path("matsim.simulation.run"), "simulation_output")
        )

        content = content.replace(
            'switzerland_network.xml.gz',
            '%s/%s/output_network.xml.gz' % (context.path("matsim.simulation.run"), "simulation_output")
        )

        content = content.replace(
            'switzerland_transit_schedule.xml.gz',
            '%s/%s/output_transitSchedule.xml.gz' % (context.path("matsim.simulation.run"), "simulation_output")
        )

        content = content.replace(
            'switzerland_transit_vehicles.xml.gz',
            '%s/%s/output_transitVehicles.xml.gz' % (context.path("matsim.simulation.run"), "simulation_output")
        )

        content = content.replace(
            'switzerland_vehicles.xml.gz',
            '%s/%s/output_vehicles.xml.gz' % (context.path("matsim.simulation.run"), "simulation_output")
        )
        


        with open("%s/config_cutter.xml" % context.path(), "w+") as f_write:
            f_write.write(content)

    
    # use the new config to run the cutter

    config_path = "%s/config_cutter.xml" % context.path()
    
    
    
    eqasim.run(context, "org.eqasim.core.scenario.cutter.RunScenarioCutterV2", [
        "--config-path", config_path,
        "--output-path", "%s/output" % context.path(),
        "--extent-path", "%s/%s" % (context.config("extent_path"), context.config("shape_name")),
        "--vdf-travel-times-path", "%s/%s" % (context.path("matsim.simulation.run"), "simulation_output/vdf.bin"),
        "--threads", context.config("threads"),
        "--prefix", context.config("extent_prefix")
    ])