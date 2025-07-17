import os.path

import matsim.runtime.eqasim as eqasim


def configure(context):
    context.stage("matsim.simulation.prepare")    
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.eqasim")
    
    context.config("use_vdf", default=False)
    context.config("threads")
    context.config("last_iteration", 60) 

    context.config("useScheduleBasedTransport", default=True)
    context.config("preventwaitingtoentertraffic", default = "no")
    context.config("writeexperiencedplans", default = "no")
    context.config("preventwaitingtoentertraffic", default = "no")
    context.config("writeexperiencedplans", default = "no")


def execute(context):
    config_path = "%s/%s" % (
        context.path("matsim.simulation.prepare"),
        context.stage("matsim.simulation.prepare")
    )
    
    if context.config("useScheduleBasedTransport"):
        scheduleBasedPTconfig = "true"
    else:
        scheduleBasedPTconfig = "false"

    preventwaitingtoentertraffic = "n"
    if context.config("preventwaitingtoentertraffic"):
        preventwaitingtoentertraffic = "y"
        print("Prevent waiting to enter traffic: " + preventwaitingtoentertraffic)

    writeExperiencedPlans = "false"
    if context.config("writeexperiencedplans"):
        writeExperiencedPlans = "true"
        print("Write experienced plans: " + writeExperiencedPlans)

    
    last_iteration = context.config("last_iteration")
    if (not context.config("use_vdf")):
        # Run simulation
        eqasim.run(context, "org.eqasim.switzerland.ch.RunSimulation", [
            "--config-path", config_path,
            "--config:controler.lastIteration", str(last_iteration),
            "--config:controler.writeEventsInterval", str(last_iteration),
            "--config:controler.writePlansInterval", str(last_iteration),
            "--config:qsim.numberOfThreads", str(min(context.config("threads"),12)),
            "--config:linkStats.writeLinkStatsInterval", str(last_iteration),
            "--config:linkStats.averageLinkStatsOverIterations", str(1),
            # if one wants to visualize outputs, trips file needs to be generated 
            # so one should set this to something other than 0, and preferebly to something 
            # that will output trips file at the end of the simulation
            "--config:controller.writeTripsInterval", str(last_iteration),
            "--config:eqasim.useScheduleBasedTransport", scheduleBasedPTconfig,
            "--preventwaitingtoentertraffic", preventwaitingtoentertraffic,
            "--config:scoring.writeExperiencedPlans", writeExperiencedPlans
        ])
    else:
        # Run simulation with vdf
        eqasim.run(context, "org.eqasim.switzerland.ch.RunVDFSimulation", [
            "--config-path", config_path,
            "--config:controler.lastIteration", str(last_iteration),
            "--config:controler.writeEventsInterval", str(last_iteration),
            "--config:controler.writePlansInterval", str(last_iteration),
            "--config:qsim.numberOfThreads", str(min(context.config("threads"),12)),
            "--config:linkStats.writeLinkStatsInterval", str(last_iteration),
            "--config:linkStats.averageLinkStatsOverIterations", str(1),
            "--config:controller.writeTripsInterval", str(0),
            "--config:eqasim.useScheduleBasedTransport", scheduleBasedPTconfig,          
        ])
    assert os.path.exists("%s/simulation_output/output_events.xml.gz" % context.path())
    
    return context.path()
