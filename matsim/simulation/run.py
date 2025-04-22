import os.path

import matsim.runtime.eqasim as eqasim


def configure(context):
    context.stage("matsim.simulation.prepare")    
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.eqasim")
    
    context.config("use_vdf", default=False)
    context.config("threads")
    context.config("last_iteration", 60)

def execute(context):
    config_path = "%s/%s" % (
        context.path("matsim.simulation.prepare"),
        context.stage("matsim.simulation.prepare")
    )
    
    last_iteration = context.config("last_iteration")
    if (not context.config("use_vdf")):
        # Run simulation
        eqasim.run(context, "org.eqasim.switzerland.RunSimulation", [
            "--config-path", config_path,
            "--config:controler.lastIteration", str(last_iteration),
            "--config:controler.writeEventsInterval", str(last_iteration),
            "--config:controler.writePlansInterval", str(last_iteration),
            "--config:qsim.numberOfThreads", str(min(context.config("threads"),8)),
            "--config:linkStats.writeLinkStatsInterval", str(last_iteration),
            "--config:linkStats.averageLinkStatsOverIterations", str(1)
        ])
    else:
        # Run simulation with vdf
        eqasim.run(context, "org.eqasim.switzerland.RunVDFSimulation", [
            "--config-path", config_path,
            "--config:controler.lastIteration", str(last_iteration),
            "--config:controler.writeEventsInterval", str(last_iteration),
            "--config:controler.writePlansInterval", str(last_iteration),
            "--config:qsim.numberOfThreads", str(min(context.config("threads"),12)),
            "--config:linkStats.writeLinkStatsInterval", str(last_iteration),
            "--config:linkStats.averageLinkStatsOverIterations", str(1)          
        ])
    assert os.path.exists("%s/simulation_output/output_events.xml.gz" % context.path())
    
    return context.path()
