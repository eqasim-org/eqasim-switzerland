"""Launch the simulation fully configured by matsim.simulation.prepare."""
import os.path
import matsim.runtime.eqasim as eqasim
from matsim.config import tools

def configure(context):
    context.stage("matsim.simulation.prepare")
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.eqasim")

    context.config("java_memory", "50G", volatile=True)
    context.config("threads", volatile=True)
    context.config("use_vdf", False)
    context.config("use_dsim", False)
    context.config("preventwaitingtoentertraffic", False)


def execute(context):
    config_path = context.stage("matsim.simulation.prepare")

    # whether to use vdf or qsim
    vdf = tools.parse_boolean(context.config("use_vdf"))
    arguments = []
    if vdf:
        # Java resets this engine switch unless it is supplied on launch.
        arguments = ["--generateNetworkEvents", "true"]
    elif tools.parse_boolean(context.config("preventwaitingtoentertraffic")):
        # The QSim capacity-easing setter is not exposed as an XML parameter.
        arguments = ["--preventwaitingtoentertraffic", "y"]

    # whether to use qsim
    dsim = tools.parse_boolean(context.config("use_dsim"))
    if dsim:
        arguments.extend([
            "--use-dsim", "true",
            "--config:dsim.threads", str(context.config("threads"))
        ])

    # Keep the explicit replanning limit even when preparation came from cache.
    arguments.extend(["--config:global.numberOfThreads", str(context.config("threads"))])

    # Run the simulation with the prepared config. The main class is chosen based on the engine.
    main_class= "org.eqasim.switzerland.ch_cmdp.RunVDFSimulation" if vdf else "org.eqasim.switzerland.ch_cmdp.RunSimulation"
    eqasim.run(context, main_class, ["--config-path", config_path, *arguments])

    # when using dsim, and with matsim 2026, the default compression is zst, withthe older version it was gz
    assert (os.path.exists(os.path.join(context.path(), "simulation_output", "output_events.xml.gz")) or
            os.path.exists(os.path.join(context.path(), "simulation_output", "output_events.xml.zst")))
    return context.path()
