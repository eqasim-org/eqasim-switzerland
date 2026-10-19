"""Launch the simulation fully configured by matsim.simulation.prepare."""
import os.path

import matsim.runtime.eqasim as eqasim


def configure(context):
    context.stage("matsim.simulation.prepare")
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.eqasim")
    context.config("java_memory", "50G", volatile=True)


def execute(context):
    simulation = context.stage("matsim.simulation.prepare")
    eqasim.run(context, simulation["main_class"],
               ["--config-path", simulation["config"], *simulation["arguments"]])
    assert os.path.exists(os.path.join(context.path(), "simulation_output", "output_events.xml.gz"))
    return context.path()
