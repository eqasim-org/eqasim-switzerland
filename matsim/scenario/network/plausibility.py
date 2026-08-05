import os
import matsim.runtime.pt2matsim as pt2matsim


def configure(context):
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.pt2matsim")
    context.stage("matsim.scenario.network.mapped")


def execute(context):
    paths = context.stage("matsim.scenario.network.mapped")

    # Do plausibility checks
    pt2matsim.run(context, "org.matsim.pt2matsim.run.CheckMappedSchedulePlausibility", [
        paths["schedule"], paths["network"], "epsg:2056", context.path()
    ])

    assert (os.path.exists("%s/allPlausibilityWarnings.csv" % context.path()))
    return context.path()
