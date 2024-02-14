import os.path
import matsim.runtime.pt2matsim as pt2matsim

def configure(context):
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.pt2matsim")
    context.stage("matsim.scenario.supply.processed")


def execute(context):
    java = context.stage("utils.java")
    jar, tmp_path = context.stage("matsim.java.pt2matsim")
    paths = context.stage("matsim.scenario.supply.processed")

    # Run plausibility checks
    pt2matsim.run(context, "org.matsim.pt2matsim.run.CheckMappedSchedulePlausibility", [
       paths["schedule"], paths["network"], "EPSG:2154", context.path()
    ])
    assert(os.path.exists("%s/allPlausibilityWarnings.csv" % context.path()))
    return context.path()
