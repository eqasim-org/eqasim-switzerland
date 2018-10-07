import subprocess as sp
import os.path

def configure(context, require):
    require.stage("matsim.java.pt2matsim")
    require.stage("matsim.network.mapped")

def execute(context):
    jar = context.stage("matsim.java.pt2matsim")
    paths = context.stage("matsim.network.mapped")

    # Do plausibility checks

    java(jar, "org.matsim.pt2matsim.run.CheckMappedSchedulePlausibility", [
        paths["schedule"], paths["network"], "EPSG:2056", context.cache_path
    ], cwd = context.cache_path)

    assert(os.path.exists("%s/allPlausibilityWarnings.csv" % context.cache_path))
    return context.cache_path
