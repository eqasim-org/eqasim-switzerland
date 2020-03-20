import subprocess as sp
import os.path

def configure(context, require):
    require.stage("matsim.java.pt2matsim")
    require.stage("matsim.network.mapped")
    require.stage("utils.java")

def execute(context):
    java = context.stage("utils.java")
    jar, tmp_path = context.stage("matsim.java.pt2matsim")
    paths = context.stage("matsim.network.mapped")

    # Do plausibility checks

    java(jar, "org.matsim.pt2matsim.run.CheckMappedSchedulePlausibility", [
        "-Djava.io.tmpdir=%s/java_tmp" % tmp_path,
        paths["schedule"], paths["network"], "EPSG:2056", context.cache_path
    ], cwd = context.cache_path)

    assert(os.path.exists("%s/allPlausibilityWarnings.csv" % context.cache_path))
    return context.cache_path
