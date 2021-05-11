import os


def configure(context):
    context.stage("matsim.java.pt2matsim")
    context.stage("matsim.network.mapped")
    context.stage("utils.java")


def execute(context):
    java = context.stage("utils.java")
    jar, tmp_path = context.stage("matsim.java.pt2matsim")
    paths = context.stage("matsim.network.mapped")

    # Do plausibility checks

    java(jar, "org.matsim.pt2matsim.run.CheckMappedSchedulePlausibility", [
        "-Djava.io.tmpdir=%s/java_tmp" % tmp_path,
        paths["schedule"], paths["network"], "epsg:2056", context.cache_path
    ], cwd=context.cache_path)

    assert (os.path.exists("%s/allPlausibilityWarnings.csv" % context.cache_path))
    return context.cache_path
