import os
import subprocess as sp


def configure(context):
    context.stage("utils.java")


def execute(context):
    java = context.stage("utils.java")

    os.mkdir("%s/java_tmp" % context.cache_path)

    sp.check_call([
        "git", "clone", "https://github.com/matsim-org/pt2matsim.git"
    ], cwd=context.cache_path)

    sp.check_call([
        "git", "checkout", "v19.10"
    ], cwd="%s/pt2matsim" % context.cache_path)

    sp.check_call([
        "mvn", "-Djava.io.tmpdir=%s/java_tmp" % context.cache_path, "package"
    ], cwd="%s/pt2matsim" % context.cache_path)

    jar = "%s/pt2matsim/target/pt2matsim-19.10-shaded.jar" % context.cache_path
    java(jar, "org.matsim.pt2matsim.run.CreateDefaultOsmConfig", ["test_config.xml"], cwd=context.cache_path)

    assert (os.path.exists("%s/test_config.xml" % context.cache_path))
    assert (os.path.exists("%s/java_tmp/GeoTools" % context.cache_path))

    return jar, "%s/java_tmp" % context.cache_path
