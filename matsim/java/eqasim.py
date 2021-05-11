import subprocess as sp


def configure(context):
    context.stage("utils.java")


def execute(context):
    java = context.stage("utils.java")

    sp.check_call([
        "git", "clone", "https://github.com/eqasim-org/eqasim-java.git"
    ], cwd=context.cache_path)

    sp.check_call([
        "git", "checkout", "v1.0.5"
    ], cwd="%s/eqasim-java" % context.cache_path)

    sp.check_call([
        "mvn", "-Pstandalone", "package"
    ], cwd="%s/eqasim-java" % context.cache_path)

    jar = "%s/eqasim-java/switzerland/target/switzerland-1.0.5.jar" % context.cache_path
    return jar
