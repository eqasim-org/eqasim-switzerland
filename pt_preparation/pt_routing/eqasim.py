import subprocess as sp
import os, os.path, shutil

import matsim.runtime.git as git
import matsim.runtime.java as java
import matsim.runtime.maven as maven

DEFAULT_EQASIM_VERSION = "2.0.0"
DEFAULT_EQASIM_BRANCH = "cmdp"
DEFAULT_EQASIM_COMMIT = "8807c5a" # "e06a8b8e21e7ad985a513da9c7f156e0c7dc9667" #"4145959"


def configure(context):
    context.stage("matsim.runtime.git")
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.maven")

    context.config("eqasim_version", DEFAULT_EQASIM_VERSION)
    context.config("eqasim_branch", DEFAULT_EQASIM_BRANCH)
    context.config("eqasim_commit", DEFAULT_EQASIM_COMMIT)
    context.config("eqasim_repository", "https://github.com/eqasim-org/eqasim-java.git")
    context.config("eqasim_path", "")
    context.config("eqasim_server_path", "")


def run(context, command, arguments):
    jar_path = context.stage("pt_preparation.pt_routing.eqasim")
    java.run(context, command, arguments, jar_path)


def execute(context):
    version = context.config("eqasim_version")

    # Normal case: we clone eqasim
    if context.config("eqasim_path") == "" or context.config("eqasim_server_path") == "":
        # Clone repository and checkout version
        branch = context.config("eqasim_branch")

        git.run(context, [
            "clone", "--single-branch", "-b", branch,
            context.config("eqasim_repository"), "eqasim-java"
        ])

        # Select the configured commit or tag
        commit = str(context.config("eqasim_commit"))

        git.run(context, [
            "checkout", commit
        ], cwd = "{}/eqasim-java".format(context.path()))


        # Build eqasim
        maven.run(context, ["-Pstandalone", "--projects", "switzerland, server", "--also-make", "package", "-DskipTests"], cwd = "%s/eqasim-java" % context.path())

    # Special case: We provide the jar directly. This is mainly used for
    # creating input to unit tests of the eqasim-java package.
    else:
        os.makedirs("%s/eqasim-java/switzerland/target" % context.path())
        shutil.copy(context.config("eqasim_path"),
            "%s/eqasim-java/switzerland/target/switzerland-%s.jar" % (context.path(), version))
        
        os.makedirs("%s/eqasim-java/server/target" % context.path())
        shutil.copy(context.config("eqasim_server_path"),
            "%s/eqasim-java/server/target/server-%s.jar" % (context.path(), version))

    local_dir = context.path()  
    switzerland_path = f"{local_dir}/eqasim-java/switzerland/target/switzerland-{version}.jar"
    server_path      = f"{local_dir}/eqasim-java/server/target/server-{version}.jar"

    return f"{switzerland_path}:{server_path}"


def validate(context):
    path = context.config("eqasim_path")

    if path == "":
        return True

    if not os.path.exists(path):
        raise RuntimeError("Cannot find eqasim at: %s" % path)

    return os.path.getmtime(path)