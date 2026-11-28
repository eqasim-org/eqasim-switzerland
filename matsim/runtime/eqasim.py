import os
import shutil
import subprocess as sp

import matsim.runtime.git as git
import matsim.runtime.java as java
import matsim.runtime.maven as maven

DEFAULT_EQASIM_VERSION = "2.0.0"
DEFAULT_EQASIM_BRANCH = "cmdp_new_developments"
DEFAULT_EQASIM_COMMIT = -1


def configure(context):
    context.stage("matsim.runtime.git")
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.maven")

    context.config("eqasim_version", DEFAULT_EQASIM_VERSION)
    context.config("eqasim_branch", DEFAULT_EQASIM_BRANCH)
    context.config("eqasim_commit", DEFAULT_EQASIM_COMMIT)
    context.config("eqasim_repository", "https://github.com/eqasim-org/eqasim-java.git")
    context.config("eqasim_path", "")
    context.config("git_binary", "git")


def execute(context):

    # Prepare the location of the built or supplied JAR.

    jar_relative_path = "eqasim-java/switzerland/target/switzerland.jar"
    jar_path = os.path.join(context.path(), jar_relative_path)

    # Clone the selected branch and build the Switzerland JAR.

    if context.config("eqasim_path") == "":
        branch = context.config("eqasim_branch")

        git.run(context, [
            "clone", "--single-branch", "-b", branch,
            context.config("eqasim_repository"), "eqasim-java"
        ])

        # The clone already selects the latest branch commit; optionally pin it.

        commit = str(context.config("eqasim_commit")).strip()
        if commit != "-1":
            git.run(context, ["checkout", commit],
                    cwd=os.path.join(context.path(), "eqasim-java"))

        # Build eqasim.

        maven.run(context, ["-Pstandalone", "--projects", "switzerland", "--also-make", "package", "-DskipTests"], cwd = "%s/eqasim-java" % context.path())

    else:

        # Copy an explicitly supplied JAR instead of building it.

        os.makedirs("%s/eqasim-java/switzerland/target" % context.path())
        shutil.copy(context.config("eqasim_path"), jar_path )

    assert os.path.exists(jar_path), "Cannot find eqasim jar at: %s" % jar_path
    return jar_relative_path


def run(context, command, arguments, cwd = None, memory = None):
    jar_path = context.stage("matsim.runtime.eqasim")
    jar_path = "{}/{}".format(context.path("matsim.runtime.eqasim"), jar_path)
    java.run(context, command, arguments, jar_path, cwd=cwd, memory=memory)


def validate(context):
    path = context.config("eqasim_path")

    if path == "":
        if str(context.config("eqasim_commit")).strip() == "-1":
            branch = context.config("eqasim_branch")
            result = sp.check_output([
                context.config("git_binary"), "ls-remote", "--exit-code", "--heads",
                context.config("eqasim_repository"), f"refs/heads/{branch}",
            ], text=True, timeout=30)
            return result.split()[0]
        return True

    if not os.path.exists(path):
        raise RuntimeError("Cannot find eqasim at: %s" % path)

    return os.path.getmtime(path)
