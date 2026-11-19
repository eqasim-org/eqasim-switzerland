import os, os.path
import shutil

import matsim.runtime.git as git
import matsim.runtime.java as java
import matsim.runtime.maven as maven

DEFAULT_BRANCH = "dev/pt-integration"
DEFAULT_COMMIT = "a7f88ee4a20fc90b141909dde417bedd3abefe79"


def configure(context):
    context.stage("matsim.runtime.git")
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.maven")

    context.config("pt2matsim_version", "26.6.1-SNAPSHOT")
    branch = context.config("pt2matsim_branch", DEFAULT_BRANCH)
    if not context.config("pt2matsim_path", ""):
        context.config("pt2matsim_commit", DEFAULT_COMMIT if branch == DEFAULT_BRANCH else None)


def execute(context):

    # Select the source version or locally supplied JAR.

    version = context.config("pt2matsim_version")
    branch  = context.config("pt2matsim_branch")
    
    if context.config("pt2matsim_path") == "":
        # Keep branch history for pinned builds so the commit survives branch advances.

        commit = context.config("pt2matsim_commit")
        depth = [] if commit else ["--depth", "1"]
        git.run(context, [
            "clone", "--branch", branch, "--single-branch", *depth,
            "https://github.com/matsim-org/pt2matsim.git", "pt2matsim"
        ])
        if commit:
            git.run(context, ["checkout", "--detach", commit], cwd=os.path.join(context.path(), "pt2matsim"))

        # Build pt2matsim
        maven.run(context, ["package", "-Dskip.surefire.tests=true"], cwd = "%s/pt2matsim" % context.path())
        jar_path = "%s/pt2matsim/target/pt2matsim-%s-shaded.jar" % (context.path(), version)

        # Test pt2matsim
        java.run(context, "org.matsim.pt2matsim.run.CreateDefaultOsmConfig", [
            "test_config.xml"
        ], jar_path)
        
        jar_path = "%s/pt2matsim/target/pt2matsim-%s-shaded.jar" % (
                        context.path(), version
                    )
    
    else:
        filename = os.path.basename(context.config("pt2matsim_path"))
        jar_path = "%s/pt2matsim/target/%s" % (context.path(), filename)
        os.makedirs("%s/pt2matsim/target" % context.path())
        
        shutil.copy(context.config("pt2matsim_path"), jar_path)
        
    return jar_path


def run(context, command, arguments, vm_arguments):
    jar_path = context.stage("matsim.runtime.pt2matsim")
    java.run(context, command, arguments, jar_path, vm_arguments)
