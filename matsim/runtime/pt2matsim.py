import subprocess as sp
import os, os.path
import shutil

import matsim.runtime.git as git
import matsim.runtime.java as java
import matsim.runtime.maven as maven

def configure(context):
    context.stage("matsim.runtime.git")
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.maven")

    context.config("pt2matsim_version", "25.8")
    context.config("pt2matsim_branch", "v25.8")
    context.config("pt2matsim_path", "") 

def run(context, command, arguments, vm_arguments):
    jar_path = context.stage("matsim.runtime.pt2matsim")    
    java.run(context, command, arguments, jar_path, vm_arguments)


def execute(context):
    version = context.config("pt2matsim_version")
    branch  = context.config("pt2matsim_branch")
    
    if context.config("pt2matsim_path") == "":
        # Clone repository and checkout version
        git.run(context, [
            "clone", "https://github.com/matsim-org/pt2matsim.git",
            "--branch", branch,
            "--single-branch", "pt2matsim",
            "--depth", "1"
        ])

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
