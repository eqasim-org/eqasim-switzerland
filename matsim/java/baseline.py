import requests
from tqdm import tqdm
import subprocess as sp
import os.path

def configure(context, require):
    require.stage("utils.java")

def execute(context):
    java = context.stage("utils.java")

    sp.check_call([
        "git", "clone", "https://github.com/matsim-eth/baseline_scenario.git"
    ], cwd = context.cache_path)

    sp.check_call([
        "git", "checkout", "origin/synpop"
    ], cwd = "%s/baseline_scenario" % context.cache_path)

    sp.check_call([
        "mvn", "-version"
    ], cwd = "%s/baseline_scenario" % context.cache_path)

    sp.check_call([
        "mvn", "-Pstandalone", "package"
    ], cwd = "%s/baseline_scenario" % context.cache_path)

    jar = "%s/baseline_scenario/target/baseline_scenario-0.0.1-SNAPSHOT.jar" % context.cache_path
    return jar
