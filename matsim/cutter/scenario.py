import os.path
import shutil
import glob
import matsim.runtime.eqasim as eqasim
import logging
logger = logging.getLogger("synpp")

def configure(context):
    context.config("cutterVersion", default="v1")
    context.config("use_vdf", default=False)
    context.config("extent_path", default="")
    context.config("extent_prefix", default="")

    if context.config("extent_path") != "" and context.config("extent_prefix") != "":
        if context.config("cutterVersion").lower() in ["v1","v.1"]:
            context.stage("matsim.cutter.runCutter", alias="cutter_run")
        elif context.config("cutterVersion").lower() in ["v2","v.2"]:
            assert context.config("use_vdf"), "VDF must be used for cutter version 2"
            context.stage("matsim.cutter.runCutterV2", alias="cutter_run")
        else:
            raise ValueError("Invalid cutter version specified:", context.config("cutterVersion"))

def execute(context):
    if context.config("extent_path") == "" or context.config("extent_prefix") == "":
        logger.warning("Extent path or extent prefix is not set. Cutter will not be executed.")
        return ""
    
    return context.stage("cutter_run")