from .penalty_calibration import REGIONS
import os
import json

def configure(context):
  pass

def execute(context):
  path = os.path.join(context.path(), "regions.json")
  with open(path, "w") as f:
    json.dump(REGIONS, f)
  
  return path


