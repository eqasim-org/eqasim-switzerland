import pipeline
import data.microcensus.trips
import yaml
import sys

config_path = "config.yml"

if len(sys.argv) > 1:
    config_path = sys.argv[1]

with open(config_path) as f:
    config = yaml.load(f)

pipeline.run(
    config["stages"],
    target_path = config["target_path"],
    config = config)
