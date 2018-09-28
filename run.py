import pipeline
import data.microcensus.trips

config = {
    "raw_data_path" : "/run/media/sebastian/shoerl_data/population/raw",
    "target_path" : "/run/media/sebastian/shoerl_data/temp",
    "stages" : [
        #"population.sociodemographics",
        "data.od.matrix"
    ]
}

pipeline.run(
    config["stages"],
    target_path = config["target_path"],
    config = config)
