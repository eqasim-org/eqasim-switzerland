import pipeline
import data.microcensus.trips

config = {
    "raw_data_path" : "/run/media/sebastian/shoerl_data/population/raw",
    "target_path" : "/run/media/sebastian/shoerl_data/temp",
    "threads" : 4,
    "stages" : [
        #"population.matching",
        #"population.sociodemographics",
        "data.microcensus.households",
        #"data.microcensus.trips",
        #"population.commute",
        #"population.primary_locations",
        #"data.od.raw",
        #"data.od.matrix",
        #"data.statpop.statpop",
        #"data.misc.spatial_structure",
        #"data.statpop.spatial_structure"
        #"data.microcensus.spatial_structure"
        #"data.spatial.types"
    ]
}

pipeline.run(
    config["stages"],
    target_path = config["target_path"],
    config = config)
