import pipeline
import data.microcensus.trips

pipeline.run([
    #"population.matching",
    "data.od.matrix"
], target_path = "/run/media/sebastian/shoerl_data/temp", config = {
    "raw_data_path" : "/run/media/sebastian/shoerl_data/population"
})
