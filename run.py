import pipeline
import data.microcensus.trips

pipeline.run([
    #"population.matching",
    "population.sociodemographics"
], target_path = "/run/media/sebastian/shoerl_data/temp", config = {
    "raw_data_path" : "/run/media/sebastian/shoerl_data/population"
})
