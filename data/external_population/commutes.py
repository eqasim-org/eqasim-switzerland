import os
import geopandas as gpd
import pandas as pd

"""
Reads the real (non-downsampled) home->work commutes of the French
population from cmdp_commutes.gpkg. Kept independent of
data.external_population.read_outputs (which depends on
synthesis.population.enriched) so that
synthesis.population.spatial.primary.work.{fixed,moving}_work_locations can
use it without creating a dependency cycle back to themselves through
synthesis.population.matching.matched_v1's use of work_locations.
"""


def configure(context):
    context.config("include_external_population", default = False)

    if context.config("include_external_population"):
        context.config("external_population_folder")


def execute(context):
    if not context.config("include_external_population"):
        return

    folder = context.config("external_population_folder")

    assert any(f.endswith("_commutes.gpkg") for f in os.listdir(folder)), f"No *_commutes.gpkg file found in {folder}"
    commutes_file = next(f for f in os.listdir(folder) if f.endswith("_commutes.gpkg"))

    commutes     = gpd.read_file(os.path.join(folder, commutes_file))[["person_id", "geometry"]]
    commutes.crs = "EPSG:2154"
    commutes     = commutes.to_crs("EPSG:2056")

    # The line's end point is the work destination.
    commutes["destination_x"] = commutes["geometry"].apply(lambda g: g.coords[-1][0])
    commutes["destination_y"] = commutes["geometry"].apply(lambda g: g.coords[-1][1])

    return pd.DataFrame(commutes[["person_id", "destination_x", "destination_y"]])
