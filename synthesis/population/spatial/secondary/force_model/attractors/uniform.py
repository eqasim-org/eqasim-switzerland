"""One attractor per destination and offered purpose, as in eqasim-france."""
import geopandas as gpd
import pandas as pd


def configure(context):
    context.stage("synthesis.population.destinations")


def execute(context):

    # Expand multi-purpose Swiss/French facilities into purpose-specific attractors.

    destinations = context.stage("synthesis.population.destinations").to_crs(2056)
    groups = [destinations.loc[destinations["offers_" + purpose].fillna(False).astype(bool),
                               ["geometry"]].assign(activity_type=purpose, weight=1.)
              for purpose in ("shop", "leisure", "other")]

    # Preserve metre coordinates; zero-purpose destinations supply no field.

    return gpd.GeoDataFrame(pd.concat(groups, ignore_index=True), geometry="geometry", crs=2056)
