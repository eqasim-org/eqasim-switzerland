import data.spatial.utils as spatial_utils


def configure(context):
    context.stage("synthesis.population.enriched")
    context.stage("data.statpop.households")


def execute(context):
    df_homes = context.stage("synthesis.population.enriched")[[
        "household_id", "home_municipality_id", "home_x", "home_y"]].rename({"home_municipality_id": "municipality_id",
                                                                             "home_x": "x",
                                                                             "home_y": "y"},
                                                                            axis=1)

    df_homes = spatial_utils.to_gpd(context, df_homes, coord_type="home")
    df_homes = df_homes.drop_duplicates(subset="household_id")

    return df_homes[["household_id", "municipality_id", "geometry"]]
