import pandas as pd
import geopandas as gpd
import data.spatial.ovgk

def configure(context):
    context.stage("synthesis.population.destinations_statent")
    context.stage("synthesis.population.spatial.primary.work.remote_locations", alias="remote_work_locations")
    context.stage("data.cross_border.interview_places")
    context.stage("data.spatial.ovgk")
    context.stage("data.spatial.municipality_types")
    context.stage("data.spatial.municipalities")
    context.config("generate_outbound_flows", False)

    if context.config("generate_outbound_flows"):
        context.stage("data.locations_fr.secondary")


def build_border_destinations(context):
    df = context.stage("data.cross_border.interview_places")[["border_crossing_point_id", "geometry"]].copy()
    df = df.rename(columns={"border_crossing_point_id": "destination_id"})

    # Truncate to int (not round) to match the int(geometry.x)/int(geometry.y) cast that
    # matsim/scenario/population.py applies when writing an activity's coordinates. Both sides
    # must use the same truncation so that a facility and the activity referencing it end up
    # with identical x/y, as required by MATSim's ScenarioValidator.
    df["destination_x"] = df.geometry.x.astype(int)
    df["destination_y"] = df.geometry.y.astype(int)
    df["number_employees"] = 0

    for column in ("offers_work", "offers_education", "offers_leisure", "offers_shop", "offers_other",
                   "offers_work_secondary", "offers_education_secondary", "offers_home_secondary"):
        df[column] = False

    df_ovgk = context.stage("data.spatial.ovgk")
    df_spatial = data.spatial.ovgk.impute(context, df_ovgk, df, ["destination_id"], chunk_size=1e3, point_type="border_facility")
    df = df.merge(df_spatial[["destination_id", "ovgk"]], how="left", on="destination_id")

    df_municipality_type = context.stage("data.spatial.municipality_types")
    df_municipalities, _ = context.stage("data.spatial.municipalities")
    df_municipalities = df_municipalities.merge(df_municipality_type)[["municipality_type", "municipality_id", "geometry"]]
    assert df.crs == df_municipalities.crs
    df = gpd.sjoin_nearest(df, df_municipalities, how="left").drop(columns=["index_right"])

    # Some border points may fall outside the municipality layer's coverage (e.g. right on
    # or just past the border); sjoin_nearest normally still finds the closest municipality,
    # but fall back to defaults so the pipeline never breaks on an unmatched point.
    df["municipality_type"] = df["municipality_type"].fillna("rural")
    df["municipality_id"]   = df["municipality_id"].fillna(-1)

    return df[["destination_id", "number_employees", "destination_x", "destination_y",
               "offers_work", "offers_education", "offers_leisure", "offers_shop", "offers_other",
               "offers_work_secondary", "offers_education_secondary", "offers_home_secondary",
               "ovgk", "municipality_id", "municipality_type", "geometry"]]


def execute(context):
    df_destinations = context.stage("synthesis.population.destinations_statent")

    # add remote work locations
    df_remote_work = context.stage("remote_work_locations")
    df_destinations = pd.concat([df_destinations, df_remote_work], ignore_index=True)

    # add the french population
    if context.config("generate_outbound_flows"):
        df_FR = context.stage("data.locations_fr.secondary")
        df_destinations = pd.concat([df_destinations, df_FR], ignore_index=True)

    # add cross-border interview / border-crossing points. These keep their
    # string "BCP_..." destination_id and have all offers_* set to False, so
    # they are not (yet) selectable by the secondary-location choice models.
    df_borders = build_border_destinations(context)
    df_destinations = pd.concat([df_destinations, df_borders], ignore_index=True)

    # Several raw BPE records can collapse onto the same destination_id (e.g.
    # one SIRET, several sites); merge them since MATSim needs unique ids.
    offers_columns = [c for c in df_destinations.columns if c.startswith("offers_")]
    other_columns  = [c for c in df_destinations.columns
                       if c not in offers_columns and c not in ("destination_id", "number_employees")]

    aggregation = {c: "any" for c in offers_columns}
    aggregation["number_employees"] = "sum"
    aggregation.update({c: "first" for c in other_columns})

    df_destinations = df_destinations.groupby("destination_id", as_index = False).agg(aggregation)
    df_destinations = gpd.GeoDataFrame(df_destinations, geometry = "geometry", crs = df_borders.crs)

    return df_destinations