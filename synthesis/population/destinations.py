import pandas as pd
import geopandas as gpd
import data.spatial.ovgk
from data.cross_border.destinations import make_entry_border_facility_id, make_exit_border_facility_id

def configure(context):
    context.stage("synthesis.population.destinations_statent")
    context.stage("synthesis.population.spatial.primary.work.remote_locations", alias="remote_work_locations")
    context.stage("data.cross_border.swiss_residents_od")
    context.stage("data.spatial.ovgk")
    context.stage("data.spatial.municipality_types")
    context.stage("data.spatial.municipalities")
    context.config("generate_outbound_flows", False)
    context.config("include_cross_border", default=False)

    if context.config("include_cross_border"):
        context.stage("data.cross_border.destinations")

    if context.config("generate_outbound_flows"):
        context.stage("data.locations_fr.secondary")


def build_border_destinations(context):
    border_frames = []

    if context.config("include_cross_border"):
        df_foreign = context.stage("data.cross_border.destinations")

        # Foreign-resident cross-border agents now expose one facility for entering
        # Switzerland and one for exiting it. Drop duplicates so shared surveyed
        # crossings remain compact while through-trip anchors stay person-specific.
        entry = df_foreign[["entry_interview_point_id", "entry_interview_geometry_point"]].copy()
        entry = entry.rename(columns={
            "entry_interview_point_id": "destination_id",
            "entry_interview_geometry_point": "geometry",
        })

        exit = df_foreign[["exit_interview_point_id", "exit_interview_geometry_point"]].copy()
        exit = exit.rename(columns={
            "exit_interview_point_id": "destination_id",
            "exit_interview_geometry_point": "geometry",
        })

        border_frames.append(pd.concat([entry, exit], ignore_index=True, sort=False))

    df_cb = context.stage("data.cross_border.swiss_residents_od")[["cross_border_person_id", "border_crossing_point"]].copy()

    # Swiss residents are sampled later as either home -> border or border -> home,
    # so both directional facilities must exist before MATSim writes facilities.
    df_cb_entry = df_cb.rename(columns={"border_crossing_point": "geometry"}).copy()
    df_cb_entry["destination_id"] = df_cb_entry["cross_border_person_id"].apply(make_entry_border_facility_id)

    df_cb_exit = df_cb.rename(columns={"border_crossing_point": "geometry"}).copy()
    df_cb_exit["destination_id"] = df_cb_exit["cross_border_person_id"].apply(make_exit_border_facility_id)

    df_cb = pd.concat([df_cb_entry, df_cb_exit], ignore_index=True, sort=False)
    df_cb = gpd.GeoDataFrame(df_cb[["destination_id", "geometry"]], geometry="geometry", crs="EPSG:2056")
    border_frames.append(df_cb)

    df = gpd.GeoDataFrame(
        pd.concat(border_frames, ignore_index=True, sort=False),
        geometry="geometry",
        crs="EPSG:2056",
    )
    df = df.dropna(subset=["destination_id", "geometry"]).drop_duplicates("destination_id")

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
    df_remote_work["destination_id"] = df_remote_work["destination_id"].astype("int64")
    df_destinations["destination_id"] = df_destinations["destination_id"].astype("int64")
    df_destinations = pd.concat([df_destinations, df_remote_work], ignore_index=True)

    # add the french population
    if context.config("generate_outbound_flows"):
        df_FR = context.stage("data.locations_fr.secondary")
        df_FR["destination_id"] = df_FR["destination_id"].astype("int64")
        df_destinations = pd.concat([df_destinations, df_FR], ignore_index=True)

    # add cross-border interview / border-crossing points. These keep their
    # string "BCP_..." destination_id and have all offers_* set to False, so
    # they are not (yet) selectable by the secondary-location choice models.
    df_borders = build_border_destinations(context)
    df_destinations = pd.concat([df_destinations, df_borders], ignore_index=True)

    return df_destinations
