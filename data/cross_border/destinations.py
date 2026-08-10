import numpy as np
import pandas as pd
import geopandas as gpd


ENTRY_SUFFIX = "_entry"
EXIT_SUFFIX = "_exit"


def make_entry_border_facility_id(border_crossing_point_id):
    # Directional suffixes keep one logical crossing point but expose two MATSim facilities.
    return f"{border_crossing_point_id}{ENTRY_SUFFIX}"


def make_exit_border_facility_id(border_crossing_point_id):
    # The exit facility is separate so MATSim can attach it to the opposite one-way link.
    return f"{border_crossing_point_id}{EXIT_SUFFIX}"

def configure(context):
    context.config("random_seed")

    context.stage("data.cross_border.match_activity_chain")
    context.stage("data.spatial.municipalities")
    context.stage("data.statent.statent")


def execute(context):
    df_all = context.stage("data.cross_border.match_activity_chain")

    df_through = df_all[df_all["label"] == "Through"]
    df         = df_all[df_all["label"] == "From-To"]

    destinations = gpd.GeoSeries.from_xy(df["destination_x"], df["destination_y"])
    destinations = gpd.GeoDataFrame(geometry = destinations, crs = "EPSG:2056")

    del df["destination_x"]
    del df["destination_y"]

    municipalities = context.stage("data.spatial.municipalities")[0]

    merged = gpd.sjoin(destinations, municipalities, 
                       how = "left", predicate="within")
    
    df["destination_municipality_id"]   = merged["municipality_id"].values
    df["destination_municipality_name"] = merged["municipality_name"].values
    
    statent         = context.stage("data.statent.statent")[["enterprise_id", "x", "y", "noga", "municipality_id", "number_employees"]].copy()
    statent.columns = ["destination_id", "destination_x", "destination_y", "noga", "municipality_id", "number_employees"]

    statent["offers_work"]           = True
    statent["offers_other"]          = True
    statent["offers_work_secondary"] = True
    statent["offers_leisure"]        = statent["noga"].str.startswith("90") | statent["noga"].str.startswith("56")
    statent["offers_education"]      = statent["noga"].str.startswith("85")
    statent["offers_shop"]           = statent["noga"].str.startswith("47")

    for purpose in ["work", "work_secondary", "other", "education", "leisure", "shop"]:
        mask_purpose         = (df["trip_purpose"] == purpose)
        mask_statent_purpose = statent[f"offers_{purpose}"]

        mun_ids = list(set(df[mask_purpose]["destination_municipality_id"]))

        for mun_id in mun_ids:
            mask_mun = df["destination_municipality_id"] == mun_id
            mask     = mask_purpose & mask_mun

            mask_statent_mun = statent["municipality_id"] == mun_id
            mask_statent     = mask_statent_purpose & mask_statent_mun

            candidates = statent[mask_statent]
            N_sample   = np.sum(mask)

            if len(candidates) == 0:
                candidates = statent[mask_statent_mun]

            weights = candidates["number_employees"]

            sampled = candidates.sample(
                n = N_sample,
                random_state = context.config("random_seed"),
                replace = True,
                weights = weights
            )

            df.loc[mask, "destination_id"] = sampled["destination_id"].values
            df.loc[mask, "destination_x"]  = sampled["destination_x"].values
            df.loc[mask, "destination_y"]  = sampled["destination_y"].values

    df = pd.concat([df, df_through])

    df["destination_id"] = df["destination_id"].fillna(-1)

    # All entries with no destination assigned have label = Through, which means that the corresponding agents are only crossing Switzerland without stopping
    # in the country. And as STATENT only covers Switzerland, we cannot assign real destinations to the people crossing the country.

    assert (
        df.loc[df["destination_id"] == -1, "label"] == "Through"
    ).all(), "Found rows with destination_id = -1 and label != 'Through'"

    # From-To agents use the surveyed interview point twice, once in each driving
    # direction. The separate IDs let the MATSim preparation patch assign distinct
    # one-way links while preserving the original crossing point as the common base.
    df["entry_interview_point_id"] = df["interview_point_id"].apply(make_entry_border_facility_id)
    df["exit_interview_point_id"] = df["interview_point_id"].apply(make_exit_border_facility_id)
    df["entry_interview_geometry_point"] = df["interview_geometry_point"]
    df["exit_interview_geometry_point"] = df["interview_geometry_point"]

    # Through agents have two real border anchors: where they enter Switzerland and
    # where they leave it. These are person-specific because the two coordinates can
    # come from different observed or projected border points.
    through_mask = df["label"] == "Through"
    df.loc[through_mask, "entry_interview_point_id"] = (
        df.loc[through_mask, "cross_border_person_id"].astype(str) + ENTRY_SUFFIX
    )
    df.loc[through_mask, "exit_interview_point_id"] = (
        df.loc[through_mask, "cross_border_person_id"].astype(str) + EXIT_SUFFIX
    )
    df.loc[through_mask, "entry_interview_geometry_point"] = gpd.GeoSeries(
        gpd.points_from_xy(df.loc[through_mask, "origin_x"], df.loc[through_mask, "origin_y"]),
        crs="EPSG:2056",
    ).values
    df.loc[through_mask, "exit_interview_geometry_point"] = gpd.GeoSeries(
        gpd.points_from_xy(df.loc[through_mask, "destination_x"], df.loc[through_mask, "destination_y"]),
        crs="EPSG:2056",
    ).values

    df = df[["cross_border_person_id", "mz_person_id", "label",
             "residence_x", "residence_y",
             "trip_mode", "trip_purpose", "destination_id",
             "origin_x", "origin_y", "destination_x", "destination_y",
             "is_border_point_projected",
             "interview_place", "interview_point_id", "interview_geometry_point",
             "entry_interview_point_id", "entry_interview_geometry_point",
             "exit_interview_point_id", "exit_interview_geometry_point",
             "origin_country", "destination_country", "origin_country_raw", "destination_country_raw"]]
    
    for col in ["mz_person_id", "residence_x", "residence_y", "destination_id",
             "origin_x", "origin_y", "destination_x", "destination_y"]:
        df[col] = df[col].astype(int)

    return df
