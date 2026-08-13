import numpy as np
import pandas as pd
import geopandas as gpd
import logging

from data.cross_border.generate_od import sjoin_within_unique

logger = logging.getLogger("synpp")

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

    merged = sjoin_within_unique(destinations.geometry, municipalities)
    
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

    # One shared, seeded RNG rather than the same random_state for every group:
    # "work", "work_secondary" and "other" are offered by every STATENT entry,
    # so with a fixed seed they drew the very same facilities in the same order.
    rng = np.random.RandomState(context.config("random_seed"))

    for purpose in ["work", "work_secondary", "other", "education", "leisure", "shop"]:
        mask_purpose         = (df["trip_purpose"] == purpose)
        mask_statent_purpose = statent[f"offers_{purpose}"]

        # NaN ids (destination outside every municipality polygon, e.g. on a
        # lake) match no row and no candidate; they are handled by the
        # nationwide fallback below.
        mun_ids = df.loc[mask_purpose, "destination_municipality_id"].dropna().unique()

        for mun_id in mun_ids:
            mask_mun = df["destination_municipality_id"] == mun_id
            mask     = mask_purpose & mask_mun

            mask_statent_mun = statent["municipality_id"] == mun_id
            mask_statent     = mask_statent_purpose & mask_statent_mun

            candidates = statent[mask_statent]
            N_sample   = np.sum(mask)

            if len(candidates) == 0:
                candidates = statent[mask_statent_mun]

            sampled = sample_destinations(candidates, N_sample, rng)

            if sampled is None:
                continue  # nothing available in this municipality, see below

            df.loc[mask, "destination_id"] = sampled["destination_id"].values
            df.loc[mask, "destination_x"]  = sampled["destination_x"].values
            df.loc[mask, "destination_y"]  = sampled["destination_y"].values

    # Fallback for the rows the loop could not serve: no municipality could be
    # determined for the surveyed destination, or the municipality holds no
    # usable STATENT entry. Rather than dropping these agents (or crashing on
    # an empty candidate set), give them the closest facility that offers their
    # purpose, which keeps them in the right region of the country.
    missing = df["destination_id"].isna() if "destination_id" in df.columns else pd.Series(True, index = df.index)

    if missing.any():
        logger.warning(
            "No destination could be sampled from the recorded municipality for %d of %d cross-border activities; falling back to the nearest facility offering the same purpose.",
            int(missing.sum()), len(df),
        )
        df = assign_nearest_destinations(df, missing, statent, destinations)

    df = pd.concat([df, df_through])

    df["destination_id"] = df["destination_id"].fillna(-1)

    # All entries with no destination assigned have label = Through, which means that the corresponding agents are only crossing Switzerland without stopping
    # in the country. And as STATENT only covers Switzerland, we cannot assign real destinations to the people crossing the country.

    assert (
        df.loc[df["destination_id"] == -1, "label"] == "Through"
    ).all(), "Found rows with destination_id = -1 and label != 'Through'"

    df = df[["cross_border_person_id", "mz_person_id", "label",
             "residence_x", "residence_y",
             "trip_mode", "trip_purpose", "destination_id",
             "origin_x", "origin_y", "destination_x", "destination_y",
             "is_border_point_projected", "origin_is_projected", "destination_is_projected",
             "origin_point_id", "destination_point_id",
             "interview_place", "interview_point_id", "interview_geometry_point",
             "origin_country", "destination_country", "origin_country_raw", "destination_country_raw"]]
    
    # destination_id is a canonical id string for real destinations (e.g.
    # "CH_STATENT_..."), or the int sentinel -1 for "Through" trips - keep it
    # out of the blanket int cast below.
    for col in ["mz_person_id", "residence_x", "residence_y",
             "origin_x", "origin_y", "destination_x", "destination_y"]:
        df[col] = df[col].astype(int)

    return df


def sample_destinations(candidates, n_sample, rng):
    """
    Draws n_sample facilities (with replacement) from candidates, weighted by
    their number of employees. Returns None when there is nothing to draw from:
    pandas raises "weights sum to zero" rather than returning an empty frame,
    both for an empty candidate set and for a set whose employee counts are all
    zero, so those cases are caught here and left to the caller's fallback.
    """

    if n_sample == 0 or len(candidates) == 0:
        return None

    weights = candidates["number_employees"]

    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = None  # uniform draw instead of failing on degenerate weights

    return candidates.sample(n = n_sample, random_state = rng, replace = True, weights = weights)


def assign_nearest_destinations(df, missing, statent, destination_points):
    """
    Gives every row flagged in `missing` the closest facility that offers its
    trip purpose, measured from the destination point recorded in the survey.
    Used when the municipality-based draw cannot serve a row, so that the agent
    still ends up in a plausible place instead of being dropped.
    """

    df = df.copy()

    # The column is float64 while it only holds NaN, and the ids are strings.
    if "destination_id" in df.columns:
        df["destination_id"] = df["destination_id"].astype(object)

    for purpose in df.loc[missing, "trip_purpose"].dropna().unique():
        mask = missing & (df["trip_purpose"] == purpose)

        offers_column = f"offers_{purpose}"
        candidates    = statent[statent[offers_column]] if offers_column in statent.columns else statent

        if len(candidates) == 0:
            candidates = statent

        candidates = gpd.GeoDataFrame(
            candidates[["destination_id", "destination_x", "destination_y"]],
            geometry = gpd.points_from_xy(candidates["destination_x"], candidates["destination_y"]),
            crs = "EPSG:2056",
        )

        nearest = gpd.sjoin_nearest(destination_points.loc[mask[mask].index], candidates, how = "left")
        nearest = nearest[~nearest.index.duplicated(keep = "first")]  # ties return one row each

        df.loc[mask, "destination_id"] = nearest["destination_id"].values
        df.loc[mask, "destination_x"]  = nearest["destination_x"].values
        df.loc[mask, "destination_y"]  = nearest["destination_y"].values

    return df