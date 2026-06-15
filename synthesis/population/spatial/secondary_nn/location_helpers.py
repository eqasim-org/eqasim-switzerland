import logging
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import h3
from numba import njit, prange
from .h3 import H3_LEVEL_NAMES
from .hierarchical_utils import SECONDARY_ACTIVITIES, PRIMARY_ACTIVITIES
from matsim.scenario.population import HOME_DESTINATION_ID

logger = logging.getLogger("synpp: location_helpers")


SECONDARY_SET = set(SECONDARY_ACTIVITIES)
PRIMARY_SET = set(PRIMARY_ACTIVITIES)


def _prepare_primary_locations(context):
    df_home = context.stage("synthesis.population.spatial.home.locations").rename(columns={"geometry": "home"})
    df_work, df_education = context.stage("synthesis.population.spatial.primary.locations")
    df_work = df_work.rename(columns={"geometry": "work", "destination_id": "work_destination_id"})
    df_education = df_education.rename(columns={"geometry": "education", "destination_id": "education_destination_id"})

    df_locations = context.stage("synthesis.population.enriched")[["person_id", "household_id"]].copy()
    df_locations = df_locations.merge(df_home[["household_id", "home"]], how="left", on="household_id")
    df_locations = df_locations.merge(df_work[["person_id", "work", "work_destination_id"]], how="left", on="person_id")
    df_locations = df_locations.merge(df_education[["person_id", "education", "education_destination_id"]], how="left", on="person_id")

    df_locations["home_destination_id"] = HOME_DESTINATION_ID
    df_locations[["work_destination_id","education_destination_id", "home_destination_id"]] = df_locations[["work_destination_id","education_destination_id", "home_destination_id"]].fillna(0)
    return df_locations[["person_id", "home", "work", "education", "work_destination_id","education_destination_id", "home_destination_id"]
                        ].sort_values(by="person_id").reset_index(drop=True)


def _prepare_person_attributes(context):
    df = context.stage("synthesis.population.enriched").copy()
    required = ["person_id", "age", "sex", "employed", "income_class", "car_availability"]
    out = df[required].copy()
    out["employed"] = (pd.to_numeric(out["employed"], errors="coerce").fillna(0) == 1).astype(int)
    return out[["person_id", "age", "sex", "employed", "income_class", "car_availability"]]


@njit(fastmath=True)
def _euclidean(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return float(np.sqrt(dx * dx + dy * dy))

_ = _euclidean(0.5,0.5,1.0,1.0)  # warm up numba

def _build_level_attributes(h3_data, h3_geo_level0, all_h3):
    if all_h3 is None:
        raise RuntimeError("Expected all_h3 to be provided")

    df = h3_geo_level0.set_index("h3_index")

    # ensure all requested H3 indices exist
    missing = set(all_h3) - set(df.index)
    if missing:
        raise RuntimeError(f"Missing H3 indices: {list(missing)[:5]} ...")

    # reindex once 
    df = df.reindex(all_h3)

    # extract centroid coordinates (vectorized)
    df["x"] = df["centroid"].x
    df["y"] = df["centroid"].y

    # check for nans
    cols = ["num_statent", "employees", "urban_core", "urban", "education", "shop", "leisure",
            "sport", "gastronomy", "accommodation", "cultural", "ovgk_share_a", "ovgk_share_b", "ovgk_share_c", "ovgk_share_d", "ovgk_share_none", "outside_fraction"]
    assert df[cols].isna().sum().sum() == 0, "Found NaNs in coarse attributes, please check the data preparation steps."

    # build dict
    attrs = df[["x", "y"] + cols].to_dict(orient="index")

    return attrs


def build_probability(num_employees, ovgk_none, car_available, alpha=0.7):
    weight = np.asarray(num_employees) ** alpha
    if not car_available:
        weight[ovgk_none] *= 0.01  # penalize locations with no OVGK if no car available
    total_weight = weight.sum()
    return weight / total_weight if total_weight > 0 else np.ones_like(weight) / len(weight)



def _prepare_destination_level2_index(context, h3_data, h3_geo):
    df_dest = h3_data["destinations"][["destination_id", H3_LEVEL_NAMES[1], H3_LEVEL_NAMES[-1]]]
    
    offer_cols = [f"offers_{p}" for p in SECONDARY_ACTIVITIES]
    df_dest_attributes = context.stage("synthesis.population.destinations")[["destination_id", "number_employees", "ovgk", "geometry", *offer_cols]]
    df_dest = df_dest.merge(df_dest_attributes, on="destination_id", how="left")

    # Validate columns    
    missing = [c for c in offer_cols if c not in df_dest.columns]
    if missing:
        raise RuntimeError(f"Missing columns in destinations data: {missing}")

    # Reusable full-frame arrays (boolean/numeric only — indexed with boolean mask below).
    no_ovgk = (df_dest["ovgk"].to_numpy() == "None")
    employees = df_dest["number_employees"].to_numpy()
    dest_ids = df_dest["destination_id"].to_numpy()
    geoms = df_dest["geometry"].values

    # Convert offer columns once
    offer_masks = {purpose: df_dest[f"offers_{purpose}"].fillna(False).to_numpy(dtype=bool) for purpose in SECONDARY_ACTIVITIES}

    index = {purpose: {} for purpose in SECONDARY_ACTIVITIES}
    fallback = {purpose: {} for purpose in SECONDARY_ACTIVITIES}
    index_xy = {purpose: {} for purpose in SECONDARY_ACTIVITIES}
    fallback_xy = {purpose: {} for purpose in SECONDARY_ACTIVITIES}
    car_available_probabilities = {purpose: {} for purpose in SECONDARY_ACTIVITIES}
    no_car_available_probabilities = {purpose: {} for purpose in SECONDARY_ACTIVITIES}

    for purpose in SECONDARY_ACTIVITIES:
        mask = offer_masks[purpose]
        sub = df_dest.loc[mask].reset_index(drop=True)
        sub_no_ovgk = no_ovgk[mask]
        sub_employees = employees[mask]
        sub_dest_ids = dest_ids[mask]
        sub_geoms = geoms[mask]

        # ---- finest level ----
        for level_2, idx in sub.groupby(H3_LEVEL_NAMES[-1]).indices.items():
            g_l2 = sub_geoms[idx]
            index[purpose][level_2] = list(zip(sub_dest_ids[idx], g_l2))
            index_xy[purpose][level_2] = np.stack(
                [np.array([g.x for g in g_l2], dtype=np.float64),
                 np.array([g.y for g in g_l2], dtype=np.float64)], axis=1)
            emp = sub_employees[idx]
            nov = sub_no_ovgk[idx]
            car_available_probabilities[purpose][level_2] = build_probability(emp, nov, car_available=True)
            no_car_available_probabilities[purpose][level_2] = build_probability(emp, nov, car_available=False)

        # ---- middle level ----
        for level_1, idx in sub.groupby(H3_LEVEL_NAMES[1]).indices.items():
            g_l1 = sub_geoms[idx]
            fallback[purpose][level_1] = list(zip(sub_dest_ids[idx], g_l1))
            fallback_xy[purpose][level_1] = np.stack(
                [np.array([g.x for g in g_l1], dtype=np.float64),
                 np.array([g.y for g in g_l1], dtype=np.float64)], axis=1)
            emp = sub_employees[idx]
            nov = sub_no_ovgk[idx]
            car_available_probabilities[purpose][level_1] = build_probability(emp, nov, car_available=True)
            no_car_available_probabilities[purpose][level_1] = build_probability(emp, nov, car_available=False)

    return (index, fallback, car_available_probabilities, no_car_available_probabilities, index_xy, fallback_xy)


def _reverse_tree(h3_tree):
    l1_to_siblings = {}
    for l1_dict in h3_tree.values():
        l1_list = list(l1_dict.keys())
        for l1 in l1_list:
            l1_to_siblings[l1] = [s for s in l1_list if s != l1]
    return l1_to_siblings




def _get_first_location(grp, home_x, home_y, work_x, work_y, edu_x, edu_y, has_work, has_education,
                        work_destination_id, education_destination_id, home_destination_id):
    # Cache first row (much cheaper than repeated .iat)
    first = grp.iloc[0]
    first_preceding = first["preceding_purpose"]
    added_a_trip = False

    if first_preceding not in PRIMARY_SET:
        # For synthetic prepend, always assume the person starts from home.
        first_primary = "home"
        added_a_trip = True
        new_row = {
            'person_id': first["person_id"],
            'mz_person_id': first["mz_person_id"],
            'trip_id': first["trip_id"] - 1,
            'trip_index': first["trip_index"] - 1,
            'departure_time': first["departure_time"],
            'arrival_time': first["arrival_time"],
            'preceding_purpose': first_primary,
            'following_purpose': first_preceding,
            'trip_duration': first["trip_duration"],
            'mode': first["mode"],
            'daily_longest_distance_from_home': first["daily_longest_distance_from_home"],
            'daily_crowfly_total': first["daily_crowfly_total"],
            'crowfly_consumed_before_trip': 0.0,
            'trip_position_class': 0.0,
            'daily_longest_distance_from_work': first["daily_longest_distance_from_work"],
            'departure_time_normalized': max(0, first["departure_time_normalized"]-8.0/24.0),
            "activity_duration_h": 8.0,  # assume 8h duration for the first activity, consistent with mz_chains.py
            "target_distance": first["trip_origin_distance_from_home"],
            "activity_chain": first["activity_chain"]
        }

        # Faster than concat for single row prepend
        grp = pd.concat(
            [pd.DataFrame([new_row]), grp],
            ignore_index=True,
            copy=False
        )

        # Update first_preceding after modification
        first_preceding = first_primary

    # Determine starting location
    current_x, current_y = home_x, home_y
    origin_id = home_destination_id

    if first_preceding == "work" and has_work:
        current_x, current_y = work_x, work_y
        origin_id = work_destination_id

    elif first_preceding == "education" and has_education:
        current_x, current_y = edu_x, edu_y
        origin_id = education_destination_id

    return grp, (current_x, current_y), added_a_trip, origin_id