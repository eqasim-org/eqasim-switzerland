import logging

import numpy as np

logger = logging.getLogger("synpp")


def configure(context):
    context.config("random_seed")

    context.stage("data.cross_border.destinations")
    context.stage("data.cross_border.network_projection")
    context.stage("data.microcensus.persons")


PERSON_FIELDS = ["person_id", "age", "car_availability", "employed", "driving_license", "sex", 
                 "home_x", "home_y",
                 "subscriptions_ga", "subscriptions_halbtax", "subscriptions_verbund", "subscriptions_strecke",
                 "household_id", "is_car_passenger", 
                 "statpop_person_id", "statpop_household_id", "mz_person_id", "mz_head_id", 
                 "has_walk_loop_trip", "has_car_loop_trip", "has_car_passenger_loop_trip", "has_pt_loop_trip", "has_bike_loop_trip",
                 "income_class",
                 "number_of_cars_class", "number_of_bikes_class",
                 "origin_country", "destination_country", "origin_country_raw", "destination_country_raw"]


def execute(context):
    df         = context.stage("data.cross_border.destinations")
    df_refined = context.stage("data.cross_border.network_projection")
    mz_persons = context.stage("data.microcensus.persons")

    population = df[["cross_border_person_id", "mz_person_id", "origin_x", "origin_y",
                      "residence_x", "residence_y", "origin_is_projected", "trip_mode",
                      "origin_country", "destination_country",
                      "origin_country_raw", "destination_country_raw"]].copy()

    population["household_id"] = population["cross_border_person_id"].values

    # data.cross_border.network_projection refines the teleported ("projected")
    # origin of cross-border respondents whose real home was too far from the
    # Swiss border onto a real point on the MATSim car network, closer to an
    # actual road entry point than the raw survey interview place coordinate
    # (see that stage's docstring). Use the refined point as this population's
    # home location wherever one is available, falling back to the raw
    # interview place coordinate otherwise (e.g. if network_projection found
    # no origin-projected records, or ran before some rows existed).
    #
    # network_projection's output uses generic real_x/y + interview_x/y
    # column names (it refines both "From-To"/"Through" origins and "Through"
    # destinations with the same algorithm) - alias this population's own
    # origin-side columns to those names before merging.
    #
    # Merge keys are cast to int first to match data.cross_border.destinations'
    # own int cast of these same columns - both sides trace back to the same
    # underlying data.cross_border.generate_od float values, left untouched in
    # between, so the truncation is identical and the merge is exact.
    merge_keys = ["real_x", "real_y", "interview_x", "interview_y"]
    population = population.rename(columns = {
        "residence_x": "real_x", "residence_y": "real_y",
        "origin_x": "interview_x", "origin_y": "interview_y",
    })
    if len(df_refined) > 0:
        df_refined = df_refined[merge_keys + ["refined_x", "refined_y"]].copy()
        df_refined[merge_keys] = df_refined[merge_keys].astype(int)
        # Two distinct float combinations could in principle truncate to the
        # same int key - drop_duplicates keeps this a clean one-to-many-safe
        # left join (any one of the near-identical refined points is an
        # acceptable choice in that rare case) instead of silently
        # multiplying population rows.
        df_refined = df_refined.drop_duplicates(subset = merge_keys, keep = "first")
        population = population.merge(df_refined, on = merge_keys, how = "left")
    else:
        population["refined_x"] = np.nan
        population["refined_y"] = np.nan

    has_refined_origin = population["refined_x"].notna()
    use_refined_origin  = population["origin_is_projected"] & has_refined_origin

    # network_projection deliberately never refines PT trips (see its
    # docstring) - excluded here too so they don't inflate this warning,
    # which is meant to flag unexpected gaps (e.g. no network route found for
    # a projected car trip), not this by-design one.
    is_car_trip = population["trip_mode"].isin(["car", "car_passenger"])
    missing_refined_origin = population["origin_is_projected"] & is_car_trip & ~has_refined_origin

    if missing_refined_origin.any():
        logger.warning(
            "%d origin-projected car/car_passenger cross-border agents had no matching "
            "data.cross_border.network_projection entry and fall back to the raw (unrefined) interview "
            "place coordinate as their home location.",
            missing_refined_origin.sum(),
        )

    population["home_x"] = np.where(use_refined_origin, population["refined_x"], population["interview_x"])
    population["home_y"] = np.where(use_refined_origin, population["refined_y"], population["interview_y"])

    population["subscriptions_ga"]      = False
    population["subscriptions_halbtax"] = False

    population["has_walk_loop_trip"]          = False
    population["has_car_passenger_loop_trip"] = False
    population["has_car_loop_trip"]           = False
    population["has_bike_loop_trip"]          = False
    population["has_pt_loop_trip"]            = False

    population["mz_head_id"]           = population["mz_person_id"].values
    population["statpop_person_id"]    = 0
    population["statpop_household_id"] = 0

    mz_persons = mz_persons[["person_id", "age", "sex", "car_availability",
                             "employed", "driving_license", 
                             "subscriptions_verbund", "subscriptions_strecke",
                             "is_car_passenger", "income_class",
                             "number_of_cars_class", "number_of_bikes_class"]]
    
    population = population.merge(mz_persons,
                                  how = "left",
                                  left_on = "mz_person_id",
                                  right_on = "person_id")
    
    del population["person_id"]
    population["person_id"] = population["cross_border_person_id"]
    
    population = population[PERSON_FIELDS]

    return population