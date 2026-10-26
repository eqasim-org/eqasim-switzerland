"""Read-only pipeline input preparation and common survey/population semantics."""

import logging

import numpy as np
import pandas as pd

from data.utils import coerce_boolean_series
from .donors import validate_ids
from .features import prepare_features

logger = logging.getLogger("synpp")


def prepare_inputs(context, const):
    source = context.stage("data.microcensus.persons").copy()
    population = context.stage("synthesis.population.sampled").copy()
    trips = context.stage("data.microcensus.trips")[0].copy()
    validate_ids(source, "person_id", "Microcensus")
    validate_ids(population, "person_id", "Synthetic population")
    if trips.duplicated(["person_id", "trip_id"]).any():
        raise ValueError("Microcensus trips contain duplicate person/trip keys")
    # Preserve the V1 filter without importing or modifying any V1 code.
    employed_ids = set(source.loc[pd.to_numeric(source["employed"]).eq(1), "person_id"])
    work = trips["purpose"].eq("work") | trips["origin_purpose"].eq("work")
    inconsistent = set(trips.loc[work, "person_id"]) - employed_ids
    day = context.config("specific_day_scenario")
    if day in ("workday", "weekend"):
        day_mask = coerce_boolean_series(source[day], name=day)
    elif day in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"):
        day_mask = source["day"].eq(day)
    else:
        raise ValueError(f"Unimplemented day for scenario: {day}")
    border = coerce_boolean_series(source["is_crossing_the_border"], name="is_crossing_the_border")
    weights = pd.to_numeric(source["person_weight"], errors="coerce")
    valid_weight = np.isfinite(weights) & weights.gt(0)
    invalid_weights = int((day_mask & ~border & ~valid_weight).sum())
    if invalid_weights:
        logger.warning("V2 excluded %d donors with invalid/non-positive survey weights", invalid_weights)
    source = source.loc[day_mask & ~border & valid_weight & ~source["person_id"].isin(inconsistent)].copy()
    source["person_weight"] = weights.loc[source.index].astype(float)
    source = source.loc[pd.to_numeric(source["age"], errors="raise") >= const.MZ_AGE_THRESHOLD]
    chains = context.stage("data.microcensus.activity_chains")[["person_id", "activity_chain"]].copy()
    validate_ids(chains, "person_id", "Microcensus activity chains")
    source = source.merge(chains, on="person_id", how="left", validate="one_to_one")
    if source["activity_chain"].isna().any():
        raise ValueError("Eligible microcensus donors are missing activity chains")
    source["home_only"] = source["activity_chain"].eq("home")
    driver_ids = set(trips.loc[trips["mode"].isin(["car", "car_loop"]), "person_id"])
    source["uses_car_driver"] = source["person_id"].isin(driver_ids)
    source = source.rename(columns={"person_id": "mz_id"}).sort_values("mz_id").reset_index(drop=True)
    work_locations = context.stage("work_locations")[["person_id", "work_location_type", "commute_distance"]].copy()
    population = population.merge(work_locations, on="person_id", how="left", validate="one_to_one", sort=False)
    employed = pd.to_numeric(population["employed"], errors="raise")
    employed = employed.isin([1, 3]) if const.census == "are_synpop" else employed.eq(const.EMPLOYED)
    if population.loc[employed, "work_location_type"].isna().any():
        raise ValueError("Employed synthetic persons are missing work locations")
    collective = population.get("collective_housing_resident", pd.Series(False, index=population.index))
    population["home_only"] = coerce_boolean_series(collective, name="collective_housing_resident")
    source, population = prepare_features(source, population, const)
    relevant_trips = trips.loc[trips["person_id"].isin(source["mz_id"])].sort_values(["person_id", "trip_id"])
    return source, population, relevant_trips
