"""Read-only pipeline input preparation and common survey/population semantics."""

import logging

import numpy as np
import pandas as pd

from data.utils import coerce_boolean_series
from synthesis.population.models.non_private import home_only_mask
from synthesis.population.matching.matched_v1 import get_mz_persons
from synthesis.population.matching.commute import population_commutes, write_diagnostics
from .donors import validate_ids
from .features import prepare_features

logger = logging.getLogger("synpp")


def prepare_inputs(context, const):
    # Reuse V1 cleaning and commute imputation BEFORE day/border filtering.
    # Work on a consumer-owned copy of the shared helper result.
    source = get_mz_persons(context).copy()
    population = context.stage("synthesis.population.sampled").copy()
    trips = context.stage("data.microcensus.trips")[0].copy()
    validate_ids(source, "person_id", "Microcensus")
    validate_ids(population, "person_id", "Synthetic population")
    if trips.duplicated(["person_id", "trip_id"]).any():
        raise ValueError("Microcensus trips contain duplicate person/trip keys")
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
    source = source.loc[day_mask & ~border & valid_weight].copy()
    source["person_weight"] = weights.loc[source.index].astype(float)
    source = source.loc[pd.to_numeric(source["age"], errors="raise") >= const.MZ_AGE_THRESHOLD]
    chains = context.stage("data.microcensus.activity_chains")[["person_id", "activity_chain"]].copy()
    validate_ids(chains, "person_id", "Microcensus activity chains")
    source = source.merge(chains, on="person_id", how="left", validate="one_to_one")
    if source["activity_chain"].isna().any():
        raise ValueError("Eligible microcensus donors are missing activity chains")
    source["home_only"] = source["activity_chain"].eq("home")
    # Observed non-travel days remain valid training examples and ordinary donors.
    driver_ids = set(trips.loc[trips["mode"].isin(["car", "car_loop"]), "person_id"])
    source["uses_car_driver"] = source["person_id"].isin(driver_ids)
    source = source.rename(columns={"person_id": "mz_id"}).sort_values("mz_id").reset_index(drop=True)
    work_locations = context.stage("work_locations")[["person_id", "work_location_type", "commute_distance"]].copy()
    # Match on the longer available work/education commute, as in V1.
    education = context.stage("education_locations") if const.census == "statpop" else None
    # V1 holds out the explicit home-only scenario before anchor preparation.
    # Restore those rows afterward to reconstruct counts on full households.
    held = home_only_mask(population)
    active = population_commutes(population.loc[~held].copy(), work_locations, education)
    write_diagnostics(source, active, context.path())
    population = (pd.concat([active, population.loc[held]], ignore_index=True)
                  .set_index("person_id").loc[population["person_id"]].reset_index())
    employed = pd.to_numeric(population["employed"], errors="raise")
    employed = employed.isin([1, 3]) if const.census == "are_synpop" else employed.eq(const.EMPLOYED)
    population["home_only"] = home_only_mask(population)
    if population.loc[employed & ~population["home_only"], "work_location_type"].isna().any():
        raise ValueError("Employed synthetic persons are missing work locations")
    # Keep the full household roster for feature reconstruction; the stage
    # excludes explicitly home-only residents from prediction, as in V1.
    source, population = prepare_features(source, population, const)
    relevant_trips = trips.loc[trips["person_id"].isin(source["mz_id"])].sort_values(["person_id", "trip_id"])
    return source, population, relevant_trips
