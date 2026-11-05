"""V1-compatible categories and residence/day eligibility for V3."""

import numpy as np
import pandas as pd

from data.utils import coerce_boolean_series
from synthesis.population.matching.matched_v1 import get_mz_persons
from synthesis.population.matching.commute import population_commutes
from synthesis.population.models.non_private import home_only_mask

ADULT = ["age_class", "sex", "car_availability", "employed", "employment_status", "ovgk",
         "population_density_class", "N_children_under_12", "sp_region", "broad_commute_distance_class",
         "commute_distance_class", "income_class", "canton_id", "work_location_type", "urban"]
ADULT_MANDATORY = ADULT[:10]
CHILD = ["age_class", "sex", "ovgk", "population_density_class", "sp_region", "urban", "canton_id"]
CHILD_MANDATORY = CHILD[:5]
YOUTH = ["age_class", "sex", "employed", "car_availability", "ovgk", "population_density_class",
         "employment_status", "sp_region", "broad_commute_distance_class", "urban", "work_location_type",
         "commute_distance_class", "canton_id"]
YOUTH_MANDATORY = YOUTH[:8]
HARD = {"child": ["age_class", "sex"], "youth": ["age_class", "sex", "employed", "car_availability"],
        "adult": ["sex", "car_availability", "employed", "employment_status"]}


def scenario_mask(persons, day):
    if day in ("workday", "weekend"):
        return coerce_boolean_series(persons[day], name=day)
    if day in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"):
        return persons.day.eq(day)
    raise ValueError(f"Unimplemented scenario day: {day}")


def prepare_inputs(context, const):
    donors = get_mz_persons(context).copy()
    valid = scenario_mask(donors, context.config("specific_day_scenario"))
    valid &= ~coerce_boolean_series(donors.is_crossing_the_border, name="is_crossing_the_border")
    donors = donors.loc[valid].rename(columns={"person_id": "mz_id"}).copy()
    roster = context.stage("synthesis.population.sampled").copy()
    if roster.person_id.duplicated().any() or donors.mz_id.duplicated().any():
        raise ValueError("V3 requires unique population and survey identifiers")
    held = home_only_mask(roster)
    population = population_commutes(roster.loc[~held].copy(), context.stage("work_locations"),
        context.stage("education_locations") if const.census == "statpop" else None)
    employed = pd.to_numeric(population.employed, errors="raise").eq(const.EMPLOYED)
    if population.loc[employed, "work_location_type"].isna().any():
        raise ValueError("Employed synthetic persons are missing work locations")
    population["employed"] = employed.astype(int)
    donors["employed"] = pd.to_numeric(donors.employed, errors="raise").eq(1).astype(int)
    donors["car_availability"] = pd.to_numeric(donors.car_availability, errors="raise").ne(const.CAR_AVAILABILITY_NEVER).astype(int)
    donors["N_children_under_12"] = donors.N_children_under_12.ne(0)
    chains = context.stage("data.microcensus.activity_chains")[["person_id", "activity_chain"]]
    donors = donors.merge(chains, left_on="mz_id", right_on="person_id", validate="one_to_one", how="left")
    if donors.activity_chain.isna().any():
        raise ValueError("Eligible survey donors are missing activity chains")
    donors["home_only"] = donors.activity_chain.eq("home")
    for frame in (donors, population):
        for column in ("sex", "employment_status", "car_availability"):
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
        distance = frame.commute_distance.fillna(-1)
        frame["commute_distance_class"] = np.digitize(distance, np.array([-10, 0, 1, 3, 6, 9, 12, 15, 20, 50, 1000]) * 1000)
        frame["broad_commute_distance_class"] = np.digitize(distance, np.array([-10, 0, 3, 8, 30, 1000]) * 1000)
        frame["work_location_type"] = frame.work_location_type.fillna("none").map({"none": 0, "fixed": 1, "remote": 2, "moving": 3}).fillna(-1).astype(int)
        if const.census == "statpop":
            bounds = np.asarray(context.config("matching_density_bounds"), dtype=float)
            if not (np.isfinite(bounds).all() and (bounds > 0).all() and (np.diff(bounds) > 0).all()):
                raise ValueError("V3 density bounds must be positive, finite and increasing")
            density = pd.to_numeric(frame.population_density, errors="coerce")
            frame["population_density_class"] = np.where(np.isfinite(density) & density.ge(0),
                np.digitize(density, bounds * np.pi * (const.POPULATION_DENSITY_RADIUS / 1000.) ** 2), -1)
            frame["age_class"] = np.digitize(frame.age, [6, 15, 18, 24, 40, 51, 65, 80])
            frame["income_class"] = frame.income_class.astype(int).map({0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3})
            frame["urban"] = frame.municipality_type.astype(str).eq("urbancore").astype(int)
            frame["ovgk"] = frame.ovgk.ne("None").astype(int)
            for column in ("sp_region", "canton_id"):
                frame[column] = frame[column].astype(int)
    return donors.sort_values("mz_id").reset_index(drop=True), population, roster
