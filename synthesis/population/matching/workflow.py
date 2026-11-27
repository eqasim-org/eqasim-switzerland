"""Common input and output workflow for the three whole-diary matchers.

Public entry points:
* preprocess: load the same roster and apply an algorithm-specific feature adapter.
* eligible_mask: apply the shared age and explicit home-only bypass rules.
* postprocess: restore bypasses and return mappings in original population order.

The donor-search algorithms remain in matched_v1/2/3. This module never changes
upstream frames. V2 preparation preserves complete households before deriving
household counts; only its matching target is restricted to eligible residents.
"""
from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd

from data.utils import coerce_boolean_series
from synthesis.population.models.non_private import home_only_mask
from . import opening_hours, spatial_feature
from .commute import population_commutes, write_diagnostics

logger = logging.getLogger("synpp")


@dataclass
class MatchingInputs:
    """Prepared algorithm features plus the unmodified roster for finalisation."""

    const: object
    roster: pd.DataFrame
    source: pd.DataFrame
    population: pd.DataFrame
    trips: object = None

    @property
    def eligible(self):
        return eligible_mask(self.population, self.const)


def load_population(context):
    """Load one independent copy of the complete roster, preserving its order."""
    roster = context.stage("synthesis.population.sampled").copy()
    _validate_ids(roster, "person_id", "Synthetic population")
    return roster


def eligible_mask(population, const):
    """Select donor recipients identically for V1, V2 and V3.

    Non-travel survey days are valid donors. Only explicit population bypasses
    and people below the survey age threshold are outside recipient matching.
    Collective residents who receive home-only *donors* are still recipients.
    """
    if const.census == "statpop":
        age = pd.to_numeric(population.age, errors="raise")
        if age.isna().any():
            raise ValueError("Matching requires non-missing recipient ages")
        eligible = age.ge(const.MZ_AGE_THRESHOLD)
    elif const.census == "are_synpop":
        age = pd.to_numeric(population.age_class, errors="raise")
        if age.isna().any():
            raise ValueError("Matching requires non-missing recipient age classes")
        eligible = age.gt(0)
    else:
        raise ValueError(f"Unsupported census for matching: {const.census}")
    return eligible & ~home_only_mask(population)


def preprocess(context, algorithm, const=None, *, include_regions=True):
    """Load a common roster, then dispatch feature preparation for the matcher."""

    # Resolve inputs once; adapters receive a private copy of the same roster.

    const = context.stage("data.constants") if const is None else const
    roster = load_population(context)
    eligible_mask(roster, const)
    adapters = {"v1": prepare_v1, "v2": prepare_v2, "v3": prepare_v3}
    if algorithm not in adapters:
        raise ValueError(f"Unknown matching algorithm: {algorithm}")
    source, population, trips = adapters[algorithm](context, const, roster)
    if const.census != "statpop" and context.config(spatial_feature.OPTION) == "activity_chain_cluster":
        raise ValueError("Daily-trip zones currently require STATPOP matching")
    source, population = spatial_feature.add_regions(context, source, population)
    _validate_ids(population, "person_id", "Prepared population")
    _validate_ids(source, "mz_id", "Survey donors")

    # Regions belong to prepared data; raw administrative identifiers are retained.

    if include_regions:
        source, population = opening_hours.add_regions(context, source, population)
    expected = set(roster.loc[eligible_mask(roster, const), "person_id"])
    actual = set(population.loc[eligible_mask(population, const), "person_id"])
    if expected != actual:
        raise RuntimeError(f"{algorithm} preparation changed the eligible recipient cohort")
    return MatchingInputs(const, roster, source, population, trips)


def initial_mapping(prepared):
    """Create one row per original resident; bypasses have donor ID -1."""
    roster = prepared.roster
    ids = ["person_id"] + (["household_id"] if "household_id" in roster else [])
    return roster[ids].reset_index(drop=True).assign(mz_person_id=-1)


def postprocess(prepared, mapping, removed):
    """Restore bypasses, record removals, and preserve the downstream tuple contract."""

    # Validate IDs before aligning results to the original, untouched population.

    roster = prepared.roster
    _validate_ids(mapping, "person_id", "Matching result")
    if not mapping.person_id.isin(roster.person_id).all():
        raise ValueError("Matcher returned recipients absent from the original roster")
    removed = set(removed)
    if not removed.issubset(set(roster.person_id)):
        raise ValueError("Matcher returned removal IDs absent from the original roster")
    ids = ["person_id"] + (["household_id"] if "household_id" in roster else [])
    result = roster[ids].copy()
    result["mz_person_id"] = pd.to_numeric(
        result.person_id.map(mapping.set_index("person_id").mz_person_id), errors="raise")
    observed = result.mz_person_id.dropna().to_numpy(dtype=float)
    if not np.isfinite(observed).all() or not np.equal(observed, np.floor(observed)).all():
        raise ValueError("Donor IDs must be finite integers or missing")
    eligible = eligible_mask(roster, prepared.const)
    held = home_only_mask(roster)

    # Missing active rows are explicit removals, including household-level removals.
    # This also reports rows removed by V1's merges but absent from its removal set.

    absent = ~roster.person_id.isin(mapping.person_id) & ~held
    unmatched = eligible & (result.mz_person_id.isna() | result.mz_person_id.lt(0))
    removed.update(roster.loc[absent | unmatched, "person_id"])
    removed.difference_update(roster.loc[held, "person_id"])

    # A failed eligible collective resident removes their entire household for
    # every algorithm. Include already-matched and below-age members as well.
    # Apply this after bypass restoration: explicit household removal takes
    # precedence over an individual home-only bypass in the same household.

    collective = coerce_boolean_series(roster.get("collective_housing_resident",
        pd.Series(False, index=roster.index)), name="collective_housing_resident")
    failed_collective = eligible & collective & roster.person_id.isin(removed)
    if failed_collective.any() and "household_id" in roster:
        households = roster.loc[failed_collective, "household_id"].dropna().unique()
        household_members = roster.household_id.isin(households)
        removed.update(roster.loc[household_members, "person_id"])
        logger.info("Shared matching removal: %d failed collective recipients; "
                    "%d households, %d household members removed",
                    int(failed_collective.sum()), len(households), int(household_members.sum()))
    result.loc[~eligible, "mz_person_id"] = -1
    result = result.loc[~result.person_id.isin(removed)].reset_index(drop=True)
    result["mz_person_id"] = pd.to_numeric(result.mz_person_id, errors="raise").fillna(-1).astype("int64")
    logger.info("Shared matching output: %d input residents, %d retained, %d removed; "
                "%d explicit home-only bypasses restored without donors",
                len(roster), len(result), len(removed), int((held & ~roster.person_id.isin(removed)).sum()))
    return result, removed


def _validate_ids(frame, column, label):
    if frame[column].isna().any() or frame[column].duplicated().any():
        raise ValueError(f"{label} requires unique, non-missing {column}")


def scenario_mask(persons, day):
    """Interpret day flags explicitly, including string booleans in older caches."""
    if day in ("workday", "weekend"):
        return coerce_boolean_series(persons[day], name=day)
    if day in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"):
        return persons.day.eq(day)
    raise ValueError(f"Unimplemented scenario day: {day}")


def survey_for_day(context):
    """Use the same cleaned domestic survey-day selection for all adapters."""
    source = load_survey(context).copy()
    _validate_ids(source, "person_id", "Microcensus")
    keep = scenario_mask(source, context.config("specific_day_scenario"))
    keep &= ~coerce_boolean_series(source.is_crossing_the_border, name="is_crossing_the_border")
    return source.loc[keep].copy()


# Survey cleaning is shared with existing callers of matched_v1.get_mz_persons.

def load_survey(context):
    """Clean inconsistent primary-work diaries and prepare usual commute predictors.

    Preserve genuine home-only days. The historical exclusion checks primary
    work purposes only; changing that policy belongs in this function.
    """

    # Read survey records without mutating the cached person/trip frames.

    df_persons = context.stage("data.microcensus.persons")
    df_trips   = context.stage("data.microcensus.trips")[0]

    # Exclude diaries with primary work when employment is inconsistent.
    employed_persons = set(df_persons[df_persons["employed"] == True]["person_id"])
    work_activities = ["work"]
    persons_with_work_purpose = set(df_trips[(df_trips["origin_purpose"].isin(work_activities)) |
                                             (df_trips["purpose"].isin(work_activities))]["person_id"])
    persons_to_remove = persons_with_work_purpose - employed_persons
    logger.info("Donor eligibility: excluded %d non-employed survey respondents whose diary contains work activities",
                len(persons_to_remove))

    df_persons = df_persons[~df_persons["person_id"].isin(persons_to_remove)]
    from synthesis.population.matching.commute import donor_commutes
    return donor_commutes(df_persons, context.stage("data.microcensus.commute")["education"],
                           seed=context.config("random_seed") + 1349)



def prepare_v1(context, const, roster):
    """Keep the recursive matcher's original categorical encodings and order."""

    # Clean donors, attach residence regions and infer usual commuting attributes.

    scenario_day = context.config("specific_day_scenario")

    df_source = survey_for_day(context).rename(columns={"person_id": "mz_id"})
    df_source["canton_id"] = df_source["canton_id"].astype(int)

    df_population = roster.copy()
    held_out_home = df_population.loc[home_only_mask(df_population)].copy()
    df_population = df_population.loc[~home_only_mask(df_population)].copy()
    logger.info("V1 start: %d recipients, %d home-only bypasses, %d day/border-eligible donors; day=%s; "
                "opening-hours region mask=%s", len(df_population) + len(held_out_home), len(held_out_home),
                len(df_source), scenario_day, context.config("consider_opening_hours"))
    df_population["sex"] = df_population["sex"].astype(int)

    # add commute distance and work location type
    from synthesis.population.matching.commute import population_commutes, write_diagnostics
    education = context.stage("education_locations") if const.census == "statpop" else None
    df_population = population_commutes(df_population, context.stage("work_locations"), education)
    write_diagnostics(df_source, df_population, context.path())

    assert df_population.loc[df_population.employed == const.EMPLOYED, "work_location_type"].notna().all(), "Some employed agents are missing commute distance"

    df_population[ "commute_distance"]   = df_population["commute_distance"].fillna(-1)
    df_population[ "work_location_type"] = df_population["work_location_type"].fillna("none")

    # transform work_location_type to int
    DICT_WORK_LOCATION_TYPE = {"none": 0, "fixed": 1, "remote": 2, "moving": 3}

    df_population["work_location_type"] = df_population["work_location_type"].map(DICT_WORK_LOCATION_TYPE)
    df_source["work_location_type"]     = df_source["work_location_type"].map(DICT_WORK_LOCATION_TYPE)

    # add commute distance classes
    COMMUTE_DISTANCE_BOUNDS = np.array([-10, 0, 1, 3, 6, 9, 12, 15, 20, 50, 1000]) * 1e3 # convert km -> m

    df_population["commute_distance_class"] = np.digitize(df_population["commute_distance"], COMMUTE_DISTANCE_BOUNDS)
    df_source["commute_distance_class"]     = np.digitize(df_source["commute_distance"], COMMUTE_DISTANCE_BOUNDS)

    BROAD_COMMUTE_DISTANCE_BOUNDS = np.array([-10, 0, 3, 8, 30, 1000]) * 1e3

    df_population["broad_commute_distance_class"] = np.digitize(df_population["commute_distance"], BROAD_COMMUTE_DISTANCE_BOUNDS)
    df_source["broad_commute_distance_class"]     = np.digitize(df_source["commute_distance"], BROAD_COMMUTE_DISTANCE_BOUNDS)

    # this is now necessary to include, to make sure employement is well matched, otherwise we get errors in work location assignment
    df_source["employed"]     = df_source["employed"].astype(int)
    df_population["employed"] = (df_population["employed"] == const.EMPLOYED).astype(int)

    # Model stages use string encodings for categorical features internally.
    # Statistical matching compares values directly, so normalize both sides.
    df_source["employment_status"] = pd.to_numeric(
        df_source["employment_status"], errors="raise").astype("int64")
    df_population["employment_status"] = pd.to_numeric(
        df_population["employment_status"], errors="raise").astype("int64")

    df_source["N_children_under_12"] = df_source["N_children_under_12"].ne(0)  # presence of children under 12

    df_source["sex"] = df_source["sex"].astype(int)

    var_raw = pd.to_numeric(df_source["car_availability"], errors="coerce")
    df_source["car_availability"] = np.where(var_raw == const.CAR_AVAILABILITY_NEVER, 0, 1).astype("int64")

    # checkig if any duplicates
    assert df_population["person_id"].duplicated().sum()==0, "Duplicate person_id found in population dataframe. Please ensure person_id is unique for each individual in the population."
    assert df_source["mz_id"].duplicated().sum()==0, "Duplicate mz_id found in source dataframe. Please ensure mz_id is unique for each individual in the source."

    # Construct exactly the categorical bins used by the recursive donor search.

    if const.census == "statpop":
        # Density is already a resident count within 2.5 km on both sides.
        # Configured density boundaries are in residents/km²; -1 is unknown.
        density_bounds = np.asarray(context.config("matching_density_bounds"), dtype=float)
        density_bounds = density_bounds * np.pi * (const.POPULATION_DENSITY_RADIUS / 1000.) ** 2
        for frame in (df_source, df_population):
            density = pd.to_numeric(frame["population_density"], errors="coerce")
            frame["population_density_class"] = np.where(
                np.isfinite(density) & density.ge(0), np.digitize(density, density_bounds), -1)

        # age class
        AGE_CLASS_UPPER_BOUNDS = [6, 15, 18, 24, 40, 51, 65, 80]
        df_population["age_class"] = np.digitize(df_population["age"], AGE_CLASS_UPPER_BOUNDS)
        df_source["age_class"]     = np.digitize(df_source["age"], AGE_CLASS_UPPER_BOUNDS)

        # income classes
        df_population["income_class"] = df_population["income_class"].astype(int)
        df_source["income_class"]     = df_source["income_class"].astype(int)

        INCOME_CLASSIFICATION = {0: 1, 1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3} # map 6 income classes to 3 classes (low, medium, high)

        df_population["income_class"] = df_population["income_class"].map(INCOME_CLASSIFICATION)
        df_source["income_class"]     = df_source["income_class"].map(INCOME_CLASSIFICATION)

        # further cleaning
        df_source["household_size_class"] = df_source["household_size_class"].clip(upper=2)
        df_population["household_size"]   = df_population["household_size"].clip(upper=2)

        df_population["number_of_cars_class"] = df_population["number_of_cars_class"].clip(upper=3)
        df_source["number_of_cars_class"]     = df_source["number_of_cars_class"].clip(upper=3)
        df_population["marital_status"]    = df_population["marital_status"].astype(int)
        df_population["car_availability"]  = df_population["car_availability"].astype(int)
        df_population["municipality_type"] = df_population["municipality_type"].astype(str)
        df_population["urban"]             = (df_population["municipality_type"].isin(["urbancore"])).astype(int)
        df_population["sp_region"]         = df_population["sp_region"].astype(int)
        df_population["canton_id"]         = df_population["canton_id"].astype(int)
        df_population["ovgk"]              = (df_population["ovgk"] != "None").astype(int)

        df_source["municipality_type"] = df_source["municipality_type"].astype(str)
        df_source["urban"]             = (df_source["municipality_type"].isin(["urbancore"])).astype(int)
        df_source["sp_region"]         = df_source["sp_region"].astype(int)
        df_source["canton_id"]         = df_source["canton_id"].astype(int)
        df_source["ovgk"]              = (df_source["ovgk"] != "None").astype(int)

    return df_source, df_population, None


def prepare_v2(context, const, roster):
    """Preserve continuous/count inputs and the full households used by V2."""
    from .V2.features import prepare_features

    # Reuse V1 cleaning and commute imputation BEFORE day/border filtering.
    # Work on a consumer-owned copy of the shared helper result.
    source = survey_for_day(context)
    population = roster.copy()
    trips = context.stage("data.microcensus.trips")[0].copy()
    if trips.duplicated(["person_id", "trip_id"]).any():
        raise ValueError("Microcensus trips contain duplicate person/trip keys")
    weights = pd.to_numeric(source["person_weight"], errors="coerce")
    valid_weight = np.isfinite(weights) & weights.gt(0)
    invalid_weights = int((~valid_weight).sum())
    if invalid_weights:
        logger.warning("V2 excluded %d donors with invalid/non-positive survey weights", invalid_weights)
    source = source.loc[valid_weight].copy()
    source["person_weight"] = weights.loc[source.index].astype(float)
    source = source.loc[pd.to_numeric(source["age"], errors="raise") >= const.MZ_AGE_THRESHOLD]
    chains = context.stage("data.microcensus.activity_chains")[["person_id", "activity_chain"]].copy()
    _validate_ids(chains, "person_id", "Microcensus activity chains")
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


def prepare_v3(context, const, roster):
    """Build the existing V3 categorical features for order-independent search."""

    # Prepare the shared domestic survey and active population anchors.

    donors = survey_for_day(context).rename(columns={"person_id": "mz_id"})
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
    # Apply V3-specific types and categories without altering the raw roster.

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
    return donors.sort_values("mz_id").reset_index(drop=True), population, None
