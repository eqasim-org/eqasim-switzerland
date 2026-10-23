"""Calibrate border demand without using vehicle occupancy as vehicle volume.

The two OD stages call these helpers separately for Swiss and foreign residents.
All weights represent one survey-day direction; balancing the opposite direction
is a separate activity-chain decision, not another calibration factor.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("synpp")

# These are support checks, not targets fitted to a particular crossing. Smaller
# cells share a residual pooled factor rather than an unstable local ratio.
MIN_OBSERVATIONS = 10
MIN_EFFECTIVE_OBSERVATIONS = 5


def calibrate_demand(reference, observations):
    """Return calibrated observations and a ledger including unsupported targets.

    Car (including motorcycles) uses 2015 vehicle totals by interview-place ID.
    Rail/coach keep their existing person-based national OD/purpose controls.
    No coordinate-availability filter may precede this function.
    """
    result = observations.copy()
    result["uncalibrated_weight"] = result["weight"]
    result["scaling_factor"] = 1.0
    result["calibration_source"] = "uncontrolled"
    records = []
    for frame in (reference, observations):
        if (~np.isfinite(frame["weight"]) | frame["weight"].lt(0)).any():
            raise ValueError("Cross-border survey weights must be finite and nonnegative.")

    # Keep all road-driver mass, including observations with impossible occupancy.
    old = reference.loc[reference["trip_mode"].eq("car")]
    new = result.loc[result["trip_mode"].eq("car")]
    target = old.groupby("interview_place_id")["weight"].sum()
    grouped = new.groupby("interview_place_id")["weight"]
    current = grouped.sum()
    squares = new.assign(squared = new["weight"] ** 2).groupby("interview_place_id")["squared"].sum()
    count = new.loc[new["weight"].gt(0)].groupby("interview_place_id").size()
    effective = current ** 2 / squares.replace(0, np.nan)
    supported = current.index[(count.reindex(current.index, fill_value = 0) >= MIN_OBSERVATIONS)
                              & (effective >= MIN_EFFECTIVE_OBSERVATIONS)
                              & current.index.isin(target.index)]

    # Pool only the remaining groups. Well-supported local controls stay exact;
    # missing 2021 groups remain visible in the ledger, never silently disappear.
    remaining_target = target.drop(supported, errors = "ignore").sum()
    remaining_current = current.drop(supported, errors = "ignore").sum()
    pooled_factor = remaining_target / remaining_current if remaining_current > 0 else np.nan
    for group in target.index.union(current.index):
        raw = float(current.get(group, 0))
        wanted = float(target.get(group, 0))
        if group in supported:
            factor, source = wanted / raw, "interview_group_vehicle"
        elif group not in target.index and remaining_target == 0 and raw > 0:
            # Absence from the reference is not an observed zero. With no
            # residual control, use the national ratio and expose the gap.
            factor = target.sum() / current.sum() if current.sum() > 0 else 1.0
            source = "no_2015_control_national_vehicle"
        elif raw > 0 and np.isfinite(pooled_factor):
            factor, source = pooled_factor, "pooled_residual_vehicle"
        else:
            factor, source = np.nan, "no_2021_support"
        mask = result["trip_mode"].eq("car") & result["interview_place_id"].eq(group)
        result.loc[mask, "scaling_factor"] = factor
        result.loc[mask, "calibration_source"] = source
        records.append(dict(interview_place_id = group, observations_2021 = int(count.get(group, 0)),
                            effective_observations_2021 = float(effective.get(group, 0)),
                            entries_2015 = wanted, entries_2021 = raw,
                            calibrated_entries = raw * factor if raw else 0.0,
                            scaling_factor = factor, calibration_source = source))

    # PT controls are outside this road-volume correction. Preserve their unit
    # and former strata rather than mixing coach vehicle counts with train users.
    keys = ["trip_mode", "trip_purpose", "origin_country", "destination_country"]
    old_pt = reference.loc[~reference["trip_mode"].eq("car")].groupby(keys)["group_weight"].sum()
    new_pt = result.loc[~result["trip_mode"].eq("car")].groupby(keys)["group_weight"].sum()
    factors = (old_pt.reindex(new_pt.index) / new_pt.replace(0, np.nan)).fillna(1.0)
    mask = ~result["trip_mode"].eq("car")
    if mask.any():
        result.loc[mask, "scaling_factor"] = factors.reindex(pd.MultiIndex.from_frame(result.loc[mask, keys])).to_numpy()
        result.loc[mask, "calibration_source"] = "legacy_pt_person"
    result["weight"] *= result["scaling_factor"]
    result["group_weight"] *= result["scaling_factor"]
    if (~np.isfinite(result["weight"]) | result["weight"].lt(0)).any():
        raise ValueError("Cross-border calibration produced invalid weights.")
    return result, pd.DataFrame(records, columns = [
        "interview_place_id", "observations_2021", "effective_observations_2021",
        "entries_2015", "entries_2021", "calibrated_entries", "scaling_factor", "calibration_source",
    ])


def repair_car_occupancy(observations, rng):
    """Repair passenger expansion only; never change the vehicle weight.

    A passenger car has at most nine occupants including its driver. Motorcycle
    records use at most two. Invalid values are sampled from valid 2021 peers,
    first by vehicle type and purpose, then vehicle type alone.
    """
    result = observations.copy()
    original = pd.to_numeric(result["nb_passengers"], errors = "coerce")
    limit = np.where(result["survey_vehicle_type"].eq(2), 2, 9)
    car = result["trip_mode"].eq("car")
    valid = original.ge(1) & original.le(limit) & original.eq(np.floor(original))
    result["reported_occupancy"] = original
    result["occupancy_source"] = "reported"
    for idx in result.index[car & ~valid]:
        row = result.loc[idx]
        peers = result.loc[car & valid & result["survey_vehicle_type"].eq(row["survey_vehicle_type"])]
        same_purpose = peers[peers["trip_purpose"].eq(row["trip_purpose"])]
        if len(same_purpose) >= MIN_OBSERVATIONS:
            peers = same_purpose
        weight_column = "uncalibrated_weight" if "uncalibrated_weight" in peers else "weight"
        if peers.empty or peers[weight_column].sum() <= 0:
            raise ValueError("No valid occupancy peers for interview %s" % row["interview_id"])
        result.loc[idx, "nb_passengers"] = peers.sample(n = 1, weights = weight_column, random_state = rng)["nb_passengers"].iloc[0]
        result.loc[idx, "occupancy_source"] = "weighted_2021_peer"
    result.loc[car, "group_weight"] = result.loc[car, "weight"] * result.loc[car, "nb_passengers"]
    return result


def write_survey_audit(context, reference, observations, ledger, swiss_residents = False):
    """Write lightweight, pre-expansion diagnostics in this stage's own folder."""
    folder = Path(context.path())
    ledger.to_csv(folder / "vehicle_calibration.csv", index = False)
    columns = ["interview_id", "interview_place_id", "interview_place", "trip_mode", "travel_cat",
               "origin_country_raw", "destination_country_raw", "weight", "uncalibrated_weight",
               "scaling_factor", "calibration_source", "reported_occupancy", "nb_passengers", "occupancy_source",
               "start_x", "start_y", "end_x", "end_y", "origin_place", "destination_place",
               "start_location_source", "end_location_source", "location_usable", "location_issue"]
    audit = observations[columns].copy()
    supported_categories = ["From CH", "To CH"] if swiss_residents else ["From CH", "To CH", "Through CH"]
    audit["assignment_scope"] = np.where(audit["travel_cat"].isin(supported_categories), "supported_category", "unsupported_category")
    audit.loc[~audit["location_usable"], "assignment_scope"] = "unresolved_location"
    audit.to_csv(folder / "survey_demand_audit.csv", index = False)
    audit.groupby(["interview_place_id", "trip_mode", "travel_cat", "assignment_scope"]).agg(
        observations = ("interview_id", "size"), calibrated_weight = ("weight", "sum"),
    ).to_csv(folder / "survey_scope_counts.csv")
    occupancy = pd.to_numeric(reference["nb_passengers"], errors = "coerce")
    invalid = reference["trip_mode"].eq("car") & (~occupancy.between(1, 9) | occupancy.ne(np.floor(occupancy)))
    reference.loc[invalid, ["interview_id", "interview_place_id", "nb_passengers", "weight"]].to_csv(
        folder / "reference_occupancy_flags.csv", index = False)
    unusable = ~observations["location_usable"]
    logger.info("Cross-border vehicle controls: %.1f target entries, %.1f calibrated; %d unresolved survey locations (%.1f car entries).",
                ledger["entries_2015"].sum(), ledger["calibrated_entries"].sum(), int(unusable.sum()),
                observations.loc[unusable & observations["trip_mode"].eq("car"), "weight"].sum())


def write_assignment_audit(context, before_exclusion, result):
    """Count generated agents, not an assumed twice-the-person crossing volume."""
    keys = ["interview_place", "trip_mode", "label"]
    before = before_exclusion.groupby(keys).size().rename("expanded_agents")
    after = result.groupby(keys).size().rename("retained_agents")
    audit = pd.concat([before, after], axis = 1).fillna(0).reset_index()
    audit["excluded_agents"] = audit["expanded_agents"] - audit["retained_agents"]
    audit.to_csv(Path(context.path()) / "assignment_counts.csv", index = False)
