"""Shared event definitions for fitting and applying journey-level car controls."""

import numpy as np
import pandas as pd


def crossing_events(observations, targets, exit_places):
    """Count entry/return and uniquely reported exit events per survey journey."""
    keys = sorted(targets)
    names = {targets[key]["interview_place"]: key for key in keys}
    categories = pd.to_numeric(observations.travel_cat, errors="coerce").fillna(
        observations.travel_cat.map({"Within CH": 1, "Through CH": 2, "From CH": 3, "To CH": 4}))
    car = observations.trip_mode.eq("car").to_numpy()
    swiss = observations.residence_country.eq("CH").to_numpy()
    visit = categories.isin([3, 4]).to_numpy() & car
    visit &= ~swiss | observations.origin_country.eq("CH").to_numpy() | observations.destination_country.eq("CH").to_numpy()
    transit = categories.eq(2).to_numpy() & ~swiss & car
    codes = pd.to_numeric(observations.get("survey_exit_code", pd.Series(index=observations.index, dtype=float)), errors="coerce")
    exit_place = codes.map(exit_places)
    exit_id = exit_place.map(names)
    entry = np.zeros((len(observations), len(keys)))
    exits = np.zeros_like(entry)
    for column, key in enumerate(keys):
        local = observations.interview_place_id.eq(key).to_numpy()
        entry[:, column] = local * (2 * visit.astype(int) + transit.astype(int))
        exits[:, column] = transit & exit_id.eq(key).to_numpy()
    return entry, exits, transit, exit_place


def journey_multipliers(observations, controls):
    """Apply exactly the same event matrix used by the target solver."""
    selected = {key: value for key, value in controls.items()
                if value.get("calibration_basis") == "journey_entry_and_exit"}
    if not selected:
        return np.ones(len(observations))
    exit_places = {}
    for control in selected.values():
        exit_places.update({int(code): control["interview_place"] for code in control["reported_exit_codes"]})
    entry, exits, _, _ = crossing_events(observations, selected, exit_places)
    log_factors = np.array([selected[key]["log_event_factor"] for key in sorted(selected)])
    return np.exp((entry + exits) @ log_factors)
