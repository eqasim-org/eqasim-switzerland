"""Shared diary counts and eligibility used by spatial zoning and its evaluation."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from data.utils import coerce_boolean_series

FEATURES = ("work_trips", "shopping_trips", "leisure_trips", "total_trips")
CLUSTER_COLUMN = "activity_chain_cluster"
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def day_mask(roster, day):
    if day in ("workday", "weekend"):
        return coerce_boolean_series(roster[day], name=day)
    if day in DAYS:
        return roster.day.eq(day)
    raise ValueError(f"Unsupported specific_day_scenario: {day}")


def effective_sample(weights):
    """Kish effective sample size, invariant to a common weight multiplier."""
    return float(weights.sum() ** 2 / np.square(weights).sum()) if len(weights) else 0.0


def daily_trip_counts(trips):
    """Count destination purposes; total includes home returns and all purposes."""
    if (
        trips[["person_id", "trip_id"]].isna().any().any()
        or trips.duplicated(["person_id", "trip_id"]).any()
    ):
        raise ValueError("Trips require unique, non-missing (person_id, trip_id)")
    purposes = trips.purpose.astype("string").str.split("_").str[0]
    values = pd.DataFrame(
        {
            "person_id": trips.person_id,
            "work_trips": purposes.eq("work").astype(int),
            "shopping_trips": purposes.isin(["shop", "shopping"]).astype(int),
            "leisure_trips": purposes.eq("leisure").astype(int),
            "total_trips": 1,
        }
    )
    return values.groupby("person_id")[list(FEATURES)].sum()


def valid_diary_mask(chains):
    """Return the diary audit prepared by data.microcensus.activity_chains."""
    if chains.person_id.isna().any() or chains.person_id.duplicated().any():
        raise ValueError("Unique, non-missing survey person IDs required")
    if "valid_diary" not in chains or chains.valid_diary.isna().any():
        raise ValueError("Rebuild activity_chains to include the prepared diary audit")
    values = coerce_boolean_series(chains.valid_diary, name="valid_diary")
    return pd.Series(values.to_numpy(), index=chains.person_id, name="valid_diary")


def implementation_token():
    """Invalidate consuming stages when shared diary definitions change."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
