"""Prepare facility IDs, integer coordinates and named activity options."""

import numpy as np
import pandas as pd
from data.cross_border.activities import THROUGH_DESTINATION_PREFIX
from data.utils import coerce_boolean_series
from .schema import HOME_DESTINATION_ID
from .tables import Facilities

OPTIONS = {
    "offers_work": "work",
    "offers_education": "education",
    "offers_other": "other",
    "offers_leisure": "leisure",
    "offers_shop": "shop",
}


def configure(context):
    context.config("use_airports", default=True)
    if context.config("use_airports"):
        context.stage("synthesis.prepared.population")
    context.stage("synthesis.population.destinations")
    context.stage("synthesis.population.enriched")
    context.config("include_cross_border", default=False)
    if context.config("include_cross_border"):
        context.stage("data.cross_border.generate_cross_border_traffic")
    context.config("include_external_population", default=True)
    if context.config("include_external_population"):
        context.stage("data.external_population.read_outputs")


def execute(context):

    # Prepare shared business facilities and resident household homes.

    destinations = context.stage("synthesis.population.destinations")
    frames = [
        _destinations(destinations),
        _homes(context.stage("synthesis.population.enriched")),
    ]

    # Include cross-border homes and through-traffic endpoints abroad.

    if context.config("include_cross_border"):
        persons, activities, _ = context.stage(
            "data.cross_border.generate_cross_border_traffic"
        )
        frames.append(_homes(persons))
        through = activities.loc[
            activities.destination_id.astype(str).str.startswith(
                THROUGH_DESTINATION_PREFIX
            )
        ]
        frames.append(
            pd.DataFrame(
                {
                    "facility_id": through.destination_id.to_numpy(),
                    "x": through.geometry.map(lambda point: point.x).to_numpy(),
                    "y": through.geometry.map(lambda point: point.y).to_numpy(),
                    "activity_options": [("other",)] * len(through),
                }
            )
        )

    # Imported activities reuse shared facility coordinates and IDs where available.

    if context.config("include_external_population"):
        persons, activities, _ = context.stage("data.external_population.read_outputs")
        valid = persons.home_x.notna() & persons.home_y.notna()
        frames.append(_homes(persons.loc[valid]))
        external = (
            activities[["destination_id", "destination_x", "destination_y"]]
            .drop_duplicates("destination_id")
            .copy()
        )
        ids = external.destination_id.astype(str)
        keep = ~ids.eq(str(HOME_DESTINATION_ID)) & ~ids.str.startswith("home")
        keep &= ~external.destination_id.isin(destinations.destination_id)
        external = external.loc[keep]
        for column in OPTIONS:
            external[column] = True
        frames.append(_destinations(external))

    # Give airport activities explicit facilities at their sampled coordinates.

    if context.config("use_airports"):
        trips = context.stage("synthesis.prepared.population").airport_trips
        for endpoint in ["origin", "destination"]:
            frames.append(pd.DataFrame({
                "facility_id": trips[endpoint + "_facility_id"],
                "x": trips[endpoint + "_x"],
                "y": trips[endpoint + "_y"],
                "activity_options": trips[endpoint + "_purpose"].map(lambda purpose: (purpose,)),
            }))

    # Enforce one unambiguous definition per serialized facility ID.

    facilities = pd.concat(frames, ignore_index=True)
    if facilities.facility_id.isna().any():
        raise ValueError("Facility IDs must not be missing")
    facilities["facility_id"] = facilities.facility_id.astype(str)
    xy = facilities[["x", "y"]].to_numpy(dtype=float)
    if not np.isfinite(xy).all():
        raise ValueError("Facility coordinates must be finite")
    facilities[["x", "y"]] = np.trunc(xy).astype(np.int64)
    duplicated = facilities.facility_id.duplicated(keep=False)
    for facility_id, group in facilities.loc[duplicated].groupby(
        "facility_id", sort=False
    ):
        if len(group[["x", "y", "activity_options"]].drop_duplicates()) != 1:
            raise ValueError(f"Conflicting definitions for facility {facility_id}")
    facilities = facilities.drop_duplicates("facility_id").reset_index(drop=True)
    return Facilities(facilities)


def _homes(persons):
    """One home facility per household; prefixes match the population writer."""
    homes = persons[["household_id", "home_x", "home_y"]].drop_duplicates(
        "household_id"
    )
    return pd.DataFrame(
        {
            "facility_id": "home" + homes.household_id.astype(str),
            "x": homes.home_x,
            "y": homes.home_y,
            "activity_options": [("home",)] * len(homes),
        }
    )


def _destinations(destinations):
    """Use flag names explicitly: column positions must not determine purposes."""
    frame = destinations[
        ["destination_id", "destination_x", "destination_y", *OPTIONS]
    ].copy()
    for flag in OPTIONS:
        frame[flag] = coerce_boolean_series(frame[flag], name=flag)
    options = [
        tuple(purpose for flag, purpose in OPTIONS.items() if bool(getattr(row, flag)))
        for row in frame.itertuples(index=False)
    ]
    return pd.DataFrame(
        {
            "facility_id": frame.destination_id,
            "x": frame.destination_x,
            "y": frame.destination_y,
            "activity_options": options,
        }
    )


def validate(context):
    from .tables import implementation_token

    return implementation_token(__file__)
