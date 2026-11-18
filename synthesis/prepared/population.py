"""Prepare and validate population tables for simulation and analysis."""

import logging
import pandas as pd
import polars as pl
import geopandas as gpd
from shapely import wkt

from data.utils import coerce_boolean_series
from synthesis.population.trips_with_departures import align_activities
from .schema import (
    HOME_DESTINATION_ID,
    VEHICLE_FIELDS,
    ACTIVITY_ATTRIBUTE_FIELDS,
    PERSON_FIELDS,
    ACTIVITY_FIELDS,
    PERSONS_DTYPES,
    BOOLEAN_PERSON_COLUMNS,
)
from .tables import Population

logger = logging.getLogger("synpp")


def configure(context):
    # The zones of the regional subscriptions of the Swiss synthetic population; false: without them
    context.config("use_subscription_zones", default=True)
    context.stage(
        (
            "synthesis.population.models.subscription_zones"
            if context.config("use_subscription_zones")
            else "synthesis.population.models.subscriptions"
        ),
        alias="synthetic_persons",
    )
    context.config("enable_mode_choice", False)
    context.stage(
        (
            "mode_choice.run"
            if context.config("enable_mode_choice")
            else "synthesis.population.trips_with_departures"
        ),
        alias="population_trips",
    )
    context.stage("synthesis.population.activities")
    context.stage("synthesis.population.spatial.locations")
    context.stage("data.spatial.cantons")
    context.config("use_freight", default=True)
    context.config("use_lcv", default=True)
    if context.config("use_freight"):
        context.stage("synthesis.freight.trips")
    if context.config("use_lcv"):
        context.stage("synthesis.lcv.trips")
    context.stage("data.constants")

    context.stage("synthesis.vehicles.vehicles")

    context.config("include_cross_border", default=False)
    if context.config("include_cross_border"):
        context.stage("data.cross_border.generate_cross_border_traffic")

    context.config("include_external_population", default=True)
    if context.config("include_external_population"):
        context.stage("data.external_population.read_outputs")
        context.stage("data.external_population.constants")
        context.stage("synthesis.population.destinations")


def execute(context):

    # Align domestic people, trips, activity times and locations.

    const = context.stage("data.constants")
    persons, activities, vehicles = _prepare_residents(context, const)

    # Convert and append optional populations using the same export schema.

    if context.config("include_external_population"):
        persons, activities, vehicles = _append_external(
            context, const, persons, activities, vehicles
        )
    if context.config("include_cross_border"):
        persons, activities, vehicles = _append_cross_border(
            context, persons, activities, vehicles
        )

    # Normalise types, retain complete plans and validate the table relationships.

    df_persons, df_activities, df_vehicles = _normalise_tables(
        persons, activities, vehicles
    )

    # Prepare optional freight/LCV tables in the order consumed by their writers.

    fleet = context.stage("synthesis.vehicles.vehicles")
    freight = (
        _prepare_freight(context.stage("synthesis.freight.trips"))
        if context.config("use_freight")
        else None
    )
    lcv = (
        _prepare_freight(context.stage("synthesis.lcv.trips"), is_lcv=True)
        if context.config("use_lcv")
        else None
    )
    freight_vehicles = (
        fleet[2].sort_values("owner_id")[VEHICLE_FIELDS].copy()
        if freight is not None
        else None
    )
    lcv_vehicles = (
        fleet[3].sort_values("owner_id")[VEHICLE_FIELDS].copy()
        if lcv is not None
        else None
    )

    # Return dataframes only. Compression and XML formatting belong to the consumer.

    return Population(
        df_persons,
        df_activities,
        df_vehicles,
        freight,
        freight_vehicles,
        lcv,
        lcv_vehicles,
    )


def _prepare_residents(context, cst):
    """Join domestic attributes and aligned plans without changing upstream tables."""
    population_trips = context.stage("population_trips")
    if isinstance(population_trips, pl.DataFrame):
        population_trips = population_trips.to_pandas()

    df_persons = context.stage("synthetic_persons").copy()
    df_activities = align_activities(
        context.stage("synthesis.population.activities").copy(), population_trips
    )
    df_vehicles = context.stage("synthesis.vehicles.vehicles")[1].copy()

    # Normalise employment to the shared binary scenario representation.
    if set(df_persons["employed"].unique()) != {0, 1}:
        df_persons["employed"] = (
            pd.to_numeric(df_persons["employed"], errors="coerce").fillna(0)
            == cst.EMPLOYED
        ).astype(int)

    # Attach following modes to activities
    df_trips = pd.DataFrame(population_trips, copy=True)[
        ["person_id", "trip_index", "mode"]
    ]
    df_trips.columns = ["person_id", "activity_index", "following_mode"]
    df_activities = pd.merge(
        df_activities, df_trips, on=["person_id", "activity_index"], how="left"
    )

    # Attach locations to activities
    df_locations = context.stage("synthesis.population.spatial.locations")
    df_activities = pd.merge(
        df_activities, df_locations, on=["person_id", "activity_index"], how="left"
    )

    # Find the loop attribute in the following_modes
    loop_modes = ["walk_loop", "car_loop", "car_passenger_loop", "pt_loop", "bike_loop"]

    unique_modes_per_agent = (
        df_activities.groupby("person_id")["following_mode"]
        .apply(lambda x: list(set(x)))
        .reset_index()
    )
    loop_columns = ["person_id"]
    for mode in loop_modes:
        col = "has_" + mode + "_trip"
        unique_modes_per_agent[col] = [
            mode in unique_modes
            for unique_modes in unique_modes_per_agent["following_mode"]
        ]
        loop_columns.append(col)

    df_persons = df_persons.merge(
        unique_modes_per_agent[loop_columns], on="person_id", how="left"
    )
    df_persons[loop_columns[1:]] = df_persons[loop_columns[1:]].fillna(False)

    # Bring in correct order (although it should already be)
    df_persons = df_persons.sort_values(by="person_id")
    df_activities = df_activities.sort_values(by=["person_id", "activity_index"])
    df_vehicles = df_vehicles.sort_values(by=["owner_id"])

    df_persons["person_type"] = "normal"
    is_crossing_the_border = coerce_boolean_series(
        df_persons["is_crossing_the_border"],
        name="is_crossing_the_border",
    )
    df_persons.loc[is_crossing_the_border, "person_type"] = "crossborder"

    # For Swiss residents crossing the border, destination_country_raw was already
    # assigned in synthesis.population.models.cross_border (matched to a specific
    # data.cross_border.swiss_residents_od record) and carried through
    # synthesis.population.models.subscriptions -- the same record
    # synthesis.population.spatial.locations used for the border activity's
    # location, so label and point stay consistent.
    df_persons["cross_border_od"] = None
    has_raw_od = df_persons["destination_country_raw"].notna()
    df_persons.loc[has_raw_od, "cross_border_od"] = (
        "CH-" + df_persons.loc[has_raw_od, "destination_country_raw"]
    )

    df_persons = df_persons.drop(columns=["destination_country_raw"])

    return df_persons, df_activities, df_vehicles


def _append_external(context, cst, df_persons, df_activities, df_vehicles):
    """Convert external attributes and snap activities onto shared facility points."""
    external_persons = context.stage("data.external_population.read_outputs")[0].copy()
    external_activities = context.stage("data.external_population.read_outputs")[
        1
    ].copy()
    external_vehicles = context.stage("data.external_population.read_outputs")[2].copy()
    ex_constants = context.stage("data.external_population.constants")

    external_persons["person_type"] = ex_constants.person_type
    external_persons["sex"] = ex_constants.convert_sex(external_persons["sex"]).astype(
        int
    )
    external_persons["pt_subscription"] = ex_constants.get_subscriptions(
        external_persons
    )
    external_persons["bike_availability"] = ex_constants.convert_bike_availability(
        external_persons["number_of_bikes_class"], cst
    )
    external_persons["car_availability"] = ex_constants.convert_car_availability(
        external_persons["car_availability"]
    )

    external_persons = external_persons[external_persons["home_x"].notna()]
    external_persons = external_persons[external_persons["home_y"].notna()]

    external_activities["destination_id"] = external_activities[
        "destination_id"
    ].astype(object)
    external_activities.loc[
        external_activities["purpose"] == "home", "destination_id"
    ] = HOME_DESTINATION_ID

    external_activities = snap_to_shared_facilities(
        external_activities, context.stage("synthesis.population.destinations")
    )

    external_activities["destination_x"] = external_activities["destination_x"].astype(
        int
    )
    external_activities["destination_y"] = external_activities["destination_y"].astype(
        int
    )

    for col in ACTIVITY_ATTRIBUTE_FIELDS:
        if not col in external_activities.columns:
            raise RuntimeError("Missing column in external activities: %s" % col)

    external_vehicles = external_vehicles[VEHICLE_FIELDS]

    df_persons = pd.concat([df_persons, external_persons])
    df_activities = pd.concat([df_activities, external_activities])
    df_vehicles = pd.concat([df_vehicles, external_vehicles])

    df_persons["mz_person_id"] = df_persons["mz_person_id"].astype(int)
    df_persons["home_x"] = df_persons["home_x"].astype(int)
    df_persons["home_y"] = df_persons["home_y"].astype(int)

    df_persons = df_persons.sort_values(by="person_id")
    df_activities = df_activities.sort_values(by=["person_id", "activity_index"])
    df_vehicles = df_vehicles.sort_values(by=["owner_id"])

    return df_persons, df_activities, df_vehicles


def _append_cross_border(context, df_persons, df_activities, df_vehicles):
    """Attach the cross-border OD label and append already generated plans."""
    cross_border_persons = context.stage(
        "data.cross_border.generate_cross_border_traffic"
    )[0].copy()
    cross_border_activities = context.stage(
        "data.cross_border.generate_cross_border_traffic"
    )[1].copy()
    cross_border_vehicles = context.stage(
        "data.cross_border.generate_cross_border_traffic"
    )[2].copy()

    cross_border_persons = cross_border_persons.sort_values(by="person_id")
    cross_border_activities = cross_border_activities.sort_values(
        by=["person_id", "activity_index"]
    )
    cross_border_vehicles = cross_border_vehicles.sort_values(by=["owner_id"])

    for col in ACTIVITY_ATTRIBUTE_FIELDS:
        if col not in cross_border_activities.columns:
            cross_border_activities[col] = 0

    cross_border_persons["person_type"] = "crossborder"

    # These agents travel <abroad> -> CH -> <abroad>, so the od label has
    # three parts (e.g. "FR-CH-FR"). data.cross_border.generate_od orders
    # every record so that it ends in CH: "From-To" agents then carry
    # destination_country_raw == "CH" and go back to their origin country,
    # while "Through" agents keep two distinct foreign endpoints.
    origin_country = cross_border_persons["origin_country_raw"]
    destination_country = cross_border_persons["destination_country_raw"]
    destination_country = destination_country.where(
        destination_country != "CH", origin_country
    )

    cross_border_persons["cross_border_od"] = (
        origin_country + "-CH-" + destination_country
    )
    cross_border_persons["pt_subscription"] = 0
    cross_border_persons["bike_availability"] = 0
    cross_border_persons["car_availability"] = 1

    df_persons = pd.concat([df_persons, cross_border_persons])
    df_activities = pd.concat([df_activities, cross_border_activities])
    df_vehicles = pd.concat([df_vehicles, cross_border_vehicles])

    df_persons["mz_person_id"] = df_persons["mz_person_id"].astype(int)
    df_persons["home_x"] = df_persons["home_x"].astype(int)
    df_persons["home_y"] = df_persons["home_y"].astype(int)

    df_persons = df_persons.sort_values(by="person_id")
    df_activities = df_activities.sort_values(by=["person_id", "activity_index"])
    df_vehicles = df_vehicles.sort_values(by=["owner_id"])

    return df_persons, df_activities, df_vehicles


def _normalise_tables(df_persons, df_activities, df_vehicles):
    """Resolve booleans/dtypes and keep sorted, complete person/plan/fleet rows."""
    df_persons["is_car_passenger"] = df_persons["is_car_passenger"].fillna(False)

    # Only the Swiss synthetic persons have subscription zones: the external and cross-border agents get none
    df_persons["subscription_zones"] = df_persons.get(
        "subscription_zones", pd.Series("", index=df_persons.index)
    ).fillna("")

    # Extra scenario markers stay available to PersonWriter; imported external
    # populations have no such classification and receive explicit False defaults.
    residence_flags = [
        "non_private_dorm_candidate_households",
        "non_private_home_only_households",
        "is_synthetic_household",
    ]
    for column in residence_flags:
        df_persons[column] = coerce_boolean_series(
            df_persons.get(column, pd.Series(False, index=df_persons.index)),
            name=column,
        )
    df_persons["non_private_model_status"] = df_persons.get(
        "non_private_model_status", pd.Series("ordinary_models", index=df_persons.index)
    ).fillna("ordinary_models")
    df_persons = df_persons[
        PERSON_FIELDS
        + ["subscription_zones"]
        + residence_flags
        + ["non_private_model_status"]
    ]
    df_activities = df_activities[ACTIVITY_FIELDS]
    df_vehicles = df_vehicles[VEHICLE_FIELDS]
    df_vehicles["owner_id"] = df_vehicles["owner_id"].astype(int)

    df_activities["is_last"] = coerce_boolean_series(
        df_activities["is_last"], name="is_last"
    )

    # Normalize logical values before astype: bool("False") is True in Python,
    # and CSV inference can produce either strings or booleans for the same field.
    for column in BOOLEAN_PERSON_COLUMNS:
        df_persons[column] = coerce_boolean_series(df_persons[column], name=column)

    # Normalise dtypes and geometry before selecting valid, complete plans.
    df_persons = df_persons.astype(PERSONS_DTYPES)
    df_activities["geometry"] = df_activities["geometry"].apply(
        lambda g: wkt.loads(g) if isinstance(g, str) else g
    )
    valid_ids = df_activities.groupby("person_id")["geometry"].apply(
        lambda g: g.notna().all()
    )
    valid_ids = valid_ids[valid_ids].index

    df_persons = df_persons[df_persons["person_id"].isin(valid_ids)]
    df_activities = df_activities[
        df_activities["person_id"].isin(df_persons["person_id"].values.tolist())
    ]
    df_vehicles = df_vehicles[
        df_vehicles["owner_id"].isin(df_persons["person_id"].values.tolist())
    ]

    # Keep one activity/location row per person and activity index.
    df_activities = df_activities.drop_duplicates(
        ["person_id", "activity_index"], keep="first"
    )

    # Make sure the minimum required columns exist (order does NOT matter)
    _require_cols(
        df_persons,
        [
            "person_id",
            "age",
            "car_availability",
            "employed",
            "driving_license",
            "sex",
            "home_x",
            "home_y",
        ],
        "df_persons",
    )
    _require_cols(
        df_activities,
        [
            "person_id",
            "activity_index",
            "start_time",
            "end_time",
            "purpose",
            "is_last",
            "geometry",
            "destination_id",
            "following_mode",
            "municipality_type",
            "municipality_id",
            "employee_density",
            "companies_density",
            "population_density",
        ],
        "df_activities",
    )
    _require_cols(df_vehicles, ["mode", "vehicle_id", "owner_id"], "df_vehicles")

    # Cast only the columns that exist (so removing/adding columns won't break)
    df_persons = df_persons.astype(
        {k: v for k, v in PERSONS_DTYPES.items() if k in df_persons.columns}
    )

    df_persons = df_persons.sort_values(by="person_id")
    df_activities = df_activities.sort_values(by=["person_id", "activity_index"])
    df_vehicles = df_vehicles.sort_values(by=["owner_id"])

    # Check table relationships before returning the read-only export bundle.

    if df_persons.person_id.duplicated().any():
        raise ValueError("Prepared population contains duplicate person IDs")
    if df_vehicles.vehicle_id.duplicated().any():
        raise ValueError("Prepared population contains duplicate owned vehicle IDs")
    last_counts = df_activities.groupby("person_id").is_last.sum()
    final_rows = df_activities.groupby("person_id", sort=False).tail(1)
    if not last_counts.eq(1).all() or not final_rows.is_last.all():
        raise ValueError(
            "Each prepared person must have exactly one final activity, at the end of the plan"
        )
    if df_activities.loc[~df_activities.is_last, "following_mode"].isna().any():
        raise ValueError(
            "Non-final prepared activities must have a following travel mode"
        )

    return df_persons, df_activities, df_vehicles


def _require_cols(df, cols, df_name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{df_name} is missing required columns: {missing}")


def snap_to_shared_facilities(external_activities, df_destinations):
    """Align imported activity points with authoritative shared facility locations.

    Imported activities and the facility inventory may reach the same site via
    different reprojection/rounding paths. Use the inventory's coordinates for
    every shared ID so that scenario consumers receive consistent locations.
    """

    coordinates = df_destinations[
        ["destination_id", "destination_x", "destination_y"]
    ].drop_duplicates("destination_id")
    coordinates = coordinates.set_index("destination_id")

    shared = external_activities["destination_id"].isin(coordinates.index)

    if shared.any():
        x = (
            external_activities.loc[shared, "destination_id"]
            .map(coordinates["destination_x"])
            .astype(float)
        )
        y = (
            external_activities.loc[shared, "destination_id"]
            .map(coordinates["destination_y"])
            .astype(float)
        )

        external_activities.loc[shared, "destination_x"] = x.values
        external_activities.loc[shared, "destination_y"] = y.values
        external_activities.loc[shared, "geometry"] = gpd.points_from_xy(
            x.values, y.values
        )

        logger.info(
            "Aligned %d external activities (%d distinct facilities) with the shared facility coordinates.",
            int(shared.sum()),
            int(external_activities.loc[shared, "destination_id"].nunique()),
        )

    return external_activities


def _prepare_freight(trips, is_lcv=False):
    """Normalise heavy/light freight to one named export schema."""
    frame = trips.copy()
    if is_lcv:
        frame = frame.rename(columns={"trip_id": "agent_id"})
        frame["mode"] = "truck"
    else:
        frame["mode"] = frame["vehicle_type"].astype(str)
    fields = [
        "agent_id",
        "origin_x",
        "origin_y",
        "destination_x",
        "destination_y",
        "departure_time",
        "mode",
    ]
    _require_cols(frame, fields, "freight trips")
    frame = frame[fields].copy()
    frame["arrival_time"] = frame.departure_time + 3600
    frame["day_end_time"] = 30 * 3600
    if frame.agent_id.isna().any() or frame.agent_id.duplicated().any():
        raise ValueError("Freight export requires unique, non-missing agent IDs")
    return frame.sort_values("agent_id").reset_index(drop=True)


def validate(context):
    from .tables import implementation_token
    from inspect import getfile

    return implementation_token(__file__, getfile(align_activities))
