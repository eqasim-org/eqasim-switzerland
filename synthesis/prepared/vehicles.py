"""Combine synthetic fleets, resolve physical types and retain emission metadata."""

import pandas as pd
from .tables import Vehicles

TYPE_FIELDS = ["type_id", "nb_seats", "length", "width", "pce", "mode"]
VEHICLE_FIELDS = ["vehicle_id", "type_id", "age", "euro"]


def configure(context):
    context.stage("synthesis.vehicles.vehicles")
    if context.config("swiss_cross_border_background_traffic", default=False):
        context.stage("synthesis.prepared.background_crossings")
    context.config("use_airports", default=True)
    if context.config("use_airports"):
        context.stage("synthesis.prepared.population")
    context.config("include_cross_border", default=False)
    if context.config("include_cross_border"):
        context.stage("data.cross_border.generate_cross_border_traffic")
    context.config("include_external_population", default=True)
    if context.config("include_external_population"):
        context.stage("data.external_population.read_outputs")


def execute(context):

    # Assemble the domestic passenger, freight and light-commercial fleets.

    types, cars, trucks, lcv = context.stage("synthesis.vehicles.vehicles")
    frames = [cars, trucks, lcv]
    passenger_indices = [0]

    # Map imported loop modes onto the existing physical vehicle types.

    if context.config("include_external_population"):
        external = context.stage("data.external_population.read_outputs")[2].copy()
        external["type_id"] = "default_" + external["mode"].str.replace(
            "_loop$", "", regex=True
        )
        passenger_indices.append(len(frames))
        frames.append(external)
    if context.config("include_cross_border"):
        crossing = context.stage("data.cross_border.generate_cross_border_traffic")[
            2
        ].copy()
        for mode in ("car", "car_passenger"):
            crossing.loc[crossing["mode"].eq(mode), "type_id"] = "default_" + mode
        passenger_indices.append(len(frames))
        frames.append(crossing)

    # Replace selected residents' passenger resources with the airport fleet.

    if context.config("use_airports"):
        population = context.stage("synthesis.prepared.population")
        resident_ids = set(population.airport_trips.loc[
            ~population.airport_trips.is_visitor, "agent_id"
        ])
        # Freight/LCV owner numbers belong to separate namespaces.
        for index in passenger_indices:
            frames[index] = frames[index].loc[
                ~frames[index].owner_id.astype(str).isin(resident_ids)
            ]
        frames.append(population.airport_vehicles)
    if context.config("swiss_cross_border_background_traffic"):
        frames.append(context.stage("synthesis.prepared.background_crossings")[1])
    vehicles = pd.concat(frames, ignore_index=True)

    # Validate type references and retain optional physical/emission descriptors.

    for frame, column in ((types, "type_id"), (vehicles, "vehicle_id")):
        if frame[column].isna().any() or frame[column].astype(str).duplicated().any():
            raise ValueError(f"Vehicle export requires unique, non-missing {column}")
    known = set(types.type_id.astype(str))
    unknown = set(vehicles.type_id.astype(str)) - known
    if unknown:
        raise ValueError(f"Vehicles refer to undefined types: {sorted(unknown)}")
    exported_types = types[TYPE_FIELDS].copy()
    exported_types["maximum_velocity"] = types.get(
        "maxVelocity", pd.Series(None, index=types.index, dtype=object)
    )
    exported_types["flow_efficiency_factor"] = types.get(
        "flowEfficiencyFactor", pd.Series(None, index=types.index, dtype=object)
    )
    # Preserve emission descriptors as ordinary columns; consumers choose their
    # own attribute names and serialization representation.
    for column in ("hbefa_cat", "hbefa_tech", "hbefa_size", "hbefa_emission"):
        exported_types[column] = types.get(
            column, pd.Series(None, index=types.index, dtype=object)
        )
    exported_vehicles = vehicles[["vehicle_id", "type_id"]].copy()
    for column in ("age", "euro"):
        exported_vehicles[column] = vehicles.get(
            column, pd.Series(None, index=vehicles.index, dtype=object)
        )
    return Vehicles(
        exported_types.reset_index(drop=True), exported_vehicles.reset_index(drop=True)
    )


def validate(context):
    from .tables import implementation_token

    return implementation_token(__file__)
