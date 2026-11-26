"""Prepare the one-way survey trips kept separately from Swiss resident diaries."""

from pathlib import Path

import pandas as pd

from data.cross_border.destinations import make_entry_border_facility_id, make_exit_border_facility_id
from synthesis.population.trips import BACKGROUND_CROSSINGS_FILE


def configure(context):
    context.stage("synthesis.population.trips")
    context.stage("synthesis.population.enriched")
    context.stage("synthesis.prepared.households")


def execute(context):

    # Read the crossings generated alongside the resident trips with the same random draws.

    path = Path(context.path("synthesis.population.trips")) / BACKGROUND_CROSSINGS_FILE
    trips = pd.read_pickle(path)
    if trips.person_id.isna().any() or not trips.person_id.is_unique:
        raise ValueError("Expected at most one background crossing per source resident")
    if not trips["mode"].isin(["car", "car_passenger", "pt"]).all():
        raise ValueError("Unknown Swiss crossing mode")

    # Use the same directional facilities as resident plans, including PT gateways.

    trips["agent_id"] = "background_ch_" + trips.person_id.astype(str)
    leaving = trips.following_purpose.eq("border")
    trips["border_facility"] = trips.mz_person_id.map(make_entry_border_facility_id)
    trips.loc[leaving, "border_facility"] = trips.loc[leaving, "mz_person_id"].map(make_exit_border_facility_id)

    # Supply the attributes the Swiss Java configurator normally copies via household membership.

    residents = context.stage("synthesis.population.enriched")
    households = context.stage("synthesis.prepared.households")
    source_households = residents.set_index("person_id").household_id.loc[trips.person_id]
    fields = ["number_of_cars_class", "income_per_capita", "sp_region", "canton_id", "canton_name", "ovgk"]
    values = households.households.set_index("household_id").loc[source_households, fields].copy()
    adults = residents.loc[residents.age.ge(18)].groupby("household_id").size()
    adult_count = values.index.map(adults).fillna(0).to_numpy()
    cars = values.number_of_cars_class.clip(upper=households.maximum_car_class)
    values["car_ownership_ratio"] = (1 - cars / adult_count).where(adult_count > 0, 0).clip(0, 1)
    values["number_of_cars"] = cars.astype(int).astype(str) + cars.eq(households.maximum_car_class).map({True: "+", False: ""})
    trips = trips.reset_index(drop=True).join(values.reset_index(drop=True))

    # Create a dedicated mode resource for each road plan using the existing vehicle types.

    road = trips["mode"].isin(["car", "car_passenger"])
    trips["vehicle_id"] = (trips.agent_id + ":" + trips["mode"]).where(road)
    fleet = trips.loc[road, ["agent_id", "mode", "vehicle_id"]].rename(columns={"agent_id": "owner_id"})
    fleet["type_id"] = "default_" + fleet["mode"]
    fleet["age"] = 0
    fleet["euro"] = 6
    return trips, fleet
