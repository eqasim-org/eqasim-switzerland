"""Write prepared population tables as MATSim population.xml.gz."""

import gzip
import io
import json
import logging
import shutil
import subprocess
from xml.sax.saxutils import escape
import pandas as pd
import matsim.writers
from matsim.writers import backlog_iterator
from data.external_population.constants import ExternalPopulationConstants
from .schema import (
    HOME_DESTINATION_ID,
    VEHICLE_FIELDS,
    ACTIVITY_ATTRIBUTES_TO_SAVE,
    PERSON_FIELDS,
    ACTIVITY_FIELDS,
    PERSONS_DTYPES,
    BOOLEAN_PERSON_COLUMNS,
    SUBSCRIPTIONS,
)

# Preserve the existing helper import for external callers.
from synthesis.prepared.population import snap_to_shared_facilities

logger = logging.getLogger("synpp")


def configure(context):
    context.stage("synthesis.prepared.population")
    context.config("population_compresslevel", default=1)
    context.config("population_use_pigz", default=True)
    context.config("population_pigz_threads", default=8)


def execute(context):

    # Read cached export tables and choose the configured compression backend.

    data = context.stage("synthesis.prepared.population")
    xml_path = "%s/population.xml" % context.path()
    gzip_path = "%s/population.xml.gz" % context.path()
    level = int(context.config("population_compresslevel"))
    requested_pigz = bool(context.config("population_use_pigz"))
    use_pigz = requested_pigz and shutil.which("pigz") is not None
    if requested_pigz and not use_pigz:
        logger.warning("pigz was not found; using Python gzip for population.xml.gz")
    output = xml_path if use_pigz else gzip_path
    open_file = open if use_pigz else gzip.open
    options = {} if use_pigz else {"compresslevel": level}

    # Stream resident plans and optional freight plans from the prepared frames.

    with open_file(output, "wb", **options) as stream:
        with io.BufferedWriter(stream, buffer_size=2 * 1024**3) as buffered:
            writer = matsim.writers.PopulationWriter(buffered)
            writer.start_population()
            _write_persons(context, writer, data)
            if data.freight_trips is not None:
                _write_freight(
                    context, writer, data.freight_trips, data.freight_vehicles
                )
            if data.lcv_trips is not None:
                _write_freight(
                    context, writer, data.lcv_trips, data.lcv_vehicles, is_lcv=True
                )
            if data.airport_trips is not None:
                _write_airports(context, writer, data.airport_trips, data.airport_vehicles)
            writer.end_population()

    # Compress the completed XML when the parallel compression backend is selected.

    if use_pigz:
        threads = max(1, int(context.config("population_pigz_threads")))
        logger.info("Compressing population.xml with pigz using %d threads", threads)
        subprocess.run(
            ["pigz", "-f", "-p", str(threads), "-" + str(level), xml_path], check=True
        )
    return gzip_path


def _write_persons(context, writer, data):
    """Consume aligned activities/vehicles once, with bounded per-person memory."""
    activities = iter(data.activities.itertuples(index=False, name="Activity"))
    vehicles = backlog_iterator(
        iter(data.vehicles.itertuples(index=False, name="Vehicle"))
    )
    written_activities = 0
    for person in context.progress(
        data.persons.itertuples(index=False, name="Person"),
        total=len(data.persons),
        label="Writing persons",
    ):
        first = next(activities)
        if first.person_id != person.person_id:
            raise ValueError("Prepared persons and activities are not aligned")
        record = PersonWriter(person)
        record.add_vehicles(_owned_vehicles(vehicles, person.person_id))
        written_activities += record.write(
            writer, first_activity=first, activity_iterator=activities
        )
    if written_activities != len(data.activities):
        raise ValueError("Prepared activities were not all written")


def _write_freight(context, writer, trips, fleet, is_lcv=False):
    """The same serializer handles the normalised heavy/light freight contract."""
    vehicles = backlog_iterator(iter(fleet.itertuples(index=False)))
    for trip in context.progress(
        trips.itertuples(index=False),
        total=len(trips),
        label="Writing LCV" if is_lcv else "Writing freight",
    ):
        record = FreightWriter(trip, is_lcv=is_lcv)
        record.add_vehicles(_owned_vehicles(vehicles, trip.agent_id))
        record.write(writer, truck=not is_lcv)


def _write_airports(context, writer, trips, fleet):
    """Write fixed, one-leg airport plans without ordinary resident attributes."""
    vehicles = backlog_iterator(iter(fleet.itertuples(index=False)))
    for trip in context.progress(trips.itertuples(index=False), total=len(trips), label="Writing airports"):
        writer.start_person(escape(trip.agent_id, {'"': '&quot;'}))
        writer.start_attributes()
        writer.add_attribute("isAirport", "java.lang.Boolean", "true")
        writer.add_attribute("isVisitor", "java.lang.Boolean", writer.true_false(trip.is_visitor))
        writer.add_attribute("subpopulation", "java.lang.String", "airport")
        writer.add_attribute("airport", "java.lang.String", escape(trip.airport))
        if not trip.is_visitor:
            writer.add_attribute("age", "java.lang.Integer", int(trip.age))
        resources = {v.mode: v.vehicle_id for v in _owned_vehicles(vehicles, trip.agent_id)}
        writer.add_attribute("vehicles", "org.matsim.vehicles.PersonVehicles", escape(json.dumps(resources)))
        writer.end_attributes()
        writer.start_plan(selected=True)
        origin = writer.location(trip.origin_x, trip.origin_y, escape(trip.origin_facility_id, {'"': '&quot;'}))
        destination = writer.location(trip.destination_x, trip.destination_y, escape(trip.destination_facility_id, {'"': '&quot;'}))
        writer.add_activity(trip.origin_purpose, origin, end_time=trip.departure_time)
        writer.add_leg(trip.mode, trip.departure_time, trip.arrival_time - trip.departure_time)
        writer.add_activity(trip.destination_purpose, destination, start_time=trip.arrival_time)
        writer.end_plan()
        writer.end_person()


def _owned_vehicles(iterator, owner_id):
    """Read this owner's contiguous vehicle block from an owner-sorted table."""
    records = []
    while iterator.has_next():
        row = iterator.next()
        if row.owner_id != owner_id:
            iterator.previous()
            break
        records.append(row)
    return records


def format_regional_products(zones):
    """The RegionalSubscriptionProducts attribute: the zones of the regional subscription per fare network, the networks
    joined by "+", e.g. ZVV:110,121+Unireso:10; empty without zones. zones: subscription_zones of
    synthesis.population.models.subscription_zones ("ZVV:110,121", several networks separated by ";").
    """
    networks = [
        part.split(":", 1) for part in str(zones or "").split(";") if ":" in part
    ]
    return "+".join(
        f"{network.strip()}:{','.join(z.strip() for z in codes.split(','))}"
        for network, codes in networks
    )


def _na_to(x, to=None):
    return to if pd.isna(x) else x


class PersonWriter:
    def __init__(self, person):
        self.person = person
        self.activities = []
        self.vehicles = []

    def add_activity(self, activity):
        self.activities.append(activity)

    def add_vehicles(self, vehicles):
        self.vehicles = vehicles

    def _write_single_activity(self, writer, activity, home_location):
        geometry = activity.geometry
        destination_id = activity.destination_id

        location = (
            home_location
            if destination_id == HOME_DESTINATION_ID
            or destination_id == str(HOME_DESTINATION_ID)
            else writer.location(
                int(geometry.x),
                int(geometry.y),
                (
                    destination_id
                    if isinstance(destination_id, str)
                    else int(destination_id)
                ),
            )
        )

        start_time = _na_to(activity.start_time, None)
        end_time = _na_to(activity.end_time, None)

        attributes = {
            "municipalityType": _na_to(activity.municipality_type, "other"),
            "municipalityId": _na_to(activity.municipality_id, -1),
            "employeeDensity": _na_to(activity.employee_density, 0.0),
            "companiesDensity": _na_to(activity.companies_density, 0.0),
            "populationDensity": _na_to(activity.population_density, 0.0),
            "ovgk": _na_to(activity.ovgk, None),
        }

        writer.add_activity(
            activity.purpose, location, start_time, end_time, attributes=attributes
        )

    def write(self, writer, first_activity=None, activity_iterator=None):
        p = self.person

        writer.start_person(str(p.person_id))

        # Attributes
        writer.start_attributes()
        writer.add_attribute("age", "java.lang.Integer", str(int(p.age)))
        writer.add_attribute(
            "employed", "java.lang.Boolean", writer.true_false(p.employed)
        )
        writer.add_attribute(
            "hasLicense", "java.lang.String", writer.yes_no(p.driving_license)
        )
        writer.add_attribute("sex", "java.lang.String", ["m", "f"][p.sex])
        writer.add_attribute("home_coordinate_x", "java.lang.Double", str(p.home_x))
        writer.add_attribute("home_coordinate_y", "java.lang.Double", str(p.home_y))
        writer.add_attribute(
            "carAvail", "java.lang.String", ["never", "always"][int(p.car_availability)]
        )

        writer.add_attribute(
            "subscriptions", "java.lang.String", SUBSCRIPTIONS[int(p.pt_subscription)]
        )
        # the zones of the regional subscription, per fare network (see format_regional_products)
        regional_products = format_regional_products(p.subscription_zones)
        if regional_products:
            writer.add_attribute(
                "RegionalSubscriptionProducts",
                "java.lang.String",
                escape(regional_products),
            )

        writer.add_attribute(
            "isCarPassenger",
            "java.lang.Boolean",
            writer.true_false(getattr(p, "is_car_passenger", False)),
        )
        # Keep assumptions discoverable in the final population XML, not only
        # intermediate CSVs. These flags do not alter simulation subpopulation.
        for field in [
            "non_private_dorm_candidate_households",
            "non_private_home_only_households",
            "is_synthetic_household",
        ]:
            writer.add_attribute(
                field, "java.lang.Boolean", writer.true_false(getattr(p, field, False))
            )
        writer.add_attribute(
            "non_private_model_status",
            "java.lang.String",
            getattr(p, "non_private_model_status", "ordinary_models"),
        )

        # loop trip flags may or may not exist -> default False
        writer.add_attribute(
            "hasWalkLoopTrip",
            "java.lang.Boolean",
            writer.true_false(getattr(p, "has_walk_loop_trip", False)),
        )
        writer.add_attribute(
            "hasCarLoopTrip",
            "java.lang.Boolean",
            writer.true_false(getattr(p, "has_car_loop_trip", False)),
        )
        writer.add_attribute(
            "hasCarPassengerLoopTrip",
            "java.lang.Boolean",
            writer.true_false(getattr(p, "has_car_passenger_loop_trip", False)),
        )
        writer.add_attribute(
            "hasPtLoopTrip",
            "java.lang.Boolean",
            writer.true_false(getattr(p, "has_pt_loop_trip", False)),
        )
        writer.add_attribute(
            "hasBikeLoopTrip",
            "java.lang.Boolean",
            writer.true_false(getattr(p, "has_bike_loop_trip", False)),
        )

        writer.add_attribute("isFreight", "java.lang.Boolean", writer.true_false(False))

        person_type = getattr(p, "person_type", "normal")

        is_crossborder = person_type == "crossborder"
        writer.add_attribute(
            "isCrossBorder", "java.lang.Boolean", writer.true_false(is_crossborder)
        )
        is_french = person_type == ExternalPopulationConstants.person_type
        writer.add_attribute(
            ExternalPopulationConstants.is_external,
            "java.lang.Boolean",
            writer.true_false(is_french),
        )

        cross_border_od = getattr(p, "cross_border_od", None)
        if pd.notna(cross_border_od):
            writer.add_attribute(
                "crossBorderOD", "java.lang.String", str(cross_border_od)
            )

        writer.add_attribute(
            "bikeAvail",
            "java.lang.String",
            ["never", "always"][int(p.bike_availability)],
        )

        if is_crossborder:
            writer.add_attribute("subpopulation", "java.lang.String", "crossborder")

        writer.add_attribute(
            "vehicles",
            "org.matsim.vehicles.PersonVehicles",
            "{{{content}}}".format(
                content=",".join(
                    [
                        '"{mode}":"{id}"'.format(mode=v.mode, id=v.vehicle_id)
                        for v in self.vehicles
                    ]
                )
            ),
        )
        writer.end_attributes()

        # Plan
        writer.start_plan(selected=True)

        home_location = writer.location(
            p.home_x, p.home_y, "home%s" % getattr(p, "household_id", 0)
        )

        written_activities = 0

        if first_activity is None:
            for i in range(len(self.activities)):
                a = self.activities[i]
                self._write_single_activity(writer, a, home_location)
                written_activities += 1

                if not a.is_last:
                    next_a = self.activities[i + 1]
                    writer.add_leg(
                        a.following_mode, a.end_time, next_a.start_time - a.end_time
                    )

        else:
            current_activity = first_activity
            while True:
                self._write_single_activity(writer, current_activity, home_location)
                written_activities += 1

                if current_activity.is_last:
                    break

                next_activity = next(activity_iterator)
                assert p.person_id == next_activity.person_id
                writer.add_leg(
                    current_activity.following_mode,
                    current_activity.end_time,
                    next_activity.start_time - current_activity.end_time,
                )
                current_activity = next_activity

        writer.end_plan()
        writer.end_person()
        return written_activities


class FreightWriter:
    """Serialize one prepared freight plan; coordinates/times are already resolved."""

    def __init__(self, freight_agent, is_lcv=False):
        self.freight_agent = freight_agent
        self.prefix = "lcv_" if is_lcv else "freight_"
        self.vehicles = []

    def add_vehicles(self, vehicles):
        self.vehicles = vehicles

    def write(self, writer, truck=True):
        trip = self.freight_agent
        writer.start_person(self.prefix + str(trip.agent_id))
        writer.start_attributes()
        writer.add_attribute("isFreight", "java.lang.Boolean", writer.true_false(True))
        writer.add_attribute("type", "java.lang.String", trip.mode)
        writer.add_attribute("subpopulation", "java.lang.String", "freight")
        vehicles = ",".join(
            '"{mode}":"{id}"'.format(mode=v.mode, id=v.vehicle_id)
            for v in self.vehicles
        )
        writer.add_attribute(
            "vehicles", "org.matsim.vehicles.PersonVehicles", "{" + vehicles + "}"
        )
        writer.end_attributes()
        writer.start_plan(selected=True)
        origin = writer.location(trip.origin_x, trip.origin_y, None)
        destination = writer.location(trip.destination_x, trip.destination_y, None)
        writer.start_activity("freight_loading", origin, 0, trip.departure_time)
        writer.start_attributes()
        writer.end_attributes()
        writer.end_activity()
        writer.add_leg(
            trip.mode, trip.departure_time, trip.arrival_time - trip.departure_time
        )
        writer.start_activity(
            "freight_unloading", destination, trip.arrival_time, trip.day_end_time
        )
        writer.start_attributes()
        writer.end_attributes()
        writer.end_activity()
        writer.end_plan()
        writer.end_person()
