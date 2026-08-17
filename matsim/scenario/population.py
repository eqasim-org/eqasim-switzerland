import gzip
import io
import shutil
import subprocess
import pandas as pd
import geopandas as gpd
from shapely import wkt

import matsim.writers
from matsim.writers import backlog_iterator
import logging
from data.external_population.constants import ExternalPopulationConstants
from synthesis.population.departure_times.trips_departures import get_best_departue_time

logger = logging.getLogger("synpp")

# If the activity has this id, the written location will be the home location of the person. 
# This was already there, I just want to make it a variable, to keep consistency in the pipeline in case this is changed somewhere.
HOME_DESTINATION_ID = -1

def _require_cols(df, cols, df_name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{df_name} is missing required columns: {missing}")


def _na_to_none(x):
    return None if pd.isna(x) else x


def snap_to_shared_facilities(external_activities, df_destinations):
    """
    Moves an external activity onto the facility it names whenever that facility
    is also part of synthesis.population.destinations.

    Both sides describe the same French BPE (or Swiss STATENT) locations, but they
    reach them through different paths: data.locations_fr.secondary snaps the point
    to a 1 m grid, while data.external_population.read_outputs reprojects the raw
    coordinate from EPSG:2154 and lets matsim.scenario.population truncate it. The
    two disagree by up to a metre. matsim/scenario/facilities.py writes the shared
    ids only once, from df_destinations, so the activity has to follow -- MATSim's
    ScenarioValidator rejects an activity that is not exactly on its facility.
    """

    coordinates = df_destinations[["destination_id", "destination_x", "destination_y"]].drop_duplicates("destination_id")
    coordinates = coordinates.set_index("destination_id")

    shared = external_activities["destination_id"].isin(coordinates.index)

    if shared.any():
        x = external_activities.loc[shared, "destination_id"].map(coordinates["destination_x"]).astype(float)
        y = external_activities.loc[shared, "destination_id"].map(coordinates["destination_y"]).astype(float)

        external_activities.loc[shared, "destination_x"] = x.values
        external_activities.loc[shared, "destination_y"] = y.values
        external_activities.loc[shared, "geometry"]      = gpd.points_from_xy(x.values, y.values)

        logger.info(
            "Moved %d external activities (%d distinct facilities) onto the coordinates written to facilities.xml.gz.",
            int(shared.sum()), int(external_activities.loc[shared, "destination_id"].nunique()),
        )

    return external_activities


def configure(context):
    context.stage("synthesis.population.models.subscriptions")
    context.stage("synthesis.population.trips")
    context.stage("synthesis.population.activities")
    context.stage("synthesis.population.spatial.locations")
    context.stage("data.spatial.cantons")
    context.config("use_freight", default = False)
    context.config("use_lcv", default = False)
    context.stage("synthesis.freight.trips")
    context.stage("synthesis.lcv.trips")
    context.stage("data.constants")

    context.stage("synthesis.vehicles.vehicles")
    context.config("population_compresslevel", default=1)
    context.config("population_use_pigz", default=True)
    context.config("population_pigz_threads", default=8)

    context.config("include_cross_border", default = False)
    if context.config("include_cross_border"):
        context.stage("data.cross_border.generate_cross_border_traffic")

    context.config("include_external_population", default = False)
    if context.config("include_external_population"):
        context.stage("data.external_population.read_outputs")
        context.stage("data.external_population.constants")
        context.stage("synthesis.population.destinations")

    context.config("correct_departure_time", default=True)
    if context.config("correct_departure_time"):
        context.stage("synthesis.population.departure_times.trips_departures")


VEHICLE_FIELDS = ["mode", "vehicle_id", "owner_id"]

ACTIVITY_ATTRIBUTES_TO_SAVE = dict(municipalityType  = "municipality_type", 
                                   municipalityId    = "municipality_id", 
                                   employeeDensity   = "employee_density", 
                                   companiesDensity  = "companies_density", 
                                   populationDensity = "population_density", 
                                   ovgk              = "ovgk")


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
            if destination_id == HOME_DESTINATION_ID or destination_id==str(HOME_DESTINATION_ID)
            else writer.location(int(geometry.x), int(geometry.y), destination_id if isinstance(destination_id, str) else int(destination_id))
        )

        start_time = _na_to_none(activity.start_time)
        end_time = _na_to_none(activity.end_time)

        attributes = {
            "municipalityType": activity.municipality_type,
            "municipalityId": activity.municipality_id,
            "employeeDensity": activity.employee_density,
            "companiesDensity": activity.companies_density,
            "populationDensity": activity.population_density,
            "ovgk": activity.ovgk,
        }

        writer.add_activity(activity.purpose, location, start_time, end_time, attributes=attributes)


    def write(self, writer, first_activity=None, activity_iterator=None):
        p = self.person

        writer.start_person(str(p.person_id))

        # Attributes
        writer.start_attributes()
        writer.add_attribute("age", "java.lang.Integer", str(int(p.age)))
        writer.add_attribute("employed", "java.lang.Boolean", writer.true_false(p.employed))
        writer.add_attribute("hasLicense", "java.lang.String", writer.yes_no(p.driving_license))
        writer.add_attribute("sex", "java.lang.String", ["m", "f"][p.sex])
        writer.add_attribute("home_coordinate_x", "java.lang.Double", str(p.home_x))
        writer.add_attribute("home_coordinate_y", "java.lang.Double", str(p.home_y))
        writer.add_attribute("carAvail", "java.lang.String", ["never", "always"][int(p.car_availability)])

        writer.add_attribute("subscriptions", "java.lang.String", ["none", "GA", "VA", "HT", "VA+HT"][int(p.pt_subscription)])

        writer.add_attribute("isCarPassenger", "java.lang.Boolean", writer.true_false(getattr(p, "is_car_passenger", False)))

        # loop trip flags may or may not exist -> default False
        writer.add_attribute("hasWalkLoopTrip", "java.lang.Boolean", writer.true_false(getattr(p, "has_walk_loop_trip", False)))
        writer.add_attribute("hasCarLoopTrip", "java.lang.Boolean", writer.true_false(getattr(p, "has_car_loop_trip", False)))
        writer.add_attribute("hasCarPassengerLoopTrip", "java.lang.Boolean", writer.true_false(getattr(p, "has_car_passenger_loop_trip", False)))
        writer.add_attribute("hasPtLoopTrip", "java.lang.Boolean", writer.true_false(getattr(p, "has_pt_loop_trip", False)))
        writer.add_attribute("hasBikeLoopTrip", "java.lang.Boolean", writer.true_false(getattr(p, "has_bike_loop_trip", False)))

        writer.add_attribute("isFreight", "java.lang.Boolean", writer.true_false(False))

        person_type = getattr(p, "person_type", "normal")
        
        is_crossborder = (person_type == "crossborder")
        writer.add_attribute("isCrossBorder", "java.lang.Boolean", writer.true_false(is_crossborder))
        is_french     = (person_type == ExternalPopulationConstants.person_type)
        writer.add_attribute(ExternalPopulationConstants.is_external, "java.lang.Boolean", writer.true_false(is_french))

        cross_border_od = getattr(p, "cross_border_od", None)
        if pd.notna(cross_border_od):
            writer.add_attribute("crossBorderOD", "java.lang.String", str(cross_border_od))

        writer.add_attribute("bikeAvail", "java.lang.String", ["never", "always"][int(p.bike_availability)])

        if is_crossborder:
            writer.add_attribute("subpopulation", "java.lang.String", "crossborder")

        writer.add_attribute(
            "vehicles",
            "org.matsim.vehicles.PersonVehicles",
            "{{{content}}}".format(content=",".join([
                "\"{mode}\":\"{id}\"".format(mode=v.mode, id=v.vehicle_id)
                for v in self.vehicles
            ]))
        )
        writer.end_attributes()

        # Plan
        writer.start_plan(selected=True)

        home_location = writer.location(p.home_x, p.home_y, "home%s" % getattr(p, "household_id", 0))

        written_activities = 0

        if first_activity is None:
            for i in range(len(self.activities)):
                a = self.activities[i]
                self._write_single_activity(writer, a, home_location)
                written_activities += 1

                if not a.is_last:
                    next_a = self.activities[i + 1]
                    writer.add_leg(a.following_mode, a.end_time, next_a.start_time - a.end_time)

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
    def __init__(self, freight_agent, is_lcv = False):
        self.freight_agent = freight_agent
        self.vehicles = []
        if is_lcv:
            self.prefix = "lcv_"
        else:
            self.prefix = "freight_"


    def add_vehicles(self, vehicles):
        self.vehicles = vehicles


    def write(self, writer, truck=True):
        writer.start_person(self.prefix + str(self.freight_agent[1]))
        # Attributes
        writer.start_attributes()
        writer.add_attribute("isFreight", "java.lang.Boolean", writer.true_false(True))
        
        if (truck):
            writer.add_attribute("type", "java.lang.String", str(self.freight_agent[7]))
        else:
            writer.add_attribute("type", "java.lang.String", "truck")

        writer.add_attribute("subpopulation", "java.lang.String", "freight")

        writer.add_attribute("vehicles", "org.matsim.vehicles.PersonVehicles", "{{{content}}}".format(content = ",".join([
                "\"{mode}\":\"{id}\"".format(mode = v[VEHICLE_FIELDS.index("mode")], id = v[VEHICLE_FIELDS.index("vehicle_id")])
                for v in self.vehicles
            ])))

        writer.end_attributes()

        # Plan
        writer.start_plan(selected=True)
        if (truck):
            start_location = writer.location(self.freight_agent[2], self.freight_agent[3], None)
            end_location = writer.location(self.freight_agent[4], self.freight_agent[5], None)
            departure_time = self.freight_agent[6]
        else:
            start_location = writer.location(self.freight_agent[4], self.freight_agent[5], None)
            end_location = writer.location(self.freight_agent[6], self.freight_agent[7], None)
            departure_time = self.freight_agent[8]
        arrival_time = departure_time + 3600

        # loading activity
        writer.start_activity("freight_loading", start_location, 0, departure_time)
        writer.start_attributes()
        writer.end_attributes()
        writer.end_activity()

        # transport leg
        if (truck):
            writer.add_leg(str(self.freight_agent[7]), departure_time, arrival_time - departure_time)
        else:
            writer.add_leg("truck", departure_time, arrival_time - departure_time)

        # unloading activity
        writer.start_activity("freight_unloading", end_location, arrival_time, 30 * 3600)
        writer.start_attributes()
        writer.end_attributes()
        writer.end_activity()

        writer.end_plan()
        writer.end_person()


PERSON_FIELDS = ["person_id", "age", "car_availability", "employed", "driving_license", "sex", 
                 "home_x", "home_y",
                 "pt_subscription", 
                 "household_id", "is_car_passenger", 
                 "has_walk_loop_trip", "has_car_loop_trip", "has_car_passenger_loop_trip", "has_pt_loop_trip", "has_bike_loop_trip",
                 "income_class", "person_type", "cross_border_od",
                 "bike_availability"]


ACTIVITY_FIELDS = ["person_id", "activity_index", "start_time", "end_time", "duration", "purpose", "is_last",
                   "geometry", "destination_id", "following_mode", "municipality_type","municipality_id",
                   "employee_density", "companies_density", "population_density", "ovgk"]


PERSONS_DTYPES = {
        "person_id": int,
        "age": int,
        "car_availability": int,
        "bike_availability": int,
        "employed": bool,
        "driving_license": bool,
        "sex": int,
        "home_x": float,
        "home_y": float,
        "pt_subscription": int,
        "household_id": int,
        "is_car_passenger": bool,
        "has_walk_loop_trip": bool,
        "has_car_loop_trip": bool,
        "has_car_passenger_loop_trip": bool,
        "has_pt_loop_trip": bool,
        "has_bike_loop_trip": bool,
        "income_class": int,
        "person_type": str,
    }


def execute(context):
    cache_path    = context.path()
    cst           = context.stage("data.constants")
    df_persons    = context.stage("synthesis.population.models.subscriptions")
    df_activities = context.stage("synthesis.population.activities")
    df_vehicles   = context.stage("synthesis.vehicles.vehicles")[1]  

    # Correct employement if required
    if set(df_persons["employed"].unique())!={0, 1}:
        df_persons["employed"] = (pd.to_numeric(df_persons["employed"], errors="coerce").fillna(0) == cst.EMPLOYED).astype(int)

    # Attach following modes to activities
    df_trips         = pd.DataFrame(context.stage("synthesis.population.trips"), copy=True)[["person_id", "trip_index", "mode"]]
    df_trips.columns = ["person_id", "activity_index", "following_mode"]
    df_activities    = pd.merge(df_activities, df_trips, on=["person_id", "activity_index"], how="left")

    # correct departure times
    if context.config("correct_departure_time"):
        df = df_activities[["person_id","activity_index", "end_time"]]
        df.columns = ['person_id', 'trip_index', 'end_time']
        df['new_end_time'] = get_best_departue_time(context, df)
        df['new_end_time'] = df['new_end_time'].fillna(df['end_time'])
        df_activities['end_time'] = df['new_end_time']
        del df

    # Attach locations to activities
    df_locations  = context.stage("synthesis.population.spatial.locations")
    df_activities = pd.merge(df_activities, df_locations, on=["person_id", "activity_index"], how="left")
        
    # Find the loop attribute in the following_modes
    loop_modes = ["walk_loop", "car_loop", "car_passenger_loop", "pt_loop", "bike_loop"]

    unique_modes_per_agent = df_activities.groupby("person_id")["following_mode"].apply(lambda x: list(set(x))).reset_index()
    loop_columns           = ["person_id"]
    for mode in loop_modes:
        col = "has_" + mode + "_trip"
        unique_modes_per_agent[col] = [mode in unique_modes for unique_modes in unique_modes_per_agent["following_mode"]]
        loop_columns.append(col)

    df_persons = df_persons.merge(unique_modes_per_agent[loop_columns], on = "person_id", how="left")
    df_persons[loop_columns] = df_persons[loop_columns].fillna(False)
    
    # Bring in correct order (although it should already be)
    df_persons    = df_persons.sort_values(by="person_id")
    df_activities = df_activities.sort_values(by=["person_id", "activity_index"])
    df_vehicles   = df_vehicles.sort_values(by=["owner_id"])

    df_persons["person_type"] = "normal"
    is_crossing_the_border = df_persons["is_crossing_the_border"].astype("boolean").fillna(False).astype(bool)
    df_persons.loc[is_crossing_the_border, "person_type"] = "crossborder"

    # For Swiss residents crossing the border, destination_country_raw was already
    # assigned in synthesis.population.models.cross_border (matched to a specific
    # data.cross_border.swiss_residents_od record) and carried through
    # synthesis.population.models.subscriptions -- the same record
    # synthesis.population.spatial.locations used for the border activity's
    # location, so label and point stay consistent.
    df_persons["cross_border_od"] = None
    has_raw_od = df_persons["destination_country_raw"].notna()
    df_persons.loc[has_raw_od, "cross_border_od"] = "CH-" + df_persons.loc[has_raw_od, "destination_country_raw"]

    df_persons = df_persons.drop(columns=["destination_country_raw"])

    if context.config("include_external_population"):
        external_persons    = context.stage("data.external_population.read_outputs")[0].copy()
        external_activities = context.stage("data.external_population.read_outputs")[1].copy()
        external_vehicles   = context.stage("data.external_population.read_outputs")[2].copy()
        ex_constants        = context.stage("data.external_population.constants")

        external_persons["person_type"]       = ex_constants.person_type
        external_persons["sex"]               = ex_constants.convert_sex(external_persons["sex"]).astype(int)
        external_persons["pt_subscription"]   = ex_constants.get_subscriptions(external_persons)
        external_persons["bike_availability"] = ex_constants.convert_bike_availability(external_persons["number_of_bikes_class"],cst)
        external_persons["car_availability"]  = ex_constants.convert_car_availability(external_persons["car_availability"])

        external_persons = external_persons[external_persons["home_x"].notna()]
        external_persons = external_persons[external_persons["home_y"].notna()]
        
        external_activities["destination_id"] = external_activities["destination_id"].astype(object)
        external_activities.loc[external_activities["purpose"] == "home", "destination_id"] = HOME_DESTINATION_ID

        external_activities = snap_to_shared_facilities(
            external_activities, context.stage("synthesis.population.destinations"))

        external_activities["destination_x"] = external_activities["destination_x"].astype(int)
        external_activities["destination_y"] = external_activities["destination_y"].astype(int)

        for col in ACTIVITY_ATTRIBUTES_TO_SAVE.values():
            if not col in external_activities.columns:
                raise RuntimeError("Missing column in external activities: %s" % col)

        external_vehicles = external_vehicles[VEHICLE_FIELDS]

        df_persons    = pd.concat([df_persons, external_persons])
        df_activities = pd.concat([df_activities, external_activities])
        df_vehicles   = pd.concat([df_vehicles, external_vehicles])

        df_persons["mz_person_id"] = df_persons["mz_person_id"].astype(int)
        df_persons["home_x"]       = df_persons["home_x"].astype(int)
        df_persons["home_y"]       = df_persons["home_y"].astype(int)

        df_persons    = df_persons.sort_values(by = "person_id")
        df_activities = df_activities.sort_values(by = ["person_id", "activity_index"])
        df_vehicles   = df_vehicles.sort_values(by = ["owner_id"])

    if context.config("include_cross_border"):
        cross_border_persons    = context.stage("data.cross_border.generate_cross_border_traffic")[0].copy()
        cross_border_activities = context.stage("data.cross_border.generate_cross_border_traffic")[1].copy()
        cross_border_vehicles   = context.stage("data.cross_border.generate_cross_border_traffic")[2].copy()

        cross_border_persons    = cross_border_persons.sort_values(by="person_id")
        cross_border_activities = cross_border_activities.sort_values(by=["person_id", "activity_index"])
        cross_border_vehicles   = cross_border_vehicles.sort_values(by=["owner_id"])
        
        for col in ACTIVITY_ATTRIBUTES_TO_SAVE.values():
            if col not in cross_border_activities.columns:
                cross_border_activities[col] = 0

        cross_border_persons["person_type"]       = "crossborder"

        # These agents travel <abroad> -> CH -> <abroad>, so the od label has
        # three parts (e.g. "FR-CH-FR"). data.cross_border.generate_od orders
        # every record so that it ends in CH: "From-To" agents then carry
        # destination_country_raw == "CH" and go back to their origin country,
        # while "Through" agents keep two distinct foreign endpoints.
        origin_country      = cross_border_persons["origin_country_raw"]
        destination_country = cross_border_persons["destination_country_raw"]
        destination_country = destination_country.where(destination_country != "CH", origin_country)

        cross_border_persons["cross_border_od"] = origin_country + "-CH-" + destination_country
        cross_border_persons["pt_subscription"]   = 0
        cross_border_persons["bike_availability"] = 0
        cross_border_persons["car_availability"]  = 1

        df_persons    = pd.concat([df_persons, cross_border_persons])
        df_activities = pd.concat([df_activities, cross_border_activities])
        df_vehicles   = pd.concat([df_vehicles, cross_border_vehicles])

        df_persons["mz_person_id"] = df_persons["mz_person_id"].astype(int)
        df_persons["home_x"]       = df_persons["home_x"].astype(int)
        df_persons["home_y"]       = df_persons["home_y"].astype(int)

        df_persons    = df_persons.sort_values(by = "person_id")
        df_activities = df_activities.sort_values(by = ["person_id", "activity_index"])
        df_vehicles   = df_vehicles.sort_values(by = ["owner_id"])
    
    df_persons["is_car_passenger"] = df_persons["is_car_passenger"].fillna(False)
    
    df_persons    = df_persons[PERSON_FIELDS]
    df_activities = df_activities[ACTIVITY_FIELDS]
    df_vehicles   = df_vehicles[VEHICLE_FIELDS]
    df_vehicles["owner_id"] = df_vehicles["owner_id"].astype(int)

    # correct types before saving the data    
    df_persons = df_persons.astype(PERSONS_DTYPES)
    df_activities["geometry"] = df_activities["geometry"].apply(lambda g: wkt.loads(g) if isinstance(g, str) else g)
    valid_ids = df_activities.groupby("person_id")["geometry"].apply(
        lambda g: g.notna().all()
    )
    valid_ids = valid_ids[valid_ids].index

    # TODO was there a reason why vehicles are not impacted by this selection of valid ids?
    df_persons    = df_persons[df_persons["person_id"].isin(valid_ids)]
    df_activities = df_activities[df_activities["person_id"].isin(df_persons["person_id"].values.tolist())]
    df_vehicles   = df_vehicles[df_vehicles["owner_id"].isin(df_persons["person_id"].values.tolist())]

    # TODO check why there are multiple activities with same attributes but only different municipality_id and municipality_types.
    df_activities = df_activities.drop_duplicates(["person_id", "activity_index"], keep = "first")

    # Make sure the minimum required columns exist (order does NOT matter)
    _require_cols(df_persons, ["person_id", "age", "car_availability", "employed", "driving_license", "sex", "home_x", "home_y"], "df_persons")
    _require_cols(df_activities, ["person_id", "activity_index", "start_time", "end_time", "purpose", "is_last",
                                "geometry", "destination_id", "following_mode", "municipality_type", "municipality_id",
                                "employee_density", "companies_density", "population_density"], "df_activities")
    _require_cols(df_vehicles, ["mode", "vehicle_id", "owner_id"], "df_vehicles")

    # Cast only the columns that exist (so removing/adding columns won't break)
    df_persons = df_persons.astype({k: v for k, v in PERSONS_DTYPES.items() if k in df_persons.columns})

    df_persons    = df_persons.sort_values(by = "person_id")
    df_activities = df_activities.sort_values(by = ["person_id", "activity_index"])
    df_vehicles   = df_vehicles.sort_values(by = ["owner_id"])

    person_iterator   = iter(df_persons.itertuples(index=False, name="Person"))
    activity_iterator = iter(df_activities.itertuples(index=False, name="Activity"))
    vehicle_iterator  = backlog_iterator(iter(df_vehicles.itertuples(index=False, name="Vehicle")))

    number_of_written_persons    = 0
    number_of_written_activities = 0
    logger.info("Starting to write population")

    population_xml_path = "%s/population.xml" % cache_path
    population_gz_path  = "%s/population.xml.gz" % cache_path
    compresslevel       = int(context.config("population_compresslevel"))

    use_pigz = bool(context.config("population_use_pigz")) and shutil.which("pigz") is not None
    if bool(context.config("population_use_pigz")) and not use_pigz:
        logger.warning("population_use_pigz=True but pigz was not found in PATH. Falling back to Python gzip.")

    output_path = population_xml_path if use_pigz else population_gz_path

    open_fn = open if use_pigz else gzip.open
    open_kwargs = {} if use_pigz else {"compresslevel": compresslevel}

    # TODO check why at some point (most probably not related to cross border or external population) the vehicles and populations get un-aligned
    with open_fn(output_path, "wb+", **open_kwargs) as f:
        with io.BufferedWriter(f, buffer_size=1024 * 1024 * 1024 * 2) as raw_writer:
            writer = matsim.writers.PopulationWriter(raw_writer)
            writer.start_population()

            with context.progress(total=len(df_persons), label="Writing persons ...") as progress:
                try:
                    while True:
                        person = next(person_iterator)
                        person_id = person.person_id
                        person_writer = PersonWriter(person)
                        vehicles = []

                        first_activity = next(activity_iterator)
                        assert person.person_id == first_activity.person_id

                        # Consume vehicles in one pass (owner_id-sorted); avoids O(N_persons * N_vehicles) scans.
                        while vehicle_iterator.has_next():
                            vehicle = vehicle_iterator.next()
                            if vehicle.owner_id != person_id:
                                vehicle_iterator.previous()
                                break
                            vehicles.append(vehicle)

                        if len(vehicles) == 0:
                            vehicles = df_vehicles[df_vehicles["owner_id"] == person_id]

                        person_writer.add_vehicles(vehicles)
                        number_of_written_activities += person_writer.write(
                            writer,
                            first_activity=first_activity,
                            activity_iterator=activity_iterator,
                        )
                        number_of_written_persons += 1
                        progress.update()
                        
                except StopIteration:
                    pass

            assert (number_of_written_activities == len(df_activities))
            assert (number_of_written_persons == len(df_persons))

            if context.config("use_freight"):
                df_freight = context.stage("synthesis.freight.trips")
                df_vehicles = context.stage("synthesis.vehicles.vehicles")[2]
                df_vehicles = df_vehicles.sort_values(by=["owner_id"])
                
                df_vehicles = df_vehicles[VEHICLE_FIELDS]
                vehicle_iterator = backlog_iterator(iter(df_vehicles[VEHICLE_FIELDS].itertuples(index = False)))

                freight_iterator = iter(df_freight.itertuples())
                number_of_written_freight = 0

                with context.progress(total=len(df_freight), label="Writing freight agents ...") as progress:
                    try:
                        while True:
                            vehicles = []
                            freight = next(freight_iterator)
                            freight_writer = FreightWriter(freight)
                            owner_id = freight[1]
                            while vehicle_iterator.has_next():
                                vehicle = vehicle_iterator.next()
                                if not vehicle[VEHICLE_FIELDS.index("owner_id")] == owner_id:
                                    vehicle_iterator.previous()
                                    break
                                else:
                                    vehicles.append(vehicle)
                            freight_writer.add_vehicles(vehicles)
                            freight_writer.write(writer)
                            number_of_written_freight += 1
                            progress.update()
                    except StopIteration:
                        pass

                assert (number_of_written_freight == len(df_freight))
            
            if context.config("use_lcv"):
                df_lcv= context.stage("synthesis.lcv.trips")
                df_vehicles = context.stage("synthesis.vehicles.vehicles")[3] ## here we obtain lcv vehicles data
                df_vehicles = df_vehicles.sort_values(by=["owner_id"])                

                df_vehicles = df_vehicles[VEHICLE_FIELDS]
                vehicle_iterator = backlog_iterator(iter(df_vehicles[VEHICLE_FIELDS].itertuples(index = False)))

                lcv_iterator = iter(df_lcv.itertuples())
                number_of_written_lcv = 0

                with context.progress(total=len(df_lcv), label="Writing lcv agents ...") as progress:
                    try:
                        while True:
                            vehicles = []
                            lcv = next(lcv_iterator)
                            lcv_writer = FreightWriter(lcv, is_lcv = True)
                            owner_id = lcv[1]
                            while vehicle_iterator.has_next():
                                vehicle = vehicle_iterator.next()
                                if not vehicle[VEHICLE_FIELDS.index("owner_id")] == owner_id:
                                    vehicle_iterator.previous()
                                    break
                                else:
                                    vehicles.append(vehicle)
                            lcv_writer.add_vehicles(vehicles)
                            lcv_writer.write(writer, False)
                            number_of_written_lcv += 1
                            progress.update()
                    except StopIteration:
                        pass

                assert (number_of_written_lcv == len(df_lcv))

            writer.end_population()

    if use_pigz:
        pigz_threads = max(1, int(context.config("population_pigz_threads")))
        logger.info("Compressing population.xml with pigz using %d threads ...", pigz_threads)
        subprocess.run([
            "pigz",
            "-f",
            "-p",
            str(pigz_threads),
            "-" + str(compresslevel),
            population_xml_path,
        ], check=True)

    return population_gz_path
