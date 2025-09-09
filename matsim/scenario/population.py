import gzip
import io
import itertools

import numpy as np
import pandas as pd
import geopandas as gpd

import matsim.writers
from matsim.writers import backlog_iterator

def configure(context):
    context.stage("synthesis.population.enriched")
    context.stage("synthesis.population.trips")
    context.stage("synthesis.population.activities")
    context.stage("synthesis.population.spatial.locations")
    context.stage("data.spatial.cantons")
    context.config("use_freight", default=False)
    context.stage("synthesis.freight.trips")

    context.stage("synthesis.vehicles.vehicles")
    context.stage("data.spatial.municipality_types")
    context.stage("data.spatial.municipalities")

VEHICLE_FIELDS = [
    "mode", "vehicle_id", "owner_id"
]

class PersonWriter:
    def __init__(self, person):
        self.person = person
        self.activities = []
        self.vehicles = []

    def add_activity(self, activity):
        self.activities.append(activity)
    
    def add_vehicles(self, vehicles):
        self.vehicles = vehicles

    def write(self, writer):
        writer.start_person(str(self.person[0]))

        # Attributes
        writer.start_attributes()
        writer.add_attribute("age", "java.lang.Integer", str(int(self.person[1])))
        writer.add_attribute("employed", "java.lang.Boolean", writer.true_false(self.person[3]))
        writer.add_attribute("hasLicense", "java.lang.String", writer.yes_no(self.person[4]))
        writer.add_attribute("sex", "java.lang.String", ["m", "f"][self.person[5]])
        writer.add_attribute("home_coordinate_x", "java.lang.Double", str(self.person[6]))
        writer.add_attribute("home_coordinate_y", "java.lang.Double", str(self.person[7]))
        writer.add_attribute("carAvail", "java.lang.String", ["always", "sometimes", "never"][int(self.person[2])])
        writer.add_attribute("ptHasGA", "java.lang.Boolean", writer.true_false(self.person[8]))
        writer.add_attribute("ptHasHalbtax", "java.lang.Boolean", writer.true_false(self.person[9]))
        writer.add_attribute("ptHasVerbund", "java.lang.Boolean", writer.true_false(self.person[10]))
        writer.add_attribute("ptHasStrecke", "java.lang.Boolean", writer.true_false(self.person[11]))
        writer.add_attribute("ptHasGleis7", "java.lang.Boolean", writer.true_false(self.person[24]))
        writer.add_attribute("ptHasJunior", "java.lang.Boolean", writer.true_false(self.person[25]))
        writer.add_attribute("isCarPassenger", "java.lang.Boolean", writer.true_false(self.person[13]))
        writer.add_attribute("hasWalkLoopTrip", "java.lang.Boolean", writer.true_false(self.person[18]))
        writer.add_attribute("hasCarLoopTrip", "java.lang.Boolean", writer.true_false(self.person[19]))
        writer.add_attribute("hasCarPassengerLoopTrip", "java.lang.Boolean", writer.true_false(self.person[20]))
        writer.add_attribute("hasPtLoopTrip", "java.lang.Boolean", writer.true_false(self.person[21]))
        writer.add_attribute("hasBikeLoopTrip", "java.lang.Boolean", writer.true_false(self.person[22]))
        writer.add_attribute("statpopPersonId", "java.lang.Long", str(self.person[14]))
        writer.add_attribute("statpopHouseholdId", "java.lang.Long", str(self.person[15]))


        writer.add_attribute("mzPersonId", "java.lang.Long", str(self.person[16]))
        writer.add_attribute("mzHeadId", "java.lang.Long", str(self.person[17]))
        writer.add_attribute("isFreight", "java.lang.Boolean", writer.true_false(False))
        writer.add_attribute("vehicles", "org.matsim.vehicles.PersonVehicles", "{{{content}}}".format(content = ",".join([
                "\"{mode}\":\"{id}\"".format(mode = v[VEHICLE_FIELDS.index("mode")], id = v[VEHICLE_FIELDS.index("vehicle_id")])
                for v in self.vehicles
            ])))
        writer.end_attributes()

        # Plan
        writer.start_plan(selected=True)

        home_location = writer.location(self.activities[0][7].x, self.activities[0][7].y, "home%s" % self.person[12])

        for i in range(len(self.activities)):
            activity = self.activities[i]
            geometry = activity[7]
            destination_id = activity[8]
            location = home_location if destination_id == -1 else writer.location(geometry.x, geometry.y,
                                                                                  int(destination_id))

            start_time = activity[2] if not np.isnan(activity[2]) else None
            end_time = activity[3] if not np.isnan(activity[3]) else None
            attributes = dict(municipalityType = activity[10], municipalityId = activity[11])
            writer.add_activity(activity[5], location, start_time, end_time, attributes = attributes)

            if not activity[6]:
                next_activity = self.activities[i + 1]
                writer.add_leg(activity[9], activity[3], next_activity[2] - activity[3])

        writer.end_plan()
        writer.end_person()


class FreightWriter:
    def __init__(self, freight_agent):
        self.freight_agent = freight_agent
        self.vehicles = []

    def add_vehicles(self, vehicles):
        self.vehicles = vehicles

    def write(self, writer):
        writer.start_person("freight_" + str(self.freight_agent[1]))
        # Attributes
        writer.start_attributes()
        writer.add_attribute("isFreight", "java.lang.Boolean", writer.true_false(True))
        writer.add_attribute("type", "java.lang.String", str(self.freight_agent[7]))
        writer.add_attribute("subpopulation", "java.lang.String", "freight")

        writer.add_attribute("vehicles", "org.matsim.vehicles.PersonVehicles", "{{{content}}}".format(content = ",".join([
                "\"{mode}\":\"{id}\"".format(mode = v[VEHICLE_FIELDS.index("mode")], id = v[VEHICLE_FIELDS.index("vehicle_id")])
                for v in self.vehicles
            ])))

        writer.end_attributes()

        # Plan
        writer.start_plan(selected=True)

        start_location = writer.location(self.freight_agent[2], self.freight_agent[3], None)
        end_location = writer.location(self.freight_agent[4], self.freight_agent[5], None)
        departure_time = self.freight_agent[6]
        arrival_time = departure_time + 3600

        # loading activity
        writer.start_activity("freight_loading", start_location, 0, departure_time)
        writer.start_attributes()
        writer.end_attributes()
        writer.end_activity()

        # transport leg
        writer.add_leg(str(self.freight_agent[7]), departure_time, arrival_time - departure_time)

        # unloading activity
        writer.start_activity("freight_unloading", end_location, arrival_time, 30 * 3600)
        writer.start_attributes()
        writer.end_attributes()
        writer.end_activity()

        writer.end_plan()
        writer.end_person()


PERSON_FIELDS = ["person_id", "age", "car_availability", "employed", "driving_license", "sex", "home_x", "home_y",
                 "subscriptions_ga", "subscriptions_halbtax", "subscriptions_verbund", "subscriptions_strecke",
                 "household_id", "is_car_passenger", "statpop_person_id", "statpop_household_id", "mz_person_id",
                 "mz_head_id", "has_walk_loop_trip", "has_car_loop_trip", "has_car_passenger_loop_trip", "has_pt_loop_trip", "has_bike_loop_trip",
                 "mz_head_id","subscriptions_gleis7","subscriptions_junior"]
ACTIVITY_FIELDS = ["person_id", "activity_index", "start_time", "end_time", "duration", "purpose", "is_last",
                   "geometry", "destination_id", "following_mode", "municipality_type","municipality_id"]


def execute(context):
    cache_path    = context.path()
    df_persons    = context.stage("synthesis.population.enriched")
    df_activities = context.stage("synthesis.population.activities")
    df_vehicles   = context.stage("synthesis.vehicles.vehicles")[1]
    df_vehicles = context.stage("synthesis.vehicles.vehicles")[1]
    df_municipality_type = context.stage("data.spatial.municipality_types")
    df_municipalities,_ = context.stage("data.spatial.municipalities")

    # Attach following modes to activities
    df_trips         = pd.DataFrame(context.stage("synthesis.population.trips"), copy=True)[["person_id", "trip_index", "mode"]]
    df_trips.columns = ["person_id", "activity_index", "following_mode"]
    df_activities    = pd.merge(df_activities, df_trips, on=["person_id", "activity_index"], how="left")

    # Attach locations to activities
    df_locations  = context.stage("synthesis.population.spatial.locations")
    df_activities = pd.merge(df_activities, df_locations, on=["person_id", "activity_index"], how="left")

    # Attach municipality to activities (TODO: Maybe this can be done in previous stages by keeping track of municipality id)
    df_municipalities = df_municipalities.merge(df_municipality_type)[["municipality_type","municipality_id", "geometry"]]
    df_activities = gpd.GeoDataFrame(df_activities, geometry="geometry", crs="EPSG:2056")
    assert df_activities.crs == df_municipalities.crs
    df_activities = gpd.sjoin_nearest(df_activities, df_municipalities, how="left").drop(columns=["index_right"]) # way faster than sjoin   


    # Replace the primary-secondary purposes with normal ones
    # Now that the secondary locations are assigned, no need to continue working with these purposes
    df_activities["purpose"] = df_activities["purpose"].replace({"home_secondary":"home",
                                                                 "work_secondary": "work",
                                                                 "education_secondary":"education"})
    
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
    
    df_persons    = df_persons[PERSON_FIELDS]
    df_activities = df_activities[ACTIVITY_FIELDS]
    df_vehicles   = df_vehicles[VEHICLE_FIELDS]
    
    person_iterator   = iter(df_persons.itertuples(index = False))
    activity_iterator = iter(df_activities.itertuples(index = False))
    vehicle_iterator  = backlog_iterator(iter(df_vehicles.itertuples(index = False)))

    number_of_written_persons    = 0
    number_of_written_activities = 0

    with gzip.open("%s/population.xml.gz" % cache_path, "wb+", compresslevel=1) as f:
        with io.BufferedWriter(f, buffer_size=1024 * 1024 * 1024 * 2) as raw_writer:
            writer = matsim.writers.PopulationWriter(raw_writer)
            writer.start_population()

            with context.progress(total=len(df_persons), label="Writing persons ...") as progress:
                try:
                    while True:
                        person = next(person_iterator)
                        person_id = person[PERSON_FIELDS.index("person_id")]
                        is_last = False

                        person_writer = PersonWriter(person)
                        vehicles = []
                        while not is_last:
                            activity = next(activity_iterator)

                            is_last = activity[6]
                            assert (person[0] == activity[0])

                            person_writer.add_activity(activity)
                            number_of_written_activities += 1
                        # Track all vehicles for person
                        while vehicle_iterator.has_next():
                            vehicle = vehicle_iterator.next()

                            if not vehicle[VEHICLE_FIELDS.index("owner_id")] == person_id:
                                vehicle_iterator.previous()
                                break
                            else:
                                vehicles.append(vehicle)
                        person_writer.add_vehicles(vehicles)
                        person_writer.write(writer)
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

            writer.end_population()

    return "%s/population.xml.gz" % cache_path
