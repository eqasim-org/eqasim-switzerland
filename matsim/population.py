import gzip
import io

import numpy as np
import pandas as pd

import matsim.writers


def configure(context):
    context.stage("synthesis.population.enriched")
    context.stage("synthesis.population.trips")
    context.stage("synthesis.population.activities")
    context.stage("synthesis.population.spatial.locations")
    context.config("use_freight", default=False)
    context.stage("synthesis.freight.trips")


class PersonWriter:
    def __init__(self, person):
        self.person = person
        self.activities = []

    def add_activity(self, activity):
        self.activities.append(activity)

    def write(self, writer):
        writer.start_person(str(self.person[1]))

        # Attributes
        writer.start_attributes()
        writer.add_attribute("age", "java.lang.Integer", str(self.person[2]))
        writer.add_attribute("employed", "java.lang.Boolean", writer.true_false(self.person[4]))
        writer.add_attribute("hasLicense", "java.lang.String", writer.yes_no(self.person[5]))
        writer.add_attribute("sex", "java.lang.String", ["m", "f"][self.person[6]])
        writer.add_attribute("carAvail", "java.lang.String", ["always", "sometimes", "never"][int(self.person[3])])
        writer.add_attribute("ptHasGA", "java.lang.Boolean", writer.true_false(self.person[9]))
        writer.add_attribute("ptHasHalbtax", "java.lang.Boolean", writer.true_false(self.person[10]))
        writer.add_attribute("ptHasVerbund", "java.lang.Boolean", writer.true_false(self.person[11]))
        writer.add_attribute("ptHasStrecke", "java.lang.Boolean", writer.true_false(self.person[12]))
        writer.add_attribute("isCarPassenger", "java.lang.Boolean", writer.true_false(self.person[14]))
        writer.add_attribute("statpopPersonId", "java.lang.Long", str(self.person[15]))
        writer.add_attribute("statpopHouseholdId", "java.lang.Long", str(self.person[16]))
        writer.add_attribute("mzPersonId", "java.lang.Long", str(self.person[17]))
        writer.add_attribute("mzHeadId", "java.lang.Long", str(self.person[18]))
        writer.add_attribute("isFreight", "java.lang.Boolean", writer.true_false(False))

        writer.end_attributes()

        # Plan
        writer.start_plan(selected=True)

        home_location = writer.location(self.activities[0][8].x, self.activities[0][8].y, "home%s" % self.person[13])

        for i in range(len(self.activities)):
            activity = self.activities[i]
            geometry = activity[8]
            destination_id = activity[9]
            location = home_location if destination_id == -1 else writer.location(geometry.x, geometry.y,
                                                                                  int(destination_id))

            start_time = activity[3] if not np.isnan(activity[3]) else None
            end_time = activity[4] if not np.isnan(activity[4]) else None

            writer.add_activity(activity[6], location, start_time, end_time)

            if not activity[7]:
                next_activity = self.activities[i + 1]
                writer.add_leg(activity[10], activity[4], next_activity[3] - activity[4])

        writer.end_plan()
        writer.end_person()


class FreightWriter:
    def __init__(self, freight_agent):
        self.freight_agent = freight_agent

    def write(self, writer):
        writer.start_person("freight_" + str(self.freight_agent[1]))

        # Attributes
        writer.start_attributes()
        writer.add_attribute("isFreight", "java.lang.Boolean", writer.true_false(True))
        writer.add_attribute("type", "java.lang.String", str(self.freight_agent[7]))
        writer.add_attribute("subpopulation", "java.lang.String", "freight")
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
                 "mz_head_id"]
ACTIVITY_FIELDS = ["person_id", "activity_index", "start_time", "end_time", "duration", "purpose", "is_last",
                   "geometry", "destination_id", "following_mode"]


def execute(context):
    cache_path = context.cache_path
    df_persons = context.stage("synthesis.population.enriched")
    df_activities = context.stage("synthesis.population.activities")

    # Attach following modes to activities
    df_trips = pd.DataFrame(context.stage("synthesis.population.trips"), copy=True)[["person_id", "trip_index", "mode"]]
    df_trips.columns = ["person_id", "activity_index", "following_mode"]
    df_activities = pd.merge(df_activities, df_trips, on=["person_id", "activity_index"], how="left")

    # Attach locations to activities
    df_locations = context.stage("synthesis.population.spatial.locations")
    df_activities = pd.merge(df_activities, df_locations, on=["person_id", "activity_index"], how="left")

    # Bring in correct order (although it should already be)
    df_persons = df_persons.sort_values(by="person_id")
    df_activities = df_activities.sort_values(by=["person_id", "activity_index"])

    df_persons = df_persons[PERSON_FIELDS]
    df_activities = df_activities[ACTIVITY_FIELDS]

    person_iterator = iter(df_persons.itertuples())
    activity_iterator = iter(df_activities.itertuples())

    number_of_written_persons = 0
    number_of_written_activities = 0

    with gzip.open("%s/population.xml.gz" % cache_path, "w+") as f:
        with io.BufferedWriter(f, buffer_size=1024 * 1024 * 1024 * 2) as raw_writer:
            writer = matsim.writers.PopulationWriter(raw_writer)
            writer.start_population()

            with context.progress(total=len(df_persons), label="Writing persons ...") as progress:
                try:
                    while True:
                        person = next(person_iterator)
                        is_last = False

                        person_writer = PersonWriter(person)

                        while not is_last:
                            activity = next(activity_iterator)
                            is_last = activity[7]
                            assert (person[1] == activity[1])

                            person_writer.add_activity(activity)
                            number_of_written_activities += 1

                        person_writer.write(writer)
                        number_of_written_persons += 1
                        progress.update()
                except StopIteration:
                    pass

            assert (number_of_written_activities == len(df_activities))
            assert (number_of_written_persons == len(df_persons))

            if context.config("use_freight"):
                df_freight = context.stage("synthesis.freight.trips")

                freight_iterator = iter(df_freight.itertuples())
                number_of_written_freight = 0

                with context.progress(total=len(df_freight), label="Writing freight agents ...") as progress:
                    try:
                        while True:
                            freight = next(freight_iterator)
                            freight_writer = FreightWriter(freight)
                            freight_writer.write(writer)
                            number_of_written_freight += 1
                            progress.update()
                    except StopIteration:
                        pass

                assert (number_of_written_freight == len(df_freight))

            writer.end_population()

    return "%s/population.xml.gz" % cache_path
