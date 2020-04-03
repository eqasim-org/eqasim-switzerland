import gzip
import io

import numpy as np
import pandas as pd

import matsim.writers


def configure(context):
    context.stage("data.microcensus.persons")
    context.stage("data.microcensus.trips")
    context.stage("matsim.mz.activities")

class PersonWriter:
    def __init__(self, person):
        self.person = person
        self.activities = []

    def add_activity(self, activity):
        self.activities.append(activity)

    def write(self, writer):
        writer.start_person(self.person[1])

        # Attributes
        writer.start_attributes()
        writer.add_attribute("mzId", "java.long.Integer", str(self.person[1]))
        writer.add_attribute("age", "java.long.Integer", str(self.person[2]))
        writer.add_attribute("employed", "java.lang.Boolean", writer.true_false(self.person[4]))
        writer.add_attribute("hasLicense", "java.lang.String", writer.yes_no(self.person[5]))
        writer.add_attribute("sex", "java.lang.String", ["m", "f"][self.person[6]])
        writer.add_attribute("carAvail", "java.lang.String", ["always", "sometimes", "never"][int(self.person[3])])
        writer.add_attribute("ptHasGA", "java.lang.Boolean", writer.true_false(self.person[9]))
        writer.add_attribute("ptHasHalbtax", "java.lang.Boolean", writer.true_false(self.person[10]))
        writer.add_attribute("ptHasVerbund", "java.lang.Boolean", writer.true_false(self.person[11]))
        writer.add_attribute("ptHasStrecke", "java.lang.Boolean", writer.true_false(self.person[12]))
        writer.add_attribute("mzWeekend", "java.lang.Boolean", writer.true_false(self.person[13]))
        writer.add_attribute("mzDate", "java.lang.String", str(self.person[14]))
        writer.add_attribute("mzWeight", "java.lang.Double", str(self.person[15]))
        writer.end_attributes()

        # Plan
        writer.start_plan(selected = True)

        home_location = writer.location(x = self.activities[0][8], y = self.activities[0][9])

        for i in range(len(self.activities)):
            activity = self.activities[i]
            location = writer.location(activity[8], activity[9], None)

            start_time = activity[3] if not np.isnan(activity[3]) else None
            end_time = activity[4] if not np.isnan(activity[4]) else None

            writer.add_activity(activity[6], location, start_time, end_time)

            if not activity[7]:
                next_activity = self.activities[i + 1]
                writer.add_leg(activity[10], activity[4], next_activity[3] - activity[4])

        writer.end_plan()
        writer.end_person()

PERSON_FIELDS = ["person_id", "age", "car_availability", "employed", "driving_license", "sex", "home_x", "home_y", "subscriptions_ga", "subscriptions_halbtax", "subscriptions_verbund", "subscriptions_strecke", "weekend", "date", "person_weight"]
ACTIVITY_FIELDS = ["person_id", "activity_id", "start_time", "end_time", "duration", "purpose", "is_last", "location_x", "location_y", "following_mode"]

def execute(context):
    cache_path = context.cache_path
    df_persons = context.stage("data.microcensus.persons")
    df_activities = context.stage("matsim.mz.activities")

    # Attach following modes to activities
    df_trips = pd.DataFrame(context.stage("data.microcensus.trips"), copy = True)[["person_id", "trip_id", "mode"]]
    df_trips.columns = ["person_id", "activity_id", "following_mode"]
    df_activities = pd.merge(df_activities, df_trips, on = ["person_id", "activity_id"], how = "left")

    # Bring in correct order (although it should already be)
    df_persons = df_persons.sort_values(by = "person_id")
    df_activities = df_activities.sort_values(by = ["person_id", "activity_id"])

    df_persons = df_persons[PERSON_FIELDS]
    df_activities = df_activities[ACTIVITY_FIELDS]

    person_iterator = iter(df_persons.itertuples())
    activity_iterator = iter(df_activities.itertuples())

    number_of_written_persons = 0
    number_of_written_activities = 0

    with gzip.open("%s/population.xml.gz" % cache_path, "w+") as f:
        with io.BufferedWriter(f, buffer_size = 1024  * 1024 * 1024 * 2) as raw_writer:
            writer = matsim.writers.PopulationWriter(raw_writer)
            writer.start_population()

            with context.progress(total = len(df_persons), label = "Writing persons ...") as progress:
                try:
                    while True:
                        person = next(person_iterator)
                        is_last = False

                        person_writer = PersonWriter(person)

                        while not is_last:
                            activity = next(activity_iterator)
                            is_last = activity[7]
                            assert(person[1] == activity[1])

                            person_writer.add_activity(activity)
                            number_of_written_activities += 1

                        person_writer.write(writer)
                        number_of_written_persons += 1
                        progress.update()
                except StopIteration:
                    pass

            writer.end_population()

            assert(number_of_written_activities == len(df_activities))
            assert(number_of_written_persons == len(df_persons))

    return "%s/population.xml.gz" % cache_path
