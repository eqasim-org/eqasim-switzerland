import gzip
from tqdm import tqdm
import data.constants as c
import numpy as np
import io
import matsim.writers

def configure(context, require):
    require.stage("population.sociodemographics")
    require.stage("population.trips")

class PersonWriter:
    def __init__(self, person):
        self.person = person
        self.trips = []

    def add_trip(self, trip):
        self.trips.append(trip)

    def write(self, writer):
         # Here we filter out young person without actvity chain
        if self.person[2] >= c.MZ_AGE_THRESHOLD:
            writer.start_person(self.person[1])

            # Attributes
            writer.start_attributes()
            writer.add_attribute("age", "java.long.Integer", str(self.person[2]))
            writer.add_attribute("employed", "java.lang.Boolean", writer.true_false(self.person[4]))
            writer.add_attribute("hasLicense", "java.lang.String", writer.yes_no(self.person[5]))
            writer.add_attribute("sex", "java.lang.String", ["m", "f"][self.person[6]])
            writer.add_attribute("carAvail", "java.lang.String", ["always", "sometimes", "never"][int(self.person[3])])
            writer.add_attribute("ptHasGA", "java.lang.Boolean", writer.true_false(self.person[9]))
            writer.add_attribute("ptHasHalbtax", "java.lang.Boolean", writer.true_false(self.person[10]))
            writer.add_attribute("ptHasVerbund", "java.lang.Boolean", writer.true_false(self.person[11]))
            writer.add_attribute("ptHasStrecke", "java.lang.Boolean", writer.true_false(self.person[12]))
            writer.end_attributes()

            # Plan
            writer.start_plan(selected = True)

            home_location = writer.location(x = self.person[7], y = self.person[8])
            location = home_location
            activity_type = "home"
            end_time = None
            start_time = None
            first = True

            for trip in self.trips:
                end_time = trip[3]

                writer.add_activity(activity_type, location, start_time, end_time)
                writer.add_leg(trip[5], trip[3], trip[4] - trip[3])

                activity_type = trip[6]
                location = writer.location(trip[7], trip[8], trip[9]) if not np.isnan(trip[7]) else home_location
                start_time = trip[4]

            writer.add_activity(activity_type, location, start_time)

            writer.end_plan()
            writer.end_person()

PERSON_FIELDS = ["person_id", "age", "car_availability", "employed", "driving_license", "sex", "home_x", "home_y", "subscriptions_ga", "subscriptions_halbtax", "subscriptions_verbund", "subscriptions_strecke"]
TRIP_FIELDS = ["person_id", "trip_id", "departure_time", "arrival_time", "mode", "purpose", "location_x", "location_y", "location_id"]

def execute(context):
    cache_path = context.cache_path
    df_persons = context.stage("population.sociodemographics")
    df_trips = context.stage("population.trips")

    df_persons = df_persons.sort_values(by = "person_id")
    df_trips = df_trips.sort_values(by = ["person_id", "trip_id"])

    df_persons = df_persons[PERSON_FIELDS]
    df_trips = df_trips[TRIP_FIELDS]

    person_iterator = iter(df_persons.itertuples())
    trip_iterator = iter(df_trips.itertuples())

    with gzip.open("%s/population.xml.gz" % cache_path, "w+") as f:
        with io.BufferedWriter(f, buffer_size = 1024  * 1024 * 1024 * 2) as raw_writer:
            writer = matsim.writers.PopulationWriter(raw_writer)
            writer.start_population()

            person_writer = None

            number_of_processed_trips = 1
            number_of_processed_persons = 1

            with tqdm(total = len(df_persons)) as progress:
                try:
                    person = next(person_iterator)
                    trip = next(trip_iterator)

                    person_writer = PersonWriter(person)
                    person_writer.add_trip(trip)

                    while True:
                        while True:
                            trip = next(trip_iterator)
                            number_of_processed_trips += 1

                            if not trip[1] == person[1]:
                                break
                            else:
                                person_writer.add_trip(trip)

                        person_writer.write(writer)

                        person = next(person_iterator)
                        number_of_processed_persons += 1

                        person_writer = PersonWriter(person)
                        person_writer.add_trip(trip)

                        progress.update()
                except StopIteration:
                    person_writer.write(writer)

            assert(number_of_processed_trips == len(df_trips))
            assert(number_of_processed_persons == len(df_persons))

            writer.end_population()

    return "%s/population.xml.gz" % cache_path
