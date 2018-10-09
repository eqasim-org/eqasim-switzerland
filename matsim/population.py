import gzip
from tqdm import tqdm
import data.constants as c
import numpy as np

def configure(context, require):
    require.stage("population.sociodemographics")
    require.stage("population.trips")
    require.config("output_path")

class PersonWriter:
    def __init__(self, person):
        self.person = person
        self.trips = []

    def add_trip(self, trip):
        self.trips.append(trip)

    def write_line(self, f, indent, content):
        line = ("  " * indent) + content + "\n"
        f.write(bytes(line, "utf-8"))

    def write_attribute(self, f, name, type, value):
        self.write_line(f, 3,
            '<attribute name="%s" class="%s">%s</attribute>' % (name, type, value)
        )

    def write_time(self, time):
        time = int(time)
        hours = time // 3600
        minutes = (time % 3600) // 60
        seconds = (time % 60)
        return "%02d:%02d:%02d" % (hours, minutes, seconds)

    def write_activity(self, f, purpose, location, start_time = None, end_time = None):
        attributes = []

        attributes.append('type="%s"' % purpose)
        attributes.append('x="%f"' % location[0])
        attributes.append('y="%f"' % location[1])

        id = location[2]
        if id is not None and not np.isnan(id): attributes.append('facility="%d"' % id)

        if start_time is not None: attributes.append('start_time="%s"' % self.write_time(start_time))
        if end_time is not None: attributes.append('end_time="%s"' % self.write_time(end_time))

        attributes = " ".join(attributes)
        self.write_line(f, 3, '<activity %s />' % attributes)

    def write_leg(self, f, departure_time, arrival_time, mode):
        attributes = []

        travle_time = arrival_time - departure_time

        attributes.append('mode="%s"' % mode)
        attributes.append('dep_time="%s"' % self.write_time(departure_time))
        attributes.append('trav_time="%s"' % self.write_time(arrival_time - departure_time))

        attributes = " ".join(attributes)
        self.write_line(f, 3, '<leg %s />' % attributes)

    def write(self, f):
        if self.person[2] >= c.MZ_AGE_THRESHOLD:
            self.write_line(f, 1, '<person id="%d">' % self.person[1])

            # Write attributes

            self.write_line(f, 2, '<attributes>')

            age = str(self.person[2])
            self.write_attribute(f, "age", "java.lang.Integer", age)

            car_availability = ["always", "sometimes", "never"][int(self.person[3])]
            self.write_attribute(f, "carAvail", "java.lang.String", car_availability)

            employed = "true" if self.person[4] else "false"
            self.write_attribute(f, "employed", "java.lang.Boolean", employed)

            license = "yes" if self.person[5] else "no"
            self.write_attribute(f, "hasLicense", "java.lang.String", license)

            sex = ["m", "f"][self.person[6]]
            self.write_attribute(f, "sex", "java.lang.Integer", sex)

            self.write_line(f, 2, '</attributes>')

            # Write plan

            self.write_line(f, 2, '<plan selected="yes">')

            home_location = (self.person[7], self.person[8], None)
            location = home_location
            purpose = "home"
            end_time = None
            start_time = None
            first = True

            for trip in self.trips:
                end_time = trip[3]

                self.write_activity(f, purpose, location, start_time, end_time)
                self.write_leg(f, trip[3], trip[4], trip[5])

                purpose = trip[6]
                location = (trip[7], trip[8], trip[9])
                start_time = trip[4]

                if np.isnan(trip[7]):
                    location = home_location

            self.write_activity(f, purpose, location)

            self.write_line(f, 2, '</plan>')
            self.write_line(f, 1, '</person>')

PERSON_FIELDS = ["person_id", "age", "car_availability", "employed", "driving_license", "sex", "home_x", "home_y"]
TRIP_FIELDS = ["person_id", "trip_id", "departure_time", "arrival_time", "mode", "purpose", "location_x", "location_y", "location_id"]

def execute(context):
    output_path = context.config["output_path"]
    df_persons = context.stage("population.sociodemographics")
    df_trips = context.stage("population.trips")

    df_persons = df_persons.sort_values(by = "person_id")
    df_trips = df_trips.sort_values(by = ["person_id", "trip_id"])

    df_persons = df_persons[PERSON_FIELDS]
    df_trips = df_trips[TRIP_FIELDS]

    person_iterator = iter(df_persons.itertuples())
    trip_iterator = iter(df_trips.itertuples())

    with gzip.open("%s/population.xml.gz" % output_path, "w+") as f:
        write_line = lambda line: f.write(bytes(line + "\n", "utf-8"))

        write_line('<?xml version="1.0" encoding="utf-8"?>')
        write_line('<!DOCTYPE population SYSTEM "http://www.matsim.org/files/dtd/population_v6.dtd">')
        write_line('<population desc="Switzerland Baseline">')

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

                    person_writer.write(f)

                    person = next(person_iterator)
                    number_of_processed_persons += 1

                    person_writer = PersonWriter(person)
                    person_writer.add_trip(trip)

                    progress.update()
            except StopIteration:
                person_writer.write(f)

        assert(number_of_processed_trips == len(df_trips))
        assert(number_of_processed_persons == len(df_persons))

        write_line('</population>')
