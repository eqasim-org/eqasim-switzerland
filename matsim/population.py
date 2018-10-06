import gzip
from tqdm import tqdm
import data.constants as c
import numpy as np

def configure(context, require):
    require.stage("population.sociodemographics")
    require.stage("population.trips")
    require.config("output_path")

def execute(context):
    output_path = context.config["output_path"]
    df_persons = context.stage("population.sociodemographics")
    df_trips = context.stage("population.trips")

    with gzip.open("%s/population.xml.gz" % output_path, "w+") as f:
        write_line = lambda line: f.write(bytes(line + "\n", "utf-8"))

        write_line('<?xml version="1.0" encoding="utf-8"?>')
        write_line('<!DOCTYPE population SYSTEM "http://www.matsim.org/files/dtd/population_v6.dtd">')
        write_line('<population desc="Switzerland Baseline">')

        for _, row in tqdm(df_persons.iterrows(), total = len(df_persons), desc = "Writing population"):
            if row["age"] >= c.MZ_AGE_THRESHOLD:
                write_line('  <person id="%d">' % row["person_id"])
                write_line('    <attributes>')

                car_availability = lambda x: ["always", "sometimes", "never"][int(x)]
                license = lambda x: "yes" if x else "no"
                employed = lambda x: "true" if x else "false"
                sex = lambda x: ["m", "f"][x]

                attributes = (
                    # Attribute name,   Java Type,              Value
                    ("age",             "java.lang.Integer",    str(row["age"])),
                    ("carAvail",        "java.lang.String",     car_availability(row["car_availability"])),
                    ("employed",        "java.lang.Boolean",    employed(row["employed"])),
                    ("hasLicense",      "java.lang.String",     license(row["driving_license"])),
                    ("sex",             "java.lang.String",     sex(row["sex"])),
                )

                for attribute in attributes:
                    write_line('      <attribute name="%s" class="%s">%s</attribute>' % attribute)

                write_line('    </attributes>')
                write_line('    <plan selected="yes">')

                home_location = (row["home_x"], row["home_y"], None)
                location = (row["home_x"], row["home_y"], None)
                purpose = "home"
                end_time = None

                for _, trip in df_trips[df_trips["person_id"] == row["person_id"]].iterrows():
                    write_line('      <activity type="%s" x="%f" y="%f" end_time="%f"%s />' % (
                        purpose, location[0], location[1], trip["departure_time"],
                        (' facility="%d"' % location[2]) if location[2] is not None else ""
                    ))

                    travel_time = trip["arrival_time"] - trip["departure_time"]

                    write_line('      <leg mode="%s" dep_time="%f" trav_time="%f" />' % (
                        trip["mode"], trip["departure_time"], travel_time
                    ))

                    facility_id = None if np.isnan(trip["location_id"]) else int(trip["location_id"])
                    location = (trip["location_x"], trip["location_y"], facility_id)
                    purpose = trip["purpose"]

                    if np.isnan(location[0]):
                        location = home_location

                write_line('      <activity type="%s" x="%f" y="%f"%s />' % (
                    purpose, location[0], location[1], (' facility="%d"' % location[2]) if location[2] is not None else ""
                ))

                write_line('    </plan>')
                write_line('  </person>')

        write_line('</population>')
