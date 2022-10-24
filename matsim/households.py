import gzip
import io

import numpy as np

import data.constants as c
import matsim.writers


def configure(context):
    context.stage("synthesis.population.enriched")

FIELDS = ["household_id", "person_id", "income_class", "age", "number_of_cars_class", "number_of_bikes_class",
          "municipality_type", "sp_region", "canton_id", "ovgk"]
INCOME_VALUES = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000]


def write_number_of_cars_class(value):
    if value == c.MAX_NUMBER_OF_CARS_CLASS:
        return "%d+" % c.MAX_NUMBER_OF_CARS_CLASS
    else:
        return str(value)


def write_bike_availability(value):
    if value == c.BIKE_AVAILABILITY_FOR_ALL:
        return "FOR_ALL"
    elif value == c.BIKE_AVAILABILITY_FOR_SOME:
        return "FOR_SOME"
    else:
        return "FOR_NONE"


def add_household(writer, household, member_ids):
    writer.start_household(household[1])
    writer.add_members(member_ids)
    writer.add_income(INCOME_VALUES[int(household[3])])

    writer.start_attributes()
    writer.add_attribute("incomeClass", "java.lang.Integer", str(int(household[3])))
    writer.add_attribute("numberOfCars", "java.lang.String", write_number_of_cars_class(household[5]))
    writer.add_attribute("bikeAvailability", "java.lang.String", write_bike_availability(household[6]))
    writer.add_attribute("municipalityType", "java.lang.String", str(household[7]))
    writer.add_attribute("spRegion", "java.lang.Integer", str(household[8]))
    writer.add_attribute("ovgk", "java.lang.String", str(household[10]))

    canton_id = str(household[9]) if not np.isnan(household[9]) else "-1"
    writer.add_attribute("cantonId", "java.lang.Double", canton_id)

    writer.end_attributes()

    writer.end_household()


def execute(context):
    cache_path = context.cache_path

    df_persons = context.stage("synthesis.population.enriched").sort_values(by=["household_id", "person_id"])
    df_persons = df_persons[FIELDS]

    with gzip.open("%s/households.xml.gz" % cache_path, "w+") as f:
        with io.BufferedWriter(f, buffer_size=1024 * 1024 * 1024 * 2) as raw_writer:
            writer = matsim.writers.HouseholdsWriter(raw_writer)
            writer.start_households()

            household = [None, None]
            member_ids = []

            for item in context.progress(df_persons.itertuples(), total=len(df_persons)):
                # if item[4] >= c.MZ_AGE_THRESHOLD: # Here we filter out young person without actvity chain
                if not household[1] == item[1]:
                    if household[0] is not None: add_household(writer, household, member_ids)
                    household, member_ids = item, [item[2]]
                else:
                    member_ids.append(item[2])

            if household[0] is not None: add_household(writer, household, member_ids)

            writer.end_households()

    return "%s/households.xml.gz" % cache_path
