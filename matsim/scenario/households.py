import gzip
import io

import numpy as np
import pandas as pd

import matsim.writers


def configure(context):
    context.stage("synthesis.population.enriched")
    context.stage("data.constants")
    context.stage("data.spatial.cantons")
    context.config("include_cross_border")
    context.stage("data.cross_border.generate_cross_border_traffic")

FIELDS = ["household_id", "person_id", "income_class", "age", "number_of_cars_class", "number_of_bikes_class",
          "municipality_type", "sp_region", "canton_id", "ovgk", "canton_name", "income_per_capita"]

INCOME_VALUES = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000]


def write_number_of_cars_class(value, c):
    if value == c.MAX_NUMBER_OF_CARS_CLASS:
        return "%d+" % c.MAX_NUMBER_OF_CARS_CLASS
    else:
        return str(value)


def write_bike_availability(value, c):
    if value == c.BIKE_AVAILABILITY_FOR_ALL:
        return "FOR_ALL"
    elif value == c.BIKE_AVAILABILITY_FOR_SOME:
        return "FOR_SOME"
    else:
        return "FOR_NONE"


def add_household(writer, household, member_ids, c):
    writer.start_household(household[1])
    writer.add_members(member_ids)
    writer.add_income(INCOME_VALUES[int(household[3])])

    writer.start_attributes()
    writer.add_attribute("incomeClass", "java.lang.Integer", str(int(household[3])))
    writer.add_attribute("numberOfCars", "java.lang.String", write_number_of_cars_class(household[5], c))
    writer.add_attribute("bikeAvailability", "java.lang.String", write_bike_availability(household[6], c))
    writer.add_attribute("municipalityType", "java.lang.String", str(household[7]))
    writer.add_attribute("spRegion", "java.lang.Integer", str(household[8]))
    writer.add_attribute("ovgk", "java.lang.String", str(household[10]))
    writer.add_attribute("cantonName", "java.lang.String", str(household[11]))
    writer.add_attribute("incomePerCapita", "java.lang.Double", str(household[12]))

    canton_id = str(household[9]) if not np.isnan(household[9]) else "-1"
    writer.add_attribute("cantonId", "java.lang.Double", canton_id)

    writer.end_attributes()

    writer.end_household()


def execute(context):
    cache_path = context.path()
    c          = context.stage("data.constants")

    df_persons = context.stage("synthesis.population.enriched").sort_values(by=["household_id", "person_id"])
    df_cantons = context.stage("data.spatial.cantons")[["canton_id","canton_name_en"]].copy()
    
    # Attach canton name to agent (TODO: do it in previous stages, keep track of canton name)
    df_cantons = df_cantons.rename(columns={"canton_name_en": "canton_name"})
    df_persons = pd.merge(df_persons, df_cantons, left_on="canton_id", right_on="canton_id", how="left")
    assert df_persons.canton_name.notnull().all(), "Not all persons have a canton name assigned. Check the canton data."

    # Attach real average income per person per household
    INCOME_CLASS_MAP = {0: 2000, 1: 3000, 2: 4500, 3: 7000, 4: 9000, 5: 11000,  6: 13000, 7: 15000, 8: 17000}
    df_persons["income"] = df_persons["income_class"].astype(int).map(INCOME_CLASS_MAP)
    df_persons["income_per_capita"] = df_persons["income"] / df_persons["household_size"].fillna(1).clip(lower=1, upper=7)

    df_persons = df_persons[FIELDS]

    if context.config("include_cross_border"):
        cross_border_persons = context.stage("data.cross_border.generate_cross_border_traffic")[0].copy()

        id_person_max = np.max(context.stage("synthesis.population.enriched").copy()["person_id"].values)
        N             = id_person_max + 1

        cross_border_persons    = cross_border_persons.sort_values(by="person_id")

        cross_border_persons["household_id"] = range(N, N + len(cross_border_persons), 1)
        cross_border_persons["person_id"]    = cross_border_persons["household_id"].values

        cross_border_persons["municipality_type"] = "crossborder"
        cross_border_persons["sp_region"]         = -1
        cross_border_persons["canton_id"]         = 0
        cross_border_persons["ovgk"]              = "crossborder"

        cross_border_persons["canton_name"] = "outsideCH"
        cross_border_persons["income_per_capita"] = 0

        cross_border_persons = cross_border_persons[FIELDS]
        df_persons = pd.concat([df_persons, cross_border_persons])


    with gzip.open("%s/households.xml.gz" % cache_path, "w+") as f:
        with io.BufferedWriter(f, buffer_size=1024 * 1024 * 1024 * 2) as raw_writer:
            writer = matsim.writers.HouseholdsWriter(raw_writer)
            writer.start_households()

            household = [None, None]
            member_ids = []

            for item in context.progress(df_persons.itertuples(), total=len(df_persons)):
                # if item[4] >= c.MZ_AGE_THRESHOLD: # Here we filter out young person without actvity chain
                if not household[1] == item[1]:
                    if household[0] is not None: add_household(writer, household, member_ids, c)
                    household, member_ids = item, [item[2]]
                else:
                    member_ids.append(item[2])

            if household[0] is not None: add_household(writer, household, member_ids, c)

            writer.end_households()

    return "%s/households.xml.gz" % cache_path
