import gzip
import io

import numpy as np
import pandas as pd

import matsim.writers

def _require_cols(df, cols, df_name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{df_name} is missing required columns: {missing}")

def _na_to_default(x, default):
    return default if pd.isna(x) else x

def configure(context):
    context.stage("synthesis.population.enriched")
    context.stage("data.constants")
    context.stage("data.spatial.cantons")

    if context.config("include_cross_border"):
        context.stage("data.cross_border.generate_cross_border_traffic")        

    context.config("include_external_population", default = False)
    if context.config("include_external_population"):
        context.stage("data.external_population.read_outputs")


FIELDS = ["household_id", "person_id", "income_class", "age", "number_of_cars_class",
          "municipality_type", "sp_region", "canton_id", "ovgk", "canton_name", "income_per_capita"]

INCOME_VALUES = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000]


def write_number_of_cars_class(value, c):
    if value == c.MAX_NUMBER_OF_CARS_CLASS:
        return "%d+" % c.MAX_NUMBER_OF_CARS_CLASS
    else:
        return str(value)

def add_household(writer, household, member_ids, c):
    # household is a namedtuple row now
    writer.start_household(household.household_id)
    writer.add_members(member_ids)
    writer.add_income(INCOME_VALUES[int(household.income_class)])

    writer.start_attributes()
    writer.add_attribute("incomeClass", "java.lang.Integer", str(int(household.income_class)))
    writer.add_attribute("numberOfCars", "java.lang.String", write_number_of_cars_class(household.number_of_cars_class, c))
    writer.add_attribute("municipalityType", "java.lang.String", str(household.municipality_type))
    writer.add_attribute("spRegion", "java.lang.Integer", str(household.sp_region))
    writer.add_attribute("ovgk", "java.lang.String", str(household.ovgk))
    writer.add_attribute("cantonName", "java.lang.String", str(household.canton_name))
    writer.add_attribute("incomePerCapita", "java.lang.Double", str(household.income_per_capita))

    canton_id = str(_na_to_default(household.canton_id, -1))
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
    df_persons["canton_id"] = df_persons["canton_id"].astype(int)
    df_persons = pd.merge(df_persons, df_cantons, left_on="canton_id", right_on="canton_id", how="left")
    assert df_persons.canton_name.notnull().all(), "Not all persons have a canton name assigned. Check the canton data."

    # Attach real average income per person per household    
    df_persons["income"] = df_persons["income_class"].astype(int).map(c.INCOME_CLASS_MAP)
    
    # Calculate income per capita using the OECD equivalence scale: https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Glossary:Equivalised_income
    df_persons['is_child'] = df_persons['age'] < 14
    num_children = df_persons.groupby('household_id')['is_child'].transform('sum')
    num_adults = df_persons['household_size'] - num_children
    assert (num_adults >= 1).all(), "All households should have at least one adult."
    equvalent_size =  1 + 0.5 * (num_adults - 1) + 0.3 * num_children
    df_persons["income_per_capita"] = df_persons["income"] / equvalent_size

    _require_cols(df_persons, ["household_id", "person_id", "income_class", "number_of_cars_class",
                           "municipality_type", "sp_region", "canton_id", "ovgk", "canton_name", "income_per_capita"], "df_persons")

    # Keep only the fields you need, but don't crash if you later add extras elsewhere
    df_persons = df_persons[[c for c in FIELDS if c in df_persons.columns]]

    if context.config("include_external_population"):
        external_persons   = context.stage("data.external_population.read_outputs")[0].copy()

        external_persons["municipality_type"] = "fr"
        external_persons["sp_region"]         = -1
        external_persons["canton_id"]         = 0
        external_persons["ovgk"]              = "fr"

        external_persons["canton_name"] = "fr"
        external_persons["income_per_capita"] = 0

        external_persons = external_persons[[c for c in FIELDS if c in external_persons.columns]]
        df_persons = pd.concat([df_persons, external_persons])

    if context.config("include_cross_border"):
        cross_border_persons = context.stage("data.cross_border.generate_cross_border_traffic")[0].copy()

        id_person_max    = np.max(context.stage("synthesis.population.enriched").copy()["person_id"].values)
        id_household_max = np.max(context.stage("synthesis.population.enriched").copy()["household_id"].values)
        id_person_max    = max(id_person_max, id_household_max)  # just in case person_id and household_id are not on the same scale
        N                = id_person_max + 1

        cross_border_persons    = cross_border_persons.sort_values(by="person_id")

        cross_border_persons["household_id"] = range(N, N + len(cross_border_persons), 1)
        cross_border_persons["person_id"]    = cross_border_persons["household_id"].values

        cross_border_persons["municipality_type"] = "crossborder"
        cross_border_persons["sp_region"]         = -1
        cross_border_persons["canton_id"]         = 0
        cross_border_persons["ovgk"]              = "crossborder"

        cross_border_persons["canton_name"] = "outsideCH"
        cross_border_persons["income_per_capita"] = 0

        cross_border_persons = cross_border_persons[[c for c in FIELDS if c in cross_border_persons.columns]]
        df_persons = pd.concat([df_persons, cross_border_persons])


    with gzip.open("%s/households.xml.gz" % cache_path, "w+") as f:
        with io.BufferedWriter(f, buffer_size=1024 * 1024 * 1024 * 2) as raw_writer:
            writer = matsim.writers.HouseholdsWriter(raw_writer)
            writer.start_households()

            household = None
            member_ids = []

            # name="HH" gives predictable attribute access even if pandas chooses defaults
            for item in context.progress(df_persons.itertuples(index=False, name="HH"), total=len(df_persons)):
                if (household is None) or (household.household_id != item.household_id):
                    if household is not None:
                        add_household(writer, household, member_ids, c)
                    household = item
                    member_ids = [item.person_id]
                else:
                    member_ids.append(item.person_id)

            if household is not None:
                add_household(writer, household, member_ids, c)


            writer.end_households()

    return "%s/households.xml.gz" % cache_path
