import io, gzip
import pandas as pd
import numpy as np
from synpp import progress

import matsim.writers as writers


def configure(context):
    context.stage("synthesis.population.destinations")
    #context.stage("synthesis.population.destinations_detailed")
    context.stage("synthesis.population.SNN_population")
    context.stage("synthesis.population.spatial.home.locations")
    context.config("use_detailed_activities")


FIELDS_DETAILED = [
    "destination_id", "destination_x", "destination_y",
    "offers_work", "offers_education", "offers_leisure", "offers_grocery", "offers_other(S)", "offers_culture", "education_type", "offers_religion", "offers_gastronomy", "offers_sport", "offers_other(L)", "offers_other", "offers_visits", "offers_volunteer",
"offers_outdoor"
]

FIELDS_NOT_DETAILED = [
    "destination_id", "destination_x", "destination_y",
    "offers_work", "offers_education", "offers_leisure", "education_type", "offers_shop", "offers_other", 
]


def make_options_detailed(item):
    options = []
    if item[4]: options.append("work")
    if item[5]: options.append("education")
    if item[6]: options.append("leisure")
    if item[7]: options.append("grocery")
    if item[8]: options.append("other(S)")
    if item[9]: options.append("culture")
    if item[11]: options.append("religion")
    if item[12]: options.append("gastronomy")
    if item[13]: options.append("sport") 
    if item[14]: options.append("other(L)")
    if item[15]: options.append("other")
    if item[16]: options.append("visits")
    if item[17]: options.append("volunteer")
    if item[18]: options.append("outdoor")
    return options

def make_options_not_detailed(item):
    options = []
    if item[4]: options.append("work")
    if item[5]: options.append("education")
    if item[6]: options.append("leisure")
    if item[8]: options.append("shop")
    if item[9]: options.append("other")
    return options


def execute(context):
    cache_path = context.cache_path
    #det_activities = context.config("use_detailed_activities")

    # First, write actual facilities (from STATENT)
    df_statent = context.stage("synthesis.population.destinations")

    if False:#det_activities:
        df_statent = context.stage("synthesis.population.destinations_detailed")
        df_statent = df_statent[FIELDS_DETAILED]

    else:
        df_statent = df_statent[FIELDS_NOT_DETAILED]

    with gzip.open("%s/facilities.xml.gz" % cache_path, "w+") as f:
        with io.BufferedWriter(f, buffer_size=1024 * 1024 * 1024 * 2) as raw_writer:
            writer = writers.FacilitiesWriter(raw_writer)
            writer.start_facilities()

            for item in context.progress(df_statent.itertuples(), total=len(df_statent)):
                writer.start_facility(item[1], item[2], item[3])
                if False:#det_activities:
                    if item[4]: writer.add_activity("work")
                    if item[5]: writer.add_activity("education")
                    if item[6]: writer.add_activity("leisure")
                    if item[7]: writer.add_activity("grocery")
                    if item[8]: writer.add_activity("other(S)")
                    if item[9]: writer.add_activity("culture")
                    if item[11]: writer.add_activity("religion")
                    if item[12]: writer.add_activity("gastronomy")
                    if item[13]: writer.add_activity("sport") 
                    if item[14]: writer.add_activity("other(L)")
                    if item[15]: writer.add_activity("other")
                    if item[16]: writer.add_activity("visits")
                    if item[17]: writer.add_activity("volunteer")
                    if item[18]: writer.add_activity("outdoor")
                    writer.end_facility()
                else:
                    if item[4]: writer.add_activity("work")
                    if item[5]: writer.add_activity("education")
                    if item[6]: writer.add_activity("leisure")
                    if item[8]: writer.add_activity("shop")
                    if item[9]: writer.add_activity("other")
                    writer.end_facility()

            # Second, write household facilities
            df_households = context.stage("synthesis.population.SNN_population")[[
                "household_id", "home_x", "home_y"
            ]].drop_duplicates("household_id")

            for item in context.progress(df_households.itertuples(), total=len(df_households), label="Homes"):
                writer.start_facility("home%s" % item[1], item[2], item[3])
                writer.add_activity("home")
                writer.end_facility()
                #progress.update()

            writer.end_facilities()


    return "facilities.xml.gz"


