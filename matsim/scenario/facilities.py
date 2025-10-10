import gzip
import io
import numpy as np

import matsim.writers


def configure(context):
    context.stage("synthesis.population.destinations")
    context.stage("synthesis.population.enriched")

    context.config("include_cross_border", default = False)
    if context.config("include_cross_border"):
        context.stage("data.cross_border.generate_cross_border_traffic")


FIELDS = [
    "destination_id", "destination_x", "destination_y",
    "offers_work", "offers_education", "offers_leisure", "offers_shop", "offers_other"
]


def make_options(item):
    options = []
    if item[4]: options.append("work")
    if item[5]: options.append("education")
    if item[6]: options.append("other")
    if item[7]: options.append("leisure")
    if item[8]: options.append("shop")
    return options


def execute(context):
    cache_path = context.path()

    # First, write actual facilities (from STATENT)
    df_statent = context.stage("synthesis.population.destinations")
    df_statent = df_statent[FIELDS]

    with gzip.open("%s/facilities.xml.gz" % cache_path, "w+") as f:
        with io.BufferedWriter(f, buffer_size=1024 * 1024 * 1024 * 2) as raw_writer:
            writer = matsim.writers.FacilitiesWriter(raw_writer)
            writer.start_facilities()

            for item in context.progress(df_statent.itertuples(), total=len(df_statent)):
                writer.start_facility(item[1], item[2], item[3])
                if item[4]: writer.add_activity("work")
                if item[5]: writer.add_activity("education")
                if item[6]: writer.add_activity("other")
                if item[7]: writer.add_activity("leisure")
                if item[8]: writer.add_activity("shop")
                writer.end_facility()

            # Second, write household facilities
            df_households = context.stage("synthesis.population.enriched")[[
                "household_id", "home_x", "home_y"
            ]].drop_duplicates("household_id")

            for item in context.progress(df_households.itertuples(), total=len(df_households), label="Homes"):
                writer.start_facility("home%s" % item[1], item[2], item[3])
                writer.add_activity("home")
                writer.end_facility()

            if context.config("include_cross_border"):
                cross_border_persons = context.stage("data.cross_border.generate_cross_border_traffic")[0].copy()

                #id_person_max = np.max(context.stage("synthesis.population.enriched").copy()["person_id"].values)
                #N             = id_person_max + 1
                id_hhl_max     = np.max(df_households["household_id"].values)
                N              = id_hhl_max + 1

                cross_border_persons    = cross_border_persons.sort_values(by="person_id")

                cross_border_persons["household_id"] = range(N, N + len(cross_border_persons), 1)

                cbs_hhl = cross_border_persons[["household_id", "home_x", "home_y"]]
                cbs_hhl["home_x"] = cbs_hhl["home_x"].astype(int)
                cbs_hhl["home_y"] = cbs_hhl["home_y"].astype(int)

                for item in context.progress(cbs_hhl.itertuples(), total=len(cbs_hhl), label="Homes - crossborder"):
                    writer.start_facility("home%s" % item[1], item[2], item[3])
                    writer.add_activity("home")
                    writer.end_facility()

            writer.end_facilities()

    return "%s/facilities.xml.gz" % cache_path
