import gzip
import io

import matsim.writers


def configure(context):
    context.stage("synthesis.population.destinations")
    context.stage("synthesis.population.enriched")


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
    cache_path = context.cache_path

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

            writer.end_facilities()

    return "%s/facilities.xml.gz" % cache_path
