import pandas as pd

def configure(context):
    context.stage("synthesis.population.destinations_statent")
    context.stage("synthesis.population.spatial.primary.work.remote_locations", alias="remote_work_locations")
    context.config("generate_outbound_flows", False)

    if context.config("generate_outbound_flows"):
        context.stage("data.locations_fr.secondary")


def execute(context):
    df_destinations = context.stage("synthesis.population.destinations_statent")

    # add remote work locations
    df_remote_work = context.stage("remote_work_locations")
    df_remote_work["destination_id"] = df_remote_work["destination_id"].astype("int64")
    df_destinations["destination_id"] = df_destinations["destination_id"].astype("int64")
    df_destinations = pd.concat([df_destinations, df_remote_work], ignore_index=True)

    # add the french population
    if context.config("generate_outbound_flows"):
        df_FR = context.stage("data.locations_fr.secondary")
        df_FR["destination_id"] = df_FR["destination_id"].astype("int64")
        df_destinations = pd.concat([df_destinations, df_FR], ignore_index=True)

    return df_destinations