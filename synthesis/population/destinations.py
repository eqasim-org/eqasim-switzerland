import pandas as pd

def configure(context):
    context.stage("synthesis.population.destinations_statent")
    context.config("generate_outbound_flows", False)

    if context.config("generate_outbound_flows"):
        context.stage("data.locations_fr.secondary")


def execute(context):
    df_destinations = context.stage("synthesis.population.destinations_statent")

    if context.config("generate_outbound_flows"):
        df_FR = context.stage("data.locations_fr.secondary")
        df_destinations = pd.concat([df_destinations, df_FR])

    #df_FR.to_file(f"{context.path()}/destinations.gpkg")

    return df_destinations