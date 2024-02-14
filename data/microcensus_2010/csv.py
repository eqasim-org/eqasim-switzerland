def configure(context):
    context.stage("data.microcensus_2010.persons")
    context.stage("data.microcensus_2010.trips")
    context.stage("data.microcensus_2010.transit")

def execute(context):
    df_persons = context.stage("data.microcensus_2010.persons")
    df_trips = context.stage("data.microcensus_2010.trips")
    df_transit = context.stage("data.microcensus_2010.transit")

    df_persons.to_csv("%s/persons.csv" % context.cache_path, sep = ";", index = None)
    df_trips.to_csv("%s/trips.csv" % context.cache_path, sep = ";", index = None)
    df_transit.to_csv("%s/transit.csv" % context.cache_path, sep = ";", index = None)
