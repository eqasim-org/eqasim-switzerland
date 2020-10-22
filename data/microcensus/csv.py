def configure(context):
    context.stage("data.microcensus.persons")
    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.transit")

def execute(context):
    df_persons = context.stage("data.microcensus.persons")
    df_trips = context.stage("data.microcensus.trips")
    df_transit = context.stage("data.microcensus.transit")

    df_persons.to_csv("%s/persons.csv" % context.cache_path, sep = ";", index = None)
    df_trips.to_csv("%s/trips.csv" % context.cache_path, sep = ";", index = None)
    df_transit.to_csv("%s/transit.csv" % context.cache_path, sep = ";", index = None)
