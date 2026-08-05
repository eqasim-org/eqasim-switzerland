def configure(context):
    pt_schedule = context.config("pt_schedule", "gtfs")

    if pt_schedule == "hafas":
        context.stage("matsim.scenario.network.convert_hafas", alias = "pt_schedule")
    elif pt_schedule == "gtfs":
        context.stage("matsim.scenario.network.convert_gtfs", alias = "pt_schedule")
    else:
        raise RuntimeError("Unknown PT schedule format: %s" % pt_schedule)
    
def execute(context):
    return context.stage("pt_schedule")