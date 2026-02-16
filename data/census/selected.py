def configure(context):
    census = context.config("census", default="statpop")

    if census == "statpop":
        context.stage("synthesis.population.scaled", alias = "census")

    elif census == "are_synpop":
        context.stage("data.are_synpop.scaled", alias = "census")

    else:
        raise RuntimeError("Unknown census: %s" % census)
    
def execute(context):
    return context.stage("census")