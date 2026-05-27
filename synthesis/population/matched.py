def configure(context):
    context.config("matching_algorithm", "matched_v1")
    
    if context.config("matching_algorithm").lower() in ["matched_v2", "v2", "2", "v.2"]:
        context.stage("synthesis.population.matching.matched_v2", alias="matched")
    else:
        context.stage("synthesis.population.matching.matched_v1", alias="matched")


def execute(context):
    return context.stage("matched")