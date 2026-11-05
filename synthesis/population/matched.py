def configure(context):
    context.config("matching_algorithm", 'v3')
    
    if context.config("matching_algorithm").lower() in ["matched_v2", "v2", "2", "v.2"]:
        context.stage("synthesis.population.matching.matched_v2", alias="matched")
    elif context.config("matching_algorithm").lower() in ["matched_v3", "v3", "3", "v.3"]:
        context.stage("synthesis.population.matching.matched_v3", alias="matched")
    elif context.config("matching_algorithm").lower() in ["matched_v1", "v1", "1", "v.1"]:
        context.stage("synthesis.population.matching.matched_v1", alias="matched")
    else:
        raise ValueError("Unknown matching_algorithm: %s" % context.config("matching_algorithm"))


def execute(context):
    return context.stage("matched")
