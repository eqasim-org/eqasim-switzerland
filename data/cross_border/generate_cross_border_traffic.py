def configure(context):
    context.config("include_cross_border", default = False)

    if context.config("include_cross_border"):
        context.stage("data.cross_border.population")
        context.stage("data.cross_border.activities")
        context.stage("data.cross_border.vehicles")

def execute(context):
    if context.config("include_cross_border"):
        population = context.stage("data.cross_border.population")
        activities = context.stage("data.cross_border.activities")
        vehicles   = context.stage("data.cross_border.vehicles")

        return population, activities, vehicles