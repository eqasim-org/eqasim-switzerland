def configure(context):
    context.stage("synthesis.population.spatial.primary.work.locations")
    context.stage("synthesis.population.spatial.primary.education.locations")


def execute(context):
    df_work = context.stage("synthesis.population.spatial.primary.work.locations")
    df_education = context.stage("synthesis.population.spatial.primary.education.locations")

    return df_work, df_education
